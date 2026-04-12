"""stage4_dynamic.py – Stage 4: Dynamic Obstacle Field.

Five dynamic obstacles (vertical cylinders, diameter ≈ 0.5 m, height 5 m)
patrol the stage area with bouncing motion at 0.5 m/s.  Their current
positions are published at 20 Hz on

    /dynamic_obstacles/locations  (geometry_msgs/PoseStamped)

One message per obstacle, identified by ``header.frame_id = "object_{i}"``.

Algorithm (centralised, RRT* with online replanning):

  1. A ``DynamicObstacleTracker`` node subscribes to the obstacle topic in
     a dedicated background executor thread and maintains a thread-safe
     dictionary of the latest (x, y) position for each obstacle ID.

  2. Before entering the obstacle field the swarm adopts a *columnN*
     (single-file) formation.  This minimises the lateral cross-section
     the group presents to approaching obstacles, reducing the probability
     that any flanking obstacle intersects the planned corridor.

  3. An initial RRT* path is planned from the column entry point to the
     exit point using the current obstacle snapshot.

  4. The formation advances waypoint-by-waypoint along the planned path.
     After every waypoint — or whenever any obstacle closes within
     ``REPLAN_TRIGGER_DIST`` of the current formation centroid — a fresh
     RRT* plan is computed from the centroid's current position.  This
     keeps the route up-to-date with obstacle motion without requiring a
     full continuous-time trajectory optimiser.

  5. A tighter mid-flight trigger (``REPLAN_TRIGGER_DIST * 0.7``) interrupts
     an in-progress goto if an obstacle is closing fast, immediately
     issuing replacement commands to all drones.

  6. On completion the swarm reforms in a line formation.

Obstacle model for RRT*:
  Obstacles are modelled as cuboid AABBs (width = depth = cylinder diameter,
  height = cylinder height) centred at the reported (x, y) position and
  inflated by ``OBSTACLE_INFLATION`` metres.  RRT* is run with a thin
  z-slice centred on the constant flight altitude, which collapses the
  problem to 2-D while still using the fully 3-D cleared-segment check;
  all tall obstacles correctly occlude the flight layer.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
import rclpy.executors
import rclpy.parameter
from geometry_msgs.msg import PoseStamped
from as2_msgs.msg import YawMode
from rclpy.node import Node

from swarm.swarm_conductor import SwarmConductor
from formation.formation_manager import FormationManager
from planners.rrts import RRTStarPlanner

# --------------------------------------------------------------------------- #
#  Tuneable parameters                                                         #
# --------------------------------------------------------------------------- #

FLIGHT_SPEED         = 0.6   # m/s  – cruise speed through dynamic field
FLIGHT_HEIGHT        = 1.2   # m   – constant flight altitude
COLUMN_SPACING       = 0.5   # m   – inter-drone gap in columnN
OBSTACLE_INFLATION   = 0.55  # m   – safety margin added to each obstacle AABB
REPLAN_INTERVAL_S    = 1.0   # s   – minimum wall-clock time between replans
REPLAN_TRIGGER_DIST  = 2.0   # m   – replan when any obstacle is closer than this
REPLAN_TRIGGER_CLOSE = 1.4   # m   – mid-flight interrupt threshold (= 0.7 × 2.0)
RRT_STEP_SIZE        = 0.45  # m   – RRT* branch step
RRT_MAX_ITER         = 3000  # iterations
RRT_GOAL_BIAS        = 0.15  # fraction of samples aimed at goal
RRT_REWIRE_RADIUS    = 1.35  # m   – rewire neighbourhood (3 × step)
WAYPOINT_TIMEOUT     = 45.0  # s   – max time to reach one waypoint
TRACKER_WARMUP_S     = 0.5   # s   – wait for tracker to receive first messages
Z_SLICE_HALF         = 0.3   # m   – half-thickness of RRT* z-slice


# --------------------------------------------------------------------------- #
#  Dynamic obstacle tracker                                                    #
# --------------------------------------------------------------------------- #

class DynamicObstacleTracker(Node):
    """Minimal ROS2 node that subscribes to /dynamic_obstacles/locations
    and stores the most recent (x, y, z) for each obstacle ID.

    All state mutations are protected by a threading.Lock so the caller
    can read positions from the main thread while this node's executor
    spins in a background thread.
    """

    _TOPIC = "/dynamic_obstacles/locations"

    def __init__(self, use_sim_time: bool = True) -> None:
        super().__init__("dynamic_obstacle_tracker")

        if use_sim_time:
            self.set_parameters([
                rclpy.parameter.Parameter(
                    "use_sim_time",
                    rclpy.parameter.Parameter.Type.BOOL,
                    True,
                )
            ])

        self._lock: threading.Lock = threading.Lock()
        self._positions: Dict[str, Tuple[float, float, float]] = {}

        self.create_subscription(
            PoseStamped,
            self._TOPIC,
            self._callback,
            20,
        )

    # ------------------------------------------------------------------ #
    #  Subscription callback                                               #
    # ------------------------------------------------------------------ #

    def _callback(self, msg: PoseStamped) -> None:
        key = msg.header.frame_id  # "object_0", "object_1", …
        pos = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        with self._lock:
            self._positions[key] = pos

    # ------------------------------------------------------------------ #
    #  Public accessors (safe to call from any thread)                    #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Return a shallow copy of all current obstacle positions."""
        with self._lock:
            return dict(self._positions)

    def count(self) -> int:
        """Number of distinct obstacle IDs seen so far."""
        with self._lock:
            return len(self._positions)

    def obstacles_as_aabbs(
        self,
        diameter: float,
        height: float,
        inflation: float,
    ) -> List[Dict[str, float]]:
        """Convert current positions to the cuboid AABB dict format
        expected by ``RRTStarPlanner`` (keys: x, y, z, w, d, h).

        The vertical centre is placed at height / 2 so the full
        cylinder is represented (floor to top).
        """
        result: List[Dict[str, float]] = []
        with self._lock:
            for (ox, oy, _oz) in self._positions.values():
                result.append({
                    "x": ox,
                    "y": oy,
                    "z": height / 2.0,
                    "w": diameter,
                    "d": diameter,
                    "h": height,
                })
        return result


# --------------------------------------------------------------------------- #
#  RRT* path planning                                                          #
# --------------------------------------------------------------------------- #

def _plan_rrt(
    start_xy:  List[float],
    goal_xy:   List[float],
    z:         float,
    tracker:   DynamicObstacleTracker,
    stage_cfg: Dict[str, Any],
    stage_size: Tuple[float, float],
    inflation: float,
) -> List[List[float]]:
    """Plan an obstacle-free RRT* path from *start_xy* to *goal_xy* at
    altitude *z*, using the current obstacle snapshot from *tracker*.

    Returns a list of [x, y, z] waypoints.  If RRT* fails to find a path
    within its iteration budget, the direct straight-line is returned as a
    fallback (the caller may still fly it; collision avoidance via replanning
    provides a safety net).
    """
    diam   = float(stage_cfg["obstacle_diameter"])
    height = float(stage_cfg["obstacle_height"])
    sc     = stage_cfg["stage_center"]

    obstacles = tracker.obstacles_as_aabbs(diam, height, inflation)

    # RRT* bounds: stage area plus a 2 m margin so the planner can route
    # around obstacles at the stage boundary.
    hx = stage_size[0] / 2.0 + 2.0
    hy = stage_size[1] / 2.0 + 2.0
    bounds = (
        (sc[0] - hx, sc[0] + hx),
        (sc[1] - hy, sc[1] + hy),
        (z - Z_SLICE_HALF, z + Z_SLICE_HALF),
    )

    planner = RRTStarPlanner(
        obstacles=obstacles,
        bounds=bounds,
        step_size=RRT_STEP_SIZE,
        max_iter=RRT_MAX_ITER,
        goal_bias=RRT_GOAL_BIAS,
        rewire_radius=RRT_REWIRE_RADIUS,
        inflation=inflation,
    )

    path3d = planner.plan(
        start=(start_xy[0], start_xy[1], z),
        goal=(goal_xy[0],  goal_xy[1],  z),
    )

    if not path3d:
        # Fallback: direct line — acceptable given continuous replanning
        return [[start_xy[0], start_xy[1], z],
                [goal_xy[0],  goal_xy[1],  z]]

    return [[p[0], p[1], z] for p in path3d]


# --------------------------------------------------------------------------- #
#  Column formation helpers                                                    #
# --------------------------------------------------------------------------- #

def _column_targets(
    centroid_wp: List[float],
    next_wp:     Optional[List[float]],
    n:           int,
    spacing:     float,
    fallback_heading: float = 0.0,
) -> List[List[float]]:
    """Return per-drone [x, y, z] targets for a columnN formation centred
    on *centroid_wp*, with drone 0 at the head.

    The column axis is aligned toward *next_wp* (or *fallback_heading* when
    the centroid is already at the waypoint).
    """
    if next_wp is not None:
        dx = next_wp[0] - centroid_wp[0]
        dy = next_wp[1] - centroid_wp[1]
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) if dist > 1e-6 else fallback_heading
    else:
        heading = fallback_heading

    local_offsets = FormationManager.get_offsets("columnN", n, spacing)
    world_offsets = FormationManager.rotate_to_world(local_offsets, heading)

    return [
        [centroid_wp[0] + ddx, centroid_wp[1] + ddy, centroid_wp[2]]
        for ddx, ddy in world_offsets
    ]


# --------------------------------------------------------------------------- #
#  Replanning trigger                                                          #
# --------------------------------------------------------------------------- #

def _closest_obstacle_m(
    centroid_xy: List[float],
    tracker:     DynamicObstacleTracker,
) -> float:
    """Minimum Euclidean XY distance from any tracked obstacle to the
    formation centroid.  Returns inf when no obstacles have been seen yet.
    """
    positions = tracker.get_positions()
    if not positions:
        return float("inf")
    cx, cy = centroid_xy
    return min(
        math.hypot(ox - cx, oy - cy)
        for (ox, oy, _) in positions.values()
    )


# --------------------------------------------------------------------------- #
#  Stage entry point                                                           #
# --------------------------------------------------------------------------- #

def run_stage4(
    conductor:  SwarmConductor,
    stage_cfg:  Dict[str, Any],
    stage_size: Tuple[float, float] = (10.0, 10.0),
    speed:      float = FLIGHT_SPEED,
    spacing:    float = COLUMN_SPACING,
    verbose:    bool  = True,
) -> None:
    """Execute Stage 4: dynamic obstacle field traversal.

    Parameters
    ----------
    conductor   : SwarmConductor
    stage_cfg   : the 'stage4' dict from the scenario YAML.  Expected keys:
                    stage_center      : [x, y]
                    start_point       : [dx, dy]  relative to stage_center
                    end_point         : [dx, dy]  relative to stage_center
                    num_obstacles     : int
                    obstacle_velocity : float (m/s)
                    obstacle_diameter : float (m)
                    obstacle_height   : float (m)
    stage_size  : (width, depth) of the bounding box within which obstacles
                  patrol (from the top-level scenario ``stage_size`` field).
    speed       : go_to cruise speed (m/s)
    spacing     : columnN inter-drone gap (m)
    verbose     : print progress messages
    """
    sc       = stage_cfg["stage_center"]
    start_xy = [sc[0] + stage_cfg["start_point"][0],
                sc[1] + stage_cfg["start_point"][1]]
    goal_xy  = [sc[0] + stage_cfg["end_point"][0],
                sc[1] + stage_cfg["end_point"][1]]
    z        = FLIGHT_HEIGHT

    init_heading = math.atan2(
        goal_xy[1] - start_xy[1],
        goal_xy[0] - start_xy[0],
    )

    # ------------------------------------------------------------------
    # 1. Start the obstacle tracker in a background executor thread.
    #    DroneAgents already own their own ROS2 nodes; we add one more
    #    lightweight subscriber node to the same process.
    # ------------------------------------------------------------------
    tracker  = DynamicObstacleTracker(use_sim_time=True)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(tracker)
    spin_thread = threading.Thread(
        target=executor.spin, daemon=True, name="obs_tracker_spin"
    )
    spin_thread.start()

    # Allow time for the first obstacle messages to arrive.
    time.sleep(TRACKER_WARMUP_S)

    if verbose:
        print(
            f"[Stage 4] Tracking {tracker.count()} obstacles.  "
            f"Start {start_xy} → Goal {goal_xy}, "
            f"heading {math.degrees(init_heading):.1f}°."
        )

    try:
        _run_traversal(
            conductor=conductor,
            start_xy=start_xy,
            goal_xy=goal_xy,
            z=z,
            init_heading=init_heading,
            tracker=tracker,
            stage_cfg=stage_cfg,
            stage_size=stage_size,
            speed=speed,
            spacing=spacing,
            verbose=verbose,
        )
    finally:
        executor.shutdown(timeout_sec=2.0)

    if verbose:
        print("[Stage 4] Complete.")


# --------------------------------------------------------------------------- #
#  Inner traversal loop                                                        #
# --------------------------------------------------------------------------- #

def _run_traversal(
    conductor:    SwarmConductor,
    start_xy:     List[float],
    goal_xy:      List[float],
    z:            float,
    init_heading: float,
    tracker:      DynamicObstacleTracker,
    stage_cfg:    Dict[str, Any],
    stage_size:   Tuple[float, float],
    speed:        float,
    spacing:      float,
    verbose:      bool,
) -> None:
    """Core execution loop (separated so the finally-block in run_stage4
    always fires even if an exception occurs here)."""

    n = conductor.n

    # ------------------------------------------------------------------
    # 2. Move the swarm to the stage entry in columnN formation.
    # ------------------------------------------------------------------
    if verbose:
        print("[Stage 4] Moving to entry in columnN formation…")
    conductor.set_all_leds("orange")
    conductor.goto_formation(
        centroid_xyz=[start_xy[0], start_xy[1], z],
        heading_rad=init_heading,
        formation="columnN",
        spacing=spacing,
        speed=speed,
    )

    # ------------------------------------------------------------------
    # 3. Plan the initial RRT* path.
    # ------------------------------------------------------------------
    if verbose:
        print("[Stage 4] Planning initial RRT* path…")
    path = _plan_rrt(
        start_xy, goal_xy, z,
        tracker, stage_cfg, stage_size,
        OBSTACLE_INFLATION,
    )
    if verbose:
        print(f"[Stage 4] Initial path has {len(path)} waypoints.")

    # ------------------------------------------------------------------
    # 4. Advance waypoint-by-waypoint with online replanning.
    # ------------------------------------------------------------------
    conductor.set_all_leds("red")
    last_replan = time.time()
    wp_idx      = 0

    while wp_idx < len(path):
        wp      = path[wp_idx]
        next_wp = path[wp_idx + 1] if wp_idx + 1 < len(path) else None

        centroid    = conductor.get_centroid()
        centroid_xy = centroid[:2]

        # ---- Check whether a fresh plan is needed before this step ----
        time_since = time.time() - last_replan
        closest    = _closest_obstacle_m(centroid_xy, tracker)
        if time_since >= REPLAN_INTERVAL_S or closest <= REPLAN_TRIGGER_DIST:
            if verbose:
                print(
                    f"[Stage 4] Replanning — closest obs {closest:.2f} m, "
                    f"{time_since:.1f} s since last plan."
                )
            new_path = _plan_rrt(
                centroid_xy, goal_xy, z,
                tracker, stage_cfg, stage_size,
                OBSTACLE_INFLATION,
            )
            if new_path and len(new_path) >= 2:
                path        = new_path
                wp_idx      = 0
                last_replan = time.time()
                wp          = path[0]
                next_wp     = path[1] if len(path) > 1 else None
                if verbose:
                    print(f"[Stage 4]   New path: {len(path)} waypoints.")

        # ---- Command the column to the next waypoint ------------------
        targets  = _column_targets(wp, next_wp, n, spacing, init_heading)
        for drone, tgt in zip(conductor.drones, targets):
            drone.cmd_goto(
                tgt[0], tgt[1], tgt[2],
                speed=speed,
                yaw_mode=YawMode.PATH_FACING,
            )

        # ---- Wait for arrival, allowing mid-flight interrupt ----------
        deadline    = time.time() + WAYPOINT_TIMEOUT
        interrupted = False

        while time.time() < deadline:
            if all(d.behaviour_idle() for d in conductor.drones):
                break  # reached waypoint normally

            # Check for a close-approach interrupt
            centroid_xy = conductor.get_centroid()[:2]
            closest     = _closest_obstacle_m(centroid_xy, tracker)

            if closest <= REPLAN_TRIGGER_CLOSE:
                if verbose:
                    print(
                        f"[Stage 4] Mid-flight interrupt — "
                        f"obstacle at {closest:.2f} m."
                    )
                new_path = _plan_rrt(
                    centroid_xy, goal_xy, z,
                    tracker, stage_cfg, stage_size,
                    OBSTACLE_INFLATION,
                )
                if new_path and len(new_path) >= 2:
                    path        = new_path
                    wp_idx      = -1   # will become 0 after the outer increment
                    last_replan = time.time()
                    interrupted = True

                    # Issue replacement commands immediately
                    new_wp   = path[0]
                    new_next = path[1] if len(path) > 1 else None
                    new_tgts = _column_targets(
                        new_wp, new_next, n, spacing, init_heading
                    )
                    for drone, tgt in zip(conductor.drones, new_tgts):
                        drone.cmd_goto(
                            tgt[0], tgt[1], tgt[2],
                            speed=speed,
                            yaw_mode=YawMode.PATH_FACING,
                        )
                    if verbose:
                        print(
                            f"[Stage 4]   Emergency path: "
                            f"{len(path)} waypoints."
                        )
                    break  # restart the wait loop for the new first waypoint

            time.sleep(0.1)

        if not interrupted:
            wp_idx += 1

    # ------------------------------------------------------------------
    # 5. Reform at goal in a line formation and signal completion.
    # ------------------------------------------------------------------
    if verbose:
        print("[Stage 4] Reforming at exit in line formation…")
    conductor.set_all_leds("green")
    conductor.goto_formation(
        centroid_xyz=[goal_xy[0], goal_xy[1], z],
        heading_rad=init_heading,
        formation="line",
        spacing=spacing,
        speed=speed,
    )
    conductor.set_formation_leds("line")
