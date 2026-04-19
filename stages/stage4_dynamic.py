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
OBSTACLE_INFLATION   = 0.85  # m   – safety margin added to each obstacle AABB
REPLAN_INTERVAL_S    = 0.5   # s   – minimum wall-clock time between replans
TARGET_DELTA_M       = 0.15  # m   – below this, skip re-issuing cmd_goto (stutter guard)
REPLAN_TRIGGER_DIST  = 3.0   # m   – replan when any obstacle is closer than this
REPLAN_TRIGGER_CLOSE = 2.1   # m   – mid-flight interrupt threshold (= 0.7 × 2.0)
RRT_STEP_SIZE        = 0.45  # m   – RRT* branch step
RRT_MAX_ITER         = 4000  # iterations
RRT_GOAL_BIAS        = 0.15  # fraction of samples aimed at goal
RRT_REWIRE_RADIUS    = 1.35  # m   – rewire neighbourhood (3 × step)
WAYPOINT_TIMEOUT     = 45.0  # s   – max time to reach one waypoint
TRACKER_WARMUP_S     = 0.5   # s   – wait for tracker to receive first messages
Z_SLICE_HALF         = 0.3   # m   – half-thickness of RRT* z-slice

# [STAGE4 SWEPT] Predictive obstacle inflation parameters. Obstacles are
# projected forward along their estimated velocity and emitted as extra
# inflated AABBs at discrete time snapshots, so RRT* plans around future
# positions rather than just the current snapshot.
# [STAGE4 REACTIVE] Horizon raised 1.5 → 4.0 s and sweep dt 0.5 → 1.0 s.
# 4 s × 0.5 m/s = 2 m of obstacle motion coverage, enough to catch
# obstacles approaching from ~3 m away. 5 snapshots/obstacle at dt=1.0
# keeps planner cost ~5× the non-swept baseline.
OBSTACLE_HORIZON_S   = 4.0   # s   – how far ahead to project obstacle motion
OBSTACLE_SWEEP_DT    = 1.0   # s   – time step between predictive snapshots
VEL_MIN_DT_S         = 0.02  # s   – lower bound for velocity finite-diff gap
VEL_MAX_DT_S         = 0.5   # s   – upper bound for velocity finite-diff gap
VEL_MAX_MAGNITUDE    = 1.5   # m/s – reject vel estimates above this (sanity)

# [STAGE4 REACTIVE] Reactive-sidestep layer parameters.
# Fires when the leader is within REACTIVE_DANGER_M of any obstacle.
# The sidestep vector is oriented perpendicular to the obstacle's
# estimated velocity (toward whichever side the leader already is on)
# and scaled to REACTIVE_SIDESTEP_M. Stationary obstacles (|v| below
# REACTIVE_MIN_VEL) get a pure away-from-obstacle push instead.
# [STAGE4 REACTIVE2] Raised 1.2 → 2.2 m so reactive sidestep fires
# BEFORE the obstacle enters the 1.2–2.1 m band where the A*
# fallback oscillation occurs (see 2026-04-18 run: 18+ rapid-fire
# interrupts at 1.2 m before reactive finally broke the lockup).
REACTIVE_DANGER_M    = 2.2   # m   – swarm-to-obstacle threshold
REACTIVE_SIDESTEP_M  = 1.8   # m   – sidestep distance
REACTIVE_MIN_VEL     = 0.1   # m/s – below this use push-away, not perp


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
        # [STAGE4 SWEPT] Per-obstacle previous sample (pos, sim_time)
        # and latest estimated velocity (vx, vy, vz). Populated lazily
        # in _callback once two samples have been seen.
        self._prev_samples: Dict[str, Tuple[Tuple[float, float, float], float]] = {}
        self._velocities: Dict[str, Tuple[float, float, float]] = {}

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
        # [STAGE4 SWEPT] Use the sim-time header stamp (not wall clock)
        # for velocity estimation, so paused sims / frame skips don't
        # corrupt the finite-difference.
        t = (float(msg.header.stamp.sec)
             + float(msg.header.stamp.nanosec) * 1e-9)
        with self._lock:
            prev = self._prev_samples.get(key)
            if prev is not None:
                prev_pos, prev_t = prev
                dt = t - prev_t
                if VEL_MIN_DT_S <= dt <= VEL_MAX_DT_S:
                    vx = (pos[0] - prev_pos[0]) / dt
                    vy = (pos[1] - prev_pos[1]) / dt
                    vz = (pos[2] - prev_pos[2]) / dt
                    if (vx * vx + vy * vy + vz * vz) ** 0.5 <= VEL_MAX_MAGNITUDE:
                        self._velocities[key] = (vx, vy, vz)
            self._positions[key] = pos
            self._prev_samples[key] = (pos, t)

    # ------------------------------------------------------------------ #
    #  Public accessors (safe to call from any thread)                    #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Return a shallow copy of all current obstacle positions."""
        with self._lock:
            return dict(self._positions)

    def get_velocities(self) -> Dict[str, Tuple[float, float, float]]:
        """[STAGE4 SWEPT] Return a shallow copy of estimated obstacle
        velocities. Obstacles for which no velocity has yet been
        estimated (i.e. only one pose sample seen so far) are omitted.
        """
        with self._lock:
            return dict(self._velocities)

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

    def obstacles_as_swept_aabbs(
        self,
        diameter: float,
        height: float,
        inflation: float,
        horizon_s: float,
        dt: float,
    ) -> List[Dict[str, float]]:
        """[STAGE4 SWEPT] Predictive obstacle AABBs.

        For each tracked obstacle, emit one inflated AABB per time
        snapshot t ∈ {0, dt, 2dt, …, horizon_s}, positioned at
        pos + vel * t (horizontal components only — patrol obstacles
        don't move vertically). The RRT* planner sees each snapshot as
        a separate static obstacle, so a moving obstacle becomes a
        "trail" of inflated volumes along its predicted trajectory,
        and both _clear and the RRT* tree-building route around it.

        If no velocity is yet known for a given obstacle (only one
        pose sample seen so far), only the t=0 snapshot is emitted —
        equivalent to the non-swept AABB for that obstacle.

        Parameters
        ----------
        diameter   : cylinder diameter (m) — maps to w, d in the AABB
        height     : cylinder height (m)
        inflation  : unused here (planner applies inflation itself),
                     kept in the signature for API parity with
                     obstacles_as_aabbs.
        horizon_s  : how far ahead to project (s)
        dt         : step between snapshots (s)
        """
        result: List[Dict[str, float]] = []
        if horizon_s < 0.0 or dt <= 0.0:
            return self.obstacles_as_aabbs(diameter, height, inflation)
        n_steps = int(math.floor(horizon_s / dt)) + 1
        with self._lock:
            positions = dict(self._positions)
            velocities = dict(self._velocities)
        for key, (ox, oy, _oz) in positions.items():
            vx, vy, _vz = velocities.get(key, (0.0, 0.0, 0.0))
            for k in range(n_steps):
                t = k * dt
                result.append({
                    "x": ox + vx * t,
                    "y": oy + vy * t,
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

    # [STAGE4 SWEPT] Swap the snapshot-obstacle call for the swept
    # (predictive) obstacle list, so RRT* routes around future
    # positions rather than just current ones.
    obstacles = tracker.obstacles_as_swept_aabbs(
        diam, height, inflation,
        horizon_s=OBSTACLE_HORIZON_S,
        dt=OBSTACLE_SWEEP_DT,
    )
    # [STAGE4 REACTIVE] Diagnostic so we can confirm from logs that the
    # swept layer is actually biting (AABB count should be ~5× the
    # obstacle count, and velocity count should equal obstacle count
    # once warmed up).
    _n_vel = len(tracker.get_velocities())
    print(
        f"[Stage 4]   plan input: {len(obstacles)} swept AABBs "
        f"({tracker.count()} obstacles × snapshots, "
        f"{_n_vel} velocities estimated)"
    )

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

def _compute_reactive_sidestep(
    conductor,
    tracker: "DynamicObstacleTracker",
) -> Optional[Tuple[float, float, Tuple[float, float], float]]:
    """[STAGE4 REACTIVE2] Scan ALL (drone, obstacle) pairs and return
    (sx, sy, (ox, oy), danger_m) when any drone is within
    REACTIVE_DANGER_M of any obstacle. None otherwise.

    Previous (leader-only) version missed trailing-drone approaches:
    if the column is perpendicular to the obstacle's motion, a
    trailing drone can be closer to the obstacle than the leader and
    no sidestep would fire (2026-04-18 run: "near miss for the last
    drone in the formation").

    Sidestep origin is the CLOSEST drone's position, not always the
    leader's — this keeps the perpendicular-to-velocity bias on the
    correct side for the threatened drone. All drones are translated
    by the same (sx, sy) so formation geometry is preserved.

    danger_m (returned) is the min drone-to-obstacle distance; used
    for log output.
    """
    positions = tracker.get_positions()
    if not positions:
        return None
    velocities = tracker.get_velocities()

    # Scan all (drone, obstacle) pairs.
    best_d: float = float("inf")
    best_obs_key: Optional[str] = None
    best_drone_xy: Optional[Tuple[float, float]] = None
    for drone in conductor.drones:
        dx_, dy_, _dz = drone.xyz
        for key, (ox, oy, _oz) in positions.items():
            d = math.hypot(ox - dx_, oy - dy_)
            if d < best_d:
                best_d = d
                best_obs_key = key
                best_drone_xy = (dx_, dy_)

    if best_obs_key is None or best_d > REACTIVE_DANGER_M:
        return None

    ox, oy, _oz = positions[best_obs_key]
    vx, vy, _vz = velocities.get(best_obs_key, (0.0, 0.0, 0.0))
    # "lx, ly" here is the closest-drone position — used both for
    # sidestep-side bias and to keep the variable naming stable
    # relative to the mental model ("my anchor drone").
    lx, ly = best_drone_xy  # type: ignore[misc]
    rel_x, rel_y = lx - ox, ly - oy
    rel_mag = math.hypot(rel_x, rel_y) or 1e-9
    v_mag = math.hypot(vx, vy)

    if v_mag >= REACTIVE_MIN_VEL:
        # Perpendicular to obstacle velocity, biased to the closest
        # drone's side.
        perp_x, perp_y = -vy, vx
        if perp_x * rel_x + perp_y * rel_y < 0.0:
            perp_x, perp_y = -perp_x, -perp_y
        pm = math.hypot(perp_x, perp_y) or 1e-9
        dx, dy = perp_x / pm, perp_y / pm
    else:
        # Stationary obstacle — push radially away.
        dx, dy = rel_x / rel_mag, rel_y / rel_mag

    return (
        dx * REACTIVE_SIDESTEP_M,
        dy * REACTIVE_SIDESTEP_M,
        (ox, oy),
        best_d,
    )


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
    # [DELTA GUARD] last commanded target per drone — None on first tick
    _last_cmd_targets: list = [None] * conductor.n

    while wp_idx < len(path):
        wp      = path[wp_idx]
        next_wp = path[wp_idx + 1] if wp_idx + 1 < len(path) else None

        centroid    = conductor.get_centroid()
        centroid_xy = centroid[:2]
        leader_xy   = conductor.drones[0].xyz[:2]  # [LEADER FIX] column ref point

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
                leader_xy, goal_xy, z,
                tracker, stage_cfg, stage_size,
                OBSTACLE_INFLATION,
            )
            if new_path and len(new_path) >= 2:
                path        = new_path
                wp_idx      = min(1, len(path) - 1)  # [LEADER FIX] skip current-pos waypoint
                last_replan = time.time()
                wp          = path[wp_idx]
                next_wp     = path[wp_idx + 1] if wp_idx + 1 < len(path) else None
                if verbose:
                    print(f"[Stage 4]   New path: {len(path)} waypoints.")

        # ---- Command the column to the next waypoint ------------------
        targets  = _column_targets(wp, next_wp, n, spacing, init_heading)
        # [DELTA GUARD] only re-issue cmd_goto if target moved > TARGET_DELTA_M
        _issued_any = False
        for _i, (drone, tgt) in enumerate(zip(conductor.drones, targets)):
            if _target_changed(_i, tgt[0], tgt[1], tgt[2], _last_cmd_targets):
                drone.cmd_goto(
                    tgt[0], tgt[1], tgt[2],
                    speed=speed,
                    yaw_mode=YawMode.PATH_FACING,
                )
                _issued_any = True
        # [IDLE RACE FIX] wait for all drones to leave IDLE before
        # trusting a later idle signal as "arrived". Only waits if we
        # actually issued a command this tick (delta guard may skip all).
        if _issued_any:
            _start_deadline = time.time() + 1.0
            while time.time() < _start_deadline:
                if not any(d.behaviour_idle() for d in conductor.drones):
                    break
                time.sleep(0.02)
            else:
                if verbose:
                    _idle = [d.behaviour_idle() for d in conductor.drones]
                    print(f"[Stage 4]   WARN: drones did not leave idle in 1s: {_idle}")

        # ---- Wait for arrival, allowing mid-flight interrupt ----------
        deadline    = time.time() + WAYPOINT_TIMEOUT
        interrupted = False

        while time.time() < deadline:
            if all(d.behaviour_idle() for d in conductor.drones):
                break  # reached waypoint normally

            # Check for a close-approach interrupt
            centroid_xy = conductor.get_centroid()[:2]
            leader_xy   = conductor.drones[0].xyz[:2]  # [LEADER FIX]
            closest     = _closest_obstacle_m(centroid_xy, tracker)

            # [STAGE4 REACTIVE] Priority 1 — if any obstacle is within
            # REACTIVE_DANGER_M of the leader, sidestep NOW. Planning
            # won't help: if the leader is already inside an inflated
            # obstacle, RRT*'s plan() returns [] and _plan_rrt's
            # fallback quietly emits a straight line — the exact bug
            # we saw in the 3.01 m → 0.26 m approach trace. An APF-style
            # sidestep pushes the swarm perpendicular to the obstacle's
            # motion (or radially away for stationary obstacles),
            # buying the replanner room to work on the next tick.
            # [STAGE4 REACTIVE2] Reactive trigger is swarm-wide now —
            # the `closest` variable uses leader-only distance, but
            # the sidestep helper scans ALL drones and only returns a
            # non-None result when some drone is within
            # REACTIVE_DANGER_M (=2.2 m). Dropping the leader-only gate
            # here fixes the trailing-drone near-miss case.
            sidestep = _compute_reactive_sidestep(conductor, tracker)
            if sidestep is not None:
                sx, sy, (ox_, oy_), danger_m = sidestep
                if verbose:
                    print(
                        f"[Stage 4] REACTIVE SIDESTEP — "
                        f"{danger_m:.2f} m drone-obs, push "
                        f"({sx:+.2f}, {sy:+.2f}) m."
                    )
                # Rewrite the current path as [leader, sidestep, goal]
                # so the outer loop issues column targets for the
                # sidestep waypoint next.
                sidestep_wp = [leader_xy[0] + sx, leader_xy[1] + sy, z]
                goal_wp     = [goal_xy[0],       goal_xy[1],       z]
                path        = [[leader_xy[0], leader_xy[1], z],
                               sidestep_wp, goal_wp]
                wp_idx      = 1
                last_replan = time.time()
                interrupted = True

                # [STAGE4 UNHOVER] Hover-hold removed — cmd_hover() calls
                # motion_ref_handler.hover() which disables the
                # trajectory_generator without re-enabling it; subsequent
                # go_to behaviours could not form trajectories cleanly.
                # Altitude sag during the handover is acceptable here
                # because the sidestep is followed by an immediate
                # cmd_goto batch below, and the delta guard re-commands
                # a fresh target anyway.
                # Issue column targets pointing at the sidestep wp
                # straight away so we don't wait for the outer loop.
                new_tgts = _column_targets(
                    sidestep_wp, goal_wp, n, spacing, init_heading
                )
                for _i, (drone, tgt) in enumerate(zip(conductor.drones, new_tgts)):
                    drone.cmd_goto(
                        tgt[0], tgt[1], tgt[2],
                        speed=speed,
                        yaw_mode=YawMode.PATH_FACING,
                    )
                    # Update delta-guard cache so the outer loop
                    # doesn't immediately re-issue the same thing.
                    _last_cmd_targets[_i] = (tgt[0], tgt[1], tgt[2])
                break  # restart wait loop for the sidestep waypoint

            if closest <= REPLAN_TRIGGER_CLOSE:
                if verbose:
                    print(
                        f"[Stage 4] Mid-flight interrupt — "
                        f"obstacle at {closest:.2f} m."
                    )
                new_path = _plan_rrt(
                    leader_xy, goal_xy, z,
                    tracker, stage_cfg, stage_size,
                    OBSTACLE_INFLATION,
                )
                if new_path and len(new_path) >= 2:
                    path        = new_path
                    wp_idx      = min(1, len(path) - 1)  # [LEADER FIX] skip current-pos waypoint
                    last_replan = time.time()
                    interrupted = True

                    # Issue replacement commands immediately
                    new_wp   = path[wp_idx]
                    new_next = path[wp_idx + 1] if wp_idx + 1 < len(path) else None
                    new_tgts = _column_targets(
                        new_wp, new_next, n, spacing, init_heading
                    )
                    # [STAGE4 UNHOVER] Hover-hold removed — see top of
                    # file for rationale. The subsequent cmd_goto batch
                    # will re-engage a fresh go_to behaviour cleanly.
                    # [DELTA GUARD] only re-issue cmd_goto if target moved
                    _issued_any_mid = False
                    for _i, (drone, tgt) in enumerate(zip(conductor.drones, new_tgts)):
                        if _target_changed(_i, tgt[0], tgt[1], tgt[2], _last_cmd_targets):
                            drone.cmd_goto(
                                tgt[0], tgt[1], tgt[2],
                                speed=speed,
                                yaw_mode=YawMode.PATH_FACING,
                            )
                            _issued_any_mid = True
                    # [STAGE4 UNHOVER] Reverted to the simple idle-race
                    # wait from patch_stage4_idle_race.py — wait for
                    # drones to transition out of IDLE, without
                    # re-hovering during the wait. The per-tick
                    # cmd_hover() was disabling the trajectory generator
                    # repeatedly.
                    if _issued_any_mid:
                        _mid_deadline = time.time() + 1.0
                        while time.time() < _mid_deadline:
                            if not any(d.behaviour_idle() for d in conductor.drones):
                                break
                            time.sleep(0.02)
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


def _target_changed(idx: int, x: float, y: float, z: float,
                    last: list, thresh: float = None) -> bool:
    """Return True if (x,y,z) differs from last[idx] by > thresh (defaults
    to TARGET_DELTA_M). Updates last[idx] on True, leaves it on False."""
    if thresh is None:
        thresh = TARGET_DELTA_M
    prev = last[idx]
    if prev is None:
        last[idx] = (x, y, z)
        return True
    dx = x - prev[0]
    dy = y - prev[1]
    dz = z - prev[2]
    if (dx * dx + dy * dy + dz * dz) ** 0.5 > thresh:
        last[idx] = (x, y, z)
        return True
    return False


# [LEADER FIX applied]

# [IDLE RACE FIX applied]

# [RRT TUNING] applied: OBSTACLE_INFLATION 0.55 -> 0.85,
# REPLAN_TRIGGER_DIST 2.0 -> 2.5, REPLAN_TRIGGER_CLOSE 1.4 -> 1.8,
# RRT_MAX_ITER 3000 -> 4000. See patch_stage4_rrt_tuning.py rationale.

# [STAGE4 FASTER REPLAN] applied: REPLAN_INTERVAL_S 1.0 -> 0.5,
# REPLAN_TRIGGER_DIST 2.5 -> 3.0, REPLAN_TRIGGER_CLOSE 1.8 -> 2.1.
# Halved replan cycle so stale-path window is smaller than what
# 0.85 m obstacle inflation buys. See patch_stage4_faster_replan.py.

# [STAGE4 SWEPT] applied: DynamicObstacleTracker now estimates obstacle
# velocities from consecutive sim-time-stamped samples, and exposes
# obstacles_as_swept_aabbs(...) which emits predictive AABBs at
# t = 0, 0.5, 1.0, 1.5 s. _plan_rrt consumes the swept list, so RRT*
# plans dodge paths around where obstacles *will be*, defeating the
# _clear early-exit that previously let the straight line through.
# See patch_stage4_swept.py rationale.

# [STAGE4 REACTIVE] applied: horizon 1.5→4.0 s, sweep dt 0.5→1.0 s;
# reactive APF-style sidestep layered onto the mid-flight interrupt
# (fires at REACTIVE_DANGER_M = 1.2 m, pushing perpendicular to the
# obstacle's velocity or radially away if stationary); plan-input
# diagnostic added to _plan_rrt. See patch_stage4_reactive.py.

# [STAGE4 REACTIVE2] applied: REACTIVE_DANGER_M 1.2 → 2.2 m;
# _compute_reactive_sidestep scans all drones and uses closest-drone
# origin; both reactive and replan branches hover-hold IDLE drones
# before issuing cmd_gotos; replan idle-race wait re-issues hover on
# each tick for any drone still in IDLE. See patch_stage4_reactive2.py.

# [STAGE4 UNHOVER] applied: removed the three cmd_hover() call sites
# introduced by patch_stage4_reactive2. Keeps the danger-radius bump
# (2.2 m) and swarm-aware sidestep from reactive2. See
# patch_stage4_unhover.py for rationale.
