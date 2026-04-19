"""stage3_forest.py – Stage 3: Forest Traversal.

Seven cylindrical trees (static obstacles) are placed in the stage area.
The swarm must navigate from a start point to an end point without
colliding with the trees.

Algorithm (centralised, per-drone A* with altitude staggering):

  1. Convert CW2 tree positions to the obstacle format expected by the
     CW1 OccupancyGrid3D / astar_search_3d (cuboid AABB).
  2. Compute initial formation offsets (compressed line, perpendicular
     to the start→end heading) and offset start/goal positions for every
     drone individually.
  3. Run 3-D A* for each drone from its offset start to its offset goal
     through the shared occupancy grid.
  4. To resolve any spatial conflicts between planned paths, drones in
     the back half of the formation fly 0.3 m higher than those in front.
     This altitude stratification ensures paths do not share a cell even
     when they must take similar routes through narrow corridors.
  5. All drones execute their paths concurrently; each drone follows a
     sequence of go_to waypoints and waits for each before advancing.

Obstacle inflation is set to cover the drone body radius plus a safety
margin (0.35 m total, well above the 0.09 m crazyflie body radius).
"""

import math
import time
from typing import List, Dict, Any, Optional

from as2_msgs.msg import YawMode

from swarm.swarm_conductor import SwarmConductor
from formation.formation_manager import FormationManager

# Import CW1-derived planners
from planners.grid_map import OccupancyGrid3D, GridConfig
from planners.astar import astar_search_3d
from planners.path_utils import (
    cells_to_world_path_3d,
    simplify_world_path_3d,
)

# ---- tuneable parameters --------------------------------------------------

FLIGHT_SPEED      = 0.5   # m/s  — slow through trees
FLIGHT_HEIGHT_BASE = 1.2  # m  — nominal cruise altitude (shared centre)
# [STAGE3 FANOUT] Tight altitude tier per drone (centred around
# FLIGHT_HEIGHT_BASE). Breaks exact-cell conflicts where two A* paths
# happen to share a narrow gap, without spreading so far that outer
# drones skim the tree canopy. N=5 → z ∈ [1.00, 1.40] m.
ALT_STEP          = 0.10  # m
FORMATION_SPACING = 0.5   # m  — goal line-reform spacing (not ingress)
# [STAGE3 FANOUT] Ingress lateral spread — drones start spread this far
# apart perpendicular to the travel axis, so the entry line spans most
# of the forest width and the swarm enters through distinct gaps.
# N=5 → lateral offsets ≈ [-2.4, -1.2, 0.0, +1.2, +2.4] m.
FANOUT_SPACING    = 1.2   # m
GRID_RESOLUTION   = 0.20  # m  — A* grid resolution
OBSTACLE_INFLATION = 0.35 # m  — safety radius added to each tree
# Retained from the convoy patch: proximity-advance tolerance for
# _fly_convoy. Used per-drone to skip the IDLE wait at each A* kink.
CONVOY_SPACING    = 0.6   # m  — retained for API compat; unused by fanout
CONVOY_ARRIVE_TOL = 0.35  # m
# [STAGE3 FANOUT] Per-drone launch stagger (s) to prevent pile-up at
# the entry row while drones accelerate to cruise.
LAUNCH_STAGGER_S  = 0.4


def _trees_to_aabb_obstacles(stage_cfg: Dict[str, Any]) -> List[Dict[str, float]]:
    """Convert CW2 stage3 tree list to the AABB obstacle format used by
    grid_map.py (keys: x, y, z, w, d, h — all in world coordinates).

    Trees are vertical cylinders; we bound them with a square cross-
    section of side equal to the cylinder diameter so the existing
    OccupancyGrid3D axis-aligned box machinery can mark them occupied.
    """
    sc     = stage_cfg["stage_center"]
    diam   = float(stage_cfg["obstacle_diameter"])
    height = float(stage_cfg["obstacle_height"])
    obs    = []
    for pos in stage_cfg["obstacles"]:
        obs.append({
            "x": pos[0] + sc[0],
            "y": pos[1] + sc[1],
            "z": height / 2.0,   # z is the vertical centre
            "w": diam,
            "d": diam,
            "h": height,
        })
    return obs


def _build_grid(obstacles: List[Dict[str, float]],
                start_xyz: List[float],
                goal_xyz:  List[float],
                n_drones:  int,
                lateral_offsets: List[float],
                resolution: float,
                inflation:  float) -> OccupancyGrid3D:
    """Build an OccupancyGrid3D that covers all drone start/goal positions
    and all tree obstacles, inflated by *inflation* metres.

    We add a generous bounding margin so paths never hit the grid edge.
    """
    margin = 2.0  # m

    # Collect all relevant XY positions to determine grid extents
    all_x = [start_xyz[0], goal_xyz[0]]
    all_y = [start_xyz[1], goal_xyz[1]]
    for lat in lateral_offsets:
        # lateral offsets are perpendicular to heading; heading ≈ x-axis here
        all_x += [start_xyz[0], goal_xyz[0]]
        all_y += [start_xyz[1] + lat, goal_xyz[1] + lat]
    for o in obstacles:
        all_x.append(o["x"])
        all_y.append(o["y"])

    min_x = min(all_x) - margin
    max_x = max(all_x) + margin
    min_y = min(all_y) - margin
    max_y = max(all_y) + margin
    min_z = 0.0
    max_z = FLIGHT_HEIGHT_BASE + (n_drones // 2) * ALT_STEP + 1.0

    grid = OccupancyGrid3D(
        min_x=min_x, max_x=max_x,
        min_y=min_y, max_y=max_y,
        min_z=min_z, max_z=max_z,
        resolution_m=resolution,
    )

    for o in obstacles:
        half_w = o["w"] / 2.0 + inflation
        half_d = o["d"] / 2.0 + inflation
        half_h = o["h"] / 2.0 + inflation
        grid.mark_box_occupied(
            min_x=o["x"] - half_w, max_x=o["x"] + half_w,
            min_y=o["y"] - half_d, max_y=o["y"] + half_d,
            min_z=o["z"] - half_h, max_z=o["z"] + half_h,
        )
    return grid


def _plan_drone_path(grid: OccupancyGrid3D,
                     start: List[float],
                     goal:  List[float]) -> Optional[List[List[float]]]:
    """Plan a 3-D A* path for one drone from start to goal.

    Returns a list of (x, y, z) world-frame waypoints, or None on failure.
    The path is simplified by greedy line-of-sight pruning (string pulling)
    to reduce unnecessary waypoints.
    """
    start_cell = grid.world_to_grid(*start)
    goal_cell  = grid.world_to_grid(*goal)

    cells = astar_search_3d(grid, start_cell, goal_cell)
    if cells is None:
        return None

    world_path = cells_to_world_path_3d(grid, cells)
    simplified = simplify_world_path_3d(grid, world_path)
    return simplified


def _fly_path(drone, waypoints: List[List[float]],
              speed: float, verbose: bool) -> None:
    """Legacy entry point — kept for compatibility. Delegates to
    _fly_convoy with zero initial delay and the default proximity
    tolerance."""
    _fly_convoy(drone, waypoints, initial_delay_s=0.0, speed=speed)


def _fly_convoy(drone, waypoints: List[List[float]],
                initial_delay_s: float,
                speed: float,
                tol_m: float = CONVOY_ARRIVE_TOL,
                per_wp_timeout: float = 30.0) -> None:
    """[STAGE3 CONVOY] Fly a drone along a convoy path with a staggered
    start and proximity-based advance between waypoints.

    Each drone in the convoy uses the SAME path as the leader but delays
    its first waypoint command by `initial_delay_s` seconds. This turns
    column spacing into temporal separation, so when the path bends
    around a tree every drone traces the same curve time-shifted.

    Parameters
    ----------
    drone           : DroneAgent
    waypoints       : list of [x, y, z] in world frame
    initial_delay_s : seconds to wait before issuing the first cmd_goto
                      (= drone_index * CONVOY_SPACING / speed)
    speed           : go_to cruise speed (m/s)
    tol_m           : XY proximity tolerance for advancing to next wp
    per_wp_timeout  : per-waypoint safety timeout (s)
    """
    if initial_delay_s > 0.0:
        time.sleep(initial_delay_s)
    tol2 = tol_m * tol_m
    for wp in waypoints:
        drone.cmd_goto(
            wp[0], wp[1], wp[2],
            speed=speed,
            yaw_mode=YawMode.PATH_FACING,
        )
        deadline = time.time() + per_wp_timeout
        while time.time() < deadline:
            p = drone.xyz
            dx = p[0] - wp[0]
            dy = p[1] - wp[1]
            if dx * dx + dy * dy < tol2:
                break
            time.sleep(0.05)


def run_stage3(conductor: SwarmConductor,
               stage_cfg: Dict[str, Any],
               speed: float = FLIGHT_SPEED,
               spacing: float = FORMATION_SPACING,
               verbose: bool = True) -> None:
    """[STAGE3 FANOUT] Execute Stage 3: forest traversal with N parallel
    A* paths and a wide lateral fan-out at ingress.

    Supersedes the convoy approach. Each drone plans its own path through
    the forest from a fanned-out start XY to a tightly-grouped goal XY.
    Tight altitude staggering (ALT_STEP centred around FLIGHT_HEIGHT_BASE)
    breaks exact-cell conflicts; per-drone launch staggering prevents
    pile-up at the start row. Proximity-based waypoint advance via
    _fly_convoy keeps each drone moving smoothly through the A* kinks.

    Parameters
    ----------
    conductor  : SwarmConductor
    stage_cfg  : the 'stage3' dict from the scenario YAML
    speed      : cruise speed (m/s)
    spacing    : lateral spacing for the goal line reform (NOT ingress;
                 ingress uses FANOUT_SPACING).
    verbose    : print progress messages
    """
    sc = stage_cfg["stage_center"]
    start_xy = [sc[0] + stage_cfg["start_point"][0],
                sc[1] + stage_cfg["start_point"][1]]
    goal_xy  = [sc[0] + stage_cfg["end_point"][0],
                sc[1] + stage_cfg["end_point"][1]]

    obstacles = _trees_to_aabb_obstacles(stage_cfg)
    n = conductor.n

    # Travel axis (from YAML start to YAML goal). Perpendicular to this
    # is the lateral axis along which we fan the drones out at ingress.
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    heading = math.atan2(dy, dx)
    perp_angle = heading + math.pi / 2.0

    # Centred index in [-(n-1)/2 … +(n-1)/2]. For N=5: [-2, -1, 0, 1, 2].
    def _ci(i: int) -> float:
        return i - (n - 1) / 2.0

    # Per-drone lateral offsets at start (wide fanout) and at goal
    # (tight reform spacing), plus per-drone altitude tier.
    lateral_start = [_ci(i) * FANOUT_SPACING for i in range(n)]
    lateral_goal  = [_ci(i) * spacing        for i in range(n)]
    z_by_drone    = [FLIGHT_HEIGHT_BASE + _ci(i) * ALT_STEP
                     for i in range(n)]

    # Rotate lateral offsets into the world frame (perpendicular to
    # heading) to build each drone's start and goal XYZ.
    starts: List[List[float]] = []
    goals:  List[List[float]] = []
    for i in range(n):
        ls, lg = lateral_start[i], lateral_goal[i]
        starts.append([
            start_xy[0] + ls * math.cos(perp_angle),
            start_xy[1] + ls * math.sin(perp_angle),
            z_by_drone[i],
        ])
        goals.append([
            goal_xy[0]  + lg * math.cos(perp_angle),
            goal_xy[1]  + lg * math.sin(perp_angle),
            z_by_drone[i],
        ])

    if verbose:
        print(f"[Stage 3] Start {start_xy} → Goal {goal_xy}, "
              f"heading {math.degrees(heading):.1f}°, "
              f"{len(obstacles)} trees, {n} drones (parallel A*).")
        print(f"[Stage 3]   Ingress lateral fanout: "
              f"{lateral_start[0]:+.2f} … {lateral_start[-1]:+.2f} m")
        print(f"[Stage 3]   Altitude tiers: "
              f"{min(z_by_drone):.2f} … {max(z_by_drone):.2f} m")

    # --- Build a shared occupancy grid covering all drones' extents. ---
    grid = _build_grid(
        obstacles=obstacles,
        start_xyz=starts[0],
        goal_xyz=goals[0],
        n_drones=n,
        lateral_offsets=lateral_start,
        resolution=GRID_RESOLUTION,
        inflation=OBSTACLE_INFLATION,
    )

    # --- Plan one A* per drone. ---
    paths: List[List[List[float]]] = []
    for i in range(n):
        path = _plan_drone_path(grid, starts[i], goals[i])
        if path is None:
            if verbose:
                print(f"[Stage 3] WARNING: A* found no path for drone {i}; "
                      "using straight-line fallback. "
                      "Raise OBSTACLE_INFLATION or FANOUT_SPACING.")
            path = [starts[i], goals[i]]
        else:
            if verbose:
                print(f"[Stage 3]   Drone {i}: {len(path)} waypoints "
                      f"(start x-offset {lateral_start[i]:+.2f} m, "
                      f"z {z_by_drone[i]:.2f} m)")
        paths.append(path)

    # --- Move each drone to its fanned-out start position. ---
    conductor.set_all_leds("green")
    if verbose:
        print("[Stage 3] Moving drones to fanned-out start positions…")
    conductor.goto_positions(
        positions=starts,
        speed=speed,
        yaw_mode=YawMode.PATH_FACING,
    )

    # --- Fire all drones in parallel, with small launch stagger. ---
    if verbose:
        print(f"[Stage 3] Parallel traversal: stagger "
              f"{LAUNCH_STAGGER_S:.2f} s/drone.")
    conductor.set_all_leds("cyan")

    import threading

    def fly_member(i: int) -> None:
        _fly_convoy(
            conductor.drones[i],
            waypoints=paths[i],
            initial_delay_s=i * LAUNCH_STAGGER_S,
            speed=speed,
        )

    threads = [
        threading.Thread(target=fly_member, args=(i,), daemon=True)
        for i in range(n)
    ]
    for t in threads:
        t.start()
    max_t = (n - 1) * LAUNCH_STAGGER_S + 180.0
    for t in threads:
        t.join(timeout=max_t)

    # --- Reform to a tight line at the goal (centred on YAML goal). ---
    if verbose:
        print("[Stage 3] Reforming (line) at goal…")
    conductor.goto_formation(
        centroid_xyz=goal_xy + [FLIGHT_HEIGHT_BASE],
        heading_rad=heading,
        formation="line",
        spacing=spacing,
        speed=speed,
    )
    conductor.set_formation_leds("line")

    if verbose:
        print("[Stage 3] Complete.")

# [STAGE3 FANOUT] applied: supersedes the convoy approach. Drones now
# fan out laterally at ingress (FANOUT_SPACING = 1.2 m) and each plans
# its own A* path through the forest, with tight altitude staggering
# (ALT_STEP = 0.10 m) for cell-conflict breakage and a small per-drone
# launch stagger (LAUNCH_STAGGER_S = 0.4 s). _fly_convoy is retained
# unchanged. See patch_stage3_fanout.py rationale.
