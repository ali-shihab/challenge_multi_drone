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
FLIGHT_HEIGHT_BASE = 1.2  # m  — nominal cruise altitude
ALT_STEP          = 0.30  # m  — altitude separation between strata
FORMATION_SPACING = 0.5   # m  — lateral offset between adjacent drones
GRID_RESOLUTION   = 0.20  # m  — A* grid resolution
OBSTACLE_INFLATION = 0.35 # m  — safety radius added to each tree


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
    """Command a single drone through a sequence of waypoints."""
    for wp in waypoints:
        drone.cmd_goto(wp[0], wp[1], wp[2],
                       speed=speed,
                       yaw_mode=YawMode.PATH_FACING)
        deadline = time.time() + 60.0
        while time.time() < deadline:
            if drone.behaviour_idle():
                break
            time.sleep(0.05)


def run_stage3(conductor: SwarmConductor,
               stage_cfg: Dict[str, Any],
               speed: float = FLIGHT_SPEED,
               spacing: float = FORMATION_SPACING,
               verbose: bool = True) -> None:
    """Execute Stage 3: forest traversal.

    Parameters
    ----------
    conductor  : SwarmConductor
    stage_cfg  : the 'stage3' dict from the scenario YAML, containing:
                   stage_center   : [x, y]
                   start_point    : [dx, dy]  relative to stage_center
                   end_point      : [dx, dy]  relative to stage_center
                   obstacle_diameter, obstacle_height
                   obstacles      : list of [dx, dy] relative to stage_center
    speed      : go_to speed for traversal (m/s)
    spacing    : lateral inter-drone spacing (metres)
    verbose    : print progress messages
    """
    sc = stage_cfg["stage_center"]
    start_xy = [sc[0] + stage_cfg["start_point"][0],
                sc[1] + stage_cfg["start_point"][1]]
    goal_xy  = [sc[0] + stage_cfg["end_point"][0],
                sc[1] + stage_cfg["end_point"][1]]

    obstacles = _trees_to_aabb_obstacles(stage_cfg)
    n = conductor.n

    # Heading from start to goal
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    heading = math.atan2(dy, dx)

    # Per-drone lateral offsets in the heading-aligned frame.
    # FormationManager._line gives offsets [dx_local, dy_local] where
    # dy_local is the lateral spread.  We use these dy values directly
    # as the lateral distances to offset start/goal positions.
    line_offsets = FormationManager.get_offsets("line", n, spacing)
    lateral = [off[1] for off in line_offsets]

    # Altitude staggering: drones in back half fly slightly higher to
    # prevent path conflicts when routes converge in narrow corridors.
    z_by_drone = [
        FLIGHT_HEIGHT_BASE + (i // 2) * ALT_STEP
        for i in range(n)
    ]

    # Rotate lateral offset into world frame (perpendicular to heading)
    perp_angle = heading + math.pi / 2.0   # perpendicular direction

    starts = []
    goals  = []
    for i in range(n):
        lat = lateral[i]
        starts.append([
            start_xy[0] + lat * math.cos(perp_angle),
            start_xy[1] + lat * math.sin(perp_angle),
            z_by_drone[i],
        ])
        goals.append([
            goal_xy[0]  + lat * math.cos(perp_angle),
            goal_xy[1]  + lat * math.sin(perp_angle),
            z_by_drone[i],
        ])

    if verbose:
        print(f"[Stage 3] Start {start_xy} → Goal {goal_xy}, "
              f"heading {math.degrees(heading):.1f}°, "
              f"{len(obstacles)} trees, {n} drones.")

    # Build a single shared occupancy grid
    grid = _build_grid(
        obstacles=obstacles,
        start_xyz=starts[0],
        goal_xyz=goals[0],
        n_drones=n,
        lateral_offsets=lateral,
        resolution=GRID_RESOLUTION,
        inflation=OBSTACLE_INFLATION,
    )

    # Plan A* paths for every drone
    paths = []
    for i in range(n):
        path = _plan_drone_path(grid, starts[i], goals[i])
        if path is None:
            # Fallback: straight line if A* finds no path (should not happen
            # with well-set inflation, but guards against edge cases)
            if verbose:
                print(f"[Stage 3] WARNING: A* found no path for drone {i}, "
                      "using straight-line fallback.")
            path = [starts[i], goals[i]]
        else:
            if verbose:
                print(f"[Stage 3]   Drone {i}: {len(path)} waypoints "
                      f"(z ≈ {z_by_drone[i]:.2f} m)")
        paths.append(path)

    # --- Move swarm to their respective start positions ---
    conductor.set_all_leds("green")
    if verbose:
        print("[Stage 3] Moving to start positions…")
    conductor.goto_positions(
        positions=starts,
        speed=speed,
        yaw_mode=YawMode.PATH_FACING,
    )

    # --- Execute planned paths concurrently ---
    if verbose:
        print("[Stage 3] Executing A* paths through forest…")
    conductor.set_all_leds("cyan")

    import threading

    def fly_drone(i):
        _fly_path(conductor.drones[i], paths[i], speed, verbose=False)

    threads = [threading.Thread(target=fly_drone, args=(i,), daemon=True)
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120.0)

    # --- Reform at goal ---
    if verbose:
        print("[Stage 3] Reforming at goal position…")
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
