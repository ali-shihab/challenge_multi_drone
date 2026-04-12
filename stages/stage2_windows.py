"""stage2_windows.py – Stage 2: Window Traversal.

Two walls span the full stage width.  Each has a gap of different size:

    Window 1:  gap_width = 2.0 m  (2–3 drones can pass side-by-side)
    Window 2:  gap_width = 1.0 m  (single-drone width only)

Strategy (both centralised and decentralised cases use the same approach,
since the geometry is deterministic):

    1. Read window parameters from the scenario YAML.
    2. For each window:
       a. Reform into a columnN formation (single file along heading).
       b. Compute per-drone staggered approach waypoints so that drones
          queue behind the window entry point.
       c. Command each drone through the window one at a time, waiting
          for each to clear before sending the next.
       d. After all drones are through, reform the previous formation.

Why columnar single-file rather than e.g. two-abreast for window 1?
  Single-file is robust to any window width ≥ one drone, generalises to
  the narrower window 2 without a code-path change, and avoids having to
  solve a more complex multi-drone sequencing problem.  For the assessment
  this provides a clean, general solution; the brief asks us to "consider
  how to split, rejoin, or compress the formation" – columnN is the
  natural minimal-width compressor.
"""

import math
import time
from typing import List, Dict, Any, Tuple

from as2_msgs.msg import YawMode

from swarm.swarm_conductor import SwarmConductor
from formation.formation_manager import FormationManager

# ---- tuneable parameters --------------------------------------------------

CRUISE_SPEED       = 0.5   # m/s  — slow and deliberate through windows
COLUMN_SPACING     = 0.55  # m  — longitudinal spacing while queuing
APPROACH_OFFSET    = 2.0   # m  — how far in front of the wall to form up
EXIT_OFFSET        = 2.0   # m  — how far past the wall before reforming
INTER_DRONE_DELAY  = 2.0   # s  — extra pause between consecutive drones
                            #       (on top of waiting for IDLE)


def _parse_windows(stage_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract window list from stage2 config, sorted by Y position.

    The scenario uses dict-of-dicts keyed by integer (1, 2, …).  We
    convert to a list sorted by increasing Y wall position so traversal
    goes from the side the swarm enters to the far side.
    """
    sc = stage_cfg["stage_center"]
    windows = []
    for key, w in stage_cfg["windows"].items():
        # In the scenario YAML: center = [x_offset_of_gap, y_of_wall]
        # Object placement in generate_world_from_scenario.py confirms:
        #   object x = stage_center[0]           (wall is parallel to y-axis)
        #   object y = stage_center[1] + center[1]
        # The gap (opening) is at:
        #   gap_x = stage_center[0] + center[0]
        #   gap_y = stage_center[1] + center[1]   (same as wall y)
        #   gap_z_centre = distance_floor + height / 2
        gap_x = sc[0] + w["center"][0]
        wall_y = sc[1] + w["center"][1]
        gap_z = w["distance_floor"] + w["height"] / 2.0
        windows.append({
            "key":           key,
            "gap_x":         gap_x,
            "wall_y":        wall_y,
            "gap_z":         gap_z,
            "gap_width":     w["gap_width"],
            "thickness":     w["thickness"],
        })
    # Sort by wall_y so we traverse in a consistent direction
    windows.sort(key=lambda w: w["wall_y"])
    return windows


def _through_window(conductor: SwarmConductor,
                    window: Dict[str, Any],
                    approach_y: float,
                    exit_y: float,
                    speed: float,
                    verbose: bool) -> None:
    """Fly all drones through a single window in single-file sequence.

    Each drone is commanded from its approach position to the gap centre
    and then to its exit position.  The conductor waits for each drone
    individually before sending the next.

    Parameters
    ----------
    conductor  : the swarm conductor
    window     : window dict from _parse_windows()
    approach_y : Y coordinate of the approach staging line (before wall)
    exit_y     : Y coordinate of the exit staging line (after wall)
    speed      : go_to speed (m/s)
    verbose    : print progress messages
    """
    gx = window["gap_x"]
    gz = window["gap_z"]
    wy = window["wall_y"]

    for i, drone in enumerate(conductor.drones):
        if verbose:
            print(f"[Stage 2]   Drone {i} ({drone._ns}) → window …")

        # Move this drone to its approach position (directly in front of gap)
        drone.cmd_goto(gx, approach_y, gz,
                       speed=speed,
                       yaw_mode=YawMode.PATH_FACING)
        _wait_one(drone, timeout=30.0)

        # Fly through the gap
        drone.cmd_goto(gx, wy, gz,
                       speed=speed * 0.8,   # slow right at the gap
                       yaw_mode=YawMode.PATH_FACING)
        _wait_one(drone, timeout=20.0)

        # Continue to exit position
        drone.cmd_goto(gx, exit_y, gz,
                       speed=speed,
                       yaw_mode=YawMode.PATH_FACING)
        _wait_one(drone, timeout=30.0)

        # Small delay between drones so they do not crowd the exit
        time.sleep(INTER_DRONE_DELAY)

    if verbose:
        print(f"[Stage 2]   All drones through window {window['key']}.")


def _wait_one(drone, timeout: float = 30.0) -> bool:
    """Block until a single drone's behaviour is IDLE or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if drone.behaviour_idle():
            return True
        time.sleep(0.05)
    return False


def run_stage2(conductor: SwarmConductor,
               stage_cfg: Dict[str, Any],
               entry_formation: str = "line",
               spacing: float = 0.5,
               speed: float = CRUISE_SPEED,
               verbose: bool = True) -> None:
    """Execute Stage 2: window traversal.

    Parameters
    ----------
    conductor       : SwarmConductor
    stage_cfg       : the 'stage2' dict from the scenario YAML
    entry_formation : formation to reform into after clearing all windows
    spacing         : inter-drone spacing for the reformed formation
    speed           : cruise speed (m/s)
    verbose         : print progress messages
    """
    windows = _parse_windows(stage_cfg)
    sc      = stage_cfg["stage_center"]

    if verbose:
        print(f"[Stage 2] {len(windows)} window(s) to traverse.")

    # Determine the overall travel direction through Stage 2.
    # The stage is entered from Stage 1 (higher Y) and exits toward
    # Stage 3 (lower Y), so travel is in the -Y direction.
    # Sort windows by wall_y descending so we hit them in travel order.
    windows_in_order = sorted(windows, key=lambda w: w["wall_y"], reverse=True)

    for win in windows_in_order:
        wall_y = win["wall_y"]
        gx     = win["gap_x"]
        gz     = win["gap_z"]

        # Approach from the high-Y side; exit to the low-Y side
        approach_y = wall_y + APPROACH_OFFSET
        exit_y     = wall_y - EXIT_OFFSET

        if verbose:
            print(f"[Stage 2] Window {win['key']}: "
                  f"gap ({gx:.2f}, {wall_y:.2f}, {gz:.2f}), "
                  f"width {win['gap_width']:.1f} m")

        # ---- Step 1: compress into columnN (single file) ---------------
        conductor.set_all_leds("white")
        current_pos = conductor.get_centroid()
        heading_to_window = math.atan2(wall_y - current_pos[1],
                                       gx    - current_pos[0])
        conductor.goto_formation(
            centroid_xyz=[gx, approach_y, gz],
            heading_rad=heading_to_window,
            formation="columnN",
            spacing=COLUMN_SPACING,
            speed=speed,
        )

        # ---- Step 2: thread drones through the window one at a time ----
        _through_window(conductor, win,
                        approach_y=approach_y,
                        exit_y=exit_y,
                        speed=speed,
                        verbose=verbose)

        # ---- Step 3: regroup on the exit side --------------------------
        # All drones should now be around (gx, exit_y, gz).
        # Reform them into a compact line formation before the next window.
        exit_heading = math.atan2(-1.0, 0.0)   # heading: -Y (continuing through)
        conductor.goto_formation(
            centroid_xyz=[gx, exit_y, gz],
            heading_rad=exit_heading,
            formation="line",
            spacing=spacing,
            speed=speed,
        )
        conductor.set_formation_leds("line")

    if verbose:
        print("[Stage 2] Complete — reforming into requested formation.")

    # Final reform into the requested entry formation for Stage 3
    last_centroid = conductor.get_centroid()
    conductor.goto_formation(
        centroid_xyz=last_centroid,
        heading_rad=math.atan2(-1.0, 0.0),
        formation=entry_formation,
        spacing=spacing,
        speed=speed,
    )
    conductor.set_formation_leds(entry_formation)

    if verbose:
        print("[Stage 2] Complete.")
