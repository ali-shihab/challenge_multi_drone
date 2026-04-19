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
# [STAGE2 PREALIGN] COLUMN_SPACING tightened 0.55 -> 0.40 so a 5-drone
# column (4*0.40 = 1.6 m) fits between walls at y=7.5 and y=3.5 with
# headroom. APPROACH_OFFSET reduced 2.0 -> 1.0 so the column tail stays
# well clear of the OTHER wall when compressed at approach_y. EXIT_OFFSET
# raised 2.0 -> 2.5 so every drone fully clears the traversed wall before
# regrouping (tail leads leader by column_length on exit).
COLUMN_SPACING     = 0.40  # m  — longitudinal spacing while queuing
APPROACH_OFFSET    = 1.0   # m  — how far in front of the wall to form up
EXIT_OFFSET        = 2.5   # m  — how far past the wall before reforming
# [STAGE2 NORTH START] Extra offset used to pre-position drones on the
# far north side of window 1 before the window-traversal loop starts.
PRE_APPROACH_OFFSET = 3.5   # metres north of window 1 on approach
INTER_DRONE_DELAY  = 2.0   # s  — legacy constant, unused after columnN
                            #       traversal (retained for compat)


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
    """[STAGE2 COLUMN] Thread the swarm through a window as a columnN.

    Assumes the swarm is already compressed into a columnN at
    (gap_x, approach_y, gap_z) with heading aligned along -Y — which is
    exactly what run_stage2 does immediately before calling us.

    One goto_formation command moves the column's virtual centroid from
    approach_y to exit_y. Because columnN puts every drone at x_local=0,
    each drone's x is locked to gap_x both at start and end, and (since
    go_to interpolates linearly in world frame) x stays at gap_x
    throughout. The column flows through the gap like a snake: drone i
    crosses the wall plane at t = (approach_y - wall_y + i·spacing)/v,
    so physical separation in y enforces temporal staggering with no
    explicit inter-drone delay.

    Parameters
    ----------
    conductor  : the swarm conductor (already in columnN behind the gap)
    window     : window dict from _parse_windows()
    approach_y : Y coordinate of the approach staging line (before wall)
    exit_y     : Y coordinate of the exit staging line (after wall)
    speed      : cruise speed (m/s); traversal slows to 0.7× through gap
    verbose    : print progress messages
    """
    gx = window["gap_x"]
    gz = window["gap_z"]
    wy = window["wall_y"]

    # Travel heading: from approach_y to exit_y with exit < approach is -Y.
    # (math.atan2 handles the sign correctly for any approach/exit pair.)
    heading_rad = math.atan2(exit_y - approach_y, 0.0)

    if verbose:
        n = conductor.n
        column_len = (n - 1) * COLUMN_SPACING
        print(
            f"[Stage 2]   Threading column ({n} drones, "
            f"{column_len:.2f} m long) through gap at "
            f"x={gx:.2f} wall_y={wy:.2f} z={gz:.2f} (gap {window['gap_width']:.1f} m)"
        )

    # Single command for the whole column: centroid from approach to exit.
    # columnN keeps drone 0 at the centroid, drones 1..n-1 staggered
    # behind along +heading direction. With heading=-Y (travel direction)
    # the "behind" direction is +Y, i.e. all drones start/end at gap_x,
    # spaced by COLUMN_SPACING in y.
    conductor.goto_formation(
        centroid_xyz=[gx, exit_y, gz],
        heading_rad=heading_rad,
        formation="columnN",
        spacing=COLUMN_SPACING,
        speed=speed * 0.7,   # slow during window crossing for safety
    )

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

    # [STAGE2 PREALIGN] travel_heading survives across iterations so the
    # post-loop final reform can use the last-segment heading.
    travel_heading = math.atan2(-1.0, 0.0)
    # [STAGE2 NORTH START] Pre-position swarm well NORTH of window 1
    # before entering the traversal loop, so the dynamic side-detection
    # consistently routes drones SOUTH through each window.
    if windows_in_order:
        _first_win = windows_in_order[0]
        _north_y = _first_win["wall_y"] + PRE_APPROACH_OFFSET
        _gx = _first_win["gap_x"]
        _gz = _first_win["gap_z"]
        if verbose:
            print(f"[Stage 2] Pre-approach: moving to "
                  f"({_gx:.1f}, {_north_y:.1f}, {_gz:.1f}) — north of window 1.")
        conductor.goto_formation(
            centroid_xyz=[_gx, _north_y, _gz],
            heading_rad=math.atan2(-1.0, 0.0),
            formation="line",
            spacing=spacing,
            speed=speed,
        )

    for win in windows_in_order:
        wall_y = win["wall_y"]
        gx     = win["gap_x"]
        gz     = win["gap_z"]

        # [STAGE2 PREALIGN] Pick travel side based on swarm's current
        # centroid vs the wall. approach_y must be on the drones' OWN
        # side so compression doesn't clip the wall; exit_y on the far
        # side. travel_heading points from approach toward exit.
        cpos = conductor.get_centroid()
        drones_above = cpos[1] > wall_y
        if drones_above:
            approach_y     = wall_y + APPROACH_OFFSET
            exit_y         = wall_y - EXIT_OFFSET
            travel_heading = math.atan2(-1.0, 0.0)   # -Y
            direction_sign = -1
        else:
            approach_y     = wall_y - APPROACH_OFFSET
            exit_y         = wall_y + EXIT_OFFSET
            travel_heading = math.atan2( 1.0, 0.0)   # +Y
            direction_sign = +1

        if verbose:
            direction_label = "-Y" if drones_above else "+Y"
            print(f"[Stage 2] Window {win['key']}: "
                  f"gap ({gx:.2f}, {wall_y:.2f}, {gz:.2f}), "
                  f"width {win['gap_width']:.1f} m "
                  f"(drones at y={cpos[1]:.2f}, travel {direction_label})")

        # [STAGE2 PREALIGN] Any OTHER wall whose y lies strictly between
        # our current y and this wall's approach_y must be crossed
        # through its own gap, not clipped at the wrong x. Detour.
        y_lo = min(cpos[1], approach_y)
        y_hi = max(cpos[1], approach_y)
        intermediate = [
            ow for ow in windows_in_order
            if ow is not win and y_lo < ow["wall_y"] < y_hi
        ]
        intermediate.sort(key=lambda w: abs(w["wall_y"] - cpos[1]))

        conductor.set_all_leds("white")
        cur_y = cpos[1]

        # ---- Step 1a: detour through every intermediate wall's gap ----
        for other in intermediate:
            # Lateral+z to other.gap_x at current y (no y change → no
            # wall is crossed by any drone during this move).
            conductor.goto_formation(
                centroid_xyz=[other["gap_x"], cur_y, other["gap_z"]],
                heading_rad=travel_heading,
                formation="columnN",
                spacing=COLUMN_SPACING,
                speed=speed,
            )
            # Slide through the other wall's gap to its far side
            # (direction_sign of travel, by EXIT_OFFSET past).
            past_y = other["wall_y"] + direction_sign * EXIT_OFFSET
            conductor.goto_formation(
                centroid_xyz=[other["gap_x"], past_y, other["gap_z"]],
                heading_rad=travel_heading,
                formation="columnN",
                spacing=COLUMN_SPACING,
                speed=speed,
            )
            cur_y = past_y

        # ---- Step 1b: compress at THIS wall's gap_x at current y -----
        # Lateral+z to (gx, cur_y, gz). All intermediates are now on
        # the correct side so no wall lies between cur_y and approach_y.
        conductor.goto_formation(
            centroid_xyz=[gx, cur_y, gz],
            heading_rad=travel_heading,
            formation="columnN",
            spacing=COLUMN_SPACING,
            speed=speed,
        )

        # ---- Step 1c: slide forward to approach_y (same side) --------
        conductor.goto_formation(
            centroid_xyz=[gx, approach_y, gz],
            heading_rad=travel_heading,
            formation="columnN",
            spacing=COLUMN_SPACING,
            speed=speed,
        )

        # ---- Step 2: thread the column through the window ------------
        _through_window(conductor, win,
                        approach_y=approach_y,
                        exit_y=exit_y,
                        speed=speed,
                        verbose=verbose)

        # ---- Step 3: regroup in line on the exit side ----------------
        conductor.goto_formation(
            centroid_xyz=[gx, exit_y, gz],
            heading_rad=travel_heading,
            formation="line",
            spacing=spacing,
            speed=speed,
        )
        conductor.set_formation_leds("line")

    if verbose:
        print("[Stage 2] Complete — reforming into requested formation.")

    # [STAGE2 PREALIGN] Final reform uses the LAST iteration's travel
    # heading so the formation ends oriented along the actual direction
    # the swarm was moving, not a hardcoded -Y.
    last_centroid = conductor.get_centroid()
    conductor.goto_formation(
        centroid_xyz=last_centroid,
        heading_rad=travel_heading,
        formation=entry_formation,
        spacing=spacing,
        speed=speed,
    )
    conductor.set_formation_leds(entry_formation)

    if verbose:
        print("[Stage 2] Complete.")

# [STAGE2 NORTH START] applied: pre-loop pre-positioning at
# (window1.gap_x, window1.wall_y + 3.5, window1.gap_z) inserted
# before the traversal loop. See patch_stage2_north_start.py.
