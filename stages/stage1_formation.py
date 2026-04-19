"""stage1_formation.py – Stage 1: Changing Formations on a Circular Trajectory.

The swarm's virtual centroid traces a horizontal circle whose centre and
diameter are read directly from the scenario YAML.  On each waypoint the
formation is updated from a cycling list; LEDs change colour with the
formation for easy visual confirmation.

Centralised approach (this module):
  A single conductor computes all drone targets from a common centroid and
  formation offsets, then commands every drone simultaneously.  This is
  the "virtual structure" centralised strategy.

The decentralised approach (BoidsConductor) can optionally be used for
the same trajectory by passing boids=True to run_stage1(); drones then
independently apply separation/alignment/cohesion forces plus a formation-
target attraction rather than receiving direct position commands.
"""

import math
import time
from typing import List, Dict, Any

from as2_msgs.msg import YawMode

from swarm.swarm_conductor import SwarmConductor
from formation.formation_manager import FormationManager

# ---- tuneable parameters (can be overridden via scenario or CLI) ----------

CRUISE_SPEED      = 0.6   # m/s  — comfortable, stable formation transitions
TAKEOFF_HEIGHT    = 1.2   # m
FORMATION_SPACING = 0.45  # m  — tight enough for 5 drones, clear of circle
WAYPOINTS_PER_LAP = 16    # 22.5 ° increments → smooth circle
LAPS              = 1     # one full lap is enough to show all formations

# Number of waypoints between formation changes.
# With 16 wp/lap and 8 formations, change every 2 waypoints.
CHANGE_EVERY      = 2


def _circle_waypoints(cx: float, cy: float,
                      radius: float, n: int, z: float) -> List[List[float]]:
    """Generate *n* equally-spaced waypoints on a horizontal circle.

    Waypoints are ordered counter-clockwise starting at angle 0
    (positive-x direction from the centre).

    Parameters
    ----------
    cx, cy  : world coordinates of the circle centre
    radius  : circle radius (metres)
    n       : number of waypoints
    z       : constant altitude (metres)
    """
    return [
        [cx + radius * math.cos(2.0 * math.pi * i / n),
         cy + radius * math.sin(2.0 * math.pi * i / n),
         z]
        for i in range(n)
    ]


def _tangent_heading(waypoints: List[List[float]], index: int) -> float:
    """Return the tangent heading (radians) at waypoint *index*.

    The tangent is the direction from the current waypoint to the next,
    giving the forward axis for the formation frame.
    """
    nxt = (index + 1) % len(waypoints)
    dx = waypoints[nxt][0] - waypoints[index][0]
    dy = waypoints[nxt][1] - waypoints[index][1]
    return math.atan2(dy, dx)


def run_stage1(conductor: SwarmConductor,
               stage_cfg: Dict[str, Any],
               formations: List[str] = None,
               speed: float = CRUISE_SPEED,
               spacing: float = FORMATION_SPACING,
               n_waypoints: int = WAYPOINTS_PER_LAP,
               laps: int = LAPS,
               verbose: bool = True) -> None:
    """Execute Stage 1: circular formation flight.

    Parameters
    ----------
    conductor   : SwarmConductor managing the drone swarm
    stage_cfg   : the 'stage1' dict from the scenario YAML, containing:
                    stage_center : [x, y]
                    trajectory   : {diameter: float}
                    formations   : list of formation name strings (optional)
    formations  : override the formation list from the scenario
    speed       : cruise speed for go_to commands (m/s)
    spacing     : inter-drone spacing (metres)
    n_waypoints : number of waypoints per full lap
    laps        : number of complete laps to fly
    verbose     : print progress messages
    """
    cx, cy   = stage_cfg["stage_center"]
    diameter = stage_cfg["trajectory"]["diameter"]
    radius   = diameter / 2.0
    z        = TAKEOFF_HEIGHT

    # Formation list: prefer the scenario YAML list, then the override, then default
    if formations is None:
        formations = stage_cfg.get(
            "formations",
            ["line", "v", "square", "orbit", "grid", "staggered", "columnN", "free"]
        )

    if verbose:
        print(f"[Stage 1] Centre ({cx:.1f}, {cy:.1f}), radius {radius:.2f} m, "
              f"{n_waypoints} wp/lap, {laps} lap(s), "
              f"{len(formations)} formations: {formations}")

    # Pre-compute waypoints for one full lap
    waypoints = _circle_waypoints(cx, cy, radius, n_waypoints, z)

    formation_index = 0
    current_formation = formations[0]

    # Move to the first waypoint before starting the lap, using a
    # column formation to avoid lateral drift during transit.
    if verbose:
        print("[Stage 1] Transiting to start waypoint in column formation…")
    conductor.goto_formation(
        centroid_xyz=waypoints[0],
        heading_rad=_tangent_heading(waypoints, 0),
        formation="columnN",
        spacing=spacing,
        speed=speed,
    )
    conductor.set_formation_leds("columnN")

    # --- Main loop: fly the circle, changing formation periodically -------
    # [STAGE1 PIPELINE] We dispatch each waypoint non-blocking and wait
    # only for proximity (not full IDLE), so the swarm never stops
    # between waypoints. At CHANGE_EVERY=2 the drones have ~2 s of
    # forward motion to settle into each new formation, comfortably
    # more than the ~0.75 s needed to rearrange by FORMATION_SPACING.
    PROX_TOL_M   = 0.7   # loose tolerance → smooth forward motion
    PROX_TIMEOUT = 1.5    # per-waypoint proximity timeout (s)
    for lap in range(laps):
        if verbose:
            print(f"[Stage 1] Starting lap {lap + 1}/{laps}")

        for wp_idx in range(n_waypoints):
            # Change formation at regular intervals
            if wp_idx % CHANGE_EVERY == 0:
                current_formation = formations[formation_index % len(formations)]
                formation_index += 1
                conductor.set_formation_leds(current_formation)
                if verbose:
                    print(f"[Stage 1]   wp {wp_idx:2d}/{n_waypoints} → "
                          f"formation '{current_formation}'")

            heading = _tangent_heading(waypoints, wp_idx)
            targets = conductor.goto_formation(
                centroid_xyz=waypoints[wp_idx],
                heading_rad=heading,
                formation=current_formation,
                spacing=spacing,
                speed=speed,
                yaw_mode=YawMode.PATH_FACING,
                wait=False,
            )
            # "free" formation returns True (no-op) instead of a targets
            # list. Skip the proximity wait in that case.
            if isinstance(targets, list):
                conductor.wait_near_positions(
                    targets, tol_m=PROX_TOL_M, timeout=PROX_TIMEOUT,
                )

    if verbose:
        print("[Stage 1] Complete.")

# [STAGE1 RETUNE] applied: PROX_TOL_M 0.35 -> 0.7, PROX_TIMEOUT 4.0 -> 1.5.
# Retargets now happen while drone is at cruise speed, before go_to
# deceleration starts. See patch_stage1_retune.py rationale.
