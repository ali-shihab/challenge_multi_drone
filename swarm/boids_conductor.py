"""boids_conductor.py – Decentralised swarm conductor using the Boids model.

This module provides a drop-in alternative to SwarmConductor for the parts
of the mission that require — or can demonstrate — decentralised control.

Background
----------
Reynolds' Boids model (1987) describes emergent flocking behaviour from three
purely local rules, each computed from the states of nearby neighbours only:

  Separation  – steer away from neighbours that are too close (collision
                avoidance, short-range repulsion).
  Alignment   – steer toward the mean heading of near neighbours (velocity
                matching, medium-range).
  Cohesion    – steer toward the mean position of far neighbours (group
                attraction, long-range).

These are augmented with a fourth rule specific to CW2's formation demands:

  Formation target – steer toward the drone's assigned formation slot
                     (defined as a rigid offset from the formation centroid).

Each drone runs its own update thread at ``UPDATE_HZ`` Hz.  At every tick the
drone reads the self-reported positions and yaws of its team-mates (available
via the AS2 ``self_localization/pose`` subscriptions that DroneInterfaceBase
already maintains) and independently decides its next commanded position.
No global coordinator is involved in the velocity computation.

The ``BoidsConductor`` class presents the same high-level API as
``SwarmConductor`` (``goto_formation``, ``goto_positions``, ``takeoff``,
``land``, ``set_all_leds``, etc.) so it can be substituted with a single
command-line flag in the mission entrypoint.

Implementation notes
--------------------
AS2 uses *position* control (``go_to``), not velocity control.  Boids
produces velocity vectors; we integrate them over ``DT = 1/UPDATE_HZ``
seconds to obtain a target displacement:

    target_xyz = current_xyz + clamp(velocity, MAX_SPEED) * DT

A new ``cmd_goto`` is issued at each tick.  Because AS2's go_to behaviour
continuously tracks a moving target when issued successive commands, the
drones effectively track the Boids velocity field.

Altitude is handled separately: drones are commanded to their assigned
formation altitude at all times; the Boids x-y plane dynamics do not
alter z.

Thread safety
-------------
``DroneInterfaceBase.position`` and ``DroneInterfaceBase.orientation`` are
properties backed by a subscriber callback; they may be updated concurrently
by the AS2 spin thread.  In CPython, float attribute reads are atomic at the
interpreter level (they are single-reference loads), so no additional lock is
needed for reading ``xyz`` / ``yaw`` from multiple threads.
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

from as2_msgs.msg import YawMode

from swarm.drone_agent import DroneAgent
from formation.formation_manager import FormationManager

# --------------------------------------------------------------------------- #
#  Tuneable Boids parameters                                                   #
# --------------------------------------------------------------------------- #

UPDATE_HZ        = 10.0  # Hz – Boids update rate per drone
DT               = 1.0 / UPDATE_HZ

# Rule radii (metres, XY only)
SEP_RADIUS       = 0.60  # m – neighbours closer than this trigger separation
ALIGN_RADIUS     = 2.00  # m – neighbourhood for alignment
COHESION_RADIUS  = 3.00  # m – neighbourhood for cohesion

# Rule weights (dimensionless scale factors for the velocity contribution)
W_SEPARATION     = 2.5   # strong short-range repulsion
W_ALIGNMENT      = 0.8   # moderate velocity matching
W_COHESION       = 0.5   # gentle group attraction
W_TARGET         = 3.0   # strong pull toward formation slot

# Speed constraints (m/s)
MAX_SPEED        = 1.0   # m/s – maximum commanded speed
MIN_SPEED        = 0.05  # m/s – commands below this are suppressed

# Convergence threshold
ARRIVE_TOLERANCE = 0.15  # m  – XYZ radius considered "arrived"

# Safety minimum XY separation between any two targets
_MIN_SEPARATION_M = 0.30  # metres


# --------------------------------------------------------------------------- #
#  Low-level Boids rule functions                                              #
# --------------------------------------------------------------------------- #

def _separation(
    my_pos:    List[float],
    neighbour_positions: List[List[float]],
) -> Tuple[float, float]:
    """Steer away from neighbours that are closer than SEP_RADIUS.

    Returns an (x, y) velocity vector in m/s.  The magnitude of the
    repulsion grows as the inverse of the distance (stronger when very
    close).
    """
    vx = vy = 0.0
    for npos in neighbour_positions:
        dx = my_pos[0] - npos[0]
        dy = my_pos[1] - npos[1]
        dist = math.hypot(dx, dy)
        if 0 < dist < SEP_RADIUS:
            # Weight inversely by distance so nearby drones push harder
            scale = (SEP_RADIUS - dist) / (SEP_RADIUS * dist + 1e-9)
            vx += dx * scale
            vy += dy * scale
    return (vx, vy)


def _alignment(
    my_pos:    List[float],
    neighbour_positions: List[List[float]],
    neighbour_yaws:      List[float],
) -> Tuple[float, float]:
    """Steer toward the mean heading of neighbours within ALIGN_RADIUS.

    Yaw is used as a proxy for velocity direction (valid when drones
    operate in PATH_FACING mode).  Returns an (x, y) velocity vector.
    """
    sx = sy = 0.0
    count = 0
    for npos, nyaw in zip(neighbour_positions, neighbour_yaws):
        dx = npos[0] - my_pos[0]
        dy = npos[1] - my_pos[1]
        if math.hypot(dx, dy) < ALIGN_RADIUS:
            sx += math.cos(nyaw)
            sy += math.sin(nyaw)
            count += 1
    if count == 0:
        return (0.0, 0.0)
    # Return normalised mean heading vector (unit magnitude, scaled later)
    mag = math.hypot(sx, sy)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (sx / mag, sy / mag)


def _cohesion(
    my_pos:    List[float],
    neighbour_positions: List[List[float]],
) -> Tuple[float, float]:
    """Steer toward the mean position of neighbours within COHESION_RADIUS.

    Returns an (x, y) velocity vector pointing from the drone's current
    position toward the local centre of mass.
    """
    sx = sy = 0.0
    count = 0
    for npos in neighbour_positions:
        dx = npos[0] - my_pos[0]
        dy = npos[1] - my_pos[1]
        if math.hypot(dx, dy) < COHESION_RADIUS:
            sx += npos[0]
            sy += npos[1]
            count += 1
    if count == 0:
        return (0.0, 0.0)
    cx = sx / count - my_pos[0]
    cy = sy / count - my_pos[1]
    mag = math.hypot(cx, cy)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (cx / mag, cy / mag)


def _target_attraction(
    my_pos:    List[float],
    target:    List[float],
) -> Tuple[float, float]:
    """Steer toward the formation slot target position.

    Uses a proportional controller: the velocity magnitude equals the
    distance to the target (capped at 1.0), so the drone decelerates
    smoothly as it approaches.
    """
    dx = target[0] - my_pos[0]
    dy = target[1] - my_pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return (0.0, 0.0)
    # Magnitude: proportional to distance but capped at 1 (unit vector)
    scale = min(1.0, dist)
    return (dx / dist * scale, dy / dist * scale)


def _clamp_speed(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
    """Scale a 2-D velocity vector so its magnitude does not exceed max_speed."""
    mag = math.hypot(vx, vy)
    if mag < 1e-9 or mag <= max_speed:
        return (vx, vy)
    return (vx / mag * max_speed, vy / mag * max_speed)


# --------------------------------------------------------------------------- #
#  Per-drone update thread                                                     #
# --------------------------------------------------------------------------- #

def _boids_thread(
    idx:        int,
    drones:     List[DroneAgent],
    targets:    List[List[float]],    # formation slot targets [x, y, z]
    stop_event: threading.Event,
    speed:      float,
) -> None:
    """Boids update loop for drone at index *idx*.

    Runs at UPDATE_HZ until *stop_event* is set.  At each tick:
      1. Snapshot neighbour states.
      2. Compute the four Boids rule velocity contributions.
      3. Combine with tunable weights.
      4. Integrate to a target position and issue cmd_goto.
    """
    me = drones[idx]

    while not stop_event.is_set():
        tick_start = time.time()

        my_pos = list(me.xyz)    # [x, y, z] from AS2
        my_z   = targets[idx][2]  # maintain assigned altitude

        neighbours = [d for j, d in enumerate(drones) if j != idx]
        n_pos  = [list(d.xyz)  for d in neighbours]
        n_yaws = [d.yaw        for d in neighbours]

        # Compute rule velocity components (2-D, in XY plane)
        sep_v  = _separation(my_pos, n_pos)
        ali_v  = _alignment(my_pos, n_pos, n_yaws)
        coh_v  = _cohesion(my_pos, n_pos)
        tgt_v  = _target_attraction(my_pos, targets[idx])

        # Weighted sum
        vx = (W_SEPARATION * sep_v[0]
              + W_ALIGNMENT  * ali_v[0]
              + W_COHESION   * coh_v[0]
              + W_TARGET     * tgt_v[0])
        vy = (W_SEPARATION * sep_v[1]
              + W_ALIGNMENT  * ali_v[1]
              + W_COHESION   * coh_v[1]
              + W_TARGET     * tgt_v[1])

        vx, vy = _clamp_speed(vx, vy, MAX_SPEED)

        # Only issue a command if the velocity is above the minimum threshold
        if math.hypot(vx, vy) > MIN_SPEED:
            tx = my_pos[0] + vx * DT
            ty = my_pos[1] + vy * DT
            me.cmd_goto(tx, ty, my_z, speed=speed,
                        yaw_mode=YawMode.PATH_FACING)

        # Maintain the target update rate
        elapsed = time.time() - tick_start
        sleep_t = max(0.0, DT - elapsed)
        stop_event.wait(timeout=sleep_t)


# --------------------------------------------------------------------------- #
#  BoidsConductor                                                              #
# --------------------------------------------------------------------------- #

class BoidsConductor:
    """Decentralised swarm conductor using the Boids flocking model.

    Provides the same high-level API as ``SwarmConductor`` so it can be
    used as a drop-in replacement controlled by a ``--approach`` flag.

    Parameters
    ----------
    drones : list of DroneAgent
        Shared list of drone objects.  The same objects may also be held by a
        SwarmConductor if both conductors are constructed (only one should be
        active at a time).
    verbose : bool
        Print Boids debug output.
    """

    def __init__(
        self,
        drones:  List[DroneAgent],
        verbose: bool = False,
    ) -> None:
        self.drones:   List[DroneAgent] = drones
        self.n:        int              = len(drones)
        self.verbose:  bool             = verbose

    # ------------------------------------------------------------------ #
    #  Arm / offboard                                                      #
    # ------------------------------------------------------------------ #

    def arm_and_offboard(self) -> bool:
        ok = True
        for drone in self.drones:
            ok = drone.arm()      and ok
            ok = drone.offboard() and ok
        return ok

    # ------------------------------------------------------------------ #
    #  Collective takeoff / land                                           #
    # ------------------------------------------------------------------ #

    def takeoff(
        self,
        height:  float = 1.2,
        speed:   float = 0.7,
        timeout: float = 30.0,
    ) -> bool:
        """Simultaneous takeoff with a blocking wait."""
        for drone in self.drones:
            drone.cmd_takeoff(height, speed)
        return self.wait_all(timeout)

    def land(
        self,
        speed:   float = 0.4,
        timeout: float = 30.0,
    ) -> bool:
        for drone in self.drones:
            drone.cmd_land(speed)
        return self.wait_all(timeout)

    # ------------------------------------------------------------------ #
    #  Formation flight (Boids)                                           #
    # ------------------------------------------------------------------ #

    def goto_formation(
        self,
        centroid_xyz: List[float],
        heading_rad:  float,
        formation:    str,
        spacing:      float,
        speed:        float = 1.0,
        yaw_mode:     int   = YawMode.PATH_FACING,
        yaw_angle:    Optional[float] = None,
        timeout:      float = 60.0,
        wait:         bool  = True,
    ) -> bool:
        """Fly the swarm to a formation centred on *centroid_xyz* using Boids.

        Each drone's formation slot is the attractor in the Boids model;
        the other rules (separation, alignment, cohesion) provide local
        collision avoidance and flocking during the transit.

        Returns True when every drone is within ``ARRIVE_TOLERANCE`` of its
        target, or False on timeout.
        """
        local_offsets = FormationManager.get_offsets(formation, self.n, spacing)
        if local_offsets is None:
            return True  # "free" — hold current positions

        world_offsets = FormationManager.rotate_to_world(local_offsets, heading_rad)
        targets = [
            [centroid_xyz[0] + dx,
             centroid_xyz[1] + dy,
             centroid_xyz[2]]
            for dx, dy in world_offsets
        ]

        return self._fly_boids(targets, speed=speed, timeout=timeout)

    def goto_positions(
        self,
        positions: List[List[float]],
        speed:     float = 1.0,
        yaw_mode:  int   = YawMode.PATH_FACING,
        yaw_angle: Optional[float] = None,
        timeout:   float = 60.0,
        wait:      bool  = True,
    ) -> bool:
        """Command each drone to an explicit [x, y, z] position via Boids."""
        if len(positions) != self.n:
            raise ValueError(
                f"Expected {self.n} positions, got {len(positions)}"
            )
        return self._fly_boids(positions, speed=speed, timeout=timeout)

    # ------------------------------------------------------------------ #
    #  Internal Boids execution                                           #
    # ------------------------------------------------------------------ #

    def _fly_boids(
        self,
        targets: List[List[float]],
        speed:   float,
        timeout: float,
    ) -> bool:
        """Launch per-drone Boids update threads and block until all drones
        arrive within ``ARRIVE_TOLERANCE`` of their targets, or timeout.

        The stop event is set as soon as ALL drones have converged, which
        terminates all update threads gracefully.
        """
        stop_event = threading.Event()
        threads = [
            threading.Thread(
                target=_boids_thread,
                args=(i, self.drones, targets, stop_event, speed),
                daemon=True,
                name=f"boids_{i}",
            )
            for i in range(self.n)
        ]

        for t in threads:
            t.start()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._all_arrived(targets):
                if self.verbose:
                    print("[Boids] All drones converged.")
                break
            time.sleep(0.05)
        else:
            if self.verbose:
                dists = [
                    math.sqrt(sum((a - b) ** 2 for a, b in
                                  zip(d.xyz, tgt)))
                    for d, tgt in zip(self.drones, targets)
                ]
                print(
                    f"[Boids] Timeout. Distances: "
                    f"{[f'{dist:.2f}' for dist in dists]}"
                )

        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

        return self._all_arrived(targets)

    def _all_arrived(self, targets: List[List[float]]) -> bool:
        """Return True when every drone is within ARRIVE_TOLERANCE of its
        target in 3-D Euclidean distance."""
        for drone, tgt in zip(self.drones, targets):
            pos  = drone.xyz
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, tgt)))
            if dist > ARRIVE_TOLERANCE:
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Wait helpers                                                        #
    # ------------------------------------------------------------------ #

    def wait_all(self, timeout: float = 60.0) -> bool:
        """Block until every drone's active AS2 behaviour is IDLE."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(d.behaviour_idle() for d in self.drones):
                return True
            time.sleep(0.05)
        return False

    def wait_for_poses(self, timeout: float = 10.0) -> bool:
        """Block until all drones have published at least one pose (z is not NaN)."""
        import math
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(not math.isnan(d.xyz[2]) for d in self.drones):
                return True
            time.sleep(0.1)
        return False

    # ------------------------------------------------------------------ #
    #  Pose utilities                                                      #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> List[List[float]]:
        return [d.xyz for d in self.drones]

    def get_centroid(self) -> List[float]:
        positions = self.get_positions()
        return [
            sum(p[i] for p in positions) / self.n
            for i in range(3)
        ]

    # ------------------------------------------------------------------ #
    #  LED                                                                 #
    # ------------------------------------------------------------------ #

    def set_all_leds(self, colour: str) -> None:
        for drone in self.drones:
            drone.set_led(colour)

    def set_formation_leds(self, formation: str) -> None:
        colour = FormationManager.COLOURS.get(formation, "off")
        self.set_all_leds(colour)

    # ------------------------------------------------------------------ #
    #  Shutdown                                                            #
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        for drone in self.drones:
            drone.shutdown()
