"""swarm_conductor.py – Centralised multi-drone conductor.

SwarmConductor owns N DroneAgent objects and provides a high-level API
for collective arm/takeoff/land and formation-flight commands.

Control strategy (centralised leader-virtual-centre):
  1. A virtual centroid waypoint is computed externally (by a stage module).
  2. Per-drone target positions are computed as:
         pos_i = centroid + R(heading) · offset_i
     where R is the rotation from heading-aligned frame to world frame
     and offset_i comes from FormationManager.
  3. All drones receive go_to commands simultaneously; the conductor
     blocks until every drone's behaviour is IDLE (goal reached).

Thread safety: DroneInterface spin threads are managed internally by
AS2; the conductor only touches drone objects from the calling thread.
"""

import math
import time
from typing import List, Optional

from as2_msgs.msg import YawMode

from swarm.drone_agent import DroneAgent
from formation.formation_manager import FormationManager

# Minimum allowed XY separation between any two target positions.
# Below this, a formation command is refused to prevent collision.
_MIN_SEPARATION_M = 0.30   # metres (crazyfly body radius ≈ 0.09 m + margin)


class SwarmConductor:
    """Centralised conductor for a fixed-size swarm of DroneAgents.

    Parameters
    ----------
    namespaces : list of str
        Ordered list of drone namespaces (e.g. ["drone0", …, "drone4"]).
        Drone 0 is the logical leader / front drone in every formation.
    verbose : bool
        Passed to each DroneAgent constructor.
    use_sim_time : bool
        Passed to each DroneAgent constructor.
    """

    def __init__(self, namespaces: List[str],
                 verbose: bool = False,
                 use_sim_time: bool = True) -> None:
        self.drones: List[DroneAgent] = [
            DroneAgent(ns, verbose=verbose, use_sim_time=use_sim_time)
            for ns in namespaces
        ]
        self.n: int = len(self.drones)

    # ------------------------------------------------------------------ #
    #  Arm / mode                                                          #
    # ------------------------------------------------------------------ #

    def arm_and_offboard(self) -> bool:
        """Arm all drones and switch to offboard mode.

        Returns True only if every drone succeeds at both steps.
        """
        ok = True
        for drone in self.drones:
            ok = drone.arm()      and ok
            ok = drone.offboard() and ok
        return ok

    # ------------------------------------------------------------------ #
    #  Collective takeoff / land                                           #
    # ------------------------------------------------------------------ #

    def takeoff(self, height: float = 1.2,
                speed: float = 0.7,
                timeout: float = 30.0) -> bool:
        """Command all drones to take off to *height* simultaneously.

        Returns True when all drones have reached the target height
        (or False on timeout).
        """
        for drone in self.drones:
            drone.cmd_takeoff(height, speed)
        return self.wait_all(timeout)

    def land(self, speed: float = 0.4,
             timeout: float = 30.0) -> bool:
        """Command all drones to land simultaneously."""
        for drone in self.drones:
            drone.cmd_land(speed)
        return self.wait_all(timeout)

    # ------------------------------------------------------------------ #
    #  Formation flight                                                    #
    # ------------------------------------------------------------------ #

    def goto_formation(self,
                       centroid_xyz: List[float],
                       heading_rad: float,
                       formation: str,
                       spacing: float,
                       speed: float = 1.0,
                       yaw_mode: int = YawMode.PATH_FACING,
                       yaw_angle: Optional[float] = None,
                       timeout: float = 60.0) -> bool:
        """Move every drone to its position within a formation.

        The formation is centred on *centroid_xyz* and oriented so that
        the local x-axis (forward) points in the direction *heading_rad*.

        Parameters
        ----------
        centroid_xyz : [x, y, z]  –  formation centre in the earth frame
        heading_rad  : world yaw of the formation's forward direction
        formation    : formation name understood by FormationManager
        spacing      : inter-drone spacing (metres)
        speed        : cruise speed for go_to commands (m/s)
        yaw_mode     : YawMode.PATH_FACING (default) or YawMode.FIXED_YAW
        yaw_angle    : used only when yaw_mode == FIXED_YAW
        timeout      : seconds before wait_all returns False

        Returns True when all drones have reached their targets.
        Raises RuntimeError if any pair of targets is too close.
        """
        local_offsets = FormationManager.get_offsets(formation, self.n, spacing)

        if local_offsets is None:
            # "free" formation – drones hold current positions
            return True

        world_offsets = FormationManager.rotate_to_world(local_offsets, heading_rad)
        targets = [
            [centroid_xyz[0] + dx,
             centroid_xyz[1] + dy,
             centroid_xyz[2]]
            for dx, dy in world_offsets
        ]

        self._assert_targets_safe(targets)

        for drone, (tx, ty, tz) in zip(self.drones, targets):
            drone.cmd_goto(tx, ty, tz,
                           speed=speed,
                           yaw_mode=yaw_mode,
                           yaw_angle=yaw_angle)

        return self.wait_all(timeout)

    def goto_positions(self,
                       positions: List[List[float]],
                       speed: float = 1.0,
                       yaw_mode: int = YawMode.PATH_FACING,
                       yaw_angle: Optional[float] = None,
                       timeout: float = 60.0) -> bool:
        """Send each drone to an explicitly specified [x, y, z] position.

        *positions* must have the same length as the number of drones.
        Useful for stage-specific manoeuvres where generic formation
        geometry is too rigid (e.g. window traversal approach points).
        """
        if len(positions) != self.n:
            raise ValueError(
                f"Expected {self.n} positions, got {len(positions)}"
            )
        self._assert_targets_safe(positions)
        for drone, pos in zip(self.drones, positions):
            drone.cmd_goto(pos[0], pos[1], pos[2],
                           speed=speed,
                           yaw_mode=yaw_mode,
                           yaw_angle=yaw_angle)
        return self.wait_all(timeout)

    # ------------------------------------------------------------------ #
    #  Wait helpers                                                        #
    # ------------------------------------------------------------------ #

    def wait_all(self, timeout: float = 60.0) -> bool:
        """Block until every drone's active behaviour reaches IDLE.

        Polls at 50 ms intervals to avoid spinning the CPU.
        Returns True on success, False on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(d.behaviour_idle() for d in self.drones):
                return True
            time.sleep(0.05)
        return False

    def wait_for_poses(self, timeout: float = 10.0) -> bool:
        """Block until all drones have published at least one pose.

        The DroneInterfaceBase initialises position to [0, 0, 0], which
        is indistinguishable from a drone genuinely at the origin.  We
        instead wait until the z component is > 0 (drones spawn above
        ground) as a proxy for a real pose update.

        Returns True when poses are available; False on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(d.xyz[2] > 0.05 for d in self.drones):
                return True
            time.sleep(0.1)
        return False

    # ------------------------------------------------------------------ #
    #  Pose utilities                                                      #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> List[List[float]]:
        """Return current [x, y, z] position of every drone."""
        return [d.xyz for d in self.drones]

    def get_centroid(self) -> List[float]:
        """Mean [x, y, z] of all drone positions."""
        positions = self.get_positions()
        return [
            sum(p[i] for p in positions) / self.n
            for i in range(3)
        ]

    # ------------------------------------------------------------------ #
    #  LED                                                                 #
    # ------------------------------------------------------------------ #

    def set_all_leds(self, colour: str) -> None:
        """Set every drone's LED to a named palette colour."""
        for drone in self.drones:
            drone.set_led(colour)

    def set_formation_leds(self, formation: str) -> None:
        """Set LEDs to the colour associated with a formation name."""
        colour = FormationManager.COLOURS.get(formation, "off")
        self.set_all_leds(colour)

    # ------------------------------------------------------------------ #
    #  Safety                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_targets_safe(targets: List[List[float]]) -> None:
        """Raise RuntimeError if any two targets are closer than the
        minimum allowed separation in the XY plane."""
        n = len(targets)
        for i in range(n):
            for j in range(i + 1, n):
                dx = targets[i][0] - targets[j][0]
                dy = targets[i][1] - targets[j][1]
                dist = math.hypot(dx, dy)
                if dist < _MIN_SEPARATION_M:
                    raise RuntimeError(
                        f"Target pair ({i}, {j}) only {dist:.3f} m apart "
                        f"(minimum is {_MIN_SEPARATION_M} m). "
                        "Reduce formation speed, increase spacing, or "
                        "check the formation configuration."
                    )

    # ------------------------------------------------------------------ #
    #  Shutdown                                                            #
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Cleanly shut down every drone's ROS2 node."""
        for drone in self.drones:
            drone.shutdown()
