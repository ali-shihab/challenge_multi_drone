"""drone_agent.py – DroneInterface extension for CW2 swarm control.

Wraps as2_python_api.DroneInterface with:
  - LED ring control
  - Unified current-behaviour handle for generic wait/idle checks
  - Convenience go-to / takeoff / land helpers (all non-blocking)
  - Pose accessors that delegate to the built-in DroneInterfaceBase
    properties (position → [x,y,z], orientation → [roll,pitch,yaw])

The AS2 DroneInterfaceBase already subscribes to
  /{namespace}/self_localization/pose
and exposes:
  self.position   → list[float]  [x, y, z]  (metres, earth frame)
  self.orientation → list[float] [roll, pitch, yaw]  (radians)
so no additional subscriptions are needed here.
"""

import math
from typing import Optional

from as2_msgs.msg import BehaviorStatus, YawMode
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA


class DroneAgent(DroneInterface):
    """Single-drone agent used by the CW2 swarm conductor.

    Parameters
    ----------
    namespace : str
        ROS2 namespace (e.g. "drone0").
    verbose : bool
        Pass-through to DroneInterface.
    use_sim_time : bool
        Pass-through to DroneInterface.
    """

    # Named LED colours (0-255 per channel)
    _LED_PALETTE: dict = {
        "red":     (255,   0,   0),
        "green":   (  0, 255,   0),
        "blue":    (  0,   0, 255),
        "yellow":  (255, 255,   0),
        "cyan":    (  0, 255, 255),
        "magenta": (255,   0, 255),
        "white":   (255, 255, 255),
        "orange":  (255, 128,   0),
        "off":     (  0,   0,   0),
    }

    def __init__(self, namespace: str,
                 verbose: bool = False,
                 use_sim_time: bool = True) -> None:
        super().__init__(namespace,
                         verbose=verbose,
                         use_sim_time=use_sim_time)
        self._ns = namespace
        # Track the most recently started behaviour module so we can
        # poll its .status without knowing which behaviour it was.
        self._active_behaviour: Optional[BehaviorHandler] = None
        self._led_pub = self.create_publisher(
            ColorRGBA, f"/{namespace}/leds/control", 10)

    # ------------------------------------------------------------------
    # Pose (delegated to DroneInterfaceBase built-in properties)
    # ------------------------------------------------------------------

    @property
    def xyz(self) -> list:
        """Current [x, y, z] position in the earth frame (metres)."""
        return self.position  # inherited from DroneInterfaceBase

    @property
    def yaw(self) -> float:
        """Current yaw angle in the earth frame (radians)."""
        return self.orientation[2]  # orientation → [roll, pitch, yaw]

    def dist_to(self, other: "DroneAgent") -> float:
        """Euclidean XY distance to another agent (metres)."""
        p, q = self.xyz, other.xyz
        return math.hypot(p[0] - q[0], p[1] - q[1])

    # ------------------------------------------------------------------
    # LED
    # ------------------------------------------------------------------

    def set_led_rgb(self, r: int, g: int, b: int) -> None:
        """Publish an LED colour (0-255 per channel)."""
        msg = ColorRGBA()
        msg.r = r / 255.0
        msg.g = g / 255.0
        msg.b = b / 255.0
        msg.a = 1.0
        self._led_pub.publish(msg)

    def set_led(self, colour: str) -> None:
        """Set LED by palette name (e.g. 'red', 'green', 'off')."""
        r, g, b = self._LED_PALETTE.get(colour, (0, 0, 0))
        self.set_led_rgb(r, g, b)

    # ------------------------------------------------------------------
    # Behaviour lifecycle
    # ------------------------------------------------------------------

    def _start(self, module_name: str, *args) -> None:
        """Start a named AS2 behaviour module non-blocking and save
        the handle so behaviour_idle() can poll it."""
        module = getattr(self, module_name)
        self._active_behaviour = module
        module(*args)

    def behaviour_idle(self) -> bool:
        """Return True when the tracked behaviour has finished
        (status == IDLE) or no behaviour has been started."""
        if self._active_behaviour is None:
            return True
        return self._active_behaviour.status == BehaviorStatus.IDLE

    # ------------------------------------------------------------------
    # Flight commands (non-blocking wrappers)
    # ------------------------------------------------------------------

    def cmd_takeoff(self, height: float = 1.2,
                    speed: float = 0.7) -> None:
        """Arm, set offboard, then take off to *height* at *speed*.
        Non-blocking; poll behaviour_idle() to detect completion."""
        self._start("takeoff", height, speed, False)

    def cmd_land(self, speed: float = 0.4) -> None:
        """Land at *speed*.  Non-blocking."""
        self._start("land", speed, False)

    def cmd_goto(self, x: float, y: float, z: float,
                 speed: float = 1.0,
                 yaw_mode: int = YawMode.PATH_FACING,
                 yaw_angle: Optional[float] = None) -> None:
        """Fly to (x, y, z) in the earth frame at *speed*.  Non-blocking.

        Parameters
        ----------
        yaw_mode : YawMode constant
            PATH_FACING  – drone faces its direction of travel (default).
            FIXED_YAW    – drone holds the angle given by *yaw_angle*.
        yaw_angle : float | None
            Used only when yaw_mode == FIXED_YAW.
        """
        self._start("go_to", x, y, z, speed,
                    yaw_mode, yaw_angle, "earth", False)
