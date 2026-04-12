"""formation_manager.py – Formation geometry for the CW2 swarm.

FormationManager.get_offsets(name, n, spacing) returns a list of n
per-drone position offsets from the formation centroid, expressed in
the heading-aligned local frame:

    x_local  =  forward  (direction of travel)
    y_local  =  left

These are rotated into the world / earth frame by rotate_to_world().

Formation names match the scenario YAML exactly:
    "line", "v", "square", "orbit", "grid", "staggered", "columnN", "free"

The module is pure Python (no ROS dependencies) so it can be unit-tested
without a running ROS2 environment.
"""

import math
from typing import List, Optional

# Type alias: a 2-D offset [dx_local, dy_local]
Offset2D = List[float]


class FormationManager:
    """Static factory for formation offset lists.

    All methods are class-methods or static-methods; no instance is needed.
    """

    # Supported formation names (matches scenario YAML)
    NAMES = frozenset({
        "line", "v", "square", "orbit", "grid", "staggered", "columnN", "free"
    })

    # Suggested LED colour per formation (for visual differentiation)
    COLOURS = {
        "line":      "red",
        "v":         "green",
        "square":    "blue",
        "orbit":     "yellow",
        "grid":      "cyan",
        "staggered": "magenta",
        "columnN":   "white",
        "free":      "off",
    }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_offsets(cls, name: str, n: int,
                    spacing: float) -> Optional[List[Offset2D]]:
        """Return n heading-frame offsets [dx, dy] from the centroid.

        Returns None for 'free' (drones hold current positions).
        Raises ValueError for unknown names.

        Drone index 0 occupies the 'lead' position in every formation.
        """
        if name not in cls.NAMES:
            raise ValueError(
                f"Unknown formation '{name}'. "
                f"Valid names: {sorted(cls.NAMES)}"
            )
        if n <= 0:
            return []
        if name == "free":
            return None

        fn = {
            "line":      cls._line,
            "v":         cls._v,
            "square":    cls._square,
            "orbit":     cls._orbit,
            "grid":      cls._grid,
            "staggered": cls._staggered,
            "columnN":   cls._column,
        }[name]
        return fn(n, spacing)

    @staticmethod
    def rotate_to_world(offsets: List[Offset2D],
                        heading_rad: float) -> List[Offset2D]:
        """Rotate heading-frame offsets into the world (earth) frame.

        Uses the standard 2-D rotation matrix:
            R = [[cos θ, -sin θ],
                 [sin θ,  cos θ]]
        where θ = heading_rad is the yaw of the formation's direction of
        travel, measured CCW from the +x world axis.
        """
        c = math.cos(heading_rad)
        s = math.sin(heading_rad)
        return [
            [c * dx - s * dy, s * dx + c * dy]
            for dx, dy in offsets
        ]

    # ------------------------------------------------------------------ #
    #  Individual formation implementations                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _line(n: int, spacing: float) -> List[Offset2D]:
        """Lateral line perpendicular to the heading direction.

        Drones are evenly distributed about y=0 in the local frame.
        All drones share x=0 (same forward position).

            ← y+ (left)
        [4] [3] [2] [1] [0]   (n=5, drone 0 at centre)
        """
        return [[0.0, (i - (n - 1) / 2.0) * spacing] for i in range(n)]

    @staticmethod
    def _v(n: int, spacing: float) -> List[Offset2D]:
        """V-shape with drone 0 at the forward tip.

        Arms extend rearward and outward at 45°:
            odd indices  → left  arm (y > 0)
            even nonzero → right arm (y < 0)

             0
           1   2
         3       4
        """
        offsets = [[0.0, 0.0]]  # lead drone at tip
        for i in range(1, n):
            depth = math.ceil(i / 2)          # how many steps back
            side  = 1 if i % 2 == 1 else -1  # left (+y) or right (-y)
            offsets.append([
                -depth * spacing,
                 side  * depth * spacing,
            ])
        return offsets

    @staticmethod
    def _square(n: int, spacing: float) -> List[Offset2D]:
        """Square arrangement: drone 0 at centre, others at corners then
        mid-edges.  For n ≤ 5, canonical square.

        Assign positions in priority order:
            centre, front-left, front-right, back-right, back-left,
            mid-front, mid-right, mid-back, mid-left, …
        """
        h = spacing / 2.0
        priority = [
            [  0.0,   0.0],   # 0: centre
            [  h,     h  ],   # 1: front-left
            [  h,    -h  ],   # 2: front-right
            [ -h,    -h  ],   # 3: back-right
            [ -h,     h  ],   # 4: back-left
            # mid-edge positions for n > 5
            [  h,     0.0],   # 5: front-mid
            [  0.0,  -h  ],   # 6: right-mid
            [ -h,     0.0],   # 7: back-mid
            [  0.0,   h  ],   # 8: left-mid
        ]
        if n <= len(priority):
            return priority[:n]
        # Fallback: outer orbit ring for remaining drones
        extra = FormationManager._orbit(n - len(priority), spacing * 1.5)
        return priority + extra

    @staticmethod
    def _orbit(n: int, spacing: float) -> List[Offset2D]:
        """Circular ring: all drones equally spaced on a circle.

        The ring radius is chosen so that the arc-length between adjacent
        drones equals *spacing*:
            radius = n * spacing / (2π)

        Drone 0 is placed at the forward apex (angle = 0 in local frame,
        i.e., at [+radius, 0]).
        """
        if n == 1:
            return [[0.0, 0.0]]
        radius = (n * spacing) / (2.0 * math.pi)
        return [
            [radius * math.cos(2.0 * math.pi * i / n),
             radius * math.sin(2.0 * math.pi * i / n)]
            for i in range(n)
        ]

    @staticmethod
    def _grid(n: int, spacing: float) -> List[Offset2D]:
        """Rectangular grid with minimum forward depth.

        cols = ceil(√n)  (number of lateral columns)
        rows = ceil(n / cols)  (number of depth rows)

        Drone assignment fills left-to-right, front-to-back:
            x_offset = -(row * spacing)   (rearward per row)
            y_offset = (col - (cols-1)/2) * spacing   (centred laterally)
        """
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        offsets = []
        for i in range(n):
            row = i // cols
            col = i % cols
            offsets.append([
                -(row * spacing),
                 (col - (cols - 1) / 2.0) * spacing,
            ])
        return offsets

    @staticmethod
    def _staggered(n: int, spacing: float) -> List[Offset2D]:
        """Two-row brick / staggered pattern.

        Front row (even drone indices): x = 0, evenly spread laterally.
        Back row  (odd drone indices):  x = -spacing, shifted half a
                                        spacing laterally to fill gaps.

        Example (n=5):
            front: drones 0, 2, 4  at y = -s, 0, +s
            back:  drones 1, 3     at y = -s/2, +s/2
        """
        front_idx = [i for i in range(n) if i % 2 == 0]
        back_idx  = [i for i in range(n) if i % 2 == 1]
        offsets: List[Optional[Offset2D]] = [None] * n

        n_f = len(front_idx)
        for j, idx in enumerate(front_idx):
            dy = (j - (n_f - 1) / 2.0) * spacing
            offsets[idx] = [0.0, dy]

        n_b = len(back_idx)
        for j, idx in enumerate(back_idx):
            # centre the back row, then shift by half-spacing
            dy = (j - (n_b - 1) / 2.0) * spacing + spacing / 2.0
            offsets[idx] = [-spacing, dy]

        return offsets  # type: ignore[return-value]

    @staticmethod
    def _column(n: int, spacing: float) -> List[Offset2D]:
        """Single file along the heading direction.

        Drone 0 leads at the front; each successive drone trails behind.
        Used for window traversal where lateral clearance is minimal.
        """
        return [[-i * spacing, 0.0] for i in range(n)]
