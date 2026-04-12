from __future__ import annotations
import math
from typing import Dict, List, Tuple
from planners.grid_map import OccupancyGrid2D, OccupancyGrid3D

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

def cells_to_world_path(grid, cells): return [grid.grid_to_world(c) for c in cells]
def cells_to_world_path_3d(grid, cells): return [grid.grid_to_world(c) for c in cells]

def simplify_world_path(grid, path_xy):
    """Greedy 2D line-of-sight string-pulling."""
    if len(path_xy)<=2: return path_xy[:]
    simplified=[path_xy[0]]; i=0
    while i<len(path_xy)-1:
        j=len(path_xy)-1
        while j>i+1:
            if grid.line_is_free_world(path_xy[i],path_xy[j]): break
            j-=1
        simplified.append(path_xy[j]); i=j
    return simplified

def simplify_world_path_3d(grid, path_3d):
    """Greedy 3D line-of-sight string-pulling (skips intermediate waypoints when LOS is clear)."""
    if len(path_3d)<=2: return path_3d[:]
    simplified=[path_3d[0]]; i=0
    while i<len(path_3d)-1:
        j=len(path_3d)-1
        while j>i+1:
            if grid.line_is_free_world(path_3d[i],path_3d[j]): break
            j-=1
        simplified.append(path_3d[j]); i=j
    return simplified

def densify_stride(path_xy, stride):
    if stride<=1 or len(path_xy)<=2: return path_xy[:]
    out=[path_xy[0]]
    for i in range(1,len(path_xy)-1,stride): out.append(path_xy[i])
    if out[-1]!=path_xy[-1]: out.append(path_xy[-1])
    return out

def heading_between(a, b, fallback_yaw):
    dx=b[0]-a[0]; dy=b[1]-a[1]
    return math.atan2(dy,dx) if abs(dx)>1e-9 or abs(dy)>1e-9 else fallback_yaw

def heading_between_3d(a, b, fallback_yaw):
    dx=b[0]-a[0]; dy=b[1]-a[1]
    return math.atan2(dy,dx) if abs(dx)>1e-9 or abs(dy)>1e-9 else fallback_yaw

def build_subgoals_from_xy_path(*, path_xy, start_z, goal_z, final_yaw):
    """Convert XY path to 3D subgoals with linearly interpolated Z."""
    if not path_xy: return []
    n=len(path_xy); out=[]
    for i,xy in enumerate(path_xy):
        z=goal_z if n==1 else start_z+(i/(n-1))*(goal_z-start_z)
        yaw=heading_between(path_xy[i],path_xy[i+1],final_yaw) if i<n-1 else final_yaw
        out.append({"x":float(xy[0]),"y":float(xy[1]),"z":float(z),"yaw":float(yaw)})
    return out

def build_subgoals_from_3d_path(*, path_3d, final_yaw):
    """Convert 3D path (list of (x,y,z)) to subgoal dicts. Yaw faces along XY projection."""
    if not path_3d: return []
    n=len(path_3d); out=[]
    for i,pt in enumerate(path_3d):
        yaw=heading_between_3d(path_3d[i],path_3d[i+1],final_yaw) if i<n-1 else final_yaw
        out.append({"x":float(pt[0]),"y":float(pt[1]),"z":float(pt[2]),"yaw":float(yaw)})
    return out
