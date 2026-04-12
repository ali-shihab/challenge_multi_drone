from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

Point3 = Tuple[float, float, float]

@dataclass
class OrderedMission:
    ordered_ids: List[str]
    original_ids: List[str]
    estimated_length_before_m: float
    estimated_length_after_m: float

def viewpoint_position(vp): return (float(vp["x"]),float(vp["y"]),float(vp["z"]))
def euclidean(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def path_length(start, ordered_ids, viewpoints):
    total=0.0; cur=start
    for vid in ordered_ids:
        nxt=viewpoint_position(viewpoints[vid]); total+=euclidean(cur,nxt); cur=nxt
    return total

def path_length_with_matrix(ordered_ids, dist_matrix, start_dists):
    if not ordered_ids: return 0.0
    total=start_dists.get(ordered_ids[0],0.0)
    for i in range(len(ordered_ids)-1):
        total+=dist_matrix.get(ordered_ids[i],{}).get(ordered_ids[i+1],0.0)
    return total

def nearest_neighbour_order(start, viewpoint_ids, viewpoints, dist_matrix=None, start_dists=None):
    remaining=set(viewpoint_ids); order=[]; cur_id=None
    while remaining:
        if cur_id is None:
            best_id=min(remaining, key=lambda v: start_dists.get(v,float("inf")) if start_dists else euclidean(start,viewpoint_position(viewpoints[v])))
        else:
            if dist_matrix is not None:
                best_id=min(remaining, key=lambda v: dist_matrix.get(cur_id,{}).get(v,float("inf")))
            else:
                cp=viewpoint_position(viewpoints[cur_id])
                best_id=min(remaining, key=lambda v: euclidean(cp,viewpoint_position(viewpoints[v])))
        order.append(best_id); cur_id=best_id; remaining.remove(best_id)
    return order

def two_opt_open_path(start, ordered_ids, viewpoints, max_passes=20, dist_matrix=None, start_dists=None):
    """2-opt improvement for open path. Uses A* dist_matrix if provided."""
    if len(ordered_ids)<4: return ordered_ids[:]
    def length(ids):
        if dist_matrix is not None and start_dists is not None:
            return path_length_with_matrix(ids,dist_matrix,start_dists)
        return path_length(start,ids,viewpoints)
    best=ordered_ids[:]; best_len=length(best); improved=True; passes=0
    while improved and passes<max_passes:
        improved=False; passes+=1
        for i in range(len(best)-1):
            for j in range(i+1,len(best)):
                candidate=best[:i]+list(reversed(best[i:j+1]))+best[j+1:]
                cand_len=length(candidate)
                if cand_len+1e-9<best_len: best=candidate; best_len=cand_len; improved=True
        if not improved: break
    return best

def compute_astar_distance_matrix(scenario, viewpoint_ids, viewpoints, start,
                                   grid_resolution=0.5, inflation_m=0.5, bounds_margin_m=2.0):
    """Compute pairwise A*-path distances. Falls back to Euclidean on error."""
    try:
        from planners.grid_map import GridConfig, build_occupancy_grid_3d_from_scenario
        from planners.astar import astar_search_3d
        from planners.path_utils import cells_to_world_path_3d
        config=GridConfig(resolution_m=grid_resolution,inflation_m=inflation_m,bounds_margin_m=bounds_margin_m)
        grid=build_occupancy_grid_3d_from_scenario(scenario,start_xyz=start,config=config)
        def _path_dist(a,b):
            cells=astar_search_3d(grid,grid.world_to_grid(a[0],a[1],a[2]),grid.world_to_grid(b[0],b[1],b[2]))
            if cells is None: return euclidean(a,b)*2.0
            pts=cells_to_world_path_3d(grid,cells)
            return sum(euclidean(pts[i],pts[i+1]) for i in range(len(pts)-1))
    except Exception:
        def _path_dist(a,b): return euclidean(a,b)

    positions={vid:viewpoint_position(viewpoints[vid]) for vid in viewpoint_ids}
    dist_matrix={}; start_dists={}
    for a in viewpoint_ids:
        dist_matrix[a]={}; pa=positions[a]; start_dists[a]=_path_dist(start,pa)
        for b in viewpoint_ids:
            dist_matrix[a][b]=0.0 if a==b else _path_dist(pa,positions[b])
    return dist_matrix, start_dists

def build_ordered_mission(scenario, strategy="input", start=(0.0,0.0,1.0),
                          use_astar_distances=False, grid_resolution=0.5,
                          inflation_m=0.5, bounds_margin_m=2.0):
    viewpoints=scenario["viewpoint_poses"]; original_ids=list(viewpoints.keys())
    dist_matrix=None; start_dists=None
    if use_astar_distances and strategy in ("nn","nn_2opt"):
        try:
            dist_matrix,start_dists=compute_astar_distance_matrix(
                scenario,original_ids,viewpoints,start,
                grid_resolution=grid_resolution,inflation_m=inflation_m,bounds_margin_m=bounds_margin_m)
        except Exception: dist_matrix=None; start_dists=None
    before=path_length(start,original_ids,viewpoints)
    if strategy=="input":
        ordered=original_ids[:]
    elif strategy=="nn":
        ordered=nearest_neighbour_order(start,original_ids,viewpoints,dist_matrix=dist_matrix,start_dists=start_dists)
    elif strategy=="nn_2opt":
        ordered=nearest_neighbour_order(start,original_ids,viewpoints,dist_matrix=dist_matrix,start_dists=start_dists)
        ordered=two_opt_open_path(start,ordered,viewpoints,dist_matrix=dist_matrix,start_dists=start_dists)
    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")
    after=path_length(start,ordered,viewpoints)
    return OrderedMission(ordered_ids=ordered,original_ids=original_ids,
                          estimated_length_before_m=before,estimated_length_after_m=after)
