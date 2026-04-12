from __future__ import annotations
import heapq, math
from typing import Dict, List, Optional, Tuple
from planners.grid_map import GridCell, GridCell3, OccupancyGrid2D, OccupancyGrid3D

def heuristic(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def reconstruct_path(came_from, current):
    path=[current]
    while current in came_from: current=came_from[current]; path.append(current)
    path.reverse(); return path

def nearest_free_cell(grid, start, max_radius=20):
    if grid.is_free(start): return start
    si,sj=start
    for r in range(1,max_radius+1):
        for di in range(-r,r+1):
            for dj in range(-r,r+1):
                if max(abs(di),abs(dj))!=r: continue
                c=(si+di,sj+dj)
                if grid.is_free(c): return c
    return None

def astar_search(grid, start, goal):
    start=nearest_free_cell(grid,start); goal=nearest_free_cell(grid,goal)
    if start is None or goal is None: return None
    open_heap=[]; heapq.heappush(open_heap,(heuristic(start,goal),0,start))
    came_from={}; g_score={start:0.0}; closed=set(); tie=1
    while open_heap:
        _,_,current=heapq.heappop(open_heap)
        if current in closed: continue
        closed.add(current)
        if current==goal: return reconstruct_path(came_from,current)
        for nxt,step_cost in grid.neighbors8(current):
            tentative=g_score[current]+step_cost
            if tentative<g_score.get(nxt,float("inf")):
                came_from[nxt]=current; g_score[nxt]=tentative
                heapq.heappush(open_heap,(tentative+heuristic(nxt,goal),tie,nxt)); tie+=1
    return None

# --- 3D A* (26-connectivity, Euclidean heuristic) ---

def heuristic3d(a, b):
    dx=a[0]-b[0]; dy=a[1]-b[1]; dz=a[2]-b[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def reconstruct_path3d(came_from, current):
    path=[current]
    while current in came_from: current=came_from[current]; path.append(current)
    path.reverse(); return path

def nearest_free_cell_3d(grid, start, max_radius=20):
    if grid.is_free(start): return start
    si,sj,sk=start
    for r in range(1,max_radius+1):
        for di in range(-r,r+1):
            for dj in range(-r,r+1):
                for dk in range(-r,r+1):
                    if max(abs(di),abs(dj),abs(dk))!=r: continue
                    c=(si+di,sj+dj,sk+dk)
                    if grid.is_free(c): return c
    return None

def astar_search_3d(grid, start, goal):
    """3D A* with 26-connectivity and Euclidean heuristic.
    Returns list of grid cells from start to goal, or None if no path found."""
    start=nearest_free_cell_3d(grid,start); goal=nearest_free_cell_3d(grid,goal)
    if start is None or goal is None: return None
    open_heap=[]; heapq.heappush(open_heap,(heuristic3d(start,goal),0,start))
    came_from={}; g_score={start:0.0}; closed=set(); tie=1
    while open_heap:
        _,_,current=heapq.heappop(open_heap)
        if current in closed: continue
        closed.add(current)
        if current==goal: return reconstruct_path3d(came_from,current)
        for nxt,step_cost in grid.neighbors26(current):
            tentative=g_score[current]+step_cost
            if tentative<g_score.get(nxt,float("inf")):
                came_from[nxt]=current; g_score[nxt]=tentative
                heapq.heappush(open_heap,(tentative+heuristic3d(nxt,goal),tie,nxt)); tie+=1
    return None
