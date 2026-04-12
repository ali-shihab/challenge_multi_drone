from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
GridCell = Tuple[int, int]
GridCell3 = Tuple[int, int, int]

@dataclass
class GridConfig:
    resolution_m: float = 0.5
    inflation_m: float = 0.6
    bounds_margin_m: float = 2.0

class OccupancyGrid2D:
    """2D occupancy grid built from scenario obstacles projected onto XY."""
    def __init__(self, *, min_x, max_x, min_y, max_y, resolution_m):
        self.min_x = float(min_x); self.max_x = float(max_x)
        self.min_y = float(min_y); self.max_y = float(max_y)
        self.resolution_m = float(resolution_m)
        self.width = max(1, int(math.ceil((self.max_x - self.min_x) / self.resolution_m)) + 1)
        self.height = max(1, int(math.ceil((self.max_y - self.min_y) / self.resolution_m)) + 1)
        self.occupied: set = set()

    def in_bounds(self, cell):
        i, j = cell
        return 0 <= i < self.width and 0 <= j < self.height

    def world_to_grid(self, x, y):
        i = min(max(int(round((x - self.min_x) / self.resolution_m)), 0), self.width - 1)
        j = min(max(int(round((y - self.min_y) / self.resolution_m)), 0), self.height - 1)
        return (i, j)

    def grid_to_world(self, cell):
        i, j = cell
        return (self.min_x + i * self.resolution_m, self.min_y + j * self.resolution_m)

    def is_occupied(self, cell): return cell in self.occupied
    def is_free(self, cell): return self.in_bounds(cell) and cell not in self.occupied

    def mark_rect_occupied(self, *, min_x, max_x, min_y, max_y):
        c0 = self.world_to_grid(min_x, min_y); c1 = self.world_to_grid(max_x, max_y)
        imin, imax = sorted((c0[0], c1[0])); jmin, jmax = sorted((c0[1], c1[1]))
        for i in range(imin, imax + 1):
            for j in range(jmin, jmax + 1):
                if self.in_bounds((i, j)): self.occupied.add((i, j))

    def neighbors8(self, cell):
        i, j = cell; out = []
        for di, dj, cost in [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                              (-1,-1,math.sqrt(2.0)),(-1,1,math.sqrt(2.0)),
                              (1,-1,math.sqrt(2.0)),(1,1,math.sqrt(2.0))]:
            nxt = (i+di, j+dj)
            if self.is_free(nxt): out.append((nxt, cost))
        return out

    def line_is_free_world(self, a, b):
        dist = math.hypot(b[0]-a[0], b[1]-a[1])
        n = max(2, int(math.ceil(dist / max(1e-6, self.resolution_m * 0.5))))
        for k in range(n + 1):
            t = k / n
            if not self.is_free(self.world_to_grid(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))):
                return False
        return True


class OccupancyGrid3D:
    """3D occupancy grid. Obstacles inflated by inflation_m in all three dimensions."""
    def __init__(self, *, min_x, max_x, min_y, max_y, min_z, max_z, resolution_m):
        self.min_x=float(min_x); self.max_x=float(max_x)
        self.min_y=float(min_y); self.max_y=float(max_y)
        self.min_z=float(min_z); self.max_z=float(max_z)
        self.resolution_m=float(resolution_m)
        self.nx=max(1,int(math.ceil((self.max_x-self.min_x)/self.resolution_m))+1)
        self.ny=max(1,int(math.ceil((self.max_y-self.min_y)/self.resolution_m))+1)
        self.nz=max(1,int(math.ceil((self.max_z-self.min_z)/self.resolution_m))+1)
        self.occupied: set = set()

    def in_bounds(self, cell):
        i,j,k=cell; return 0<=i<self.nx and 0<=j<self.ny and 0<=k<self.nz

    def world_to_grid(self, x, y, z):
        i=min(max(int(round((x-self.min_x)/self.resolution_m)),0),self.nx-1)
        j=min(max(int(round((y-self.min_y)/self.resolution_m)),0),self.ny-1)
        k=min(max(int(round((z-self.min_z)/self.resolution_m)),0),self.nz-1)
        return (i,j,k)

    def grid_to_world(self, cell):
        i,j,k=cell
        return (self.min_x+i*self.resolution_m, self.min_y+j*self.resolution_m, self.min_z+k*self.resolution_m)

    def is_free(self, cell): return self.in_bounds(cell) and cell not in self.occupied

    def mark_box_occupied(self, *, min_x, max_x, min_y, max_y, min_z, max_z):
        c0=self.world_to_grid(min_x,min_y,min_z); c1=self.world_to_grid(max_x,max_y,max_z)
        imin,imax=sorted((c0[0],c1[0])); jmin,jmax=sorted((c0[1],c1[1])); kmin,kmax=sorted((c0[2],c1[2]))
        for i in range(imin,imax+1):
            for j in range(jmin,jmax+1):
                for k in range(kmin,kmax+1):
                    if self.in_bounds((i,j,k)): self.occupied.add((i,j,k))

    _NEIGHBOR_DELTAS = [(di,dj,dk) for di in(-1,0,1) for dj in(-1,0,1) for dk in(-1,0,1)
                        if not(di==0 and dj==0 and dk==0)]

    def neighbors26(self, cell):
        i,j,k=cell; out=[]
        for di,dj,dk in self._NEIGHBOR_DELTAS:
            nxt=(i+di,j+dj,k+dk)
            if self.is_free(nxt):
                out.append((nxt, math.sqrt(di*di+dj*dj+dk*dk)))
        return out

    def line_is_free_world(self, a, b):
        dist=math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2)
        n=max(2,int(math.ceil(dist/max(1e-6,self.resolution_m*0.5))))
        for step in range(n+1):
            t=step/n
            x=a[0]+t*(b[0]-a[0]); y=a[1]+t*(b[1]-a[1]); z=a[2]+t*(b[2]-a[2])
            if not self.is_free(self.world_to_grid(x,y,z)): return False
        return True


def _obstacle_vertical_span(obs):
    zc=float(obs["z"]); h=float(obs["h"])
    return (zc-0.5*h, zc+0.5*h)

def _viewpoint_positions_xy(scenario):
    return [(float(vp["x"]),float(vp["y"])) for vp in scenario["viewpoint_poses"].values()]

def build_occupancy_grid_from_scenario(scenario, *, start_xy, flight_z, config):
    """Build a 2D occupancy grid projecting obstacles at the given flight altitude."""
    points_xy=[start_xy]+_viewpoint_positions_xy(scenario)
    min_x=min(p[0] for p in points_xy); max_x=max(p[0] for p in points_xy)
    min_y=min(p[1] for p in points_xy); max_y=max(p[1] for p in points_xy)
    for obs in scenario.get("obstacles",{}).values():
        ox,oy,w,d=float(obs["x"]),float(obs["y"]),float(obs["w"]),float(obs["d"])
        min_x=min(min_x,ox-0.5*w); max_x=max(max_x,ox+0.5*w)
        min_y=min(min_y,oy-0.5*d); max_y=max(max_y,oy+0.5*d)
    min_x-=config.bounds_margin_m; max_x+=config.bounds_margin_m
    min_y-=config.bounds_margin_m; max_y+=config.bounds_margin_m
    grid=OccupancyGrid2D(min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y,resolution_m=config.resolution_m)
    z_band_half=max(0.5,config.inflation_m)
    for obs in scenario.get("obstacles",{}).values():
        z0,z1=_obstacle_vertical_span(obs)
        if flight_z<(z0-z_band_half) or flight_z>(z1+z_band_half): continue
        ox,oy,w,d=float(obs["x"]),float(obs["y"]),float(obs["w"]),float(obs["d"])
        hw=0.5*w+config.inflation_m; hd=0.5*d+config.inflation_m
        grid.mark_rect_occupied(min_x=ox-hw,max_x=ox+hw,min_y=oy-hd,max_y=oy+hd)
    return grid

def build_occupancy_grid_3d_from_scenario(scenario, *, start_xyz, config):
    """Build a 3D occupancy grid covering all viewpoints and obstacles, inflated in 3D."""
    viewpoints=scenario.get("viewpoint_poses",{})
    points=[start_xyz]+[(float(vp["x"]),float(vp["y"]),float(vp["z"])) for vp in viewpoints.values()]
    min_x=min(p[0] for p in points); max_x=max(p[0] for p in points)
    min_y=min(p[1] for p in points); max_y=max(p[1] for p in points)
    min_z=min(p[2] for p in points); max_z=max(p[2] for p in points)
    min_z=min(min_z,0.0)
    for obs in scenario.get("obstacles",{}).values():
        ox,oy,oz=float(obs["x"]),float(obs["y"]),float(obs["z"])
        w,d,h=float(obs["w"]),float(obs["d"]),float(obs["h"])
        min_x=min(min_x,ox-0.5*w); max_x=max(max_x,ox+0.5*w)
        min_y=min(min_y,oy-0.5*d); max_y=max(max_y,oy+0.5*d)
        min_z=min(min_z,oz-0.5*h); max_z=max(max_z,oz+0.5*h)
    m=config.bounds_margin_m
    min_x-=m; max_x+=m; min_y-=m; max_y+=m
    min_z=max(0.0,min_z-m); max_z+=m
    grid=OccupancyGrid3D(min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y,
                         min_z=min_z,max_z=max_z,resolution_m=config.resolution_m)
    inf=config.inflation_m
    for obs in scenario.get("obstacles",{}).values():
        ox,oy,oz=float(obs["x"]),float(obs["y"]),float(obs["z"])
        w,d,h=float(obs["w"]),float(obs["d"]),float(obs["h"])
        grid.mark_box_occupied(
            min_x=ox-0.5*w-inf, max_x=ox+0.5*w+inf,
            min_y=oy-0.5*d-inf, max_y=oy+0.5*d+inf,
            min_z=oz-0.5*h-inf, max_z=oz+0.5*h+inf,
        )
    return grid
