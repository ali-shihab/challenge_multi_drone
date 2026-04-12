"""RRT* (Rapidly-exploring Random Tree Star) 3D path planner."""
from __future__ import annotations
import math, random
from typing import Any, Dict, List, Optional, Tuple

Point3 = Tuple[float, float, float]

def _dist3(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

class _Node:
    __slots__=("pos","parent","cost")
    def __init__(self,pos,parent=None,cost=0.0): self.pos=pos;self.parent=parent;self.cost=cost

class RRTStarPlanner:
    """RRT* planner for 3D obstacle-aware path planning.

    Parameters
    ----------
    obstacles : list of dicts with keys x,y,z,w,d,h (center + full dimensions)
    bounds    : ((min_x,max_x),(min_y,max_y),(min_z,max_z))
    step_size : max branch length (m)
    max_iter  : max iterations
    goal_bias : fraction of samples directed at goal
    rewire_radius : neighbourhood radius for rewiring (m)
    inflation : obstacle inflation margin (m)
    """
    def __init__(self, obstacles, bounds, step_size=0.5, max_iter=2000,
                 goal_bias=0.1, rewire_radius=1.5, inflation=0.5):
        self.bounds=bounds; self.step_size=step_size; self.max_iter=max_iter
        self.goal_bias=goal_bias; self.rewire_radius=rewire_radius; self.inflation=inflation
        self._aabbs=[]
        for obs in obstacles:
            ox,oy,oz=float(obs["x"]),float(obs["y"]),float(obs["z"])
            hw=0.5*float(obs["w"])+inflation; hd=0.5*float(obs["d"])+inflation; hh=0.5*float(obs["h"])+inflation
            self._aabbs.append((ox-hw,ox+hw,oy-hd,oy+hd,oz-hh,oz+hh))

    def _in_obs(self,p):
        x,y,z=p
        for x0,x1,y0,y1,z0,z1 in self._aabbs:
            if x0<=x<=x1 and y0<=y<=y1 and z0<=z<=z1: return True
        return False

    def _clear(self,a,b,n=0):
        d=_dist3(a,b)
        if d<1e-9: return not self._in_obs(a)
        steps=max(2,n or int(math.ceil(d/(self.step_size*0.5))))
        for k in range(steps+1):
            t=k/steps
            if self._in_obs((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),a[2]+t*(b[2]-a[2]))): return False
        return True

    def _sample(self,goal):
        if random.random()<self.goal_bias: return goal
        (x0,x1),(y0,y1),(z0,z1)=self.bounds
        return (random.uniform(x0,x1),random.uniform(y0,y1),random.uniform(z0,z1))

    def _steer(self,fr,to):
        d=_dist3(fr,to)
        if d<=self.step_size: return to
        t=self.step_size/d
        return (fr[0]+t*(to[0]-fr[0]),fr[1]+t*(to[1]-fr[1]),fr[2]+t*(to[2]-fr[2]))

    def _nearest(self,nodes,pos): return min(nodes,key=lambda n:_dist3(n.pos,pos))
    def _near(self,nodes,pos): r=self.rewire_radius; return [n for n in nodes if _dist3(n.pos,pos)<=r]

    def plan(self, start, goal):
        """Run RRT* and return smoothed (x,y,z) waypoints, or [] on failure."""
        if self._clear(start,goal): return [start,goal]
        root=_Node(start); nodes=[root]; goal_node=None
        for _ in range(self.max_iter):
            rand=self._sample(goal); nearest=self._nearest(nodes,rand); new_pos=self._steer(nearest.pos,rand)
            if self._in_obs(new_pos) or not self._clear(nearest.pos,new_pos): continue
            near_nodes=self._near(nodes,new_pos)
            best_p=nearest; best_c=nearest.cost+_dist3(nearest.pos,new_pos)
            for n in near_nodes:
                c=n.cost+_dist3(n.pos,new_pos)
                if c<best_c and self._clear(n.pos,new_pos): best_p=n; best_c=c
            new_node=_Node(new_pos,parent=best_p,cost=best_c); nodes.append(new_node)
            for n in near_nodes:
                c=new_node.cost+_dist3(new_node.pos,n.pos)
                if c<n.cost and self._clear(new_node.pos,n.pos): n.parent=new_node; n.cost=c
            if _dist3(new_pos,goal)<=self.step_size and self._clear(new_pos,goal):
                gc=new_node.cost+_dist3(new_pos,goal)
                if goal_node is None or gc<goal_node.cost:
                    goal_node=_Node(goal,parent=new_node,cost=gc)
        if goal_node is None: return []
        path=[]; cur=goal_node
        while cur is not None: path.append(cur.pos); cur=cur.parent
        path.reverse(); return self._smooth(path)

    def _smooth(self,path):
        if len(path)<=2: return path
        s=[path[0]]; i=0
        while i<len(path)-1:
            j=len(path)-1
            while j>i+1:
                if self._clear(path[i],path[j]): break
                j-=1
            s.append(path[j]); i=j
        return s
