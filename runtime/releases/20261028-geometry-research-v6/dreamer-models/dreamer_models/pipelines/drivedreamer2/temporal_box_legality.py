"""Deterministic sampled-world legality checks for the box-only pilot.

This is a finite-time no-worsening test, not a proof of continuous collision
freedom or a human label/content audit. All obstacles include non-visible boxes.
"""
import math
import numpy as np
import torch
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

class TemporalBoxLegality:
    def __init__(self,renderer,*,max_delta_speed=.1,max_delta_acceleration=.3,
                 max_delta_yaw_rate=.05,area_tolerance=1e-6):
        self.renderer=renderer
        self.limits=(max_delta_speed,max_delta_acceleration,max_delta_yaw_rate,area_tolerance)
        if not all(math.isfinite(v) and v>0 for v in self.limits):raise ValueError('Positive fixed physical thresholds required')
        bundle=renderer.bundle
        if not bundle.get('physical_records') or not bundle.get('drivable_polygons'):raise ValueError('Full world obstacles and raw drivable polygons required')
        polygons=[Polygon(x['exterior'],x['holes']) for x in bundle['drivable_polygons']]
        if not all(p.is_valid and not p.is_empty for p in polygons):raise ValueError('Invalid original drivable map')
        self.road=unary_union(polygons)
        self.baseline=[self._shapes(r) for r in bundle['physical_records']]
        self.last_report=None

    @staticmethod
    def _shapes(state):
        centers=state['centers_world'].detach().cpu().numpy();sizes=state['sizes_wlh'].detach().cpu().numpy();rotations=state['rotations_world'].detach().cpu().numpy()
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        shapes=[]
        for c,s,r in zip(centers,sizes,rotations):
            points=(signs*s[[1,0,2]]/2)@r.T+c
            shapes.append((MultiPoint(points[:,:2]).convex_hull,float(points[:,2].min()),float(points[:,2].max())))
        return shapes

    @staticmethod
    def _overlap(a,b):
        if min(a[2],b[2])<=max(a[1],b[1]):return 0.
        return a[0].intersection(b[0]).area

    def __call__(self,geometry):
        u=geometry['normalized_u'];self.renderer.build_geometry(u)
        controls=(u.detach()*self.renderer.scale).cpu()
        d=self.renderer.duration
        # Analytic maximum of the smoothstep first and second derivatives.
        speed=float(controls[:,:2].norm(dim=-1).max())*1.5/d
        acceleration=float(controls[:,:2].norm(dim=-1).max())*6/d**2
        yaw_rate=float(controls[:,2].abs().max())*1.5/d
        violations=[];max_road=0.;max_overlap=0.;checks=0
        for value,limit,name in zip((speed,acceleration,yaw_rate),self.limits[:3],('delta_speed','delta_acceleration','delta_yaw_rate')):
            if value>limit:violations.append(name)
        for record,base in zip(self.renderer.bundle['physical_records'],self.baseline):
            edited=self._shapes(self.renderer.world_state(record,u))
            for i,ident in enumerate(record['instance_ids']):
                if ident not in self.renderer.selected:continue
                checks+=1
                road_increase=edited[i][0].difference(self.road).area-base[i][0].difference(self.road).area
                max_road=max(max_road,road_increase)
                if road_increase>self.limits[3]:violations.append('new_road_exit')
                for j in range(len(edited)):
                    if j==i:continue
                    increase=self._overlap(edited[i],edited[j])-self._overlap(base[i],base[j])
                    max_overlap=max(max_overlap,increase)
                    if increase>self.limits[3]:violations.append('new_or_increased_box_overlap')
        if checks==0:violations.append('no_selected_observations')
        self.last_report=dict(passed=not violations,violations=sorted(set(violations)),selected_observations=checks,max_road_outside_area_increase=max_road,max_collision_area_increase=max_overlap,max_delta_speed=speed,max_delta_acceleration=acceleration,max_delta_yaw_rate=yaw_rate,scope='Full source world boxes at 48 camera sampling times; analytic edit kinematics; no-new road exit/overlap; continuous-time collision and generated-content validity not certified')
        return not violations
