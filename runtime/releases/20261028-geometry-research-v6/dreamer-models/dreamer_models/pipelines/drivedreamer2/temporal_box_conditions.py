"""Identity-aligned world-box edits on actual asynchronous DD2 camera frames.

Map rasters are fixed verified source conditions in this box-only pilot. The
single compact world dx/dy/yaw vector per selected instance drives every frame
and current keyframe labels. Forward uses exact native raster pixels; backward
uses an explicitly lower-resolution soft raster STE. No pixel/feature variables.
"""
import torch
import numpy as np
from torch.nn import functional as F
from .dd2_fixed_input_adapter import RenderedDD2Conditions
from .physical_box_conditions import (NativeBoxRasterizer, world_to_camera_boxes,
    native_camera_corners,crop_native_corners,project_corners,soft_box_raster,
    full_camera_corners,SOURCE_SIZE)


class TemporalBoxConditions:
    def __init__(self,bundle,*,repository_root=None,xy_budget_m=.05,yaw_budget_rad=.01,
                 output_size=(256,448),soft_size=(64,112),antialias=True,physical_legality=None):
        self.bundle=bundle;self.native=NativeBoxRasterizer(repository_root)
        self.selected=tuple(bundle['selected_instance_ids']);self.output_size=tuple(output_size);self.soft_size=tuple(soft_size)
        self.scale=torch.tensor([xy_budget_m,xy_budget_m,yaw_budget_rad],dtype=torch.double)
        if not self.selected or (self.scale<=0).any() or not torch.isfinite(self.scale).all():raise ValueError('Positive physical budgets and selected IDs required')
        self.duration=(bundle['end_us']-bundle['start_us'])/1e6
        if self.duration<=0:raise ValueError('Real increasing temporal support required')
        self.antialias=antialias;self.physical_legality=physical_legality
        if len(bundle['records'])!=48 or len(bundle['camera_order'])!=6:raise ValueError('Eight genuine frames and six views required')
        self.baseline=[self._project(r,torch.zeros((len(self.selected),3),dtype=torch.double)) for r in bundle['records']]

    def initial_u(self):return torch.zeros((len(self.selected),3),dtype=torch.double)

    def _weights(self,timestamp_us):
        t=(timestamp_us-self.bundle['start_us'])/1e6
        # Hold the edit constant outside the clip: both time derivatives vanish.
        # At the endpoints use the interior one-sided acceleration (C1, not C2).
        if t<0:return 0.,0.,0.
        if t>self.duration:return 1.,0.,0.
        s=t/self.duration
        return s*s*(3-2*s),6*s*(1-s)/self.duration,6*(1-2*s)/self.duration**2

    def _delta(self,ids,timestamp,u):
        indices={value:i for i,value in enumerate(self.selected)}
        rows=[u[indices[i]]*self.scale.to(u) if i in indices else u.new_zeros(3) for i in ids]
        controls=torch.stack(rows) if rows else u.new_zeros((0,3))
        w,dw,ddw=self._weights(timestamp)
        return controls*w,controls*dw,controls*ddw

    def world_state(self,record,u):
        delta,velocity_delta,acceleration_delta=self._delta(record['instance_ids'],record.get('geometry_timestamp_us',record.get('timestamp_us')),u)
        centers=record['centers_world'].to(u)+torch.cat((delta[:,:2],torch.zeros_like(delta[:,:1])),-1)
        angle=delta[:,2];c,s=angle.cos(),angle.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
        rz=torch.stack((c,-s,z,s,c,z,z,z,one),-1).reshape(-1,3,3)
        state=dict(centers_world=centers,sizes_wlh=record['sizes_wlh'].to(u),rotations_world=rz@record['rotations_world'].to(u))
        if 'velocities_world' in record:
            state['velocities_world']=record['velocities_world'].to(u)+torch.cat((velocity_delta[:,:2],torch.zeros_like(velocity_delta[:,:1])),-1)
        return state

    def build_geometry(self,u):
        if not isinstance(u,torch.Tensor) or not u.is_floating_point() or u.shape!=(len(self.selected),3) or not torch.isfinite(u.detach()).all() or (u.detach().abs()>1).any():raise ValueError('Only bounded compact world dx/dy/yaw controls accepted')
        return dict(scene_id=self.bundle['scene_id'],normalized_u=u,selected_instance_ids=self.selected,
                    keyframe_states=[self.world_state(r,u) for r in self.bundle['keyframes']])

    def _project(self,r,u):
        state=self.world_state(r,u)
        computed,centers,orient=world_to_camera_boxes(**state,camera_to_world=r['camera_to_world'])
        original,_,_=world_to_camera_boxes(r['centers_world'].to(u),r['sizes_wlh'].to(u),r['rotations_world'].to(u),r['camera_to_world'])
        # Source float32 camera coordinates are an audited representation of the
        # same world boxes. Preserve their exact zero-edit values while applying
        # only the world-derived position/orientation change.
        boxes=r['native_camera_boxes'].to(u)+(computed-original)
        array=boxes.detach().cpu().numpy().astype(np.float32)
        order=sorted(range(len(array)),key=lambda i:float(array[i,0]**2+array[i,1]**2+array[i,2]**2),reverse=True)
        hard3d=self.native.boxes.boxes3d_to_corners3d(array[order],rot_axis=1)
        if len(order) and not (hard3d[...,2].mean(-1)>0).all():raise ValueError('Native camera boxes behind camera')
        hard=self.native.boxes.corners3d_to_corners2d(self.native.boxes.crop_corners3d(hard3d.copy(),z_b=.5,threshold=0),r['intrinsic'].numpy())
        if not np.isfinite(hard).all() or (hard.size and np.abs(hard).max()>=2**30):raise ValueError('Unsafe native projection')
        soft=project_corners(crop_native_corners(native_camera_corners(boxes[order])),r['intrinsic'])
        full=full_camera_corners(centers,state['sizes_wlh'],orient)
        return dict(order=order,hard=hard,soft=soft,hard3d=hard3d,full=full,class_ids=tuple(r['class_ids'][i] for i in order))

    def _valid_views(self,g):
        self.build_geometry(g['normalized_u'])
        if g['scene_id']!=self.bundle['scene_id'] or tuple(g['selected_instance_ids'])!=self.selected:raise ValueError('Scene/instance identity changed')
        views=[]
        for r,base in zip(self.bundle['records'],self.baseline):
            view=self._project(r,g['normalized_u'])
            if view['order']!=base['order']:raise ValueError('Discrete occlusion ordering changed')
            selected=[i for i,ident in enumerate(r['instance_ids']) if ident in self.selected]
            if selected:
                raw=view['full'][selected].detach().cpu().numpy().astype(np.float32)
                if not self.native.boxes.corners3d_in_image(raw,r['intrinsic'].numpy(),SOURCE_SIZE).all():raise ValueError('Selected object visibility changed')
                for i in selected:
                    pos=view['order'].index(i)
                    if view['hard3d'][pos,:,2].min()<=.55:raise ValueError('Selected object approaches near-plane discontinuity')
            views.append(view)
        return views

    def geometry_is_valid(self,g):
        try:
            self._valid_views(g)
            expected=self.build_geometry(g['normalized_u'])
            if len(g['keyframe_states'])!=len(expected['keyframe_states']):return False
            for actual,reference in zip(g['keyframe_states'],expected['keyframe_states']):
                if set(actual)!=set(reference):return False
                for key in actual:
                    if not torch.allclose(actual[key].detach(),reference[key].detach(),atol=0,rtol=0,equal_nan=True):return False
            if self.physical_legality is None:return False
            result=self.physical_legality(g)
            if not isinstance(result,bool):raise TypeError('Physical legality must return bool')
            return result
        except (ValueError,KeyError):return False

    def __call__(self,g):
        views=self._valid_views(g);u=g['normalized_u'];images=[[None]*6 for _ in range(8)]
        for r,view in zip(self.bundle['records'],views):
            hard=self.native.resized(view['hard'],view['class_ids'],self.output_size,antialias=self.antialias).to(u)
            if torch.is_grad_enabled() and u.requires_grad and any(i in self.selected for i in r['instance_ids']):
                soft=soft_box_raster(view['soft'],view['class_ids'],output_size=self.soft_size,skeleton=self.native.skeleton,sigma_pixels=8.,hard_corners=view['hard'].astype(np.int32))
                soft=F.interpolate(soft[None],size=self.output_size,mode='bilinear',align_corners=False)[0]
                hard=hard+(soft-soft.detach())
            images[r['frame']][r['view']]=hard
        maps=self.bundle['source_conditions']['input_dict']['grounding_downsampler_input']
        boxes=torch.stack([torch.cat(frame,-1) for frame in images]).to(dtype=maps.dtype)
        return RenderedDD2Conditions(self.bundle['scene_id'],tuple(self.bundle['camera_order']),dict(grounding_downsampler_input=maps.to(u.device),box_downsampler_input=boxes))
