"""Differentiable native LiDAR labels from the same edited world tracks."""
import torch
from .physical_box_conditions import pose_matrix

class TemporalDetectionLabels:
    def __init__(self,renderer,infos):
        self.renderer=renderer;self.infos=tuple(infos)
        if len(infos)!=len(renderer.bundle['keyframes']):raise ValueError('One native info for each actual keyframe required')
        self.transforms=[];self.base=[]
        for record,info in zip(renderer.bundle['keyframes'],infos):
            if record['sample_token']!=info['token'] or info['scene_token']!=renderer.bundle['scene_id']:raise ValueError('Scene/keyframe mismatch')
            e=pose_matrix(dict(translation=info['ego2global_translation'],rotation=info['ego2global_rotation']))
            l=pose_matrix(dict(translation=info['lidar2ego_translation'],rotation=info['lidar2ego_rotation']))
            transform=torch.linalg.inv(e@l);self.transforms.append(transform)
            state=renderer.world_state(record,renderer.initial_u());computed=self._convert(state,transform)
            original=torch.cat((torch.as_tensor(info['gt_boxes'],dtype=torch.double),torch.as_tensor(info['gt_velocity'],dtype=torch.double)),dim=-1)
            if computed.shape!=original.shape:raise ValueError('Original annotation count differs')
            if not torch.allclose(computed[:,:6],original[:,:6],atol=2e-4,rtol=1e-6):raise ValueError('Original world/LiDAR boxes differ')
            angle=torch.atan2(torch.sin(computed[:,6]-original[:,6]),torch.cos(computed[:,6]-original[:,6]))
            if angle.abs().max()>2e-4:raise ValueError('Original world/LiDAR yaw convention differs')
            if not torch.allclose(computed[:,7:],original[:,7:],atol=2e-4,rtol=1e-6,equal_nan=True):raise ValueError('Original world/LiDAR velocities differ')
            self.base.append((computed,original))

    @staticmethod
    def _convert(state,transform):
        centers=state['centers_world'];transform=transform.to(centers);rotation=transform[:3,:3]
        xyz=centers@rotation.T+transform[:3,3]
        orientation=rotation[None]@state['rotations_world']
        yaw=torch.atan2(-orientation[:,0,1],orientation[:,0,0])
        velocity=state['velocities_world'];velocity=torch.cat((velocity[:,:2],torch.zeros_like(velocity[:,:1])),-1)@rotation.T
        return torch.cat((xyz,state['sizes_wlh'][:,[1,0,2]],yaw[:,None],velocity[:,:2]),-1)

    def __call__(self,geometry,index,*,bottom_center=False):
        expected=self.renderer.build_geometry(geometry['normalized_u'])
        # Ignore externally substituted labels; derive again from the source u.
        value=self._convert(expected['keyframe_states'][index],self.transforms[index])
        base,original=self.base[index];base=base.to(value);original=original.to(value)
        delta=value-base
        yaw=torch.atan2(torch.sin(delta[:,6]),torch.cos(delta[:,6]))
        # Unknown native velocity labels are fixed zero after dataset sanitization.
        velocity=torch.where(torch.isfinite(original[:,7:]),original[:,7:]+torch.nan_to_num(delta[:,7:]),torch.zeros_like(delta[:,7:]))
        result=torch.cat((original[:,:6]+delta[:,:6],(original[:,6]+yaw)[:,None],velocity),-1)
        if bottom_center:
            result=torch.cat((result[:,:2],(result[:,2]-result[:,5]/2)[:,None],result[:,3:]),-1)
        return result
