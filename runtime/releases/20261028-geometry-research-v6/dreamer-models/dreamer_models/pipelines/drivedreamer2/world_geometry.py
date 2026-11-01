"""World map nodes and identity-aligned temporal boxes: the ONLY editable leaves.

The renderer and task label adapters consume the same returned state. This module
checks identity, metric budgets, fixed z/size, temporal derivatives and graph
connectivity. A native map/topology/road/collision validator is mandatory: keeping
connectivity alone does not prove geometric topology or expert-route validity.
"""
from dataclasses import dataclass
import math
import torch


def _fixed(value, label):
    value = torch.as_tensor(value)
    if value.requires_grad or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(label + ' must be finite fixed floating data')
    return value.detach().clone()


@dataclass(frozen=True)
class GeometryBudgets:
    map_xy_m: float
    box_xy_m: float
    box_yaw_rad: float
    max_speed_mps: float
    max_acceleration_mps2: float
    max_yaw_rate_rps: float

    def __post_init__(self):
        for key,value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(key + ' must be finite and nonnegative')


def _connected(value, u):
    if value is u:return True
    pending=[value.grad_fn];seen=set()
    while pending:
        node=pending.pop()
        if node is None or node in seen:continue
        seen.add(node)
        if node is u.grad_fn and u.grad_fn is not None:return True
        if getattr(node,'variable',None) is u:return True
        pending.extend(n for n,_ in node.next_functions)
    return False


class WorldGeometry:
    def __init__(self, *, scene_id, map_nodes_world, map_elements, centers_world,
                 yaw_world, sizes_wlh, timestamps_s, instance_ids, class_ids,
                 temporal_basis, budgets, native_legality, anchor_first_frame=True,
                 map_edit_mask=None):
        if not scene_id or not callable(native_legality):
            raise ValueError('Explicit scene and native physical legality callback required')
        self.scene_id, self.native_legality = scene_id, native_legality
        self.budgets, self.anchor_first_frame = budgets, anchor_first_frame
        self.maps = _fixed(map_nodes_world, 'map nodes')
        self.centers = _fixed(centers_world, 'centers')
        self.yaw = _fixed(yaw_world, 'yaw')
        self.sizes = _fixed(sizes_wlh, 'sizes')
        self.times = _fixed(timestamps_s, 'timestamps')
        self.basis = _fixed(temporal_basis, 'temporal basis')
        if self.maps.ndim != 2 or self.maps.shape[1] != 3:
            raise ValueError('Map nodes must be [M,3] world coordinates, never rasters')
        if self.centers.ndim != 3 or self.centers.shape[2] != 3:
            raise ValueError('Centers must be [T,N,3]')
        t,n,_ = self.centers.shape
        if t < 2 or n < 1 or self.yaw.shape != (t,n) or self.sizes.shape != (n,3) or not (self.sizes>0).all():
            raise ValueError('Require real multi-frame tracks with fixed positive dimensions')
        if self.times.shape != (t,) or not (self.times[1:]>self.times[:-1]).all():
            raise ValueError('Timestamps must be strictly increasing real seconds')
        if self.basis.ndim != 2 or self.basis.shape[0] != t or self.basis.shape[1] < 1 or (self.basis<0).any() or not torch.allclose(self.basis.sum(1),torch.ones_like(self.times)):
            raise ValueError('Fixed temporal basis must be a convex interpolation matrix')
        self.instance_ids, self.class_ids = tuple(instance_ids),tuple(class_ids)
        if len(self.instance_ids)!=n or len(set(self.instance_ids))!=n or len(self.class_ids)!=n:
            raise ValueError('Require fixed unique instance IDs and classes across all frames')
        # An explicit dense contiguous track is required. Birth/death/occlusion
        # tracks must be split by the data adapter; do not fill missing frames.
        self.elements = tuple((str(e['id']),str(e['semantic']),tuple(e['node_indices'])) for e in map_elements)
        if len({e[0] for e in self.elements})!=len(self.elements):
            raise ValueError('Map element IDs must be unique')
        for _,_,indices in self.elements:
            if len(indices)<2 or any(not isinstance(i,int) or i<0 or i>=len(self.maps) for i in indices):
                raise ValueError('Invalid map connectivity')
        if map_edit_mask is None:
            map_edit_mask = torch.ones((len(self.maps),2),dtype=torch.bool)
        self.map_mask = torch.as_tensor(map_edit_mask).clone()
        if self.map_mask.dtype != torch.bool or self.map_mask.shape != (len(self.maps),2):
            raise ValueError('Map edit mask must be fixed boolean [M,2]')
        self.map_mask &= budgets.map_xy_m > 0
        box_scale = self.centers.new_tensor([budgets.box_xy_m,budgets.box_xy_m,budgets.box_yaw_rad])
        self.box_scale = box_scale.expand(n,self.basis.shape[1],3).clone()
        # First-frame basis coefficients are not editable. For other basis
        # choices, reject rather than subtracting an offset that doubles budgets.
        if anchor_first_frame:
            self.box_scale[:,self.basis[0]>0,:] = 0
        self.box_mask = self.box_scale > 0
        self.map_count,self.box_count = int(self.map_mask.sum()),int(self.box_mask.sum())
        self.parameter_count = self.map_count+self.box_count
        if not self.parameter_count:
            raise ValueError('No editable world geometry coordinates remain')

    def initial_u(self, *, device=None, dtype=None):
        return torch.zeros(self.parameter_count,device=device,dtype=dtype or self.centers.dtype)

    def build_geometry(self,u):
        if not isinstance(u,torch.Tensor) or not u.is_floating_point() or u.shape!=(self.parameter_count,) or not torch.isfinite(u.detach()).all():
            raise ValueError('u must be the compact finite world-geometry parameter vector')
        if bool((u.detach().abs()>1).any()):
            raise ValueError('Normalized world coordinate budget exceeded')
        md = u.new_zeros(self.map_mask.shape).masked_scatter(self.map_mask.to(u.device),u[:self.map_count])*self.budgets.map_xy_m
        map_nodes = self.maps.to(u)+torch.cat((md,u.new_zeros((len(md),1))),-1)
        controls = u.new_zeros(self.box_mask.shape).masked_scatter(self.box_mask.to(u.device),u[self.map_count:])*self.box_scale.to(u)
        delta = torch.einsum('tk,nkc->tnc',self.basis.to(u),controls)
        centers = self.centers.to(u)+torch.cat((delta[...,:2],torch.zeros_like(delta[...,:1])),-1)
        yaw = self.yaw.to(u)+delta[...,2]
        dt = (self.times[1:]-self.times[:-1]).to(u)
        segment_velocity = (centers[1:]-centers[:-1])/dt[:,None,None]
        # Timestamp-aware central derivatives, with one-sided endpoints.
        if len(self.times)>2:
            middle = (centers[2:]-centers[:-2])/(self.times[2:]-self.times[:-2]).to(u)[:,None,None]
            velocities = torch.cat((segment_velocity[:1],middle,segment_velocity[-1:]),0)
            acceleration = 2*(segment_velocity[1:]-segment_velocity[:-1])/(dt[1:]+dt[:-1])[:,None,None]
        else:
            velocities = segment_velocity.expand(2,-1,-1)
            acceleration = centers[:0]
        yaw_diff = yaw[1:]-yaw[:-1]
        yaw_rate = torch.atan2(yaw_diff.sin(),yaw_diff.cos())/dt[:,None]
        return dict(scene_id=self.scene_id, normalized_u=u, map_nodes_world=map_nodes,
                    map_elements=self.elements, centers_world=centers, yaw_world=yaw,
                    sizes_wlh=self.sizes.to(u), velocities_world=velocities,
                    segment_velocities_world=segment_velocity, acceleration_world=acceleration,
                    yaw_rate=yaw_rate, instance_ids=self.instance_ids, class_ids=self.class_ids,
                    timestamps_s=self.times.to(u), single_keyframe_only=False)

    def geometry_is_valid(self,g):
        try:
            rebuilt = self.build_geometry(g['normalized_u'])
            if set(g)!=set(rebuilt):return False
            for key,expected in rebuilt.items():
                actual=g[key]
                if torch.is_tensor(expected):
                    if not torch.is_tensor(actual) or actual.shape!=expected.shape or not torch.equal(actual.detach(),expected.detach()):return False
                    if expected.requires_grad and (not actual.requires_grad or not _connected(actual,g["normalized_u"])):return False
                elif actual != expected:return False
            b=self.budgets
            if (g['segment_velocities_world'].norm(dim=-1)>b.max_speed_mps).any():return False
            if (g['acceleration_world'].norm(dim=-1)>b.max_acceleration_mps2).any():return False
            if (g['yaw_rate'].abs()>b.max_yaw_rate_rps).any():return False
            for _,_,indices in self.elements:
                points=g['map_nodes_world'][list(indices)]
                if ((points[1:]-points[:-1]).norm(dim=-1)<1e-6).any():return False
            result=self.native_legality(g)
            if not isinstance(result,bool):raise TypeError('Native legality must return bool')
            return result
        except (KeyError,ValueError,IndexError):
            return False

    def export(self,u):
        g=self.build_geometry(u.detach())
        if not self.geometry_is_valid(g):raise ValueError('Refusing invalid geometry export')
        return {k:v.detach().cpu().tolist() if torch.is_tensor(v) else v for k,v in g.items()}
