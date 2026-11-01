"""Compose raw divider-node and temporal-box geometry in one native DD2 input.

Only world coordinates are controls. Each component retains its own physical
constraints and native-forward/soft-surrogate derivative. Drivable surfaces
remain fixed; this does not certify expert-route validity for edited objects.
"""
import torch
from .dd2_fixed_input_adapter import RenderedDD2Conditions

class JointWorldConditions:
    def __init__(self,map_renderer,box_renderer):
        if map_renderer.bundle is not box_renderer.bundle:
            raise ValueError('Map and boxes must be built from the same scene bundle')
        self.map_renderer=map_renderer;self.box_renderer=box_renderer
        self.bundle=box_renderer.bundle;self.box_shape=box_renderer.initial_u().shape
        self.map_count=map_renderer.initial_u().numel();self.last_report={}

    def initial_u(self):
        return torch.cat((self.map_renderer.initial_u().flatten(),self.box_renderer.initial_u().flatten()))

    def split_u(self,u):
        if not torch.is_tensor(u) or not u.is_floating_point() or u.shape!=self.initial_u().shape or not torch.isfinite(u.detach()).all() or (u.detach().abs()>1).any():
            raise ValueError('Expected bounded map XY followed by per-instance box dx/dy/yaw')
        return u[:self.map_count],u[self.map_count:].reshape(self.box_shape)

    def build_geometry(self,u):
        map_u,box_u=self.split_u(u)
        return dict(scene_id=self.bundle['scene_id'],normalized_u=u,
                    map=self.map_renderer.build_geometry(map_u),box=self.box_renderer.build_geometry(box_u))

    def geometry_is_valid(self,g):
        try:
            map_u,box_u=self.split_u(g['normalized_u'])
            if set(g)!={'scene_id','normalized_u','map','box'} or g['scene_id']!=self.bundle['scene_id']:
                return False
            if not torch.equal(g['map']['normalized_u'],map_u) or not torch.equal(g['box']['normalized_u'],box_u):
                return False
            map_valid=self.map_renderer.geometry_is_valid(g['map'])
            box_valid=self.box_renderer.geometry_is_valid(g['box'])
            self.last_report=dict(valid=map_valid and box_valid,map=self.map_renderer.last_report,
                                  box=self.box_renderer.physical_legality.last_report,
                                  scope='Combined existing map and box constraints; drivable-area surfaces fixed. Expert trajectory validity requires a separate object interaction audit.')
            return map_valid and box_valid
        except (KeyError,ValueError,TypeError):
            return False

    def __call__(self,g):
        if not self.geometry_is_valid(g):raise ValueError('Invalid combined world geometry: '+str(self.last_report))
        maps=self.map_renderer(g['map']);boxes=self.box_renderer(g['box'])
        return RenderedDD2Conditions(self.bundle['scene_id'],tuple(self.bundle['camera_order']),
            dict(grounding_downsampler_input=maps.input_dict['grounding_downsampler_input'],
                 box_downsampler_input=boxes.input_dict['box_downsampler_input']))

    def segmentation(self,g,**kwargs):
        if not self.geometry_is_valid(g):raise ValueError('Invalid combined world geometry')
        return self.map_renderer.segmentation(g['map'],**kwargs)
