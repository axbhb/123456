"""Bounded edits of one real divider node, shared across camera conditions/labels.

This map-only adapter keeps boxes and road-area polygons fixed, and updates
linked lane polygons through their shared raw node. Exact native DD2
rasterization is the forward; projected world-segment distance is the explicitly
approximate STE backward. It does not optimize condition pixels.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .dd2_fixed_input_adapter import RenderedDD2Conditions
from .native_hdmap_conditions import NativeHDMapRenderer


class WorldDividerConditions:
    def __init__(self,bundle,*,raw_root,repository_root,location,line_token,node_token,xy_budget_m=.1,max_turn_change_rad=.08,soft_size=(64,112)):
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        self.bundle=bundle;self.native=NativeHDMapRenderer(raw_root,repository_root,location)
        self.line_token=line_token;self.node_token=node_token;self.budget=float(xy_budget_m);self.max_turn=float(max_turn_change_rad);self.soft_size=soft_size
        if not np.isfinite(self.budget) or self.budget<=0:raise ValueError('Positive metric node budget required')
        lines={r['token']:r for r in self.native.map.line};self.line=lines[line_token];self.ids=tuple(self.line['node_tokens'])
        if node_token not in self.ids[1:-1] or self.ids.count(node_token)!=1:raise ValueError('Only an existing internal divider node can move')
        self.index=self.ids.index(node_token)
        owners=[r for r in self.native.map.lane_divider+self.native.map.road_divider if r['line_token']==line_token]
        if len(owners)!=1:raise ValueError('Expected one native divider semantic element')
        self.element_id=owners[0]['token']
        if any(node_token in r['node_tokens'] for key,r in lines.items() if key!=line_token):raise ValueError('Shared junction node is anchored')
        self.linked_polygons=[r for r in self.native.map.polygon if node_token in r['exterior_node_tokens'] or any(node_token in h['node_tokens'] for h in r['holes'])]
        polygon_ids={r['token'] for r in self.linked_polygons}
        lane_polygon_ids={r['polygon_token'] for r in self.native.map.lane}
        if not polygon_ids.issubset(lane_polygon_ids):raise ValueError('This divider adapter supports linked lane polygons; road-area edits need their own constraints')
        for layer in ['road_segment','drivable_area','walkway','ped_crossing','stop_line','carpark_area']:
            if any(polygon_ids.intersection(row.get('polygon_tokens',[row.get('polygon_token')])) for row in getattr(self.native.map,layer)):
                raise ValueError('Selected node would alter a non-lane surface')
        self.base_polygons={row['token']:self.native.map.extract_polygon(row['token']) for row in self.linked_polygons}
        self.neighbor_polygons=[self.native.map.extract_polygon(row['polygon_token']) for row in self.native.map.lane if row['polygon_token'] not in polygon_ids]
        self.polygon_relations={key:[(shape.intersects(other),shape.touches(other),shape.intersection(other).area>1e-6) for other in self.neighbor_polygons] for key,shape in self.base_polygons.items()}
        self.xy=torch.tensor([self.native.original_xy[x] for x in self.ids],dtype=torch.double)
        self.base_line=LineString(self.xy.numpy());self.base_lengths=np.linalg.norm(np.diff(self.xy.numpy(),axis=0),axis=1)
        self.other_lines=[self.native.map.extract_line(r['line_token']) for r in self.native.map.lane_divider+self.native.map.road_divider if r['line_token']!=line_token]
        self.relations=[(self.base_line.intersects(x),self.base_line.crosses(x),self.base_line.touches(x)) for x in self.other_lines]
        self.distances=[self.base_line.distance(x) for x in self.other_lines]
        polygons=[self.native.map.extract_polygon(t) for r in self.native.map.drivable_area for t in r['polygon_tokens']]
        self.road=unary_union(polygons);self.base_outside=self.base_line.difference(self.road).length
        self.camera=[]
        from pyquaternion import Quaternion
        for r in bundle['records']:
            sd=self.native.nusc.get('sample_data',r['camera_token']);pose=self.native.nusc.get('ego_pose',sd['ego_pose_token']);cal=self.native.nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
            yaw=self.native.functions['quaternion_yaw'](Quaternion(pose['rotation']));c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,-s],[s,c]])
            local=(self.xy.numpy()-np.asarray(pose['translation'])[:2])@rotation
            if np.abs(local).max()>51.2-self.budget*2:raise ValueError('Selected line approaches native patch clipping')
            self.camera.append(dict(rotation=torch.tensor(rotation),translation=torch.tensor(pose['translation'][:2]),camera_rotation=torch.tensor(Quaternion(cal['rotation']).rotation_matrix),camera_translation=torch.tensor(cal['translation']),intrinsic=torch.tensor(cal['camera_intrinsic'])))
        self.last_report={}

    def initial_u(self):return torch.zeros(2,dtype=torch.double)

    def build_geometry(self,u):
        if not torch.is_tensor(u) or not u.is_floating_point() or u.shape!=(2,) or not torch.isfinite(u).all() or (u.detach().abs()>1).any():raise ValueError('Expected bounded world-node XY controls')
        selector=u.new_zeros((len(self.ids),1));selector[self.index]=1
        xy=self.xy.to(u)+selector*(u*self.budget)[None,:]
        return dict(scene_id=self.bundle['scene_id'],normalized_u=u,line_token=self.line_token,element_id=self.element_id,node_tokens=self.ids,node_xy_world=xy,linked_lane_polygon_tokens=tuple(row['token'] for row in self.linked_polygons))

    def geometry_is_valid(self,g):
        from shapely.geometry import LineString,Polygon
        try:
            expected=self.build_geometry(g['normalized_u'])
            if set(g)!=set(expected):return False
            for key,value in expected.items():
                if torch.is_tensor(value):
                    if not torch.is_tensor(g[key]) or not torch.equal(g[key].detach(),value.detach()):return False
                elif g[key]!=value:return False
            xy=g['node_xy_world'].detach().cpu().numpy();line=LineString(xy);delta=np.diff(xy,axis=0);lengths=np.linalg.norm(delta,axis=1)
            turn=np.arctan2(delta[:,1],delta[:,0]);base=np.diff(self.xy.numpy(),axis=0);base_turn=np.arctan2(base[:,1],base[:,0]);angle=np.arctan2(np.sin(np.diff(turn)-np.diff(base_turn)),np.cos(np.diff(turn)-np.diff(base_turn)))
            relations=[(line.intersects(x),line.crosses(x),line.touches(x)) for x in self.other_lines]
            separation=[line.distance(x) for x in self.other_lines]
            issues=[]
            if not line.is_simple or (lengths<.9*self.base_lengths).any():issues.append('collapsed_or_self_crossed_line')
            if np.abs(angle).max()>self.max_turn:issues.append('excess_turn_change')
            if relations!=self.relations:issues.append('divider_intersection_topology_changed')
            if any(old>.2 and new<max(.2,old-.2) for old,new in zip(self.distances,separation)):issues.append('excess_divider_clearance_reduction')
            outside=line.difference(self.road).length
            if outside>self.base_outside+1e-6:issues.append('increased_off_road_length')
            linked_areas={}
            def coords(tokens):return [xy[self.index] if token==self.node_token else self.native.original_xy[token] for token in tokens]
            for row in self.linked_polygons:
                shape=Polygon(coords(row['exterior_node_tokens']),[coords(h['node_tokens']) for h in row['holes']]);base_shape=self.base_polygons[row['token']]
                if not shape.is_valid or shape.is_empty or len(shape.interiors)!=len(base_shape.interiors):issues.append('invalid_linked_lane_polygon');continue
                if abs(shape.area/base_shape.area-1)>.05:issues.append('excess_lane_area_change')
                if shape.difference(self.road).area>base_shape.difference(self.road).area+1e-6:issues.append('lane_surface_leaves_drivable_area')
                relations=[(shape.intersects(other),shape.touches(other),shape.intersection(other).area>1e-6) for other in self.neighbor_polygons]
                if relations!=self.polygon_relations[row['token']]:issues.append('lane_polygon_adjacency_changed')
                linked_areas[row['token']]=shape.area
            self.last_report=dict(linked_lane_areas_m2=linked_areas,valid=not issues,violations=issues,max_turn_change_rad=float(np.abs(angle).max()),max_xy_change_m=float(np.abs(xy-self.xy.numpy()).max()),off_road_length_m=outside,scope='Native divider topology and road containment; endpoints, drivable-area/road-segment surfaces and boxes fixed; shared lane polygons updated. Not a lane-design or expert-planning certificate.')
            return not issues
        except (KeyError,ValueError,TypeError):return False

    def install(self,g):
        if not self.geometry_is_valid(g):raise ValueError('Invalid world divider geometry: '+str(self.last_report))
        self.native.set_world_nodes({self.node_token:g['node_xy_world'][self.index].detach().cpu().numpy()})

    def _soft(self,g,record_index):
        xy=g['node_xy_world'];meta=self.camera[record_index]
        # Fixed fractions per original segment; no learnable raster or embedding.
        portions=[]
        for index,length in enumerate(self.base_lengths):
            fraction=torch.linspace(0,1,max(2,int(np.ceil(length))+1),dtype=xy.dtype,device=xy.device)
            pts=xy[index][None,:]*(1-fraction[:,None])+xy[index+1][None,:]*fraction[:,None]
            portions.append(pts[:-1] if index<len(self.base_lengths)-1 else pts)
        xy=torch.cat(portions);local=(xy-meta['translation'].to(xy))@meta['rotation'].to(xy)
        xyz=torch.cat((local,torch.zeros_like(local[:,:1])),-1);camera=(xyz-meta['camera_translation'].to(xy))@meta['camera_rotation'].to(xy)
        homogeneous=camera@meta['intrinsic'].to(xy).T;pixels=homogeneous[:,:2]/homogeneous[:,2:].clamp_min(.5)
        height,width=self.soft_size;pixels=pixels*xy.new_tensor([width/1600,height/900])
        valid=(camera[:-1,2].detach()>.5)&(camera[1:,2].detach()>.5)
        if not valid.any():return xy.sum()*0+xy.new_zeros((height,width))
        start=pixels[:-1][valid];end=pixels[1:][valid];direction=end-start
        yy,xx=torch.meshgrid(torch.arange(height,dtype=xy.dtype,device=xy.device)+.5,torch.arange(width,dtype=xy.dtype,device=xy.device)+.5,indexing='ij');grid=torch.stack((xx,yy),-1)
        rel=grid[None]-start[:,None,None,:];along=(rel*direction[:,None,None,:]).sum(-1)/direction.square().sum(-1).clamp_min(1e-10)[:,None,None];closest=rel-along.clamp(0,1)[...,None]*direction[:,None,None,:];distance=closest.square().sum(-1)
        return torch.exp(-distance/(2*.8**2)).amax(0)

    def __call__(self,g):
        self.install(g);u=g['normalized_u'];frames=[[None]*6 for _ in range(8)]
        for i,r in enumerate(self.bundle['records']):
            hard=self.native.condition(r['camera_token'],self.bundle['scene_id']).to(u)
            if torch.is_grad_enabled() and u.requires_grad:
                soft=F.interpolate(self._soft(g,i)[None,None],size=(256,448),mode='bilinear',align_corners=False)[0,0]
                # DD2's channel reversal puts dividers in RGB blue; road contours
                # and crossings have priority over dividers in the native raster.
                visible=(hard[0]<-.99)&(hard[1]<-.99)
                proxy=torch.stack((soft*0,soft*0,2*soft*visible.to(soft)))
                hard=hard+(proxy-proxy.detach())
            frames[r['frame']][r['view']]=hard
        original=self.bundle['source_conditions']['input_dict'];maps=torch.stack([torch.cat(row,-1) for row in frames]).to(dtype=original['grounding_downsampler_input'].dtype)
        return RenderedDD2Conditions(self.bundle['scene_id'],tuple(self.bundle['camera_order']),dict(grounding_downsampler_input=maps,box_downsampler_input=original['box_downsampler_input'].to(u.device)))

    def segmentation(self,g,*,source_path,lidar2ego,ego2global,classes):
        self.install(g)
        hard=torch.as_tensor(self.native.segmentation(source_path,lidar2ego,ego2global,classes),dtype=g['normalized_u'].dtype,device=g['normalized_u'].device)
        if not torch.is_grad_enabled() or not g['normalized_u'].requires_grad:return hard
        xy=g['node_xy_world'];transform=np.asarray(ego2global)@np.asarray(lidar2ego);yaw=np.arctan2(transform[1,0],transform[0,0]);c,s=np.cos(yaw),np.sin(yaw);rotation=xy.new_tensor([[c,-s],[s,c]])
        values=torch.arange(200,dtype=xy.dtype,device=xy.device)*.5-49.75
        xx,yy=torch.meshgrid(values,values,indexing='ij');world=torch.stack((xx,yy),-1)@rotation.T+xy.new_tensor(transform[:2,3])
        start,end=xy[:-1],xy[1:];direction=end-start;rel=world[None]-start[:,None,None,:];along=(rel*direction[:,None,None,:]).sum(-1)/direction.square().sum(-1)[:,None,None];distance=(rel-along.clamp(0,1)[...,None]*direction[:,None,None,:]).square().sum(-1)
        soft=torch.exp(-distance/(2*.5**2)).amax(0);proxy=torch.stack([soft if name=='divider' else soft*0 for name in classes])
        return hard+(proxy-proxy.detach())
