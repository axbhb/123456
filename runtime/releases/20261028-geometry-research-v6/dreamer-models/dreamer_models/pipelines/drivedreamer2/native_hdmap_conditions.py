"""Original DD2 map rendering from editable nuScenes world-coordinate nodes.

No learned weights or independent raster parameters. Native camera-map conversion
retains DD2's yaw-only local ground-plane convention and discrete rasterization.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np


class NativeHDMapRenderer:
    def __init__(self, raw_root, repository_root, location, nusc=None):
        import cv2
        from decimal import Decimal, ROUND_HALF_UP
        from PIL import Image
        from pyquaternion import Quaternion
        from shapely import affinity, ops
        from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
        import matplotlib.style as style
        if 'seaborn-whitegrid' not in style.library and 'seaborn-v0_8-whitegrid' in style.library:
            style.library['seaborn-whitegrid']=style.library['seaborn-v0_8-whitegrid']
        from nuscenes import NuScenes
        from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
        self.location=location
        self.nusc=nusc if nusc is not None else NuScenes(version='v1.0-mini',dataroot=str(raw_root),verbose=False)
        self.map=NuScenesMap(dataroot=str(raw_root),map_name=location)
        self.explorer=NuScenesMapExplorer(self.map)
        path=Path(repository_root)/'dreamer-datasets/dd_scripts/converters/nuscenes_converter.py'
        tree=ast.parse(path.read_text())
        names={'quaternion_yaw','get_map_geom','get_ped_crossing_line','line_geoms_to_vectors','one_type_line_geom_to_vectors','sample_pts_from_line','poly_geoms_to_vectors','preprocess_map','line_geom_to_mask','overlap_filter','round_dec','mask_for_lines','get_discrete_degree','view_points_depth'}
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in functions}!=names:raise ValueError('Native map functions missing')
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NuScenesConverter')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_hdmap')
        ns=dict(np=np,cv2=cv2,Decimal=Decimal,ROUND_HALF_UP=ROUND_HALF_UP,Image=Image,Quaternion=Quaternion,affinity=affinity,ops=ops,LineString=LineString,box=box,MultiPolygon=MultiPolygon,MultiLineString=MultiLineString,NuScenesMapExplorer=NuScenesMapExplorer)
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions+[method],type_ignores=[])),str(path),'exec'),ns)
        self.functions=ns
        self.receiver=SimpleNamespace(nusc=self.nusc,nusc_maps={location:self.map},map_explorer={location:self.explorer},class2label={'road_divider':0,'lane_divider':0,'ped_crossing':1,'contours':2,'others':-1})
        self.nodes={row['token']:row for row in self.map.node}
        self.original_xy={token:(row['x'],row['y']) for token,row in self.nodes.items()}

    def set_world_nodes(self, edits):
        """Install absolute XY for named existing nodes; reset previous edits first."""
        for token,xy in edits.items():
            if token not in self.nodes or np.asarray(xy).shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid raw map node')
        for token,xy in self.original_xy.items():self.nodes[token]['x'],self.nodes[token]['y']=xy
        for token,xy in edits.items():self.nodes[token]['x'],self.nodes[token]['y']=map(float,xy)

    def image(self, camera_token, scene_token):
        return self.functions['_get_hdmap'](self.receiver,camera_token,scene_token)

    def condition(self,camera_token,scene_token,output_size=(256,448)):
        import torch
        from PIL import Image
        image=self.image(camera_token,scene_token).resize((output_size[1],output_size[0]),Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float().div(255).sub(.5).div(.5)

    def segmentation(self,source_path,lidar2ego,ego2global,classes):
        """Execute the actual BEVFusion map-label class on this edited map object."""
        if not hasattr(self,'_seg_class'):
            from typing import Dict,Any,Tuple
            tree=ast.parse(Path(source_path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadBEVSegmentation');node.decorator_list=[]
            class Aliases:
                bool=np.bool_
                long=np.int64
                def __getattr__(self,key):return getattr(np,key)
            namespace=dict(np=Aliases(),Dict=Dict,Any=Any,Tuple=Tuple)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source_path),'exec'),namespace)
            self._seg_class=namespace['LoadBEVSegmentation']
        target=self._seg_class.__new__(self._seg_class);target.patch_size=(100.,100.);target.canvas_size=(200,200);target.classes=classes;target.maps={self.location:self.map}
        return target(dict(lidar_aug_matrix=np.eye(4),lidar2ego=np.asarray(lidar2ego),ego2global=np.asarray(ego2global),location=self.location))['gt_masks_bev']
