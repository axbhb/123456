"""Derive VAD object/motion supervision using its installed native converter.

Only original selected world-box annotations change; expert trajectories,
cameras, identities, semantic classes and object sizes remain fixed.
"""
import ast,copy,types,math
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box


def edited_annotation_view(nusc,renderer,u):
    view=copy.copy(nusc);view.sample_annotation=list(nusc.sample_annotation)
    selected=set(renderer.selected)
    def changes(annotation):
        timestamp=nusc.get('sample',annotation['sample_token'])['timestamp']
        delta,velocity,_=renderer._delta([annotation['instance_token']],timestamp,u)
        return delta[0].detach().cpu().numpy(),velocity[0].detach().cpu().numpy()
    for index,annotation in enumerate(nusc.sample_annotation):
        if annotation['instance_token'] not in selected:continue
        row=copy.deepcopy(annotation);delta,_=changes(row)
        row['translation']=(np.asarray(row['translation'])+np.r_[delta[:2],0.]).tolist()
        row['rotation']=(Quaternion(axis=[0,0,1],angle=float(delta[2]))*Quaternion(row['rotation'])).elements.tolist()
        view.sample_annotation[index]=row
    def velocity(self,token,max_time_diff=1.5):
        original=nusc.box_velocity(token,max_time_diff=max_time_diff)
        annotation=nusc.get('sample_annotation',token)
        if annotation['instance_token'] in selected:
            _,change=changes(annotation);original=original+np.r_[change[:2],0.]
        return original
    view.box_velocity=types.MethodType(velocity,view)
    return view


def native_vad_object_labels(nusc,row,converter_path):
    tree=ast.parse(converter_path.read_text())
    converter=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fill_trainval_infos')
    def assignment(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    block=next(n.body for n in ast.walk(converter) if isinstance(n,ast.If) and any(assignment(x,'annotations') for x in n.body))
    start=next(i for i,n in enumerate(block) if assignment(n,'annotations'))
    end=next(i for i,n in enumerate(block) if assignment(n,'ego_his_trajs'))
    source=ast.Module(body=copy.deepcopy(block[start:end]),type_ignores=[])
    sample=nusc.get('sample',row['token']);sd=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    pose=nusc.get('ego_pose',sd['ego_pose_token']);sensor=nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
    annotations=[nusc.get('sample_annotation',t) for t in sample['anns']]
    if len(annotations)!=len(row['gt_boxes']):raise ValueError('Native object ordering/count changed')
    mapping={ann['category_name']:name for ann,name in zip(annotations,row['gt_names'])}
    categories={ann['category_name']:int(feat[8]) for ann,feat in zip(annotations,row['gt_agent_lcf_feat'])}
    environment=dict(np=np,math=math,Quaternion=Quaternion,Box=Box,nusc=nusc,sample=sample,
                     boxes=nusc.get_sample_data(sample['data']['LIDAR_TOP'])[1],
                     pose_record=pose,cs_record=sensor,e2g_r_mat=Quaternion(pose['rotation']).rotation_matrix,
                     l2e_r_mat=Quaternion(sensor['rotation']).rotation_matrix,
                     NuScenesDataset=types.SimpleNamespace(NameMapping=mapping),cat2idx=categories,fut_ts=6)
    rpy=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quart_to_rpy')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(rpy)],type_ignores=[])),str(converter_path),'exec'),environment)
    exec(compile(ast.fix_missing_locations(source),str(converter_path),'exec'),environment)
    values=environment
    return dict(gt_boxes=values['gt_boxes'],gt_names=values['names'],gt_velocity=values['velocity'].reshape(-1,2),
                gt_agent_fut_trajs=values['gt_fut_trajs'].reshape(-1,12).astype(np.float32),
                gt_agent_fut_masks=values['gt_fut_masks'].astype(np.float32),gt_agent_lcf_feat=values['agent_lcf_feat'].astype(np.float32),
                gt_agent_fut_yaw=values['gt_fut_yaw'].astype(np.float32),gt_agent_fut_goal=values['gt_fut_goal'].astype(np.float32))
