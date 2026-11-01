"""Export all four genuine keyframes with synchronized native 2D/3D labels."""
import copy,importlib.util,pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .temporal_detection_labels import TemporalDetectionLabels


def load_source_infos(root,bundle):
    if bundle.get('source_infos_path'):
        path=Path(bundle['source_infos_path'])
    else:
        roles={'train_a':'train','train_b':'attack_dev'}
        if bundle['role'] not in roles:
            raise ValueError('Additional scene requires explicit source_infos_path')
        path=Path(root)/'assets/streampetr_subset'/('streampetr_'+roles[bundle['role']]+'.pkl')
    source=pickle.loads(path.read_bytes())
    bytoken={x['token']:x for x in source['infos']}
    if len(bytoken)!=len(source['infos']):raise ValueError('Repeated native source token')
    tokens=[x['sample_token'] for x in bundle['keyframes']]
    if len(tokens)!=len(set(tokens)) or not set(tokens)<=set(bytoken):
        raise ValueError('Native source is missing or repeats requested keyframes')
    rows=[bytoken[token] for token in tokens]
    if any(row['scene_token']!=bundle['scene_id'] for row in rows):
        raise ValueError('Native source scene differs from physical bundle')
    return rows


def export_native_dataset(root,renderer,geometry,video,output):
    from pyquaternion import Quaternion
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if tuple(video.shape)!=(1,8,6,3,256,448) or not torch.isfinite(video).all() or video.min()<0 or video.max()>1:raise ValueError('Canonical finite 8-frame 6-view RGB required')
    source=load_source_infos(root,renderer.bundle);labels=TemporalDetectionLabels(renderer,source)
    from nuscenes import NuScenes
    nusc=NuScenes(version='v1.0-mini',dataroot=str(root/'assets/nuscenes_mini_temporal_camera_v1'),verbose=False)
    runtime=root/'frameworks/20261028-v1/runtime_sources/streampetr'
    spec=importlib.util.spec_from_file_location('_native_export_converter',runtime/'tools/data_converter/nuscenes_converter.py');converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    states=renderer.build_geometry(geometry['normalized_u'])['keyframe_states'];edits={}
    for record,state in zip(renderer.bundle['keyframes'],states):
        for j,token in enumerate(record['annotation_tokens']):edits[token]={k:v[j].detach().cpu().numpy() for k,v in state.items()}
    get_box,box_velocity=nusc.get_box,nusc.box_velocity
    def edited_box(token):
        b=get_box(token)
        if token in edits:
            e=edits[token];b.center=e['centers_world'].copy();b.wlh=e['sizes_wlh'].copy();b.orientation=Quaternion(matrix=e['rotations_world'])
        return b
    nusc.get_box=edited_box
    nusc.box_velocity=lambda token:edits[token]['velocities_world'].copy() if token in edits else box_velocity(token)
    infos=[];zero=bool((geometry['normalized_u'].detach()==0).all());max_zero_2d_error=0.
    for i,src in enumerate(source):
        info=copy.deepcopy(src);native=labels(geometry,i).detach().cpu().numpy()
        info['gt_boxes']=native[:,:7];info['gt_velocity']=native[:,7:]
        fields={k:[] for k in ['bboxes2d','bboxes3d_cams','labels2d','centers2d','depths','bboxes_ignore','visibilities']}
        slot=renderer.bundle['trainable_slots'][i]
        for c,(name,cam) in enumerate(info['cams'].items()):
            view=renderer.bundle['camera_order'].index(name)
            row=next(r for r in renderer.bundle['records'] if r['frame']==slot and r['view']==view)
            if row['camera_token']!=cam['sample_data_token'] or row['sample_token']!=info['token']:raise ValueError('Native camera/keyframe identity mismatch')
            anns=converter.get_2d_boxes(nusc,cam['sample_data_token'],visibilities=['','1','2','3','4'],mono3d=True)
            b2=[];b3=[];classes=[];centers=[];depths=[];ignore=[];visibility=[]
            for ann in anns:
                if ann is None or ann.get('ignore',False):continue
                x,y,w,h=ann['bbox']
                if max(0,min(x+w,1600)-max(x,0))*max(0,min(y+h,900)-max(y,0))==0 or ann['area']<=0 or w<1 or h<1 or ann['category_name'] not in converter.nus_categories:continue
                box=[x,y,x+w,y+h]
                if ann.get('iscrowd',False):ignore.append(box);continue
                b2.append(box);b3.append(ann['bbox_cam3d']);classes.append(ann['category_id']);centers.append(ann['center2d'][:2]);depths.append(ann['center2d'][2]);visibility.append(ann['visibility_token'])
            current=dict(bboxes2d=np.asarray(b2,dtype=np.float32).reshape(-1,4),bboxes3d_cams=np.asarray(b3,dtype=np.float32).reshape(-1,7),labels2d=np.asarray(classes,dtype=np.int64),centers2d=np.asarray(centers,dtype=np.float32).reshape(-1,2),depths=np.asarray(depths,dtype=np.float32),bboxes_ignore=np.asarray(ignore,dtype=np.float32).reshape(-1,4),visibilities=visibility)
            if zero:
                for key in ['bboxes2d','bboxes3d_cams','labels2d','centers2d','depths']:
                    old=np.asarray(src[key][c]);new=current[key]
                    if old.size!=new.size or not np.allclose(old.reshape(new.shape),new,atol=2e-3,rtol=1e-5):raise ValueError('Zero-edit original 2D/3D label mismatch: '+key)
                    if new.size:max_zero_2d_error=max(max_zero_2d_error,float(np.abs(old.reshape(new.shape)-new).max()))
            current['bboxes2d']*=np.array([448/1600,256/900,448/1600,256/900],dtype=np.float32)
            current['bboxes_ignore']*=np.array([448/1600,256/900,448/1600,256/900],dtype=np.float32)
            current['centers2d']*=np.array([448/1600,256/900],dtype=np.float32)
            for key in fields:fields[key].append(current[key])
            pixels=(video[0,slot,view].detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
            path=output/'images'/info['token']/(name+'.png');path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels).save(path)
            cam['data_path']=str(path);cam['cam_intrinsic']=np.diag([448/1600,256/900,1.])@np.asarray(cam['cam_intrinsic'])
        info.update(fields);info.update(prev='',next='',sweeps=[],frame_idx=0);infos.append(info)
    path=output/'native_infos.pkl';path.write_bytes(pickle.dumps(dict(infos=infos,metadata=dict(version='v1.0-mini')),protocol=4))
    from mmcv import Config
    cfg=Config.fromfile(str(root/'frameworks/20261028-v1/clean_e2e/clean_config.py'));cfg.data.train.ann_file=str(path);cfg.data.train.seq_split_num=1
    cfg.dump(str(output/'native_config.py'))
    return dict(video_point=video.detach().clone(),u_point=geometry['normalized_u'].detach().clone(),infos=infos,source_infos=source,config=cfg,labels=labels,ann_file=path,zero_edit_max_label_error=max_zero_2d_error,slots=tuple(renderer.bundle['trainable_slots']))
