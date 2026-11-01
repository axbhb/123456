"""Prepare native 61-frame geometric inputs; RGB is not fed to MagicDrive."""
import os,sys,json,pickle,shutil,importlib.util,argparse
p=argparse.ArgumentParser();p.add_argument("--clip",required=True);p.add_argument("--output",required=True);p.add_argument("--profile-first-frame",action="store_true");args=p.parse_args()
from pathlib import Path
R=Path('/srv/table2');W=R/'generator_deployment_20261028';S=W/'sources/magicdrive_video';B=S/'third_party/bevfusion'
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.chdir(str(S))
# Reuse compiled extensions from the same BEVFusion revision and Torch 1.11.
compiled=R/'model_readiness_20261028/preparation/sources/bevfusion_db75150'
for f in (compiled/'mmdet3d/ops').rglob('*.so'):
 dst=B/f.relative_to(compiled);dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(f,dst)
p=B/'mmdet3d/ops/spconv/conv.py';s=p.read_text().replace('@CONV_LAYERS.register_module()', '@CONV_LAYERS.register_module(force=True)');p.write_text(s)
sys.path[:0]=[str(S),str(B),str(S/'third_party/diffusers/src')]
sys.path.append('/home/user/gzx/envs/wmautoattack/lib/python3.9/site-packages')
sys.path.append('/home/user/anaconda3/envs/torch200/lib/python3.9/site-packages')
pure=W/'cpu_python_deps';pure.mkdir(exist_ok=True)
for package in ['omegaconf','antlr4']:
 dst=pure/package
 if not dst.exists():
  for parent in [R/'envs/dd2_gpu/lib/python3.9/site-packages',Path('/home/user/anaconda3/envs/torch200/lib/python3.9/site-packages'),Path('/home/user/gzx/envs/wmautoattack/lib/python3.9/site-packages')]:
   if (parent/package).is_dir():shutil.copytree(parent/package,dst);break
sys.path.insert(0,str(pure))
import torch,numpy as np
from nuscenes import NuScenes
from omegaconf import OmegaConf
from mmdet3d.datasets import build_dataset
from magicdrive.dataset import collate_fn_single
# Camera/map generation does not read LiDAR files; retain all original transforms.
p=S/'tools/data_converter/nuscenes_converter.py';s=p.read_text().replace('        mmcv.check_file_exist(lidar_path)','        # Camera-only preparation: LiDAR points are not consumed.');p.write_text(s)
spec=importlib.util.spec_from_file_location('magicdrive_native_converter',str(p));converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
D=R/'assets/nuscenes_mini_temporal_camera_v1';nusc=NuScenes(version='v1.0-mini',dataroot=str(D),verbose=False)
plan=json.loads(Path(args.clip).read_text());rows=pickle.load((R/'datasets/streampetr_mini_scene_283_121/infos_train.pkl').open('rb'))['infos'];by_token={i['token']:i for i in rows};base=[by_token[t] for t in plan['tokens']];W=Path(args.output);W.mkdir(parents=True,exist_ok=True)
# Native annotations are at irregular ~2 Hz intervals. Resample ego poses
# to six substeps per interval, preserving the model's 6-frame keyframe cadence.
from pyquaternion import Quaternion
from bisect import bisect_right
scene=base[0]['scene_token'];streams={}
for cam in converter.camera_types:
 seq=[r for r in nusc.sample_data if r.get('channel')==cam and nusc.get('sample',r['sample_token'])['scene_token']==scene]
 seq.sort(key=lambda x:x['timestamp']);streams[cam]=([r['timestamp'] for r in seq],seq)
def add_record(table,record):
 items=getattr(nusc,table);nusc._token2ind[table][record['token']]=len(items);items.append(record)
def virtual_camera(cam,timestamp,sample_token,tag):
 times,records=streams[cam];idx=bisect_right(times,timestamp);idx=min(max(idx,1),len(times)-1)
 left,right=records[idx-1],records[idx];alpha=(timestamp-left['timestamp'])/(right['timestamp']-left['timestamp'])
 assert 0<=alpha<=1,('Pose extrapolation is not required',cam,timestamp)
 lp=nusc.get('ego_pose',left['ego_pose_token']);rp=nusc.get('ego_pose',right['ego_pose_token'])
 pose=dict(lp,token=tag+'-pose',timestamp=timestamp,translation=((1-alpha)*np.array(lp['translation'])+alpha*np.array(rp['translation'])).tolist(),rotation=Quaternion.slerp(Quaternion(lp['rotation']),Quaternion(rp['rotation']),amount=alpha).elements.tolist())
 add_record('ego_pose',pose)
 rec=dict(left,token=tag,ego_pose_token=pose['token'],timestamp=timestamp,sample_token=sample_token,is_key_frame=False)
 add_record('sample_data',rec);return rec
infos=[]
for index,key in enumerate(base[:11]):
 sample=nusc.get('sample',key['token'])
 if index:
  previous=base[index-1]
  for sub in range(1,6):
   timestamp=round(previous['timestamp']+(key['timestamp']-previous['timestamp'])*sub/6)
   cams=[virtual_camera(cam,timestamp,sample['token'],'magicdrive-interp-%d-%d-%s'%(index,sub,cam)) for cam in converter.camera_types]
   frame=converter._sample_data_transform(nusc,sample,max_sweeps=0,ego_pose_token=cams[0]['ego_pose_token'],cam_tokens=[c['token'] for c in cams],keyframe=False)
   frame['token']=sample['token']+';-'+str(6-sub);frame['timestamp']=timestamp;infos.append(frame)
 infos.append(converter._sample_data_transform(nusc,sample,max_sweeps=0,keyframe=True))
assert len(infos)==61
keyframes=[i for i,x in enumerate(infos) if x['is_key_frame']]
assert keyframes==list(range(0,61,6))
carrier_replacements=0
for info in infos:
 sample=nusc.get('sample',info['token'].split(';')[0])
 for cam,ci in info['cams'].items():
  path=Path(ci['data_path'])
  if not path.is_file():
   carrier=Path(nusc.get_sample_data_path(sample['data'][cam]));assert carrier.is_file()
   ci['geometry_sample_data_token']=ci['sample_data_token'];ci['data_path']=str(carrier);carrier_replacements+=1
ann=W/'magicdrive_real_12hz_infos.pkl'
with ann.open('wb') as f:pickle.dump({'infos':infos,'metadata':{'version':'v1.0-mini'},'scene_tokens':[[i['token'] for i in infos]]},f)
cfg=OmegaConf.load(str(R/'assets/generators/magicdrive_t_60f/hydra/config.yaml'))
cfg.dataset.dataset_root=str(D)+'/'
for pipe in [cfg.dataset.test_pipeline,cfg.dataset.train_pipeline]:
 for transform in pipe:
  if transform.type=='LoadBEVSegmentation':transform.cache_file=None
cfg.dataset.data.val.ann_file=str(ann);cfg.dataset.data.val.start_on_keyframe=True
cfg.dataset.data.val.start_on_firstframe=True
val=build_dataset(OmegaConf.to_container(cfg.dataset.data.val,resolve=True))
import time
if args.profile_first_frame:
 frame=val.get_data_info(0)[0];val.pre_pipeline(frame)
 for step in val.pipeline.transforms:
  start=time.monotonic();frame=step(frame)
  print(json.dumps({'transform':type(step).__name__,'seconds':time.monotonic()-start}),flush=True)
 raise SystemExit(0)
native_pipeline=val.pipeline
prepared_frames=[0]
def timed_pipeline(frame):
 start=time.monotonic();result=native_pipeline(frame);prepared_frames[0]+=1
 print(json.dumps({'phase':'native_frame_preprocessing','frame':prepared_frames[0],'total':61,'seconds':time.monotonic()-start}),flush=True)
 return result
val.pipeline=timed_pipeline
example=val[0]
augmentations=np.stack([frame['img_aug_matrix'].data.numpy() for frame in example])
assert np.allclose(augmentations,augmentations[0,0])
aug=augmentations[0,0]
source_to_generated=[[float(aug[0,0]),float(aug[0,1]),float(aug[0,3])],[float(aug[1,0]),float(aug[1,1]),float(aug[1,3])],[0,0,1]]
batch=collate_fn_single(example,tokenizer=None,template=cfg.dataset.template,bbox_mode=cfg.model.bbox_mode,bbox_view_shared=cfg.model.bbox_view_shared,is_train=False,keyframe_rate=6)
# Export tensors and strings only, so inference has no MMCV dependency.
batch={k:batch[k] for k in ['captions','bev_map_with_aux','camera_param','kwargs']}
torch.save(batch,W/'magicdrive_real_61frame_batch.pt')
record={'status':'prepared','frames':61,'keyframes':keyframes,'timestamps':[i['timestamp'] for i in infos],'native_converter':True,'pose_resampling':'Linear translation and quaternion SLERP, six substeps per real annotated interval; no extrapolation','rgb_model_input':False,'carrier_image_replacements':carrier_replacements,'carrier_note':'Existing keyframe RGB supplies image dimensions to native preprocessing only; no carrier pixels are sent to the generator or represented as real non-keyframe video. Camera poses, map geometry and keyframe boxes come from native metadata.','tensor_shapes':{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)}}
(W/'magicdrive_input_status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))

status=json.loads((W/'magicdrive_input_status.json').read_text());status.update(camera_order=list(cfg.dataset.view_order));status['source_to_generated']=source_to_generated;status['geometry_verified_against_native_img_aug_matrix']=True;(W/'magicdrive_input_status.json').write_text(json.dumps(status,indent=2))
