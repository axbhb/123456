"""Build a clean generated-camera training item and four official mini_val frames. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
import json,pickle,copy,hashlib,tarfile,sys,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
R=Path('/srv/table2');S=R/'frameworks/20261028-v1';O=S/'clean_e2e';O.mkdir(exist_ok=True)
D=R/'assets/nuscenes_mini_temporal_camera_v1';N=R/'assets/dd2_native_temporal_subset'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hook(x):
 if '__ndarray__' in x:return np.asarray(x['__ndarray__'],dtype=x['dtype']).reshape(x['shape'])
 return x
def check(ok,msg):
 if not ok:raise ValueError(msg)
manifest=json.load(open(N/'manifest.json'));rows=json.load(open(N/'attack_development/labels.json'),object_hook=hook)
gen=json.load(open(S/'dd2_native_gpu_smoke/result.json'));check(gen['status']=='native_generation_passed' and not gen['attack_update_executed'],'Only clean native generated artifacts accepted')
video_path=Path(gen['video_tensor']);video=torch.load(video_path,map_location='cpu');check(tuple(video.shape)==(8,3,256,2688),'Unexpected native tensor shape');check(bool(torch.isfinite(video).all()),'Nonfinite video')
check(float(video.min())>=0 and float(video.max())<=1,'Out-of-range video')
sd={x['token']:x for x in json.load(open(D/'v1.0-mini/sample_data.json'))}
samples={x['token']:x for x in json.load(open(D/'v1.0-mini/sample.json'))}
infos=pickle.load(open(R/'assets/streampetr_subset/streampetr_attack_dev.pkl','rb'))['infos'];bytoken={x['token']:x for x in infos}
accepted=[];rejected=[];mapping=[];frame_images=[]
for t in range(8):
 records=[rows[c*8+t] for c in range(6)];tokens={x['sample_token'] for x in records};reasons=[]
 if len(tokens)!=1:reasons.append('camera_sample_tokens_differ')
 if not all(sd[x['cam_token']]['is_key_frame'] for x in records):reasons.append('contains_nonkey_camera_frames')
 if reasons:rejected.append({'frame':t,'reasons':reasons});continue
 token=records[0]['sample_token'];check(token in bytoken,'Missing matching official StreamPETR info');src=bytoken[token];info=copy.deepcopy(src)
 check(info['scene_token']==gen['scene_token'],'Wrong generated scene')
 camera_rows={x['cam_type'].upper():x for x in records};order=list(src['cams'])
 for camera_index,cam_name in enumerate(order):
  row=camera_rows[cam_name];cam=info['cams'][cam_name];meta=sd[row['cam_token']];pano_index=manifest['camera_order'].index(cam_name)
  check(row['cam_token']==cam['sample_data_token'],'Camera token mismatch')
  check(row['cam_timestamp']==cam['timestamp']==meta['timestamp'],'Timestamp mismatch')
  check(meta['sample_token']==token,'Wrong camera sample')
  check(np.allclose(row['calib']['cam_intrinsic'][:3,:3],cam['cam_intrinsic'],atol=1e-4,rtol=1e-6),'Native/official intrinsics differ')
  width,height=meta['width'],meta['height'];check((width,height)==(1600,900),'Unexpected source size')
  tile=video[t,:,:,448*pano_index:448*(pano_index+1)].permute(1,2,0).float().numpy();pixels=np.rint(tile*255).astype(np.uint8)
  target=O/'generated_images'/token/(cam_name+'.png');target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels,'RGB').save(target)
  check(np.array_equal(np.asarray(Image.open(target)),pixels),'PNG round trip differs')
  scale=np.diag([448/width,256/height,1.]);original_k=np.asarray(cam['cam_intrinsic']).copy();cam['cam_intrinsic']=scale@original_k;cam['data_path']=str(target)
  info['bboxes2d'][camera_index]=np.asarray(src['bboxes2d'][camera_index],dtype=np.float32).reshape(-1,4)*np.array([448/width,256/height,448/width,256/height],dtype=np.float32)
  info['centers2d'][camera_index]=np.asarray(src['centers2d'][camera_index],dtype=np.float32).reshape(-1,2)*np.array([448/width,256/height],dtype=np.float32)
  boxes_cam=np.asarray(src['bboxes3d_cams'][camera_index]);projection_error=0.
  if len(boxes_cam):
   xyz=boxes_cam[:,:3].T;old=original_k@xyz;new=cam['cam_intrinsic']@xyz
   projection_error=float(np.max(np.abs(new[:2]/new[2:]-np.array([[448/width],[256/height]])*(old[:2]/old[2:]))));check(projection_error<1e-8,'Projection resize mismatch')
  frame_images.append({'frame':t,'camera':cam_name,'pano_index':pano_index,'dataset_index':camera_index,'sample_data_token':row['cam_token'],'timestamp_us':row['cam_timestamp'],'path':str(target),'sha256':digest(target),'projection_resize_max_error_px':projection_error})
 for key in ['gt_boxes','gt_names','gt_velocity','valid_flag','num_lidar_pts','num_radar_pts']:
  check(np.array_equal(info[key],src[key],equal_nan=True) if np.asarray(src[key]).dtype.kind not in 'OU' else np.array_equal(info[key],src[key]),'Changed 3D label '+key)
 info.update(prev='',next='',sweeps=[],frame_idx=0);accepted.append(info);mapping.append({'frame':t,'sample_token':token,'scene_token':info['scene_token'],'native_camera_order':manifest['camera_order'],'streampetr_camera_order':order})
check(len(accepted)>=1,'No fully aligned generated keyframe')
# Preserve exact original bytes for the official clean validation images.
archive=R/'assets/clean_mini_val_images.tar'
with tarfile.open(archive) as tar:
 vm=json.load(tar.extractfile('manifest.json'))
 for row in vm['files']:
  dst=(D/row['path']).resolve();check(D.resolve() in dst.parents,'Unsafe archive path');member=tar.getmember(row['path']);check(member.isfile(),'Not a regular file');payload=tar.extractfile(member).read()
  check(len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'],'Validation image hash mismatch');dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():check(digest(dst)==row['sha256'],'Existing clean validation file differs')
  else:dst.write_bytes(payload)
val=pickle.load(open(S/'clean_mini_val_selected.pkl','rb'));valid_infos=val['infos']
from nuscenes.utils.splits import create_splits_scenes
official_val=set(create_splits_scenes()['mini_val']);scenes={x['token']:x['name'] for x in json.load(open(D/'v1.0-mini/scene.json'))}
check(set(vm['sample_tokens'])=={x['token'] for x in valid_infos},'Validation archive tokens mismatch')
for info in valid_infos:
 check(scenes[info['scene_token']] in official_val,'Not official mini_val')
 check(info['scene_token'] not in {x['scene_token'] for x in accepted},'Training/validation scene leakage')
 check(info['timestamp']==samples[info['token']]['timestamp'],'Validation timestamp mismatch')
 for name,cam in info['cams'].items():
  rel='samples/'+cam['data_path'].split('/samples/',1)[1];cam['data_path']=str(D/rel);meta=sd[cam['sample_data_token']]
  check(meta['is_key_frame'] and meta['sample_token']==info['token'] and meta['timestamp']==cam['timestamp'],'Validation camera mismatch');check((D/rel).is_file(),'Missing validation image')
valid_infos.sort(key=lambda x:(x['timestamp']))
for filename,payload in [('generated_train.pkl',{'infos':accepted,'metadata':{'version':'v1.0-mini'}}),('clean_val.pkl',{'infos':valid_infos,'metadata':{'version':'v1.0-mini'}})]:
 (O/filename).write_bytes(pickle.dumps(payload,protocol=4))
# The native 2D transform must remain training=True to transform auxiliary targets.
sys.path.insert(0,str(S/'runtime_sources/streampetr'))
from mmcv import Config
cfg=Config.fromfile(str(S/'runtime_sources/streampetr/smoke_v100.py'))
cfg.data.train.ann_file=str(O/'generated_train.pkl');cfg.data.train.seq_split_num=1
for step in cfg.data.train.pipeline:
 if step['type']=='ResizeCropFlipRotImage':
  step['training']=True;step['data_aug_conf'].update(H=256,W=448,resize_lim=(704/448,704/448),final_dim=(256,704),bot_pct_lim=(0.,0.),rot_lim=(0.,0.),rand_flip=False)
 if step['type']=='GlobalRotScaleTransImage':step.update(rot_range=(0.,0.),scale_ratio_range=(1.,1.),translation_std=(0.,0.,0.))
for split in ['val','test']:cfg.data[split].update(ann_file=str(O/'clean_val.pkl'),test_mode=True)
cfg.dump(str(O/'clean_config.py'))
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
train=build_dataset(cfg.data.train);test=build_dataset(cfg.data.test)
item=train[0];check(tuple(item['img'].data.shape)==(1,6,3,256,704),'Unexpected generated training shape')
def tensors(value):
 if isinstance(value,torch.Tensor):return [value]
 if isinstance(value,(list,tuple)):return [t for v in value for t in tensors(v)]
 raise TypeError('Unexpected target container '+str(type(value)))
for key in ['gt_bboxes','centers2d','depths']:
 for target_tensor in tensors(item[key].data):check(target_tensor.dtype==torch.float32,'Native 2D target must be float32: '+key)
for i in range(len(test)):test[i]
check(not torch.cuda.is_initialized(),'Preparation initialized CUDA')
report={'status':'clean_alignment_cpu_passed','gpu_used':False,'generated_frames_available':8,'accepted_generated_frames':len(accepted),'rejected_frames':rejected,'mapping':mapping,'image_files':frame_images,'video_sha256':digest(video_path),'generation_result_sha256':digest(S/'dd2_native_gpu_smoke/result.json'),'generated_train_sha256':digest(O/'generated_train.pkl'),'clean_val_sha256':digest(O/'clean_val.pkl'),'validation_source':'official mini_val; first two frames from each of two scenes','validation_frames':len(valid_infos),'validation_scene_names':[scenes[x] for x in sorted({i['scene_token'] for i in valid_infos})],'validation_tokens':[x['token'] for x in valid_infos],'train_scene_tokens':list({x['scene_token'] for x in accepted}),'native_train_batch_shape':list(item['img'].data.shape),'native_validation_pipeline_passed':True,'labels':'Inherited unmodified 3D scene annotations; 2D targets and K scaled by 448/1600,256/900. Metadata alignment does not establish generated object fidelity.','scope':'One clean generated training item, four real official mini_val frames; not a full benchmark.'}
(O/'alignment.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['status','accepted_generated_frames','validation_frames','validation_scene_names','native_train_batch_shape']}))
