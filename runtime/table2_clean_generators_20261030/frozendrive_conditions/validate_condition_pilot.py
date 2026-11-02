import os,json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
import project_uniscene_mini as p
sample=p.nusc.get('sample',os.environ['FROZEN_SAMPLE_TOKEN']);lid=p.nusc.get('sample_data',sample['data']['LIDAR_TOP'])
def matrix(trans,rot):
 t=np.eye(4);t[:3,:3]=Quaternion(rot).rotation_matrix;t[:3,3]=trans;return t
def sensor_global(sd):
 c=p.nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);e=p.nusc.get('ego_pose',sd['ego_pose_token']);return matrix(e['translation'],e['rotation'])@matrix(c['translation'],c['rotation'])
lc=p.nusc.get('calibrated_sensor',lid['calibrated_sensor_token']);le=p.nusc.get('ego_pose',lid['ego_pose_token']);details=[]
occ=np.load(Path(p.OCCUPANCY_DIR)/sample['token']/(lid['token']+'.npy'),allow_pickle=True);assert occ.ndim==2 and occ.shape[1]==4 and len(occ)>0;assert np.all((occ[:,:3]>=0)&(occ[:,:3]<[800,800,64]));assert np.isin(occ[:,3],np.arange(17)).all()
for cam in p.cameras:
 sd=p.nusc.get('sample_data',sample['data'][cam]);a=p.obtain_sensor2top(p.nusc,sd['token'],(lc['translation'],),Quaternion(lc['rotation']).rotation_matrix,le['translation'],Quaternion(le['rotation']).rotation_matrix,cam);t=matrix(a['sensor2lidar_translation'],[1,0,0,0]);t[:3,:3]=a['sensor2lidar_rotation'];ref=np.linalg.inv(sensor_global(lid))@sensor_global(sd);assert np.allclose(t,ref,atol=1e-7,rtol=1e-7)
 cs=p.nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);intrinsic=np.asarray(cs['camera_intrinsic']);expected=np.array([[.5,0.,0.],[0.,.5,-2.],[0.,0.,1.]])@intrinsic;assert np.allclose(p.projection_intrinsics(intrinsic,[448,800]),expected)
 f=Path(p.OUTPUT_IMG_DIR)/('sample_'+sample['token'])/('proj_'+cam+'.npz');z=np.load(f);lab=z['labels'];dist=z['distances'];assert lab.shape==(448,800) and dist.shape==(448,800);assert np.isin(lab,np.arange(-1,17)).all();hit=lab>=0;assert hit.any();assert np.isfinite(dist[hit]).all() and (dist[hit]>=max(0.,0.5-np.sqrt(3)*0.125)-1e-5).all();assert not np.isnan(dist).any();details.append({'camera':cam,'valid_fraction':float(hit.mean()),'min_surface_range_m':float(dist[hit].min()),'native_filter':'voxel corner distance > 0.5m; entry surface may be up to one voxel diagonal closer','labels':np.unique(lab).tolist(),'transform_max_error':float(np.abs(t-ref).max())})
r={'status':'passed','sample_token':sample['token'],'occupancy_voxels':len(occ),'occupancy_shape':[800,800,64],'voxel_size':0.125,'point_cloud_range':[-50,-50,-5,50,50,3],'cameras':details,'temporal_scope':'one real 2Hz training keyframe; not a 12Hz video condition sequence'};out=Path(p.OUTPUT_IMG_DIR).parent/'condition_validation.json';out.write_text(json.dumps(r,indent=2));print(json.dumps(r))

# Run the actual model-side pipeline in its own environment after real maps exist.
import subprocess
R=Path('/srv/table2')
if os.environ.get('FROZEN_SKIP_NATIVE_BATCH')!='1':subprocess.run([str(R/'envs/frozendrive/bin/python'),str(Path(__file__).with_name('validate_native_batch.py'))],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'))
