from common import *
import numpy as np,cv2
from pyquaternion import Quaternion
from PIL import Image
rows=pickle.load((C/'infos_train.pkl').open('rb'))['infos'];selected=json.loads((O/'selection.json').read_text())['selected_tokens'];by_token={r['token']:r for r in rows};base=Path(os.environ.get('TABLE2_FROZEN_REFERENCE_CLEAN',str(R/'outputs/table2_clean_mini_283_121/frozendrive')));matrix=np.array(json.loads((C/'dataset.json').read_text())['source_to_generated']);patches=[];started=time.time();anchor_path=O/'asymmetry/anchor_overrides.json';anchor_overrides=json.loads(anchor_path.read_text()) if anchor_path.exists() else {}
def pose(rot,trans):
 p=np.eye(4);p[:3,:3]=Quaternion(rot).rotation_matrix;p[:3,3]=trans;return p
for token in selected:
 row=by_token[token];folder=Path(os.environ.get('TABLE2_ROAD_CACHE',str(base/'conditions/frames')))/token
 if not (folder/'prepared.npz').exists():folder=base/'condition_pilot'
 z=np.load(folder/'prepared.npz');points=z['semantic_points'];road=points[points[:,3]==11,:3]
 transform=pose(row['lidar2ego_rotation'],row['lidar2ego_translation']);road=road@transform[:3,:3].T+transform[:3,3];near=road[(np.abs(road[:,0])>4)&(np.abs(road[:,0])<20)&(np.abs(road[:,1])<8)]
 if len(near)<30:raise ValueError('insufficient measured road plane '+token)
 target=np.array(anchor_overrides.get(token,[8.,2.]));ego_world=pose(row['ego2global_rotation'],row['ego2global_translation']);world_candidates=near@ego_world[:3,:3].T+ego_world[:3,3];eligible=np.zeros(len(near),dtype=bool)
 for camera,cam in row['cams'].items():
  camera_world=pose(cam['ego2global_rotation'],cam['ego2global_translation'])@pose(cam['sensor2ego_rotation'],cam['sensor2ego_translation']);inverse=np.linalg.inv(camera_world);q=world_candidates@inverse[:3,:3].T+inverse[:3,3];uv=q@np.array(cam['cam_intrinsic']).T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8);fx=np.rint(.5*uv[:,0]).astype(int);fy=np.rint(.5*uv[:,1]-2).astype(int);valid=(q[:,2]>.1)&(fx>=0)&(fx<800)&(fy>=0)&(fy<448);indices=np.where(valid)[0]
  z=np.load(base/'conditions/projected'/('sample_'+token)/('proj_'+camera+'.npz'));dist=z['distances'][fy[indices],fx[indices]];labels=z['labels'][fy[indices],fx[indices]];expected=np.linalg.norm(q[indices],axis=1);eligible[indices]|=(dist>0)&(expected<=dist+.75)&np.isin(labels,[11,12])
 if not eligible.any():raise ValueError('no reconstructed visible road anchor '+token)
 candidates=near[eligible];chosen_anchor=candidates[np.argmin(np.linalg.norm(candidates[:,:2]-target,axis=1))];nearest=chosen_anchor[:2];local=near[np.linalg.norm(near[:,:2]-nearest,axis=1)<1.5]
 A=np.column_stack([local[:,:2],np.ones(len(local))]);coeff=np.linalg.lstsq(A,local[:,2],rcond=None)[0];residual=np.abs(A@coeff-local[:,2]);keep=residual<=np.quantile(residual,.8);coeff=np.linalg.lstsq(A[keep],local[keep,2],rcond=None)[0]
 center=chosen_anchor.copy();u=np.array([1.,0.,coeff[0]]);u/=np.linalg.norm(u);v=np.cross(np.array([-coeff[0],-coeff[1],1.]),u);v/=np.linalg.norm(v);corners=np.stack([center-.5*u-.5*v,center+.5*u-.5*v,center+.5*u+.5*v,center-.5*u+.5*v]);ego_world=pose(row['ego2global_rotation'],row['ego2global_translation']);world=corners@ego_world[:3,:3].T+ego_world[:3,3]
 patches.append({'id':token,'scene_token':row['scene_token'],'world_corners':world.tolist(),'surface':'measured drivable semantic point cloud least-squares plane','fit_points':len(local),'trimmed_plane_rmse_m':float(np.sqrt(np.mean((A[keep]@coeff-local[keep,2])**2))),'side_meters':1.0})
report={'patches':patches,'frames':{},'static_world_patch_policy':'one 1m x 1m world patch anchored per selected keyframe, projected consistently into all selected views/frames of that scene; nonselected training images stay Clean','depth_source':'FrozenDrive measured/reconstructed training occupancy, not test data','seconds':0}
for token in selected:
 row=by_token[token];cams=list(row['cams']);image=Image.open(next(iter(row['cams'].values()))['data_path']);width,height=image.size;visible=[]
 for patch in patches:
  if patch['scene_token']!=row['scene_token']:continue
  per_view=[]
  for camera in cams:
   cam=row['cams'][camera];camera_world=pose(cam['ego2global_rotation'],cam['ego2global_translation'])@pose(cam['sensor2ego_rotation'],cam['sensor2ego_translation']);world_camera=np.linalg.inv(camera_world);world=np.array(patch['world_corners']);q=world@world_camera[:3,:3].T+world_camera[:3,3]
   if q[:,2].min()<=.1:per_view.append(None);continue
   pixels=q@np.array(cam['cam_intrinsic']).T;pixels=pixels[:,:2]/pixels[:,2:];pixels=np.column_stack([pixels,np.ones(4)])@matrix.T;pixels=pixels[:,:2]
   H=cv2.getPerspectiveTransform(np.array([[0,0],[63,0],[63,63],[0,63]],dtype=np.float32),pixels.astype(np.float32));inv=np.linalg.inv(H);yy,xx=np.mgrid[:height,:width];uv=np.stack([xx,yy,np.ones_like(xx)],-1)@inv.T;uv=uv[...,:2]/uv[...,2:];mask=(uv[...,0]>=0)&(uv[...,0]<=63)&(uv[...,1]>=0)&(uv[...,1]<=63)
   if not np.any(mask):per_view.append(None);continue
   fraction=uv/63;world_pixels=world[0]+fraction[...,0,None]*(world[1]-world[0])+fraction[...,1,None]*(world[3]-world[0]);expected=np.linalg.norm(world_pixels-camera_world[:3,3],axis=-1)
   projected=base/'conditions/projected'/('sample_'+token)/('proj_'+camera+'.npz');z=np.load(projected);mapx=((xx-matrix[0,2])/matrix[0,0]*.5).astype(np.float32);mapy=((yy-matrix[1,2])/matrix[1,1]*.5-2).astype(np.float32);dist=cv2.remap(z['distances'],mapx,mapy,cv2.INTER_NEAREST);labels=cv2.remap(z['labels'].astype(np.float32),mapx,mapy,cv2.INTER_NEAREST).astype(int)
   # The occupancy surface establishes both visibility and a road-only support.
   mask &= (dist>0)&np.isfinite(dist)&(expected<=dist+.75)&np.isin(labels,[11,12])
   if np.count_nonzero(mask)<2:per_view.append(None);continue
   per_view.append({'camera':camera,'homography':H.tolist(),'pixels':int(mask.sum())});p=O/'asymmetry/projections'/token;p.mkdir(parents=True,exist_ok=True);np.savez_compressed(p/(patch['id']+'_'+camera+'.npz'),grid=(2*uv/63-1).astype(np.float32),mask=mask.astype(np.float32))
  if any(x is not None for x in per_view):visible.append({'patch_id':patch['id'],'views':per_view})
 report['frames'][token]={'visible_patches':visible,'visible_pixels':sum(v['pixels'] for item in visible for v in item['views'] if v)}
report['seconds']=time.time()-started;report['physically_editable_frames']=sum(v['visible_pixels']>0 for v in report['frames'].values());atomic(O/'asymmetry/geometry.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['frames','patches']}))
