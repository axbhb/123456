from common import *
import numpy as np,torch,cv2
from PIL import Image
from torch.nn import functional as F
from mmcv.parallel import collate,scatter
from mmcv.runner import load_checkpoint,wrap_fp16_model
from mmdet3d.models import build_model
sys.path.insert(0,str(R/'baseline_deployment_20261030/upstream/apgd'))
from autoattack.autopgd_base import APGDAttack
parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=N);a=parser.parse_args()
out=O/'asymmetry';out.mkdir(parents=True,exist_ok=True);start=time.time();state={'method':'Asymmetry-inspired-physical-patch-v1','status':'starting','steps':200,'optimizer':'Adam','learning_rate':.01,'patch_side_meters':1.0,'texture_size':[64,64],'gradient':'native forward; BPDA for uint8 quantization and PIL/cv2 resampling','temporal_context':'one immediately preceding Clean training frame, detached and restored for every candidate','completed':[]}
def save(**kw):state.update(kw);state['active_attempt_seconds']=time.time()-start;state['seconds']=state['active_attempt_seconds']+state.get('reused_frame_seconds',0);atomic(out/'attack_status.json',state)
save()
try:
 cfg,data=load_training();seed_all();torch.set_num_threads(1)
 model=build_model(cfg.model);load_checkpoint(model,str(cfg.load_from),map_location='cpu',strict=True)
 for m in model.modules():
  if hasattr(m,'with_cp'):m.with_cp=False
  if hasattr(m,'use_checkpoint'):m.use_checkpoint=False
 model.requires_grad_(False);wrap_fp16_model(model);model=model.cuda().train()
 memory_keys=['memory_embedding','memory_reference_point','memory_timestamp','memory_egopose','memory_velo']
 selected=json.loads((O/'selection.json').read_text())['selected_tokens'];matrix=np.asarray(json.loads((C/'dataset.json').read_text())['source_to_generated'])
 aug=next(t for t in data.pipeline.transforms if t.__class__.__name__=='ResizeCropFlipRotImage');orig_sample=aug._sample_augmentation;captured={}
 def capture():
  values=orig_sample();captured['augmentation']=values;return values
 aug._sample_augmentation=capture
 mean=torch.tensor(cfg.img_norm_cfg['mean'],device='cuda')[None,:,None,None];std=torch.tensor(cfg.img_norm_cfg['std'],device='cuda')[None,:,None,None]
 infos={x['token']:i for i,x in enumerate(data.data_infos)}
 geometry=json.loads((out/'geometry.json').read_text())
 if geometry['physically_editable_frames']!=N:raise ValueError('Not all selected frames have valid measured road projections')
 (out/'textures').mkdir(exist_ok=True)
 state['static_world_patch_policy']=geometry['static_world_patch_policy']
 state['geometry_source']=geometry['depth_source']
 class DetAPGD(APGDAttack):
  def dlr_loss(self,scores,labels):return scores[:,0]
 def batch_at(i,seed):
  seed_all(seed);return scatter(collate([data[i]],samples_per_gpu=1),[0])[0]
 for position,token in enumerate(selected[:a.limit]):
  folder=out/'images'/token;receipt=folder/'result.json'
  if receipt.exists():
   reused=json.loads(receipt.read_text());state['completed'].append(reused);state['reused_frame_seconds']=state.get('reused_frame_seconds',0)+reused['seconds'];continue
  t0=time.time();index=infos[token];info=data.data_infos[index];cams=list(info['cams']);native=np.stack([np.asarray(Image.open(info['cams'][cam]['data_path']).convert('RGB')) for cam in cams]);height,width=native.shape[1:3]
  x=torch.from_numpy(native.copy()).permute(0,3,1,2).float().cuda()/255;x=x.reshape(1,18,height,width)
  model.pts_bbox_head.reset_memory();model.test_flag=False
  if index>0 and data.data_infos[index-1]['token']==info.get('prev'):
   previous=batch_at(index-1,10000+index);previous['prev_exists'].zero_()
   with torch.no_grad(),torch.cuda.amp.autocast():model(return_loss=True,**previous)
   del previous
  memory={k:getattr(model.pts_bbox_head,k).detach().clone() if torch.is_tensor(getattr(model.pts_bbox_head,k)) else None for k in memory_keys}
  batch=batch_at(index,10000+index);resize,dims,crop,flip,rotate=captured['augmentation'];assert rotate==0
  if memory['memory_embedding'] is None:batch['prev_exists'].zero_()
  else:batch['prev_exists'].fill_(1)
  # Native forward follows exactly RestoreGeneratorCanvas then PIL augmentation.
  def native_preprocess(z):
   rgb=(z.detach().reshape(6,3,height,width).permute(0,2,3,1).cpu().numpy()*255).round().clip(0,255).astype(np.uint8);items=[]
   for image in rgb:
    restored=cv2.warpAffine(image[:,:,::-1].astype(np.float32),np.linalg.inv(matrix)[:2],(1600,900),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    image=Image.fromarray(restored.astype(np.uint8)).resize(dims).crop(crop)
    if flip:image=image.transpose(Image.FLIP_LEFT_RIGHT)
    image=image.rotate(rotate);items.append(np.asarray(image)[:,:,::-1].copy())
   v=torch.from_numpy(np.stack(items)).permute(0,3,1,2).float().cuda();return (v-mean)/std
  yy,xx=torch.meshgrid(torch.arange(900,device='cuda'),torch.arange(1600,device='cuda'),indexing='ij');gx=matrix[0,0]*xx+matrix[0,1]*yy+matrix[0,2];gy=matrix[1,0]*xx+matrix[1,1]*yy+matrix[1,2]
  grid=torch.stack((2*(gx+.5)/width-1,2*(gy+.5)/height-1),-1).float()[None].expand(6,-1,-1,-1)
  def preprocess(z):
   v=F.grid_sample(z.reshape(6,3,height,width),grid,align_corners=False,padding_mode='zeros')
   v=F.interpolate(v,size=(dims[1],dims[0]),mode='bicubic',align_corners=False,antialias=True)
   l,t,r,b=crop;v=F.pad(v,(0,max(0,r-v.shape[-1]),0,max(0,b-v.shape[-2])))[:,:,t:b,l:r]
   if flip:v=v.flip(-1)
   approx=(v*255-mean)/std;exact=native_preprocess(z)
   return approx+(exact-approx).detach()
  observed=native_preprocess(x);native_error=float((observed-batch['img'][0,0]).abs().max());assert native_error<1e-4,('native preprocess mismatch',native_error,tuple(batch['img'].shape))
  records=[]
  def oracle(z):
   seed_all(20000+index)
   for k,value in memory.items():setattr(model.pts_bbox_head,k,value.clone() if torch.is_tensor(value) else None)
   model.test_flag=False;current=dict(batch);current['img']=preprocess(z)[None,None]
   with torch.cuda.amp.autocast():losses=model(return_loss=True,**current)
   loss=sum(v.mean() if torch.is_tensor(v) else sum(t.mean() for t in v) for k,v in losses.items() if 'loss' in k)
   if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite detector loss')
   records.append(float(loss.detach()));save(status='optimizing',frame=position+1,token=token,objective_calls=len(records),last_loss=records[-1])
   return (loss*128).reshape(1,1)
  initial_loss=float(oracle(x.detach())[0,0].detach())/128
  projections=geometry['frames'][token]['visible_patches'];chosen=next((p for p in projections if p['patch_id']==token),projections[0]);patch_id=chosen['patch_id'];grids=[];masks=[]
  for camera in cams:
   path=out/'projections'/token/(patch_id+'_'+camera+'.npz')
   if path.exists() and any(v is not None and v['camera']==camera for v in chosen['views']):
    a=np.load(path);grids.append(torch.from_numpy(a['grid']));masks.append(torch.from_numpy(a['mask'])[None])
   else:grids.append(torch.zeros(height,width,2));masks.append(torch.zeros(1,height,width))
  patch_grid=torch.stack(grids).cuda();mask=torch.stack(masks).cuda();texture_path=out/'textures'/(patch_id+'.pt');texture=torch.nn.Parameter(torch.load(texture_path,map_location='cuda') if texture_path.exists() else torch.full((1,3,64,64),.5,device='cuda'));optimizer=torch.optim.Adam([texture],lr=.01);best_texture=texture.detach().clone();best_value=-float('inf')
  def compose_texture(value):
   painted=F.grid_sample(value.expand(6,-1,-1,-1),patch_grid,mode='bilinear',align_corners=True);return (x.reshape(6,3,height,width)*(1-mask)+painted*mask).reshape_as(x)
  for iteration in range(200):
   optimizer.zero_grad();score=oracle(compose_texture(texture)).sum()
   value=float(score.detach())
   if value>best_value:best_value=value;best_texture=texture.detach().clone()
   (-score).backward()
   if texture.grad is None or not torch.isfinite(texture.grad).all():raise FloatingPointError('invalid physical patch derivative')
   optimizer.step()
   with torch.no_grad():texture.clamp_(0,1)
  torch.save(best_texture.cpu(),texture_path);best=compose_texture(best_texture).detach();best_loss=torch.tensor([best_value],device='cuda')
  arrays=(best.detach().reshape(6,3,height,width).permute(0,2,3,1).cpu().numpy()*255).round().clip(0,255).astype(np.uint8);folder.mkdir(parents=True,exist_ok=True)
  changed=0;paths={}
  for camera,image,clean in zip(cams,arrays,native):
   diff=np.abs(image.astype(np.int16)-clean.astype(np.int16));assert diff.max()<=255
   changed+=int(np.any(diff));path=folder/(camera+'.png');Image.fromarray(image).save(path);paths[camera]=str(path)
  stored=torch.from_numpy(arrays.copy()).permute(0,3,1,2).float().cuda().reshape_as(x)/255
  with torch.no_grad():final_loss=float(oracle(stored)[0,0])/128
  record={'token':token,'paths':paths,'initial_loss':initial_loss,'stored_final_loss':final_loss,'best_continuous_loss':float(best_loss[0])/128,'loss_trace':records,'changed_cameras':changed,'max_uint8_delta':int(np.abs(arrays.astype(np.int16)-native.astype(np.int16)).max()),'native_preprocess_max_error':native_error,'seconds':time.time()-t0,'augmentation':{'resize':resize,'dims':dims,'crop':crop,'flip':bool(flip)},'previous_context_token':data.data_infos[index-1]['token'] if memory['memory_embedding'] is not None else None}
  atomic(receipt,record);state['completed'].append(record);save()
  del batch,memory,x,best,stored,texture,optimizer,patch_grid,mask,best_texture,score;model.pts_bbox_head.reset_memory();torch.cuda.empty_cache()
 # Reproject the final static textures into every selected frame, including cross-frame visibility.
 for record in state['completed']:
  token=record['token'];info=data.data_infos[infos[token]];changed=0;max_delta=0;visible_pixels=0
  for camera,meta in info['cams'].items():
   native=np.asarray(Image.open(meta['data_path']).convert('RGB')).copy();hh,ww=native.shape[:2];canvas=torch.from_numpy(native.copy()).permute(2,0,1).float()/255;support=torch.zeros(hh,ww,dtype=torch.bool)
   for patch in geometry['frames'][token]['visible_patches']:
    projection=out/'projections'/token/(patch['patch_id']+'_'+camera+'.npz');texture_path=out/'textures'/(patch['patch_id']+'.pt')
    if not projection.exists() or not texture_path.exists() or not any(v is not None and v['camera']==camera for v in patch['views']):continue
    a=np.load(projection);grid=torch.from_numpy(a['grid'])[None];mask=torch.from_numpy(a['mask'])[None];texture=torch.load(texture_path,map_location='cpu');paint=F.grid_sample(texture,grid,mode='bilinear',align_corners=True)[0];canvas=canvas*(1-mask)+paint*mask;support|=mask[0].bool()
   image=(canvas*255).round().clamp(0,255).permute(1,2,0).byte().numpy();diff=np.abs(image.astype(np.int16)-native.astype(np.int16));assert not np.any(diff[~support.numpy()]);changed+=int(np.any(diff));max_delta=max(max_delta,int(diff.max()));visible_pixels+=int(support.sum());Image.fromarray(image).save(record['paths'][camera])
  record.update(changed_cameras=changed,max_uint8_delta=max_delta,final_static_projection_pixels=visible_pixels,loss_scope='per-frame optimization before final cross-frame texture recomposition')
  atomic(out/'images'/token/'result.json',record)
 rows=pickle.load((C/'infos_train.pkl').open('rb'));replace={r['token']:r['paths'] for r in state['completed']}
 if len(replace)!=N:save(status='partial_probe_complete');sys.exit(0)
 for row in rows['infos']:
  if row['token'] in replace:
   for cam,path in replace[row['token']].items():row['cams'][cam]['data_path']=path
 pickle.dump(rows,(out/'infos_train.pkl').open('wb'));cfg.data.train.ann_file=str(out/'infos_train.pkl');cfg.work_dir=str(out/'train');cfg.runner.max_iters=ITERS;cfg.checkpoint_config.interval=ITERS;cfg.dump(str(out/'streampetr.py'))
 save(status='completed',frames=N,changed_cameras=sum(r['changed_cameras'] for r in state['completed']),peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
except BaseException as e:
 if isinstance(e,SystemExit):raise
 save(status='failed',error=str(e),traceback=traceback.format_exc());raise
