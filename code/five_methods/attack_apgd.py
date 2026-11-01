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
out=O/'apgd';out.mkdir(parents=True,exist_ok=True);start=time.time();state={'method':'APGD-DetLoss-v1','status':'starting','steps':20,'restarts':1,'epsilon':8/255,'gradient':'native forward; BPDA for uint8 quantization and PIL/cv2 resampling','temporal_context':'one immediately preceding Clean training frame, detached and restored for every candidate','completed':[]}
def save(**kw):state.update(kw);state['seconds']=time.time()-start;atomic(out/'attack_status.json',state)
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
 class DetAPGD(APGDAttack):
  def dlr_loss(self,scores,labels):return scores[:,0]
 def batch_at(i,seed):
  seed_all(seed);return scatter(collate([data[i]],samples_per_gpu=1),[0])[0]
 for position,token in enumerate(selected[:a.limit]):
  folder=out/'images'/token;receipt=folder/'result.json'
  if receipt.exists():state['completed'].append(json.loads(receipt.read_text()));continue
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
  initial=oracle(x.requires_grad_(True));g=torch.autograd.grad(initial.sum(),x)[0];assert torch.isfinite(g).all() and g.abs().sum()>0
  x=x.detach();initial_loss=records[-1]
  attack=DetAPGD(oracle,n_iter=20,n_restarts=1,eps=8/255,norm='Linf',loss='dlr',seed=0,device=x.device);attack.init_hyperparam(x)
  seed_all(index);best,_,best_loss,_=attack.attack_single_run(x,torch.zeros(1,dtype=torch.long,device='cuda'),x_init=x)
  if float(best_loss[0])/128<initial_loss:best=x
  arrays=(best.detach().reshape(6,3,height,width).permute(0,2,3,1).cpu().numpy()*255).round().clip(0,255).astype(np.uint8);folder.mkdir(parents=True,exist_ok=True)
  changed=0;paths={}
  for camera,image,clean in zip(cams,arrays,native):
   diff=np.abs(image.astype(np.int16)-clean.astype(np.int16));assert diff.max()<=8
   changed+=int(np.any(diff));path=folder/(camera+'.png');Image.fromarray(image).save(path);paths[camera]=str(path)
  stored=torch.from_numpy(arrays.copy()).permute(0,3,1,2).float().cuda().reshape_as(x)/255
  with torch.no_grad():final_loss=float(oracle(stored)[0,0])/128
  record={'token':token,'paths':paths,'initial_loss':initial_loss,'stored_final_loss':final_loss,'best_continuous_loss':float(best_loss[0])/128,'loss_trace':records,'changed_cameras':changed,'max_uint8_delta':int(np.abs(arrays.astype(np.int16)-native.astype(np.int16)).max()),'native_preprocess_max_error':native_error,'seconds':time.time()-t0,'augmentation':{'resize':resize,'dims':dims,'crop':crop,'flip':bool(flip)},'previous_context_token':data.data_infos[index-1]['token'] if memory['memory_embedding'] is not None else None}
  atomic(receipt,record);state['completed'].append(record);save()
  del batch,memory,x,g,best,initial,stored;model.pts_bbox_head.reset_memory();torch.cuda.empty_cache()
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
