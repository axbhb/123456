"""FrozenDrive native autoregressive generation; reuse completed per-token outputs."""
import os,sys,json,time,traceback,fcntl,subprocess,argparse,shutil
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,default=283);args=ap.parse_args()
R=Path('/srv/table2');WD=R/'frozendrive_deployment_20261029/source/WorldDreamer';O=R/'outputs/table2_clean_mini_283_121/frozendrive';(O/'frames').mkdir(exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
state={'status':'waiting_for_existing_gpu_lock','started':time.time(),'attack':False,'completed_frames':0,'total_frames':283,'requested_end':args.end}
def save(**kw):
 state.update(kw);p=O/'generation_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
pairs=json.loads((O/'native_generation_pairs.json').read_text());assert len(pairs)==283 and 1<=args.end<=283
# Reuse the already verified first-frame pilot without resampling it.
first=pairs[0];p=O/'generation_pilot';old=json.loads((p/'result.json').read_text())
if old.get('status')=='passed' and old.get('sample_token')==first['token']:
 d=O/'frames'/first['token'];d.mkdir(exist_ok=True)
 if not (d/'result.json').exists():
  for im in p.glob('CAM_*.png'):shutil.copy2(str(im),str(d/im.name))
  old.update(reference_token=first['reference_token'],is_first_frame=True,reused_from=str(p));(d/'result.json').write_text(json.dumps(old,indent=2))
save()
try:
 with (R/'frameworks/gpu_budget_20261028.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
  gpu=max([x.split(',') for x in rows.strip().splitlines()],key=lambda x:int(x[1]))[0].strip();os.environ['CUDA_VISIBLE_DEVICES']=gpu
  os.chdir(str(WD));sys.path[:0]=[str(WD),str(WD/'third_party/bevfusion')]
  import torch,numpy as np
  from PIL import Image
  from omegaconf import OmegaConf
  from accelerate.utils import set_seed
  import dataset
  from dataset import collate_fn_singleframe
  from mmdet3d.datasets import build_dataset
  from projects.dreamer.utils.test_utils import build_pipe
  cfg=OmegaConf.load(str(O/'native_clean_config.yaml'));set_seed(cfg.seed)
  pipe=None;dtype=None;ds=None
  for pair in pairs[:args.end]:
   token=pair['token'];folder=O/'frames'/token;folder.mkdir(exist_ok=True);done=folder/'result.json'
   prev=json.loads(done.read_text()) if done.exists() else {}
   if prev.get('status')=='passed':
    assert prev['sample_token']==token
    for cam in cfg.dataset.view_order_nuscenes:
     with Image.open(folder/(cam+'.png')) as im:assert im.size==(800,448);im.verify()
    save(completed_frames=pair['index']+1);continue
   save(status='running',phase='native_getitem',gpu=gpu,index=pair['index'],sample_token=token,reference_token=pair['reference_token'],is_first_frame=pair['is_first_frame'])
   for cond_token in {token,pair['reference_token']}:
    for cam in cfg.dataset.view_order_nuscenes:assert (Path(cfg.dataset.dataset_root_occ)/('sample_'+cond_token)/('proj_'+cam+'.npz')).exists()
   if pipe is None:
    save(phase='loading_official_models');torch.linalg.solve(torch.eye(2,device='cuda'),torch.ones(2,device='cuda'));torch.cuda.empty_cache();pipe,dtype=build_pipe(cfg,'cuda');ds=build_dataset(OmegaConf.to_container(cfg.dataset.data.val.nuscenes,resolve=True))
   # Resume the native global RNG stream when the predecessor has a checkpoint.
   # The already generated first-frame pilot remains an immutable input artifact.
   if pair['index']>0:
    rng_path=O/'frames'/pairs[pair['index']-1]['token']/'rng_after.pt'
    if rng_path.exists():
     rng=torch.load(rng_path,map_location='cpu');torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
   torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},folder/'rng_before.pt')
   example=ds[pair['index']]
   batch=collate_fn_singleframe([example],template=cfg.dataset.template,tokenizer=pipe.tokenizer,is_train=False,bbox_mode=cfg.model.bbox_mode,bbox_view_shared=cfg.model.bbox_view_shared,bbox_drop_ratio=cfg.runner.bbox_drop_ratio,bbox_add_ratio=cfg.runner.bbox_add_ratio,bbox_add_num=cfg.runner.bbox_add_num)
   assert example['metas'].data['token']==token
   if not pair['is_first_frame']:
    ref=O/'frames'/pair['reference_token'];rr=json.loads((ref/'result.json').read_text());assert rr['status']=='passed' and rr['sample_token']==pair['reference_token']
    from torchvision.transforms.functional import to_tensor,normalize
    refs=[normalize(to_tensor(Image.open(ref/(cam+'.png')).convert('RGB')),[.5,.5,.5],[.5,.5,.5]) for cam in cfg.dataset.view_order_nuscenes]
    batch['ref_images'][0]=torch.stack(refs).contiguous().float()
   rel=batch['relative_pose'].to(dtype)
   kwargs=dict(prompt=batch['captions'],bev_hdmap=batch['bev_hdmap'],camera_param=batch['camera_param'].to(dtype),rel_pose=rel,ref_rel_pose=torch.eye(rel.size(-1),dtype=rel.dtype,device=rel.device).expand_as(rel),ref_images=batch['ref_images'].to(dtype),layout_canvas=batch['layout_canvas'].to(dtype),occupancy=batch['occupancy'].to(dtype),ref_layout_canvas=batch['ref_layout_canvas'].to(dtype),ref_occupancy=batch['ref_occupancy'].to(dtype),height=cfg.dataset.image_size[0],width=cfg.dataset.image_size[1],generator=torch.Generator(device=pipe.device).manual_seed(cfg.seed),bev_controlnet_kwargs=batch['kwargs'],is_first_frame=pair['is_first_frame'],use_first_frame=cfg.get('use_first_frame',False),**OmegaConf.to_container(cfg.runner.pipeline_param,resolve=True))
   def progress(step,t,latents):
    if not torch.isfinite(latents).all():raise RuntimeError('Nonfinite native sampled latents')
    save(phase='native_sampling',completed_steps=step+1,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
   save(phase='native_sampling',completed_steps=0);start=time.time()
   with torch.inference_mode():images=pipe(**kwargs,callback=progress).images[0]
   torch.cuda.synchronize();elapsed=time.time()-start;assert len(images)==6
   stats=[]
   for cam,img in zip(cfg.dataset.view_order_nuscenes,images):
    assert img.size==(800,448);img.save(folder/(cam+'.png'));a=np.asarray(img);stats.append({'camera':cam,'mean':float(a.mean()),'std':float(a.std()),'min':int(a.min()),'max':int(a.max())})
   torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},folder/'rng_after.pt')
   result={'status':'passed','sample_token':token,'reference_token':pair['reference_token'],'is_first_frame':pair['is_first_frame'],'reference_source':'native first-frame input' if pair['is_first_frame'] else 'previous generated six-camera PNGs','seed':cfg.seed,'sampling_seconds':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'images':stats,'finished':time.time(),'image_size':list(cfg.dataset.image_size),'pipeline_params':OmegaConf.to_container(cfg.runner.pipeline_param,resolve=True)}
   tmp=done.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(done);save(completed_frames=pair['index']+1)
  save(status='generation_complete' if args.end==283 else 'partial_complete',phase='finished',finished=time.time())
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
