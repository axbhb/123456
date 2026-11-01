"""One native FrozenDrive frame on real reconstructed training conditions."""
import os,sys,json,time,traceback,fcntl,subprocess
from pathlib import Path
R=Path('/srv/table2');WD=R/'frozendrive_deployment_20261029/source/WorldDreamer';O=R/'outputs/table2_clean_mini_283_121/frozendrive';P=O/'generation_pilot';P.mkdir(exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
state={'status':'waiting_for_existing_gpu_lock','started':time.time(),'attack':False}
def save(**kw):
 state.update(kw);p=P/'result.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
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
  pairs=json.loads((O/'native_generation_pairs.json').read_text());pair=pairs[0];token=pair['token'];assert pair['is_first_frame']
  for cam in cfg.dataset.view_order_nuscenes:assert (Path(cfg.dataset.dataset_root_occ)/('sample_'+token)/('proj_'+cam+'.npz')).exists()
  save(status='running',phase='loading_official_models',gpu=gpu,sample_token=token,seed=cfg.seed,image_size=list(cfg.dataset.image_size),pipeline_params=OmegaConf.to_container(cfg.runner.pipeline_param,resolve=True))
  torch.linalg.solve(torch.eye(2,device='cuda'),torch.ones(2,device='cuda'));torch.cuda.empty_cache()
  pipe,dtype=build_pipe(cfg,'cuda')
  save(phase='native_getitem')
  ds=build_dataset(OmegaConf.to_container(cfg.dataset.data.val.nuscenes,resolve=True));example=ds[0]
  batch=collate_fn_singleframe([example],template=cfg.dataset.template,tokenizer=pipe.tokenizer,is_train=False,bbox_mode=cfg.model.bbox_mode,bbox_view_shared=cfg.model.bbox_view_shared,bbox_drop_ratio=cfg.runner.bbox_drop_ratio,bbox_add_ratio=cfg.runner.bbox_add_ratio,bbox_add_num=cfg.runner.bbox_add_num)
  assert example['metas'].data['token']==token
  rel=batch['relative_pose'].to(dtype)
  kwargs=dict(prompt=batch['captions'],bev_hdmap=batch['bev_hdmap'],camera_param=batch['camera_param'].to(dtype),rel_pose=rel,ref_rel_pose=torch.eye(rel.size(-1),dtype=rel.dtype,device=rel.device).expand_as(rel),ref_images=batch['ref_images'].to(dtype),layout_canvas=batch['layout_canvas'].to(dtype),occupancy=batch['occupancy'].to(dtype),ref_layout_canvas=batch['ref_layout_canvas'].to(dtype),ref_occupancy=batch['ref_occupancy'].to(dtype),height=cfg.dataset.image_size[0],width=cfg.dataset.image_size[1],generator=torch.Generator(device=pipe.device).manual_seed(cfg.seed),bev_controlnet_kwargs=batch['kwargs'],is_first_frame=True,use_first_frame=cfg.get('use_first_frame',False),**OmegaConf.to_container(cfg.runner.pipeline_param,resolve=True))
  def progress(step,t,latents):
   if not torch.isfinite(latents).all():raise RuntimeError('Nonfinite native sampled latents')
   save(phase='native_sampling',completed_steps=step+1,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
  save(phase='native_sampling',completed_steps=0);start=time.time()
  with torch.inference_mode():images=pipe(**kwargs,callback=progress).images[0]
  torch.cuda.synchronize();elapsed=time.time()-start;assert len(images)==6
  stats=[]
  for cam,img in zip(cfg.dataset.view_order_nuscenes,images):
   assert img.size==(800,448);img.save(P/(cam+'.png'));a=np.asarray(img);stats.append({'camera':cam,'mean':float(a.mean()),'std':float(a.std()),'min':int(a.min()),'max':int(a.max())})
  save(status='passed',phase='six_camera_frame_generated',sampling_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,images=stats,finished=time.time(),scope='one first training frame; full autoregressive sequence and downstream fine-tuning pending')
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
