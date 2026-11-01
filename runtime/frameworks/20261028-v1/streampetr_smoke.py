"""Strict pretrained StreamPETR CPU preflight or one-update/two-frame GPU smoke."""
import argparse,os,sys,time,json,hashlib,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['cpu-preflight','gpu-smoke'],default='cpu-preflight');a=p.parse_args();a.runtime=a.runtime.resolve();a.output=a.output.resolve()
if a.mode=='cpu-preflight':os.environ['CUDA_VISIBLE_DEVICES']='-1'
elif os.environ.get('PHYSCOND_GPU_GUARD')!='1':raise RuntimeError('Launch GPU smoke through run_gpu_smoke.py')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
a.output.mkdir(parents=True,exist_ok=True);started=time.time();r={'status':'started','mode':a.mode,'gpu_used':a.mode=='gpu-smoke','model_forward_passed':False,'training_passed':False,'AMOTA':None,'benchmark_metrics_computed':False}
def save():
 tmp=a.output/'result.tmp';tmp.write_text(json.dumps(r,indent=2));tmp.replace(a.output/'result.json')
save()
try:
 import torch,numpy as np
 torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
 import random;random.seed(0)
 r['torch']=torch.__version__
 sys.path.insert(0,str(a.runtime.resolve()));os.chdir(a.runtime)
 from mmcv import Config
 import projects.mmdet3d_plugin
 from mmdet3d.models import build_model
 from mmdet3d.datasets import build_dataset
 from mmcv.parallel import collate,MMDataParallel
 from mmcv.runner import build_optimizer,wrap_fp16_model
 cfg=Config.fromfile(str(a.runtime.resolve()/'smoke_v100.py'))
 checkpoint=Path(cfg.load_from)
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451':raise ValueError('Unexpected checkpoint hash')
 t=time.time();model=build_model(cfg.model)
 try:weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
 except TypeError:weights=torch.load(checkpoint,map_location='cpu')
 state=weights['state_dict'];model.load_state_dict(state,strict=True);del weights,state
 r.update(strict_checkpoint_load=True,model_load_seconds=time.time()-t,parameters=sum(x.numel() for x in model.parameters()));save()
 train=build_dataset(cfg.data.train);item=train[0]
 r['train_frames']=len(train);r['native_train_batch_shape']=list(item['img'].data.shape);r['native_train_pipeline_passed']=True
 testcfg=cfg.data.test.copy();testcfg['test_mode']=True;test=build_dataset(testcfg);test_item=test[0];r['native_test_pipeline_passed']=True
 if a.mode=='cpu-preflight':
  assert not torch.cuda.is_initialized();r.update(status='cpu_preflight_passed',cuda_initialized=False)
 else:
  assert torch.cuda.device_count()==1
  wrap_fp16_model(model)  # Enable native force_fp32 loss decorators, as in MMCV AMP hook.
  model=MMDataParallel(model.cuda(),device_ids=[0]);model.train();optimizer=build_optimizer(model.module,cfg.optimizer)
  scaler=torch.cuda.amp.GradScaler();batch=collate([item],samples_per_gpu=1)
  before=next(x for x in model.parameters() if x.requires_grad).detach().clone()
  cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();numpy_rng=np.random.get_state();python_rng=random.getstate()
  buffers={n:b.detach().clone() for n,b in model.module.named_buffers()}
  torch.cuda.synchronize();t=time.time();r['amp_attempts']=[]
  for attempt in range(16):
   optimizer.zero_grad(set_to_none=True);model.module.pts_bbox_head.reset_memory()
   torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng);np.random.set_state(numpy_rng);random.setstate(python_rng)
   with torch.no_grad():
    for n,b in model.module.named_buffers():b.copy_(buffers[n])
   attempt_started=time.time();scale=float(scaler.get_scale())
   with torch.cuda.amp.autocast():
    losses=model(return_loss=True,**batch);loss,logs=model.module._parse_losses(losses)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native training loss')
   r['model_forward_passed']=True
   scaler.scale(loss).backward();scaler.unscale_(optimizer)
   grads=[x.grad for x in model.parameters() if x.grad is not None]
   if not grads:raise RuntimeError('Missing model gradients')
   bad=[n for n,x in model.named_parameters() if x.grad is not None and not bool(torch.isfinite(x.grad).all())]
   r['amp_attempts'].append({'scale':scale,'nonfinite_tensors':len(bad),'nonfinite_examples':bad[:3],'loss':float(loss.detach().cpu())});save()
   if bad:
    scaler.step(optimizer);scaler.update()  # Native GradScaler skips this overflowed update.
    del loss,losses,grads
    continue
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),35)
   if not torch.isfinite(norm):raise FloatingPointError('Nonfinite aggregate gradient norm')
   scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
   change=(next(x for x in model.parameters() if x.requires_grad).detach()-before).abs().max().item()
   if change==0:raise RuntimeError('No measured optimizer update')
   r.update(training_passed=True,successful_optimizer_updates=1,first_update_seconds=time.time()-t,successful_update_seconds=time.time()-attempt_started,loss=float(loss.detach().cpu()),gradient_norm=float(norm.cpu()),parameter_max_change=change,optimizer_precision='torch AMP with native MMCV fp16 flags; dynamic loss scale retries, one successful update; not final timing benchmark');save()
   break
  else:raise FloatingPointError('Gradients remained nonfinite after bounded dynamic loss-scale recovery')
  del buffers
  del batch,item,before,loss,losses,grads;optimizer.zero_grad(set_to_none=True);model.eval();predictions=[];times=[]
  for i in range(2):
   b=collate([test[i]],samples_per_gpu=1);torch.cuda.synchronize();t=time.time()
   with torch.no_grad(),torch.cuda.amp.autocast():output=model(return_loss=False,rescale=True,**b)
   torch.cuda.synchronize();times.append(time.time()-t);predictions.extend(output)
  path=test._format_bbox([x['pts_bbox'] for x in predictions],str(a.output/'detections'))
  detections=json.loads(Path(path).read_text())['results'];sys.path.insert(0,str(a.runtime/'nusc_tracking'));from pub_tracker import PubTracker
  tracker=PubTracker(hungarian=False,max_age=3);records=[];previous=None
  for info in test.data_infos[:2]:
   timestamp=info['timestamp']/1e6;dt=0 if previous is None else timestamp-previous
   tracks=tracker.step_centertrack(detections[info['token']],dt,.2);records.append({'sample_token':info['token'],'tracks':len(tracks),'ids':[x['tracking_id'] for x in tracks]});previous=timestamp
  r.update(status='gpu_smoke_passed',model_forward_passed=True,inference_frames=2,inference_seconds=times,tracking_frames=records,tracking_evaluation=False,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
except Exception as e:r.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()[-6000:])
r['total_seconds']=time.time()-started;save();print(json.dumps(r))
if r['status']=='failed':sys.exit(2)
