"""One clean DD2-derived optimizer update with paired official mini_val subset evaluation."""
import os,sys,json,time,hashlib,traceback,random,argparse,copy
from collections import OrderedDict
from pathlib import Path
if os.environ.get('PHYSCOND_GPU_GUARD')!='1':raise RuntimeError('Use the shared GPU guard')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();W=a.work.resolve();S=W.parent;runtime=S/'runtime_sources/streampetr';start=time.time()
r={'status':'started','experiment':'clean_generated_data_finetune_smoke','attack_executed':False,'successful_optimizer_updates':0,'full_benchmark':False}
def save():
 temp=W/'result.tmp';temp.write_text(json.dumps(r,indent=2,default=lambda x:x.item()));temp.replace(W/'result.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
save()
try:
 import numpy as np,torch
 torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0)
 if torch.cuda.device_count()!=1:raise RuntimeError('Single visible GPU required')
 sys.path.insert(0,str(runtime));os.chdir(runtime)
 from mmcv import Config
 import projects.mmdet3d_plugin
 from mmdet3d.datasets import build_dataset
 from mmdet3d.models import build_model
 from mmcv.parallel import collate,MMDataParallel
 from mmcv.runner import build_optimizer,wrap_fp16_model
 cfg=Config.fromfile(str(W/'clean_config.py'));alignment=json.load(open(W/'alignment.json'))
 assert alignment['status']=='clean_alignment_cpu_passed'
 assert sha(W/'generated_train.pkl')==alignment['generated_train_sha256']
 assert sha(W/'clean_val.pkl')==alignment['clean_val_sha256']
 for record in alignment['image_files']:assert sha(record['path'])==record['sha256']
 ckpt=Path(cfg.load_from);expected='e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451';assert sha(ckpt)==expected
 model=build_model(cfg.model);state=torch.load(ckpt,map_location='cpu');model.load_state_dict(state['state_dict'],strict=True);del state
 wrap_fp16_model(model);model=MMDataParallel(model.cuda(),device_ids=[0]);train=build_dataset(cfg.data.train);test=build_dataset(cfg.data.test)
 assert {x['scene_token'] for x in train.data_infos}.isdisjoint({x['scene_token'] for x in test.data_infos})
 assert len(train)==alignment['accepted_generated_frames'] and len(test)==4
 r.update(strict_checkpoint_load=True,checkpoint_sha256=expected,train_frames=len(train),val_frames=len(test),validation_protocol='Official mini_val first two keyframes per scene (four frames); native devkit metrics, not full mini_val',alignment_sha256=sha(W/'alignment.json'))
 from nuscenes import NuScenes
 from nuscenes.eval.common.data_classes import EvalBoxes
 from nuscenes.eval.common.loaders import load_gt,load_prediction,add_center_dist,filter_eval_boxes
 from nuscenes.eval.detection.data_classes import DetectionBox
 from nuscenes.eval.detection.config import config_factory
 from nuscenes.eval.detection.evaluate import DetectionEval
 nusc=NuScenes(version='v1.0-mini',dataroot=str(cfg.data.test.data_root),verbose=False);dcfg=config_factory('detection_cvpr_2019')
 all_gt=load_gt(nusc,'mini_val',DetectionBox,verbose=False);selected=[x['token'] for x in test.data_infos];gt=EvalBoxes()
 for token in selected:gt.add_boxes(token,all_gt[token])
 gt=filter_eval_boxes(nusc,add_center_dist(nusc,gt),dcfg.class_range,verbose=False)
 r['gt_counts']={c:sum(b.detection_name==c for b in gt.all) for c in dcfg.class_names}
 def evaluate(phase):
  model.eval();model.module.pts_bbox_head.reset_memory();model.module.prev_scene_token=None;preds=[];times=[]
  with torch.no_grad():
   for i in range(len(test)):
    batch=collate([test[i]],samples_per_gpu=1);torch.cuda.synchronize();t=time.time()
    with torch.cuda.amp.autocast():out=model(return_loss=False,rescale=True,**batch)
    torch.cuda.synchronize();times.append(time.time()-t);preds.extend(out)
  path=test._format_bbox([x['pts_bbox'] for x in preds],str(W/phase/'detections'))
  pred,_=load_prediction(path,dcfg.max_boxes_per_sample,DetectionBox,verbose=False)
  assert set(pred.sample_tokens)==set(selected)==set(gt.sample_tokens)
  pred=filter_eval_boxes(nusc,add_center_dist(nusc,pred),dcfg.class_range,verbose=False)
  # Use the unchanged official evaluator on explicitly restricted GT/prediction sets.
  evaluator=DetectionEval.__new__(DetectionEval);evaluator.cfg=dcfg;evaluator.verbose=False;evaluator.gt_boxes=gt;evaluator.pred_boxes=pred
  metrics,md=evaluator.evaluate();full=metrics.serialize()
  (W/phase/'metrics_summary.json').write_text(json.dumps(full,indent=2))
  (W/phase/'metrics_details.json').write_text(json.dumps(md.serialize()))
  return {'mAP':full['mean_ap'],'NDS':full['nd_score'],'tp_errors':full['tp_errors'],'class_AP':full['mean_dist_aps'],'frames':len(test),'inference_seconds':times,'prediction_path':path,'gt_boxes_after_filter':len(gt.all),'predicted_boxes_after_filter':len(pred.all)}
 r['before']=evaluate('before');r['status']='baseline_eval_passed';save()
 model.module.pts_bbox_head.reset_memory();model.module.prev_scene_token=None;model.train()
 optimizer=build_optimizer(model.module,cfg.optimizer);scaler=torch.cuda.amp.GradScaler(init_scale=2048.)
 item=train[0];batch=collate([item],samples_per_gpu=1);r['train_batch_shape']=list(item['img'].data.shape)
 parameter=next(x for x in model.parameters() if x.requires_grad);before=parameter.detach().clone()
 cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();np_rng=np.random.get_state();py_rng=random.getstate();buffers={n:b.detach().clone() for n,b in model.module.named_buffers()}
 torch.cuda.synchronize();t0=time.time();r['amp_attempts']=[]
 for attempt in range(16):
  optimizer.zero_grad(set_to_none=True);model.module.pts_bbox_head.reset_memory();model.module.prev_scene_token=None
  torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng);np.random.set_state(np_rng);random.setstate(py_rng)
  with torch.no_grad():
   for n,b in model.module.named_buffers():b.copy_(buffers[n])
  t=time.time();scale=float(scaler.get_scale())
  with torch.cuda.amp.autocast():losses=model(return_loss=True,**batch);loss,logs=model.module._parse_losses(losses)
  if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite clean training loss')
  scaler.scale(loss).backward();scaler.unscale_(optimizer)
  grads=[x.grad for x in model.parameters() if x.grad is not None];assert grads
  bad=[n for n,x in model.named_parameters() if x.grad is not None and not bool(torch.isfinite(x.grad).all())]
  r['amp_attempts'].append({'scale':scale,'nonfinite_tensors':len(bad),'loss':float(loss.detach().cpu())});save()
  if bad:
   scaler.step(optimizer);scaler.update();del loss,losses,grads;continue
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
  if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient norm')
  scaler.step(optimizer);scaler.update();torch.cuda.synchronize();change=float((parameter.detach()-before).abs().max())
  if change<=0:raise RuntimeError('Optimizer did not update parameters')
  r.update(status='clean_finetune_passed',successful_optimizer_updates=1,loss=float(loss.detach().cpu()),loss_terms=logs,gradient_norm=float(norm),parameter_max_change=change,successful_update_seconds=time.time()-t,first_update_seconds=time.time()-t0,learning_rate=cfg.optimizer.lr,optimizer='AdamW; native trainable parameter groups; one update, no schedule benchmark');save();break
 else:raise FloatingPointError('No finite update within bounded AMP retries')
 optimizer.zero_grad(set_to_none=True);del optimizer,buffers,batch,item,before,loss,losses,grads
 model.module.pts_bbox_head.reset_memory();model.module.prev_scene_token=None
 updated_path=W/'clean_finetuned_one_update.pth'
 native_state=model.module.state_dict();cpu_state=OrderedDict((k,v.detach().cpu()) for k,v in native_state.items());cpu_state._metadata=copy.deepcopy(native_state._metadata)
 torch.save({'state_dict':cpu_state,'meta':{'source_checkpoint_sha256':expected,'successful_optimizer_updates':1,'training_data_sha256':alignment['generated_train_sha256'],'classes':list(cfg.class_names),'scope':'clean DD2 generated frame; minimal plumbing check'}},updated_path)
 saved=torch.load(updated_path,map_location='cpu');model.module.load_state_dict(saved['state_dict'],strict=True)
 assert saved['state_dict']._metadata==native_state._metadata
 assert all(torch.equal(v.detach().cpu(),saved['state_dict'][k]) for k,v in model.module.state_dict().items())
 del saved,native_state,cpu_state
 r.update(updated_checkpoint=str(updated_path),updated_checkpoint_sha256=sha(updated_path),updated_checkpoint_reload_passed=True);save()
 r['after']=evaluate('after');r['delta_after_minus_before']={k:r['after'][k]-r['before'][k] for k in ['mAP','NDS']}
 assert sha(ckpt)==expected
 r.update(status='clean_e2e_passed',native_checkpoint_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,clean_validation_evaluated=True,scope='Pipeline correctness only: one generated six-camera sample, one optimizer update, four clean official mini_val frames. No full-benchmark or efficacy claim.')
except Exception as e:r.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()[-10000:])
r['total_seconds']=time.time()-start;save();print(json.dumps({k:r.get(k) for k in ['status','error','successful_optimizer_updates','before','after','total_seconds']}))
if r['status']=='failed':sys.exit(2)
