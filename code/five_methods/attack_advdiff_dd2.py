from common import *
import subprocess,numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=N);a=p.parse_args()
out=O/'advdiff';rpc=O/'advdiff_rpc';rpc.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True);start=time.time();state={'method':'AdvDiff-inspired-DD2-v1','status':'starting','native_steps':30,'guided_steps':10,'guidance_strength':.025,'inner_steps_per_sampling_step':1,'initial_noise':'paired with original DD2 clip seed','gradient':'detached UNet prediction, differentiable predicted-clean VAE decode, native StreamPETR loss gradient via BPDA preprocessing','guidance_rule':'Euler predicted-clean sample shifted by strength*sigma^2*RMS-normalized d(detloss)/d(latent); converted back to native prediction parameterization','scope':'independent original-condition clip regeneration for each selected keyframe','memory_policy':'frozen UNet CPU-offloaded; two-pass VAE decode and detector CPU-offload prevent simultaneous graphs; original resolution and conditions retained','completed':[]}
def save(**kw):state.update(kw);state['active_attempt_seconds']=time.time()-start;state['seconds']=state['active_attempt_seconds']+state.get('reused_frame_seconds',0);atomic(out/'attack_status.json',state)
worker=None
try:
 sys.path[:0]=[str(R/'frameworks/20261028-v1/import_overlay'),str(R/'releases/20261028-geometry-research-v6/dreamer-models'),str(R/'releases/20261028-geometry-research-v6/dreamer-datasets'),str(R/'frameworks/20261028-v1')]
 import torch
 from dreamer_models import DriveDreamer2Pipeline
 from build_merged_wo_img_pipeline import build_verified_wo_img_cpu,place_complete_wo_img
 from dreamer_models.models.drivedreamer2.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
 from dreamer_models.models.drivedreamer2.grounding_downsampler import GroundingDownSampler
 torch.set_num_threads(1);selected=json.loads((O/'selection.json').read_text())['selected_tokens'];plans=json.loads((PLAN/'plan.json').read_text())['generators']['dd2'];by_token={t:clip for clip in plans for t in clip['export_tokens']};source=pickle.load((C/'infos_train.pkl').open('rb'));info_by_token={r['token']:r for r in source['infos']}
 save(status='loading_native_generator');model=build_verified_wo_img_cpu(R/'assets/dd2_merged_fp16',R/'assets/dd2_base',native_pipeline_class=DriveDreamer2Pipeline,unet_class=UNetSpatioTemporalConditionModel,grounding_class=GroundingDownSampler);model.frame_num=8
 for m in vars(model).values():
  if isinstance(m,torch.nn.Module):m.eval().requires_grad_(False)
 model.unet.enable_forward_chunking(chunk_size=1,dim=0);place_complete_wo_img(model,'cuda:0')
 # The detector uses its original compiled Torch1.11 environment in a child of this guarded job.
 env=os.environ.copy();env.pop('PYTORCH_CUDA_ALLOC_CONF',None);env.pop('PYTORCH_ALLOC_CONF',None);env['PYTHONPATH']=':'.join(map(str,[W,T,D,S]))
 for old in ['request.json','response.json']:
  p=rpc/old
  if p.exists():p.unlink()
 worker_log=(rpc/'worker.log').open('a');worker=subprocess.Popen([str(P),str(W/'detector_rpc.py')],env=env,cwd=str(W),stdout=worker_log,stderr=subprocess.STDOUT)
 request_id=0
 def detector(token,images=None,finish=False):
  global request_id
  request_id+=1;ident=str(request_id)+'_'+token
  request={'id':ident,'token':token,'finish':finish}
  if images is not None:
   np.save(rpc/'input.npy',images);request['array']=str(rpc/'input.npy')
  atomic(rpc/'request.json',request);deadline=time.time()+240
  while time.time()<deadline:
   if worker.poll() is not None:raise RuntimeError('Detector worker exited: '+(rpc/'worker.log').read_text()[-3500:])
   reply=rpc/'response.json'
   if reply.exists():
    response=json.loads(reply.read_text())
    if response.get('id')==ident:
     if finish:return response
     return response,np.load(response['gradient']).copy()
   time.sleep(.1)
  raise TimeoutError('Detector derivative response timeout')
 original_step=model.scheduler.step
 for index,token in enumerate(selected):
  folder=out/'images'/token;receipt=folder/'result.json'
  if receipt.exists():
   reused=json.loads(receipt.read_text());state['completed'].append(reused);state['reused_frame_seconds']=state.get('reused_frame_seconds',0)+reused['seconds'];continue
  if len(state['completed'])>=a.limit:break
  folder.mkdir(parents=True,exist_ok=True);clip=by_token[token];clip_dir=out/'conditions'/('%03d'%clip['index']);clip_dir.mkdir(parents=True,exist_ok=True);conditions=clip_dir/'dd2_conditions.pt';t0=time.time()
  if not conditions.exists():
   save(status='preparing_original_conditions',token=token,clip_index=clip['index'])
   with (clip_dir/'prepare.log').open('a') as f:subprocess.run([sys.executable,str(W/'prepare_conditions.py'),'--generator','dd2','--clip',clip['path'],'--output',str(clip_dir)],stdout=f,stderr=subprocess.STDOUT,check=True)
  data=torch.load(conditions,map_location='cpu');meta=json.loads((clip_dir/'dd2_input_status.json').read_text());camera_order=meta['camera_order'];detector_order=list(info_by_token[token]['cams']);slot=clip['tokens'].index(token)*2;counter=[0];logs=[]
  def guided_step(noise,timestep,sample,*args,**kwargs):
   step=counter[0];counter[0]+=1
   if step<20:return original_step(noise,timestep,sample,*args,**kwargs)
   sigma=model.scheduler.sigmas[step].to(device=sample.device,dtype=torch.float32);kind=model.scheduler.config.prediction_type
   # Release frozen denoiser weights during VAE backward; native predictions stay unchanged.
   model.unet.to('cpu');torch.cuda.empty_cache()
   def predict_clean(z):
    prediction=noise.detach().float()
    if kind=='v_prediction':return prediction*(-sigma/torch.sqrt(sigma*sigma+1))+z/(sigma*sigma+1)
    if kind=='epsilon':return z-sigma*prediction
    raise ValueError('Unsupported detached-prediction gradient '+kind)
   def decode_views(x0):
    decoded=model.decode_latents(x0[:,slot:slot+1].to(model.vae.dtype),num_frames=1,decode_chunk_size=1,clip_latents=False)[0,:,0];views=torch.stack([decoded[:,:,448*v:448*(v+1)] for v in range(6)]);images=(views/2+.5).clamp(0,1);return images[[camera_order.index(c) for c in detector_order]]
   # Do not hold VAE and detector graphs on the GPU at the same time.
   with torch.no_grad():
    reference_images=decode_views(predict_clean(sample.detach().float()));image_array=reference_images.cpu().numpy().copy();del reference_images
   torch.cuda.empty_cache();response,image_gradient=detector(token,image_array)
   # The RPC worker has returned its frozen model to CPU and released its cache.
   with torch.enable_grad():
    z=sample.detach().float().requires_grad_(True);x0=predict_clean(z);ordered=decode_views(x0);recompute_error=float((ordered.detach().cpu()-torch.from_numpy(image_array)).abs().max());adjoint=torch.from_numpy(image_gradient).to(device=ordered.device,dtype=ordered.dtype);gradient=torch.autograd.grad((ordered*adjoint*128).sum(),z)[0].float()/128
    if not torch.isfinite(gradient).all() or not bool(gradient.norm()>0):raise FloatingPointError('Zero or nonfinite DD2 latent guidance gradient')
    rms=gradient.square().mean().sqrt().clamp_min(1e-12);guided_x0=x0.detach()+.025*sigma.square()*gradient/rms
    if kind=='v_prediction':guided_noise=(guided_x0-sample.float()/(sigma*sigma+1))/(-sigma/torch.sqrt(sigma*sigma+1))
    else:guided_noise=(sample.float()-guided_x0)/sigma
    logs.append({'step':step,'sigma':float(sigma),'detector_loss':response['loss'],'image_gradient_l2':response['gradient_l2'],'latent_gradient_rms':float(rms),'predicted_clean_shift_rms':float((guided_x0-x0.detach()).square().mean().sqrt()),'VAE_recompute_max_RGB_error':recompute_error});save(status='guided_native_sampling',frame=index+1,token=token,completed_steps=step+1,last_guidance=logs[-1])
   del ordered,adjoint,gradient,z,x0;torch.cuda.empty_cache()
   model.unet.to(sample.device);torch.cuda.empty_cache()
   return original_step(guided_noise.to(noise.dtype),timestep,sample,*args,**kwargs)
  model.scheduler.step=guided_step;save(status='native_sampling',frame=index+1,token=token);torch.manual_seed(clip['seed'])
  with torch.no_grad():images=model(prompt_embeddings=data['prompt_embeds'].cuda().half(),input_dict={k:v.cuda().half() for k,v in data['input_dict'].items()},height=256,width=2688,num_frames=8,num_inference_steps=30,min_guidance_scale=2,max_guidance_scale=5,scheduled_sampling_beta=1.,decode_chunk_size=1,generator=torch.Generator(device='cuda').manual_seed(clip['seed']),output_type='pil').frames[0]
  model.scheduler.step=original_step;paths={};changed=0
  for v,camera in enumerate(camera_order):
   image=images[slot].crop((v*448,0,(v+1)*448,256));path=folder/(camera+'.png');image.save(path);paths[camera]=str(path);clean=np.asarray(Image.open(info_by_token[token]['cams'][camera]['data_path']).convert('RGB'));changed+=int(np.any(np.asarray(image)!=clean))
  record={'token':token,'paths':paths,'clip_index':clip['index'],'paired_seed':clip['seed'],'guidance_logs':logs,'changed_cameras':changed,'seconds':time.time()-t0,'geometry_statement':'Native HDMap/3D-box conditions and calibration retained; visual alignment audit pending. Not an exact guarantee of generated object-label correspondence.'};atomic(receipt,record);state['completed'].append(record);detector(token,finish=True);save();del images,data;torch.cuda.empty_cache()
  # Rebuildable, task-local condition batches mirror the Clean runner's cache policy.
  if a.limit==N and conditions.exists():conditions.unlink()
 if len(state['completed'])<N:save(status='partial_probe_complete');sys.exit(0)
 for row in source['infos']:
  record=next((r for r in state['completed'] if r['token']==row['token']),None)
  if record:
   for camera,path in record['paths'].items():row['cams'][camera]['data_path']=path
 pickle.dump(source,(out/'infos_train.pkl').open('wb'))
 # Config generation is delegated to the detector environment to retain mmcv serialization.
 code="import sys;sys.path.insert(0,"+repr(str(W))+");from common import *;from mmcv import Config;cfg=Config.fromfile(str(C/'streampetr.py'),import_custom_modules=False);cfg.data.train.ann_file=str(O/'advdiff/infos_train.pkl');cfg.work_dir=str(O/'advdiff/train');cfg.dump(str(O/'advdiff/streampetr.py'))"
 subprocess.run([str(P),'-c',code],check=True,env=env);save(status='completed',frames=N,changed_cameras=sum(r['changed_cameras'] for r in state['completed']),peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
except BaseException as e:
 if isinstance(e,SystemExit):raise
 save(status='failed',error=str(e),traceback=traceback.format_exc());raise
finally:
 if worker and worker.poll() is None:
  worker.terminate()
  try:worker.wait(timeout=5)
  except subprocess.TimeoutExpired:worker.kill();worker.wait()
