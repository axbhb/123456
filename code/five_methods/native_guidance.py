from common import *
import subprocess,numpy as np,atexit
class NativeGuide:
 def __init__(self,token,camera_order,detector_order,save):
  self.token=token;self.order=[camera_order.index(c) for c in detector_order];self.save=save;self.logs=[];self.rpc=O/'advdiff_rpc';self.rpc.mkdir(parents=True,exist_ok=True);self.worker=None;self.ident=0;save(runtime_modules=runtime_summary());atexit.register(self.abort)
 def query(self,images=None,finish=False):
  if self.worker is None:
   for n in ['request.json','response.json']:
    p=self.rpc/n
    if p.exists():p.unlink()
   env=os.environ.copy();env['TABLE2_RPC_TOKEN']=self.token;env.pop('PYTORCH_CUDA_ALLOC_CONF',None);env.pop('PYTORCH_ALLOC_CONF',None);env['PYTHONPATH']=':'.join(map(str,[W,T,D,S]));self.log=(self.rpc/'worker.log').open('a');self.worker=subprocess.Popen([str(P),str(W/'detector_rpc.py')],env=env,stdout=self.log,stderr=subprocess.STDOUT)
  self.ident+=1;ident=str(self.ident)+'_'+self.token;request=dict(id=ident,token=self.token,finish=finish)
  if images is not None:np.save(self.rpc/'input.npy',images);request['array']=str(self.rpc/'input.npy')
  atomic(self.rpc/'request.json',request);deadline=time.time()+300
  while time.time()<deadline:
   if self.worker.poll() is not None:raise RuntimeError('detector worker exited '+(self.rpc/'worker.log').read_text()[-3000:])
   p=self.rpc/'response.json'
   if p.exists():
    reply=json.loads(p.read_text())
    if reply.get('id')==ident:
     if finish:return reply
     return reply,np.load(reply['gradient']).copy()
   time.sleep(.1)
  raise TimeoutError('native detector RPC timeout')
 def abort(self):
  if self.worker is not None and self.worker.poll() is None:
   self.worker.terminate()
   try:self.worker.wait(timeout=10)
   except subprocess.TimeoutExpired:self.worker.kill();self.worker.wait(timeout=10)
 def finish(self):
  if self.worker is not None:
   if self.worker.poll() is None:self.query(finish=True);self.worker.wait(timeout=30)
   self.log.close()
 def refine(self,x0,sigma,vae,offload,indices=None,panorama=False,scale=None,step=None):
  import torch
  for module in offload:module.cpu()
  torch.cuda.empty_cache();vae.to('cuda');scale=float(scale if scale is not None else vae.config.scaling_factor);indices=list(range(len(x0))) if indices is None else list(indices);selected=x0[indices].detach();dtype=next(vae.parameters()).dtype
  def decode(z):
   
   with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled() and dtype!=torch.float32,dtype=torch.get_autocast_gpu_dtype()):raw=vae.decode(z.to(dtype)/scale)
   raw=raw.sample if hasattr(raw,'sample') else raw
   if isinstance(raw,(tuple,list)):raw=raw[0]
   return (raw/2+.5).clamp(0,1)
  with torch.no_grad():
   if panorama:
    raw=decode(selected);width=raw.shape[-1]//6;reference=torch.cat([raw[:,:,:,width*i:width*(i+1)] for i in range(6)],dim=0)
   else:reference=torch.cat([decode(z) for z in selected.split(1)],dim=0)
   image_array=reference[self.order].float().cpu().numpy().copy();del reference
  torch.cuda.empty_cache();reply,image_grad=self.query(image_array);inverse=[self.order.index(i) for i in range(6)];image_grad=image_grad[inverse];gradient=torch.zeros_like(selected,dtype=torch.float32);errors=[]
  with torch.enable_grad():
   if panorama:
    z=selected.float().detach().requires_grad_(True);raw=decode(z);width=raw.shape[-1]//6;views=torch.cat([raw[:,:,:,width*i:width*(i+1)] for i in range(6)],dim=0);errors.append(float((views.detach().float().cpu()-torch.from_numpy(image_array[inverse])).abs().max()));adjoint=torch.from_numpy(image_grad).to(views.device,dtype=views.dtype);gradient=torch.autograd.grad((views*adjoint*128).sum(),z)[0].float()/128;del raw,views,adjoint,z
   else:
    for i in range(6):
     z=selected[i:i+1].float().detach().requires_grad_(True);view=decode(z);errors.append(float((view.detach().float().cpu()-torch.from_numpy(image_array[inverse][i:i+1])).abs().max()));adjoint=torch.from_numpy(image_grad[i:i+1]).to(view.device,dtype=view.dtype);gradient[i:i+1]=torch.autograd.grad((view*adjoint*128).sum(),z)[0].float()/128;del view,adjoint,z;torch.cuda.empty_cache()
  if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm()>0):raise FloatingPointError('invalid native guidance derivative')
  rms=gradient.square().mean().sqrt();shift=.025*float(sigma)**2*gradient/rms.clamp_min(1e-12);guided=x0.detach().clone();guided[indices]=(selected.float()+shift).to(x0.dtype);self.logs.append(dict(step=step,sigma=float(sigma),detector_loss=reply['loss'],image_gradient_l2=reply['gradient_l2'],predicted_clean_gradient_rms=float(rms),predicted_clean_shift_rms=float(shift.square().mean().sqrt()),VAE_recompute_max_RGB_error=max(errors)));self.save(phase='native_detector_guidance',guidance_steps=len(self.logs),last_guidance=self.logs[-1]);del gradient,shift,selected;torch.cuda.empty_cache()
  for module in offload:module.to(x0.device)
  torch.cuda.empty_cache();return guided
 def install_diffusers(self,pipe,indices):
  import torch
  assert pipe.scheduler.predict_x0, 'native scheduler must return predicted clean sample'
  original=pipe.scheduler.convert_model_output;counter=[0]
  def convert(model_output,timestep,sample,*args,**kwargs):
   x0=original(model_output,timestep,sample,*args,**kwargs);step=counter[0];counter[0]+=1
   if step<len(pipe.scheduler.timesteps)-10:return x0
   alpha=pipe.scheduler.alphas_cumprod[int(timestep)];sigma=((1-alpha)/alpha).sqrt();return self.refine(x0,sigma,pipe.vae,[pipe.unet,pipe.controlnet],indices=indices,step=step)
  pipe.scheduler.convert_model_output=convert
