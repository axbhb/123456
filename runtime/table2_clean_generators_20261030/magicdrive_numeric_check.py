"""Compare native two-frame and framewise ControlNet on actual conditions."""
import json,torch

def install(pipe,report):
 original=pipe.controlnet.forward
 for m in pipe.controlnet.modules():
  if m.__class__.__name__!='FeedForward':continue
  old=m.forward
  def ff(x,*args,_old=old,**kwargs):
   shape=x.shape;flat=x.reshape(-1,shape[-1])
   if len(flat)<=8192:return _old(x,*args,**kwargs)
   return torch.cat([_old(z,*args,**kwargs) for z in flat.split(8192)],0).reshape(*shape[:-1],-1)
  m.forward=ff
 def check(sample,timestep,camera_param,**kwargs):
  batch=sample.shape[0]
  def select(value,start,end,cast=None):
   if isinstance(value,torch.Tensor):
    value=value[start:end] if value.ndim and value.shape[0]==batch else value
    return value.to(dtype=cast) if cast is not None and value.is_floating_point() else value
   if isinstance(value,dict):return {k:select(v,start,end,cast) for k,v in value.items()}
   return value
  def forward(start,end,cast):
   values=original(select(sample,start,end,cast),select(timestep,start,end,cast),select(camera_param,start,end,cast),**{k:select(v,start,end,cast) for k,v in kwargs.items()})
   return [x.detach().float().cpu() for x in list(values[0])+[values[1],values[2]]]
  records={}
  for dtype,name in [(torch.float16,'fp16'),(torch.float32,'fp32')]:
   pipe.controlnet.to(dtype=dtype)
   with torch.inference_mode(),torch.autocast('cuda',enabled=False):
    ref=forward(0,2,dtype);a=forward(0,1,dtype);b=forward(1,2,dtype)
   stats=[]
   for y,u,v in zip(ref,a,b):
    x=torch.cat([u,v],0);d=x-y
    stats.append({'max_abs':float(d.abs().max()),'reference_abs_max':float(y.abs().max()),'relative_l2':float(d.norm()/y.norm().clamp_min(1e-12)),'rms_error':float(d.square().mean().sqrt()),'outside_fp16_allclose_fraction':float((d.abs()>0.01+0.01*y.abs()).float().mean()),'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all())})
   records[name]=stats;report(numerical_comparison=records)
   del ref,a,b
  report(status='completed',phase='numeric_verification',fp32_max_relative_l2=max(x['relative_l2'] for x in records['fp32']))
  raise SystemExit(0)
 pipe.controlnet.forward=check
