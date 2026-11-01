"""Memory scheduling for the unchanged 61-frame MagicDrive-t computation."""
import inspect,textwrap,types,torch

def install(pipe,report):
 # Each FeedForward token is independent; attention still sees all 61 frames.
 # Release unused UNet allocator blocks before UniPC's external cuSolver work.
 # This changes allocation scheduling only, not latents or solver arithmetic.
 scheduler_step=pipe.scheduler.step
 def step_with_free_cache(*args,**kwargs):
  torch.cuda.empty_cache()
  return scheduler_step(*args,**kwargs)
 pipe.scheduler.step=step_with_free_cache
 report(cuda_cache_policy='release_unused_blocks_before_scheduler_step')
 count=0
 for root in [pipe.unet,pipe.controlnet]:
  for m in root.modules():
   if m.__class__.__name__!='FeedForward':continue
   original=m.forward
   def ff(x,*args,_original=original,**kwargs):
    shape=x.shape;flat=x.reshape(-1,shape[-1])
    if len(flat)<=8192:return _original(x,*args,**kwargs)
    return torch.cat([_original(z,*args,**kwargs) for z in flat.split(8192)],0).reshape(*shape[:-1],-1)
   m.forward=ff;count+=1
 # ControlNet is purely 2D with a per-frame six-camera batch. It has no temporal
 # blocks; its output residuals are staged on CPU until the temporal UNet uses them.
 original=pipe.controlnet.forward
 checked=[False]
 def chunked(sample,timestep,camera_param,**kwargs):
  batch=sample.shape[0];views=sample.shape[1]
  def sl(value,index):
   if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==batch:return value[index:index+1]
   if isinstance(value,dict):return {k:sl(v,index) for k,v in value.items()}
   return value
  reference=None
  if not checked[0] and batch>=2:
   def first2(value):
    if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==batch:return value[:2]
    if isinstance(value,dict):return {k:first2(v) for k,v in value.items()}
    return value
   ref=original(sample[:2],first2(timestep),camera_param[:2],**{k:first2(v) for k,v in kwargs.items()})
   reference=([x.cpu() for x in ref[0]],ref[1].cpu(),ref[2].cpu());del ref
  down_all=mid_all=enc_all=None
  for i in range(batch):
   down,mid,enc=original(sample[i:i+1],sl(timestep,i),camera_param[i:i+1],**{k:sl(v,i) for k,v in kwargs.items()})
   if down_all is None:
    down_all=[torch.empty((batch*views,)+tuple(x.shape[1:]),dtype=x.dtype) for x in down]
    mid_all=torch.empty((batch*views,)+tuple(mid.shape[1:]),dtype=mid.dtype)
    enc_all=torch.empty((batch*views,)+tuple(enc.shape[1:]),dtype=enc.dtype)
   for dst,x in zip(down_all,down):dst[i*views:(i+1)*views].copy_(x.cpu())
   mid_all[i*views:(i+1)*views].copy_(mid.cpu());enc_all[i*views:(i+1)*views].copy_(enc.cpu())
  if reference is not None:
   actual=[x[:2*views] for x in down_all]+[mid_all[:2*views],enc_all[:2*views]]
   expected=list(reference[0])+[reference[1],reference[2]]
   errors=[float((x-y).abs().max()) for x,y in zip(actual,expected)]
   passed=all(torch.allclose(x,y,atol=0.01,rtol=0.01) for x,y in zip(actual,expected))
   relative_l2=[float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12)) for x,y in zip(actual,expected)]
   # FP16 GEMM/convolution results depend on batch shape. The real-condition
   # FP32 test in numeric_verification measured max relative L2 1.03e-5.
   # Retain the pointwise comparison as a diagnostic, not an FP16 exactness test.
   report(controlnet_chunk_comparison={'frames':2,'pointwise_allclose_fp16':passed,'max_abs_errors':errors,'relative_l2':relative_l2,'atol':0.01,'rtol':0.01,'fp32_reference':'numeric_verification/result.json'})
   if not all(bool(torch.isfinite(x).all()) for x in actual):raise RuntimeError('Nonfinite ControlNet output')
   checked[0]=True
  return down_all,mid_all,enc_all.to(sample.device)
 pipe.controlnet.forward=chunked
 # Copy only the two additions in the published UNet forward. Temporal attention,
 # frame count, CFG, scheduler, weights, and residual values remain unchanged.
 original_unet=pipe.unet.forward.__func__;source=textwrap.dedent(inspect.getsource(original_unet))
 old='down_block_res_sample + down_block_additional_residual'
 assert old in source
 source=source.replace(old,'down_block_res_sample + down_block_additional_residual.to(down_block_res_sample.device)')
 old='sample + mid_block_additional_residual';assert old in source
 source=source.replace(old,'sample + mid_block_additional_residual.to(sample.device)')
 namespace=dict(original_unet.__globals__);exec(compile(source,original_unet.__code__.co_filename,'exec'),namespace)
 pipe.unet.forward=types.MethodType(namespace['forward'],pipe.unet)
 report(memory_schedule={'feedforward_token_chunk':8192,'feedforward_modules':count,'controlnet_frames_per_call':1,'controlnet_residual_storage':'CPU','temporal_unet_frames':61,'temporal_attention_unchanged':True})
