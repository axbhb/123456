"""One real, normal Panacea+ generation through the official sampling method."""
import os,sys,json,time,traceback,argparse,gc,types
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='0');parser.add_argument('--steps',type=int,default=25);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=20261030);args=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=Path('/srv/table2');W=R/'generator_deployment_20261028';S=W/'sources/panacea_plus';O=Path(args.output);O.mkdir(parents=True,exist_ok=True)
os.chdir(str(S));sys.path.insert(0,str(S))
state={'status':'running','started':time.time(),'gpu':args.gpu,'sampling_steps':args.steps,'attack':False}
def save(**kw):
 state.update(kw);(O/'result.json').write_text(json.dumps(state,indent=2))
try:
 import torch,numpy as np
 from PIL import Image
 from omegaconf import OmegaConf
 from sgm.util import instantiate_from_config
 torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 cfg=OmegaConf.load(str(S/'configs/inference_nuscenes.yaml'))
 cfg.model.params.sampler_config.params.num_steps=args.steps
 cfg.model.params.log_keys=[] # Text-preview font is not part of generation.
 for emb in cfg.model.params.conditioner_config.params.emb_models:
  if emb.target.endswith('FrozenOpenCLIPEmbedder'):emb.params.version=None
 save(phase='constructing_model')
 model=instantiate_from_config(cfg.model)
 ckpt=R/'assets/generators/panacea_plus/panaceaplus_40k_deepspeed.ckpt'
 save(phase='loading_official_checkpoint')
 sd=torch.load(str(ckpt),map_location='cpu',mmap=True)
 if 'state_dict' in sd:sd=sd['state_dict']
 sd={k.removeprefix('_forward_module.'):v for k,v in sd.items()}
 result=model.load_state_dict(sd,strict=False)
 missing_parameters=sorted(set(result.missing_keys)&set(dict(model.named_parameters())))
 save(missing_keys=list(result.missing_keys),unexpected_keys=list(result.unexpected_keys),missing_parameters=missing_parameters)
 if missing_parameters:raise RuntimeError('Official checkpoint lacks parameters: '+repr(missing_parameters[:20]))
 del sd;gc.collect()
 model=model.eval().requires_grad_(False)
 # Keep VAE fp32 for numerical stability; encode/decode one complete panorama at a time.
 vae=model.first_stage_model
 encode_original=vae.encode
 def encode_chunked(x):
  return torch.cat([encode_original(frame.float()) for frame in x.split(1)],dim=0)
 vae.encode=encode_chunked
 torch.cuda.reset_peak_memory_stats()
 batch=torch.load(args.input,map_location='cpu')
 batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
 save(phase='conditioning',input_shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},precision={'denoiser':'fp16 weights and autocast','vae':'fp32, one panorama per encode/decode','conditioning':'native fp32 weights with fp16 autocast'},sequential_cfg=True,skipped_diagnostics=['input reconstruction preview'])
 start=time.time()
 with torch.no_grad():
  model.conditioner.to('cuda')
  batch_uc=batch.copy();batch_uc['txt']=['' for _ in batch['txt']]
  with torch.autocast('cuda',dtype=torch.float16):
   c,uc=model.conditioner.get_unconditional_conditioning(batch,batch_uc=batch_uc,force_uc_zero_embeddings=[])
  assert all(torch.isfinite(v).all() for d in (c,uc) for v in d.values() if isinstance(v,torch.Tensor))
  save(conditioning_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
  # Shared VAE is also moved to CPU here; only the denoiser stays on GPU during sampling.
  model.conditioner.cpu();del batch,batch_uc;gc.collect();torch.cuda.empty_cache()
  model.model.half().to('cuda');model.denoiser.to('cuda')
  # Lightning caches the root device even when its submodules are moved separately.
  model._device=torch.device('cuda',0)
  save(phase='native_sampling',denoiser_parameter_gib=sum(p.numel()*p.element_size() for p in model.model.parameters())/1024**3)
  from native_guidance import NativeGuide
  context=json.loads(os.environ['TABLE2_GUIDANCE_CONTEXT']);guide=NativeGuide(context['token'],context['camera_order'],context['detector_order'],save);slot=context['slot']
  completed=[0]
  def sequential_denoise(sampler,x,denoiser,sigma,cond,uncond):
   # Same native CFG formula, evaluating its two branches without doubling the video batch.
   u=denoiser(x,sigma,uncond)
   v=denoiser(x,sigma,cond)
   guided=sampler.guider(torch.cat([u,v],dim=0),sigma)
   if not torch.isfinite(guided).all():raise RuntimeError('Nonfinite guided denoiser output')
   if completed[0]>=args.steps-10:
    guided=guide.refine(guided,sigma[0],vae,[model.model],indices=[slot],panorama=True,scale=model.scale_factor,step=completed[0])
   completed[0]+=1
   save(completed_steps=completed[0],peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
   return guided
  model.sampler.denoise=types.MethodType(sequential_denoise,model.sampler)
  with torch.autocast('cuda',dtype=torch.float16):
   latent=model.sample(c,shape=(4,32,384),uc=uc,batch_size=8)
  guide.finish()
  (O/'guidance.json').write_text(json.dumps(guide.logs,indent=2))
  assert torch.isfinite(latent).all(),'Nonfinite sampled latent'
  model.model.cpu();del c,uc;gc.collect();torch.cuda.empty_cache()
  vae.to('cuda');save(phase='decoding')
  samples=torch.cat([model.decode_first_stage(frame).float().cpu() for frame in latent.split(1)],dim=0)
 out={'samples':samples}
 samples=out['samples'].float().cpu()
 assert samples.shape==(8,3,256,3072),samples.shape
 assert torch.isfinite(samples).all(),'Nonfinite generated output'
 # Native PNG views are sufficient; no duplicate floating-point video cache.
 frames=[]
 for t,x in enumerate(samples):
  if t!=slot:continue
  a=((x.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
  im=Image.fromarray(a);im.save(str(O/('panorama_%02d.png'%t)));frames.append(im.resize((1536,128)))
  for v in range(6):Image.fromarray(a[:,v*512:(v+1)*512]).save(str(O/('view%d_frame%02d.png'%(v,t))))
 frames[0].save(str(O/'preview.gif'),save_all=True,append_images=frames[1:],duration=500,loop=0)
 save(status='passed',phase='finished',shape=list(samples.shape),sampling_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,finished=time.time())
except Exception as e:
 save(peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if 'torch' in globals() and torch.cuda.is_initialized() else None)
 save(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
