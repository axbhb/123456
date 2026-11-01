"""Construct only a frozen wo_img DD2 pipeline from verified complete artifacts.
All model loading starts on CPU. No random model placeholders or fallback assets.
The native generation implementations are inherited unchanged; added guards only
reject unsupported conditioning. Downsamplers follow the native loader's attrs.
"""
import os,json,hashlib
from pathlib import Path
import torch
from diffusers import AutoencoderKLTemporalDecoder,EulerDiscreteScheduler
from transformers import CLIPImageProcessor
from load_merged_dd2_cpu import load_complete_dd2_cpu,_sha

ROOT_MODULES=('unet','vae','grounding_downsampler','box_downsampler')

def guarded_wo_img_class(native_pipeline_class):
 class VerifiedWoImgPipeline(native_pipeline_class):
  def _prepare_prompt_context(self,prompt_embeddings,*args,**kwargs):
   if not isinstance(prompt_embeddings,torch.Tensor) or prompt_embeddings.ndim<2 or prompt_embeddings.shape[0]!=1:
    raise ValueError('Supply a fixed prompt tensor with batch size one')
   if prompt_embeddings.requires_grad or not torch.isfinite(prompt_embeddings).all():
    raise ValueError('Prompt must be finite, detached and fixed')
   if prompt_embeddings.device!=self.unet.device or prompt_embeddings.dtype!=self.unet.dtype:
    raise ValueError('Prompt device/dtype must match the loaded UNet')
   return super()._prepare_prompt_context(prompt_embeddings,*args,**kwargs)
  def _extract_raw_condition_features(self,input_dict,*args,**kwargs):
   if not isinstance(input_dict,dict) or set(input_dict)-{'grounding_downsampler_input','box_downsampler_input'}:
    raise ValueError('Verified wo_img supports raw map/box tensors only; img_cond/video_cond are explicitly forbidden')
   if not input_dict:raise ValueError('Raw physical conditions are required')
   return super()._extract_raw_condition_features(input_dict,*args,**kwargs)
  def _encode_image(self,*args,**kwargs):
   raise RuntimeError('Image encoding is forbidden for this fixed-prompt wo_img-only pipeline')
 return VerifiedWoImgPipeline

def component_report(pipe):
 result={}
 for name in ROOT_MODULES:
  module=getattr(pipe,name);params=list(module.parameters())
  result[name]={'parameters':sum(p.numel() for p in params),'devices':sorted({str(p.device) for p in params}),'dtypes':sorted({str(p.dtype) for p in params}),'all_frozen':all(not p.requires_grad for p in params),'eval':not module.training}
 result['image_encoder']=None if pipe.image_encoder is None else type(pipe.image_encoder).__name__
 result['native_component_keys']=sorted(pipe.components)
 return result

def build_verified_wo_img_cpu(bundle_dir,base_assets_dir,*,native_pipeline_class,unet_class,grounding_class):
 bundle,base=Path(bundle_dir),Path(base_assets_dir)
 manifest=json.loads((bundle/'pipeline_assets_manifest.json').read_text())
 expected={'svd/vae/config.json','svd/vae/diffusion_pytorch_model.fp16.safetensors','svd/scheduler/scheduler_config.json','svd/feature_extractor/preprocessor_config.json'}
 if {r['relative'] for r in manifest}!=expected:raise ValueError('Incomplete base asset manifest')
 for row in manifest:
  path=base/row['relative']
  if path.stat().st_size!=row['bytes'] or _sha(path)!=row['sha256']:raise ValueError('Base artifact mismatch: '+row['relative'])
 groups=load_complete_dd2_cpu(bundle,unet_class=unet_class,grounding_class=grounding_class)
 vae=AutoencoderKLTemporalDecoder.from_pretrained(str(base/'svd'),subfolder='vae',variant='fp16',torch_dtype=torch.float16,local_files_only=True)
 scheduler=EulerDiscreteScheduler.from_pretrained(str(base/'svd'),subfolder='scheduler',local_files_only=True)
 feature=CLIPImageProcessor.from_pretrained(str(base/'svd'),subfolder='feature_extractor',local_files_only=True)
 cls=guarded_wo_img_class(native_pipeline_class)
 pipe=cls(vae=vae,image_encoder=None,unet=groups['unet'],scheduler=scheduler,feature_extractor=feature)
 # Native DD2 load_weights assigns these attributes. register_modules would add
 # unsupported config keys to DiffusionPipeline.components (constructor has only
 # the five native base components), so do not falsely register extra components.
 pipe.grounding_downsampler=groups['grounding_downsampler'];pipe.box_downsampler=groups['box_downsampler']
 for name in ROOT_MODULES:getattr(pipe,name).eval().requires_grad_(False)
 if set(pipe.components)!={'vae','image_encoder','unet','scheduler','feature_extractor'}:raise RuntimeError('Native pipeline component registration mismatch')
 if pipe.image_encoder is not None:raise RuntimeError('Unexpected image encoder')
 for name in ROOT_MODULES:
  if any(p.device.type!='cpu' or p.dtype!=torch.float16 for p in getattr(pipe,name).parameters()):raise RuntimeError('Invalid CPU FP16 component: '+name)
 return pipe

def place_complete_wo_img(pipe,device):
 device=torch.device(device)
 if device.type=='cuda':
  if device.index!=0 or os.environ.get('PHYSCOND_GPU_GUARD')!='1':raise PermissionError('CUDA placement requires the shared authorized single-GPU launcher')
 elif device.type!='cpu':raise ValueError('Unsupported execution device')
 pipe.to(device)
 # Native pipeline.to only traverses constructor components. Move both extra
 # condition modules explicitly before capturing the loaded model identity.
 for name in ROOT_MODULES:getattr(pipe,name).to(device=device,dtype=torch.float16).eval().requires_grad_(False)
 for name in ROOT_MODULES:
  if any(p.device!=device or p.dtype!=torch.float16 for p in getattr(pipe,name).parameters()):raise RuntimeError('Component placement mismatch: '+name)
 return pipe
