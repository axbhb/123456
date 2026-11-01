"""Normal MagicDrive-t generation with its official 61-frame configuration."""
import os,sys,json,time,argparse,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261030);args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=Path('/srv/table2');W=R/'generator_deployment_20261028';S=W/'sources/magicdrive_video';A=R/'assets/generators';O=Path(args.output);O.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(S),str(S/'third_party/bevfusion'),str(S/'third_party/diffusers/src')];os.chdir(str(S))
state={'status':'running','started':time.time(),'gpu':args.gpu,'attack':False,'frames':61}
def save(**kw):state.update(kw);(O/'result.json').write_text(json.dumps(state,indent=2))
try:
 import torch,numpy as np
 from PIL import Image
 from omegaconf import OmegaConf
 from transformers import CLIPTextModel,CLIPTokenizer
 from diffusers import AutoencoderKL,UniPCMultistepScheduler
 from magicdrive.networks.unet_addon_rawbox import BEVControlNetModel
 from magicdrive.networks.unet_2d_condition_multiview_t import UNet2DConditionModelMultiviewT
 from magicdrive.pipeline.pipeline_bev_controlnet_t import StableDiffusionBEVControlNetTPipeline
 cfg=OmegaConf.load(str(A/'magicdrive_t_60f/hydra/config.yaml'));dtype=torch.float16
 save(phase='initializing_cuda_solver')
 # UniPC first calls cuSolver at step 3. Initialize before UNet cache fills VRAM.
 with torch.inference_mode():
  solver_probe=torch.linalg.solve(torch.eye(2,device='cuda',dtype=torch.float32),torch.ones(2,device='cuda',dtype=torch.float32))
  if not torch.equal(solver_probe,torch.ones_like(solver_probe)):raise RuntimeError('CUDA solver initialization failed')
  del solver_probe
 torch.cuda.synchronize();torch.cuda.empty_cache()
 save(phase='loading_official_components',cuda_solver_initialized=True)
 control,ci=BEVControlNetModel.from_pretrained(str(A/'magicdrive_t_60f/controlnet'),torch_dtype=dtype,output_loading_info=True)
 unet,ui=UNet2DConditionModelMultiviewT.from_pretrained(str(A/'magicdrive_t_60f/unet'),torch_dtype=dtype,output_loading_info=True)
 save(controlnet_loading=ci,unet_loading=ui)
 if ci.get('missing_keys') or ui.get('missing_keys'):raise RuntimeError('Missing published model parameters')
 for module in unet.modules():
  if hasattr(module,'_sc_attn_index'):module._sc_attn_index=OmegaConf.to_container(cfg.model.sc_attn_index,resolve=True)
 text=CLIPTextModel.from_pretrained(str(A/'sd15_base/text_encoder'),variant='fp16',torch_dtype=dtype,local_files_only=True)
 tokenizer=CLIPTokenizer.from_pretrained(str(A/'sd15_base/tokenizer'),local_files_only=True)
 vae=AutoencoderKL.from_pretrained(str(A/'sd15_base/vae'),variant='fp16',torch_dtype=dtype,local_files_only=True)
 scheduler=UniPCMultistepScheduler.from_pretrained(str(A/'sd15_base/scheduler'),local_files_only=True)
 pipe=StableDiffusionBEVControlNetTPipeline(vae=vae,text_encoder=text,unet=unet,controlnet=control,scheduler=scheduler,tokenizer=tokenizer)
 for component in (pipe.unet,pipe.controlnet,pipe.vae,pipe.text_encoder):component.eval().requires_grad_(False)
 pipe.enable_xformers_memory_efficient_attention();pipe.to('cuda')
 from magicdrive_memory import install
 install(pipe,save)
 batch=torch.load(args.input,map_location='cpu')
 params=OmegaConf.to_container(cfg.runner.pipeline_param,resolve=True)
 save(phase='native_sampling',pipeline_params=params,image_height=224,image_width=400,attention_slicing=False,attention_backend="xformers_cutlass")
 torch.cuda.reset_peak_memory_stats();start=time.time()
 with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
  images=pipe(prompt=batch['captions'],image=batch['bev_map_with_aux'],camera_param=batch['camera_param'].to(dtype=dtype,device='cuda'),height=224,width=400,bev_controlnet_kwargs=batch['kwargs'],generator=torch.Generator(device='cuda').manual_seed(args.seed),callback=lambda i,t,z:save(completed_steps=i+1,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,reserved_gib=torch.cuda.memory_reserved()/1024**3,device_free_gib=torch.cuda.mem_get_info()[0]/1024**3),**params).images
 assert len(images)==61 and all(len(frame)==6 for frame in images)
 frames=[]
 for t,views in enumerate(images):
  for v,im in enumerate(views):im.save(str(O/('view%d_frame%03d.png'%(v,t))))
  canvas=Image.new('RGB',(1200,448))
  for v,im in enumerate(views):canvas.paste(im,((v%3)*400,(v//3)*224))
  frames.append(canvas.resize((900,336)))
 frames[0].save(str(O/'preview.gif'),save_all=True,append_images=frames[1:],duration=83,loop=0)
 save(status='passed',phase='finished',sampling_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,finished=time.time(),png_frames=366)
except Exception as e:save(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
