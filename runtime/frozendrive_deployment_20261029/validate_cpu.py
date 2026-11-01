"""CPU-only post-download validation; never launches video generation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import json,time,sys,traceback,gc
from pathlib import Path
W=Path(__file__).resolve().parent;WD=W/'source/WorldDreamer';state={'gpu_used':False,'phase':'waiting_for_assets'}
def save(): (W/'cpu_validation.json').write_text(json.dumps(state,indent=2))
try:
 save();deadline=time.time()+43200
 while time.time()<deadline:
  es=json.loads((W/'environment_status.json').read_text()) if (W/'environment_status.json').exists() else {}
  ws=json.loads((W/'weights_status.json').read_text()) if (W/'weights_status.json').exists() else {}
  if es.get('phase')=='ready' and ws.get('phase')=='download_complete':break
  if es.get('phase')=='failed' or ws.get('phase')=='failed':raise RuntimeError('Deployment prerequisite failed; see environment/weights status')
  time.sleep(30)
 else:raise TimeoutError('Deployment waiting window elapsed')
 os.chdir(WD);sys.path.insert(0,str(WD));import torch,diffusers,transformers
 torch.set_num_threads(2)
 state.update(phase='loading_models',versions={'torch':torch.__version__,'diffusers':diffusers.__version__,'transformers':transformers.__version__},components={});save()
 from projects.dreamer.networks.unet_2d_condition_multiview import UNet2DConditionModelMultiview
 from projects.dreamer.networks.unet_addon_single_ref_depth import SingleRefControlNetModel
 weights=WD/'dreamer-log/SDv1.5_mv_single_ref_nus/weight-E5-S200000'
 for name,cls in [('unet',UNet2DConditionModelMultiview),('controlnet',SingleRefControlNetModel)]:
  model,info=cls.from_pretrained(str(weights/name),torch_dtype=torch.float32,output_loading_info=True)
  issues={k:v for k,v in info.items() if v}
  if issues:raise RuntimeError(name+' loading discrepancies: '+str(issues))
  state['components'][name]={'parameters':sum(p.numel() for p in model.parameters()),'loading_info':info,'device':'cpu'};save();del model;gc.collect()
 from projects.dreamer.networks.clip_embedder import FrozenOpenCLIPImageEmbedderV2
 model=FrozenOpenCLIPImageEmbedderV2(arch='ViT-B-32',device='cpu',model_path=str(WD/'pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin'))
 with torch.no_grad():out=model(torch.zeros(1,3,224,224))
 if not torch.isfinite(out).all():raise RuntimeError('Nonfinite CLIP output')
 state['components']['clip']={'output_shape':list(out.shape),'device':'cpu'};save();del model;gc.collect()
 from diffusers import AutoencoderKL
 from transformers import CLIPTextModel,CLIPTokenizer
 base=WD/'pretrained/stable-diffusion-v1-5'
 for name,cls in [('vae',AutoencoderKL),('text_encoder',CLIPTextModel),('tokenizer',CLIPTokenizer)]:
  model=cls.from_pretrained(str(base),subfolder=name,local_files_only=True);state['components'][name]={'loaded':True};del model;gc.collect();save()
 image_proj=torch.load(weights/'image_proj/image_proj_model.bin',map_location='cpu');state['components']['image_proj']={'tensors':{k:list(v.shape) for k,v in image_proj.items()}}
 assert not torch.cuda.is_initialized()
 state.update(phase='passed',cuda_initialized=False,video_generation_tested=False);save()
except Exception:
 state.update(phase='failed',error=traceback.format_exc());save();raise
