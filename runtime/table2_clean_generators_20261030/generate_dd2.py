"""Normal frozen DD2 wo_img inference, with no perturbations or attack update."""
import os,sys,json,time,traceback,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',required=True);p.add_argument('--seed',type=int,default=20261030);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
R=Path('/srv/table2');F=R/'frameworks/20261028-v1';O=Path(a.output);O.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(F/'import_overlay'),str(R/'releases/20261028-geometry-research-v6/dreamer-models'),str(R/'releases/20261028-geometry-research-v6/dreamer-datasets'),str(F)]
state={'status':'running','started':time.time(),'attack':False,'seed':a.seed}
def save(**kw):state.update(kw);(O/'result.json').write_text(json.dumps(state,indent=2))
save()
try:
 import torch
 from dreamer_models import DriveDreamer2Pipeline
 from build_merged_wo_img_pipeline import build_verified_wo_img_cpu,place_complete_wo_img
 from dreamer_models.models.drivedreamer2.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
 from dreamer_models.models.drivedreamer2.grounding_downsampler import GroundingDownSampler
 torch.set_num_threads(1);torch.manual_seed(a.seed)
 model=build_verified_wo_img_cpu(R/'assets/dd2_merged_fp16',R/'assets/dd2_base',native_pipeline_class=DriveDreamer2Pipeline,unet_class=UNetSpatioTemporalConditionModel,grounding_class=GroundingDownSampler)
 model.frame_num=8
 for m in vars(model).values():
  if isinstance(m,torch.nn.Module):m.eval().requires_grad_(False)
 model.unet.enable_forward_chunking(chunk_size=1,dim=0);place_complete_wo_img(model,'cuda:0')
 data=torch.load(a.input,map_location='cpu');save(phase='native_sampling');start=time.time();torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  images=model(prompt_embeddings=data['prompt_embeds'].cuda().half(),input_dict={k:v.cuda().half() for k,v in data['input_dict'].items()},height=256,width=2688,num_frames=8,num_inference_steps=30,min_guidance_scale=2,max_guidance_scale=5,scheduled_sampling_beta=1.,decode_chunk_size=1,generator=torch.Generator(device='cuda').manual_seed(a.seed),output_type='pil').frames[0]
 assert len(images)==8
 for t,im in enumerate(images):
  assert im.size==(2688,256)
  for v in range(6):im.crop((v*448,0,(v+1)*448,256)).save(O/('view%d_frame%02d.png'%(v,t)))
 save(status='passed',phase='finished',sampling_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,finished=time.time())
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
