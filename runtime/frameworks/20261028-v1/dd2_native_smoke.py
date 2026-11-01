"""Strict real-weight DD2 native generation preflight/smoke; no attack efficacy claim."""
import os,sys,argparse,json,time,hashlib,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu-load','gpu-native'],required=True);p.add_argument('--role',choices=['calibration','attack_development'],default='attack_development');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.mode=='cpu-load':os.environ['CUDA_VISIBLE_DEVICES']='-1'
elif os.environ.get('PHYSCOND_GPU_GUARD')!='1':raise RuntimeError('Use the shared GPU guard')
R=Path('/srv/table2');P=R/'frameworks/20261028-v1';A=R/'assets';a.output.mkdir(parents=True,exist_ok=True);start=time.time();r={'status':'started','mode':a.mode,'gpu_used':a.mode=='gpu-native','attack_update_executed':False,'learned_generator_executed':False}
def save():
 q=a.output/'result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(q)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
save()
try:
 manifest=json.loads((P/'dd2_required_weights_manifest.json').read_text());r['verified_assets']=[]
 for row in manifest:
  q=A/row['relative'];assert q.is_file() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],str(q);r['verified_assets'].append(row['relative'])
 status=json.loads((P/'dd2_native_conditions_status.json').read_text());assert status['status']=='native_conditions_ready';record=status['roles'][a.role];condition_path=Path(record['path']);assert sha(condition_path)==record['sha256']
 sys.path[:0]=[str(P/'import_overlay'),str(R/'releases/20261028-fixed-input-v4/dreamer-models'),str(R/'releases/20261028-fixed-input-v4/dreamer-datasets')]
 import torch
 from dreamer_models import DriveDreamer2Pipeline
 torch.set_num_threads(1);torch.manual_seed(0)
 data=torch.load(condition_path,map_location='cpu')
 assert set(data['input_dict'])=={'grounding_downsampler_input','box_downsampler_input'},'This complete loader is only for fixed-prompt wo_img'
 sys.path.insert(0,str(P))
 from build_merged_wo_img_pipeline import build_verified_wo_img_cpu,place_complete_wo_img,component_report
 from dreamer_models.models.drivedreamer2.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
 from dreamer_models.models.drivedreamer2.grounding_downsampler import GroundingDownSampler
 model=build_verified_wo_img_cpu(A/'dd2_merged_fp16',A/'dd2_base',native_pipeline_class=DriveDreamer2Pipeline,unet_class=UNetSpatioTemporalConditionModel,grounding_class=GroundingDownSampler)
 model.frame_num=8
 r['weight_load_mode']='complete merged FP16, verified equal to native loader then FP16';r['image_encoder']='None: unused; guarded class rejects image/video conditioning and encoder calls';r['cpu_components']=component_report(model)
 modules={id(x):x for x in vars(model).values() if isinstance(x,torch.nn.Module)}
 for module in modules.values():module.eval();module.requires_grad_(False)
 model.unet.enable_forward_chunking(chunk_size=1,dim=0)
 r.update(native_weight_loading_passed=True,all_weights_frozen=all(not x.requires_grad for m in modules.values() for x in m.parameters()),frame_num=8,camera_order=data['camera_order'],scene_token=data['scene_token'],data_sampling='native stride3 with original nonuniform timestamps preserved',load_seconds=time.time()-start)
 if a.mode=='cpu-load':assert not torch.cuda.is_initialized();r.update(status='cpu_native_load_passed',cuda_initialized=False)
 else:
  place_complete_wo_img(model,'cuda:0');r['gpu_components']=component_report(model)
  inputs={k:v.to('cuda:0',dtype=torch.float16) for k,v in data['input_dict'].items()};prompt=data['prompt_embeds'].to('cuda:0',dtype=torch.float16)
  kwargs=dict(height=256,width=2688,num_frames=8,num_inference_steps=30,min_guidance_scale=2,max_guidance_scale=5,scheduled_sampling_beta=1.,first_frame=True,decode_chunk_size=1,output_type='pt')
  r.update(status='native_generation_running',generation_kwargs=kwargs,seed=0,playback_fps=4,conditioning_fps=7);save();torch.cuda.synchronize();t=time.time()
  with torch.no_grad():video=model(prompt,input_dict=inputs,generator=torch.Generator(device='cuda:0').manual_seed(0),**kwargs).frames
  torch.cuda.synchronize();r['generation_seconds']=time.time()-t
  assert len(video)==1 and tuple(video[0].shape)==(8,3,256,2688)
  v=video[0].detach().float().cpu();assert torch.isfinite(v).all() and float(v.min())>=0 and float(v.max())<=1;torch.save(v,a.output/'native_video.pt')
  from PIL import Image
  frames=[Image.fromarray((x.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')) for x in v]
  frames[0].save(a.output/'native_preview.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
  r.update(status='native_generation_passed',learned_generator_executed=True,shape=list(v.shape),pixel_range=[float(v.min()),float(v.max())],peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,video_tensor=str(a.output/'native_video.pt'),preview=str(a.output/'native_preview.gif'))
except Exception as e:r.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()[-5000:])
r['total_seconds']=time.time()-start;save();print(json.dumps(r))
if r['status']=='failed':sys.exit(2)
