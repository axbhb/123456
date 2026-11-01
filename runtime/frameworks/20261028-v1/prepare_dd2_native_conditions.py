"""Build real DD2 native conditions from verified LMDB records, CPU only."""
import os,sys,json,time,hashlib,tarfile,traceback,importlib,types
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='1'
R=Path('/srv/table2');P=R/'frameworks/20261028-v1';A=R/'assets';src=A/'dd2_native_temporal_subset.tar';prompt=A/'dd2_base/clip_text_transform_after_pool_panoramic.pkl';out=A/'dd2_native_temporal_subset';state={'status':'waiting_assets','gpu_used':False,'started':time.time(),'deadline':time.time()+1800,'learned_generator_executed':False}
def save():
 q=P/'dd2_native_conditions_status.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(q)
save()
try:
 while (not src.exists() or not prompt.exists()) and time.time()<state['deadline']:time.sleep(10)
 if not src.exists() or not prompt.exists():raise TimeoutError('Native clip assets missing')
 assert hashlib.sha256(src.read_bytes()).hexdigest()=='90a25a8b83c40b29481f74305a544b6b5fcec6961a30f44e34075541597312f4'
 assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='3248abaa779dadd427c197aae79c55e8149e104daa6d9ee28d33de081af0710b'
 out.mkdir(exist_ok=True)
 with tarfile.open(src) as tar:
  manifest=json.load(tar.extractfile('manifest.json'))
  for row in manifest['files']:
   path=(out/row['path']).resolve();assert out.resolve() in path.parents
   member=tar.getmember(row['path']);assert member.isfile();payload=tar.extractfile(member).read();assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 sys.path[:0]=[str(P/'import_overlay'),str(R/'releases/20261028-fixed-input-v4/dreamer-models'),str(R/'releases/20261028-fixed-input-v4/dreamer-datasets')]
 import numpy as np,torch
 from PIL import Image
 torch.set_num_threads(1)
 native=P/'sources/drivedreamer2/dreamer-train/projects/DriveDreamer2/drivedreamer2'
 if not native.exists():raise FileNotFoundError(str(native))
 package=types.ModuleType('_dd2_native_conditions');package.__path__=[str(native)];sys.modules[package.__name__]=package
 module=importlib.import_module(package.__name__+'.drivedreamer2_transforms');Transform=module.DriveDreamer2_Transform
 transform=Transform(resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=True,embed_map_path=str(prompt),text_embed_dim=1024,box_normal=True)
 def hook(d):
  if '__ndarray__' in d:
   a=np.asarray(d['__ndarray__'],dtype=d['dtype']);assert a.size==int(__import__('math').prod(d['shape']));return a.reshape(d['shape'])
  return d
 state.update(status='rendering_native_conditions',source_transform_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),roles={});save()
 for role,info in manifest['roles'].items():
  rows=json.loads((out/role/'labels.json').read_text(),object_hook=hook);assert len(rows)==48
  maps=torch.empty((8,3,256,2688),dtype=torch.float16);boxes=torch.empty((8,19,256,2688),dtype=torch.float16);embed=None
  for i,row in enumerate(rows):
   assert row['cam_type'].upper()==manifest['camera_order'][i//8] and row['scene_token']==info['scene_token']
   row['image']=Image.open(out/role/'images'/('%03d.bin'%i)).convert('RGB');row['image_hdmap']=Image.open(out/role/'hdmaps'/('%03d.bin'%i)).convert('RGB');v=transform(row);c,t=divmod(i,8)
   for x in ['grounding_downsampler_input','box_downsampler_input']:assert torch.isfinite(v[x]).all()
   maps[t,:,:,448*c:448*(c+1)]=v['grounding_downsampler_input'];boxes[t,:,:,448*c:448*(c+1)]=v['box_downsampler_input']
   e=torch.as_tensor(v['prompt_embeds'])
   if embed is None:embed=e
   else:assert torch.equal(embed,e)
  # Match the native tester: a batch of 48 prompt embeddings, selecting [0:1].
  emb=embed.unsqueeze(0).half();target=out/(role+'_conditions.pt')
  torch.save({'input_dict':{'grounding_downsampler_input':maps,'box_downsampler_input':boxes},'prompt_embeds':emb,'camera_order':manifest['camera_order'],'scene_token':info['scene_token'],'source_manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),'native_source_index_stride':3,'actual_timestamps_us':info['camera_timestamps_us']},target)
  state['roles'][role]={'frames':8,'maps_shape':list(maps.shape),'boxes_shape':list(boxes.shape),'prompt_shape':list(emb.shape),'condition_dtype':str(maps.dtype),'maps_range':[float(maps.min()),float(maps.max())],'boxes_range':[float(boxes.min()),float(boxes.max())],'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'native_sampling_has_gaps':any(abs(dt-.25)>.015 for a in info['actual_intervals_seconds'].values() for dt in a)};save()
  del maps,boxes
 assert not torch.cuda.is_initialized();state.update(status='native_conditions_ready',cuda_initialized=False)
except Exception as e:state.update(status='failed',error=str(e),traceback=traceback.format_exc()[-4000:])
state['finished']=time.time();save();print(json.dumps(state))

if state['status']=='failed':sys.exit(2)
