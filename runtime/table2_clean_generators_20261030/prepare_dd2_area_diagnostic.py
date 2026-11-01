"""Render unchanged training conditions using DD2's native map/box preprocessing."""
import os,sys,json,pickle,copy,argparse,ast,importlib,types,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
p=argparse.ArgumentParser();p.add_argument('--clip',required=True);p.add_argument('--output',required=True);args=p.parse_args()
R=Path('/srv/table2');Release=R/'releases/20261028-geometry-research-v6';O=Path(args.output);O.mkdir(parents=True,exist_ok=True)
import numpy as np,torch
from PIL import Image
from pyquaternion import Quaternion
import matplotlib.style as style
if 'seaborn-whitegrid' not in style.library and 'seaborn-v0_8-whitegrid' in style.library: style.library['seaborn-whitegrid']=style.library['seaborn-v0_8-whitegrid']
torch.set_num_threads(1)
framework=R/'frameworks/20261028-v1'
sys.path[:0]=[str(framework/'import_overlay'),str(R/'releases/20261028-fixed-input-v4/dreamer-models'),str(R/'releases/20261028-fixed-input-v4/dreamer-datasets')]
from nuscenes import NuScenes
from dreamer_datasets import boxes3d_utils
raw=R/'assets/nuscenes_mini_temporal_camera_v1'
nusc=NuScenes(version='v1.0-mini',dataroot=str(raw),verbose=False)
native=framework/'sources/drivedreamer2/dreamer-train/projects/DriveDreamer2/drivedreamer2'
pkg=types.ModuleType('_mini_native_conditions');pkg.__path__=[str(native)];sys.modules[pkg.__name__]=pkg
transform=importlib.import_module(pkg.__name__+'.drivedreamer2_transforms').DriveDreamer2_Transform(resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=True,embed_map_path=str(R/'assets/dd2_base/clip_text_transform_after_pool_panoramic.pkl'),text_embed_dim=1024,box_normal=True)
geom=types.ModuleType('_mini_geometry');geom.__path__=[str(Release/'dreamer-models/dreamer_models/pipelines/drivedreamer2')];sys.modules[geom.__name__]=geom
Renderer=importlib.import_module(geom.__name__+'.native_hdmap_conditions').NativeHDMapRenderer
converter=Release/'dreamer-datasets/dd_scripts/converters/nuscenes_converter.py'
tree=ast.parse(converter.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NuScenesConverter')
nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='transform_matrix']+[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_get_cam_calib','_get_cam_label')]
if len(nodes)!=3: raise ValueError('Native converter functions missing')
ns=dict(np=np,math=math,Quaternion=Quaternion,boxes3d_utils=boxes3d_utils)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(converter),'exec'),ns)
receiver=types.SimpleNamespace(nusc=nusc);renderers={}
def renderer(scene):
    location=nusc.get('log',scene['log_token'])['location']
    if location not in renderers: renderers[location]=Renderer(raw,Release,location,nusc=nusc)
    return renderers[location]
def label(camera_token,index,frame):
    sd=nusc.get('sample_data',camera_token);sample=nusc.get('sample',sd['sample_token']);scene=nusc.get('scene',sample['scene_token']);calib=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);sensor=nusc.get('sensor',calib['sensor_token'])
    row=dict(data_index=index,scene_token=scene['token'],sample_token=sample['token'],cam_type=sensor['channel'].lower(),cam_token=camera_token,cam_timestamp=sd['timestamp'],calib=ns['_get_cam_calib'](receiver,camera_token),scene_description=scene['description'],frame_idx=frame,video_length=8)
    row.update(ns['_get_cam_label'](receiver,camera_token));row['ori_labels3d']=row.pop('labels3d')
    return row,scene,sd
def render(row,scene,sd,image_path=None):
    image_path=raw/sd['filename'] if image_path is None else image_path
    if not image_path.is_file(): raise FileNotFoundError(str(image_path))
    with Image.open(image_path) as im: image=im.convert('RGB')
    hdmap=renderer(scene).image(sd['token'],scene['token'])
    inputs=copy.deepcopy(row);inputs.update(image=image,image_hdmap=hdmap)
    value=transform(inputs)
    return value,hdmap

# Diagnostic only: compare native int32 area ordering to Python-integer ordering.
import inspect,textwrap,types,warnings
area_audit={'calls':0,'calls_with_changed_pixels':0,'changed_pixels':0,'max_abs_change':0.,'native_warning_counts':{}}
method_name=None
for name in dir(type(transform)):
 value=getattr(type(transform),name)
 if callable(value):
  try:
   source=inspect.getsource(value)
   if 'w_1*h_1<w_2*h_2' in source:method_name=name;break
  except (TypeError,OSError):pass
assert method_name is not None
original=getattr(transform,method_name);source=textwrap.dedent(inspect.getsource(original.__func__))
for name,a,b in [('w_1','corner[0,0]','corner[2,0]'),('h_1','corner[0,1]','corner[2,1]'),('w_2','corner[4,0]','corner[6,0]'),('h_2','corner[4,1]','corner[6,1]')]:
 source=source.replace('abs('+a+'-'+b+')','abs(int('+a+')-int('+b+'))')
namespace=dict(original.__func__.__globals__);exec(source,namespace);wide=types.MethodType(namespace[method_name],transform)
def compare(*args,**kwargs):
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',RuntimeWarning);native=original(*args,**kwargs)
 fixed=wide(*args,**kwargs);different=int(np.count_nonzero(native!=fixed));area_audit['calls']+=1;area_audit['changed_pixels']+=different;area_audit['calls_with_changed_pixels']+=bool(different)
 area_audit['max_abs_change']=max(area_audit['max_abs_change'],float(np.max(np.abs(native-fixed))))
 for warning in caught:
  key=str(warning.message);area_audit['native_warning_counts'][key]=area_audit['native_warning_counts'].get(key,0)+1
 (O/'area_comparison.json').write_text(json.dumps(area_audit,indent=2))
 return native
setattr(transform,method_name,compare)

plan=json.loads(Path(args.clip).read_text())
rows=pickle.load((R/'datasets/streampetr_mini_scene_283_121/infos_train.pkl').open('rb'))['infos'];allowed={i['token'] for i in rows}
assert set(plan['tokens'])<=allowed
ref=torch.load(R/'assets/dd2_aligned_clips/train_b_conditions.pt',map_location='cpu');order=list(ref['camera_order']);del ref
maps=torch.empty((8,3,256,2688),dtype=torch.float16);boxes=torch.empty((8,19,256,2688),dtype=torch.float16);embed=None;records=[];terminal=[]
for v,cam in enumerate(order):
 for j,token in enumerate(plan['tokens']):
  sample=nusc.get('sample',token);keysd=nusc.get('sample_data',sample['data'][cam])
  following=nusc.get('sample',sample['next']) if sample['next'] else None
  candidates=[];nextsd=keysd
  if following is not None:
   midpoint=(keysd['timestamp']+nusc.get('sample_data',following['data'][cam])['timestamp'])/2
   while nextsd['next']:
    nextsd=nusc.get('sample_data',nextsd['next']);candidates.append(nextsd)
    if nextsd['timestamp']>=midpoint:break
   midsd=min(candidates,key=lambda q:abs(q['timestamp']-midpoint)) if candidates else keysd
  else:
   midsd=keysd;terminal.append({'camera':cam,'frame':2*j+1})
  for frame,sd in [(2*j,keysd),(2*j+1,midsd)]:
   row,scene,sd=label(sd['token'],v*8+frame,frame)
   assert scene['token']==plan['scene_token']
   # RGB is not consumed by wo_img; a real train keyframe supplies dimensions if a sweep image is absent.
   source=raw/sd['filename'];carrier=source if source.is_file() else raw/keysd['filename']
   values,_=render(row,scene,sd,carrier)
   for name,target in [('grounding_downsampler_input',maps),('box_downsampler_input',boxes)]:target[frame,:,:,448*v:448*(v+1)]=values[name].half()
   e=torch.as_tensor(values['prompt_embeds']).unsqueeze(0).half()
   if embed is None:embed=e
   else:assert torch.equal(embed,e)
   records.append({'camera':cam,'frame':frame,'camera_token':sd['token'],'timestamp':sd['timestamp']})
for t in (maps,boxes,embed):assert torch.isfinite(t).all()
torch.save({'input_dict':{'grounding_downsampler_input':maps,'box_downsampler_input':boxes},'prompt_embeds':embed},O/'dd2_conditions.pt')
(O/'dd2_input_status.json').write_text(json.dumps({'status':'prepared','camera_order':order,'samples':plan['tokens'],'source_to_generated':[[448/1600,0,0],[0,256/900,0],[0,0,1]],'records':records,'terminal_padding':terminal,'rgb_model_input':False},indent=2))
print('Prepared',plan['index'])
