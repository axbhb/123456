"""Convert only the fixed training split with FrozenDrive's native converter (CPU)."""
import os,sys,json,pickle,collections,importlib.util,time,traceback
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';WD=R/'frozendrive_deployment_20261029/source/WorldDreamer';D=R/'datasets/frozendrive_mini_conditions';O=R/'outputs/table2_clean_mini_283_121/frozendrive';O.mkdir(parents=True,exist_ok=True)
os.chdir(str(WD));sys.path.insert(0,str(WD));sys.path.insert(1,str(WD/'third_party/bevfusion'))
state={'status':'running','gpu_used':False,'started':time.time()}
def save(**kw):
 state.update(kw);p=O/'native_infos_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
try:
 save(phase='native_converter')
 import numpy as np
 import dataset
 from nuscenes.nuscenes import NuScenes
 from omegaconf import OmegaConf
 from hydra import initialize_config_dir,compose
 from mmdet3d.datasets import build_dataset
 path=WD/'tools/data_converter/nuscenes_converter.py';spec=importlib.util.spec_from_file_location('frozendrive_native_converter',str(path));conv=importlib.util.module_from_spec(spec);spec.loader.exec_module(conv)
 nusc=NuScenes(version='v1.0-mini',dataroot=str(D),verbose=False)
 train=pickle.load((R/'datasets/streampetr_mini_scene_283_121/infos_train.pkl').open('rb'))['infos'];test=pickle.load((R/'datasets/streampetr_mini_scene_283_121/infos_test.pkl').open('rb'))['infos']
 assert len(train)==283 and len(test)==121;assert not {i['scene_token'] for i in train}&{i['scene_token'] for i in test}
 groups=collections.OrderedDict();infos=[]
 for row in sorted(train,key=lambda i:(i['scene_token'],i['timestamp'])):
  sample=nusc.get('sample',row['token']);info=conv._sample_data_transform(nusc,sample,max_sweeps=0,keyframe=True)
  assert info['token']==row['token'];assert len(info['cams'])==6
  for cam,c in info['cams'].items():assert Path(c['data_path']).is_file(),c['data_path']
  assert len(info['gt_boxes'])==len(info['gt_names'])==len(info['obj_ids'])
  infos.append(info);groups.setdefault(sample['scene_token'],[]).append(sample['token'])
 ann=O/'native_train_infos.pkl'
 with ann.open('wb') as f:pickle.dump({'infos':infos,'metadata':{'version':'v1.0-mini'},'scene_tokens':list(groups.values())},f)
 with initialize_config_dir(config_dir=str(WD/'configs'),version_base=None):cfg=compose(config_name='test_config_nus_448_depth_infl_ref_attn')
 cfg.dataset.dataset_root_nuscenes=str(D);cfg.dataset.dataset_root_occ=str(O/'conditions/projected');cfg.dataset.dataset_cache_file_nuscenes=[None,None]
 for split in ['train','val','test']:cfg.dataset.data[split].nuscenes.ann_file=str(ann)
 cfg.resume_from_checkpoint=str(WD/'dreamer-log/SDv1.5_mv_single_ref_nus/weight-E5-S200000');cfg.runner.num_workers=0
 OmegaConf.save(cfg,str(O/'native_clean_config.yaml'))
 params=OmegaConf.to_container(cfg.dataset.data.val.nuscenes,resolve=True);assert params['fps']==2 and params['test_mode']
 # Exercise native metadata/clip construction without substituting missing occupancy maps.
 params['pipeline']=[];ds=build_dataset(params)
 current=[ds.data_infos[c[-1]]['token'] for c in ds.clip_infos];assert len(current)==283 and len(set(current))==283 and set(current)=={x['token'] for x in train}
 firsts=[];pairs=[]
 for idx,pair in enumerate(ds.clip_infos):
  ref=ds.data_infos[pair[0]]['token'];cur=ds.data_infos[pair[-1]]['token'];is_first=bool(ds.data_infos[pair[-1]].get('is_first_frame',False))
  sr=nusc.get('sample',ref);sc=nusc.get('sample',cur);assert sr['scene_token']==sc['scene_token'];assert ref==cur if is_first else sr['next']==cur
  if is_first:firsts.append(cur)
  pairs.append({'index':idx,'reference_token':ref,'token':cur,'scene_token':sc['scene_token'],'is_first_frame':is_first,'timestamp':sc['timestamp']})
 assert len(firsts)==7
 (O/'native_generation_pairs.json').write_text(json.dumps(pairs,indent=2));save(status='passed',phase='native_metadata_ready_occupancy_pending',frames=len(infos),scenes=len(groups),pairs=len(ds),first_frame_pairs=len(firsts),fps=params['fps'],ann_file=str(ann),config=str(O/'native_clean_config.yaml'),image_size=list(cfg.dataset.image_size),camera_order=list(cfg.dataset.view_order_nuscenes),finished=time.time());print(json.dumps(state))
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
