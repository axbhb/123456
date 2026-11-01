"""Prepare an explicit, scene-disjoint StreamPETR camera split from native infos."""
import argparse,os,pickle,json,copy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--train-infos',type=Path,required=True);p.add_argument('--test-infos',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--version',default='v1.0-trainval');a=p.parse_args();r=a.root.resolve();d=a.output.resolve();d.mkdir(parents=True,exist_ok=True);template=r/'datasets/streampetr_mini_scene_283_121';sys.path.insert(0,str(template));from mmcv import Config
sample_table={x['token']:x for x in json.loads((a.raw_root/a.version/'sample.json').read_text())};scene_table={x['token']:x for x in json.loads((a.raw_root/a.version/'scene.json').read_text())};splits={}
for name,path in [('train',a.train_infos),('test',a.test_infos)]:
 data=pickle.load(path.open('rb'));infos=data['infos'];assert infos
 for row in infos:
  record=sample_table[row['token']];row.setdefault('scene_token',record['scene_token']);row.setdefault('scene_name',scene_table[row['scene_token']]['name']);row.setdefault('prev',record['prev']);row.setdefault('next',record['next'])
  for key in (['bboxes2d','labels2d','depths'] if name=='train' else [])+['lidar2ego_rotation','lidar2ego_translation','ego2global_rotation','ego2global_translation']:assert key in row,(row['token'],key,'use native StreamPETR 2D/ego/temporal converter')
  assert len(row['cams'])==6
  for cam in row['cams'].values():
   path=Path(cam['data_path']);text=str(path).replace(chr(92),'/');i=text.find('samples/')
   if i>=0 and (a.raw_root/text[i:]).is_file():path=a.raw_root/text[i:]
   assert path.is_file(),str(path);cam['data_path']=str(path.absolute())
 data['infos']=sorted(infos,key=lambda x:(x['scene_token'],x['timestamp']));data.setdefault('metadata',{})['version']=a.version;pickle.dump(data,(d/('infos_'+name+'.pkl')).open('wb'));splits[name]=data['infos']
assert not {x['scene_token'] for x in splits['train']}&{x['scene_token'] for x in splits['test']}
assert not {x['token'] for x in splits['train']}&{x['token'] for x in splits['test']}
module=(template/'mini_camera_dataset.py').read_text();module=module.replace("for split in ['mini_train','mini_val']:","for split in (['mini_train','mini_val'] if self.version=='v1.0-mini' else ['train','val']):");module=module.replace("'Custom 121-frame mini test subset; not official mini_val'","str(len(tokens))+' explicit test keyframes; native nuScenes detection scoring'");(d/'mini_camera_dataset.py').write_text(module)
cfg=Config.fromfile(str(template/'streampetr_mini.py'),import_custom_modules=False)
for name in ['train','test','val']:
 cfg.data[name].ann_file=str(d/('infos_train.pkl' if name=='train' else 'infos_test.pkl'));cfg.data[name].data_root=str(a.raw_root.absolute())
cfg.runner.max_iters=len(splits['train']);cfg.checkpoint_config.interval=len(splits['train']);cfg.dump(str(d/'streampetr_mini.py'));(d/'dataset_summary.json').write_text(json.dumps(dict(version=a.version,train_frames=len(splits['train']),test_frames=len(splits['test']),train_scenes=len({x['scene_token'] for x in splits['train']}),test_scenes=len({x['scene_token'] for x in splits['test']}),raw_root=str(a.raw_root.absolute())),indent=2));print((d/'dataset_summary.json').read_text())
