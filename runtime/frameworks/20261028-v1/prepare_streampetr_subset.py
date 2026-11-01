"""Subset official StreamPETR mini infos to transferred complete scenes; no model execution."""
import argparse,copy,hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
TRUSTED={'nuscenes2d_temporal_infos_train_mini.pkl':'090cf34e5c41dfac7ad59c2c45557d44fcd2626f9239ad1135da593a5a15bca2','nuscenes2d_temporal_infos_val_mini.pkl':'bd2a2d812c30139af4c238ea5703fc77c943f358eac11b226cfc641d9f09b820'}
ROLES={'scene-0061':'train','scene-0553':'calibration','scene-0655':'attack_dev'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--runtime-data-root',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 raw= a.data_root/'v1.0-mini';samples={s['token']:s for s in json.loads((raw/'sample.json').read_text())};scenes={s['token']:s for s in json.loads((raw/'scene.json').read_text())};sd=json.loads((raw/'sample_data.json').read_text());keyimages={s['filename']:s for s in sd if s.get('is_key_frame')}
 out={r:[] for r in ROLES.values()};seen=set();origins=[];image_count=0
 for name,expected in TRUSTED.items():
  data=(ROOT/'assets'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
  payload=pickle.loads(data);origins.append({'file':name,'sha256':expected,'original_frames':len(payload['infos'])})
  for original in payload['infos']:
   token=original['token']
   if token not in samples:continue
   scene=scenes[samples[token]['scene_token']];role=ROLES.get(scene['name'])
   if role is None:continue
   assert token not in seen;seen.add(token);info=copy.deepcopy(original)
   assert info['scene_token']==scene['token'] and info['timestamp']==samples[token]['timestamp']
   assert len(info['cams'])==6
   for cam_name,cam in info['cams'].items():
    rel='samples/'+cam['data_path'].split('/samples/',1)[1];part=Path(rel)
    assert '..' not in part.parts and (a.data_root/part).is_file()
    sample_data=keyimages[rel];assert sample_data['sample_token']==token and sample_data['timestamp']==cam['timestamp']
    assert np.isfinite(cam['cam_intrinsic']).all() and np.isfinite(cam['sensor2lidar_rotation']).all()
    cam['data_path']=a.runtime_data_root.rstrip('/')+'/'+rel;image_count+=1
   if '/samples/' in info['lidar_path']:info['lidar_path']=a.runtime_data_root.rstrip('/')+'/samples/'+info['lidar_path'].split('/samples/',1)[1]
   for k in ['bboxes2d','labels2d','centers2d','depths','bboxes_ignore']:assert len(info[k])==6
   for i in range(6):
    assert len(info['bboxes2d'][i])==len(info['labels2d'][i])==len(info['centers2d'][i])==len(info['depths'][i])
    for k in ['bboxes2d','centers2d','depths']:assert np.isfinite(info[k][i]).all()
   assert np.isfinite(info['gt_boxes']).all();out[role].append(info)
 assert len(seen)==sum(1 for s in samples.values() if scenes[s['scene_token']]['name'] in ROLES)
 a.output_dir.mkdir(parents=True,exist_ok=True);reports=[]
 for role,infos in out.items():
  infos.sort(key=lambda x:x['timestamp']);tokens={i['token'] for i in infos}
  for info in infos:
   assert not info['prev'] or info['prev'] in tokens
   assert not info['next'] or info['next'] in tokens
  path=a.output_dir/('streampetr_'+role+'.pkl');path.write_bytes(pickle.dumps({'infos':infos,'metadata':{'version':'v1.0-mini'}},protocol=4))
  reports.append({'role':role,'path':str(path),'frames':len(infos),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 report={'status':'passed_metadata_and_image_presence','source':origins,'roles':reports,'verified_images':image_count,'native_compose_passed':False,'model_forward_passed':False,'gpu_used':False,'scope':'clean real data framework smoke only; LiDAR paths retained as metadata, point files not transferred','current_scenes':3,'formal_100_scene_protocol_ready':False}
 (a.output_dir/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
