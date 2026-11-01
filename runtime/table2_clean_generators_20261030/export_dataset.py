import argparse,json,pickle,copy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--generator',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';D=R/'datasets/streampetr_mini_scene_283_121';O=R/'outputs/table2_clean_mini_283_121'/a.generator
from PIL import Image
import numpy as np
from mmcv import Config
plan=json.loads((W/'plan.json').read_text());source=pickle.load((D/'infos_train.pkl').open('rb'));test=pickle.load((D/'infos_test.pkl').open('rb'));expected={i['token'] for i in source['infos']};export={};matrix=None
if a.generator=='frozendrive':
 matrix=np.array([[.5,0.,0.],[0.,.5,-2.],[0.,0.,1.]])
 pairs=json.loads((O/'native_generation_pairs.json').read_text())
 for pair in pairs:
  tok=pair['token'];folder=O/'frames'/tok;res=folder/'result.json'
  if a.partial and not res.exists():
   pilot=O/'generation_pilot';rp=pilot/'result.json'
   if rp.exists() and json.loads(rp.read_text()).get('sample_token')==tok:folder=pilot;res=rp
  if not res.exists() or json.loads(res.read_text()).get('status')!='passed':
   if a.partial:continue
   raise RuntimeError('FrozenDrive frame incomplete: '+tok)
  order=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT'];paths={}
  for cam in order:
   q=folder/(cam+'.png')
   with Image.open(q) as im:
    assert im.size==(800,448);im.verify()
   paths[cam]=str(q)
  export[tok]=paths
else:
 prefix={'panacea_plus':'panacea','magicdrive':'magicdrive','dd2':'dd2'}[a.generator]
 stride={'panacea_plus':1,'magicdrive':6,'dd2':2}[a.generator]
 for clip in plan['generators'][a.generator]:
  folder=O/'clips'/('%03d'%clip['index']);res=folder/'result.json'
  if not res.exists() or json.loads(res.read_text()).get('status')!='passed':
   if a.partial:continue
   raise RuntimeError('Generation incomplete: '+str(folder))
  meta=json.loads((folder/(prefix+'_input_status.json')).read_text());order=meta['camera_order']
  H=np.array(meta['source_to_generated'])
  if matrix is None:matrix=H
  else:assert np.allclose(matrix,H)
  for tok in clip['export_tokens']:
   paths={};slot=clip['tokens'].index(tok)*stride
   for v,cam in enumerate(order):
    filename=('view%d_frame%03d.png' if a.generator=='magicdrive' else 'view%d_frame%02d.png')%(v,slot)
    q=folder/filename
    with Image.open(q) as im:im.verify()
    paths[cam]=str(q)
   assert tok not in export;export[tok]=paths
assert export and not set(export)&{i['token'] for i in test['infos']}
if not a.partial:assert set(export)==expected
infos=[]
for row in source['infos']:
 if row['token'] not in export:continue
 row=copy.deepcopy(row)
 for cam,value in row['cams'].items():value['data_path']=export[row['token']][cam]
 infos.append(row)
ann=O/('infos_train_partial.pkl' if a.partial else 'infos_train.pkl')
pickle.dump({'infos':infos,'metadata':source['metadata']},ann.open('wb'))
cfg=Config.fromfile(str(D/'streampetr_mini.py'),import_custom_modules=False);cfg.data.train.ann_file=str(ann)
cfg.data.train.pipeline.insert(1,dict(type='RestoreGeneratorCanvas',source_to_generated=matrix.tolist()))
cfg.custom_imports.imports.append('synthetic_camera_pipeline');cfg.work_dir=str(O/'train');cfg.dump(str(O/('streampetr_partial.py' if a.partial else 'streampetr.py')))
# Restoring the missing top strip does not fabricate camera content. Native training
# crops start below it at every configured resize (minimum y=900-256/0.38).
missing_top=max(0,-matrix[1,2]/matrix[1,1]);assert missing_top<900-256/0.38
report={'status':'partial_validation' if a.partial else 'ready','generator':a.generator,'frames':len(infos),'images':len(infos)*6,'training_tokens':list(export),'test_frames':121,'source_to_generated':matrix.tolist(),'geometry':'native pixels inverse-warped to original camera canvas before unchanged StreamPETR augmentation; original K, 3D and 2D labels retained','missing_top_original_pixels':missing_top,'missing_top_excluded_by_training_crop':True}
(O/('dataset_partial.json' if a.partial else 'dataset.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='training_tokens'}))
