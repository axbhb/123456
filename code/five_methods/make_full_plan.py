from common import *
import collections
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=D);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=pickle.load((a.dataset/'infos_train.pkl').open('rb'));test=pickle.load((a.dataset/'infos_test.pkl').open('rb'));groups=collections.OrderedDict()
for row in sorted(source['infos'],key=lambda x:(x['scene_token'],x['timestamp'])):groups.setdefault(row['scene_token'],[]).append(row)
assert not set(groups)&{x['scene_token'] for x in test['infos']};plan=dict(train_frames=len(source['infos']),test_frames=len(test['infos']),train_scenes=len(groups),test_scenes=len({x['scene_token'] for x in test['infos']}),generators={});a.output.mkdir(parents=True,exist_ok=True)
for generator,size,stride in [('dd2',4,4),('magicdrive',11,10),('panacea_plus',8,8)]:
 clips=[]
 for scene,rows in groups.items():
  if len(rows)<size:raise ValueError(generator+' needs a complete native clip; scene '+scene+' has '+str(len(rows))+' frames')
  starts=list(range(0,len(rows)-size+1,stride))
  if starts[-1]!=len(rows)-size:starts.append(len(rows)-size)
  seen=set()
  for start in starts:
   tokens=[x['token'] for x in rows[start:start+size]];export=[t for t in tokens if t not in seen];seen.update(tokens);record=dict(index=len(clips),scene_token=scene,tokens=tokens,export_tokens=export,seed=20261030+len(clips));path=a.output/'plans'/generator/('%03d.json'%len(clips));path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2));record['path']=str(path);clips.append(record)
  assert len(seen)==len(rows)
 plan['generators'][generator]=clips
atomic(a.output/'plan.json',plan);print({g:len(v) for g,v in plan['generators'].items()})
