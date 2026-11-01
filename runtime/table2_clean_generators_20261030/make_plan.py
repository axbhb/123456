from pathlib import Path
import pickle,json,collections
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';D=R/'datasets/streampetr_mini_scene_283_121'
W.mkdir(exist_ok=True)
x=pickle.load((D/'infos_train.pkl').open('rb'));test=pickle.load((D/'infos_test.pkl').open('rb'))
groups=collections.OrderedDict()
for i in sorted(x['infos'],key=lambda i:(i['scene_token'],i['timestamp'])):groups.setdefault(i['scene_token'],[]).append(i)
assert len(x['infos'])==283 and len(test['infos'])==121
assert not set(groups)&{i['scene_token'] for i in test['infos']}
plan={'train_frames':283,'test_frames':121,'train_scenes':7,'test_scenes':3,'training_seeds':[0,1,2],'training_iterations':283,'source':'normal synthesis only; all conditions from train split','generators':{}}
for gen,size,stride in [('panacea_plus',8,8),('magicdrive',11,10),('dd2',4,4)]:
 clips=[]
 for scene,rows in groups.items():
  starts=list(range(0,len(rows)-size+1,stride))
  if starts[-1]!=len(rows)-size:starts.append(len(rows)-size)
  seen=set()
  for start in starts:
   tokens=[i['token'] for i in rows[start:start+size]]
   export=[t for t in tokens if t not in seen];seen.update(tokens)
   clip={'index':len(clips),'scene_token':scene,'tokens':tokens,'export_tokens':export,'seed':20261030+len(clips)}
   f=W/'plans'/gen/('%03d.json'%len(clips));f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(clip,indent=2));clip['path']=str(f);clips.append(clip)
  assert len(seen)==len(rows)
 plan['generators'][gen]=clips
(W/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({k:len(v) for k,v in plan['generators'].items()}))
