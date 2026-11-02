"""Prepare every fixed training keyframe, retaining native reconstruction settings."""
import os,sys,json,subprocess,time,traceback
from pathlib import Path
R=Path('/srv/table2');F=Path(__file__).parent
O=R/'outputs/table2_clean_mini_283_121/frozendrive';C=O/'conditions';C.mkdir(exist_ok=True);(C/'projected').mkdir(exist_ok=True);(C/'frames').mkdir(exist_ok=True)
state={'status':'running','started':time.time(),'completed_frames':0,'total_frames':283}
def save(**kw):
 state.update(kw);p=C/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
try:
 pilot=json.loads((O/'condition_pilot/status.json').read_text())
 if pilot.get('status')!='passed':raise RuntimeError('First real condition pilot is not complete')
 pairs=json.loads((O/'native_generation_pairs.json').read_text());assert len(pairs)==283 and len({p['token'] for p in pairs})==283
 for i,pair in enumerate(pairs):
  token=pair['token'];dest=O/'condition_pilot' if token==pilot['sample_token'] else C/'frames'/token
  dest.mkdir(exist_ok=True);s=dest/'status.json'
  old=json.loads(s.read_text()) if s.exists() else {}
  save(phase='frame_preparation',index=i,sample_token=token,frame_output=str(dest))
  if old.get('status')!='passed':
   with (dest/'supervisor.log').open('a') as log:
    subprocess.run([sys.executable,'-u',str(F/'run_nksr_pilot.py'),'--sample-token',token,'--output',str(dest),'--skip-native-batch'],stdout=log,stderr=subprocess.STDOUT,check=True)
  validated=json.loads((dest/'condition_validation.json').read_text());assert validated['status']=='passed' and validated['sample_token']==token and len(validated['cameras'])==6
  actual=dest/'projected'/('sample_'+token);assert len(list(actual.glob('proj_CAM_*.npz')))==6
  link=C/'projected'/actual.name
  if not link.exists():link.symlink_to(actual,target_is_directory=True)
  assert link.resolve()==actual.resolve()
  save(completed_frames=i+1)
 save(status='completed',phase='all_training_conditions_ready',finished=time.time(),scope='283 training keyframe conditions; generation and downstream training still pending')
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
