"""Resume the existing MagicDrive clips, then run the three Clean seeds."""
import sys,json,time,subprocess,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=R/'outputs/table2_clean_mini_283_121'
state={'status':'running','started':time.time(),'steps':[]}
def save(**kw):
 state.update(kw);p=O/'magicdrive_resume_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
try:
 for script,phase in [('run_generator_dataset.py','generation'),('train_clean.py','finetune')]:
  save(phase=phase)
  with (W/('magicdrive_resume_'+phase+'.log')).open('a') as f:
   child=subprocess.Popen([sys.executable,str(W/script),'--generator','magicdrive'],stdout=f,stderr=subprocess.STDOUT)
   save(child_pid=child.pid);code=child.wait()
  state['steps'].append({'phase':phase,'exit_code':code});save()
  if code:raise RuntimeError('%s exited %d'%(phase,code))
 save(status='completed',finished=time.time())
except Exception as e:
 save(status='failed',error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
