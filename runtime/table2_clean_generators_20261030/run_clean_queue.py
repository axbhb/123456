"""Sequential generation and clean finetuning on the existing shared GPU lock."""
import os,sys,time,json,subprocess,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=R/'outputs/table2_clean_mini_283_121'
state={'status':'waiting_for_current_clip','steps':[],'started':time.time()}
def save():
 q=O/'queue_status.json';tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(q)
save()
# The previous supervisor finishes its in-flight clip and yields before another GPU job.
old_pid=json.loads((W/'queue_handoff.json').read_text())['pid']
while Path('/proc/%d'%old_pid).exists():time.sleep(5)
(W/'reorder_after_clip.json').unlink(missing_ok=True)
steps=[('magicdrive_smoke',['run_generator_dataset.py','--generator','magicdrive','--limit','1']),('dd2_smoke',['run_generator_dataset.py','--generator','dd2','--limit','1']),('panacea_plus_generate',['run_panacea_dataset.py']),('panacea_plus_train',['train_clean.py','--generator','panacea_plus']),('dd2_generate',['run_generator_dataset.py','--generator','dd2']),('dd2_train',['train_clean.py','--generator','dd2']),('magicdrive_generate',['run_generator_dataset.py','--generator','magicdrive']),('magicdrive_train',['train_clean.py','--generator','magicdrive'])]
for name,args in steps:
 if name.endswith('_train'):
  gen=args[-1];q=O/gen/'generation_status.json'
  if not q.exists() or json.loads(q.read_text()).get('status')!='generation_complete':
   state['steps'].append({'name':name,'status':'skipped_generation_incomplete'});save();continue
 state.update(status='running',current=name);save();start=time.time()
 with (W/(name+'_queue.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,str(W/args[0])]+args[1:],stdout=f,stderr=subprocess.STDOUT)
  state['child_pid']=p.pid;save();code=p.wait()
 state['steps'].append({'name':name,'exit_code':code,'seconds':time.time()-start});save()
state.update(status='queue_finished',finished=time.time());save()
