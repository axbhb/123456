"""Wait without allocating CUDA, then run one already-authorized StreamPETR smoke."""
import os,json,time,subprocess,fcntl
from pathlib import Path
P=Path('/srv/table2/frameworks/20261028-v1');ROOT=P.parent.parent
state={'status':'waiting_eligible_gpu','started':time.time(),'deadline':time.time()+3600,'gpu_used':False}
def save():
 q=P/'streampetr_wait_status.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(q)
with (P/'streampetr_wait.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save()
 while time.time()<state['deadline']:
  ledger=ROOT/'frameworks/gpu_budget_20261028.json';budget=json.loads(ledger.read_text()) if ledger.exists() else {'jobs':[]}
  if sum(x['charged_seconds'] for x in budget['jobs'])+900>7200:state['status']='insufficient_remaining_budget';break
  if budget.get('active'):state['status']='another_guarded_job_active';save();time.sleep(60);continue
  values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],universal_newlines=True,timeout=10)
  free=[int(x) for x in values.split()];state.update(free_mib=free,checked=time.time());save()
  if max(free)<25600:time.sleep(60);continue
  args=['/home/user/gzx/envs/wmautoattack/bin/python',str(P/'run_gpu_smoke.py'),'--job','streampetr','--python',str(ROOT/'envs/streampetr_torch111_cu113/bin/python'),'--script',str(P/'streampetr_smoke.py'),'--seconds','900','--ledger',str(ledger),'--','--runtime',str(P/'runtime_sources/streampetr'),'--output',str(P/'streampetr_gpu_smoke_dynamicamp'),'--mode','gpu-smoke']
  state['status']='guard_running';save()
  result=subprocess.run(args,timeout=940);state.update(status='guard_finished',returncode=result.returncode,gpu_used=len(json.loads(ledger.read_text()).get('jobs',[]))>len(budget['jobs']));break
 else:state['status']='waiting_deadline_reached'
 state['finished']=time.time();save()
