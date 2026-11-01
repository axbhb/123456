"""Wait for verified assets; CPU-load first, then one authorized native clip through shared GPU guard."""
import os,time,json,subprocess,fcntl
from pathlib import Path
P=Path('/srv/table2/frameworks/20261028-v1');R=P.parent.parent;deadline=time.time()+21600;s={'status':'waiting_verified_assets','started':time.time(),'deadline':deadline,'gpu_launched':False}
def save():
 q=P/'dd2_native_wait_status.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(q)
with (P/'dd2_native_wait.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
 try:
  while time.time()<deadline:
   rows=json.loads((P/'dd2_required_weights_manifest.json').read_text());missing=[x['relative'] for x in rows if not (R/'assets'/x['relative']).is_file() or (R/'assets'/x['relative']).stat().st_size!=x['bytes']]
   ready=(P/'load_merged_dd2_cpu.py').exists() and json.loads((P/'dd2_native_conditions_status.json').read_text()).get('status')=='native_conditions_ready'
   s.update(missing_assets=missing,conditions_ready=ready,checked=time.time());save()
   if not missing and ready:break
   time.sleep(60)
  else:raise TimeoutError('Asset wait deadline reached; keep resumable transfer')
  py=R/'envs/dd2_gpu/bin/python';args=[str(py),str(P/'dd2_native_smoke.py'),'--mode','cpu-load','--output',str(P/'dd2_native_cpu_load')]
  s['status']='cpu_load_preflight';save()
  result=subprocess.run(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1'),timeout=300)
  if result.returncode:raise RuntimeError('Strict CPU native loading failed; inspect saved result')
  s['status']='waiting_gpu';save();gpu_deadline=min(deadline,time.time()+3600)
  while time.time()<gpu_deadline:
   ledger=R/'frameworks/gpu_budget_20261028.json';b=json.loads(ledger.read_text())
   if sum(j['charged_seconds'] for j in b['jobs'])+1800>7200:raise RuntimeError('Insufficient shared budget for configured native clip')
   if b.get('active'):time.sleep(60);continue
   free=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],universal_newlines=True,timeout=10).split()]
   s.update(free_mib=free,checked=time.time());save()
   if max(free)>=25600:break
   time.sleep(60)
  else:raise TimeoutError('No eligible free GPU before waiting deadline')
  args=['/home/user/gzx/envs/wmautoattack/bin/python',str(P/'run_gpu_smoke.py'),'--job','drivedreamer2','--python',str(py),'--script',str(P/'dd2_native_smoke.py'),'--seconds','1800','--ledger',str(ledger),'--','--mode','gpu-native','--role','attack_development','--output',str(P/'dd2_native_gpu_smoke')]
  s['status']='guard_running';save();result=subprocess.run(args,timeout=1840);after=json.loads(ledger.read_text());s.update(status='guard_finished',returncode=result.returncode,gpu_launched=len(after['jobs'])>len(b['jobs']))
 except Exception as e:s.update(status='stopped_review_status',error_type=type(e).__name__,error=str(e))
 s['finished']=time.time();save()
