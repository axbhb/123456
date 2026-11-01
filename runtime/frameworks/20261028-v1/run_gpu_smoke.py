"""Single-GPU launcher with shared usage accounting and configurable cumulative allowance. No GPU used by --check-only."""
import argparse,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
ALLOWED={'drivedreamer','drivedreamer2','magicdrive','panacea','streampetr','vad','tracking'}
def eligible(rows):
 return sorted([r for r in rows if r['free_mib']>0],key=lambda r:r['free_mib'],reverse=True)
def remaining(ledger,limit):
 if ledger.get('active'):raise RuntimeError('An unfinished GPU job requires reconciliation before another launch')
 used=sum(float(r['charged_seconds']) for r in ledger.get('jobs',[]))
 return None if limit is None else max(0.0,float(limit)-used)
def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
CHILD=r'''
import os,sys,runpy,json
from pathlib import Path
assert os.environ.get('PHYSCOND_GPU_GUARD')=='1'
import torch
assert torch.cuda.device_count()==1
# Fixed free-memory floor and per-process cap removed by user on 2026-10-29.
# Use the GPU with most currently free memory; actual allocations determine fit.
script=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',choices=sorted(ALLOWED),required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--script',type=Path);p.add_argument('--seconds',type=int,default=900);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--check-only',action='store_true');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
 if a.args and a.args[0]=='--':a.args=a.args[1:]
 if not 7<=a.seconds<=7200:raise ValueError('Invalid per-job budget')
 root=Path('/srv/table2').resolve()
 canonical=root/'frameworks/gpu_budget_20261028.json'
 if a.ledger.resolve()!=canonical:raise ValueError('Every framework must share the canonical GPU ledger')
 auth=json.loads((root/'frameworks/20261028-v1/gpu_authorization.json').read_text())
 if auth.get('approved') is not True or 'total_gpu_seconds' not in auth:raise RuntimeError('Missing GPU authorization record')
 limit=auth['total_gpu_seconds']
 if limit is not None and (not isinstance(limit,(int,float)) or limit<0):raise ValueError('Invalid cumulative GPU allowance')
 a.ledger.parent.mkdir(parents=True,exist_ok=True)
 import fcntl
 with a.ledger.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=json.loads(a.ledger.read_text()) if a.ledger.exists() else {'jobs':[],'active':None,'total_budget_seconds':limit}
  state['total_budget_seconds']=limit
  left=remaining(state,limit)
  if left is not None and a.seconds>left:raise RuntimeError('Requested job exceeds remaining shared GPU budget')
  output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits'],universal_newlines=True,timeout=10)
  rows=[]
  for line in output.strip().splitlines():
   ident,total,free=[x.strip() for x in line.split(',')];rows.append({'uuid':ident,'total_mib':int(total),'free_mib':int(free)})
  choices=eligible(rows)
  if not choices:raise RuntimeError('No GPU has free memory; nothing launched')
  if a.check_only:print(json.dumps({'eligible':choices,'remaining_seconds':left,'gpu_initialized':False}));return
  if not a.script or not a.script.is_file():raise ValueError('A prepared model script is required')
  root=Path('/srv/table2').resolve()
  if root not in a.script.resolve().parents or root not in a.ledger.resolve().parents:raise ValueError('Script and ledger must stay in the task directory')
  gpu=choices[0];job={'id':uuid.uuid4().hex,'framework':a.job,'started':time.time(),'allowed_seconds':a.seconds,'gpu':gpu,'script':str(a.script.resolve())}
  state['active']=job;atomic(a.ledger,state);proc=None;started=time.monotonic()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PHYSCOND_GPU_GUARD='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  def interrupt(signum,frame):raise KeyboardInterrupt('Supervisor interrupted')
  signal.signal(signal.SIGTERM,interrupt)
  try:
   logpath=a.ledger.parent/(job['id']+'.log');job['log']=str(logpath)
   with logpath.open('w') as log:
    proc=subprocess.Popen([a.python,'-c',CHILD,str(a.script.resolve())]+a.args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    job['pid']=proc.pid;atomic(a.ledger,state)
    try:job['returncode']=proc.wait(timeout=max(.1,a.seconds-6-(time.monotonic()-started)));job['status']='passed_process' if job['returncode']==0 else 'failed_process'
    except subprocess.TimeoutExpired:job['status']='time_budget_stopped'
  except BaseException as e:job.update(status='interrupted',error_type=type(e).__name__)
  finally:
   if proc and proc.poll() is None:
    if os.getpgid(proc.pid)!=proc.pid:raise RuntimeError('Refusing to signal an unowned process group')
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
   job.update(charged_seconds=time.monotonic()-started,finished=time.time());state['jobs'].append(job);state['active']=None;atomic(a.ledger,state)
  print(json.dumps(job))
  if job['status']!='passed_process':sys.exit(2)
if __name__=='__main__':main()
