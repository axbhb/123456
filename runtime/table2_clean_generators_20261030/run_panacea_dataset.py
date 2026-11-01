import os,sys,json,time,subprocess,fcntl,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=R/'outputs/table2_clean_mini_283_121/panacea_plus';O.mkdir(parents=True,exist_ok=True)
state={'status':'waiting_for_existing_gpu_lock','started':time.time(),'completed_clips':0}
def save(**kw):
 state.update(kw);p=O/'generation_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
def run(cmd,log,env):
 with log.open('a') as f:subprocess.run([str(c) for c in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
save()
try:
 plan=json.loads((W/'plan.json').read_text())
 for clip in plan['generators']['panacea_plus']:
  dest=O/'clips'/('%03d'%clip['index']);dest.mkdir(parents=True,exist_ok=True);result=dest/'result.json'
  if result.exists() and json.loads(result.read_text()).get('status')=='passed':state['completed_clips']+=1;continue
  env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  save(status='running',phase='preparing_native_conditions',clip_index=clip['index'],scene_token=clip['scene_token'])
  run([R/'envs/streampetr_torch111_cu113/bin/python',W/'prepare_panacea.py','--clip',clip['path'],'--output',dest],dest/'prepare.log',env)
  save(phase='waiting_for_existing_gpu_lock')
  with (R/'frameworks/gpu_budget_20261028.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
   gpu=max([x.split(',') for x in gpus.strip().splitlines()],key=lambda x:int(x[1]))[0].strip();env['CUDA_VISIBLE_DEVICES']=gpu
   save(phase='generating',gpu=gpu)
   run([R/'envs/panacea_plus/bin/python',W/'generate_panacea.py','--gpu',gpu,'--input',dest/'panacea_real_8frame_batch.pt','--output',dest,'--seed',clip['seed']],dest/'generate.log',env)
  assert json.loads(result.read_text())['status']=='passed'
  (dest/'panacea_real_8frame_batch.pt').unlink();state['completed_clips']+=1;save()
 save(status='generation_complete',finished=time.time())
except Exception as e:
 save(status='failed',error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
