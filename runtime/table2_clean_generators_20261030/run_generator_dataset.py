import os,sys,json,time,subprocess,fcntl,traceback,argparse
p=argparse.ArgumentParser();p.add_argument("--generator",choices=["magicdrive","dd2"],required=True);p.add_argument("--limit",type=int,default=0);args=p.parse_args()
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=R/'outputs/table2_clean_mini_283_121'/args.generator;O.mkdir(parents=True,exist_ok=True)
state={'status':'waiting_for_existing_gpu_lock','started':time.time(),'completed_clips':0}
def save(**kw):
 state.update(kw);p=O/'generation_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
def run(cmd,log,env):
 with log.open('a') as f:subprocess.run([str(c) for c in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
save()
try:
 with (R/'frameworks/gpu_budget_20261028.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
  gpu=max([x.split(',') for x in gpus.strip().splitlines()],key=lambda x:int(x[1]))[0].strip()
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PHYSCOND_GPU_GUARD='1')
  save(status='running',gpu=gpu)
  plan=json.loads((W/'plan.json').read_text())
  for clip in plan['generators'][args.generator][:args.limit or None]:
   dest=O/'clips'/('%03d'%clip['index']);dest.mkdir(parents=True,exist_ok=True)
   result=dest/'result.json'
   previous=json.loads(result.read_text()) if result.exists() else {}
   same_backend=args.generator!='magicdrive' or previous.get('attention_backend')=='xformers_cutlass'
   if previous.get('status')=='passed' and same_backend:
    state['completed_clips']+=1;continue
   if previous.get('status')=='passed' and not same_backend:
    # Retain the smoke-test measurement; the full dataset uses one backend.
    (dest/'attention_slicing_smoke_result.json').write_text(json.dumps(previous,indent=2))
   save(phase='preparing_native_conditions',clip_index=clip['index'],scene_token=clip['scene_token'])
   pre_env='frozendrive' if args.generator=='magicdrive' else 'dd2_gpu'
   pre_script='prepare_'+args.generator+'.py'
   meta=dest/(args.generator+'_input_status.json')
   batch=dest/('magicdrive_real_61frame_batch.pt' if args.generator=='magicdrive' else 'dd2_conditions.pt')
   if not meta.exists() or not batch.exists():run([R/('envs/'+pre_env+'/bin/python'),W/pre_script,'--clip',clip['path'],'--output',dest],dest/'prepare.log',env)
   save(phase='generating',clip_index=clip['index'])
   batch=dest/('magicdrive_real_61frame_batch.pt' if args.generator=='magicdrive' else 'dd2_conditions.pt')
   run([R/('envs/'+pre_env+'/bin/python'),W/('generate_'+args.generator+'.py'),'--gpu',gpu,'--input',batch,'--output',dest,'--seed',clip['seed']],dest/'generate.log',env)
   assert json.loads(result.read_text())['status']=='passed'
   batch.unlink()
   state['completed_clips']+=1;save()
  save(status='partial_generation_complete' if args.limit else 'generation_complete',finished=time.time())
except Exception as e:
 save(status='failed',error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
