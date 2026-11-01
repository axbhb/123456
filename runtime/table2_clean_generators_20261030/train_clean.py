import os,sys,json,time,subprocess,fcntl,traceback,socket,argparse,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--generator',required=True);a=p.parse_args()
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';D=R/'datasets/streampetr_mini_scene_283_121';O=R/'outputs/table2_clean_mini_283_121'/a.generator;S=R/'frameworks/20261028-v1/runtime_sources/streampetr';P=R/'envs/streampetr_torch111_cu113/bin/python'
state={'status':'waiting_for_generation','generator':a.generator,'training_seeds':[0,1,2],'iterations':283,'started':time.time(),'completed_seeds':[]}
def save(**kw):
 state.update(kw);q=O/'finetune_status.json';tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(q)
def run(cmd,log,env):
 with log.open('a') as f:subprocess.run([str(c) for c in cmd],cwd=str(S),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
save()
try:
 while True:
  q=O/'generation_status.json'
  if q.exists():
   status=json.loads(q.read_text()).get('status')
   if status=='generation_complete':break
   if status=='failed':raise RuntimeError('Generator failed; inspect generation_status.json before resuming')
  time.sleep(30)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(W)+':'+str(D)+':'+str(S)+':'+env.get('PYTHONPATH',''))
 run([P,W/'export_dataset.py','--generator',a.generator],O/'export.log',env)
 save(status='waiting_for_existing_gpu_lock')
 with (R/'frameworks/gpu_budget_20261028.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
  gpu=max([x.split(',') for x in gpus.strip().splitlines()],key=lambda x:int(x[1]))[0].strip();env['CUDA_VISIBLE_DEVICES']=gpu
  for seed in [0,1,2]:
   out=O/('seed%d'%seed);out.mkdir(exist_ok=True);res=out/'after/result.json'
   if res.exists() and json.loads(res.read_text()).get('status')=='completed':state['completed_seeds'].append(seed);continue
   save(status='finetuning',seed=seed,gpu=gpu)
   checkpoint=out/'train/iter_283.pth'
   if not checkpoint.exists():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    run([P,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--master_addr=127.0.0.1','--master_port='+str(port),'tools/train.py',O/'streampetr.py','--launcher','pytorch','--seed',seed,'--no-validate','--work-dir',out/'train','--cfg-options','log_config.interval=10'],out/'train.log',env)
   run([P,W/'audit_training.py','--output',out],out/'audit.log',env)
   save(status='evaluating_real_test',seed=seed)
   run([P,W/'evaluate_clean.py','--checkpoint',checkpoint,'--output',out/'after'],out/'evaluate.log',env)
   state['completed_seeds'].append(seed);save()
 metrics=[]
 for seed in [0,1,2]:
  row=json.loads((O/('seed%d'%seed)/'after/result.json').read_text());m=row['metrics'];metrics.append({'seed':seed,'mAP':100*m['pts_bbox_NuScenes/mAP'],'NDS':100*m['pts_bbox_NuScenes/NDS'],'training_audit':json.loads((O/('seed%d'%seed)/'training_audit.json').read_text())})
 summary={'status':'completed','generator':a.generator,'train_frames':283,'test_frames':121,'seeds':metrics,'metrics':{k:{'mean':statistics.mean([r[k] for r in metrics]),'std':statistics.stdev([r[k] for r in metrics])} for k in ['mAP','NDS']},'ASR_mAP':0.0 if all(r['mAP']>0 for r in metrics) else None,'ASR_NDS':0.0 if all(r['NDS']>0 for r in metrics) else None,'protocol':'mini pilot; 7 training scenes / 3 real test scenes; normal synthetic data only; 283 scheduled finetuning iterations per seed','limitations':['Pretrained checkpoint may have seen these mini scenes.','Small custom split; not the 100-synthetic-scene paper-scale experiment.']}
 (O/'clean_summary.json').write_text(json.dumps(summary,indent=2));save(status='completed',finished=time.time())
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc());raise
