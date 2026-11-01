from common import *
import subprocess
method=sys.argv[1];out=C/('seed%d'%SEED) if method=='clean' else O/method;out.mkdir(parents=True,exist_ok=True);state=dict(method=method,generator=G,selected_frames=0 if method=='clean' else N,training_seed=SEED,planned_iterations=ITERS,status='starting',stages={},started=time.time())
def save(**kw):state.update(kw);state['elapsed_seconds']=time.time()-state['started'];atomic(out/'pipeline_status.json',state)
def stage(name,script,args=(),guard=True,python=None):
 save(status=name);python=str(python or P);cmd=[python,str(script)]+list(map(str,args))
 if guard and os.environ.get('TABLE2_USE_EXISTING_GUARD','1' if (R/'frameworks/gpu_budget_20261028.json').exists() else '0')=='1':cmd=[str(P),str(W/'run_existing_guard.py'),'--job','streampetr','--python',python,'--script',str(script),'--seconds','7200','--ledger',str(R/'frameworks/gpu_budget_20261028.json'),'--']+list(map(str,args))
 env=os.environ.copy();env['PYTHONPATH']=':'.join(map(str,[W,T,D,S]))+':'+env.get('PYTHONPATH','');env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');log=out/(name+'.log');waiting=time.time()
 while True:
  offset=log.stat().st_size if log.exists() else 0;t=time.time()
  with log.open('a') as f:p=subprocess.run(cmd,env=env,cwd=str(W),stdout=f,stderr=subprocess.STDOUT)
  if p.returncode==0:break
  with log.open() as f:f.seek(offset);tail=f.read()[-5000:]
  if 'BlockingIOError' in tail or 'An unfinished GPU job requires reconciliation' in tail:save(status='waiting_for_existing_gpu_lock',pending_stage=name);time.sleep(20);continue
  raise RuntimeError(name+' failed: '+tail)
 state['stages'][name]=dict(seconds=time.time()-t,including_wait_seconds=time.time()-waiting,log=str(log));save()
try:
 if method=='clean':
  cfg,_=load_training();cfg.runner.max_iters=ITERS;cfg.checkpoint_config.interval=ITERS;cfg.work_dir=str(out/'train');cfg.dump(str(out/'streampetr.py'));os.environ['TABLE2_OUTPUT']=str(C)
 if method!='clean' and not (O/'selection.json').exists():stage('prepare',W/'prepare.py',guard=False)
 if method in ['bait','pil'] and not (O/'roi_training.npz').exists():stage('roi_prepare',W/'prepare_roi.py',guard=False)
 if method=='asymmetry' and not (out/'geometry.json').exists():
  stage('physical_prepare',W/'prepare_physical.py',guard=False);stage('physical_anchor_retry',W/'retry_physical_anchors.py',guard=False)
 attack=out/'attack_status.json'
 if method!='clean' and (not attack.exists() or json.loads(attack.read_text()).get('status')!='completed'):stage('attack',W/('attack_'+method+'.py'),python=R/'envs/dd2_gpu/bin/python' if method=='advdiff' else P)
 from mmcv import Config
 train_cfg=Config.fromfile(str(out/'streampetr.py'),import_custom_modules=False);train_cfg.runner.max_iters=ITERS;train_cfg.checkpoint_config.interval=ITERS;train_cfg.dump(str(out/'streampetr.py'))
 ckpt=out/('train/iter_%d.pth'%ITERS);trained_now=False
 if not ckpt.exists():stage('train',W/'run_train.py',['seed%d'%SEED if method=='clean' else method]);trained_now=True
 stage('audit',W/'audit_training.py',['--output',out],False)
 prior_result=json.loads((out/'after/result.json').read_text()) if (out/'after/result.json').exists() else {}
 if trained_now or prior_result.get('checkpoint')!=str(ckpt):stage('evaluate',W/'evaluate_clean.py',['--checkpoint',ckpt,'--output',out/'after'])
 result=json.loads((out/'after/result.json').read_text());audit=json.loads((out/'training_audit.json').read_text());assert result['status']=='completed' and audit['all_model_weights_finite'];attack=dict(method='Clean',seconds=0) if method=='clean' else json.loads((out/'attack_status.json').read_text())
 summary=dict(status='completed',method=attack['method'],generator=G,selected_frames=0 if method=='clean' else N,training_seed=SEED,planned_iterations=ITERS,test_frames=result['frames'],validation_scope=os.environ.get('TABLE2_SCOPE','configured experiment'),native_metrics=result['metrics'],training_audit=audit,stage_timings=state['stages'],attack_seconds=attack['seconds'],checkpoint=str(ckpt));
 clean_path=C/('seed%d'%SEED)/'after/result.json'
 if method!='clean' and clean_path.exists():
  clean=json.loads(clean_path.read_text());summary['paired_clean_result']=str(clean_path);summary['metrics']={}
  for label,key in [('mAP','pts_bbox_NuScenes/mAP'),('NDS','pts_bbox_NuScenes/NDS')]:
   cm=clean['metrics'][key];am=result['metrics'][key];summary['metrics'][label]=dict(clean_percent=100*cm,adapted_percent=100*am,ASR_percent=100*(cm-am)/cm if cm else None)
 summary.update(train_frames=len(pickle.load((C/'infos_train.pkl').open('rb'))['infos']),dataset_directory=str(D),clean_dataset=str(C),selection_seed=SELECTION_SEED,attack_seed=ATTACK_SEED,completed_at=time.time())
 summary['replacement_percentage']=100*summary['selected_frames']/summary['train_frames']
 summary['adapted']=method!='clean';atomic(out/'summary.json',summary);save(status='completed',finished=time.time());print(json.dumps(summary))
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc());raise
