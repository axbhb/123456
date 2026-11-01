"""CPU import/component checks; never assert that a learned model has run."""
import argparse,os,json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CHILD=r'''
import os,sys,json,importlib,time,traceback,copy
from pathlib import Path
cfg=json.loads(sys.argv[1]);root=Path(sys.argv[2]);out=Path(sys.argv[3]);started=time.time()
for p in reversed(cfg['python_paths']):sys.path.insert(0,str(root/'sources'/cfg['id']/p))
r={'framework':cfg['id'],'source_commit':cfg['commit'],'gpu_used':False,'weights_loaded':False,'model_forward_passed':False,'training_passed':False,'scope':'CPU import only','python':sys.version}
try:
 import torch
 def reject(*args,**kwargs):raise RuntimeError('GPU disabled for CPU framework check')
 torch.cuda._lazy_init=reject;torch.cuda.init=reject;torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 r['torch']=torch.__version__
 if not cfg['module']:r.update(status='blocked_no_released_implementation')
 else:
  m=importlib.import_module(cfg['module']);r.update(status='import_passed',module=cfg['module'],module_file=getattr(m,'__file__',None))
 r['cuda_initialized']=torch.cuda.is_initialized()
except Exception as e:r.update(status='blocked_import',error_type=type(e).__name__,reason=str(e),traceback=traceback.format_exc()[-5000:])
r['seconds']=time.time()-started;out.write_text(json.dumps(r,indent=2));print(json.dumps({'framework':cfg['id'],'status':r['status'],'reason':r.get('reason')}))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--timeout',type=int,default=40);a=p.parse_args()
 if not 1<=a.timeout<=60:raise ValueError('CPU timeout must be <=60 seconds per import')
 a.output_dir.mkdir(parents=True,exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1')
 rows=[]
 for cfg in json.loads((ROOT/'frameworks.json').read_text()):
  target=a.output_dir/(cfg['id']+'.json')
  try:
   with (a.output_dir/(cfg['id']+'.log')).open('w') as f:r=subprocess.run([sys.executable,'-c',CHILD,json.dumps(cfg),str(ROOT),str(target.resolve())],cwd=str(ROOT/'sources'/cfg['id']),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
   item=json.loads(target.read_text()) if target.exists() else {'framework':cfg['id'],'status':'blocked_process','returncode':r.returncode}
  except subprocess.TimeoutExpired:item={'framework':cfg['id'],'status':'blocked_import_timeout'}
  rows.append(item);(a.output_dir/'summary.json').write_text(json.dumps({'checks':rows,'gpu_used':False,'full_frameworks_verified':0},indent=2))
 print(json.dumps({'import_passed':sum(r['status']=='import_passed' for r in rows),'checks':len(rows),'gpu_used':False,'full_frameworks_verified':0}))
if __name__=='__main__':main()
