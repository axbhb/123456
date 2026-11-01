"""CPU import repair in a task-local overlay; inherited packages remain read-only."""
import os,sys,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;overlay=root/'import_overlay';env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
wheels=[root/'wheels/huggingface_hub-0.25.2-py3-none-any.whl',root/'wheels/kornia-0.6.12-py2.py3-none-any.whl']
r={'started':time.time(),'gpu_used':False,'scope':'CPU import compatibility overlay, not a native validated environment'}
try:
 with (root/'overlay_install.log').open('w') as log:
  subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--disable-pip-version-check','--target',str(overlay)]+[str(p) for p in wheels],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
 env['PYTHONPATH']=str(overlay)
 with (root/'overlay_imports.log').open('w') as log:
  subprocess.run([sys.executable,str(root/'cpu_framework_probe.py'),'--output-dir',str(root/'cpu_probe_overlay'),'--timeout','40'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=350)
 r['status']='complete'
except Exception as e:r.update(status='failed',error_type=type(e).__name__,error=str(e))
r['finished']=time.time();(root/'overlay_done.json').write_text(json.dumps(r,indent=2))
