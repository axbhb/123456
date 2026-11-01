"""Build a task-local Torch 1.11 CUDA compatibility environment without modifying its source env."""
import os,sys,json,time,shutil,subprocess,hashlib,signal
from pathlib import Path
ROOT=Path('/srv/table2');STAGE=ROOT/'frameworks/20261028-v1';DEST=ROOT/'envs/streampetr_torch111_cu113';BASE=Path('/home/user/anaconda3/envs/GNN_env/bin/python');CPU=ROOT/'envs/bevformer_cpu/lib/python3.9/site-packages';WHEEL=ROOT/'assets/streampetr_cuda111/mmcv_full-1.7.0-cp39-cp39-manylinux1_x86_64.whl';HASH='9f4875aa8af6924913b28095c979ddccc793f17e4033eb5631dc85101657b174'
status={'status':'starting','started':time.time(),'deadline':time.time()+3600,'gpu_used':False,'base_environment_read_only':str(BASE.parent.parent),'destination':str(DEST),'kind':'Torch1.11 CUDA compatibility environment; inherited legacy dependency deviations must remain disclosed'}
def save():
 p=STAGE/'cuda_environment_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2));t.replace(p)
def run(name,args,timeout=60):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1',PIP_NO_INDEX='1')
 with (STAGE/(name+'.log')).open('w') as log:r=subprocess.run([str(x) for x in args],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 if r.returncode:raise RuntimeError(name+' failed; inspect '+name+'.log')
 return (STAGE/(name+'.log')).read_text()
save()
try:
 if not DEST.exists():
  run('cuda_env_create',[BASE,'-m','venv','--system-site-packages',DEST]);(DEST/'physcond_owned.json').write_text(json.dumps({'base':str(BASE),'created':time.time()}))
 elif not (DEST/'physcond_owned.json').exists():raise RuntimeError('Refusing to modify an unmarked environment')
 site=DEST/'lib/python3.9/site-packages';skip=('torch','functorch','caffe2','numpy','scipy','pil','pillow','mmcv','pip','setuptools','pkg_resources','_distutils_hack','wheel')
 status['status']='copying_cpu_compatible_dependencies';save();copied=[]
 for p in CPU.iterdir():
  if p.name.lower().startswith(skip) or p.name.endswith('.pth') or p.name=='__pycache__':continue
  if p.name=='mmdet3d' and list(p.rglob('*.so')):raise RuntimeError('mmdet3d has compiled Torch-dependent extensions; rebuild required')
  q=site/p.name
  if p.is_dir():shutil.copytree(p,q,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  elif p.is_file():shutil.copy2(p,q)
  copied.append(p.name)
 status['copied_packages']=copied;status['status']='waiting_mmcv_wheel';save()
 while not WHEEL.exists() and time.time()<status['deadline']:time.sleep(5)
 if not WHEEL.exists():raise TimeoutError('MMCV wheel unavailable before deadline')
 if hashlib.sha256(WHEEL.read_bytes()).hexdigest()!=HASH:raise ValueError('MMCV checksum mismatch')
 status['status']='installing_cuda_mmcv';save();py=DEST/'bin/python'
 run('cuda_mmcv_install',[py,'-m','pip','install','--no-index','--no-deps',WHEEL])
 check="import sys,json,torch,mmcv,mmdet,mmdet3d;import mmcv.ops;assert torch.__version__=='1.11.0+cu113';assert not torch.cuda.is_initialized();print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'mmcv':mmcv.__version__,'mmdet':mmdet.__version__,'mmdet3d':mmdet3d.__version__,'torch_file':torch.__file__,'cuda_initialized':False}))"
 output=run('cuda_env_imports',[py,'-c',check]);status['imports']=json.loads(output.strip().splitlines()[-1])
 run('cuda_env_checkpoint_preflight',[py,STAGE/'streampetr_smoke.py','--runtime',STAGE/'runtime_sources/streampetr','--output',STAGE/'streampetr_cuda_env_preflight','--mode','cpu-preflight'],90)
 status['preflight']=json.loads((STAGE/'streampetr_cuda_env_preflight/result.json').read_text());status['status']='ready_for_guarded_gpu_smoke'
except Exception as e:status.update(status='blocked',error_type=type(e).__name__,error=str(e))
status['finished']=time.time();save();print(json.dumps(status))
