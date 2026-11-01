import os,subprocess,json,time,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parent;WD=W/'source/WorldDreamer';E=R/'envs/frozendrive';python=str(E/'bin/python');state={'gpu_used':False,'phase':'xformers_extensions'}
def save(): (W/'environment_status.json').write_text(json.dumps(state,indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONPATH='',OMP_NUM_THREADS='2',MAX_JOBS='2',BUILD_VERSION='0.0.19',CUDA_HOME='/usr/local/cuda-11.3',TORCH_CUDA_ARCH_LIST='7.0',FORCE_CUDA='1',XFORMERS_DISABLE_FLASH_ATTN='1',XFORMERS_BUILD_TYPE='Release')
try:
 save();xf=WD/'third_party/xformers'
 for p in (xf/'xformers/csrc').rglob('*'):
  if p.is_file() and p.suffix in ['.cpp','.cu','.h','.cuh']:
   a=p.read_text();b=a.replace('<ATen/CUDAGeneratorImpl.h>','<ATen/cuda/CUDAGeneratorImpl.h>')
   if a!=b:p.write_text(b)
 subprocess.run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(xf)],env=env,check=True)
 state['phase']='waiting_for_aux_dependencies';save()
 for _ in range(240):
  aux=json.loads((W/'aux_status.json').read_text()) if (W/'aux_status.json').exists() else {}
  if aux.get('phase')=='ready':break
  if aux.get('phase')=='failed':raise RuntimeError('Aux dependency installation failed')
  time.sleep(15)
 else:raise TimeoutError('Aux dependency installation timeout')
 state['phase']='import_validation';save()
 out=subprocess.run([python,'-c',"import runpy;runpy.run_path('tools/test_occ.py',run_name='frozendrive_import_test');import torch,xformers,diffusers;assert not torch.cuda.is_initialized();print('CPU_ENTRYPOINT_IMPORT_OK',torch.__version__,xformers.__version__,diffusers.__version__)"],cwd=str(WD),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
 (W/'entrypoint_import.log').write_text(out.stdout)
 if out.returncode:raise RuntimeError(out.stdout[-6000:])
 state.update(phase='ready',environment=str(E),source=str(WD),cuda_initialized=False);save()
 subprocess.Popen([python,str(W/'validate_cpu.py')],cwd=str(W),env=env,stdout=open(W/'cpu_validation.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
except Exception:state.update(phase='failed',error=traceback.format_exc());save();raise
