import os,sys,subprocess,json,tarfile,traceback,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parent;S=W/'source';E=R/'envs/frozendrive';state={'gpu_used':False,'phase':'extract'}
def save(): (W/'environment_status.json').write_text(json.dumps(state,indent=2))
def run(args,cwd=None):
 print('RUN',args,flush=True);subprocess.run(args,cwd=cwd,env=env,check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONPATH='',OMP_NUM_THREADS='2',MAX_JOBS='2',BUILD_VERSION='0.0.0',CUDA_HOME='/usr/local/cuda-11.3',TORCH_CUDA_ARCH_LIST='7.0',FORCE_CUDA='1')
try:
 save();S.mkdir(exist_ok=True)
 with tarfile.open(W/'runtime_source.tar.gz') as tf:
  for m in tf.getmembers():
   (S/m.name).resolve().relative_to(S.resolve())
  tf.extractall(S)
 state['phase']='environment';save()
 base=R/'envs/streampetr_torch111_cu113'
 if not E.exists():run([str(base/'bin/python'),'-m','venv','--system-site-packages',str(E)])
 site=E/'lib/python3.9/site-packages'
 (site/'frozendrive_existing_dependencies.pth').write_text(str(base/'lib/python3.9/site-packages')+'\n')
 python=str(E/'bin/python');WD=S/'WorldDreamer'
 wheels=R/'generator_deployment_20261028/wheelhouse'
 run([python,'-m','pip','install','--no-index','--no-deps','--find-links',str(wheels),'--find-links',str(R/'frameworks/20261028-v1/cpu_extra_wheels'),'--find-links',str(R/'model_readiness_20261028/preparation/wheelhouse'),'huggingface_hub==0.25.2','transformers==4.27.4','accelerate==0.20.3','hydra-core==1.3.2','open-clip-torch==2.20.0','kornia==0.6.9','torchpack==0.3.1'])
 # Reuse missing auxiliary packages read-only; torch and compiled mmcv stay in the compatible base environment.
 import glob
 other=R/'envs/dd2_gpu/lib/python3.9/site-packages'
 for name in ['imageio','imageio_ffmpeg','decorator','proglog','h5py','h5py.libs','ftfy','regex','safetensors','omegaconf','antlr4','torchdiffeq','tensorpack','msgpack_numpy','zmq','mpi4py','torchpack','tensorboardX','termcolor','yacs','loguru']:
  candidates=[other/name,Path('/home/user/gzx/envs/wmautoattack/lib/python3.9/site-packages')/name,R/'envs/magicdrive_video/lib/python3.9/site-packages'/name]
  for src in candidates:
   if src.exists() and not (site/name).exists():
    shutil.copytree(str(src),str(site/name)) if src.is_dir() else shutil.copy2(str(src),str(site/name));break
 run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(WD/'third_party/diffusers')])
 # PyTorch 1.11 removed THC.h; these six sources only contain unused legacy includes/declarations.
 for src in (WD/'third_party/bevfusion/mmdet3d/ops').rglob('*.cpp'):
  old=src.read_text();new=old.replace('#include <THC/THC.h>','#include <ATen/cuda/CUDAContext.h>').replace('extern THCState *state;','')
  if old!=new:src.write_text(new)
 state['phase']='bevfusion_extensions';save()
 run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(WD/'third_party/bevfusion')])
 state['phase']='xformers_extensions';save()
 with tarfile.open(W/'xformers_build_headers.tar.gz') as tf:tf.extractall(S)
 xf=WD/'third_party/xformers'
 cutlass=xf/'third_party/cutlass'
 if not cutlass.exists():cutlass.symlink_to(next((xf/'third_party').glob('cutlass-*')))
 env['XFORMERS_DISABLE_FLASH_ATTN']='1';env['BUILD_VERSION']='0.0.19';env['XFORMERS_BUILD_TYPE']='Release'
 run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(xf)])
 state['phase']='import_validation';save()
 run([python,'-c',"import runpy;runpy.run_path('tools/test_occ.py',run_name='frozendrive_import_test');import torch;assert not torch.cuda.is_initialized();print('CPU_ENTRYPOINT_IMPORT_OK')"],str(WD))
 state.update(phase='ready',environment=str(E),source=str(WD));save()
except Exception:
 state.update(phase='failed',error=traceback.format_exc());save();raise
