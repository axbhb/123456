import os,subprocess,json,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parent;E=R/'envs/frozendrive';state={'phase':'installing','gpu_used':False}
(W/'aux_status.json').write_text(json.dumps(state));(W/'aux_constraints.txt').write_text('numpy==1.23.1\ntorch==1.11.0+cu113\ntorchvision==0.12.0+cu113\n')
try:
 subprocess.run([str(E/'bin/python'),'-m','pip','install','--disable-pip-version-check','--no-build-isolation','--find-links',str(W/'wheels'),'--timeout','120','--retries','6','--proxy','http://10.2.14.72:41897','-i','https://pypi.tuna.tsinghua.edu.cn/simple','-c',str(W/'aux_constraints.txt'),'h5py==3.8.0','omegaconf==2.3.0','geopandas==0.12.2','fiona==1.9.6','pyproj==3.6.1','rasterio==1.3.9','pyinstrument==4.6.2','retry==0.9.2','sympy==1.11.1','mpi4py==3.1.5','tensorpack==0.11',str(next((W/'wheels').glob('open_clip_torch-2.17.1-*.whl'))),str(next((W/'wheels').glob('kornia-0.7.2-*.whl'))),str(next((W/'wheels').glob('pyogrio-0.6.0-*.whl'))),*[str(next((W/'wheels').glob(name+'-*.whl'))) for name in ['aioboto3','aiobotocore','botocore','boto3','aioitertools']], (W/'wheels'/'kornia_rs-0.2.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl').as_uri()+'#sha256=5d5c7b8ca6ef465adb2a0f96166acc0dcae333f7adf559d914755d76163dbaf6'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PIP_NO_INDEX='0'),check=True)
 site=E/'lib/python3.9/site-packages';(site/'frozendrive_nuplan.pth').write_text(str(W/'source/WorldDreamer/third_party/nuplan-devkit')+'\n')
 state['phase']='ready'
except Exception:state.update(phase='failed',error=traceback.format_exc())
(W/'aux_status.json').write_text(json.dumps(state,indent=2))
