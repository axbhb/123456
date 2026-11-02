import os,json,subprocess,time,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030/frozendrive_conditions';V=R/'envs/nksr_reconstruction'
state={'status':'running','started':time.time(),'gpu_used':False}
def save(**kw):
 state.update(kw);p=W/'nksr_runtime_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def run(args,**kw):
 print('RUN',args,flush=True);subprocess.run([str(a) for a in args],check=True,**kw)
try:
 os.nice(10);e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='-1',CUDA_HOME='/usr/local/cuda-11.3',TORCH_CUDA_ARCH_LIST='7.0',MAX_JOBS='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PIP_NO_INDEX='1');e['PATH']=str(V/'bin')+':/usr/local/cuda-11.3/bin:'+e.get('PATH','')
 save(phase='repair_package_metadata')
 run([V/'bin/python','-m','pip','install','--no-index','--no-build-isolation','--no-deps','.'],cwd=str(W/'nksr_source/package'),env=e)
 save(phase='install_runtime_wheels')
 wheels=sorted((W/'runtime_wheels').glob('*.whl'))
 if not wheels:raise RuntimeError('No runtime wheels')
 run([V/'bin/python','-m','pip','install','--no-index','--no-deps']+wheels,env=e)
 save(phase='cpu_runtime_test')
 probe="import json,torch,nksr,open3d as o3d,numpy as np;from pycg import vis;from pykdtree.kdtree import KDTree;from torch_scatter import scatter_sum;p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]));p.estimate_normals();x=scatter_sum(torch.tensor([1.,2.,3.]),torch.tensor([0,0,1]));assert x.tolist()==[3.,3.];assert np.isfinite(np.asarray(p.normals)).all();print(json.dumps({'nksr':nksr.__version__,'torch':torch.__version__,'open3d':o3d.__version__,'scatter_sum':x.tolist(),'normal_count':len(p.normals)}))"
 p=subprocess.run([str(V/'bin/python'),'-c',probe],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=e);print(p.stdout,flush=True);print(p.stderr,flush=True)
 if p.returncode:raise RuntimeError('CPU runtime import/operation test failed: '+p.stderr[-4000:])
 save(status='cpu_runtime_ready',phase='checkpoint_and_reconstruction_pending',probe=p.stdout,finished=time.time())
except Exception as ex:
 save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
