import os,json,time,subprocess,tarfile,shutil,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030/frozendrive_conditions';V=R/'envs/nksr_reconstruction';state={'status':'running','started':time.time(),'gpu_used':False}
def save(**kw):
 state.update(kw);p=W/'chamfer_build_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
try:
 os.nice(10);e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='-1',CUDA_HOME='/usr/local/cuda-11.3',TORCH_CUDA_ARCH_LIST='7.0',MAX_JOBS='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',FORCE_CUDA='1',NVCC_FLAGS='-std=c++17',PIP_NO_INDEX='1');e['PATH']=str(V/'bin')+':/usr/local/cuda-11.3/bin:'+e.get('PATH','')
 save(phase='extract')
 with tarfile.open(str(W/'chamferdist_source.tar.gz')) as t:
  for m in t:
   p=(W/m.name).resolve()
   if W.resolve() not in p.parents or not (m.isfile() or m.isdir()):raise RuntimeError('Unexpected archive member')
   if m.isdir():p.mkdir(parents=True,exist_ok=True);continue
   p.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
 save(phase='compile')
 subprocess.run([str(V/'bin/python'),'-m','pip','install','--no-index','--no-build-isolation','--no-deps','.'],cwd=str(W/'chamferdist'),env=e,check=True)
 save(phase='cpu_knn_validation')
 code="import torch,json;from chamferdist.chamfer import knn_points;g=torch.Generator().manual_seed(7);x=torch.randn(1,103,3,generator=g);y=torch.randn(1,317,3,generator=g);a=knn_points(x,y,K=1);d=((x[:,:,None,:]-y[:,None,:,:])**2).sum(-1);v,i=d.min(-1);assert torch.equal(a.idx[...,0],i);assert torch.allclose(a.dists[...,0],v,atol=1e-6,rtol=1e-6);print(json.dumps({'status':'passed','points':[103,317],'max_squared_distance_error':float((a.dists[...,0]-v).abs().max())}))"
 p=subprocess.run([str(V/'bin/python'),'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=e);print(p.stdout);print(p.stderr)
 if p.returncode:raise RuntimeError('CPU KNN test failed: '+p.stderr[-3000:])
 save(status='passed',phase='ready',probe=p.stdout,finished=time.time())
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
