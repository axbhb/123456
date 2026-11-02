import os,sys,json,subprocess,tarfile,shutil,traceback,time
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030/frozendrive_conditions';V=R/'envs/nksr_reconstruction';BASE=R/'envs/streampetr_torch111_cu113/bin/python';state={'status':'running','started':time.time(),'gpu_used':False}
def save(**kw):
 state.update(kw);p=W/'nksr_build_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
def run(cmd,**kwargs):
 print('RUN',cmd,flush=True);subprocess.run([str(x) for x in cmd],check=True,**kwargs)
try:
 os.nice(10);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',CUDA_HOME='/usr/local/cuda-11.3',TORCH_CUDA_ARCH_LIST='7.0',MAX_JOBS='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PIP_NO_INDEX='1');env['PATH']=str(V/'bin')+':/usr/local/cuda-11.3/bin:'+env.get('PATH','')
 save(phase='create_isolated_environment')
 if not (V/'bin/python').exists():run([BASE,'-m','venv','--system-site-packages',V],env=env)
 # Reuse base libraries read-only; all installs target this separate virtualenv.
 paths=json.loads(subprocess.check_output([str(BASE),'-c','import sys,json;print(json.dumps([p for p in sys.path if p.endswith("site-packages")]))'],env=env,universal_newlines=True))
 site=V/'lib/python3.9/site-packages';site.mkdir(parents=True,exist_ok=True);(site/'existing_torch_readonly.pth').write_text('\n'.join(paths)+'\n')
 archive=W/'nksr_offline_source.tar.gz'
 with tarfile.open(str(archive),'r:gz') as t:
  for m in t:
   target=(W/m.name).resolve()
   if W.resolve() not in target.parents or not (m.isdir() or m.isfile()):raise RuntimeError('Unexpected source archive member')
   if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
   target.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
 save(phase='install_build_tool')
 wheel=next(W.glob('ninja-*.whl'));run([V/'bin/python','-m','pip','install','--no-index','--no-deps',wheel],env=env)
 save(phase='compile_extension')
 run([V/'bin/python','-m','pip','install','--no-index','--no-build-isolation','--no-deps','.'],cwd=str(W/'nksr_source/package'),env=env)
 save(phase='validate_extension_import')
 probe='import torch,importlib.util,json;from pathlib import Path;p=next(Path('+repr(str(site))+').glob("nksr/_C*.so"));s=importlib.util.spec_from_file_location("_C",str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps({"extension":str(p),"torch":torch.__version__}))'
 out=subprocess.check_output([str(V/'bin/python'),'-c',probe],env=env,universal_newlines=True);save(status='extension_ready',phase='runtime_dependencies_pending',probe=json.loads(out),finished=time.time())
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc(),finished=time.time());raise
