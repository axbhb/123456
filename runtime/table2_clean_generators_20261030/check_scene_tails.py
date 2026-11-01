"""CPU preparation of each training-scene tail, including terminal-frame handling."""
import os,json,subprocess,time,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=W/'scene_tail_checks';O.mkdir(exist_ok=True)
plan=json.loads((W/'plan.json').read_text());results=[]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for gen in ['panacea_plus','dd2','magicdrive']:
 tails={c['scene_token']:c for c in plan['generators'][gen]}
 for scene,clip in tails.items():
  dst=O/gen/('%03d'%clip['index']);dst.mkdir(parents=True,exist_ok=True)
  key='panacea' if gen=='panacea_plus' else gen
  name='streampetr_torch111_cu113' if gen=='panacea_plus' else ('dd2_gpu' if gen=='dd2' else 'frozendrive')
  start=time.time()
  with (dst/'prepare.log').open('w') as f:
   code=subprocess.call([str(R/('envs/'+name+'/bin/python')),str(W/('prepare_'+key+'.py')),'--clip',clip['path'],'--output',str(dst)],env=env,stdout=f,stderr=subprocess.STDOUT)
  results.append({'generator':gen,'scene_token':scene,'clip':clip['index'],'exit_code':code,'seconds':time.time()-start})
  for filename in ['panacea_real_8frame_batch.pt','dd2_conditions.pt','magicdrive_real_61frame_batch.pt']:
   batch=dst/filename
   if batch.is_file():batch.unlink()
  (O/'status.json').write_text(json.dumps({'status':'running','gpu_used':False,'results':results},indent=2))
(O/'status.json').write_text(json.dumps({'status':'completed','gpu_used':False,'results':results},indent=2))
