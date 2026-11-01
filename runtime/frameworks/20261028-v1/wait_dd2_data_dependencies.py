import os,time,json,hashlib,subprocess
from pathlib import Path
P=Path('/srv/table2/frameworks/20261028-v1');deadline=time.time()+1800;rows=[{"name":"lmdb-1.4.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl","sha256":"7ba5d78b0ff130b38a56b7161ceb7e27ba4364d827d2bbb251c24b06c28c64cd"},{"name":"decord-0.6.0-py3-none-manylinux2010_x86_64.whl","sha256":"51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"}]
s={'status':'waiting_dependency_wheels','gpu_used':False,'deadline':deadline}
def save():
 (P/'dd2_data_dependency_install.json').write_text(json.dumps(s,indent=2))
save()
try:
 for row in rows:
  q=P/row['name']
  while not q.exists() and time.time()<deadline:time.sleep(10)
  assert q.exists() and hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
 py='/srv/table2/envs/dd2_gpu/bin/python'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1')
 result=subprocess.run([py,'-m','pip','install','--no-index','--no-deps']+[str(P/r['name']) for r in rows],env=env,timeout=90);assert result.returncode==0
 s['status']='preparing_native_conditions';save()
 result=subprocess.run([py,str(P/'prepare_dd2_native_conditions.py')],env=env,timeout=500);s.update(status='finished',returncode=result.returncode,condition_status=json.loads((P/'dd2_native_conditions_status.json').read_text())['status'])
except Exception as e:s.update(status='failed',error=str(e))
s['finished']=time.time();save()
