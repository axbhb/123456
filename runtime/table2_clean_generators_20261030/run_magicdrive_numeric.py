from pathlib import Path
import os,fcntl,subprocess,json,time
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=R/'outputs/table2_clean_mini_283_121/magicdrive/numeric_verification';O.mkdir(exist_ok=True)
with (R/'frameworks/gpu_budget_20261028.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
 gpu=max([x.split(',') for x in rows.strip().splitlines()],key=lambda x:int(x[1]))[0].strip()
 e=os.environ.copy();e.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (O/'verify.log').open('a') as log:
  code=subprocess.call([str(R/'envs/frozendrive/bin/python'),str(W/'verify_magicdrive_numeric.py'),'--input',str(O.parent/'clips/001/magicdrive_real_61frame_batch.pt'),'--output',str(O),'--gpu',gpu,'--seed','20261031'],stdout=log,stderr=subprocess.STDOUT,env=e)
 print('Verification exit',code)
