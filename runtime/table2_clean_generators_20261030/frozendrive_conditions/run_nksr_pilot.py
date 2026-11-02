import os,sys,json,pickle,time,subprocess,fcntl,traceback
from pathlib import Path
import numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--sample-token");ap.add_argument("--output");ap.add_argument("--skip-native-batch",action="store_true");args=ap.parse_args()
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030/frozendrive_conditions';V=R/'envs/nksr_reconstruction/bin/python';O=R/'outputs/table2_clean_mini_283_121/frozendrive/condition_pilot';D=R/'datasets/frozendrive_mini_conditions'
if args.output:O=Path(args.output)
O.mkdir(parents=True,exist_ok=True)
if (O/'status.json').exists():
 (O/('status_before_resume_%d.json'%time.time())).write_text((O/'status.json').read_text())
state={'status':'running','started':time.time(),'gpu_used':False}
def save(**kw):
 state.update(kw);p=O/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def run(args,name,env):
 with (O/name).open('a') as f:subprocess.run([str(a) for a in args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
try:
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='2',NKSR_CHECKPOINT=str(W/'checkpoints/ks.pth'))
 if json.loads((W/'chamfer_build_status.json').read_text()).get('status')!='passed':raise RuntimeError('Chamfer native build/test is not ready')
 if json.loads((W/'nksr_cpu_validation.json').read_text()).get('status')!='passed':raise RuntimeError('CPU checkpoint and real-data validation not ready')
 infos=pickle.load((R/'datasets/streampetr_mini_scene_283_121/infos_train.pkl').open('rb'))['infos'];test=pickle.load((R/'datasets/streampetr_mini_scene_283_121/infos_test.pkl').open('rb'))['infos'];train_tokens={x['token'] for x in infos};assert len(infos)==283 and len(test)==121;assert not {x['scene_token'] for x in infos}&{x['scene_token'] for x in test}
 rows=json.loads((W/'nksr_cpu_validation.json').read_text())['rows'];assert {x['sample_token'] for x in rows}==train_tokens
 target=next(x for x in infos if x['token']==args.sample_token) if args.sample_token else sorted(infos,key=lambda x:(x['scene_token'],x['timestamp']))[0];token=target['token'];scenes=json.loads((D/'v1.0-mini/scene.json').read_text());scene_index=next(i for i,x in enumerate(scenes) if x['token']==target['scene_token']);prepared=O/'prepared.npz'
 save(phase='cpu_native_preparation',sample_token=token,scene_index=scene_index,scene_token=target['scene_token'])
 if not prepared.exists():run([V,W/'generate_occ_mini.py','--config_path',W/'config800.yaml','--label_mapping',W/'nuscenes.yaml','--dataroot',D,'--save_path',O,'--index_list',scene_index,'--sample_token',token,'--prepare_only',prepared],'prepare.log',e)
 z=np.load(prepared);occ_path=O/'dense_voxels_with_semantic'/token/(str(z['lidar_token'])+'.npy')
 if occ_path.exists():
  occ=np.load(occ_path,allow_pickle=True)
  assert occ.ndim==2 and occ.shape[1]==4 and len(occ)>0
  assert np.isfinite(occ).all() and np.all((occ[:,:3]>=0)&(occ[:,:3]<[800,800,64]))
  assert np.isin(occ[:,3],np.arange(17)).all()
  save(reused_reconstruction=str(occ_path),reconstructed_voxels=len(occ))
 else:
  save(status='waiting_for_existing_gpu_lock',phase='nksr_reconstruction')
  with (R/'frameworks/gpu_budget_20261028.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
   gpu=max([x.split(',') for x in gpus.strip().splitlines()],key=lambda x:int(x[1]))[0].strip();e['CUDA_VISIBLE_DEVICES']=gpu
   save(status='running',phase='nksr_reconstruction',gpu=gpu,gpu_used=True)
   run([V,W/'reconstruct_prepared.py',prepared,O],'reconstruct.log',e)
 e['FROZEN_SKIP_NATIVE_BATCH']='1' if args.skip_native_batch else '0'
 e['CUDA_VISIBLE_DEVICES']='-1';e.update(FROZEN_DATA_ROOT=str(D),FROZEN_VOXELS=str(O/'dense_voxels_with_semantic'),FROZEN_PROJECTED=str(O/'projected'),FROZEN_SAMPLE_TOKEN=token,FROZEN_OCCUPANCY_SOURCE=str(W/'projection_source'))
 save(phase='six_camera_projection');run([V,W/'project_uniscene_mini.py'],'projection.log',e)
 save(phase='validate_geometry_and_labels');run([V,W/'validate_condition_pilot.py'],'validation.log',e)
 save(status='passed',phase='first_keyframe_conditions_ready',finished=time.time())
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
