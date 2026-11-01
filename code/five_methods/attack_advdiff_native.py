from common import *
import subprocess,numpy as np
from PIL import Image
out=O/'advdiff';out.mkdir(parents=True,exist_ok=True);selected=json.loads((O/'selection.json').read_text())['selected_tokens'];source=pickle.load((C/'infos_train.pkl').open('rb'));bytoken={x['token']:x for x in source['infos']};plan=json.loads((PLAN/'plan.json').read_text());state=dict(method='AdvDiff-inspired-native-'+G+'-v1',generator=G,status='starting',guidance_strength=.025,guided_steps=10,gradient='frozen native denoiser prediction; predicted-clean VAE adjoint; StreamPETR native forward with BPDA preprocessing',scope='independent original-condition native clip regeneration per selected keyframe; only selected six camera images replace paired Clean',completed=[]);started=time.time()
def save(**kw):state.update(kw);state['seconds']=time.time()-started;atomic(out/'attack_status.json',state)
def run(cmd,log,env):
 with log.open('a') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
try:
 for token in selected:
  folder=out/'images'/token;receipt=folder/'result.json'
  if receipt.exists():state['completed'].append(json.loads(receipt.read_text()));continue
  folder.mkdir(parents=True,exist_ok=True);native=out/'native'/token;native.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['PYTHONPATH']=':'.join(map(str,[W,T,D,S]))+':'+env.get('PYTHONPATH','');env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TABLE2_NATIVE_OUTPUT=str(native));info=bytoken[token];detector_order=list(info['cams']);t0=time.time();context=dict(token=token,detector_order=detector_order)
  if G=='frozendrive':
   context.update(camera_order=detector_order,slot=0);env['TABLE2_GUIDANCE_CONTEXT']=json.dumps(context);save(status='native_sampling',token=token);run([R/'envs/frozendrive/bin/python',W/'generate_frozendrive_guided.py'],native/'generate.log',env);order=detector_order;paths={cam:native/(cam+'.png') for cam in order};seed=None
  else:
   clip=next(c for c in plan['generators'][G] if token in c['export_tokens']);magic=G=='magicdrive';prepython=R/'envs/frozendrive/bin/python' if magic else P;genpython=R/'envs/frozendrive/bin/python' if magic else R/'envs/panacea_plus/bin/python';kind='magicdrive' if magic else 'panacea';batch=native/('magicdrive_real_61frame_batch.pt' if magic else 'panacea_real_8frame_batch.pt')
   if not batch.exists():save(status='preparing_native_conditions',token=token);run([prepython,W/'prepare_conditions.py','--generator',G,'--clip',clip['path'],'--output',native],native/'prepare.log',env)
   meta=json.loads((native/(kind+'_input_status.json')).read_text());order=meta['camera_order'];slot=clip['tokens'].index(token)*(6 if magic else 1);context.update(camera_order=order,slot=slot);env['TABLE2_GUIDANCE_CONTEXT']=json.dumps(context);save(status='native_sampling',token=token);run([genpython,W/('generate_'+kind+'_guided.py'),'--gpu',os.environ.get('CUDA_VISIBLE_DEVICES','0'),'--input',batch,'--output',native,'--seed',clip['seed']],native/'generate.log',env);paths={cam:native/('view%d_frame%03d.png'%(v,slot) if magic else 'view%d_frame%02d.png'%(v,slot)) for v,cam in enumerate(order)};seed=clip['seed']
  logs=json.loads((native/'guidance.json').read_text());assert len(logs)>=10 and all(x['image_gradient_l2']>0 for x in logs);exported={};changed=0
  for cam,path in paths.items():
   image=Image.open(path).convert('RGB');clean=Image.open(info['cams'][cam]['data_path']).convert('RGB');assert image.size==clean.size;changed+=int(np.any(np.asarray(image)!=np.asarray(clean)));dest=folder/(cam+'.png');image.save(dest);exported[cam]=str(dest)
  assert changed>0;record=dict(token=token,paths=exported,paired_seed=seed,changed_cameras=changed,guidance_logs=logs,seconds=time.time()-t0);
  if G!='frozendrive' and batch.exists():batch.unlink();record['native_condition_cache_removed_after_export']=str(batch)
  atomic(receipt,record);state['completed'].append(record);save()
 for row in source['infos']:
  for record in state['completed']:
   if row['token']==record['token']:
    for cam,path in record['paths'].items():row['cams'][cam]['data_path']=path
 pickle.dump(source,(out/'infos_train.pkl').open('wb'));code="import sys;sys.path.insert(0,"+repr(str(W))+");from common import *;from mmcv import Config;cfg=Config.fromfile(str(C/'streampetr.py'),import_custom_modules=False);cfg.data.train.ann_file=str(O/'advdiff/infos_train.pkl');cfg.work_dir=str(O/'advdiff/train');cfg.runner.max_iters=ITERS;cfg.checkpoint_config.interval=ITERS;cfg.dump(str(O/'advdiff/streampetr.py'))";subprocess.run([str(P),'-c',code],env=os.environ.copy(),check=True);save(status='completed',frames=N,changed_cameras=sum(x['changed_cameras'] for x in state['completed']))
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc());raise
