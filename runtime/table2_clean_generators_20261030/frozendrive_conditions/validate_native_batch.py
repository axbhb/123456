"""Exercise the real native test pipeline and collate on the reconstructed pilot frame."""
import os,sys,json,time,traceback
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path('/srv/table2')
WD=R/'frozendrive_deployment_20261029/source/WorldDreamer'
O=R/'outputs/table2_clean_mini_283_121/frozendrive'
P=O/'condition_pilot'
sys.path.insert(0,str(WD));os.chdir(str(WD))
state={'status':'running','gpu_used':False,'started':time.time()}
def save(**kw):
 state.update(kw);(P/'native_batch_validation.json').write_text(json.dumps(state,indent=2))
try:
 import numpy as np
 import torch
 import dataset
 from dataset import collate_fn_singleframe
 from mmdet3d.datasets import build_dataset
 from omegaconf import OmegaConf
 from transformers import CLIPTokenizer
 cfg=OmegaConf.load(str(O/'native_clean_config.yaml'))
 cfg.dataset.dataset_root_occ=str(P/'projected')
 token=json.loads((P/'condition_validation.json').read_text())['sample_token']
 pairs=json.loads((O/'native_generation_pairs.json').read_text())
 pair=next(p for p in pairs if p['token']==token)
 assert pair['is_first_frame'] and pair['reference_token']==token
 save(phase='native_getitem',sample_token=token,pair_index=pair['index'])
 ds=build_dataset(OmegaConf.to_container(cfg.dataset.data.val.nuscenes,resolve=True))
 example=ds[pair['index']]
 assert example is not None
 tokenizer=CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path,subfolder='tokenizer',local_files_only=True)
 batch=collate_fn_singleframe([example],template=cfg.dataset.template,tokenizer=tokenizer,is_train=False,bbox_mode=cfg.model.bbox_mode,bbox_view_shared=cfg.model.bbox_view_shared,bbox_drop_ratio=cfg.runner.bbox_drop_ratio,bbox_add_ratio=cfg.runner.bbox_add_ratio,bbox_add_num=cfg.runner.bbox_add_num)
 assert list(batch['occupancy'].shape)==[1,6,17,448,800]
 assert list(batch['ref_images'].shape)==[1,6,3,448,800]
 assert list(batch['camera_param'].shape)==[1,6,3,7]
 assert torch.equal(batch['occupancy'],batch['ref_occupancy'])
 assert example['metas'].data['token']==token
 details=[]
 for i,cam in enumerate(cfg.dataset.view_order_nuscenes):
  z=np.load(P/'projected'/('sample_'+token)/('proj_'+cam+'.npz'))
  semantic=batch['occupancy'][0,i,1:].numpy()
  expected=np.stack([z['labels']==cls for cls in range(1,17)])
  assert np.array_equal(semantic,expected),cam+' semantic channels/camera order mismatch'
  details.append({'camera':cam,'semantic_pixels':int(expected.sum())})
 def check(value,name):
  if torch.is_tensor(value):
   assert value.device.type=='cpu',name
   assert torch.isfinite(value).all(),name
  elif isinstance(value,dict):
   for key,item in value.items():check(item,name+'.'+str(key))
 for key in ['occupancy','ref_occupancy','ref_images','pixel_values','camera_param','relative_pose','layout_canvas','ref_layout_canvas','bev_hdmap','kwargs']:check(batch[key],key)
 assert not torch.cuda.is_initialized()
 save(status='passed',phase='native_getitem_and_collate_passed',shapes={k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)},cameras=details,caption=batch['captions'],finished=time.time(),scope='one real reconstructed training keyframe; no model generation or full-dataset claim')
 print(json.dumps(state))
except Exception as ex:
 save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
