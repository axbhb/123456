from common import *
import torch,numpy as np
from PIL import Image
from omegaconf import OmegaConf
from native_guidance import NativeGuide
context=json.loads(os.environ['TABLE2_GUIDANCE_CONTEXT']);token=context['token'];dest=Path(os.environ['TABLE2_NATIVE_OUTPUT']);dest.mkdir(parents=True,exist_ok=True);base=C;wd=R/'frozendrive_deployment_20261029/source/WorldDreamer';os.chdir(wd);sys.path[:0]=[str(wd),str(wd/'third_party/bevfusion')]
import dataset
from dataset import collate_fn_singleframe
from mmdet3d.datasets import build_dataset
from projects.dreamer.utils.test_utils import build_pipe
from accelerate.utils import set_seed
cfg=OmegaConf.load(str(base/'native_clean_config.yaml'));set_seed(cfg.seed);pairs=json.loads((base/'native_generation_pairs.json').read_text());pair=next(x for x in pairs if x['token']==token)
state=dict(status='starting',token=token)
def save(**kw):state.update(kw);atomic(dest/'result.json',state)
try:
 torch.linalg.solve(torch.eye(2,device='cuda'),torch.ones(2,device='cuda'));torch.cuda.empty_cache();pipe,dtype=build_pipe(cfg,'cuda')
 for m in [pipe.unet,pipe.controlnet,pipe.vae]:m.eval().requires_grad_(False)
 ds=build_dataset(OmegaConf.to_container(cfg.dataset.data.val.nuscenes,resolve=True));rng=base/'frames'/token/'rng_before.pt'
 if rng.exists():
  state_rng=torch.load(rng,map_location='cpu');torch.set_rng_state(state_rng['cpu']);torch.cuda.set_rng_state(state_rng['cuda'])
 example=ds[pair['index']];assert example['metas'].data['token']==token;batch=collate_fn_singleframe([example],template=cfg.dataset.template,tokenizer=pipe.tokenizer,is_train=False,bbox_mode=cfg.model.bbox_mode,bbox_view_shared=cfg.model.bbox_view_shared,bbox_drop_ratio=cfg.runner.bbox_drop_ratio,bbox_add_ratio=cfg.runner.bbox_add_ratio,bbox_add_num=cfg.runner.bbox_add_num)
 if not pair['is_first_frame']:
  from torchvision.transforms.functional import to_tensor,normalize
  refs=[normalize(to_tensor(Image.open(base/'frames'/pair['reference_token']/(cam+'.png')).convert('RGB')),[.5]*3,[.5]*3) for cam in cfg.dataset.view_order_nuscenes];batch['ref_images'][0]=torch.stack(refs).contiguous().float()
 guide=NativeGuide(token,list(cfg.dataset.view_order_nuscenes),context['detector_order'],save);guide.install_diffusers(pipe,range(6));rel=batch['relative_pose'].to(dtype)
 kwargs=dict(prompt=batch['captions'],bev_hdmap=batch['bev_hdmap'],camera_param=batch['camera_param'].to(dtype),rel_pose=rel,ref_rel_pose=torch.eye(rel.size(-1),dtype=rel.dtype,device=rel.device).expand_as(rel),ref_images=batch['ref_images'].to(dtype),layout_canvas=batch['layout_canvas'].to(dtype),occupancy=batch['occupancy'].to(dtype),ref_layout_canvas=batch['ref_layout_canvas'].to(dtype),ref_occupancy=batch['ref_occupancy'].to(dtype),height=cfg.dataset.image_size[0],width=cfg.dataset.image_size[1],generator=torch.Generator(device=pipe.device).manual_seed(cfg.seed),bev_controlnet_kwargs=batch['kwargs'],is_first_frame=pair['is_first_frame'],use_first_frame=cfg.get('use_first_frame',False),**OmegaConf.to_container(cfg.runner.pipeline_param,resolve=True))
 start=time.time()
 with torch.no_grad():images=pipe(**kwargs,callback=lambda i,t,z:save(phase='sampling',completed_steps=i+1)).images[0]
 guide.finish();assert len(images)==6
 for cam,image in zip(cfg.dataset.view_order_nuscenes,images):image.save(dest/(cam+'.png'))
 atomic(dest/'guidance.json',guide.logs);save(status='passed',seconds=time.time()-start,reference_token=pair['reference_token'],reference_source='paired Clean preceding generated frame',guidance_steps=len(guide.logs))
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc());raise
