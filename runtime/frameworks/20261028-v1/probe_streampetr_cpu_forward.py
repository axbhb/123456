"""Real pretrained StreamPETR CPU inference check; no attack or training."""
import os,sys,json,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
root=Path('/srv/table2/frameworks/20261028-v1');runtime=root/'runtime_sources/streampetr'
sys.path.insert(0,str(runtime));os.chdir(runtime);out=root/'streampetr_cpu_forward_result.json';started=time.time();report={'status':'started','gpu_used':False,'attack_executed':False}
def save():out.write_text(json.dumps(report,indent=2))
save()
try:
 import torch,numpy as np
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
 from mmcv import Config
 from mmcv.parallel import collate,scatter
 import projects.mmdet3d_plugin
 from mmdet3d.models import build_model
 from mmdet3d.datasets import build_dataset
 cfg=Config.fromfile(str(runtime/'smoke_v100.py'));model=build_model(cfg.model);checkpoint=torch.load(cfg.load_from,map_location='cpu');model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint;model.eval()
 dataset=build_dataset(dict(cfg.data.test,test_mode=True));batch=scatter(collate([dataset[0]],samples_per_gpu=1),[-1])[0]
 t=time.time()
 with torch.no_grad():pred=model(return_loss=False,rescale=True,**batch)
 assert len(pred)==1
 p=pred[0]['pts_bbox'];assert torch.isfinite(p['scores_3d']).all() and torch.isfinite(p['boxes_3d'].tensor).all()
 assert not torch.cuda.is_initialized()
 report.update(status='passed_real_checkpoint_inference',strict_checkpoint_load=True,model_forward_passed=True,predicted_boxes=len(p['scores_3d']),inference_seconds=time.time()-t,sample_token=dataset.data_infos[0]['token'],cuda_initialized=False)
except Exception as e:report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()[-4500:])
report['seconds']=time.time()-started;save();print(json.dumps(report))
if report['status']=='failed':sys.exit(2)
