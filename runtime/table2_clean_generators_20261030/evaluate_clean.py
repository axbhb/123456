import argparse,os,sys,json,time,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
r=Path('/srv/table2');d=r/'datasets/streampetr_mini_scene_283_121';runtime=r/'frameworks/20261028-v1/runtime_sources/streampetr'
sys.path[:0]=[str(d),str(runtime)];os.chdir(runtime)
import numpy as np,torch
from mmcv import Config
from mmcv.parallel import collate,MMDataParallel
from mmcv.runner import wrap_fp16_model,load_checkpoint
import projects.mmdet3d_plugin,mini_camera_dataset
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
cfg=Config.fromfile(str(d/'streampetr_mini.py'));dataset=build_dataset(cfg.data.test);model=build_model(cfg.model)
load_checkpoint(model,a.checkpoint,map_location='cpu',strict=True)
wrap_fp16_model(model);model=MMDataParallel(model.cuda(),device_ids=[0]);model.eval();model.module.pts_bbox_head.reset_memory()
outputs=[];started=time.time();times=[]
for index in range(len(dataset)):
 batch=collate([dataset[index]],samples_per_gpu=1);torch.cuda.synchronize();t=time.time()
 with torch.no_grad(),torch.cuda.amp.autocast():pred=model(return_loss=False,rescale=True,**batch)
 torch.cuda.synchronize();times.append(time.time()-t)
 # Keep only CPU output tensors, not all predictions on the GPU.
 item=pred[0]['pts_bbox'];item['boxes_3d']=item['boxes_3d'].to('cpu');item['scores_3d']=item['scores_3d'].cpu();item['labels_3d']=item['labels_3d'].cpu();outputs.append(dict(pts_bbox=item))
 if (index+1)%10==0 or index+1==len(dataset):
  report=dict(status='inference',frames=index+1,total=len(dataset),seconds=time.time()-started)
  (a.output/'progress.json').write_text(json.dumps(report));print(json.dumps(report),flush=True)
metrics=dataset.evaluate(outputs,jsonfile_prefix=str(a.output/'predictions'))
report=dict(status='completed',checkpoint=a.checkpoint,frames=len(dataset),scenes=sorted({x['scene_name'] for x in dataset.data_infos}),metrics=metrics,seconds=time.time()-started,inference_seconds=sum(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,precision='AMP fp16 with native force_fp32 decorators',seed=0)
(a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
