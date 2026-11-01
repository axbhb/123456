import os,sys,argparse,json
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';D=R/'datasets/streampetr_mini_scene_283_121';S=R/'frameworks/20261028-v1/runtime_sources/streampetr'
p=argparse.ArgumentParser();p.add_argument('--generator',required=True);a=p.parse_args();O=R/'outputs/table2_clean_mini_283_121'/a.generator
os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path[:0]=[str(W),str(D),str(S)];os.chdir(S)
import numpy as np,torch
from mmcv import Config
import projects.mmdet3d_plugin,mini_camera_dataset,synthetic_camera_pipeline
from mmdet3d.datasets import build_dataset
cfg=Config.fromfile(str(O/'streampetr_partial.py'));data=build_dataset(cfg.data.train);n=len(data)
for i in range(n):
 item=data[i];im=item['img'].data;assert torch.isfinite(im).all();assert list(im.shape)==[1,6,3,256,704],im.shape
report={'status':'passed','frames':n,'pipeline':'native training with restored calibrated original canvas','gpu_used':False};(O/'pipeline_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
