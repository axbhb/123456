import os,sys,json,time,pickle,random,copy,traceback,argparse
from pathlib import Path
R=Path(os.environ.get('TABLE2_ROOT','/srv/table2')).resolve()
G=os.environ.get('TABLE2_GENERATOR','dd2')
N=int(os.environ.get('TABLE2_REPLACEMENT_COUNT','14'))
SEED=int(os.environ.get('TABLE2_SEED','0'))
SELECTION_SEED=int(os.environ.get('TABLE2_SELECTION_SEED','0'))
ATTACK_SEED=int(os.environ.get('TABLE2_ATTACK_SEED','0'))
ITERS=int(os.environ.get('TABLE2_ITERATIONS','283'))
W=Path(__file__).resolve().parent;O=Path(os.environ.get('TABLE2_OUTPUT',str(R/'outputs/table2_portable'/G)))
D=Path(os.environ.get('TABLE2_DATASET',str(R/'datasets/streampetr_mini_scene_283_121')));C=Path(os.environ.get('TABLE2_CLEAN',str(R/'outputs/table2_clean_mini_283_121'/G)))
S=R/'frameworks/20261028-v1/runtime_sources/streampetr';T=R/'table2_clean_generators_20261030'
PLAN=Path(os.environ.get('TABLE2_PLAN_DIR',str(T)))
P=R/'envs/streampetr_torch111_cu113/bin/python'
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
def seed_all(seed=ATTACK_SEED):
 import numpy as np,torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def runtime_summary():
 names=['torch','torchvision','numpy','mmcv','mmdet','mmdet3d','diffusers','transformers','xformers','nuscenes','sgm']
 return {name:dict(version=getattr(sys.modules[name],'__version__',None),module_file=getattr(sys.modules[name],'__file__',None)) for name in names if sys.modules.get(name) is not None}
def load_training():
 sys.path[:0]=[str(W),str(T),str(D),str(S)];os.chdir(S)
 import torch,numpy as np
 from mmcv import Config
 from mmdet3d.datasets import build_dataset
 import projects.mmdet3d_plugin,mini_camera_dataset,synthetic_camera_pipeline
 cfg=Config.fromfile(str(C/'streampetr.py'));data=build_dataset(cfg.data.train)
 atomic(O/'detector_runtime.json',runtime_summary())
 return cfg,data
