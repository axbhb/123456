import os,json,collections,argparse
parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
r=Path('/srv/table2');p=Path(args.output)
trained=torch.load(str(p/'train/iter_283.pth'),map_location='cpu');initial=torch.load(str(r/'assets/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth'),map_location='cpu')
steps=collections.Counter()
for value in trained.get('optimizer',{}).get('state',{}).values():
 if 'step' in value: steps[int(value['step'])]+=1
changes={}
for key in ['img_backbone.conv1.weight','pts_bbox_head.reference_points.weight']:
 if key in trained['state_dict'] and key in initial['state_dict']:
  changes[key]=float((trained['state_dict'][key].float()-initial['state_dict'][key].float()).abs().max())
finite=all(bool(torch.isfinite(v).all()) for v in trained['state_dict'].values() if torch.is_tensor(v) and v.is_floating_point())
summary=dict(checkpoint_iteration=trained.get('meta',{}).get('iter'),optimizer_step_counts={str(k):v for k,v in steps.items()},representative_parameter_max_changes=changes,all_model_weights_finite=finite,checkpoint_bytes=(p/'train/iter_283.pth').stat().st_size,amp_state=trained.get('meta',{}).get('fp16'),gpu_used=False)
(p/'training_audit.json').write_text(json.dumps(summary,indent=2,default=lambda x:float(x) if torch.is_tensor(x) else str(x)));print(json.dumps(summary,default=str))

if not finite or not steps or max(steps)<=0:raise RuntimeError('Finetuning checkpoint has no valid optimizer update or contains nonfinite weights')
