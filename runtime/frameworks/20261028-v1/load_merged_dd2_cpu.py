"""Strict CPU loading of the verified complete DD2 FP16 deployment artifact.
Caller provides the actual repository model classes and constructs the native
pipeline with its original VAE/scheduler/etc. No fallback/download/GPU use.
"""
import hashlib,json,inspect
from pathlib import Path
import torch
from safetensors.torch import load_file

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def load_complete_dd2_cpu(bundle_dir, *, unet_class, grounding_class):
 bundle=Path(bundle_dir);report=json.loads((bundle/'conversion_report.json').read_text())
 if report.get('status')!='passed_exact_native_fp16_merge' or report.get('all_values_equal_native') is not True or report.get('serialized_roundtrip_equal') is not True:
  raise ValueError('A completed native-equivalence receipt is required')
 checkpoint=bundle/'dd2_wo_img_merged.fp16.safetensors'
 if _sha(checkpoint)!=report['output']['sha256']:raise ValueError('Merged checkpoint hash mismatch')
 rows=json.loads((bundle/'parameter_manifest.json').read_text());expected={row['key']:row for row in rows}
 if len(expected)!=len(rows) or len(rows)!=report['verified_keys']:raise ValueError('Invalid parameter manifest')
 if 'assign' not in inspect.signature(torch.nn.Module.load_state_dict).parameters:
  raise RuntimeError('This memory-efficient strict loader requires load_state_dict(assign=True), e.g. the DD2 Torch2.2 environment')
 meta=json.loads((bundle/'dd2_meta.json').read_text());groups={}
 for group,cls in [('unet',unet_class),('grounding_downsampler',grounding_class),('box_downsampler',grounding_class)]:
  config=dict(meta[group]);name=config.pop('_class_name')
  if name!=cls.__name__:raise ValueError('Native model class mismatch: '+group)
  with torch.device('meta'):groups[group]=cls(**config)
 state=load_file(str(checkpoint),device='cpu')
 if set(state)!=set(expected):raise ValueError('Merged key set differs from verified manifest')
 seen=set()
 for group,module in groups.items():
  prefix=group+'.';subset={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
  for short,value in subset.items():
   key=prefix+short;row=expected[key]
   if value.dtype!=torch.float16 or list(value.shape)!=row['shape'] or not torch.isfinite(value).all():raise ValueError('Tensor shape/dtype/value mismatch: '+key)
   if hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()!=row['sha256']:raise ValueError('Tensor hash mismatch: '+key)
   seen.add(key)
  module.load_state_dict(subset,strict=True,assign=True)
  module.eval().requires_grad_(False)
  if any(p.device.type!='cpu' or p.dtype!=torch.float16 for p in module.parameters()):raise RuntimeError('Incomplete CPU FP16 bind')
 if seen!=set(state):raise ValueError('Unconsumed merged keys')
 return groups
