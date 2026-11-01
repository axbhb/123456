"""Guarded CUDA execution check; no generation or attack is performed."""
import os,json,time,sys
from pathlib import Path
assert os.environ.get('PHYSCOND_GPU_GUARD')=='1'
import torch
out=Path(sys.argv[1]);started=time.time()
x=torch.arange(16,device='cuda',dtype=torch.float32).requires_grad_(True)
y=(x.square()).sum();y.backward();torch.cuda.synchronize()
assert torch.equal(x.grad,2*x.detach())
report={'status':'passed_cuda_runtime_and_autograd','torch':torch.__version__,'cuda_build':torch.version.cuda,'device':torch.cuda.get_device_name(0),'gpu_used':True,'learned_generator_executed':False,'attack_update_executed':False,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'seconds':time.time()-started}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
