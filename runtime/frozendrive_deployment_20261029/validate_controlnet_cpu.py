import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1')
import sys,time,json,torch,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;WD=W/'source/WorldDreamer';os.chdir(WD);sys.path.insert(0,str(WD));torch.set_num_threads(2)
state={'gpu_used':False,'started':time.time()}
try:
 from projects.dreamer.networks.unet_addon_single_ref_depth import SingleRefControlNetModel
 model,info=SingleRefControlNetModel.from_pretrained(str(W.parent/'assets/generators/frozendrive/SDv1.5_mv_single_ref_nus/weight-E5-S200000/controlnet'),output_loading_info=True)
 issues={k:v for k,v in info.items() if v}
 if issues:raise RuntimeError(str(issues))
 assert not torch.cuda.is_initialized()
 state.update(status='passed',parameters=sum(p.numel() for p in model.parameters()),loading_info=info,cuda_initialized=False,seconds=time.time()-state['started'])
except Exception:state.update(status='failed',error=traceback.format_exc())
(W/'controlnet_cpu_result.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
