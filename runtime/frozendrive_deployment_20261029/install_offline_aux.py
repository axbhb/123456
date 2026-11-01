import os,subprocess,json,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;E=W.parent/'envs/frozendrive';state={'phase':'installing_offline','gpu_used':False};(W/'aux_status.json').write_text(json.dumps(state))
try:
 extras=[(p.as_uri()+'#sha256=5d5c7b8ca6ef465adb2a0f96166acc0dcae333f7adf559d914755d76163dbaf6' if p.name.startswith('kornia_rs-0.2.0-') else str(p)) for p in (W/'wheels').glob('*.whl') if p.name.startswith(('open_clip_torch-','kornia-','kornia_rs-','pyogrio-','aioboto3-','aiobotocore-','botocore-','boto3-','aioitertools-','wcwidth-'))]
 subprocess.run([str(E/'bin/python'),'-m','pip','install','--disable-pip-version-check','--no-index','--no-deps','--no-build-isolation','-r',str(W/'cached_requirements.txt'),*extras],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MAX_JOBS='2'),check=True)
 (E/'lib/python3.9/site-packages/frozendrive_nuplan.pth').write_text(str(W/'source/WorldDreamer/third_party/nuplan-devkit')+'\n');state['phase']='ready'
except Exception:state.update(phase='failed',error=traceback.format_exc())
(W/'aux_status.json').write_text(json.dumps(state,indent=2))
