from common import *
import subprocess
path=O/'asymmetry/anchor_overrides.json';overrides=json.loads(path.read_text()) if path.exists() else {}
for target in [[-8,0],[-8,3],[-8,-3],[-12,0]]:
 geometry=json.loads((O/'asymmetry/geometry.json').read_text());missing=[t for t,r in geometry['frames'].items() if r['visible_pixels']==0]
 if not missing:break
 for token in missing:overrides[token]=target
 atomic(path,overrides);result=subprocess.run([str(P),str(W/'prepare_physical.py')]);
 if result.returncode:raise RuntimeError('road anchor preparation failed')
print('final',json.loads((O/'asymmetry/geometry.json').read_text())['physically_editable_frames'])
