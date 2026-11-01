"""Reuse native condition builders with explicit dataset/version paths."""
from common import *
if a.generator=='dd2':
 layout=json.loads((W/'generator_layouts.json').read_text())['dd2'];code=code.replace("ref=torch.load(R/'assets/dd2_aligned_clips/train_b_conditions.pt',map_location='cpu');order=list(ref['camera_order']);del ref",'order='+repr(layout['camera_order']))
code=code.replace('p.write_text(s)','(p.write_text(s) if p.read_text()!=s else None)')
sys.argv=[str(path),'--clip',a.clip,'--output',a.output];exec(compile(code,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
