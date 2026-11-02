"""Run native geometric condition builders with explicit deployment/data paths."""
import argparse,os,re,sys,json
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--generator',required=True,choices=['dd2','magicdrive','panacea_plus'])
 parser.add_argument('--clip',required=True)
 parser.add_argument('--output',required=True)
 args=parser.parse_args()
 root=Path(os.environ.get('TABLE2_ROOT','/srv/table2')).resolve()
 builder_name={'dd2':'prepare_dd2.py','magicdrive':'prepare_magicdrive.py','panacea_plus':'prepare_panacea.py'}[args.generator]
 path=root/'table2_clean_generators_20261030'/builder_name
 code=path.read_text(encoding='utf8')
 code,count=re.subn(r"R=Path\('[^']+'\)",'R=Path('+repr(str(root))+')',code,count=1)
 if count!=1:raise ValueError('Native builder root assignment not found: '+str(path))
 for literal,variable in [("R/'assets/nuscenes_mini_temporal_camera_v1'",'TABLE2_RAW_DATASET'),("R/'datasets/streampetr_mini_scene_283_121'",'TABLE2_DATASET')]:
  code=code.replace(literal,"Path(os.environ.get("+repr(variable)+",str("+literal+")))")
 code=code.replace("version='v1.0-mini'","version=os.environ.get('TABLE2_NUSCENES_VERSION','v1.0-mini')")
 if args.generator=='dd2':
  layout=json.loads((Path(__file__).resolve().parent/'generator_layouts.json').read_text(encoding='utf8'))['dd2']
  original="ref=torch.load(R/'assets/dd2_aligned_clips/train_b_conditions.pt',map_location='cpu');order=list(ref['camera_order']);del ref"
  if original in code:code=code.replace(original,'order='+repr(layout['camera_order']))
 code=code.replace('p.write_text(s)','(p.write_text(s) if p.read_text()!=s else None)')
 sys.argv=[str(path),'--clip',args.clip,'--output',args.output]
 exec(compile(code,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))

if __name__=='__main__':main()
