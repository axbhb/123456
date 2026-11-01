"""Relocate only bundle-owned text/metadata. Model tensor files remain unchanged."""
import argparse,json,pickle,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--skip-pickles',action='store_true');p.add_argument('--old-root',default='/srv/table2');a=p.parse_args();root=a.root.resolve();old=a.old_root;new=str(root);changed=[]
def convert(value):
 if isinstance(value,str):return value.replace(old,new)
 if isinstance(value,list):return [convert(x) for x in value]
 if isinstance(value,tuple):return tuple(convert(x) for x in value)
 if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
 return value
for base in [root/'datasets',root/'outputs',root/'table2_clean_generators_20261030',root/'table2_portable_20261101',root/'frameworks',root/'generator_deployment_20261028',root/'frozendrive_deployment_20261029',root/'releases',root/'deployment',root/'baseline_deployment_20261030',root/'assets/dd2_merged_fp16',root/'assets/generators',root/'assets/dd2_base']:
 if not base.exists():continue
 for path in base.rglob('*'):
  if path.is_symlink():continue
  if not path.is_file():continue
  if path.suffix.lower() in ['.py','.json','.yaml','.yml','.txt','.cfg','.sh'] and path.stat().st_size<20*1024**2:
   try:text=path.read_text()
   except (UnicodeDecodeError,OSError):continue
   if old in text:path.write_text(text.replace(old,new));changed.append(str(path.relative_to(root)))
  elif not a.skip_pickles and path.suffix=='.pkl' and ('infos' in path.name or 'selected' in path.name):
   data=pickle.load(path.open('rb'));pickle.dump(convert(data),path.open('wb'));changed.append(str(path.relative_to(root)))
(root/'relocation_report.json').write_text(json.dumps(dict(old_root=old,new_root=new,changed=changed),indent=2));print(json.dumps(dict(changed_files=len(changed),new_root=new)))
