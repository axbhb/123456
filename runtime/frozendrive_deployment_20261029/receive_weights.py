import sys,time,json,subprocess,urllib.request,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parent;A=R/'assets/generators/frozendrive';WD=W/'source/WorldDreamer';items=json.loads((W/'assets.json').read_text());url=sys.argv[1];state={'gpu_used':False,'phase':'receiving','files':{}}
def save(): (W/'weights_status.json').write_text(json.dumps(state,indent=2))
try:
 for item in items:
  p=A/item['path'];p.parent.mkdir(parents=True,exist_ok=True);state['current']=item['path'];save()
  if not (p.exists() and p.stat().st_size==item['size']):
   partial=p.with_name(p.name+'.partial');deadline=time.time()+43200
   while time.time()<deadline:
    result=subprocess.run(['curl','--fail','--silent','--show-error','--connect-timeout','10','--max-time','600','--continue-at','-','--output',str(partial),url+'/'+item['path']])
    if result.returncode==0 and partial.stat().st_size==item['size']:partial.replace(p);break
    time.sleep(15)
   else:raise TimeoutError('Download window exhausted; partial files retained')
  state['files'][item['path']]={'bytes':p.stat().st_size};save()
 (WD/'dreamer-log').mkdir(exist_ok=True);link=WD/'dreamer-log/SDv1.5_mv_single_ref_nus'
 if not link.exists():link.symlink_to(A/'SDv1.5_mv_single_ref_nus')
 (WD/'pretrained').mkdir(exist_ok=True);link=WD/'pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K'
 if not link.exists():link.symlink_to(A/'CLIP-ViT-B-32-laion2B-s34B-b79K')
 base=WD/'pretrained/stable-diffusion-v1-5';base.mkdir(exist_ok=True)
 for p in (R/'assets/generators/sd15_base').rglob('*'):
  if not p.is_file():continue
  dest=base/p.relative_to(R/'assets/generators/sd15_base');dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(p)
  if '.fp16.' in dest.name:
   plain=dest.with_name(dest.name.replace('.fp16.','.'))
   if not plain.exists():plain.symlink_to(p)
 state.update(phase='download_complete',total_bytes=sum(x['size'] for x in items),weight_loading_validated=False);save()
except Exception:state.update(phase='failed',error=traceback.format_exc());save();raise
