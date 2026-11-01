import subprocess,sys,json,os,time
from pathlib import Path
root=Path(sys.argv[1]);results={}
for name,args in [('imports',['cpu_framework_probe.py','--output-dir',str(root/'cpu_probe'),'--timeout','40']),('tracking',['tracking_component_probe.py','--data-root','/srv/table2/assets/nuscenes_mini_temporal_camera_v1','--output',str(root/'tracking_component.json')])]:
 with (root/(name+'.log')).open('w') as f:
  try:r=subprocess.run([sys.executable]+args,cwd=str(root),stdout=f,stderr=subprocess.STDOUT,timeout=360);results[name]={'returncode':r.returncode}
  except subprocess.TimeoutExpired:results[name]={'timeout':True}
(root/'cpu_worker_done.json').write_text(json.dumps({'results':results,'gpu_used':False,'finished':time.time()},indent=2))
