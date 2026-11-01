import os,sys,json,time,socket,urllib.request,tarfile
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';D=R/'assets/frozendrive_mini_occupancy';D.mkdir(exist_ok=True)
original=socket.getaddrinfo
def resolve(host,*a,**kw):return original('160.16.86.14' if host=='hf-mirror.com' else host,*a,**kw)
socket.getaddrinfo=resolve
url='https://hf-mirror.com/datasets/Kevinwjw/Uniscene_mini_dataset/resolve/main/GT_occupancy_mini_12Hz.tar.gz';target=D/'GT_occupancy_mini_12Hz.tar.gz';expected=296673810;part=target.with_suffix('.partial');state={'status':'downloading','source_repo':'Kevinwjw/Uniscene_mini_dataset','expected_bytes':expected,'started':time.time()}
def save(**kw):state.update(kw);(D/'download_status.json').write_text(json.dumps(state,indent=2))
save()
try:
 for attempt in range(5):
  if target.exists() and target.stat().st_size==expected:break
  start=part.stat().st_size if part.exists() else 0
  try:
   req=urllib.request.Request(url,headers={'Range':'bytes=%d-'%start} if start else {})
   with urllib.request.urlopen(req,timeout=45) as response:
    if start and response.status!=206:start=0
    with part.open('ab' if start else 'wb') as f:
     while True:
      b=response.read(1048576)
      if not b:break
      f.write(b);save(bytes=f.tell(),attempt=attempt)
   if part.stat().st_size!=expected:raise IOError('Unexpected archive byte count')
   part.replace(target);break
  except Exception as e:save(status='retrying',error=str(e));time.sleep(5)
 if not target.exists():raise RuntimeError('Archive download incomplete')
 with tarfile.open(target,'r:gz') as tar:
  members=tar.getmembers();names=[m.name for m in members]
  # Archive remains data; do not extract until paths and native format are inspected.
 (D/'archive_members.json').write_text(json.dumps(names));save(status='downloaded',members=len(names),examples=names[:20],finished=time.time())
except Exception as e:save(status='failed',error=str(e));raise
