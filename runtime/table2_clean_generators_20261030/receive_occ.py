import urllib.request,urllib.error,time,json,tarfile
from pathlib import Path
R=Path('/srv/table2');O=R/'assets/frozendrive_native_occ_20261030';O.mkdir(exist_ok=True);url='http://10.2.14.72:39057';F=O/'GT_occupancy_mini_12Hz.tar.gz';expected=296673810
state={'status':'waiting_for_jump_download','started':time.time(),'expected_bytes':expected}
def save(**kw):
 state.update(kw);tmp=O/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(O/'status.json')
save()
try:
 while True:
  try:
   response=urllib.request.urlopen(url+'/occupancy.tar.gz',timeout=30);break
  except urllib.error.HTTPError as e:
   if e.code!=503:raise
   time.sleep(10)
 part=F.with_suffix('.lan.partial');save(status='receiving')
 with response,part.open('wb') as f:
  while True:
   b=response.read(1048576)
   if not b:break
   f.write(b);save(bytes=f.tell())
 if part.stat().st_size!=expected:raise IOError('Unexpected file size')
 part.replace(F)
 with tarfile.open(F,'r:gz') as tar:
  members=[{'name':x.name,'bytes':x.size,'regular':x.isfile()} for x in tar.getmembers()]
 (O/'archive_members.json').write_text(json.dumps(members,indent=2));save(status='downloaded',members=len(members),uncompressed_bytes=sum(x['bytes'] for x in members),examples=members[:15],finished=time.time())
 urllib.request.urlopen(urllib.request.Request(url+'/complete',method='POST',data=b''),timeout=10).close()
except Exception as e:save(status='failed',error=str(e));raise
