import json,shutil
from pathlib import Path
from pip._vendor import msgpack
W=Path(__file__).resolve().parent;items=json.loads((W/'wheel_assets.json').read_text());cache=Path.home()/'.cache/pip';sizes={x['size']:x for x in items};found={}
for p in cache.rglob('*'):
 if not p.is_file():continue
 try:
  sz=p.stat().st_size
  if not any(n<=sz<=n+20000 for n in sizes):continue
  raw=p.read_bytes()
  if raw[:5]==b'cc=4,':raw=msgpack.unpackb(raw[5:],raw=False)['response']['body']
  if len(raw) not in sizes:continue
  item=sizes[len(raw)];dest=W/'wheels'/item['filename']
  if not dest.exists():dest.write_bytes(raw)
  found[item['filename']]=dest.as_uri()+'#sha256='+item['digest']
 except (OSError,ValueError,KeyError,TypeError):continue
missing=[x['filename'] for x in items if not (W/'wheels'/x['filename']).exists()]
req=[(W/'wheels'/x['filename']).as_uri()+'#sha256='+x['digest'] for x in items if (W/'wheels'/x['filename']).exists()]
(W/'cached_requirements.txt').write_text('\n'.join(req)+'\n');(W/'missing_wheels.json').write_text(json.dumps(missing));print(json.dumps({'found':len(req),'missing':missing}))
