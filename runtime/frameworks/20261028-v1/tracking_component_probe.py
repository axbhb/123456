"""Run official association code using two real annotations; not detector/AMOTA evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,json,sys,time,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
def table(name):return json.loads((a.data_root/'v1.0-mini'/(name+'.json')).read_text())
scenes=table('scene');samples={x['token']:x for x in table('sample')};annotations={x['token']:x for x in table('sample_annotation')};instances={x['token']:x for x in table('instance')};cats={x['token']:x['name'] for x in table('category')}
scene=next(x for x in scenes if x['name']=='scene-0061');s0=samples[scene['first_sample_token']];s1=samples[s0['next']];dt=(s1['timestamp']-s0['timestamp'])/1e6
assert dt>0
first={b['instance_token']:b for b in annotations.values() if b['sample_token']==s0['token']};second={b['instance_token']:b for b in annotations.values() if b['sample_token']==s1['token']}
key=next(k for k in first if k in second and cats[instances[k]['category_token']]=='vehicle.car');b0,b1=first[key],second[key]
vel=[(b1['translation'][i]-b0['translation'][i])/dt for i in range(2)]
def det(b):return {'translation':b['translation'],'size':b['size'],'rotation':b['rotation'],'velocity':vel,'detection_name':'car','detection_score':1.0}
sys.path.insert(0,str(ROOT/'sources/streampetr/nusc_tracking'));from pub_tracker import PubTracker
tracker=PubTracker(hungarian=False,max_age=3);r0=tracker.step_centertrack([det(b0)],0,.25);r1=tracker.step_centertrack([det(b1)],dt,.25)
assert len(r0)==len(r1)==1 and r0[0]['tracking_id']==r1[0]['tracking_id']
record={'status':'passed','scope':'official tracking association on two real GT-derived boxes; no detector inference','scene':scene['name'],'sample_tokens':[s0['token'],s1['token']],'annotation_tokens':[b0['token'],b1['token']],'delta_seconds':dt,'stable_tracking_id':r1[0]['tracking_id'],'learned_model_tested':False,'AMOTA':None,'gpu_used':False,'seconds':time.time()-started}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2));print(json.dumps(record))
