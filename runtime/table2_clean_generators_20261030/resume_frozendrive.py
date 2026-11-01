"""Continue FrozenDrive Clean: temporal pilot, all conditions, generation, training."""
import os,sys,json,time,subprocess,traceback
from pathlib import Path
R=Path('/srv/table2');W=R/'table2_clean_generators_20261030';O=R/'outputs/table2_clean_mini_283_121/frozendrive';GP=R/'envs/frozendrive/bin/python';TP=R/'envs/streampetr_torch111_cu113/bin/python'
state={'status':'running','started':time.time()}
def save(**kw):
 state.update(kw);p=O/'resume_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def run(cmd,log):
 with (O/log).open('a') as f:subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 if (O/'clean_summary.json').exists() and json.loads((O/'clean_summary.json').read_text()).get('status')=='completed':save(status='completed');sys.exit(0)
 save(phase='autoregressive_two_frame_test');run([GP,W/'generate_frozendrive_sequence.py','--end',2],'sequence.log')
 save(phase='native_downstream_loading_test');run([TP,W/'export_dataset.py','--generator','frozendrive','--partial'],'partial_export.log');run([TP,W/'validate_synthetic.py','--generator','frozendrive'],'partial_validation.log')
 save(status='waiting_for_conditions',phase='full_training_conditions')
 while True:
  cs=json.loads((O/'conditions/status.json').read_text())
  if cs['status']=='completed':break
  if cs['status']=='failed':raise RuntimeError('Condition preparation failed; inspect conditions/status.json')
  time.sleep(30)
 save(status='running',phase='full_native_generation');run([GP,W/'generate_frozendrive_sequence.py'],'sequence.log')
 save(phase='clean_finetuning_and_real_test');run([TP,W/'train_clean.py','--generator','frozendrive'],'clean_train_supervisor.log')
 save(status='completed',phase='all_clean_results_ready',finished=time.time())
except Exception as ex:save(status='failed',error=str(ex),traceback=traceback.format_exc(),finished=time.time());raise
