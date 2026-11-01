"""Use both existing coordination locks before the original usage-accounted launcher."""
from pathlib import Path
import fcntl,runpy,os
R=Path(os.environ.get('TABLE2_ROOT','/srv/table2'))
# Another existing queue uses the gpu0 lock while the original launcher uses the
# canonical lock. Acquiring both existing locks avoids competing ledger writers.
with (R/'frameworks/gpu_budget_20261028_gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 runpy.run_path(str(R/'frameworks/20261028-v1/run_gpu_smoke.py'),run_name='__main__')
