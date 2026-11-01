from pathlib import Path
import subprocess
W=Path(__file__).resolve().parent;E=W.parent/'envs/frozendrive';p=E/'lib/python3.9/site-packages/moviepy'
assert not p.is_symlink();p.resolve().relative_to(E.resolve())
backup=W/'compatibility_backup';backup.mkdir(exist_ok=True);p.rename(backup/'moviepy_mixed_versions')
subprocess.run([str(E/'bin/python'),'-m','pip','install','--force-reinstall','--no-index','--no-deps','--no-build-isolation',str(W/'wheels/moviepy-1.0.3.tar.gz')],check=True)
