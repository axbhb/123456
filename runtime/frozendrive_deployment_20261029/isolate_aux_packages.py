import shutil
from pathlib import Path
W=Path(__file__).resolve().parent;site=W.parent/'envs/frozendrive/lib/python3.9/site-packages';copied=[]
for name in ['moviepy','imageio','imageio_ffmpeg','decorator','proglog','h5py','h5py.libs','ftfy','regex','safetensors','omegaconf','antlr4','torchdiffeq','tensorpack','msgpack_numpy','zmq','mpi4py','torchpack','tensorboardX','termcolor','yacs','loguru']:
 p=site/name
 if not p.is_symlink():continue
 src=p.resolve();tmp=site/(name+'.frozendrive-local-copy')
 if src.is_dir():shutil.copytree(src,tmp)
 else:shutil.copy2(src,tmp)
 p.unlink();tmp.rename(p);copied.append(name)
print('Private copies:',copied)
