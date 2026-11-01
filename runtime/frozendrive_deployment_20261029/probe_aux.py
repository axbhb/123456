import sys,importlib.util,json
from pathlib import Path
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'source/WorldDreamer/third_party/nuplan-devkit'))
names=['moviepy','h5py','ftfy','omegaconf','antlr4','geopandas','pyogrio','fiona','boto3','pyquaternion','pyinstrument','rasterio','affine','aioboto3','retry','pyarrow','sympy','networkx','mpi4py','tensorpack','termcolor','yacs','addict','sklearn']
print(json.dumps({n:importlib.util.find_spec(n) is not None for n in names}))
try:import nuplan.database.maps_db.map_api;print('NUPLAN_IMPORT_OK')
except Exception as e:import traceback;traceback.print_exc()
