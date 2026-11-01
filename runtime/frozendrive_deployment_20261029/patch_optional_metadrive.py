from pathlib import Path
W=Path(__file__).resolve().parent;p=W/'source/WorldDreamer/dataset/map_utils.py';s=p.read_text()
a='from metadrive.scenario import ScenarioDescription as SD\nfrom metadrive.type import MetaDriveType\n'
if a in s:
 s=s.replace(a,'').replace('def extract_map_features(map_api, center, radius=500):\n','def extract_map_features(map_api, center, radius=500):\n    # MetaDrive is only required by this optional NuPlan-to-simulator converter.\n    from metadrive.scenario import ScenarioDescription as SD\n    from metadrive.type import MetaDriveType\n')
 p.write_text(s)
print('MetaDrive imports deferred to the only function using them; nuScenes generation unchanged')
