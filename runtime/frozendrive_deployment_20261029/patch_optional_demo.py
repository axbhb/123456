from pathlib import Path
W=Path(__file__).resolve().parent;p=W/'source/WorldDreamer/tools/test_occ.py';s=p.read_text();imp='from data.demo_data.img_style import style_dict\n'
if imp in s:
 s=s.replace(imp,'');s=s.replace('if cfg.runner.validation_index == \'demo\' and batch_index == 0:\n','if cfg.runner.validation_index == \'demo\' and batch_index == 0:\n            # This optional demo asset is not shipped in the upstream repository.\n            from data.demo_data.img_style import style_dict\n')
 p.write_text(s)
print('Demo-only style import deferred; normal validation path preserved')
