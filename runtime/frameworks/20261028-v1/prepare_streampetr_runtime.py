"""Prepare a separate V100-compatible StreamPETR tree from the pinned source."""
import argparse,ast,difflib,hashlib,json,pprint,runpy,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--asset-root',type=Path,required=True);a=p.parse_args()
src=ROOT/'sources/streampetr';dst=a.runtime_root/'streampetr'
if dst.exists():raise ValueError('Runtime destination already exists; preserve existing work')
shutil.copytree(src,dst)
file=dst/'projects/mmdet3d_plugin/models/utils/attention.py';old=file.read_text()
imports='from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func\nfrom flash_attn.bert_padding import unpad_input, pad_input, index_first_axis'
replacement='''try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
except ModuleNotFoundError as error:
    if not (error.name or '').startswith('flash_attn'):
        raise
    flash_attn_unpadded_kvpacked_func = None
    unpad_input = pad_input = index_first_axis = None'''
assert old.count(imports)==1
new=old.replace(imports,replacement)
needle='        assert q.dtype in [torch.float16, torch.bfloat16]'
assert new.count(needle)==1
new=new.replace(needle,"        if flash_attn_unpadded_kvpacked_func is None:\n            raise RuntimeError('FlashAttention unavailable; select official PETRMultiheadAttention on V100')\n"+needle)
ast.parse(new);file.write_text(new)
original=src/'projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py'
cfg={k:v for k,v in runpy.run_path(str(original)).items() if not k.startswith('_')}
cfg['model']['pts_bbox_head']['transformer']['decoder']['transformerlayers']['attn_cfgs'][1].update(type='PETRMultiheadAttention',fp16=True)
cfg['model']['img_backbone']['pretrained']=None
cfg.update(num_gpus=1,batch_size=1,workflow=[('train',1)],dist_params={'backend':'nccl'},log_level='INFO',log_config={'interval':1,'hooks':[{'type':'TextLoggerHook'}]},checkpoint_config=None,evaluation={'interval':1000000},runner={'type':'IterBasedRunner','max_iters':1})
cfg['data'].update(samples_per_gpu=1,workers_per_gpu=0)
raw=str(a.asset_root/'nuscenes_mini_temporal_camera_v1')
for split,role in [('train','train'),('val','calibration'),('test','calibration')]:
 cfg['data'][split].update(data_root=raw,ann_file=str(a.asset_root/'streampetr_subset'/('streampetr_'+role+'.pkl')))
cfg['data']['train']['seq_split_num']=1
cfg['load_from']=str(a.asset_root/'checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth')
cfg['plugin_dir']=str(dst/'projects/mmdet3d_plugin')+'/'
config=dst/'smoke_v100.py';config.write_text('\n\n'.join(k+' = '+pprint.pformat(v,width=110,sort_dicts=False) for k,v in cfg.items())+'\n')
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='pinned/attention.py',tofile='runtime/attention.py'));(a.runtime_root/'streampetr_optional_flash.patch').write_text(patch)
report={'source_commit':'95f64702306ccdb7a78889578b2a55b5deb35b2a','runtime':str(dst),'config':str(config),'original_attention_sha256':hashlib.sha256(old.encode()).hexdigest(),'runtime_attention_sha256':hashlib.sha256(new.encode()).hexdigest(),'changes':['optional FlashAttention import; fail if unavailable Flash layer is selected','official PETRMultiheadAttention with fp16=True','one GPU, one batch, zero data workers, complete-scene grouping','no auxiliary backbone download; strict full checkpoint required'],'model_run':False}
(a.runtime_root/'streampetr_runtime.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
