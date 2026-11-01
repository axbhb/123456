from pathlib import Path
W=Path(__file__).resolve().parent;p=W/'source/WorldDreamer/third_party/bevfusion/mmdet3d/ops/spconv/conv.py';s=p.read_text()
helper='''\ndef _register_frozendrive_sparse_conv(cls):
    # MMCV 1.7 pre-registers these names; this vendored BEVFusion uses its own implementation.
    existing = CONV_LAYERS.get(cls.__name__)
    replace_builtin = existing is not None and existing.__module__ == "mmcv.ops.sparse_conv"
    return CONV_LAYERS.register_module(module=cls, force=replace_builtin)
\n'''
if '_register_frozendrive_sparse_conv' not in s:
 s=s.replace('from mmcv.cnn import CONV_LAYERS\n','from mmcv.cnn import CONV_LAYERS\n'+helper)
 s=s.replace('@CONV_LAYERS.register_module()','@_register_frozendrive_sparse_conv')
 p.write_text(s)
print('Scoped MMCV builtin sparse-convolution compatibility patch applied')
