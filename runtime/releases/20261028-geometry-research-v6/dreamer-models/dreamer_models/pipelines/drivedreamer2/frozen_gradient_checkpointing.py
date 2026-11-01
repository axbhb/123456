"""Recompute frozen DD2 blocks without changing their forward arithmetic."""
from contextlib import contextmanager
import torch
from torch.utils.checkpoint import checkpoint

def _requires_grad(value):
    if isinstance(value,torch.Tensor):return value.requires_grad
    if isinstance(value,dict):return any(_requires_grad(v) for v in value.values())
    if isinstance(value,(tuple,list)):return any(_requires_grad(v) for v in value)
    return False

@contextmanager
def checkpoint_modules(named_modules):
    originals=[];stats=dict(modules=[],checkpointed_calls=0)
    try:
        for name,module in named_modules:
            if any(p.requires_grad for p in module.parameters()):raise ValueError('Only frozen modules are supported')
            if module.training:raise ValueError('Checkpointed modules must remain in evaluation mode')
            original=module.forward;originals.append((module,original));stats['modules'].append(name)
            def make_forward(original,module):
                def wrapped(*args,**kwargs):
                    if not torch.is_grad_enabled() or not (_requires_grad(args) or _requires_grad(kwargs)):
                        return original(*args,**kwargs)
                    # DD2 restores fuser flags after forward. Replay must use the
                    # same flags even though backward happens outside that scope.
                    flags=[(m,vars(m)['enabled']) for m in module.modules() if 'enabled' in vars(m)]
                    def replay(*values,**options):
                        old=[(m,vars(m)['enabled']) for m,_ in flags]
                        try:
                            for m,enabled in flags:m.enabled=enabled
                            return original(*values,**options)
                        finally:
                            for m,enabled in old:m.enabled=enabled
                    stats['checkpointed_calls']+=1
                    return checkpoint(replay,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
                return wrapped
            module.forward=make_forward(original,module)
        yield stats
    finally:
        for module,original in reversed(originals):module.forward=original

@contextmanager
def checkpoint_frozen_dd2(pipeline, *, vae_blockwise=False):
    selected=[];prefixes=[]
    for name,module in pipeline.unet.named_modules():
        if type(module).__name__ not in {'SpatioTemporalResBlock','TransformerSpatioTemporalModel'}:continue
        if any(name==p or name.startswith(p+'.') for p in prefixes):continue
        prefixes.append(name);selected.append(('unet.'+name,module))
    if not selected:raise ValueError('Expected native DD2 spatial/temporal blocks')
    if vae_blockwise:
        selected.extend(vae_checkpoint_blocks(pipeline.vae))
    else:
        selected.append(('vae.decoder',pipeline.vae.decoder))
    selected.append(('box_downsampler',pipeline.box_downsampler))
    with checkpoint_modules(selected) as stats:yield stats


def vae_checkpoint_blocks(vae):
    """Bound decoder recomputation memory without changing native arithmetic."""
    decoder = vae.decoder
    return [('vae.decoder.mid_block', decoder.mid_block)] + [
        ('vae.decoder.up_blocks.%d' % i, block)
        for i, block in enumerate(decoder.up_blocks)
    ]
