"""Offload autograd activations without materializing broadcast model weights."""
from contextlib import contextmanager
# CPU source mode exists for deterministic broadcast-view regression tests.
import torch

def _storage_pointer(tensor):
    storage=tensor.untyped_storage() if hasattr(tensor,'untyped_storage') else tensor.storage()
    return storage.data_ptr()

@contextmanager
def offload_saved_activations(modules,stats=None,*,source_device_types=('cuda',),max_saved_bytes=None):
    if not source_device_types or set(source_device_types)-{'cuda','cpu'}:raise ValueError('Only CPU/CUDA activation sources supported')
    weights=set()
    for module in modules:
        for tensor in list(module.parameters())+list(module.buffers()):
            if tensor.device.type in source_device_types:weights.add((tensor.device,_storage_pointer(tensor)))
    stats={} if stats is None else stats
    stats.update(saved_tensors=0,saved_bytes=0,weight_views_kept=0,broadcast_views_compressed=0)
    def pack(tensor):
        if tensor.device.type not in source_device_types:return tensor
        if (tensor.device,_storage_pointer(tensor)) in weights:
            stats['weight_views_kept']+=1
            return tensor
        shape=tuple(tensor.shape)
        index=tuple(slice(0,1) if stride==0 and size>1 else slice(None) for size,stride in zip(shape,tensor.stride()))
        compact=tensor.detach()[index]
        if tuple(compact.shape)!=shape:stats['broadcast_views_compressed']+=1
        size=compact.numel()*compact.element_size()
        if max_saved_bytes is not None and stats['saved_bytes']+size>max_saved_bytes:
            raise MemoryError('Saved activations exceeded the explicit host-memory allowance; checkpoint finer blocks')
        cpu=compact.to('cpu')
        stats['saved_tensors']+=1;stats['saved_bytes']+=cpu.numel()*cpu.element_size()
        return (tensor.device,cpu,shape)
    def unpack(value):
        if isinstance(value,torch.Tensor):return value
        device,cpu,shape=value
        return cpu.to(device).expand(shape)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):yield stats
