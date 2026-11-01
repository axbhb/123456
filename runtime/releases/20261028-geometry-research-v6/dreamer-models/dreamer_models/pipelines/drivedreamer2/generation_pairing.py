"""Auditable identity for a generation and its paired clean reference.

Capture the loaded model once after placement/loading, then validate its tensor
bindings and PyTorch version counters before each generation. An identity is
invalid after load_state_dict, parameter replacement, or an ordinary in-place
weight update. Code that bypasses PyTorch tracking (for example .data writes or
external device writes) is unsupported: capture a new identity after any such
operation. This is a reproducibility contract, not a hostile-code boundary.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError('Generation configurations must contain JSON values; no repr fallback.')


def _json(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _class_name(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def _config(value, *, required=False):
    config = getattr(value, 'config', None)
    if config is None:
        if required:
            raise ValueError('Scheduler must expose its actual config for paired generation.')
        return None
    if hasattr(config, 'to_dict'):
        config = config.to_dict()
    elif not isinstance(config, Mapping) and hasattr(config, '__dict__'):
        config = vars(config)
    return _jsonable(config)


def _entries(pipeline):
    roots = {name: value for name, value in vars(pipeline).items() if isinstance(value, nn.Module)}
    components = getattr(pipeline, 'components', None)
    if isinstance(components, Mapping):
        roots.update({name: value for name, value in components.items() if isinstance(value, nn.Module)})
    if not roots:
        raise ValueError('A model identity requires loaded torch components.')
    structure, tensors = [], []
    for name, module in sorted(roots.items()):
        structure.append({'name': name, 'class': _class_name(module), 'config': _config(module),
                          'modules': [(n, _class_name(m)) for n, m in module.named_modules()]})
        for kind, items in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
            for key, tensor in items:
                tensors.append((name + '.' + kind + '.' + key, tensor))
    return structure, tensors


def _bindings(tensors):
    return tuple((name, id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                 for name, tensor in tensors)


def _update_tensor(digest, tensor):
    if tensor.layout != torch.strided or tensor.device.type not in {'cpu', 'cuda'}:
        raise ValueError('Identity hashing supports loaded dense CPU/CUDA tensors only.')
    digest.update(_json({'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}).encode('utf-8'))
    flat = tensor.detach().reshape(-1)
    # Bound the transient CPU copy; works for bfloat16 by viewing raw bytes.
    for start in range(0, flat.numel(), 262144):
        byte_array = flat[start:start + 262144].contiguous().cpu().view(torch.uint8).numpy()
        digest.update(memoryview(byte_array))


def tensor_fingerprint(tensor):
    if tensor is None:
        return None
    digest = hashlib.sha256()
    _update_tensor(digest, tensor)
    return 'sha256:' + digest.hexdigest()


@dataclass(frozen=True)
class LoadedModelIdentity:
    fingerprint: str
    pipeline_id: int
    structure_json: str
    bindings: tuple

    def validate(self, pipeline):
        structure, tensors = _entries(pipeline)
        if (id(pipeline) != self.pipeline_id or _json(structure) != self.structure_json
                or _bindings(tensors) != self.bindings):
            raise ValueError('Loaded model changed after identity capture; recapture actual weights.')
        return self.fingerprint


def capture_loaded_model_identity(pipeline, *, allow_cuda=False):
    """Hash real loaded state, once; never accepts a caller-supplied hash string.

    Explicitly call again after loading/changing weights. CUDA reads require the
    same separately authorized allow_cuda opt-in as generation. Hashing is not a
    model forward pass and does not run a download or access a checkpoint path.
    """
    structure, tensors = _entries(pipeline)
    if not tensors:
        raise ValueError('Loaded model identity needs actual state tensors.')
    for _, tensor in tensors:
        if tensor.device.type == 'cuda' and not allow_cuda:
            raise PermissionError('CUDA identity hashing requires explicit allow_cuda authorization.')
    structure_json = _json(structure)
    before = _bindings(tensors)
    digest = hashlib.sha256(('loaded-tensors-config-v1:' + structure_json).encode('utf-8'))
    for name, tensor in tensors:
        digest.update(name.encode('utf-8'))
        _update_tensor(digest, tensor)
    if before != _bindings(tensors):
        raise RuntimeError('Model state changed while its identity was being captured.')
    return LoadedModelIdentity('sha256-loaded-tensors-config-v1:' + digest.hexdigest(),
                               id(pipeline), structure_json, before)


def actual_generation_pairing(*, pipeline, model_identity, scene_id, seed, camera_names,
                              video_shape, scheduler, sampling, prompt_embeddings,
                              raw_inputs, initial_latents, clip_latents):
    if not isinstance(model_identity, LoadedModelIdentity):
        raise TypeError('Capture a LoadedModelIdentity from actual loaded weights before generation.')
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError('Generation needs the explicit source scene_id; it cannot be inferred from pixels.')
    return {
        'scene_id': scene_id,
        'model_fingerprint': model_identity.validate(pipeline),
        'seed': seed,
        'scheduler': {'class': _class_name(scheduler), 'config': _config(scheduler, required=True),
                      'sampling': _jsonable(sampling)},
        'views': list(camera_names),
        'resolution': list(video_shape[-2:]),
        'preprocessing': {'layout': 'B,T,V,C,H,W', 'decode': '(x/2+0.5).clamp(0,1)',
                          'clip_latents': clip_latents, 'camera_layout': 'panorama'},
        'fixed_context': {
            'prompt_embeddings': tensor_fingerprint(prompt_embeddings),
            'img_cond': tensor_fingerprint(raw_inputs.get('img_cond')),
            'video_cond': tensor_fingerprint(raw_inputs.get('video_cond')),
            'initial_latents': tensor_fingerprint(initial_latents),
        },
    }
