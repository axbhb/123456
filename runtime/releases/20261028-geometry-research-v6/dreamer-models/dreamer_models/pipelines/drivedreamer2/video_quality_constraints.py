"""Shared differentiable quality proxies for fixed-input PhysCond optimization.

These are paired deviation constraints, not a realism certificate. Videos use
B,T,V,C,H,W throughout; metric callbacks see flattened B*T*V,C,H,W frames.
No LPIPS network is downloaded or silently substituted. Callers must supply a
frozen perceptual metric and temporal feature extractor when those terms are
active. Independent FID/FVD and blind review remain separate evaluations.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Callable, Optional

import torch
from torch import Tensor, nn


TERMS = frozenset({'denoise', 'perceptual', 'temporal'})
PAIRING_FIELDS = frozenset({
    'scene_id', 'model_fingerprint', 'seed', 'scheduler', 'views',
    'resolution', 'preprocessing',
})


def _tensor(value, name):
    if not isinstance(value, Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if not value.is_floating_point() or value.numel() == 0:
        raise ValueError(f'{name} must be a nonempty floating-point tensor.')
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f'{name} contains non-finite values.')
    return value


def _video(value, name):
    _tensor(value, name)
    if value.ndim != 6 or any(size <= 0 for size in value.shape):
        raise ValueError(f'{name} must have nonempty B,T,V,C,H,W layout.')
    return value


def _same_tensor_contract(value, reference, name):
    _tensor(value, name)
    if value.shape != reference.shape:
        raise ValueError(f'{name} shape does not match its clean reference.')
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f'{name} device/dtype must match its clean reference.')


def _json_value(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('Pairing metadata keys must be strings.')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError('Pairing metadata must contain finite JSON values only.')


def _pairing_json(pairing, shape):
    if not isinstance(pairing, Mapping):
        raise TypeError('pairing must be a mapping.')
    missing = PAIRING_FIELDS - set(pairing)
    if missing:
        raise ValueError(f'Missing pairing metadata: {sorted(missing)}.')
    metadata = _json_value(pairing)
    batch, _, views, _, height, width = shape
    scenes = metadata['scene_id']
    if isinstance(scenes, str):
        valid_scenes = bool(scenes.strip()) and batch == 1
    else:
        valid_scenes = (isinstance(scenes, list) and len(scenes) == batch
                        and all(isinstance(x, str) and x.strip() for x in scenes))
    if not valid_scenes:
        raise ValueError('scene_id must identify each batch item (string only for B=1).')
    if not isinstance(metadata['model_fingerprint'], str) or not metadata['model_fingerprint'].strip():
        raise ValueError('model_fingerprint must be a nonempty string.')
    seeds = metadata['seed']
    valid_seed = lambda x: isinstance(x, int) and not isinstance(x, bool) and x >= 0
    if not (valid_seed(seeds) or (isinstance(seeds, list) and len(seeds) == batch
                                and all(valid_seed(x) for x in seeds))):
        raise ValueError('seed must be a nonnegative integer or one integer per batch item.')
    cameras = metadata['views']
    if not (isinstance(cameras, list) and len(cameras) == views
            and all(isinstance(x, str) and x.strip() for x in cameras)
            and len(set(cameras)) == views):
        raise ValueError('views must name every camera in tensor order, without duplicates.')
    resolution = metadata['resolution']
    if not (isinstance(resolution, list) and len(resolution) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in resolution)
            and resolution == [height, width]):
        raise ValueError('resolution must be [height, width] matching the video.')
    for name in ('scheduler', 'preprocessing'):
        value = metadata[name]
        if not isinstance(value, (str, dict)) or not value:
            raise ValueError(f'{name} must be a nonempty identifier or configuration mapping.')
    return json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _cache_key(pairing_json, shape):
    payload = json.dumps({'schema': 1, 'layout': list(shape),
                          'pairing': json.loads(pairing_json)},
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _trace(value, name):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f'{name} must be a nonempty step-key to tensor mapping.')
    for key, tensor in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f'{name} step keys must be nonempty strings.')
        _tensor(tensor, f'{name}[{key}]')
    return value


def _positive_number(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number.')
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        qualifier = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be finite and {qualifier}.')
    return number


def _frozen_callback(callback, name):
    if not callable(callback):
        raise TypeError(f'{name} must be an explicitly configured callable.')
    if isinstance(callback, nn.Module):
        if callback.training:
            raise ValueError(f'{name} must be in eval mode.')
        if any(parameter.requires_grad for parameter in callback.parameters()):
            raise ValueError(f'{name} parameters must be frozen; input gradients stay enabled.')
    return callback


class QualityReference:
    """Detached clean outputs and an exact pairing identity.

    Required pairing fields: scene_id, model_fingerprint, seed, scheduler,
    views (ordered camera identifiers), resolution ([H,W]), preprocessing.
    Include the scheduler configuration and image normalization/cropping in
    these fields; class names alone need not uniquely identify a generation.
    Extra JSON metadata is also included in cache_key. For B>1 scene_id must
    provide one identifier per item. A scalar seed identifies a batch RNG.
    clean_denoise maps explicit timestep identifiers to noise predictions.
    """

    def __init__(self, clean_video: Tensor, pairing: Mapping,
                 clean_denoise: Optional[Mapping[str, Tensor]] = None):
        _video(clean_video, 'clean_video')
        self._pairing_json = _pairing_json(pairing, clean_video.shape)
        self.clean_video = clean_video.detach().clone()
        self.cache_key = _cache_key(self._pairing_json, clean_video.shape)
        if clean_denoise is None:
            self.clean_denoise = None
        else:
            _trace(clean_denoise, 'clean_denoise')
            snapshots = {}
            for key, value in clean_denoise.items():
                if value.device != clean_video.device:
                    raise ValueError('clean_denoise must be on the clean_video device.')
                snapshots[key] = value.detach().clone()
            self.clean_denoise = MappingProxyType(snapshots)

    @property
    def pairing(self):
        return json.loads(self._pairing_json)


@dataclass(frozen=True)
class QualityResult:
    """Tensor metrics retain gradients; feasibility flags are detached checks."""

    penalty: Tensor
    per_metric: Mapping[str, Tensor]
    feasible: bool
    per_metric_feasible: Mapping[str, bool]
    normalized: Mapping[str, Tensor]
    cache_key: str


class QualityEvaluator:
    """Compute active paired quality proxies exactly once per generated video.

    thresholds selects terms from denoise, perceptual, temporal. weights must
    use the same keys (default all one). Perceptual callback takes two N,C,H,W
    tensors and returns one nonnegative distance per frame, shaped N or N,1,...
    (e.g. an externally supplied LPIPS network). Temporal callback maps frames
    to N,... features, with at least one feature dimension. Function callbacks
    must themselves use frozen deterministic components, without detaching
    the attacked input. No metric model or weights are loaded by this module.
    """

    def __init__(self, reference: QualityReference, *, thresholds: Mapping[str, float],
                 weights: Optional[Mapping[str, float]] = None,
                 perceptual_metric: Optional[Callable] = None,
                 temporal_feature_extractor: Optional[Callable] = None,
                 epsilon: float = 1e-8):
        if not isinstance(reference, QualityReference):
            raise TypeError('reference must be a QualityReference.')
        if not isinstance(thresholds, Mapping) or not thresholds:
            raise ValueError('At least one quality threshold must be configured.')
        unknown = set(thresholds) - TERMS
        if unknown:
            raise ValueError(f'Unknown quality terms: {sorted(unknown)}.')
        self.thresholds = MappingProxyType({
            name: _positive_number(value, f'threshold[{name}]')
            for name, value in thresholds.items()
        })
        if weights is None:
            weights = {name: 1.0 for name in thresholds}
        if not isinstance(weights, Mapping) or set(weights) != set(thresholds):
            raise ValueError('weights must have exactly the active quality term keys.')
        self.weights = MappingProxyType({
            name: _positive_number(value, f'weight[{name}]', allow_zero=True)
            for name, value in weights.items()
        })
        self.epsilon = _positive_number(epsilon, 'epsilon')
        self.reference = reference
        self.perceptual_metric = perceptual_metric
        self.temporal_feature_extractor = temporal_feature_extractor
        if 'denoise' in thresholds and reference.clean_denoise is None:
            raise ValueError('denoise requires clean_denoise in the reference.')
        if 'perceptual' in thresholds:
            _frozen_callback(perceptual_metric, 'perceptual_metric')
        if 'temporal' in thresholds:
            _frozen_callback(temporal_feature_extractor, 'temporal_feature_extractor')
            if reference.clean_video.shape[1] < 2:
                raise ValueError('temporal requires at least two video frames.')

    def evaluate(self, video: Tensor, *, pairing: Mapping,
                 denoise_predictions: Optional[Mapping[str, Tensor]] = None) -> QualityResult:
        _video(video, 'video')
        clean = self.reference.clean_video
        _same_tensor_contract(video, clean, 'video')
        actual_pairing = _pairing_json(pairing, video.shape)
        if actual_pairing != self.reference._pairing_json:
            raise ValueError('Attack/clean pairing mismatch; rebuild the quality reference.')
        batch, time, views, channels, height, width = video.shape
        count = batch * time * views
        attack_frames = video.reshape(count, channels, height, width)
        clean_frames = clean.reshape(count, channels, height, width)
        metrics = {}
        if 'denoise' in self.thresholds:
            _trace(denoise_predictions, 'denoise_predictions')
            clean_trace = self.reference.clean_denoise
            if set(denoise_predictions) != set(clean_trace):
                raise ValueError('Denoise timestep keys must match the clean trace exactly.')
            deviations = []
            for key, clean_prediction in clean_trace.items():
                prediction = denoise_predictions[key]
                _same_tensor_contract(prediction, clean_prediction, f'denoise_predictions[{key}]')
                # Accumulate low-precision predictions in float32 so epsilon
                # does not underflow for a zero clean noise prediction.
                compute_dtype = (torch.float32 if prediction.dtype in
                                 (torch.float16, torch.bfloat16) else prediction.dtype)
                prediction_compute = prediction.to(dtype=compute_dtype)
                clean_compute = clean_prediction.to(dtype=compute_dtype)
                deviations.append((prediction_compute - clean_compute).square().mean()
                                  / (clean_compute.square().mean() + self.epsilon))
            metrics['denoise'] = torch.stack(deviations).mean()
        if 'perceptual' in self.thresholds:
            distances = self.perceptual_metric(attack_frames, clean_frames)
            _tensor(distances, 'perceptual distances')
            if distances.ndim < 1 or distances.shape[0] != count or distances.numel() != count:
                raise ValueError('perceptual_metric must return one distance per frame.')
            if distances.device != video.device:
                raise ValueError('perceptual distances must remain on the video device.')
            if bool((distances.detach() < 0).any().item()):
                raise ValueError('perceptual distances must be nonnegative.')
            metrics['perceptual'] = distances.mean()
        if 'temporal' in self.thresholds:
            attack_features = self.temporal_feature_extractor(attack_frames)
            _tensor(attack_features, 'temporal attack features')
            with torch.no_grad():
                clean_features = self.temporal_feature_extractor(clean_frames)
                _tensor(clean_features, 'temporal clean features')
                clean_features = clean_features.detach()
            _same_tensor_contract(attack_features, clean_features, 'temporal features')
            if (attack_features.ndim < 2 or attack_features.shape[0] != count
                    or attack_features.device != video.device):
                raise ValueError('temporal extractor must return N,... features on the video device.')
            feature_shape = (batch, time, views) + tuple(attack_features.shape[1:])
            attacked = attack_features.reshape(feature_shape)
            reference = clean_features.reshape(feature_shape)
            metrics['temporal'] = ((attacked[:, 1:] - attacked[:, :-1])
                                   - (reference[:, 1:] - reference[:, :-1])).abs().mean()
        normalized = {}
        feasible = {}
        penalties = []
        for name, value in metrics.items():
            _tensor(value, f'quality metric {name}')
            if value.ndim != 0 or bool((value.detach() < 0).item()):
                raise ValueError(f'quality metric {name} must be a nonnegative scalar.')
            normalized[name] = value / self.thresholds[name]
            feasible[name] = bool((value.detach() <= self.thresholds[name]).item())
            penalties.append(self.weights[name] * (normalized[name] - 1).clamp_min(0).square())
        penalty = torch.stack(penalties).sum()
        _tensor(penalty, 'quality penalty')
        return QualityResult(
            penalty=penalty, per_metric=MappingProxyType(metrics),
            feasible=all(feasible.values()), per_metric_feasible=MappingProxyType(feasible),
            normalized=MappingProxyType(normalized), cache_key=self.reference.cache_key,
        )
