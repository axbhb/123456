"""Differentiable, fixed-condition DD2 generation for outer optimization.

This adapter supports one scene represented by a six-camera horizontal panorama.
It does not implement world-coordinate rendering, label construction, or an
attack. All model components must already reside on one execution device.
CUDA is disabled by default; allow_cuda is a technical guard, not authorization.
"""
from __future__ import annotations

import copy
import math
import random
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import torch
from torch import Tensor, nn

from .generation_pairing import (LoadedModelIdentity, actual_generation_pairing,
                                 capture_loaded_model_identity)


@dataclass(frozen=True)
class FixedInputGenerationOutput:
    video: Tensor
    denoise_predictions: Mapping[str, Tensor]
    logs: Mapping
    pairing: Mapping


_RAW_KEYS = frozenset({'grounding_downsampler_input', 'box_downsampler_input',
                       'img_cond', 'video_cond'})
_STATE_FIELDS = ('frame_num', '_cross_attention_kwargs', '_guidance_scale', '_num_timesteps')
_MISSING = object()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def _tensor(value, name):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f'{name} must be a floating-point torch.Tensor.')
    if not value.numel() or not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f'{name} must be nonempty and finite.')
    return value


def _modules(pipeline):
    roots = [value for value in vars(pipeline).values() if isinstance(value, nn.Module)]
    # DiffusionPipeline registers components as attributes, while some adapters
    # expose them only through components. Deduplicate shared modules/parameters.
    components = getattr(pipeline, 'components', None)
    if isinstance(components, Mapping):
        roots.extend(value for value in components.values() if isinstance(value, nn.Module))
    found = {}
    for root in roots:
        for module in root.modules():
            found[id(module)] = module
    if not found:
        raise ValueError('Pipeline has no loaded torch model components.')
    return tuple(found.values())


@contextmanager
def _isolated_pipeline(pipeline, device, seed):
    modules = _modules(pipeline)
    parameters = {}
    for module in modules:
        if getattr(module, '_hf_hook', None) is not None:
            raise ValueError('Offload/device hooks are unsupported; place all components on one device first.')
        for value in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            if value.device != device:
                raise ValueError('All model parameters/buffers must already be on the execution device; no weight moves are allowed.')
        for parameter in module.parameters(recurse=False):
            parameters[id(parameter)] = (parameter, parameter.requires_grad)
    flags = [(module, module.training) for module in modules]
    fusers = [(module, vars(module)['enabled']) for module in modules if 'enabled' in vars(module)]
    attributes = {name: getattr(pipeline, name, _MISSING) for name in _STATE_FIELDS}
    original_scheduler = pipeline.scheduler
    scheduler = copy.deepcopy(original_scheduler)
    if scheduler is original_scheduler:
        raise ValueError('Scheduler deepcopy must create an independent instance.')
    python_rng = random.getstate()
    numpy = sys.modules.get('numpy')
    numpy_rng = numpy.random.get_state() if numpy is not None else None
    cuda_devices = [] if device.type == 'cpu' else [device.index]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            if numpy_rng is not None:
                numpy.random.seed(seed % (2 ** 32))
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            pipeline.scheduler = scheduler
            for parameter, _ in parameters.values():
                parameter.requires_grad_(False)
            for module, _ in flags:
                module.training = False
            yield scheduler
    finally:
        pipeline.scheduler = original_scheduler
        for name, value in attributes.items():
            if value is _MISSING:
                if hasattr(pipeline, name):
                    delattr(pipeline, name)
            else:
                setattr(pipeline, name, value)
        for module, enabled in fusers:
            module.enabled = enabled
        for module, training in flags:
            module.training = training
        for parameter, requires_grad in parameters.values():
            parameter.requires_grad_(requires_grad)
        random.setstate(python_rng)
        if numpy_rng is not None:
            numpy.random.set_state(numpy_rng)


def _prepare_raw_inputs(input_dict, num_frames, height, width):
    if not isinstance(input_dict, Mapping):
        raise TypeError('input_dict must contain raw condition tensors.')
    unknown = set(input_dict) - _RAW_KEYS
    if unknown:
        raise ValueError(f'Only raw condition inputs are accepted, not features/latents: {sorted(unknown)}.')
    if not any(input_dict.get(name) is not None for name in
               ('grounding_downsampler_input', 'box_downsampler_input')):
        raise ValueError('At least one raw map/box condition tensor is required.')
    if input_dict.get('img_cond') is not None and input_dict.get('video_cond') is not None:
        raise ValueError('img_cond and video_cond are mutually exclusive native modes.')
    result = {}
    for name, value in input_dict.items():
        if value is None:
            continue
        _tensor(value, name)
        expected_frames = 1 if name == 'img_cond' else num_frames
        if value.ndim != 4 or value.shape[0] != expected_frames or tuple(value.shape[-2:]) != (height, width):
            raise ValueError(f'{name} must have native [{expected_frames},C,{height},{width}] panorama layout.')
        if name in {'img_cond', 'video_cond'} and value.requires_grad:
            raise ValueError(f'{name} is fixed context and must not require gradients.')
        # Snapshot protects a returned graph from later external input updates;
        # clone retains the raw map/box input gradient path.
        result[name] = value.clone()
    return result


def _camera_video(decoded, *, num_frames, height, width, clip_latents):
    _tensor(decoded, 'decoded video')
    if decoded.ndim != 5:
        raise ValueError('decode_latents must return B,C,T,H,W (or V,C,T,H,W for clip_latents).')
    view_width = width // 6
    if clip_latents:
        expected = (6, 3, num_frames, height, view_width)
        if tuple(decoded.shape) != expected:
            raise ValueError(f'Unexpected clipped-camera decoder output; expected {expected}.')
        result = decoded.permute(2, 0, 1, 3, 4).unsqueeze(0)
    else:
        expected = (1, 3, num_frames, height, width)
        if tuple(decoded.shape) != expected:
            raise ValueError(f'Unexpected panorama decoder output; expected {expected}.')
        result = decoded[0].reshape(3, num_frames, height, 6, view_width)
        result = result.permute(1, 3, 0, 2, 4).unsqueeze(0)
    return (result / 2 + 0.5).clamp(0, 1)


def generate_fixed_input(
    pipeline, *, prompt_embeddings: Tensor, input_dict: Mapping,
    scene_id: str, model_identity: LoadedModelIdentity,
    seed: int, camera_names, camera_layout: str = 'panorama',
    height: int, width: int, num_frames: int, num_inference_steps: int = 25,
    min_guidance_scale: float = 1.0, max_guidance_scale: float = 3.0,
    fps: int = 7, motion_bucket_id: int = 127, noise_aug_strength: float = 0.02,
    decode_chunk_size: Optional[int] = None, scheduled_sampling_beta: float = 0.3,
    clip_latents: bool = False, gradient_last_steps: Optional[int] = None,
    differentiable: bool = True, allow_cuda: bool = False,
    initial_latents: Optional[Tensor] = None,
) -> FixedInputGenerationOutput:
    """Generate final RGB tensors without updating conditions inside denoising.

    B=1 and six explicitly ordered cameras are required. prompt_embeddings,
    image/video context and optional initial noise are fixed, not attack
    variables. initial_latents follows native prepare_latents semantics: it is
    unscaled noise, before scheduler.init_noise_sigma. Full differentiation is
    the default. gradient_last_steps truncates only the earlier denoising
    trajectory; raw condition features remain connected in retained steps.
    Set differentiable=False for paired reference/candidate evaluation.

    scene_id must come from the source sample. model_identity must be captured
    from actual loaded state with capture_loaded_model_identity after placement.
    The returned pairing is built from this call and must be retained for clean
    references and independently reconstructed by native replay adapters.

    Uses the existing native helper methods, never the no_grad __call__.
    Raw raster/box rendering inputs are supported; this is not a world-coordinate
    geometry adapter. Model offload hooks/multi-device placement are rejected.
    """
    if not isinstance(allow_cuda, bool) or not isinstance(differentiable, bool):
        raise TypeError('allow_cuda and differentiable must be booleans.')
    device = torch.device(pipeline._execution_device)
    if device.type == 'cuda' and not allow_cuda:
        raise PermissionError('CUDA execution is disabled; allow_cuda=True requires separate user authorization.')
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Only explicit CPU or CUDA execution is supported.')
    if device.type == 'cuda' and device.index is None:
        raise ValueError('CUDA execution device must include an explicit index.')
    if not isinstance(model_identity, LoadedModelIdentity):
        raise TypeError('Capture a LoadedModelIdentity from actual loaded weights first.')
    model_identity.validate(pipeline)
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError('scene_id must identify the source scene for this call.')
    if differentiable and torch.is_inference_mode_enabled():
        raise ValueError('Differentiable generation cannot run under inference_mode.')
    seed = _integer(seed, 'seed', minimum=0)
    if seed >= 2 ** 63:
        raise ValueError('seed must be smaller than 2**63.')
    height = _integer(height, 'height')
    width = _integer(width, 'width')
    num_frames = _integer(num_frames, 'num_frames')
    num_inference_steps = _integer(num_inference_steps, 'num_inference_steps')
    decode_chunk_size = num_frames if decode_chunk_size is None else _integer(decode_chunk_size, 'decode_chunk_size')
    if not isinstance(clip_latents, bool):
        raise TypeError('clip_latents must be a boolean.')
    if clip_latents and num_frames % decode_chunk_size:
        raise ValueError('clip_latents decode_chunk_size must divide num_frames to avoid crossing camera boundaries.')
    if camera_layout != 'panorama':
        raise ValueError('Only explicit six-camera panorama layout is supported; no layout inference.')
    if (not isinstance(camera_names, (tuple, list)) or len(camera_names) != 6
            or not all(isinstance(name, str) and name for name in camera_names)
            or len(set(camera_names)) != 6):
        raise ValueError('camera_names must contain six unique names in left-to-right panorama order.')
    if width % 6 or (clip_latents and width % (6 * pipeline.vae_scale_factor)):
        raise ValueError('Panorama and latent width must divide exactly into six views.')
    for name, value in [('min_guidance_scale', min_guidance_scale),
                        ('max_guidance_scale', max_guidance_scale),
                        ('scheduled_sampling_beta', scheduled_sampling_beta),
                        ('noise_aug_strength', noise_aug_strength)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite.')
    if not (0 <= scheduled_sampling_beta <= 1 and 0 <= min_guidance_scale <= max_guidance_scale
            and noise_aug_strength >= 0):
        raise ValueError('Invalid guidance, sampling beta, or noise augmentation range.')
    fps = _integer(fps, 'fps')
    motion_bucket_id = _integer(motion_bucket_id, 'motion_bucket_id', minimum=0)
    if gradient_last_steps is not None:
        gradient_last_steps = _integer(gradient_last_steps, 'gradient_last_steps')
        if gradient_last_steps > num_inference_steps:
            raise ValueError('gradient_last_steps must not exceed requested denoising steps.')
    if isinstance(prompt_embeddings, Tensor) and prompt_embeddings.device != device:
        raise ValueError('prompt_embeddings must already be on the execution device.')
    _tensor(prompt_embeddings, 'prompt_embeddings')
    if prompt_embeddings.ndim < 2 or prompt_embeddings.shape[0] != 1:
        raise ValueError('Only one scene/prompt is supported; B>1 is not inferred from camera count.')
    if prompt_embeddings.requires_grad:
        raise ValueError('prompt_embeddings must be fixed, not an editable feature variable.')
    if prompt_embeddings.device != device:
        raise ValueError('prompt_embeddings must already be on the execution device.')
    # Reject off-device inputs before finite checks, which themselves execute
    # tensor operations and must never initialize CUDA in the CPU-only path.
    if isinstance(input_dict, Mapping):
        for name, value in input_dict.items():
            if isinstance(value, Tensor) and value.device != device:
                raise ValueError(f'{name} must already be on the execution device.')
    with torch.set_grad_enabled(differentiable):
        raw = _prepare_raw_inputs(input_dict, num_frames, height, width)
    for name, value in raw.items():
        if value.device != device:
            raise ValueError(f'{name} must already be on the execution device.')
    if initial_latents is not None:
        if isinstance(initial_latents, Tensor) and initial_latents.device != device:
            raise ValueError('initial_latents must already be on the execution device.')
        _tensor(initial_latents, 'initial_latents')
        expected = (1, num_frames, pipeline.vae.config.latent_channels,
                    height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)
        if initial_latents.requires_grad or tuple(initial_latents.shape) != expected:
            raise ValueError(f'initial_latents must be fixed unscaled noise of shape {expected}.')
        if initial_latents.device != device or initial_latents.dtype != prompt_embeddings.dtype:
            raise ValueError('initial_latents device/dtype must match prompt_embeddings.')
        initial_latents = initial_latents.detach().clone()
    predictions = {}
    with _isolated_pipeline(pipeline, device, seed) as scheduler:
        with torch.set_grad_enabled(differentiable):
            generator = torch.Generator(device=device).manual_seed(seed)
            context = pipeline._prepare_prompt_context(
                prompt_embeddings=prompt_embeddings.detach().clone(), height=height, width=width,
                num_frames=num_frames, num_inference_steps=num_inference_steps,
                min_guidance_scale=min_guidance_scale, max_guidance_scale=max_guidance_scale,
                fps=fps, motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength,
                num_videos_per_prompt=1, scheduled_sampling_beta=scheduled_sampling_beta,
                generator=generator, latents=initial_latents, cross_attention_kwargs=None,
            )
            if context['batch_size'] != 1 or context['num_cams'] != 1:
                raise ValueError('Native context is not a single panorama scene.')
            conditions = pipeline._prepare_condition_features(
                input_dict=raw, batch_size=1, height=height, width=width, device=device,
                prompt_dtype=context['prompt_embeddings'].dtype,
                do_classifier_free_guidance=context['do_classifier_free_guidance'],
            )
            timesteps = context['timesteps']
            if not len(timesteps):
                raise ValueError('Scheduler returned no timesteps.')
            retained = len(timesteps) if gradient_last_steps is None else gradient_last_steps
            if retained > len(timesteps):
                raise ValueError('gradient_last_steps exceeds actual scheduler timesteps.')
            first_grad_step = len(timesteps) - retained
            latents = context['latents']
            pipeline.enable_fuser(True)
            for index, timestep in enumerate(timesteps):
                if index == context['num_grounding_steps']:
                    pipeline.enable_fuser(False)
                keep_grad = differentiable and index >= first_grad_step
                if index == first_grad_step and first_grad_step:
                    latents = latents.detach()
                with torch.set_grad_enabled(keep_grad):
                    prediction = pipeline._predict_noise(
                        latents, timestep, context['prompt_embeddings'], context['added_time_ids'],
                        context['num_cams'], condition_latents=conditions['condition_latents'],
                        mask_condition=conditions['mask_condition'],
                        grounding_features=conditions['grounding_features'],
                        box_features=conditions['box_features'],
                        do_classifier_free_guidance=context['do_classifier_free_guidance'],
                        scheduler=scheduler,
                    )
                    _tensor(prediction, 'noise prediction')
                    key = f'step_{index:04d}:t={timestep.item()}'
                    predictions[key] = prediction
                    latents = scheduler.step(prediction, timestep, latents).prev_sample
            decoded = pipeline.decode_latents(latents, num_frames, decode_chunk_size, clip_latents)
            output = _camera_video(decoded, num_frames=num_frames, height=height,
                                   width=width, clip_latents=clip_latents)
            pairing = actual_generation_pairing(
                pipeline=pipeline, model_identity=model_identity, scene_id=scene_id, seed=seed,
                camera_names=camera_names, video_shape=output.shape, scheduler=scheduler,
                sampling={'requested_steps': num_inference_steps,
                          'timesteps': [step.item() for step in timesteps],
                          'min_guidance_scale': min_guidance_scale,
                          'max_guidance_scale': max_guidance_scale, 'fps': fps,
                          'motion_bucket_id': motion_bucket_id,
                          'noise_aug_strength': noise_aug_strength,
                          'scheduled_sampling_beta': scheduled_sampling_beta,
                          'decode_chunk_size': decode_chunk_size},
                prompt_embeddings=prompt_embeddings, raw_inputs=raw,
                initial_latents=initial_latents, clip_latents=clip_latents)
    logs = {
        'mode': 'fixed_input', 'input_space': 'raw_condition_raster',
        'world_coordinate_geometry': False, 'seed': seed, 'device': str(device),
        'batch_size': 1, 'camera_names': list(camera_names), 'camera_layout': camera_layout,
        'output_layout': 'B,T,V,C,H,W', 'output_range': '[0,1]',
        'clip_latents': clip_latents, 'denoise_steps': len(predictions),
        'differentiable': differentiable, 'gradient_last_steps': gradient_last_steps,
        'gradient_truncated': gradient_last_steps is not None and first_grad_step > 0,
        'differentiated_denoise_steps': retained if differentiable else 0,
        'conditions_prepared_once': True, 'scheduler_reset': True,
        'scope': 'fixed_raster_input_generation; not_world_coordinate_renderer; no_training_harm_evidence',
    }
    return FixedInputGenerationOutput(output, predictions, logs, pairing)
