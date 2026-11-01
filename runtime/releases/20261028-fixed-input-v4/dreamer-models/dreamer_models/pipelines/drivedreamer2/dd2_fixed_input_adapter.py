"""One frozen DD2 call specification for gradients and genuine native replay.

No model loading/downloads, raster construction, or GPU authorization occurs here.
The renderer must explicitly identify its source scene and ordered cameras. Native
replay calls pipeline.__call__, including its native VAE processor; it never calls
fixed_input_generation in place of the native path.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Optional

import torch
from torch import Tensor

from .fixed_input_generation import (
    FixedInputGenerationOutput, _integer, _isolated_pipeline, _prepare_raw_inputs,
    _tensor, generate_fixed_input,
)
from .generation_pairing import LoadedModelIdentity, actual_generation_pairing


@dataclass(frozen=True)
class DD2GenerationSpec:
    scene_id: str
    seed: int
    camera_names: tuple
    height: int
    width: int
    num_frames: int
    num_inference_steps: int = 25
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 3.0
    fps: int = 7
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    decode_chunk_size: Optional[int] = None
    scheduled_sampling_beta: float = 0.3
    clip_latents: bool = False
    gradient_last_steps: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError('scene_id must identify the actual source scene.')
        if not isinstance(self.camera_names, (tuple, list)):
            raise TypeError('camera_names must explicitly list panorama order.')
        names = tuple(self.camera_names)
        if len(names) != 6 or len(set(names)) != 6 or not all(isinstance(x, str) and x for x in names):
            raise ValueError('Six unique camera_names in panorama order are required.')
        object.__setattr__(self, 'camera_names', names)
        for name in ('height', 'width', 'num_frames', 'num_inference_steps', 'fps'):
            _integer(getattr(self, name), name)
        _integer(self.seed, 'seed', 0)
        _integer(self.motion_bucket_id, 'motion_bucket_id', 0)
        if self.seed >= 2 ** 63:
            raise ValueError('seed must be smaller than 2**63.')
        if self.height % 8 or self.width % 8 or self.width % 6:
            raise ValueError('Native size must divide by 8, and width must split into six views.')
        chunk = self.num_frames if self.decode_chunk_size is None else _integer(self.decode_chunk_size, 'decode_chunk_size')
        object.__setattr__(self, 'decode_chunk_size', chunk)
        if not isinstance(self.clip_latents, bool):
            raise TypeError('clip_latents must be a boolean.')
        if self.clip_latents and self.num_frames % chunk:
            raise ValueError('Clipped decode chunks cannot cross camera boundaries.')
        for name in ('min_guidance_scale', 'max_guidance_scale', 'noise_aug_strength', 'scheduled_sampling_beta'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(name + ' must be finite.')
        if not (0 <= self.min_guidance_scale <= self.max_guidance_scale
                and self.noise_aug_strength >= 0 and 0 <= self.scheduled_sampling_beta <= 1):
            raise ValueError('Invalid guidance/noise/scheduled sampling settings.')
        if self.gradient_last_steps is not None:
            _integer(self.gradient_last_steps, 'gradient_last_steps')
            if self.gradient_last_steps > self.num_inference_steps:
                raise ValueError('gradient_last_steps exceeds requested denoising steps.')

    def native_kwargs(self):
        return {name: getattr(self, name) for name in (
            'height', 'width', 'num_frames', 'num_inference_steps', 'min_guidance_scale',
            'max_guidance_scale', 'fps', 'motion_bucket_id', 'noise_aug_strength',
            'decode_chunk_size', 'scheduled_sampling_beta', 'clip_latents')}

    def sampling_identity(self, timesteps):
        return {'requested_steps': self.num_inference_steps,
                'timesteps': [step.item() for step in timesteps],
                'min_guidance_scale': self.min_guidance_scale,
                'max_guidance_scale': self.max_guidance_scale, 'fps': self.fps,
                'motion_bucket_id': self.motion_bucket_id,
                'noise_aug_strength': self.noise_aug_strength,
                'scheduled_sampling_beta': self.scheduled_sampling_beta,
                'decode_chunk_size': self.decode_chunk_size}


@dataclass(frozen=True)
class RenderedDD2Conditions:
    scene_id: str
    camera_names: tuple
    input_dict: Mapping[str, Tensor]


def _native_camera_video(frames, spec, device):
    # Actual DD2 tensor2vid returns a list of T,C,H,W tensors for output_type=pt.
    count = 6 if spec.clip_latents else 1
    width = spec.width // 6 if spec.clip_latents else spec.width
    if not isinstance(frames, (list, tuple)) or len(frames) != count:
        raise ValueError('Native pt output must be a list with the declared batch/camera layout.')
    for frame in frames:
        if not isinstance(frame, Tensor) or frame.device != device:
            raise ValueError('Native pt frames must remain tensors on the execution device.')
        _tensor(frame, 'native frames')
        if tuple(frame.shape) != (spec.num_frames, 3, spec.height, width):
            raise ValueError('Native pt frame layout is not the declared T,C,H,W.')
        if bool(((frame < 0) | (frame > 1)).any().item()):
            raise ValueError('Native processor must produce normalized [0,1] RGB tensors.')
    if spec.clip_latents:
        return torch.stack(tuple(frames), dim=1).unsqueeze(0)
    return torch.stack(frames[0].split(spec.width // 6, dim=-1), dim=1).unsqueeze(0)


class DD2FixedInputAdapter:
    """Inject __call__ and native_replay into FixedInputResearchRunner.

    Fixed prompt/context/noise tensors are detached private snapshots. They are
    never provided by the per-iteration renderer. The renderer must be pure and
    deterministic for a given geometry and return raw map/box tensors with a
    retained torch graph when gradients are enabled. This class is not thread
    safe: stateful pipeline calls must be serialized.
    """
    def __init__(self, *, pipeline, spec: DD2GenerationSpec, prompt_embeddings: Tensor,
                 render_conditions: Callable, model_identity: LoadedModelIdentity,
                 fixed_context: Optional[Mapping[str, Tensor]] = None,
                 initial_latents: Optional[Tensor] = None, allow_cuda: bool = False):
        if not isinstance(spec, DD2GenerationSpec) or not isinstance(model_identity, LoadedModelIdentity):
            raise TypeError('Explicit DD2GenerationSpec and captured LoadedModelIdentity are required.')
        if not isinstance(allow_cuda, bool):
            raise TypeError('allow_cuda must be a boolean.')
        self.pipeline, self.spec = pipeline, spec
        self.allow_cuda, self.model_identity = allow_cuda, model_identity
        self.device = torch.device(pipeline._execution_device)
        self._guard_device()
        model_identity.validate(pipeline)
        self._validate_processor()
        if not callable(render_conditions):
            raise TypeError('A deterministic raw-condition renderer is required.')
        self.render_conditions = render_conditions
        self._prompt = self._snapshot(prompt_embeddings, 'prompt_embeddings')
        if self._prompt.ndim < 2 or self._prompt.shape[0] != 1:
            raise ValueError('A single fixed scene/prompt embedding is required.')
        fixed_context = {} if fixed_context is None else fixed_context
        if not isinstance(fixed_context, Mapping) or set(fixed_context) - {'img_cond', 'video_cond'}:
            raise ValueError('fixed_context may contain only img_cond or video_cond.')
        if len(fixed_context) > 1:
            raise ValueError('img_cond and video_cond are mutually exclusive.')
        self._context = {key: self._snapshot(value, key) for key, value in fixed_context.items()}
        for name, value in self._context.items():
            frames = 1 if name == 'img_cond' else spec.num_frames
            if value.ndim != 4 or value.shape[0] != frames or tuple(value.shape[-2:]) != (spec.height, spec.width):
                raise ValueError('Fixed context must use native panorama layout.')
        self._noise = None if initial_latents is None else self._snapshot(initial_latents, 'initial_latents')
        if self._noise is not None:
            expected = (1, spec.num_frames, pipeline.vae.config.latent_channels,
                        spec.height // pipeline.vae_scale_factor, spec.width // pipeline.vae_scale_factor)
            if tuple(self._noise.shape) != expected or self._noise.dtype != self._prompt.dtype:
                raise ValueError('initial_latents must be unscaled noise matching native shape and prompt dtype.')
        if spec.clip_latents and spec.width % (6 * pipeline.vae_scale_factor):
            raise ValueError('Latent panorama must divide exactly into six cameras.')
        self._fixed_versions = self._versions()

    def _guard_device(self):
        current = torch.device(self.pipeline._execution_device)
        if current != self.device:
            raise ValueError('Pipeline execution device changed after adapter construction.')
        if self.device.type == 'cuda' and not self.allow_cuda:
            raise PermissionError('CUDA execution requires separate explicit authorization.')
        if self.device.type not in {'cpu', 'cuda'} or (self.device.type == 'cuda' and self.device.index is None):
            raise ValueError('Only explicit CPU or indexed CUDA execution is supported.')

    def _validate_processor(self):
        processor = getattr(self.pipeline, 'image_processor', None)
        config = getattr(processor, 'config', None)
        normalize = config.get('do_normalize') if isinstance(config, Mapping) else getattr(config, 'do_normalize', None)
        if not callable(getattr(processor, 'postprocess', None)) or normalize is not True:
            raise ValueError('Native VAE processor must explicitly use do_normalize=True for canonical pt RGB.')

    def _snapshot(self, tensor, name):
        if not isinstance(tensor, Tensor) or tensor.device != self.device:
            raise ValueError(name + ' must already be on the execution device.')
        _tensor(tensor, name)
        if tensor.requires_grad:
            raise ValueError(name + ' is fixed context and must not require gradients.')
        return tensor.detach().clone()

    def _versions(self):
        values = [self._prompt, *self._context.values()]
        if self._noise is not None:
            values.append(self._noise)
        return tuple((id(x), x._version) for x in values)

    def _render(self, geometry, differentiable):
        self._guard_device()
        self.model_identity.validate(self.pipeline)
        if self._versions() != self._fixed_versions:
            raise ValueError('Frozen prompt/context/noise snapshots were modified.')
        if not isinstance(differentiable, bool):
            raise TypeError('differentiable must be a boolean.')
        with torch.set_grad_enabled(differentiable):
            rendered = self.render_conditions(geometry)
            if not isinstance(rendered, RenderedDD2Conditions):
                raise TypeError('Renderer must return RenderedDD2Conditions with actual scene/camera provenance.')
            if rendered.scene_id != self.spec.scene_id or tuple(rendered.camera_names) != self.spec.camera_names:
                raise ValueError('Rendered condition scene/camera order differs from the frozen generation spec.')
            if (not isinstance(rendered.input_dict, Mapping)
                    or not rendered.input_dict
                    or set(rendered.input_dict) - {'grounding_downsampler_input', 'box_downsampler_input'}):
                raise ValueError('Renderer may return raw map/box inputs only; no context/features/latents.')
            for name, value in rendered.input_dict.items():
                if not isinstance(value, Tensor) or value.device != self.device:
                    raise ValueError(name + ' must already be on the execution device.')
            raw = dict(rendered.input_dict)
            raw.update(self._context)
            return _prepare_raw_inputs(raw, self.spec.num_frames, self.spec.height, self.spec.width)

    def __call__(self, geometry, *, differentiable=True):
        raw = self._render(geometry, differentiable)
        return generate_fixed_input(
            self.pipeline, prompt_embeddings=self._prompt, input_dict=raw,
            scene_id=self.spec.scene_id, model_identity=self.model_identity,
            seed=self.spec.seed, camera_names=self.spec.camera_names,
            initial_latents=self._noise, gradient_last_steps=self.spec.gradient_last_steps,
            differentiable=differentiable, allow_cuda=self.allow_cuda,
            **self.spec.native_kwargs())

    def native_replay(self, geometry):
        self._validate_processor()
        raw = self._render(geometry, False)
        predictions = {}
        seen_timesteps = []

        def collect_noise(pipeline, index, timestep, values):
            if pipeline is not self.pipeline or index != len(predictions):
                raise RuntimeError('Native denoising callbacks are not in the expected order.')
            prediction = values['noise_pred']
            _tensor(prediction, 'native noise prediction')
            key = f'step_{index:04d}:t={timestep.item()}'
            predictions[key] = prediction.detach().clone()
            seen_timesteps.append(timestep.detach().clone())
            return {}  # Observe only: never replace latents or conditions.

        with _isolated_pipeline(self.pipeline, self.device, self.spec.seed) as scheduler:
            with torch.no_grad():
                frames = self.pipeline(
                    prompt_embeddings=self._prompt.clone(), input_dict=raw,
                    generator=torch.Generator(device=self.device).manual_seed(self.spec.seed),
                    latents=None if self._noise is None else self._noise.clone(),
                    num_videos_per_prompt=1, output_type='pt', return_dict=False,
                    callback_on_step_end=collect_noise,
                    callback_on_step_end_tensor_inputs=['noise_pred'],
                    cross_attention_kwargs=None, causal_module=None, causal_inputs=None,
                    **self.spec.native_kwargs())
                video = _native_camera_video(frames, self.spec, self.device)
                timesteps = scheduler.timesteps
                if (not len(timesteps) or len(seen_timesteps) != len(timesteps)
                        or not all(torch.equal(a, b) for a, b in zip(seen_timesteps, timesteps))):
                    raise RuntimeError('Native replay did not complete the declared scheduler trajectory.')
                pairing = actual_generation_pairing(
                    pipeline=self.pipeline, model_identity=self.model_identity,
                    scene_id=self.spec.scene_id, seed=self.spec.seed,
                    camera_names=self.spec.camera_names, video_shape=video.shape,
                    scheduler=scheduler, sampling=self.spec.sampling_identity(timesteps),
                    prompt_embeddings=self._prompt, raw_inputs=raw,
                    initial_latents=self._noise, clip_latents=self.spec.clip_latents)
        logs = {'mode': 'native_fixed_input_replay', 'native_call_executed': True,
                'seed': self.spec.seed, 'camera_names': list(self.spec.camera_names),
                'camera_layout': 'panorama', 'output_layout': 'B,T,V,C,H,W',
                'output_range': '[0,1]', 'device': str(self.device), 'batch_size': 1,
                'denoise_steps': len(predictions), 'differentiable': False,
                'scheduler_reset': True, 'conditions_prepared_once': True,
                'scope': 'native_replay_contract; not_checkpoint_or_attack_validation'}
        return FixedInputGenerationOutput(video.detach(), predictions, logs, pairing)
