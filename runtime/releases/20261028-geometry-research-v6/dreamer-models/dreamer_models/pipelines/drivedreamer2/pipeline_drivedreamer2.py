# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Any,List, Optional, Union

import numpy as np
import PIL.Image
import torch
import copy
import torch.nn.functional as F

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.models.attention import GatedSelfAttentionDense

from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import (
    BaseOutput, 
    logging
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .drivedreamer2_loaders import DriveDreamer2_LoaderMixin
from .attack_optimization import (
    AttackObjectiveResult,
    AttackOptimizer,
    AttackStepContext,
    should_switch_stage,
    validate_objective_result,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


@dataclass
class DriveDreamer2PipelineOutput(BaseOutput):

    frames: Union[List[PIL.Image.Image], np.ndarray]

@dataclass
class DriveDreamer2AttackOutput(BaseOutput):

    clean_frames: Union[List[PIL.Image.Image], np.ndarray]
    attacked_frames: Union[List[PIL.Image.Image], np.ndarray]
    logs: Dict[str, Any]
    deltas: Optional[Dict[str, torch.Tensor]] = None

class DriveDreamer2Pipeline(DiffusionPipeline,DriveDreamer2_LoaderMixin):

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents", "noise_pred", "prompt_embeddings", "added_time_ids"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14,clip_latents=False):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        if clip_latents:
            batch, _, channels, height, width = latents.shape
            width = width//6
            latents = latents.reshape(batch,num_frames,channels,height,6,width).permute(0,4,1,2,3,5).flatten(0,2)
        else:
            latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents
        # frames = self.vae.decode(latents, num_frames=num_frames,return_dict=False)[0]
        # return frames
        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self,height, width):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,#TODO cfg具体参数设置好需要调整为//2
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def enable_fuser(self, enabled=True):
        for module in self.unet.modules():
            if type(module) is GatedSelfAttentionDense:
                module.enabled = enabled

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def _ensure_tensor(self, value, device, dtype):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _duplicate_for_cfg(self, value, do_classifier_free_guidance, zero_uncond=False):
        if value is None or not do_classifier_free_guidance:
            return value
        if zero_uncond:
            return torch.cat([torch.zeros_like(value), value], dim=0)
        return torch.cat([value, value], dim=0)

    def _extract_raw_condition_features(self, input_dict, batch_size, height, width, device, prompt_dtype):
        input_dict = input_dict or {}
        condition_latents = None
        mask_condition = None

        img_cond = input_dict.get('img_cond', None)
        if img_cond is not None:
            img_cond = self._ensure_tensor(img_cond, device, torch.float16)
            condition_latent = self.vae.encode(img_cond).latent_dist.sample()
            condition_latent = condition_latent * self.vae.config.scaling_factor
            condition_latent = condition_latent.to(self.dtype)[:, None]

            condition_latents = torch.zeros(
                (batch_size, self.frame_num, *condition_latent.shape[2:]),
                dtype=self.dtype,
                device=device,
            )
            condition_latents[:, 0] = condition_latent[:, 0]

        video_cond = input_dict.get('video_cond', None)
        if video_cond is not None:
            video_cond = self._ensure_tensor(video_cond, device, torch.float16)
            mask_condition = torch.zeros(
                (self.frame_num, 6, 1, height, width // 6),
                dtype=self.dtype,
                device=video_cond.device,
            )
            mask_condition[:, 1] = 1
            mask_condition = mask_condition.permute(0, 2, 3, 1, 4).flatten(3, 4)
            mask_image = video_cond * mask_condition
            mask_condition = torch.nn.functional.interpolate(
                mask_condition,
                size=(height // 8, width // 8),
            )[None]
            condition_latents = self.vae.encode(mask_image).latent_dist.sample()
            condition_latents = condition_latents * self.vae.config.scaling_factor
            condition_latents = condition_latents.to(self.dtype)[None]

        grounding_features = None
        grounding_downsampler_input = input_dict.get('grounding_downsampler_input', None)
        if grounding_downsampler_input is not None:
            grounding_downsampler_input = self._ensure_tensor(grounding_downsampler_input, device, prompt_dtype)
            gd_device = self.grounding_downsampler.device
            self.grounding_downsampler.to(device)
            grounding_features = self.grounding_downsampler(grounding_downsampler_input)
            self.grounding_downsampler.to(gd_device)
            grounding_features = grounding_features.reshape(batch_size, self.frame_num, *grounding_features.shape[1:])

        box_features = None
        box_downsampler_input = input_dict.get('box_downsampler_input', None)
        if box_downsampler_input is not None:
            box_downsampler_input = self._ensure_tensor(box_downsampler_input, device, prompt_dtype)
            bd_device = self.box_downsampler.device
            self.box_downsampler.to(device)
            box_features = self.box_downsampler(box_downsampler_input)
            self.box_downsampler.to(bd_device)
            box_features = box_features.reshape(batch_size, self.frame_num, *box_features.shape[1:])

        return dict(
            condition_latents=condition_latents,
            mask_condition=mask_condition,
            grounding_features=grounding_features,
            box_features=box_features,
        )

    def _maybe_apply_causal_module(
        self,
        grounding_features,
        box_features,
        causal_module=None,
        causal_inputs=None,
        return_causal_details=False,
    ):
        if causal_module is None or causal_inputs is None:
            return grounding_features, box_features, None

        causal_outputs = causal_module(
            grounding_features=grounding_features,
            box_features=box_features,
            causal_inputs=causal_inputs,
            return_details=return_causal_details,
        )
        grounding_features = causal_outputs.get('edited_grounding_features', grounding_features)
        box_features = causal_outputs.get('edited_box_features', box_features)
        return grounding_features, box_features, causal_outputs

    def _prepare_prompt_context(
        self,
        prompt_embeddings,
        height,
        width,
        num_frames,
        num_inference_steps,
        min_guidance_scale,
        max_guidance_scale,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        num_videos_per_prompt,
        scheduled_sampling_beta,
        generator,
        latents,
        cross_attention_kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        self.frame_num = num_frames

        self.check_inputs(height, width)
        self._cross_attention_kwargs = cross_attention_kwargs

        batch_size = prompt_embeddings.shape[0]
        num_cams = batch_size
        device = self._execution_device
        do_classifier_free_guidance = max_guidance_scale > 1.0

        if do_classifier_free_guidance:
            negative_prompt_embeddings = torch.zeros_like(prompt_embeddings)
            prompt_embeddings = torch.cat([negative_prompt_embeddings, prompt_embeddings])

        fps = fps - 1
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            prompt_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        ).to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            self.frame_num,
            self.vae.config.latent_channels,
            height,
            width,
            prompt_embeddings.dtype,
            device,
            generator,
            latents,
        )

        num_grounding_steps = int(scheduled_sampling_beta * len(timesteps))
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale
        self._num_timesteps = len(timesteps)

        return dict(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_cams=num_cams,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeddings=prompt_embeddings,
            added_time_ids=added_time_ids,
            timesteps=timesteps,
            latents=latents,
            num_grounding_steps=num_grounding_steps,
        )

    def _prepare_condition_features(
        self,
        input_dict,
        batch_size,
        height,
        width,
        device,
        prompt_dtype,
        do_classifier_free_guidance,
        causal_module=None,
        causal_inputs=None,
        return_causal_details=False,
    ):
        raw_features = self._extract_raw_condition_features(
            input_dict=input_dict,
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            prompt_dtype=prompt_dtype,
        )
        grounding_features, box_features, causal_outputs = self._maybe_apply_causal_module(
            raw_features['grounding_features'],
            raw_features['box_features'],
            causal_module=causal_module,
            causal_inputs=causal_inputs,
            return_causal_details=return_causal_details,
        )

        condition_latents = self._duplicate_for_cfg(raw_features['condition_latents'], do_classifier_free_guidance)
        mask_condition = self._duplicate_for_cfg(raw_features['mask_condition'], do_classifier_free_guidance)
        grounding_features = self._duplicate_for_cfg(grounding_features, do_classifier_free_guidance, zero_uncond=True)
        box_features = self._duplicate_for_cfg(box_features, do_classifier_free_guidance, zero_uncond=True)

        return dict(
            condition_latents=condition_latents,
            mask_condition=mask_condition,
            grounding_features=grounding_features,
            box_features=box_features,
            raw_grounding_features=raw_features['grounding_features'],
            raw_box_features=raw_features['box_features'],
            causal_outputs=causal_outputs,
        )

    def _predict_noise(
        self,
        latents,
        t,
        prompt_embeddings,
        added_time_ids,
        num_cams,
        condition_latents=None,
        mask_condition=None,
        grounding_features=None,
        box_features=None,
        do_classifier_free_guidance=False,
        scheduler=None,
    ):
        scheduler = self.scheduler if scheduler is None else scheduler
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        if mask_condition is not None:
            latent_model_input = torch.cat([latent_model_input, mask_condition], dim=2)
        if condition_latents is not None:
            latent_model_input = torch.cat([latent_model_input, condition_latents], dim=2)
        if grounding_features is not None:
            latent_model_input = torch.cat([latent_model_input, grounding_features], dim=2)
        if box_features is not None:
            latent_model_input = torch.cat([latent_model_input, box_features], dim=2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeddings,
            added_time_ids=added_time_ids,
            num_cams=num_cams,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return noise_pred

    def _decode_output(self, latents, num_frames, decode_chunk_size, clip_latents, output_type):
        if output_type == "latent":
            return latents
        frames = self.decode_latents(latents, num_frames, decode_chunk_size, clip_latents)
        return tensor2vid(frames, self.image_processor, output_type=output_type)

    def _should_switch_attack_stage(self, step_idx, l_diff_value, attack_config):
        return should_switch_stage(step_idx, l_diff_value, attack_config)

    def _decode_attack_video_tensor(self, latents, num_frames, decode_chunk_size, clip_latents):
        frames = self.decode_latents(latents, num_frames, decode_chunk_size, clip_latents)
        return frames.permute(0, 2, 1, 3, 4).contiguous()

    def attack_generate(
        self,
        prompt_embeddings: Optional[torch.FloatTensor] = None,
        input_dict: dict = None,
        attack_config: Optional[Dict[str, Any]] = None,
        sscd_embedder=None,
        target_embedding: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        scheduled_sampling_beta: float = 0.3,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_latents: bool = False,
        causal_module=None,
        causal_inputs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        stage2_objective: Optional[Callable[[AttackStepContext], AttackObjectiveResult]] = None,
    ):
        """Run online two-stage condition updates with an optional Stage 2 score.

        The callback is separate from attack_config so its surrogate model is
        never deep-copied. It replaces the legacy attack target, while quality
        and perturbation penalties remain controlled by the existing settings.
        See docs/physcond_optimization.md for the callback and replay contracts.
        """
        attack_config = copy.deepcopy(attack_config or {})
        attack_enabled = bool(attack_config.get('enable', True))
        if not attack_enabled:
            clean_output = self(
                prompt_embeddings=prompt_embeddings,
                input_dict=input_dict,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
                num_videos_per_prompt=num_videos_per_prompt,
                scheduled_sampling_beta=scheduled_sampling_beta,
                generator=generator,
                latents=latents,
                output_type=output_type,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_latents=clip_latents,
                causal_module=causal_module,
                causal_inputs=causal_inputs,
                return_dict=True,
            )
            output = DriveDreamer2AttackOutput(
                clean_frames=clean_output.frames,
                attacked_frames=clean_output.frames,
                logs={'enabled': False, 'steps': [], 'stage_switch_step': None},
            )
            return output if return_dict else (clean_output.frames, clean_output.frames, output.logs)

        attack_mode = attack_config.get('mode', 'targeted')
        if attack_mode not in ['targeted', 'untargeted']:
            raise ValueError(f'Unsupported attack mode: {attack_mode}')
        if stage2_objective is not None and not callable(stage2_objective):
            raise TypeError('stage2_objective must be callable or None.')
        if (
            stage2_objective is None
            and attack_mode == 'targeted'
            and (sscd_embedder is None or target_embedding is None)
        ):
            raise ValueError('Targeted attack requires both sscd_embedder and target_embedding.')

        context = self._prepare_prompt_context(
            prompt_embeddings=prompt_embeddings,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_videos_per_prompt=num_videos_per_prompt,
            scheduled_sampling_beta=scheduled_sampling_beta,
            generator=generator,
            latents=latents,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else context['num_frames']
        condition_features = self._prepare_condition_features(
            input_dict=input_dict,
            batch_size=context['batch_size'],
            height=context['height'],
            width=context['width'],
            device=context['device'],
            prompt_dtype=context['prompt_embeddings'].dtype,
            do_classifier_free_guidance=context['do_classifier_free_guidance'],
            causal_module=causal_module,
            causal_inputs=causal_inputs,
        )

        clean_latents = context['latents'].clone()
        attacked_latents = context['latents'].clone()

        base_grounding = condition_features['grounding_features']
        base_box = condition_features['box_features']
        active_branches = attack_config.get('branches', 'both')
        attack_hdmap = active_branches in ['hdmap', 'both']
        attack_box = active_branches in ['3dbox', 'box', 'both']
        attack_space = str(attack_config.get('attack_space', 'feature')).lower()
        if attack_space not in {'feature', 'raw'}:
            raise ValueError(
                f'Unsupported attack space: {attack_space}. '
                "Expected one of ['feature', 'raw']."
            )
        if attack_space == 'raw' and attack_box:
            raise ValueError(
                'Raw-condition transfer currently supports the HDMap branch only. '
                'Use attack.branches=hdmap.'
            )
        raw_parameterization = str(
            attack_config.get('raw_parameterization', 'additive')
        ).lower()
        if raw_parameterization not in {'additive', 'global_translation'}:
            raise ValueError(
                f'Unsupported raw_parameterization={raw_parameterization}. '
                "Expected one of ['additive', 'global_translation']."
            )
        raw_flow_max_pixels = float(
            attack_config.get('raw_flow_max_pixels', 2.0)
        )
        if raw_flow_max_pixels < 0:
            raise ValueError(
                'raw_flow_max_pixels must be non-negative, got '
                f'{raw_flow_max_pixels}.'
            )

        delta_grounding = None
        delta_box = None
        momentum_grounding = None
        momentum_box = None
        grounding_mask = None
        box_mask = None
        raw_grounding_input = None
        raw_panel_start = None
        raw_panel_end = None
        raw_value_min = float(attack_config.get('raw_value_min', -1.0))
        raw_value_max = float(attack_config.get('raw_value_max', 1.0))
        if raw_value_min >= raw_value_max:
            raise ValueError(
                f'Expected raw_value_min < raw_value_max, got '
                f'{raw_value_min} >= {raw_value_max}.'
            )
        attack_method = str(attack_config.get('method', 'physcond_wma')).lower()
        optimizer = AttackOptimizer(attack_config, attack_method, attack_mode)

        delta_clamp = attack_config.get('delta_clamp', 0.15)
        delta_clamp = None if delta_clamp is None else float(delta_clamp)
        random_start = bool(attack_config.get('random_start', attack_method != 'clean'))
        random_start_scale = float(attack_config.get('random_start_scale', 1.0))
        attack_seed = int(attack_config.get('attack_seed', 3407))

        def _initial_delta(
            base_feature,
            seed_offset,
            clamp_value,
            external_delta=None,
        ):
            if external_delta is not None:
                delta = self._ensure_tensor(
                    external_delta,
                    base_feature.device,
                    torch.float32,
                )
                if tuple(delta.shape) != tuple(base_feature.shape):
                    raise ValueError(
                        'Transferred perturbation shape mismatch: '
                        f'expected {tuple(base_feature.shape)}, got {tuple(delta.shape)}.'
                    )
                delta = delta.clone()
                if clamp_value is not None:
                    delta.clamp_(-clamp_value, clamp_value)
            else:
                delta = torch.zeros_like(base_feature, dtype=torch.float32)
            if (
                external_delta is None
                and random_start
                and clamp_value is not None
                and clamp_value > 0
            ):
                delta_generator = torch.Generator(device=delta.device)
                delta_generator.manual_seed(attack_seed + seed_offset)
                delta.uniform_(
                    -clamp_value * random_start_scale,
                    clamp_value * random_start_scale,
                    generator=delta_generator,
                )
            return delta.requires_grad_(True)

        if attack_hdmap:
            if base_grounding is None:
                raise ValueError('HDMap attack branch requested but grounding features are missing.')
            grounding_attack_base = base_grounding
            if attack_space == 'raw':
                raw_grounding_input = input_dict.get('grounding_downsampler_input')
                if raw_grounding_input is None:
                    raise ValueError(
                        'Raw HDMap attack requires grounding_downsampler_input.'
                    )
                raw_grounding_input = self._ensure_tensor(
                    raw_grounding_input,
                    context['device'],
                    torch.float32,
                ).detach()
                raw_num_cameras = int(
                    attack_config.get('raw_num_cameras', context['num_cams'])
                )
                raw_camera_index = int(
                    attack_config.get('raw_camera_index', 0)
                )
                if raw_num_cameras <= 0:
                    raise ValueError(
                        f'raw_num_cameras must be positive, got {raw_num_cameras}.'
                    )
                if not 0 <= raw_camera_index < raw_num_cameras:
                    raise ValueError(
                        f'raw_camera_index={raw_camera_index} is outside '
                        f'[0, {raw_num_cameras}).'
                    )
                if raw_grounding_input.shape[-1] % raw_num_cameras != 0:
                    raise ValueError(
                        'Raw HDMap width must be divisible by raw_num_cameras: '
                        f'{raw_grounding_input.shape[-1]} vs {raw_num_cameras}.'
                    )
                panel_width = raw_grounding_input.shape[-1] // raw_num_cameras
                raw_panel_start = raw_camera_index * panel_width
                raw_panel_end = raw_panel_start + panel_width
                if raw_parameterization == 'global_translation':
                    # A single temporally coherent (dx, dy) displacement is
                    # directly transferable between models that consume the
                    # same HDMap raster, independent of encoder architecture.
                    grounding_attack_base = torch.zeros(
                        (1, 2),
                        device=context['device'],
                        dtype=torch.float32,
                    )
                    grounding_clamp = raw_flow_max_pixels
                    external_grounding = input_dict.get(
                        'fixed_delta_grounding_flow'
                    )
                else:
                    grounding_attack_base = raw_grounding_input
                    grounding_clamp = delta_clamp
                    external_grounding = input_dict.get(
                        'fixed_delta_grounding'
                    )
            else:
                grounding_clamp = delta_clamp
                external_grounding = input_dict.get(
                    'fixed_delta_grounding'
                )
            delta_grounding = _initial_delta(
                grounding_attack_base,
                0,
                grounding_clamp,
                external_grounding if attack_method == 'transfer' else None,
            )
            momentum_grounding = torch.zeros_like(delta_grounding)
            if attack_space == 'raw':
                if raw_parameterization == 'global_translation':
                    grounding_mask = torch.ones_like(delta_grounding)
                else:
                    grounding_mask = torch.zeros_like(delta_grounding)
                    grounding_mask[..., raw_panel_start:raw_panel_end] = 1
            else:
                grounding_mask = torch.ones_like(delta_grounding)
                if context['do_classifier_free_guidance']:
                    grounding_mask[:context['batch_size']] = 0
            with torch.no_grad():
                delta_grounding.mul_(grounding_mask)
            if attack_space == 'raw':
                # Condition preparation temporarily moves the downsampler to the
                # execution device and then restores it. Raw-space optimization
                # needs to re-encode the attacked condition at every denoising
                # step, so keep the module colocated with the raw tensor here.
                self.grounding_downsampler.to(context['device'])
        if attack_box:
            if base_box is None:
                raise ValueError('3D-box attack branch requested but box features are missing.')
            delta_box = _initial_delta(
                base_box,
                1,
                delta_clamp,
                input_dict.get('fixed_delta_box') if attack_method == 'transfer' else None,
            )
            momentum_box = torch.zeros_like(delta_box)
            box_mask = torch.ones_like(delta_box)
            if context['do_classifier_free_guidance']:
                box_mask[:context['batch_size']] = 0
            with torch.no_grad():
                delta_box.mul_(box_mask)

        if delta_grounding is None and delta_box is None:
            clean_output = self(
                prompt_embeddings=prompt_embeddings,
                input_dict=input_dict,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
                num_videos_per_prompt=num_videos_per_prompt,
                scheduled_sampling_beta=scheduled_sampling_beta,
                generator=generator,
                latents=latents,
                output_type=output_type,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_latents=clip_latents,
                causal_module=causal_module,
                causal_inputs=causal_inputs,
                return_dict=True,
            )
            output = DriveDreamer2AttackOutput(
                clean_frames=clean_output.frames,
                attacked_frames=clean_output.frames,
                logs={'enabled': False, 'steps': [], 'stage_switch_step': None},
            )
            return output if return_dict else (clean_output.frames, clean_output.frames, output.logs)

        stage_logs = []
        stage = optimizer.initial_stage
        stage_switch_step = None
        last_align_score = None
        align_every_k = max(1, int(attack_config.get('sscd_every_k', 1)))
        use_momentum = optimizer.use_momentum
        use_quality_constraint = optimizer.use_quality_constraint
        update_enabled = optimizer.update_enabled

        def _merge_feature(base_feature, delta):
            if base_feature is None:
                return None
            if delta is None:
                return base_feature
            merged = base_feature.float() + delta
            return merged.to(base_feature.dtype)

        def _project_delta_in_place(delta):
            if delta is delta_grounding:
                mask = grounding_mask
            elif delta is delta_box:
                mask = box_mask
            else:
                mask = None
            if mask is not None:
                delta.mul_(mask)
            clamp_limit = delta_clamp
            if (
                attack_space == 'raw'
                and raw_parameterization == 'global_translation'
                and delta is delta_grounding
            ):
                clamp_limit = raw_flow_max_pixels
            if clamp_limit is not None:
                delta.clamp_(-clamp_limit, clamp_limit)
            if (
                attack_space == 'raw'
                and raw_parameterization == 'additive'
                and delta is delta_grounding
                and raw_grounding_input is not None
            ):
                delta.copy_(
                    (raw_grounding_input + delta)
                    .clamp(raw_value_min, raw_value_max)
                    .sub(raw_grounding_input)
                )

        def _build_attacked_raw_grounding():
            if raw_parameterization == 'additive':
                return (
                    raw_grounding_input + delta_grounding
                ).clamp(raw_value_min, raw_value_max)

            clean_panel = raw_grounding_input[
                ...,
                raw_panel_start:raw_panel_end,
            ]
            panel_height = clean_panel.shape[-2]
            panel_width = clean_panel.shape[-1]
            flow = delta_grounding.expand(clean_panel.shape[0], -1)
            theta = torch.eye(
                2,
                3,
                device=clean_panel.device,
                dtype=clean_panel.dtype,
            )[None].repeat(clean_panel.shape[0], 1, 1)
            # dx/dy describe the content displacement in pixels. affine_grid
            # samples from the opposite direction to move content accordingly.
            theta[:, 0, 2] = (
                -2.0 * flow[:, 0].to(clean_panel.dtype)
                / max(panel_width - 1, 1)
            )
            theta[:, 1, 2] = (
                -2.0 * flow[:, 1].to(clean_panel.dtype)
                / max(panel_height - 1, 1)
            )
            grid = F.affine_grid(
                theta,
                clean_panel.shape,
                align_corners=True,
            )
            attacked_panel = F.grid_sample(
                clean_panel,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            )
            return torch.cat(
                [
                    raw_grounding_input[..., :raw_panel_start],
                    attacked_panel,
                    raw_grounding_input[..., raw_panel_end:],
                ],
                dim=-1,
            )

        def _encode_attacked_raw_grounding():
            attacked_raw = _build_attacked_raw_grounding()
            grounding_features = self.grounding_downsampler(
                attacked_raw.to(context['prompt_embeddings'].dtype)
            )
            grounding_features = grounding_features.reshape(
                context['batch_size'],
                context['num_frames'],
                *grounding_features.shape[1:],
            )
            return self._duplicate_for_cfg(
                grounding_features,
                context['do_classifier_free_guidance'],
                zero_uncond=True,
            )

        with torch.no_grad():
            if delta_grounding is not None:
                _project_delta_in_place(delta_grounding)
            if delta_box is not None:
                _project_delta_in_place(delta_box)

        clean_scheduler = copy.deepcopy(self.scheduler)
        attacked_scheduler = copy.deepcopy(self.scheduler)
        self.enable_fuser(True)
        num_warmup_steps = len(context['timesteps']) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, t in enumerate(context['timesteps']):
                if step_idx == context['num_grounding_steps']:
                    self.enable_fuser(False)

                with torch.no_grad():
                    clean_noise_pred = self._predict_noise(
                        clean_latents,
                        t,
                        context['prompt_embeddings'],
                        context['added_time_ids'],
                        context['num_cams'],
                        condition_latents=condition_features['condition_latents'],
                        mask_condition=condition_features['mask_condition'],
                        grounding_features=base_grounding,
                        box_features=base_box,
                        do_classifier_free_guidance=context['do_classifier_free_guidance'],
                        scheduler=clean_scheduler,
                    )
                    next_clean_latents = clean_scheduler.step(clean_noise_pred, t, clean_latents).prev_sample

                attacked_grounding = (
                    _encode_attacked_raw_grounding()
                    if attack_space == 'raw' and delta_grounding is not None
                    else _merge_feature(base_grounding, delta_grounding)
                )
                attacked_box = _merge_feature(base_box, delta_box)
                attacked_noise_pred = self._predict_noise(
                    attacked_latents,
                    t,
                    context['prompt_embeddings'],
                    context['added_time_ids'],
                    context['num_cams'],
                    condition_latents=condition_features['condition_latents'],
                    mask_condition=condition_features['mask_condition'],
                    grounding_features=attacked_grounding,
                    box_features=attacked_box,
                    do_classifier_free_guidance=context['do_classifier_free_guidance'],
                    scheduler=attacked_scheduler,
                )
                next_attacked_latents = attacked_scheduler.step(
                    attacked_noise_pred,
                    t,
                    attacked_latents,
                ).prev_sample

                l_diff = F.mse_loss(attacked_noise_pred, clean_noise_pred.detach(), reduction='mean')
                if (
                    attack_method in {'physcond_wma', 'no_quality', 'no_momentum'}
                    and stage == 'stage1'
                    and optimizer.should_switch(step_idx, float(l_diff.detach().cpu()))
                ):
                    stage = 'stage2'
                    stage_switch_step = step_idx

                perturb_norm = attacked_noise_pred.new_tensor(0.0)
                active_deltas = []
                active_momentums = []
                if delta_grounding is not None:
                    perturb_norm = perturb_norm + delta_grounding.pow(2).mean()
                    active_deltas.append(delta_grounding)
                    active_momentums.append(('grounding', momentum_grounding))
                if delta_box is not None:
                    perturb_norm = perturb_norm + delta_box.pow(2).mean()
                    active_deltas.append(delta_box)
                    active_momentums.append(('box', momentum_box))

                align_score = None
                if (
                    stage2_objective is None
                    and attack_mode == 'targeted'
                    and stage == 'stage2'
                    and (step_idx % align_every_k == 0)
                ):
                    decoded_video = self._decode_attack_video_tensor(
                        next_attacked_latents,
                        context['num_frames'],
                        decode_chunk_size,
                        clip_latents,
                    )[0]
                    align_score = sscd_embedder.score_video(decoded_video, target_embedding.to(decoded_video.device))
                    last_align_score = float(align_score.detach().cpu())

                custom_score = None
                custom_metrics = {}
                if (
                    stage2_objective is not None
                    and update_enabled
                    and stage == 'stage2'
                    and step_idx % optimizer.objective_every_k == 0
                ):
                    custom_score, custom_metrics = validate_objective_result(
                        stage2_objective(AttackStepContext(
                            step_idx=step_idx,
                            timestep=int(t.item()) if torch.is_tensor(t) else int(t),
                            clean_noise_pred=clean_noise_pred.detach(),
                            attacked_noise_pred=attacked_noise_pred,
                            attacked_latents=next_attacked_latents,
                            l_diff=l_diff,
                            perturb_norm=perturb_norm,
                            decode_video=lambda: self._decode_attack_video_tensor(
                                next_attacked_latents,
                                context['num_frames'],
                                decode_chunk_size,
                                clip_latents,
                            ),
                        ))
                    )

                objective, quality_penalty = optimizer.compose(
                    stage, l_diff, perturb_norm,
                    align_score=align_score,
                    custom_score=custom_score,
                    custom_objective=stage2_objective is not None,
                )
                grads = optimizer.step(
                    stage, objective, active_deltas,
                    [tensor for _, tensor in active_momentums],
                    _project_delta_in_place,
                )

                clean_latents = next_clean_latents.detach()
                attacked_latents = next_attacked_latents.detach()

                finite_grads = [grad.detach() for grad in grads if grad is not None]
                step_log = {
                    'step': step_idx,
                    'timestep': int(t.item()) if torch.is_tensor(t) else int(t),
                    'stage': stage,
                    'method': attack_method,
                    'objective': float(objective.detach().cpu()),
                    'l_diff': float(l_diff.detach().cpu()),
                    'quality_penalty': float(quality_penalty.detach().cpu()),
                    'perturb_norm': float(perturb_norm.detach().cpu()),
                    'align_score': None if align_score is None else float(align_score.detach().cpu()),
                    'grad_l2': None if not finite_grads else float(
                        torch.sqrt(sum(grad.float().pow(2).sum() for grad in finite_grads)).cpu()
                    ),
                    'grad_linf': None if not finite_grads else float(
                        max(grad.float().abs().max() for grad in finite_grads).cpu()
                    ),
                    'delta_grounding_abs': None if delta_grounding is None else float(delta_grounding.detach().abs().mean().cpu()),
                    'delta_box_abs': None if delta_box is None else float(delta_box.detach().abs().mean().cpu()),
                }
                if stage2_objective is not None:
                    step_log['custom_objective_score'] = (
                        None if custom_score is None else float(custom_score.detach().cpu())
                    )
                    step_log['custom_objective_metrics'] = custom_metrics
                stage_logs.append(step_log)
                # Do not retain callback graphs beyond the current update.
                custom_score = None

                if step_idx == len(context['timesteps']) - 1 or ((step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        def _feature_stats(base_feature, delta):
            if base_feature is None or delta is None:
                return None
            base = base_feature.detach().float()
            perturb = delta.detach().float()
            attacked = base + perturb
            flat_base = base.reshape(-1)
            flat_attacked = attacked.reshape(-1)
            delta_l2 = perturb.pow(2).sum().sqrt()
            base_l2 = base.pow(2).sum().sqrt()
            return {
                'numel': int(perturb.numel()),
                'l1_mean': float(perturb.abs().mean().cpu()),
                'l2': float(delta_l2.cpu()),
                'linf': float(perturb.abs().max().cpu()),
                'relative_l2': float((delta_l2 / base_l2.clamp_min(1e-12)).cpu()),
                'cosine_similarity': float(
                    F.cosine_similarity(flat_base[None], flat_attacked[None]).cpu()
                ),
                'changed_activation_ratio': float((perturb.abs() > 1e-6).float().mean().cpu()),
            }

        final_delta_grounding = delta_grounding
        grounding_stats_base = base_grounding
        grounding_stats_name = 'grounding_feature'
        if attack_space == 'raw' and raw_grounding_input is not None:
            if raw_parameterization == 'global_translation':
                final_delta_grounding = (
                    _build_attacked_raw_grounding()
                    .sub(raw_grounding_input)
                )
            else:
                final_delta_grounding = (
                    (raw_grounding_input + delta_grounding)
                    .clamp(raw_value_min, raw_value_max)
                    .sub(raw_grounding_input)
                )
            grounding_stats_base = raw_grounding_input
            grounding_stats_name = 'grounding_input'

        final_condition_stats = {
            grounding_stats_name: _feature_stats(
                grounding_stats_base,
                final_delta_grounding,
            ),
            'box_feature': _feature_stats(base_box, delta_box),
        }

        # The final autograd graph is otherwise retained by loop-local references
        # and can make the temporal VAE decode exceed a single 48 GB GPU.
        del attacked_noise_pred
        del clean_noise_pred
        del objective
        del quality_penalty
        del l_diff
        del grads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        decode_offload = bool(attack_config.get('decode_offload', True))
        offloaded_modules = []
        if decode_offload and clean_latents.is_cuda:
            for module_name in ('unet', 'grounding_downsampler', 'box_downsampler'):
                module = getattr(self, module_name, None)
                if module is not None:
                    offloaded_modules.append((module_name, module))
                    module.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            clean_frames = self._decode_output(
                clean_latents,
                context['num_frames'],
                decode_chunk_size,
                clip_latents,
                output_type,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attacked_frames = self._decode_output(
                attacked_latents,
                context['num_frames'],
                decode_chunk_size,
                clip_latents,
                output_type,
            )

        if offloaded_modules:
            for _, module in offloaded_modules:
                module.to(context['device'])
        self.maybe_free_model_hooks()

        logs = {
            'enabled': True,
            'mode': attack_mode,
            'method': attack_method,
            'optimization_objective': (
                'custom' if stage2_objective is not None else (
                    'sscd_alignment' if attack_mode == 'targeted' else 'denoising_discrepancy'
                )
            ),
            'custom_objective_config': (
                {
                    'weight': optimizer.objective_weight,
                    'every_k': optimizer.objective_every_k,
                }
                if stage2_objective is not None else None
            ),
            'branches': active_branches,
            'attack_space': attack_space,
            'attack_variable': (
                (
                    'raw_hdmap_global_translation'
                    if raw_parameterization == 'global_translation'
                    else 'normalized_raw_hdmap_input'
                )
                if attack_space == 'raw'
                else 'post_downsampler_physical_condition_feature'
            ),
            'epsilon_linf': (
                raw_flow_max_pixels
                if (
                    attack_space == 'raw'
                    and raw_parameterization == 'global_translation'
                )
                else delta_clamp
            ),
            'raw_parameterization': (
                raw_parameterization if attack_space == 'raw' else None
            ),
            'raw_flow_max_pixels': (
                raw_flow_max_pixels
                if (
                    attack_space == 'raw'
                    and raw_parameterization == 'global_translation'
                )
                else None
            ),
            'raw_flow_pixels': (
                delta_grounding.detach().float().cpu().tolist()
                if (
                    attack_space == 'raw'
                    and raw_parameterization == 'global_translation'
                    and delta_grounding is not None
                )
                else None
            ),
            'raw_value_min': raw_value_min if attack_space == 'raw' else None,
            'raw_value_max': raw_value_max if attack_space == 'raw' else None,
            'raw_camera_index': (
                int(attack_config.get('raw_camera_index', 0))
                if attack_space == 'raw'
                else None
            ),
            'raw_num_cameras': (
                int(attack_config.get('raw_num_cameras', context['num_cams']))
                if attack_space == 'raw'
                else None
            ),
            'random_start': random_start,
            'attack_seed': attack_seed,
            'uses_momentum': use_momentum,
            'uses_quality_constraint': use_quality_constraint,
            'stage_switch_step': stage_switch_step,
            'last_align_score': last_align_score,
            'condition_stats': final_condition_stats,
            'steps': stage_logs,
        }
        output = DriveDreamer2AttackOutput(
            clean_frames=clean_frames,
            attacked_frames=attacked_frames,
            logs=logs,
            deltas={
                'grounding': (
                    None
                    if final_delta_grounding is None
                    else final_delta_grounding.detach().float().cpu()
                ),
                'grounding_flow': (
                    delta_grounding.detach().float().cpu()
                    if (
                        attack_space == 'raw'
                        and raw_parameterization == 'global_translation'
                        and delta_grounding is not None
                    )
                    else None
                ),
                'box': None if delta_box is None else delta_box.detach().float().cpu(),
            },
        )
        if not return_dict:
            return clean_frames, attacked_frames, logs
        return output

    def generate_fixed_input(self, **kwargs):
        """Generate final tensor video with fixed raw conditions and isolated state.

        See fixed_input_generation.generate_fixed_input for the explicit CPU
        guard, camera layout, fixed-input contract and gradient approximations.
        """
        from .fixed_input_generation import generate_fixed_input

        return generate_fixed_input(self, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeddings: Optional[torch.FloatTensor] = None,
        input_dict: dict = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        scheduled_sampling_beta: float = 0.3,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        video_condition: bool= False,
        clip_latents: bool = False,
        causal_module=None,
        causal_inputs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        first_frame: bool=False
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        context = self._prepare_prompt_context(
            prompt_embeddings=prompt_embeddings,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_videos_per_prompt=num_videos_per_prompt,
            scheduled_sampling_beta=scheduled_sampling_beta,
            generator=generator,
            latents=latents,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else context['num_frames']

        prompt_embeddings = context['prompt_embeddings']
        added_time_ids = context['added_time_ids']
        timesteps = context['timesteps']
        latents = context['latents']
        num_cams = context['num_cams']
        do_classifier_free_guidance = context['do_classifier_free_guidance']
        num_frames = context['num_frames']
        num_grounding_steps = context['num_grounding_steps']

        condition_features = self._prepare_condition_features(
            input_dict=input_dict,
            batch_size=context['batch_size'],
            height=context['height'],
            width=context['width'],
            device=context['device'],
            prompt_dtype=prompt_embeddings.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            causal_module=causal_module,
            causal_inputs=causal_inputs,
        )

        condition_latents = condition_features['condition_latents']
        mask_condition = condition_features['mask_condition']
        grounding_downsampler_input = condition_features['grounding_features']
        box_downsampler_input = condition_features['box_features']

        self.enable_fuser(True)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == num_grounding_steps:
                    self.enable_fuser(False)
                noise_pred = self._predict_noise(
                    latents,
                    t,
                    prompt_embeddings,
                    added_time_ids,
                    num_cams,
                    condition_latents=condition_latents,
                    mask_condition=mask_condition,
                    grounding_features=grounding_downsampler_input,
                    box_features=box_downsampler_input,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            
            frames = self.decode_latents(latents, num_frames, decode_chunk_size,clip_latents)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return DriveDreamer2PipelineOutput(frames=frames)


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out
