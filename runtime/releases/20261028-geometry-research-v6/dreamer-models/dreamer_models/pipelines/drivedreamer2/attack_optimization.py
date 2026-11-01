"""Two-stage PhysCond optimization, independent of model/scheduler execution.

All objectives are maximized. This module preserves the legacy online per-step
updates; it does not implement a downstream training-harm surrogate.
"""

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Callable, Mapping, Optional, Union

import torch


SUPPORTED_ATTACK_METHODS = frozenset({
    'clean', 'random', 'condition_pgd', 'condition_pgd_quality',
    'physcond_wma', 'no_stage1', 'no_quality', 'no_momentum', 'transfer',
})


@dataclass(frozen=True)
class AttackStepContext:
    """Inputs to an optional Stage 2 objective, valid only during this step.

    decode_video() lazily decodes attacked_latents with gradients. Its output is
    float32 [B, F, C, H, W] in the VAE's raw output scale (usually near [-1, 1],
    not clamped or converted to PIL). With clip_latents, camera views are folded
    into B as in the pipeline decoder. The callback owns downstream preprocessing
    and labels. Do not retain this context or its tensors across steps.
    """

    step_idx: int
    timestep: int
    clean_noise_pred: torch.Tensor
    attacked_noise_pred: torch.Tensor
    attacked_latents: torch.Tensor
    l_diff: torch.Tensor
    perturb_norm: torch.Tensor
    decode_video: Callable[[], torch.Tensor]


@dataclass(frozen=True)
class AttackObjectiveResult:
    """A differentiable scalar to maximize and scalar diagnostic values.

    score excludes the quality and perturbation penalties: the optimizer adds
    those separately. Metrics are detached before entering pipeline logs.
    """

    score: torch.Tensor
    metrics: Mapping[str, Union[Real, torch.Tensor]] = field(default_factory=dict)


def validate_objective_result(result):
    """Validate the hook contract without detaching the optimization score."""
    if not isinstance(result, AttackObjectiveResult):
        raise TypeError('stage2_objective must return AttackObjectiveResult.')
    score = result.score
    if not isinstance(score, torch.Tensor) or not score.is_floating_point():
        raise TypeError('Objective score must be a floating-point tensor.')
    if score.ndim != 0:
        raise ValueError('Objective score must be a scalar (0-dimensional tensor).')
    if not score.requires_grad:
        raise ValueError('Objective score must retain its gradient graph.')
    if not bool(torch.isfinite(score.detach()).item()):
        raise ValueError('Objective score must be finite.')
    if not isinstance(result.metrics, Mapping):
        raise TypeError('Objective metrics must be a mapping of scalar values.')
    metrics = {}
    for name, value in result.metrics.items():
        if not isinstance(name, str):
            raise TypeError('Objective metric names must be strings.')
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or value.is_complex():
                raise ValueError(f'Objective metric {name!r} must be a real scalar.')
            value = value.detach().item()
        if not isinstance(value, Real):
            raise TypeError(f'Objective metric {name!r} must be numeric.')
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'Objective metric {name!r} must be finite.')
        metrics[name] = value
    return score, metrics


def should_switch_stage(step_idx, l_diff_value, config):
    """Legacy switch policy: check before computing the current step objective."""
    strategy = config.get('switch_strategy', 'hybrid')
    switch_step = int(config.get('switch_step', 0) or 0)
    min_stage1_steps = int(config.get('min_stage1_steps', switch_step) or 0)
    tau = float(config.get('tau', 0.0))
    if (step_idx + 1) < min_stage1_steps:
        return False
    switch_by_step = switch_step > 0 and (step_idx + 1) >= switch_step
    switch_by_loss = l_diff_value < tau if tau > 0 else False
    if strategy == 'fixed':
        return switch_by_step
    if strategy == 'threshold':
        return switch_by_loss
    return switch_by_step or switch_by_loss


class AttackOptimizer:
    """Configuration, loss composition and projected ascent for both stages.

    The caller owns delta/momentum tensors, stage state and the projection
    function (including branch masks, input ranges and geometry constraints).
    No model weights or latent trajectories are managed here.
    """

    def __init__(self, config: Mapping, method: str, mode: str):
        if method not in SUPPORTED_ATTACK_METHODS:
            raise ValueError(
                f'Unsupported attack method: {method}. '
                f'Expected one of {sorted(SUPPORTED_ATTACK_METHODS)}.'
            )
        if mode not in {'targeted', 'untargeted'}:
            raise ValueError(f'Unsupported attack mode: {mode}')
        self.config = dict(config)
        self.method = method
        self.mode = mode
        self.lambda_r = float(config.get('lambda_R', 0.01))
        self.lambda_l = float(config.get('lambda_L', 1.0))
        self.lambda_tar = float(config.get('lambda_tar', 1.0))
        self.quality_weight = float(config.get('quality_weight', self.lambda_l))
        self.step_size_a = float(config.get('step_size_A', 0.01))
        self.step_size_b = float(config.get('step_size_B', 0.02))
        self.momentum = float(config.get('momentum', 0.9))
        self.tau = float(config.get('tau', 0.0))
        self.objective_weight = float(config.get('objective_weight', 1.0))
        self.objective_every_k = max(1, int(config.get('objective_every_k', 1)))
        self.use_momentum = (
            method not in {'no_momentum', 'condition_pgd', 'condition_pgd_quality'}
            and self.momentum > 0
        )
        self.use_quality_constraint = method not in {'condition_pgd', 'no_quality'}
        self.update_enabled = method not in {'clean', 'random', 'transfer'}
        self.initial_stage = (
            'stage2' if method in {
                'condition_pgd', 'condition_pgd_quality', 'no_stage1',
                'random', 'clean', 'transfer',
            } else 'stage1'
        )

    def should_switch(self, step_idx, l_diff_value):
        return should_switch_stage(step_idx, l_diff_value, self.config)

    def compose(
        self, stage, l_diff, perturb_norm, align_score=None,
        custom_score=None, custom_objective=False,
    ):
        quality_penalty = (
            torch.relu(l_diff - self.tau).pow(2) if self.tau > 0 else l_diff
        )
        if not self.update_enabled:
            return l_diff.new_tensor(0.0), quality_penalty
        if stage == 'stage1':
            objective = -self.quality_weight * quality_penalty - self.lambda_r * perturb_norm
        elif stage == 'stage2':
            if custom_objective:
                score = l_diff.new_tensor(0.0) if custom_score is None else custom_score
                objective = self.objective_weight * score - self.lambda_r * perturb_norm
            elif self.mode == 'targeted':
                score = l_diff.new_tensor(0.0) if align_score is None else align_score
                objective = self.lambda_tar * score - self.lambda_r * perturb_norm
            else:
                objective = self.lambda_l * l_diff - self.lambda_r * perturb_norm
            if self.use_quality_constraint:
                objective = objective - self.quality_weight * quality_penalty
        else:
            raise ValueError(f'Unsupported attack stage: {stage}')
        return objective, quality_penalty

    def step(self, stage, objective, deltas, momentums, project):
        """Differentiate once, ascend in-place, and project each updated delta.

        Stage 1 never changes momentum. Disconnected deltas are skipped, including
        projection, to retain the behavior of the original pipeline.
        """
        if not self.update_enabled:
            return [None for _ in deltas]
        if stage not in {'stage1', 'stage2'}:
            raise ValueError(f'Unsupported attack stage: {stage}')
        if len(deltas) != len(momentums):
            raise ValueError('Each delta requires a corresponding momentum tensor.')
        grads = torch.autograd.grad(objective, deltas, allow_unused=True)
        with torch.no_grad():
            for delta, momentum_tensor, grad in zip(deltas, momentums, grads):
                if grad is None:
                    continue
                if stage == 'stage2' and self.use_momentum:
                    normalized_grad = grad / grad.abs().mean().clamp_min(1e-12)
                    momentum_tensor.mul_(self.momentum).add_(
                        (1 - self.momentum) * normalized_grad
                    )
                    direction = momentum_tensor.sign()
                else:
                    direction = grad.sign()
                step_size = self.step_size_a if stage == 'stage1' else self.step_size_b
                delta.add_(step_size * direction)
                project(delta)
        return grads
