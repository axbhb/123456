"""Outer-loop PhysCond policy. No diffusion-step updates or task implementations.

The callback must generate a final video with one fixed input, build current labels,
and evaluate development-only task harm plus shared quality. All returned histories
are detached. These controls establish an optimization protocol, not attack efficacy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Mapping, Optional

import torch


class EvaluationBudgetExhausted(RuntimeError):
    """The evaluator cannot afford another measurement within its declared budget."""


@dataclass(frozen=True)
class FixedInputConfig:
    outer_iterations: int
    step_size_A: float
    step_size_B: float
    stage1_harm_weight: float
    stage1_quality_weight: float
    stage2_quality_weight: float
    perturbation_weight: float
    stage1_min_steps: int
    stage1_max_steps: int
    harm_gate_by_task: Mapping[str, float]
    momentum: float = 0.9
    backtrack_factor: float = 0.5
    max_backtracks: int = 4
    mode: str = 'two_stage'
    allow_zero_stage1_harm_ablation: bool = False
    stage1_geometry_weight: float = 0.0
    stage2_geometry_weight: float = 0.0
    require_stability_gate: bool = False
    monotone_ascent: bool = False
    direction_rule: str = 'sign'

    def __post_init__(self):
        if self.direction_rule not in ('sign', 'linf_normalized'):
            raise ValueError('Unknown ascent direction rule')
        if not isinstance(self.monotone_ascent, bool):
            raise TypeError('monotone_ascent must be boolean')
        if not isinstance(self.require_stability_gate, bool):
            raise TypeError('require_stability_gate must be boolean')
        for name in ('outer_iterations', 'stage1_min_steps', 'stage1_max_steps', 'max_backtracks'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        if self.outer_iterations < 1 or self.stage1_max_steps < max(1, self.stage1_min_steps):
            raise ValueError('Invalid iteration/stage bounds')
        if self.mode not in ('two_stage', 'stage_a_only', 'stage_b_only'):
            raise ValueError('Unknown optimization mode')
        for name in ('step_size_A', 'step_size_B', 'stage1_harm_weight', 'stage1_quality_weight',
                     'stage2_quality_weight', 'perturbation_weight', 'momentum', 'backtrack_factor',
                     'stage1_geometry_weight', 'stage2_geometry_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if self.step_size_A <= 0 or self.step_size_B <= 0:
            raise ValueError('Step sizes must be positive in normalized input units')
        if self.stage1_harm_weight == 0 and not self.allow_zero_stage1_harm_ablation:
            raise ValueError('Zero stage-A harm must be explicitly declared an ablation')
        if not 0 <= self.momentum < 1 or not 0 < self.backtrack_factor < 1:
            raise ValueError('Invalid momentum/backtracking coefficient')
        if not self.harm_gate_by_task or any(not math.isfinite(v) for v in self.harm_gate_by_task.values()):
            raise ValueError('Provide finite, predeclared gates for all active tasks')


@dataclass
class FixedInputEvaluation:
    harm_score: torch.Tensor
    quality_penalty: torch.Tensor
    per_task_harm: Mapping[str, torch.Tensor]
    quality_feasible: bool
    geometry_feasible: bool
    metrics: Mapping[str, float] = field(default_factory=dict)

    geometry_score: Optional[torch.Tensor] = None
    stage_a_harm_score: Optional[torch.Tensor] = None
    stage_a_per_task_harm: Optional[Mapping[str, torch.Tensor]] = None
    stability_per_task_harm: Optional[Mapping[str, torch.Tensor]] = None

    def validate(self, tasks):
        for label, mapping in (("stage_a", self.stage_a_per_task_harm), ("stability", self.stability_per_task_harm)):
            if mapping is not None and (set(mapping) != set(tasks) or any(v.numel()!=1 or not torch.isfinite(v.detach()).all() for v in mapping.values())):
                raise ValueError(label + " gates must match all tasks with finite scalars")
        for value in (self.geometry_score, self.stage_a_harm_score):
            if value is not None and (not isinstance(value, torch.Tensor) or value.numel()!=1 or not torch.isfinite(value.detach()).all()):
                raise ValueError("Auxiliary scores must be finite scalar tensors")
        if self.geometry_score is not None and abs(float(self.geometry_score.detach())) > 1.000001:
            raise ValueError("Geometry auxiliary score must be bounded in [-1,1]")
        if set(self.per_task_harm) != set(tasks):
            raise ValueError('Evaluation tasks must exactly match active task gates')
        for name, value in {'harm_score': self.harm_score, 'quality_penalty': self.quality_penalty,
                            **dict(self.per_task_harm)}.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value.detach()).all():
                raise ValueError(name + ' must be a finite scalar tensor')
        if self.quality_penalty.detach().item() < 0:
            raise ValueError('Quality penalty must be nonnegative')
        if not isinstance(self.quality_feasible, bool) or not isinstance(self.geometry_feasible, bool):
            raise TypeError('Feasibility checks must be explicit booleans')
        for key, value in self.metrics.items():
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise ValueError('Metrics must contain detached finite numbers: ' + key)
        return self


@dataclass
class FixedInputResult:
    best_u: torch.Tensor
    final_u: torch.Tensor
    final_momentum: torch.Tensor
    best_harm: float
    phase: str
    status: str
    history: List[Dict]
    evaluations: int
    gradient_evaluations: int
    stage_b_entered: bool
    final_task_gate_met: bool
    best_gated_u: Optional[torch.Tensor]
    best_gated_harm: Optional[float]


class FixedInputPhysCondOptimizer:
    """Projected ascent in budget-normalized coordinates u in [-1, 1].

    Physical conversion delta=S*u and geometry legality belong to the adapter.
    Callbacks run with autograd, including gate evaluations: virtual training needs
    first derivatives even when no outer derivative is requested. Candidate quality
    checks never advance phase using scores from the preceding input.
    """
    def __init__(self, config: FixedInputConfig):
        self.config = config

    def compose(self, stage, harm_score, quality_penalty, u, geometry_score=None):
        if stage not in ('A', 'B'):
            raise ValueError('Stage must be A or B')
        cfg = self.config
        beta = cfg.stage1_harm_weight if stage == 'A' else 1.0
        weight = cfg.stage1_quality_weight if stage == 'A' else cfg.stage2_quality_weight
        geo_weight = cfg.stage1_geometry_weight if stage == "A" else cfg.stage2_geometry_weight
        if geo_weight and geometry_score is None:
            raise ValueError("Configured geometry auxiliary requires an explicit score or declared zero fallback")
        auxiliary = 0 if geometry_score is None else geo_weight * geometry_score
        return beta * harm_score + auxiliary - weight * quality_penalty - cfg.perturbation_weight * u.square().mean()

    def _task_gate(self, evaluation):
        return all(evaluation.per_task_harm[k].detach().item() >= threshold
                   for k, threshold in self.config.harm_gate_by_task.items())

    def optimize(self, initial_u: torch.Tensor,
                 evaluate: Callable[[torch.Tensor], FixedInputEvaluation], *,
                 contextual_evaluation: bool = False) -> FixedInputResult:
        cfg = self.config
        if not isinstance(contextual_evaluation, bool):
            raise TypeError('contextual_evaluation must be boolean')
        if not isinstance(initial_u, torch.Tensor) or not initial_u.is_floating_point() or initial_u.numel() == 0:
            raise TypeError('initial_u must be a nonempty floating tensor')
        if not torch.isfinite(initial_u).all() or (initial_u.detach().abs() > 1).any():
            raise ValueError('Initial normalized physical input is outside [-1,1]')
        u = initial_u.detach().clone()
        momentum = torch.zeros_like(u)
        phase = 'B' if cfg.mode == 'stage_b_only' else 'A'
        stage_b_entered = phase == 'B'
        evaluations = 0
        gradients = 0
        history = []
        best_u = u.clone()
        best_harm = -math.inf
        best_gated_u, best_gated_harm = None, None
        final_gate = False
        status = 'iteration_budget_exhausted'

        def score(point, *, need_gradient):
            nonlocal evaluations
            evaluations += 1
            with torch.enable_grad():
                # External generator/task processes only need a VJP for the
                # current point. Trial candidates still run full forward scores.
                value = evaluate(point, stage=phase, need_gradient=need_gradient) if contextual_evaluation else evaluate(point)
                return value.validate(cfg.harm_gate_by_task)

        def consider(point, evaluation):
            nonlocal best_u, best_harm, best_gated_u, best_gated_harm
            value = float(evaluation.harm_score.detach())
            if value > best_harm:
                best_u, best_harm = point.detach().clone(), value
            if self._task_gate(evaluation) and (best_gated_harm is None or value > best_gated_harm):
                best_gated_u, best_gated_harm = point.detach().clone(), value

        for step in range(cfg.outer_iterations):
            u_var = u.detach().requires_grad_(True)
            try:
                current = score(u_var, need_gradient=True)
            except EvaluationBudgetExhausted:
                status = 'evaluation_budget_exhausted'
                break
            if not current.quality_feasible or not current.geometry_feasible:
                status = 'initial_or_current_input_infeasible'
                break
            consider(u, current)
            final_gate = self._task_gate(current)
            optimization_harm = current.stage_a_harm_score if phase == "A" and current.stage_a_harm_score is not None else current.harm_score
            objective = self.compose(phase, optimization_harm, current.quality_penalty, u_var, current.geometry_score)
            gradient, = torch.autograd.grad(objective, u_var)
            gradients += 1
            if not torch.isfinite(gradient).all():
                status = 'nonfinite_gradient'
                break
            proposed_momentum = (cfg.momentum * momentum + gradient.detach() /
                                 gradient.detach().abs().mean().clamp_min(1e-12)) if phase == 'B' else momentum
            raw_direction = proposed_momentum if phase == 'B' else gradient.detach()
            # Positive scalar normalization preserves relative components and inner
            # products. Coordinatewise sign does not preserve multi-task ascent.
            magnitude = raw_direction.abs().max()
            direction = raw_direction.sign() if cfg.direction_rule == 'sign' else raw_direction / torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
            if cfg.direction_rule == 'linf_normalized' and not bool(direction.abs().max() > 0):
                status = 'stationary_surrogate'
                break
            step_size = cfg.step_size_B if phase == 'B' else cfg.step_size_A
            accepted = False
            accepted_evaluation = current
            attempts = []
            budget_exhausted = False
            for retry in range(cfg.max_backtracks + 1):
                candidate = (u + step_size * cfg.backtrack_factor ** retry * direction).clamp(-1, 1)
                try:
                    trial = score(candidate.detach().requires_grad_(True), need_gradient=False)
                except EvaluationBudgetExhausted:
                    status = 'evaluation_budget_exhausted'
                    budget_exhausted = True
                    attempts.append({'backtrack': retry, 'budget_exhausted': True})
                    break
                feasible = trial.quality_feasible and trial.geometry_feasible
                # Selection uses the common B score over every measured feasible
                # trial, even when the current A objective rejects the update.
                # This does not commit u/momentum or advance the phase.
                if feasible:
                    consider(candidate, trial)
                trial_harm = trial.stage_a_harm_score if phase == 'A' and trial.stage_a_harm_score is not None else trial.harm_score
                trial_objective = self.compose(phase, trial_harm, trial.quality_penalty, candidate, trial.geometry_score) if feasible else None
                improved = feasible and float(trial_objective.detach()) >= float(objective.detach())
                attempts.append({'backtrack': retry, 'quality_feasible': trial.quality_feasible,
                                 'geometry_feasible': trial.geometry_feasible,
                                 'harm': float(trial.harm_score.detach()),
                                 'stage_objective': float(trial_objective.detach()) if feasible else None,
                                 'current_stage_objective': float(objective.detach()),
                                 'objective_nondecreasing': improved})
                if feasible and (not cfg.monotone_ascent or improved):
                    u = candidate.detach().clone()
                    momentum = proposed_momentum.detach().clone()
                    accepted_evaluation = trial
                    accepted = True
                    break
            # On rejection both u and momentum remain at their previous values.
            # All gate/selection scores below correspond to the retained current u.
            value = float(accepted_evaluation.harm_score.detach())
            consider(u, accepted_evaluation)
            final_gate = self._task_gate(accepted_evaluation)
            history.append({'outer_step': step, 'stage': phase, 'accepted': accepted,
                            'attempts': attempts, 'harm_score': value,
                            'per_task_harm': {k: float(v.detach()) for k, v in accepted_evaluation.per_task_harm.items()},
                            'quality_penalty': float(accepted_evaluation.quality_penalty.detach()),
                            'quality_feasible': accepted_evaluation.quality_feasible,
                            'geometry_feasible': accepted_evaluation.geometry_feasible,
                            'gradient_norm': float(gradient.detach().norm()),
                            'normalized_u_max': float(u.abs().max()),
                            'task_gate_met': final_gate,
                            'metrics': dict(accepted_evaluation.metrics)})
            if budget_exhausted:
                break
            if cfg.monotone_ascent and not accepted:
                # At an unchanged deterministic point the same direction and
                # step sizes would be retried forever. Preserve u and momentum.
                status = 'line_search_failed'
                break
            if phase == 'A' and cfg.mode == 'two_stage':
                a_harms = accepted_evaluation.stage_a_per_task_harm or accepted_evaluation.per_task_harm
                a_gate = all(float(a_harms[k].detach()) >= v for k,v in cfg.harm_gate_by_task.items())
                stable = accepted_evaluation.stability_per_task_harm
                stable_gate = not cfg.require_stability_gate or (stable is not None and all(float(stable[k].detach()) >= v for k,v in cfg.harm_gate_by_task.items()))
                history[-1]['stage_a_task_gate_met'] = a_gate
                history[-1]['stability_gate_met'] = stable_gate
                if step + 1 >= cfg.stage1_min_steps and a_gate and stable_gate:
                    phase = 'B'
                    stage_b_entered = True
                    momentum.zero_()
                elif step + 1 >= cfg.stage1_max_steps:
                    status = 'stage_a_gate_unreachable'
                    break
        return FixedInputResult(best_u=best_u, final_u=u, final_momentum=momentum,
                                best_harm=best_harm, phase=phase, status=status,
                                history=history, evaluations=evaluations,
                                gradient_evaluations=gradients, stage_b_entered=stage_b_entered,
                                final_task_gate_met=final_gate,
                                best_gated_u=best_gated_u, best_gated_harm=best_gated_harm)
