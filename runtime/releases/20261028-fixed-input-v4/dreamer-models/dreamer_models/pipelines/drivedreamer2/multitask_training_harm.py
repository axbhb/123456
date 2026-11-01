"""Fixed-scale aggregation of task-specific one-step training harms.

Each task keeps its native inner loss and its own surrogate parameter space.
Only relative outer harms are normalized and aggregated. All adapters consume
one shared final video [B,T,V,C,H,W] and the same current perturbed geometry;
differentiable video AND label paths remain connected to the physical input.

This module does not compute or subtract quality penalties. The outer optimizer
applies the shared video-quality penalty exactly once. It does not evaluate held-
out tasks: planning used for optimization, calibration, gating or selection must
not subsequently be described as an unseen task.

For multiple checkpoints, supply a sequence of separately constructed objective
instances per task. Their relative harms are averaged BEFORE task normalization
and aggregation. There is no implicit checkpoint loading or model-state update.
"""

from dataclasses import dataclass, field
import math
from numbers import Real
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Protocol, Sequence

import torch

from .attack_optimization import AttackObjectiveResult, validate_objective_result
from .training_harm_objective import OneStepTrainingHarmObjective


class TaskAdapter(Protocol):
    """Task-specific data/label construction and native scalar loss contract.

    Use loss and selected_parameter_names when constructing the corresponding
    OneStepTrainingHarmObjective. build_batch must rebuild labels from current
    geometry and preserve any differentiable label path. validate_labels must
    check the task's annotation/geometry constraints and return a Python bool;
    invalid or unverifiable labels must not be silently accepted. Discrete label
    generation or matching approximations belong in the adapter documentation.
    """

    selected_parameter_names: Optional[Sequence[str]]

    def build_batch(self, final_video: torch.Tensor, perturbed_geometry, scene_meta): ...

    def validate_labels(self, batch, perturbed_geometry, scene_meta) -> bool: ...

    def loss(self, functional_forward: Callable, batch) -> torch.Tensor: ...


@dataclass(frozen=True)
class MultiTaskHarmResult(AttackObjectiveResult):
    """Differentiable per-task values plus detached scalar diagnostic metrics.

    raw_harm and normalized_harm retain their graphs for optional gradient
    diagnostics in the shared physical-input space. weights are the aggregation
    derivative weights, not weights applied inside any task's virtual SGD.
    """

    raw_harm: Mapping[str, torch.Tensor] = field(default_factory=dict)
    normalized_harm: Mapping[str, torch.Tensor] = field(default_factory=dict)
    weights: Mapping[str, torch.Tensor] = field(default_factory=dict)


def _task_names(names, label, *, allow_empty=False):
    if isinstance(names, (str, bytes)):
        raise TypeError(f'{label} must be a sequence of task names.')
    result = tuple(names)
    if not allow_empty and not result:
        raise ValueError(f'{label} must not be empty.')
    if any(not isinstance(name, str) or not name or '/' in name for name in result):
        raise ValueError(f'{label} must contain non-empty strings without slashes.')
    if len(set(result)) != len(result):
        raise ValueError(f'{label} must not contain duplicate names.')
    return result


def _fixed_number(value, name, *, positive=False):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.requires_grad:
            raise ValueError(f'{name} must be a fixed real scalar without gradients.')
        value = value.detach().item()
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f'{name} must be a real number.')
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        limit = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {limit}.')
    return value


class MultiTaskTrainingHarm:
    """Aggregate independent tasks using locked calibration scales.

    task_objectives and task_adapters must have exactly active_tasks keys. Do not
    construct/pass objectives or adapters for held_out_tasks. scales are copied
    as positive Python floats, never inferred from the current attack. Record
    calibration provenance outside this numerical evaluator. task_weights are
    fixed nonnegative weights summing to one for mean aggregation; softmin uses
    its unweighted definition and therefore rejects supplied task_weights.

    The unshifted soft-min is -T*logsumexp(-h/T), so all-zero task harms produce
    -T*log(K), not zero. Stage gates must inspect individual normalized_harm.
    """

    def __init__(
        self,
        task_objectives: Mapping,
        task_adapters: Mapping[str, TaskAdapter],
        scales: Mapping,
        *,
        active_tasks: Sequence[str],
        held_out_tasks: Sequence[str] = (),
        aggregation: str = 'mean',
        temperature: float = 1.0,
        task_weights: Optional[Mapping] = None,
    ):
        self.active_tasks = _task_names(active_tasks, 'active_tasks')
        self.held_out_tasks = _task_names(held_out_tasks, 'held_out_tasks', allow_empty=True)
        if set(self.active_tasks).intersection(self.held_out_tasks):
            raise ValueError('active_tasks and held_out_tasks must be disjoint.')
        active = set(self.active_tasks)
        for label, mapping in (
            ('task_objectives', task_objectives), ('task_adapters', task_adapters),
            ('scales', scales),
        ):
            if not isinstance(mapping, Mapping):
                raise TypeError(f'{label} must be a mapping.')
            if set(mapping) != active:
                raise ValueError(
                    f'{label} must contain exactly active_tasks; held-out or '
                    'inactive task entries are forbidden.'
                )
        if aggregation not in {'mean', 'softmin'}:
            raise ValueError("aggregation must be 'mean' or 'softmin'.")
        self.aggregation = aggregation
        self.temperature = _fixed_number(temperature, 'temperature', positive=True)
        self.scales = MappingProxyType({
            name: _fixed_number(scales[name], f'scale[{name}]', positive=True)
            for name in self.active_tasks
        })
        if task_weights is not None and aggregation != 'mean':
            raise ValueError('task_weights are only supported for mean aggregation.')
        if task_weights is None:
            weights = {name: 1.0 / len(active) for name in self.active_tasks}
        else:
            if not isinstance(task_weights, Mapping) or set(task_weights) != active:
                raise ValueError('task_weights must contain exactly active_tasks.')
            weights = {
                name: _fixed_number(task_weights[name], f'weight[{name}]')
                for name in self.active_tasks
            }
            if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError('task_weights must sum to one; they are not renormalized implicitly.')
        self.task_weights = MappingProxyType(weights)
        objective_pool, adapters, identifiers = {}, {}, {}
        for name in self.active_tasks:
            entry = task_objectives[name]
            pool = (entry,) if isinstance(entry, OneStepTrainingHarmObjective) else tuple(entry)
            if not pool or any(not isinstance(obj, OneStepTrainingHarmObjective) for obj in pool):
                raise TypeError(f'task_objectives[{name}] must contain one-step objective instances.')
            if any(obj.score_mode != 'relative' for obj in pool):
                raise ValueError(f'Task {name} must use relative training harm, not absolute loss.')
            adapter = task_adapters[name]
            if not callable(getattr(adapter, 'build_batch', None)):
                raise TypeError(f'Task {name} adapter requires build_batch.')
            if not callable(getattr(adapter, 'validate_labels', None)):
                raise TypeError(f'Task {name} adapter requires validate_labels.')
            objective_pool[name] = pool
            adapters[name] = adapter
            identifiers[name] = tuple(
                obj.state_id if obj.state_id is not None else f'state_{index}'
                for index, obj in enumerate(pool)
            )
        self.task_objectives = MappingProxyType(objective_pool)
        self.task_adapters = MappingProxyType(adapters)
        self.state_ids = MappingProxyType(identifiers)

    def evaluate(self, final_video, perturbed_geometry, scene_meta=None) -> MultiTaskHarmResult:
        if not isinstance(final_video, torch.Tensor) or not final_video.is_floating_point():
            raise TypeError('final_video must be a floating-point tensor.')
        if final_video.ndim != 6 or any(size <= 0 for size in final_video.shape):
            raise ValueError('final_video must have non-empty [B,T,V,C,H,W] dimensions.')
        raw_harm, normalized, metrics = {}, {}, {}
        # Labels and preprocessing may depend differentiably on geometry/video;
        # an outer no_grad for candidate checks must not silently sever them.
        with torch.enable_grad():
            for name in self.active_tasks:
                adapter = self.task_adapters[name]
                batch = adapter.build_batch(final_video, perturbed_geometry, scene_meta)
                with torch.no_grad():
                    labels_valid = adapter.validate_labels(batch, perturbed_geometry, scene_meta)
                if not isinstance(labels_valid, bool):
                    raise TypeError(f'Task {name} validate_labels must return a Python bool.')
                if not labels_valid:
                    raise ValueError(f'Task {name} labels are invalid or unverifiable.')
                scores = []
                for index, objective in enumerate(self.task_objectives[name]):
                    result = objective.evaluate(batch, state_id=objective.state_id)
                    score, state_metrics = validate_objective_result(result)
                    # Scalar device copies retain their gradient paths if task
                    # surrogates were explicitly placed on different devices.
                    scores.append(score.to(device=final_video.device))
                    for metric, value in state_metrics.items():
                        metrics[f'task/{name}/state/{index}/{metric}'] = value
                raw = torch.stack(scores).mean()
                raw_harm[name] = raw
                normalized[name] = raw / self.scales[name]
                metrics[f'task/{name}/raw_harm'] = raw.detach().item()
                metrics[f'task/{name}/normalized_harm'] = normalized[name].detach().item()
                metrics[f'task/{name}/scale'] = self.scales[name]
                metrics[f'task/{name}/state_count'] = float(len(scores))
                metrics[f'task/{name}/labels_valid'] = 1.0
            values = torch.stack([normalized[name] for name in self.active_tasks])
            if self.aggregation == 'mean':
                weights = values.new_tensor([self.task_weights[name] for name in self.active_tasks])
                score = (weights * values).sum()
            else:
                weights = torch.softmax(-values / self.temperature, dim=0)
                score = -self.temperature * torch.logsumexp(-values / self.temperature, dim=0)
            weight_mapping = dict(zip(self.active_tasks, weights.unbind()))
            for name, weight in weight_mapping.items():
                metrics[f'task/{name}/aggregate_weight'] = weight.detach().item()
            metrics.update({
                'aggregate_score': score.detach().item(),
                'mean_normalized_harm': values.detach().mean().item(),
                'min_normalized_harm': values.detach().min().item(),
                'active_task_count': float(len(self.active_tasks)),
                'temperature': self.temperature,
            })
            result = MultiTaskHarmResult(
                score=score, metrics=MappingProxyType(metrics),
                raw_harm=MappingProxyType(raw_harm),
                normalized_harm=MappingProxyType(normalized),
                weights=MappingProxyType(weight_mapping),
            )
            validate_objective_result(result)
            return result
