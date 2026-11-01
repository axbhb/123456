"""Connect final-video generation, current geometry labels, task harm and quality.

Real task adapters and a native replay function must be supplied explicitly. This
module does not silently substitute toy models for an unavailable driving model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Optional

import torch

from .video_quality_constraints import _pairing_json

from .fixed_input_optimization import (
    FixedInputConfig, FixedInputEvaluation, FixedInputPhysCondOptimizer,
)


@dataclass
class ResearchRunOutput:
    optimization: Any
    replay_video: Optional[torch.Tensor]
    perturbed_geometry: Any
    logs: Mapping[str, Any]


class FixedInputResearchRunner:
    def __init__(self, *, generator: Callable, native_replay: Callable,
                 build_geometry: Callable, geometry_is_valid: Callable,
                 task_harm, quality, pairing: Mapping[str, Any],
                 scene_meta: Mapping[str, Any], config: FixedInputConfig,
                 replay_atol: float = 1e-5, replay_rtol: float = 1e-4,
                 stage_a_task_harm=None, stability_task_harm=None, geometry_auxiliary=None):
        if scene_meta.get('split_role') != 'attack_development':
            raise ValueError('Optimization requires an explicitly declared attack_development split')
        if scene_meta.get('scene_id') != pairing.get('scene_id'):
            raise ValueError('Optimization scene and clean-reference scene differ')
        if set(config.harm_gate_by_task) != set(task_harm.active_tasks):
            raise ValueError('Optimizer gates must exactly match active objectives')
        if replay_atol < 0 or replay_rtol < 0:
            raise ValueError('Replay tolerances must be nonnegative')
        for evaluator in (stage_a_task_harm, stability_task_harm):
            if evaluator is not None and (tuple(evaluator.active_tasks) != tuple(task_harm.active_tasks) or tuple(evaluator.held_out_tasks) != tuple(task_harm.held_out_tasks)):
                raise ValueError("All stage/stability evaluators must share active and held-out tasks")
        if config.require_stability_gate and stability_task_harm is None:
            raise ValueError("Independent development stability evaluator required")
        if (config.stage1_geometry_weight or config.stage2_geometry_weight) and geometry_auxiliary is None:
            raise ValueError("Geometry auxiliary was configured but not supplied")
        self.stage_a_task_harm, self.stability_task_harm = stage_a_task_harm, stability_task_harm
        self.geometry_auxiliary = geometry_auxiliary
        self.generator, self.native_replay = generator, native_replay
        self.build_geometry, self.geometry_is_valid = build_geometry, geometry_is_valid
        self.task_harm, self.quality = task_harm, quality
        self.pairing, self.scene_meta = dict(pairing), dict(scene_meta)
        self.config = config
        self.replay_atol, self.replay_rtol = replay_atol, replay_rtol
        self.generation_calls = 0
        self.task_harm_calls = 0

    def _actual_pairing(self, output, context):
        actual = getattr(output, 'pairing', None)
        if not isinstance(actual, Mapping):
            raise ValueError(context + ' must provide pairing from its actual generation call')
        actual_json = _pairing_json(actual, output.video.shape)
        expected_json = _pairing_json(self.pairing, output.video.shape)
        if actual_json != expected_json:
            raise ValueError(context + ' actual pairing differs from the clean reference')
        # Reject contradictory old-style logs as well as full-identity mismatch.
        if output.logs.get('seed') != actual['seed']:
            raise ValueError(context + ' seed log contradicts actual pairing')
        if list(output.logs.get('camera_names', ())) != list(actual['views']):
            raise ValueError(context + ' camera_names log contradicts actual pairing')
        for name in ('model_fingerprint', 'scheduler', 'preprocessing'):
            if name in output.logs and output.logs[name] != actual[name]:
                raise ValueError(context + ' ' + name + ' log contradicts actual pairing')
        return json.loads(actual_json)

    def evaluate(self, u):
        geometry = self.build_geometry(u)
        legal = self.geometry_is_valid(geometry)
        if not isinstance(legal, bool):
            raise TypeError('Geometry validity must be an explicit boolean')
        if not legal:
            zero = u.sum() * 0
            return FixedInputEvaluation(zero, zero, {k: zero for k in self.task_harm.active_tasks},
                                        False, False, {'geometry_rejected_before_generation': 1.0})
        output = self.generator(geometry, differentiable=True)
        self.generation_calls += 1
        actual_pairing = self._actual_pairing(output, 'Generator')
        video = output.video
        quality = self.quality.evaluate(video, pairing=actual_pairing,
                                        denoise_predictions=output.denoise_predictions)
        harm = self.task_harm.evaluate(video, geometry, self.scene_meta)
        self.task_harm_calls += 1
        metrics = dict(harm.metrics)
        for name, value in quality.per_metric.items():
            metrics['quality/' + name] = float(value.detach())
        for name, value in quality.normalized.items():
            metrics['quality_normalized/' + name] = float(value.detach())
        auxiliary = self.geometry_auxiliary.evaluate(video, geometry) if self.geometry_auxiliary is not None else None
        if auxiliary is not None:
            metrics.update(auxiliary.metrics)
        a_harm = self.stage_a_task_harm.evaluate(video, geometry, self.scene_meta) if self.stage_a_task_harm is not None else None
        stable = self.stability_task_harm.evaluate(video, geometry, self.scene_meta) if self.stability_task_harm is not None else None
        self.task_harm_calls += int(a_harm is not None) + int(stable is not None)
        return FixedInputEvaluation(harm.score, quality.penalty, harm.normalized_harm,
                                    quality.feasible, legal and (auxiliary is None or auxiliary.feasible), metrics,
                                    geometry_score=auxiliary.score if auxiliary is not None else None,
                                    stage_a_harm_score=a_harm.score if a_harm is not None else None,
                                    stage_a_per_task_harm=a_harm.normalized_harm if a_harm is not None else None,
                                    stability_per_task_harm=stable.normalized_harm if stable is not None else None)

    def run(self, initial_u):
        self.generation_calls = 0
        self.task_harm_calls = 0
        result = FixedInputPhysCondOptimizer(self.config).optimize(initial_u, self.evaluate)
        if result.best_harm == -float('inf'):
            raise RuntimeError('No feasible input found; refusing to export an invalid dataset')
        selected_u = result.best_gated_u if result.best_gated_u is not None else result.best_u
        geometry = self.build_geometry(selected_u)
        differentiable_path = self.generator(geometry, differentiable=False)
        self.generation_calls += 1
        self._actual_pairing(differentiable_path, 'Generator replay')
        # This must use the native final generator with fixed input and paired seed.
        replay = self.native_replay(geometry)
        native_pairing = self._actual_pairing(replay, 'Native replay')
        if differentiable_path.video.shape != replay.video.shape or not torch.allclose(
                differentiable_path.video.detach(), replay.video.detach(),
                atol=self.replay_atol, rtol=self.replay_rtol):
            raise RuntimeError('Fixed-input/native replay mismatch; no experiment pool exported')
        replay_quality = self.quality.evaluate(replay.video, pairing=native_pairing,
                                               denoise_predictions=replay.denoise_predictions)
        replay_aux = self.geometry_auxiliary.evaluate(replay.video, geometry) if self.geometry_auxiliary is not None else None
        if not replay_quality.feasible or not self.geometry_is_valid(geometry) or (replay_aux is not None and not replay_aux.feasible):
            raise RuntimeError('Native replay failed quality or geometry constraints')
        # Virtual training still needs parameter derivatives for a detached native video.
        with torch.enable_grad():
            replay_harm = self.task_harm.evaluate(replay.video.detach(), geometry, self.scene_meta)
        self.task_harm_calls += 1
        final_evaluation = FixedInputEvaluation(
            replay_harm.score, replay_quality.penalty, replay_harm.normalized_harm,
            replay_quality.feasible, True).validate(self.config.harm_gate_by_task)
        native_task_gate = all(
            final_evaluation.per_task_harm[k].detach().item() >= threshold
            for k, threshold in self.config.harm_gate_by_task.items())
        failed_statuses = {'stage_a_gate_unreachable', 'nonfinite_gradient',
                           'initial_or_current_input_infeasible'}
        pool_eligible = (result.best_gated_u is not None and native_task_gate
                         and result.status not in failed_statuses)
        maximum_difference = float((replay.video.detach() - differentiable_path.video.detach()).abs().max())
        # Failed ablations remain inspectable in logs; their video cannot enter a pool.
        return ResearchRunOutput(result, replay.video.detach() if pool_eligible else None, geometry,
                                 {'generation_calls': self.generation_calls,
                                  'task_harm_calls': self.task_harm_calls,
                                  'native_replays': 1, 'replay_max_abs_difference': maximum_difference,
                                  'replay_quality_feasible': replay_quality.feasible,
                                  'selected_normalized_u': selected_u.detach().cpu().tolist(),
                                  'native_per_task_harm': {k: float(v.detach()) for k, v in replay_harm.normalized_harm.items()},
                                  'native_task_gate_met': native_task_gate,
                                  'pool_eligible': pool_eligible,
                                  'termination_status': result.status,
                                  'active_tasks': list(self.task_harm.active_tasks),
                                  'held_out_tasks': list(self.task_harm.held_out_tasks),
                                  'scope': self.scene_meta.get('experiment_scope', 'unspecified'),
                                  'final_test_used': False})
