"""Differentiable one-step training-harm objective for the PhysCond Stage 2 hook.

This is a short-horizon training proxy, not evidence of full retraining damage.
The caller supplies a surrogate, fixed clean batches, a differentiable video-to-
batch adapter, and a scalar detector loss. No MMDetection3D dependency is needed:
loss_fn(functional_forward, batch) may call functional_forward(*args, **kwargs)
with the detector's native loss arguments and aggregate its losses.

Parameters and buffers are snapshotted at construction. Every call starts from
that same pretrained state. By default all parameters are virtually adapted;
select the detection head with parameter_names for an affordable real-model
pilot. Originally frozen parameters can still be selected. The real model's
weights, buffers, .grad fields, and per-module training flags are preserved.

The default score is validation loss after a poisoned virtual SGD step minus
validation loss after an otherwise matched clean-reference step. Explicit
score_mode='post_update' maximizes the former absolute loss. Neither is the
frozen detector's instantaneous loss on the poisoned samples.

training_mode=False disables standard dropout and BN running-stat updates while
retaining parameter/input gradients. If enabled, virtual training buffers are
isolated and propagated to validation, and poison/reference paths start from the
same PyTorch RNG state. Use matching batch shapes for matching stochastic layers.
BN running-stat updates follow PyTorch's non-differentiable buffer semantics;
the default eval mode avoids this omitted buffer-gradient path.
The caller must disable or fix Python/NumPy randomness and random augmentation;
the generic batch/loss adapter owns deterministic preprocessing. Instances briefly
change the surrogate's mode and must not share that model with concurrent calls.
"""

from contextlib import contextmanager
import math
import inspect
from typing import Callable, Optional, Sequence

import torch
from torch import nn
try:
    from torch.func import functional_call as _functional_call
except ImportError:  # PyTorch 1.13 for legacy MMDetection3D / MMCV stacks.
    from torch.nn.utils.stateless import functional_call as _functional_call

from .attack_optimization import AttackObjectiveResult, AttackStepContext


def _legacy_strict_state(model, state):
    """Implement strict/tied state semantics absent from torch 1.13.0's API."""
    # 1.13 also lacks named_parameters(remove_duplicate=False). Walk registered
    # state directly so shared modules and distinct tied parameters keep aliases.
    named = []
    def visit(module, prefix, ancestors):
        if id(module) in ancestors:
            raise ValueError('Cyclic module registration is unsupported.')
        ancestors = ancestors | {id(module)}
        for mapping in (module._parameters, module._buffers):
            named.extend((prefix + name, tensor) for name, tensor in mapping.items()
                         if tensor is not None)
        for name, child in module._modules.items():
            if child is not None:
                visit(child, prefix + name + '.', ancestors)
    visit(model, '', set())
    all_names = {name for name, _ in named}
    unexpected = sorted(set(state) - all_names)
    groups = {}
    slots = {}
    for name, tensor in named:
        groups.setdefault(id(tensor), []).append(name)
        parent, _, attribute = name.rpartition('.')
        owner = model.get_submodule(parent) if parent else model
        slots.setdefault((id(owner), attribute), []).append(name)
    missing = [names[0] for names in groups.values() if not any(name in state for name in names)]
    if missing or unexpected:
        raise RuntimeError(f'Invalid virtual state: missing keys {sorted(missing)}; unexpected keys {unexpected}.')
    expanded = {}
    for names in groups.values():
        provided = [state[name] for name in names if name in state]
        if any(value is not provided[0] for value in provided[1:]):
            raise ValueError(f'Conflicting virtual tensors for tied state aliases: {names}.')
        for name in names:
            expanded[name] = provided[0]
    # 1.13's class-based replacer maps a shared module attribute to the last
    # alias visited. Read reassigned buffers from that physical attribute slot.
    last_alias = {name: names[-1] for names in slots.values() for name in names}
    return expanded, last_alias


def _call_with_virtual_state(model, parameters, buffers, args, kwargs):
    if set(parameters).intersection(buffers):
        raise ValueError('Parameter and buffer state names must be disjoint.')
    supported = inspect.signature(_functional_call).parameters
    options = {}
    if 'strict' in supported:
        options['strict'] = True
    if 'tie_weights' in supported:
        options['tie_weights'] = True
    merged = {**parameters, **buffers}
    aliases = {}
    if 'strict' not in supported or 'tie_weights' not in supported:
        merged, aliases = _legacy_strict_state(model, merged)
    # Both modern state swapping and the 1.13 class-based backend must restore
    # the original objects, including when the model raises during forward.
    originals = [
        (module, dict(module._parameters), dict(module._buffers), module.__class__,
         {name: module.__dict__[name] for name in ('_orig_class', '_attr_to_path')
          if name in module.__dict__})
        for module in model.modules()
    ]
    try:
        return _functional_call(model, merged, args, kwargs, **options)
    finally:
        # A single explicit mapping preserves reassignments under both APIs.
        for name in parameters:
            parameters[name] = merged[aliases.get(name, name)]
        for name in buffers:
            buffers[name] = merged[aliases.get(name, name)]
        for module, original_parameters, original_buffers, original_class, private in originals:
            module.__class__ = original_class
            for name in ('_orig_class', '_attr_to_path'):
                if name in private:
                    module.__dict__[name] = private[name]
                else:
                    module.__dict__.pop(name, None)
            module._parameters.clear()
            module._parameters.update(original_parameters)
            module._buffers.clear()
            module._buffers.update(original_buffers)


@contextmanager
def _temporary_mode(model, training):
    states = [(module, module.training) for module in model.modules()]
    try:
        for module, _ in states:
            module.training = training
        yield
    finally:
        for module, original in states:
            module.training = original


def _checked_loss(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f'{name} must be a floating-point tensor.')
    if value.ndim != 0:
        raise ValueError(f'{name} must be a scalar (0-dimensional tensor).')
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError(f'{name} must be finite.')
    return value


class OneStepTrainingHarmObjective:
    """Callable Stage 2 objective differentiating exactly through one SGD step.

    Args:
        model: Pretrained downstream surrogate, already on its intended device.
        loss_fn: loss_fn(functional_forward, batch) -> scalar Tensor.
        poison_batch_builder: builder(decoded_video, context) -> batch. Preserve
            the decoded video's gradient path and use the same annotation
            protocol as the clean-reference batch. May be None when using
            evaluate(poison_batch) directly.
        clean_train_batch: Fixed real/clean portion of the inner training mix.
        validation_batch: Independent fixed clean batch for the outer loss;
            do not use the final reported test split here.
        inner_lr: Positive plain virtual SGD learning rate, without momentum or
            weight decay. Include intended loss regularization in loss_fn.
        poison_fraction: Fraction in [0, 1]. Inner loss is (1-rho)*clean_train_loss
            + rho*synthetic_train_loss. Normalize each component per example in
            loss_fn; batch sizes need not be equal for this weighted mixture.
        parameter_names: Exact named_parameters keys updated virtually. Others
            stay fixed. None selects every parameter.
        clean_reference_batch: Unattacked synthetic counterpart, required by the
            default relative score. Match poison count, annotation protocol and
            shape. The reference batch must be independent of the attack.
        score_mode: 'relative' (default) or 'post_update'. A supplied reference
            is evaluated and logged in either mode.
        training_mode: Inner forward mode; outer validation always uses eval.
        state_id: Optional identifier of this immutable pretrained snapshot. It
            is metadata, not a checkpoint selector; use separate instances for
            different checkpoints.

    Returns:
        AttackObjectiveResult with a differentiable score to maximize and detached
        scalar metrics. Quality/perturbation penalties remain the optimizer's job.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        poison_batch_builder: Optional[Callable],
        clean_train_batch,
        validation_batch,
        *,
        inner_lr: float,
        poison_fraction: float,
        parameter_names: Optional[Sequence[str]] = None,
        clean_reference_batch=None,
        score_mode: str = 'relative',
        training_mode: bool = False,
        state_id: Optional[str] = None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError('model must be a torch.nn.Module.')
        if not callable(loss_fn):
            raise TypeError('loss_fn must be callable.')
        if poison_batch_builder is not None and not callable(poison_batch_builder):
            raise TypeError('poison_batch_builder must be callable or None.')
        if state_id is not None and (not isinstance(state_id, str) or not state_id.strip()):
            raise ValueError('state_id must be a non-empty string or None.')
        inner_lr, poison_fraction = float(inner_lr), float(poison_fraction)
        if not math.isfinite(inner_lr) or inner_lr <= 0:
            raise ValueError('inner_lr must be finite and positive.')
        if not math.isfinite(poison_fraction) or not 0 <= poison_fraction <= 1:
            raise ValueError('poison_fraction must be finite and in [0, 1].')
        if score_mode not in {'relative', 'post_update'}:
            raise ValueError("score_mode must be 'relative' or 'post_update'.")
        if score_mode == 'relative' and clean_reference_batch is None:
            raise ValueError('relative scoring requires clean_reference_batch.')
        if not isinstance(training_mode, bool):
            raise TypeError('training_mode must be a bool.')

        self.model = model
        self.loss_fn = loss_fn
        self.poison_batch_builder = poison_batch_builder
        self.clean_train_batch = clean_train_batch
        self.validation_batch = validation_batch
        self.clean_reference_batch = clean_reference_batch
        self.inner_lr = inner_lr
        self.poison_fraction = poison_fraction
        self.score_mode = score_mode
        self.training_mode = training_mode
        self.state_id = state_id
        self._initial_parameters = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        self._initial_buffers = {
            name: value.detach().clone() for name, value in model.named_buffers()
        }
        if isinstance(parameter_names, (str, bytes)):
            raise TypeError('parameter_names must be a sequence of exact names, not a string.')
        names = tuple(self._initial_parameters if parameter_names is None else parameter_names)
        if not names:
            raise ValueError('At least one parameter must be selected for virtual SGD.')
        if not all(isinstance(name, str) for name in names):
            raise TypeError('parameter_names must contain strings.')
        if len(set(names)) != len(names):
            raise ValueError('parameter_names must not contain duplicates.')
        unknown = set(names).difference(self._initial_parameters)
        if unknown:
            raise ValueError(f'Unknown surrogate parameter names: {sorted(unknown)}')
        if any(not self._initial_parameters[name].is_floating_point() for name in names):
            raise TypeError('Selected parameters must be floating point.')
        self.parameter_names = names
        # Do not enumerate or initialize CUDA when the surrogate is on CPU.
        self._cuda_devices = sorted({
            tensor.device.index
            for tensor in (*self._initial_parameters.values(), *self._initial_buffers.values())
            if tensor.device.type == 'cuda'
        })

    def _new_state(self):
        selected = set(self.parameter_names)
        parameters = {
            name: value.clone().requires_grad_(name in selected)
            for name, value in self._initial_parameters.items()
        }
        buffers = {name: value.clone() for name, value in self._initial_buffers.items()}
        return parameters, buffers

    def _loss(self, parameters, buffers, batch, *, training, name):
        def forward(*args, **kwargs):
            return _call_with_virtual_state(self.model, parameters, buffers, args, kwargs)

        with _temporary_mode(self.model, training):
            return _checked_loss(self.loss_fn(forward, batch), name)

    def _virtual_update(self, synthetic_batch, *, create_graph):
        parameters, buffers = self._new_state()
        clean_loss = self._loss(
            parameters, buffers, self.clean_train_batch,
            training=self.training_mode, name='clean_train_loss',
        )
        synthetic_loss = self._loss(
            parameters, buffers, synthetic_batch,
            training=self.training_mode, name='synthetic_train_loss',
        )
        mixed_loss = (
            (1 - self.poison_fraction) * clean_loss
            + self.poison_fraction * synthetic_loss
        )
        if not mixed_loss.requires_grad:
            raise ValueError('The mixed training loss must depend on selected surrogate parameters.')
        selected = tuple(parameters[name] for name in self.parameter_names)
        gradients = torch.autograd.grad(
            mixed_loss, selected, create_graph=create_graph, allow_unused=True,
        )
        if all(gradient is None for gradient in gradients):
            raise ValueError('The mixed training loss does not use any selected parameters.')
        updated = dict(parameters)
        for name, parameter, gradient in zip(self.parameter_names, selected, gradients):
            if gradient is not None:
                updated[name] = parameter - self.inner_lr * gradient
        if not create_graph:
            updated = {name: value.detach() for name, value in updated.items()}
        validation_loss = self._loss(
            updated, buffers, self.validation_batch, training=False,
            name='post_update_val_loss',
        )
        return validation_loss, clean_loss, synthetic_loss, mixed_loss

    def evaluate(self, poison_batch, *, state_id=None) -> AttackObjectiveResult:
        """Score a task batch directly, preserving all differentiable label paths.

        The caller builds the batch from the final video and current geometry.
        state_id, if supplied, must match this instance's fixed snapshot. It does
        not load or mutate model state. Instantiate one objective per checkpoint
        and average their relative harms before aggregating different tasks.
        """
        if state_id is not None and state_id != self.state_id:
            raise ValueError(
                'state_id does not match this fixed snapshot; construct a separate '
                'OneStepTrainingHarmObjective for each checkpoint.'
            )
        # BN can alter supplied virtual buffers, never the original buffers.
        # fork_rng restores caller state on failure, too. CPU models do not
        # enumerate devices or perform CUDA operations.
        with torch.enable_grad(), torch.random.fork_rng(devices=self._cuda_devices):
            cpu_rng = torch.get_rng_state()
            cuda_rng = {
                device: torch.cuda.get_rng_state(device) for device in self._cuda_devices
            }
            initial_parameters, initial_buffers = self._new_state()
            with torch.no_grad():
                baseline = self._loss(
                    initial_parameters, initial_buffers, self.validation_batch,
                    training=False, name='pre_update_val_loss',
                )
            # Baseline evaluation must not alter the stochastic training path.
            torch.set_rng_state(cpu_rng)
            for device, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            post, clean, poison, mixed = self._virtual_update(poison_batch, create_graph=True)
            metrics = {
                'pre_update_val_loss': baseline.detach(),
                'post_update_val_loss': post.detach(),
                'clean_train_loss': clean.detach(),
                'poison_train_loss': poison.detach(),
                'mixed_train_loss': mixed.detach(),
                'poison_fraction': self.poison_fraction,
                'inner_lr': self.inner_lr,
            }
            score = post
            if self.clean_reference_batch is not None:
                torch.set_rng_state(cpu_rng)
                for device, state in cuda_rng.items():
                    torch.cuda.set_rng_state(state, device)
                reference, _, _, _ = self._virtual_update(
                    self.clean_reference_batch, create_graph=False,
                )
                reference = reference.detach()
                difference = post - reference
                metrics['reference_post_update_val_loss'] = reference
                metrics['training_harm_delta'] = difference.detach()
                if self.score_mode == 'relative':
                    score = difference
            return AttackObjectiveResult(score=score, metrics=metrics)

    def __call__(self, context: AttackStepContext) -> AttackObjectiveResult:
        """Compatibility wrapper for the legacy online Stage 2 callback."""
        if self.poison_batch_builder is None:
            raise TypeError('The context callback requires poison_batch_builder; use evaluate(batch).')
        # Keep the legacy wrapper's decoder/builder RNG isolation. The nested
        # evaluator preserves its own RNG boundary for direct batch callers.
        with torch.enable_grad(), torch.random.fork_rng(devices=self._cuda_devices):
            decoded_video = context.decode_video()
            poison_batch = self.poison_batch_builder(decoded_video, context)
            return self.evaluate(poison_batch)
