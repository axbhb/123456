"""Paired differentiable K-step SGD/momentum/AdamW trajectories.

The supplied optimizer is snapshotted, never stepped. Uses its selected parameter
 groups and initial state; schedules/AMP/clipping are NOT implicitly reproduced.
Unselected parameters stay fixed. Repeated evaluate(batch) reuses that batch at
each step; evaluate_sequence is required for a real sampler's changing slots.
Model-specific Python caches must be isolated by the native loss adapter.
"""
from contextlib import contextmanager
import copy
import math
import random
import torch
from .training_harm_objective import OneStepTrainingHarmObjective
from .attack_optimization import AttackObjectiveResult


@contextmanager
def _paired_rng(devices):
    py_state = random.getstate()
    try:
        import numpy as np
        np_state = np.random.get_state()
    except ImportError:
        np = None
    with torch.random.fork_rng(devices=devices):
        cpu = torch.get_rng_state()
        cuda = {d: torch.cuda.get_rng_state(d) for d in devices}
        def reset():
            random.setstate(py_state)
            if np is not None:
                np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            for d, state in cuda.items():
                torch.cuda.set_rng_state(state, d)
        try:
            yield reset
        finally:
            reset()


class ShortTrajectoryTrainingHarmObjective(OneStepTrainingHarmObjective):
    def __init__(self, *args, steps, optimizer=None, clean_train_batches=None,
                 clean_reference_batches=None, step_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        if self.score_mode != 'relative':
            raise ValueError('Short trajectories require a paired relative score')
        self.steps = steps
        self.clean_batches = tuple(clean_train_batches) if clean_train_batches is not None else (self.clean_train_batch,) * steps
        self.reference_batches = tuple(clean_reference_batches) if clean_reference_batches is not None else (self.clean_reference_batch,) * steps
        if len(self.clean_batches) != steps or len(self.reference_batches) != steps or any(x is None for x in self.reference_batches):
            raise ValueError('Provide exactly K paired normal/reference batches')
        weights = tuple(step_weights) if step_weights is not None else (0.,) * (steps-1) + (1.,)
        if len(weights) != steps or any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.):
            raise ValueError('Step weights must be nonnegative and sum to one')
        self.step_weights = tuple(float(w) for w in weights)
        self.optimizer_kind = 'sgd'
        self._groups = {}
        self._optimizer_state = {}
        if optimizer is None:
            for name in self.parameter_names:
                self._groups[name] = dict(lr=self.inner_lr, momentum=0., dampening=0., weight_decay=0., nesterov=False)
        else:
            if type(optimizer) not in (torch.optim.SGD, torch.optim.AdamW):
                raise TypeError('Only native SGD and AdamW optimizer semantics are supported')
            self.optimizer_kind = 'adamw' if type(optimizer) is torch.optim.AdamW else 'sgd'
            names = {id(p): n for n, p in self.model.named_parameters()}
            selected = set(self.parameter_names)
            for group in optimizer.param_groups:
                if group.get('maximize', False) or group.get('amsgrad', False):
                    raise ValueError('maximize/AMSGrad are not supported')
                options = {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
                if not isinstance(options['lr'], (int, float)) or not math.isfinite(options['lr']) or options['lr'] <= 0:
                    raise ValueError('Require positive fixed scalar group learning rates')
                for p in group['params']:
                    name = names.get(id(p))
                    if name not in selected:
                        continue
                    if name in self._groups:
                        raise ValueError('Duplicate selected optimizer parameter')
                    self._groups[name] = options
                    state = optimizer.state.get(p, {})
                    self._optimizer_state[name] = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in state.items()}
            if set(self._groups) != selected:
                raise ValueError('Optimizer must contain every selected model parameter')
        for name, state in self._optimizer_state.items():
            expected = {'step','exp_avg','exp_avg_sq'} if self.optimizer_kind == 'adamw' else {'momentum_buffer'}
            if set(state) - expected or (state and self.optimizer_kind == 'adamw' and set(state) != expected):
                raise ValueError('Unsupported/incomplete initial optimizer state for ' + name)
            for key, value in state.items():
                if key == 'step':
                    scalar = float(value)
                    if scalar < 0 or not math.isfinite(scalar) or scalar != int(scalar):
                        raise ValueError('Invalid optimizer step')
                elif value.shape != self._initial_parameters[name].shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid optimizer tensor for ' + name)
                elif key == 'exp_avg_sq' and (value < 0).any():
                    raise ValueError('Negative AdamW variance for ' + name)

    def _step(self, name, p, g, state):
        options = self._groups[name]
        lr, decay = options['lr'], options.get('weight_decay', 0.)
        if self.optimizer_kind == 'sgd':
            g = g + decay*p if decay else g
            momentum = options.get('momentum', 0.)
            if momentum:
                old = state.get('momentum_buffer')
                buf = g if old is None else momentum*old + (1-options.get('dampening', 0.))*g
                state['momentum_buffer'] = buf
                g = g + momentum*buf if options.get('nesterov', False) else buf
            return p-lr*g
        beta1, beta2 = options['betas']
        step = int(state.get('step', 0)) + 1
        m = beta1*state.get('exp_avg', torch.zeros_like(p)) + (1-beta1)*g
        v = beta2*state.get('exp_avg_sq', torch.zeros_like(p)) + (1-beta2)*g.square()
        state.update(step=step, exp_avg=m, exp_avg_sq=v)
        # Exact forward sqrt; choose the zero subgradient at zero variance.
        variance = v/(1-beta2**step)
        positive = variance > 0
        root = torch.where(positive, torch.where(positive, variance, torch.ones_like(variance)).sqrt(), torch.zeros_like(variance))
        denom = root + options['eps']
        return p*(1-lr*decay) - lr*(m/(1-beta1**step))/denom

    def _trajectory(self, batches, *, create_graph):
        params, buffers = self._new_state()
        state = {n: {k: v.clone() if torch.is_tensor(v) else v for k,v in s.items()} for n,s in self._optimizer_state.items()}
        values = []
        for index, synthetic in enumerate(batches):
            normal = self._loss(params, buffers, self.clean_batches[index], training=self.training_mode, name='normal_loss')
            replacement = self._loss(params, buffers, synthetic, training=self.training_mode, name='slot_loss')
            mixed = (1-self.poison_fraction)*normal + self.poison_fraction*replacement
            selected = tuple(params[n] for n in self.parameter_names)
            gradients = torch.autograd.grad(mixed, selected, create_graph=create_graph, allow_unused=True)
            if all(g is None for g in gradients):
                raise ValueError('No selected parameter participates in virtual training')
            for name, p, g in zip(self.parameter_names, selected, gradients):
                if g is not None:
                    if not torch.isfinite(g.detach()).all():
                        raise ValueError('Nonfinite virtual gradient: ' + name)
                    params[name] = self._step(name, p, g, state.setdefault(name, {}))
            if not create_graph:
                params = {n: p.detach().requires_grad_(n in self.parameter_names) for n,p in params.items()}
                state = {n: {k: v.detach() if torch.is_tensor(v) else v for k,v in s.items()} for n,s in state.items()}
            # Validation may mutate virtual buffers. It must not change the next
            # training step's state or RNG progression.
            with _paired_rng(self._cuda_devices):
                vb = {n: b.clone() for n,b in buffers.items()}
                with torch.set_grad_enabled(create_graph):
                    values.append(self._loss(params, vb, self.validation_batch, training=False, name='dev_loss'))
        return values

    def evaluate_sequence(self, poison_batches, *, state_id=None):
        batches = tuple(poison_batches)
        if len(batches) != self.steps:
            raise ValueError('Provide exactly K replacement batches')
        if state_id is not None and state_id != self.state_id:
            raise ValueError('state_id differs from the immutable snapshot')
        with torch.enable_grad(), _paired_rng(self._cuda_devices) as reset:
            post = self._trajectory(batches, create_graph=True)
            reset()
            reference = self._trajectory(self.reference_batches, create_graph=False)
            differences = [p-r.detach() for p,r in zip(post, reference)]
            score = sum(w*d for w,d in zip(self.step_weights, differences))
            metrics = {'training_harm_delta': score.detach(), 'virtual_steps': self.steps,
                       'poison_fraction': self.poison_fraction,
                       'optimizer_adamw': float(self.optimizer_kind == 'adamw')}
            for i,(p,r) in enumerate(zip(post,reference),1):
                metrics['trajectory/%d/post' % i] = p.detach()
                metrics['trajectory/%d/reference' % i] = r.detach()
            return AttackObjectiveResult(score, metrics)

    def evaluate(self, poison_batch, *, state_id=None):
        return self.evaluate_sequence((poison_batch,)*self.steps, state_id=state_id)
