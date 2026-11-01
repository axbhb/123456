"""Measured finite-difference directions for low-dimensional native geometry.

The callback returns task scores and existing feasibility, never test-set metrics.
At constrained boundaries use a measured one-sided secant; missing both sides is
reported explicitly. These are finite-step native secants, not exact derivatives.
"""
import math
import torch


def estimate_task_secants(point, current_scores, measure, *, step):
    if point.ndim != 1 or not point.is_floating_point() or not torch.isfinite(point).all():
        raise ValueError('Expected finite vector geometry')
    if not math.isfinite(step) or step <= 0:
        raise ValueError('Finite-difference step must be positive')
    values = torch.as_tensor(current_scores, dtype=torch.float64)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError('Expected finite task scores')
    gradient = torch.zeros(values.numel(), point.numel(), dtype=torch.float64)
    records = []
    for axis in range(point.numel()):
        samples = {}
        for sign in (1, -1):
            trial = point.detach().clone()
            trial[axis] = (trial[axis]+sign*step).clamp(-1,1)
            displacement = float(trial[axis]-point[axis])
            if displacement == 0:
                continue
            scores, feasible = measure(trial)
            scores = torch.as_tensor(scores, dtype=torch.float64)
            if scores.shape != values.shape or not torch.isfinite(scores).all():
                raise ValueError('Invalid measured task scores')
            samples[sign] = (scores, displacement, feasible)
        plus = samples.get(1); minus = samples.get(-1)
        plus = plus if plus is not None and plus[2] else None
        minus = minus if minus is not None and minus[2] else None
        if plus is not None and minus is not None:
            gradient[:,axis] = (plus[0]-minus[0])/(plus[1]-minus[1]); method='central'
        elif plus is not None or minus is not None:
            side = plus if plus is not None else minus
            gradient[:,axis] = (side[0]-values)/side[1]; method='one_sided'
        else:
            method='unmeasured_feasible_direction'
        records.append(dict(axis=axis,method=method,samples={str(k):dict(scores=v[0].tolist(),displacement=v[1],feasible=v[2]) for k,v in samples.items()}))
    return gradient, records
