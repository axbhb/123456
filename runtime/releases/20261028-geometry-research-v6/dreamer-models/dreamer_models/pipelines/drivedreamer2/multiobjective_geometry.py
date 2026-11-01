"""Small shared-geometry MGDA solve, using actual per-task u-gradients.

Sener & Koltun (NeurIPS 2018), arXiv:1810.04650, Eq. (3).
This solves the convex-hull problem directly, not the representation-space bound.
Positive task-norm normalization is explicit. With nonzero exact gradients the
nonzero solution is common ascent locally; STE, noisy VJPs, projection and finite
steps still require measured evaluation. Do not take sign() or add momentum to
this direction and claim to preserve that property.
"""
from itertools import combinations
import torch


def minimum_norm_common_ascent(task_gradients, *, normalize=True):
    if not isinstance(task_gradients, torch.Tensor) or task_gradients.ndim != 2:
        raise ValueError('Expected a task-by-geometry gradient matrix')
    if not task_gradients.is_floating_point() or not torch.isfinite(task_gradients).all():
        raise ValueError('Task gradients must be finite floating point')
    n, dimensions = task_gradients.shape
    if not 1 <= n <= 8 or dimensions == 0:
        raise ValueError('Exact face enumeration supports one to eight nonempty task gradients')
    original = task_gradients.detach().double().cpu()
    norms = original.norm(dim=1)
    zero = norms == 0
    if bool(zero.any()):
        weights = zero.double()/zero.sum()
        return dict(direction=torch.zeros_like(task_gradients[0]), weights=weights.tolist(),
                    task_norms=norms.tolist(), zero_tasks=zero.nonzero().flatten().tolist(),
                    directional_derivatives=[0.]*n, stationary=True)
    scaled = original / norms[:, None] if normalize else original
    # A common scale avoids a numerically singular KKT system for tiny VJPs.
    common_scale = scaled.abs().max()
    work = scaled / common_scale
    gram = work @ work.T
    best = None
    for size in range(1, n+1):
        for face in combinations(range(n), size):
            idx = torch.tensor(face)
            q = gram[idx][:, idx]
            system = torch.zeros(size+1, size+1, dtype=torch.float64)
            system[:size,:size] = q
            system[:size,size] = 1
            system[size,:size] = 1
            rhs = torch.zeros(size+1, dtype=torch.float64); rhs[-1] = 1
            solution = torch.linalg.lstsq(system, rhs, driver='gelsd').solution
            if (system @ solution-rhs).abs().max() > 1e-9 or solution[:size].min() < -1e-10:
                continue
            weights = torch.zeros(n, dtype=torch.float64)
            weights[idx] = solution[:size].clamp_min(0)
            weights /= weights.sum()
            value = float(weights @ gram @ weights)
            if best is None or value < best[0]:
                best = value, weights
    if best is None:
        raise RuntimeError('Failed to solve the small gradient convex-hull problem')
    weights = best[1]
    direction = weights @ scaled
    # A zero convex-hull point means no strict common first-order ascent. Do not
    # normalize roundoff into a unit update.
    stationary = float(direction.norm()) <= 1e-10*float(scaled.norm(dim=1).max())
    if stationary:
        direction.zero_()
    return dict(direction=direction.to(task_gradients), weights=weights.tolist(),
                task_norms=norms.tolist(), zero_tasks=[], stationary=stationary,
                directional_derivatives=(original @ direction).tolist())
