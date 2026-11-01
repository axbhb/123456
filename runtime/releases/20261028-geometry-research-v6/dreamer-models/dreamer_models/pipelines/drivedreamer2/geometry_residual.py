"""Bounded auxiliary on verified aligned surface patches, separate from quality.

A provider recomputes sampling grids from current world geometry and fixed camera
calibration. It must supply verified road-plane/depth correspondences, not box
corners. Visibility masks are detached; candidate identities stay fixed. Coverage
failure follows a predeclared reject or zero-auxiliary policy, never cherry picks.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass
class SurfacePairs:
    candidate_ids: tuple
    left_indices: torch.Tensor
    right_indices: torch.Tensor
    left_grid: torch.Tensor
    right_grid: torch.Tensor
    visible: torch.Tensor
    evidence: str


@dataclass
class GeometryResidualResult:
    score: torch.Tensor
    feasible: bool
    metrics: dict


class GeometryResidual:
    def __init__(self, *, provider, descriptor, reference_video, reference_geometry,
                 calibration_id, scale, min_coverage, max_residual, missing_policy='reject'):
        if not callable(provider) or not callable(descriptor) or not calibration_id:
            raise ValueError('Verified provider, descriptor and calibration provenance required')
        if not math.isfinite(scale) or scale<=0 or not 0<min_coverage<=1 or not math.isfinite(max_residual) or max_residual<0:
            raise ValueError('Invalid geometry residual calibration')
        if missing_policy not in ('reject','zero_auxiliary'):
            raise ValueError('Unknown coverage failure policy')
        if isinstance(descriptor,torch.nn.Module):
            if descriptor.training or any(p.requires_grad for p in descriptor.parameters()):
                raise ValueError('Descriptor must be frozen and in eval mode')
        self.provider,self.descriptor=provider,descriptor
        self.scale,self.min_coverage,self.max_residual=scale,min_coverage,max_residual
        self.calibration_id,self.missing_policy=calibration_id,missing_policy
        self.candidate_ids=None
        with torch.no_grad():
            self.reference,self.reference_coverage=self._residual(reference_video,reference_geometry)
        self.reference=self.reference.detach()

    def _residual(self,video,geometry):
        if video.ndim!=6 or not torch.isfinite(video.detach()).all():
            raise ValueError('Finite [B,T,V,C,H,W] video required')
        pairs=self.provider(geometry)
        if not isinstance(pairs,SurfacePairs) or pairs.evidence not in ('verified_static_road_plane','verified_surface_depth'):
            raise ValueError('Missing verified surface correspondence evidence')
        ids=tuple(pairs.candidate_ids)
        if not ids or len(set(ids))!=len(ids):raise ValueError('Nonempty unique fixed candidate IDs required')
        if self.candidate_ids is None:self.candidate_ids=ids
        if ids!=self.candidate_ids:raise ValueError('Correspondence candidate pool changed')
        e=len(ids)
        flat=video.reshape(-1,*video.shape[3:])
        patches=[]
        for indices,grid in ((pairs.left_indices,pairs.left_grid),(pairs.right_indices,pairs.right_grid)):
            if indices.shape!=(e,) or indices.dtype!=torch.long or (indices<0).any() or (indices>=len(flat)).any():
                raise ValueError('Invalid fixed video indices')
            if grid.ndim!=4 or grid.shape[0]!=e or grid.shape[-1]!=2 or not torch.isfinite(grid.detach()).all():
                raise ValueError('Invalid surface sampling grid')
            if (grid.detach().abs()>1).any():raise ValueError('Out-of-view sampling must be handled by verified provider')
            patch=F.grid_sample(flat[indices.to(video.device)],grid.to(video),mode='bilinear',padding_mode='zeros',align_corners=False)
            features=self.descriptor(patch)
            if not torch.is_tensor(features) or features.shape[0]!=e or not torch.isfinite(features.detach()).all():raise ValueError('Invalid aligned descriptors')
            patches.append(F.normalize(features.reshape(e,-1),dim=-1,eps=1e-8))
        visible=pairs.visible
        if visible.shape!=(e,) or visible.dtype!=torch.bool or visible.requires_grad:
            raise ValueError('Visibility must be a detached boolean candidate mask')
        mask=visible.to(video)
        coverage=float(mask.mean().detach())
        distance=(patches[0]-patches[1]).square().mean(-1)
        residual=(distance*mask).sum()/mask.sum().clamp_min(1)
        return residual,coverage

    def evaluate(self,video,geometry):
        residual,coverage=self._residual(video,geometry)
        available=min(coverage,self.reference_coverage)>=self.min_coverage
        if available:
            score=((residual-self.reference.to(residual))/self.scale).clamp(-1,1)
            feasible=bool(residual.detach()<=self.max_residual)
        else:
            score=video.sum()*0
            feasible=self.missing_policy=='zero_auxiliary'
        return GeometryResidualResult(score,feasible,{'geometry/available':float(available),
            'geometry/coverage':coverage,'geometry/reference_coverage':self.reference_coverage,
            'geometry/residual':float(residual.detach()),'geometry/score':float(score.detach()),
            'geometry/zero_fallback':float(not available and feasible)})
