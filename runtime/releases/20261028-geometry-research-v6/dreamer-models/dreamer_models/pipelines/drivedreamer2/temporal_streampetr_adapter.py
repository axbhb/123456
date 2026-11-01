"""Native forward with declared image-resampling STE and current label VJP."""
import copy
import numpy as np
import torch
from torch.nn import functional as F

class TemporalStreamPETRAdapter:
    def __init__(self,export,renderer):
        from mmdet3d.datasets import build_dataset
        from mmcv.parallel import collate,scatter
        self.export=export;self.renderer=renderer;self.dataset=build_dataset(export['config'].data.train)
        self.batches=[scatter(collate([self.dataset[i]],samples_per_gpu=1),[-1])[0] for i in range(len(self.dataset))]
        if [x['token'] for x in self.dataset.data_infos]!=[x['token'] for x in export['infos']]:raise ValueError('Native dataset reordered source keyframes')
        self.camera_indices=[tuple(renderer.bundle['camera_order'].index(k) for k in info['cams']) for info in export['infos']]
        self.mean=torch.tensor([123.675,116.28,103.53])[None,:,None,None]
        self.std=torch.tensor([58.395,57.12,57.375])[None,:,None,None]

    def batch(self,video,geometry,index):
        if not torch.equal(video.detach(),self.export['video_point']) or not torch.equal(geometry['normalized_u'].detach(),self.export['u_point']):raise ValueError('Video/geometry differs from the native exported forward point; re-export before evaluating')
        hard=self.batches[index];result=dict(hard);slot=self.export['slots'][index]
        rgb=video[0,slot,list(self.camera_indices[index])]
        soft=F.interpolate(rgb.float()*255,size=(402,704),mode='bicubic',align_corners=False,antialias=True)[:,:,146:402,:]
        soft=(soft-self.mean.to(soft))/self.std.to(soft)
        soft=soft[None,None]
        if soft.shape!=hard['img'].shape:raise ValueError('Native image preprocessing shape changed')
        result['img']=hard['img'].to(soft)+(soft-soft.detach())
        native=hard['gt_bboxes_3d'][0][0]
        full=self.export['labels'](geometry,index,bottom_center=True).to(native.tensor)
        # Native filtering is discrete; verify the exact retained identities by
        # their source position/size, and reject ambiguity or mismatched targets.
        differences=(native.tensor[:,None,:6]-full.detach()[None,:,:6]).abs().amax(-1)
        matches=differences<3e-4
        if not bool((matches.sum(-1)==1).all()):raise ValueError('Native filtered label mapping is ambiguous')
        indices=matches.to(torch.int64).argmax(-1);values=full[indices]
        yaw_delta=torch.atan2(torch.sin(values[:,6]-native.tensor[:,6]),torch.cos(values[:,6]-native.tensor[:,6]))
        if yaw_delta.detach().abs().max()>3e-4 or not torch.allclose(values[:,7:].detach(),native.tensor[:,7:],atol=3e-4,rtol=1e-5):raise ValueError('Native yaw/velocity preprocessing mismatch')
        values=native.tensor+(values-values.detach())
        result['gt_bboxes_3d']=[[type(native)(values,box_dim=values.shape[-1],with_yaw=True,origin=(.5,.5,0))]]
        return result
