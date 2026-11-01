"""Restore native generator pixels to the original calibrated camera canvas."""
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
@PIPELINES.register_module()
class RestoreGeneratorCanvas:
 def __init__(self,source_to_generated,source_size=(1600,900)):
  self.matrix=np.asarray(source_to_generated,dtype=np.float64);self.size=tuple(source_size)
  assert self.matrix.shape==(3,3) and np.isfinite(self.matrix).all()
 def __call__(self,results):
  inverse=np.linalg.inv(self.matrix)[:2]
  results['img']=[cv2.warpAffine(x,inverse,self.size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for x in results['img']]
  shape=results['img'][0].shape
  results['ori_shape']=shape;results['img_shape']=shape;results['pad_shape']=shape
  return results
