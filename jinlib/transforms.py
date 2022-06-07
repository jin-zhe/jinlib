from PIL import ImageEnhance
import torchvision
import torch

from .pytorch import to_numpy

class Permute:
  '''
  Transform that conducts the given permutation on the tensor
  '''
  def __init__(self, dims):
    self.dims = dims

  def __call__(self, tensor: torch.Tensor):
    return torch.permute(tensor, self.dims)

class ToNPImage:
  '''
  Transform that converts a batch of image tensors to a batch of numpy image ndarrays.
  Currently only for R,G,B channel images
  '''
  def __init__(self):
    self.batch_dims = (0,2,3,1)
    self.image_dims = (1,2,0)

  def __call__(self, tensor: torch.Tensor):
    num_dims = len(tensor.size())
    if num_dims == 4:
      tensor = torch.permute(tensor, self.batch_dims)
    elif num_dims == 3:
      tensor = torch.permute(tensor, self.image_dims)
    else:
      raise ValueError(f'Unknown conversion for tensor with {num_dims} dimensions!')
    return to_numpy(tensor)
    

class Unnormalize(torchvision.transforms.Normalize):
  '''
  Returns unnormalization transform (usually for visual checking).
  Args:
    mean: Mean of size (C,1) used in forward transform
    std: Standard deviation of size (C,1) used in forward transform
  '''
  def __init__(self, mean, std, inplace=False):
    self.unnorm_mean = mean
    self.unnorm_std = std
    mean = mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]]
    std = std=[1/std[0], 1/std[1], 1/std[2]]
    super().__init__(mean, std, inplace=inplace)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(mean={self.unnorm_mean}, std={self.unnorm_std})"

class ImageJitter:
  # Adapted from: https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/1356595bc1551ee1f505e4764416416c7d5d4672/additional_transforms.py#L15
  def __init__(self, transformdict):
    transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
    self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

  def __call__(self, img):
    out = img
    randtensor = torch.rand(len(self.transforms))

    for i, (transformer, alpha) in enumerate(self.transforms):
      r = alpha*(randtensor[i]*2.0 -1.0) + 1
      out = transformer(out).enhance(r).convert('RGB')

    return out
