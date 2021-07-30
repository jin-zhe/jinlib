from PIL import ImageEnhance
import torch

class ImageJitter(object):
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