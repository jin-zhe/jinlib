from argparse import ArgumentError
from pathlib import Path
import logging
import copy
import os

from sklearn.model_selection import train_test_split
import torchvision

from torch.utils.data import DataLoader
import torch

from .Subset import Subset

def choose_device():
  '''
  Automatically choose the best device
  '''
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def freeze(net):
  '''
  Freeze the parameters in the given network
  '''
  parameters = net.parameters()
  for param in parameters:
    param.requires_grad = False

def get_nn(name: str, kwargs: dict):
  '''
  Return torch.nn attribute
  '''
  if hasattr(torch.nn, name):  # if attribute exists in torch.nn
    attribute = getattr(torch.nn, name)
    return attribute(**kwargs)
  else:
    raise AttributeError('Attribute \'{}\' does not exist in torch.nn!'.format(name))

def get_activation(choice: str, kwargs: dict):
  '''
  Return activation of choice
  Semantic wrapper for `get_nn`
  '''
  try:
    return get_nn(choice, kwargs)
  except AttributeError:
    raise ValueError('Unknown activation choice \'{}\'!'.format(choice))

def get_layer(choice: str, kwargs: dict):
  '''
  Return layer of choice
  Semantic wrapper for `get_nn`
  '''
  try:
    return get_nn(choice, kwargs)
  except AttributeError:
    raise ValueError('Unknown layer choice \'{}\'!'.format(choice))

def get_loss_fn(choice: str, kwargs: dict):
  '''
  Return the loss function of choice
  Semantic wrapper for `get_nn`
  '''
  try:
    return get_nn(choice, kwargs)
  except AttributeError:
    raise ValueError('Unknown loss function choice \'{}\'!'.format(choice))

def get_optimizer(choice: str, model: torch.nn.Module, **kwargs):
  '''
  Return the optimizer of choice
  '''
  if hasattr(torch.optim, choice):
    return getattr(torch.optim, choice)(model.parameters(), **kwargs)
  else:
    raise ValueError('Unknown optimization choice!')

def get_subset_loader(dataset, sample_indices, batch_size, shuffle, transform=None, drop_last=False, num_workers=8):
  '''
  Return the dataloader for a subset of the given dataset according to sample indices provided
  '''
  data = Subset(dataset, sample_indices, transform)
  return DataLoader(data,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )

def rand_select_indices(n, index_length):
  '''
  Randomly select n unique indices from 0..index_length-1
  '''
  T_index_permutation = torch.randperm(index_length)
  return T_index_permutation[:n]

def rand_select_items(n, tensor_1d):
  '''
  Randomly select n items without replacement from a 1-d tensor
  '''
  n_indices = rand_select_indices(n, len(tensor_1d))
  return tensor_1d[n_indices]

def split_indices(labels, ratio, random_state=42):
  '''
  Split dataset indices into 2 parts with stratification
  '''
  sample_indices = range(len(labels))
  proportion_2 = ratio[1]/(ratio[0]+ratio[1])
  indices_1, indices_2, labels_1, labels_2 = train_test_split(
    sample_indices, labels, test_size=proportion_2, stratify=labels,
    random_state=random_state)
  return indices_1, indices_2, labels_1, labels_2

def batch_as_figure(T_samples, title):
  '''
  Transforms training batch tensor to figure for plotting.
  To display in matplotlib:
    plt.imshow(figure)
    plt.title(title, wrap=True)
    plt.show()
  '''
  T_samples = (T_samples.cpu() * 255).int()
  T_grid = torchvision.utils.make_grid(T_samples)
  figure = T_grid.permute(1, 2, 0).int().numpy()
  return figure

def set_rand_seed(seed=42):
  torch.manual_seed(seed)

def to_device(device, *tensors):
  '''
  Copies tensors to device and return
  '''
  out = [t.to(device) for t in tensors]
  return out[0] if len(out) == 1 else out

def copy_params(model):
  '''
  Performs deep copy of model parameters and return
  '''
  return copy.deepcopy(model.state_dict())

def one_hot(T_labels, num_classes):
  '''
  Converts a tensor of labels into one-hot vectors.
  Args:
    labels: (IntTensor) class labels, sized [N,].
    num_classes: (int) number of classes.
  Returns:
    (tensor) one-hot vectors, sized [N, num_classess].
  '''
  T_identity = torch.eye(num_classes) 
  return T_identity[T_labels]

def save_checkpoint(file_path: Path, checkpoint: dict):
  '''
  Saves model and training parameters at given file_path.
  Args:
    file_path: (Path) file path where checkpoint is to be saved
    checkpoint: (dict) contains model's state_dict
  '''
  dir_path = file_path.parent
  if not dir_path.exists():
    print("Directory {} does not exist! Making directory".format(str(dir_path.resolve())))
    os.mkdir(str(dir_path.resolve()))
  torch.save(checkpoint, file_path)

def load_model_state(model: torch.nn.Module, state_dict: dict, state_key: str = 'model_state'):
  '''
  Loads model with given state_dict.
  Args:
    model: (torch.nn.Module) model for which the parameters are loaded
    state_dict: (dict) loaded state_dict file
    state_key: (string) key in state_dict for value corresponding to model state dict
  '''
  model.load_state_dict(state_dict[state_key])
  model.to(choose_device())

def load_optimizer_state(optimizer: torch.optim.Optimizer, state_dict: dict, state_key='optim_state'):
  '''
  Loads optimizer with given state_dict.
  Args:
    optimizer: (torch.optim.Optimizer) the optimizer object whose state is to be loaded
    state_dict: (dict) loaded state_dict file
    state_key: (string) key in state_dict dict for value corresponding to optimization state dict
  '''
  optimizer.load_state_dict(state_dict[state_key])

  # See this issue: https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
  converted = False
  for state in optimizer.state.values():
    for k, v in state.items():
      if torch.is_tensor(v):
        src_device = v.device
        dst_device = choose_device()
        if src_device != dst_device:
          state[k] = v.to(dst_device)
          converted = True

  if converted:
    logging.info('Converted loaded optimizer state_dict tensors from {} to {}.'.format(src_device, dst_device))

def load_checkpoint(ckpt_path: Path, model: torch.nn.Module, load_optimizer: bool = True, optimizer: torch.optim.Optimizer = None):
  '''
  Loads state_dict from ckpt_path. If also indicated to load optimizer (default)
  and optimization object is provided, loads state_dict of optimizer assuming it
  is in the checkpoint.
  Args:
    ckpt_path: (string) file path of checkpoint to be loaded
    model: (torch.nn.Module) model for which the parameters are loaded
    load_optimizer: (bool) flag for loading optimizer (default True)
    optimizer: (torch.optim.Optimizer) optimizer whose state is to be loaded if
               `load_optimizer` is set to True
  '''  
  if not ckpt_path.is_file():
    raise FileNotFoundError("Checkpoint doesn't exist! {}".format(str(ckpt_path.resolve())))

  state_dict = torch.load(str(ckpt_path.resolve()))
  load_model_state(model, state_dict)
  if load_optimizer:
    if optimizer is None:
      raise ArgumentError('load_optimizer flag is {} but optimizer is {}!'.format(load_optimizer, type(optimizer).__name__))
    load_optimizer_state(optimizer, state_dict)
  else:
    logging.info('Not loading optimizer.')
  return state_dict

def denormalize(T_image, mean, std):
  """
  Undo normalization and return denormalized image (usually for visual checking).
  Args:
    tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
    mean: Mean of size (C,1) used in forward transform
    std: Standard deviation of size (C,1) used in forward transform
  """
  return torchvision.transforms.Normalize(
    mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
    std=[1/std[0], 1/std[1], 1/std[2]]
  )(T_image)
