from argparse import ArgumentError
from pathlib import Path
import logging
import shutil
import copy
import os

from sklearn.model_selection import train_test_split
import numpy as np
import torchvision

from torch.utils.data import DataLoader
import torch

from .RunningAverage import RunningAverage
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

def get_optimizer(choice: str, kwargs: dict, model: torch.nn.Module):
  '''
  Return the optimizer of choice
  '''
  if choice == 'Adam':
    return torch.optim.Adam(model.parameters(), **kwargs)
  if choice == 'SGD':
    return torch.optim.SGD(model.parameters(), **kwargs)
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
  return [t.to(device) for t in tensors]

def copy_params(model):
  '''
  Performs deep copy of model parameters and return
  '''
  return copy.deepcopy(model.state_dict())

def evaluate(model, data_loader, get_loss=False, get_acc=False,
  get_conf_mat=False, loss_fn=None, num_labels=-1):
  '''
  Returns the model's loss and accuracy on data from data loader
  '''
  # Input validation
  if get_loss:
    running_loss = RunningAverage()
    if not loss_fn:
      raise ValueError('loss_fn must be specified!')
  if get_acc:
    running_acc = RunningAverage()
  if get_conf_mat:
    if num_labels == -1:
      raise ValueError('num_labels must be specified!')
    confusion_matrix = torch.zeros(num_labels, num_labels, dtype=torch.float64)

  
  # Evaluate model
  model.eval()
  with torch.no_grad():
    for T_samples, T_labels in data_loader:
      T_samples, T_labels = to_device(choose_device(), T_samples, T_labels)
      batch_size = T_samples.size(0)
      T_outputs = model(T_samples)
      _, T_predictions = torch.max(T_outputs, 1)
      if get_loss:
        loss = loss_fn(T_outputs, T_labels)
        loss_stat = loss.item() * batch_size
        running_loss.update(loss_stat, batch_size)
      if get_acc:
        acc_stat = torch.sum(T_predictions == T_labels).item()
        running_acc.update(acc_stat, batch_size)
      if get_conf_mat:
        for true_label, pred_label in zip(T_labels.view(-1), T_predictions.view(-1)):
          confusion_matrix[true_label.long()][pred_label.long()] += 1
        confusion_matrix = confusion_matrix / confusion_matrix.sum(0) # normalize
  # Returns
  return_vals = []
  if get_loss:
    return_vals.append(running_loss())
  if get_acc:
    return_vals.append(running_acc())
  if get_conf_mat:
    return_vals.append(confusion_matrix)
  
  return return_vals

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

def update_writer(writer, train_loss, train_acc, val_loss, val_acc, epoch):
  writer.add_scalar('Loss/train', train_loss, epoch)
  writer.add_scalar('Loss/val', val_loss, epoch)
  if train_acc:
    writer.add_scalar('Accuracy/train', train_acc, epoch)
  if val_acc:
    writer.add_scalar('Accuracy/val', val_acc, epoch)

def save_checkpoint(dir_path: Path, state, is_best, last_filename='last.pth.tar', best_filename='best.pth.tar'):
  '''
  Adapted from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
  Saves model and training parameters at dir_path + last_filename.
  If is_best==True, also saves dir_path + 'best.pth.tar'
  Args:
    dir_path: (Path) directory path where checkpoint is to be saved
    state: (dict) contains model's state_dict
    is_best: (bool) True if it is the best model seen till now
    last_filename: (string) filename for the last checkpoint
    best_filename: (string) filename for the best checkpoint
  '''
  file_path = dir_path / last_filename
  if not dir_path.exists():
    print("Directory {} does not exist! Making directory".format(str(dir_path.resolve())))
    os.mkdir(str(dir_path.resolve()))
  torch.save(state, file_path)
  if is_best:
    best_file_path = dir_path / best_filename
    shutil.copyfile(str(file_path.resolve()), str(best_file_path.resolve()))

def load_model(model: torch.nn.Module, state_dict: dict, state_key: str = 'model_state'):
  '''
  Loads model with given state_dict.
  Args:
    model: (torch.nn.Module) model for which the parameters are loaded
    state_dict: (dict) loaded state_dict file
    state_key: (string) key in state_dict for value corresponding to model state dict
  '''
  model.load_state_dict(state_dict[state_key])
  model.to(choose_device())

def load_optimizer(optimizer: torch.optim.Optimizer, state_dict: dict, state_key='optim_state'):
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

def load_checkpoint(ckpt_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
  '''
  Loads state_dict from file_path. If optimization is provided, loads state_dict
  of optimization assuming it is present in checkpoint.
  Args:
    ckpt_path: (string) file path of checkpoint to be loaded
    model: (torch.nn.Module) model for which the parameters are loaded
    optimizer: (torch.optim.Optimizer) optimizer whose state is also to be loaded
  '''  
  if not ckpt_path.is_file():
    raise FileNotFoundError("Checkpoint doesn't exist! {}".format(str(ckpt_path.resolve())))

  state_dict = torch.load(str(ckpt_path.resolve()))
  logging.info('Resuming epoch {} from checkpoint {}'.format(state_dict['epoch'], str(ckpt_path.resolve())))
  load_model(model, state_dict)
  if optimizer:
    load_optimizer(optimizer, state_dict)
  return state_dict

def load_best_checkpoint(exp_dir: Path, model: torch.nn.Module, optim_choice: str = None, optim_kwargs: dict = None,
  best_filename='best.pth.tar'):
  '''
  Resumes model with the last checkpoint in the experiment directory.
  Args:
    exp_dir: (string) directory path for experiment
    model: (torch.nn.Module) model for which the parameters are loaded
    optim_choice: (str) the name of optimization to be used
    optim_kwargs: (dict) the key word arguments for initializing optimzer
    best_filename: (string) filename for the best checkpoint within `exp_dir`
  '''
  try:
    best_ckpt_path = exp_dir / best_filename
    return load_checkpoint(best_ckpt_path, model, optim_choice, optim_kwargs)
  except FileNotFoundError:
    print('No best checkpoints to resume!')
    return None

def resume_last_checkpoint(exp_dir: Path, model: torch.nn.Module, optim_choice: str = None, optim_kwargs: dict = None,
  last_filename='last.pth.tar'):
  '''
  Resumes model with the last checkpoint in the experiment directory.
  Args:
    exp_dir: (string) directory path for experiment
    model: (torch.nn.Module) model for which the parameters are loaded
    optim_choice: (str) the name of optimization to be used
    optim_kwargs: (dict) the key word arguments for initializing optimzer
    last_filename: (string) filename for the last checkpoint within `exp_dir`
  '''
  try:
    last_ckpt_path = exp_dir / last_filename
    return load_checkpoint(last_ckpt_path, model, optim_choice, optim_kwargs)
  except FileNotFoundError:
    print('No prior checkpoints to resume!')
    return None

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
