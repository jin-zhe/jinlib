from jinlib.visualisation import plot_classwise_accuracies, plot_confusion_matrix
from jinlib.pytorch import set_deterministic
from jinlib import Experiment

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from types import SimpleNamespace
from pathlib import Path
'''
This is a re-adaptation of PyTorch guide https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html using jinlib framework:
'''

class Net(nn.Module):
  def __init__(self, activation):
    super().__init__()
    self.activation = activation
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(self.activation(self.conv1(x)))
    x = self.pool(self.activation(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.fc3(x)
    return x

class CIFAR10Classifier(Experiment):
  def __init__(self, experiment_dir):
    super().__init__(experiment_dir)
    self.transforms = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    self.classes = (
      'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  def _init_model(self):
    self.model = Net(activation=self.activation)

  def _init_dataset(self):
    download_root = './data'
    train_dataset = torchvision.datasets.CIFAR10(
      root=download_root, train=True, download=True, transform=self.transforms)
    validation_dataset = torchvision.datasets.CIFAR10(
      root=download_root, train=False, download=True, transform=self.transforms)
    self.dataset = SimpleNamespace(train=train_dataset, validation=validation_dataset)

  def _init_dataloaders(self):
    self.train_loader = DataLoader(
      self.dataset.train, batch_size=self.batch_size, shuffle=False, num_workers=2)
    self.validation_loader = DataLoader(
      self.dataset.validation, batch_size=self.batch_size, shuffle=False, num_workers=2)
    self.test_loader = self.analyze_loader = self.validation_loader

  def _init_iter_stats(self):
    super()._init_iter_stats()
    if self.context == 'analyze':
      self.curr_iter_stats['true_labels'] = []
      self.curr_iter_stats['pred_labels'] = []

  def _update_iter_stats(self, T_out, batch):
    super()._update_iter_stats(T_out, batch)
    if self.context == 'analyze':
      _, T_predictions = torch.max(T_out, 1)
      self.curr_iter_stats['true_labels'] += list(self.batch_y(batch).cpu().numpy())
      self.curr_iter_stats['pred_labels'] += list(T_predictions.cpu().numpy())

  def analyze(self):
    super().analyze()
    true_labels = self.curr_iter_stats['true_labels']
    pred_labels = self.curr_iter_stats['pred_labels']
    conf_mat = confusion_matrix(true_labels, pred_labels, normalize='true')
    cls_acc = conf_mat.diagonal()
    plot_confusion_matrix(self.classes, conf_mat, self.experiment_dir)
    plot_classwise_accuracies(self.classes, cls_acc, self.experiment_dir)

def main():
  set_deterministic()
  exp1 = CIFAR10Classifier(Path('experiment_1'))
  exp1.train()            # train for 5 epochs (see experiment config file)
  exp1.train(resume=True) # train for another 5 epochs, resuming from best checkpoint
  exp1.analyze()          # analyze based on best checkpoint

  exp2 = CIFAR10Classifier(Path('experiment_2'))
  exp2.train()
  exp2.analyze()

if __name__ == '__main__': main()
