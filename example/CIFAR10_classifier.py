#TODO
import sys
sys.path.append('..')

from jinlib import Experiment

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms

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
    train_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=self.transforms)
    validation_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=self.transforms)
    self.dataset = SimpleNamespace(train=train_dataset, validation=validation_dataset)

  def _init_dataloaders(self):
    self.train_loader = DataLoader(
      self.dataset.train, batch_size=self.batch_size, shuffle=False, num_workers=2)
    self.validation_loader = DataLoader(
      self.dataset.validation, batch_size=self.batch_size, shuffle=False, num_workers=2)

  def _validation_epoch_end(self):
    pass

  def _test_epoch_end(self):
    pass

  def _analyze_epoch_end(self):
    pass

  def test(self):
    pass

  def analyze(self):
    pass

def main():
  exp1 = CIFAR10Classifier(Path('experiment_1'))
  exp1.train()

if __name__ == '__main__': main()
