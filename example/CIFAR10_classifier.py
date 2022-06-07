from jinlib.pytorch import set_deterministic, to_numpy
from jinlib.transforms import Unnormalize, ToNPImage
from jinlib import Experiment

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
import wandb

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
    self.dataset_root_dir = './data' # Update as needed
    self.transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize(self.config.transforms.mean, self.config.transforms.std)]
    )
    self.untransform = transforms.Compose(
      [Unnormalize(self.config.transforms.mean, self.config.transforms.std),
      ToNPImage()]
    )
    self.class_names = (
      'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  def _init_model(self):
    self.model = Net(activation=self.activation)

  def _init_dataset(self):
    train_dataset = torchvision.datasets.CIFAR10(
      root=self.dataset_root_dir, train=True, download=True, transform=self.transform)
    validation_dataset = torchvision.datasets.CIFAR10(
      root=self.dataset_root_dir, train=False, download=True, transform=self.transform)
    self.dataset = SimpleNamespace(train=train_dataset, validation=validation_dataset)

  def _init_dataloaders(self):
    self.train_loader = DataLoader(
      self.dataset.train, batch_size=self.config.batch_size, shuffle=False, num_workers=2)
    self.validation_loader = DataLoader(
      self.dataset.validation, batch_size=self.config.batch_size, shuffle=False, num_workers=2)
    self.test_loader = self.analyze_loader = self.validation_loader

  def _validation_epoch_begin(self):
    super()._validation_epoch_begin()
    self._validation_data = []

  def _validation_epoch_end(self):
    wandb.log({'validation/predictions': self._validation_data}, step=self.curr_epoch_stats['epoch'])
    super()._validation_epoch_end()

  def _analyze_epoch_begin(self):
    super()._analyze_epoch_begin()
    self._analyze_data = dict(images=[], y_true=[], preds=[])

  def _analyze_epoch_end(self):
    # Plot confusion matrix
    cm = wandb.plot.confusion_matrix(y_true=self._analyze_data['y_true'], preds=self._analyze_data['preds'],
      class_names=self.class_names, title='Confusion Matrix')
    wandb.log({"analyze/confusion-matrix": cm})

    # Chart predictions table
    # For more ideas, see https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA
    headers = ["Image", "Groundtruth", "Prediction"]
    groundtruths = [self.class_names[gt] for gt in self._analyze_data['y_true']]
    predictions = [self.class_names[pred] for pred in self._analyze_data['preds']]
    rows = list(zip(self._analyze_data['images'], groundtruths, predictions))
    analysis_table = wandb.Table(data=rows, columns=headers)
    wandb.log({'analyze/predictions': analysis_table})

    super()._analyze_epoch_end()

  def _update_batch_stats(self, batch_in, batch_out):
    super()._update_batch_stats(batch_in, batch_out)
    batch_x = self.batch_x(batch_in)
    batch_y = self.batch_y(batch_in)

    if self.is_validation_context() or self.is_analyze_context(): # shared variables for both contexts
      _, batch_predictions = torch.max(batch_out, 1)
      batch_images = self.untransform(batch_x)
      batch_y, batch_predictions = to_numpy(batch_y, batch_predictions)
      groundtruths = [self.class_names[gt] for gt in batch_y]
      predictions = [self.class_names[pred] for pred in batch_predictions]

    if self.is_validation_context():
      strfmt = 'Groundtruth: {}\nPrediction:  {}'
      self._validation_data += [wandb.Image(im, caption=strfmt.format(groundtruths[i],predictions[i])) for i, im in enumerate(batch_images)]

    if self.is_analyze_context():
      self._analyze_data['y_true'] += batch_y.tolist()
      self._analyze_data['preds'] += batch_predictions.tolist()
      self._analyze_data['images'] += [wandb.Image(im) for im in batch_images]

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
