# jinlib

Library package for my line of PyTorch work. Includes convenience functions that are more semantic-oriented and improves readability. Also sports a lightweight framework that abstracts away standard routines and allows the user to quickly implement and compare various experiments.

To give some sense of just how handy the framework brought by the [`Experiment` class](jinlib/Experiment.py) and library is for whipping up quick experiments, you may wish to compare between [our code](example/CIFAR10_classifier.py) and [PyTorch's guide](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for the same CIFAR10 classifier experiment.

## Overview
Every individual experiment is identified by a directory. Experiment settings are fully described by the `config.yml` within their respective directories. The [`example.config.yml`](example.config.yml) provides all the configurations supported out-of-the-box:
```yml
---
evaluation_metrics:         # The metrics which will be calculated for every epoch
  - loss                    # Can be omitted as loss is a compulsory metric
  - accuracy*               # Asterisk indicates to set as criterion metric for selecting best epoch
activation:                 # Activation function
  choice: ReLU              # Same name as function in in torch.nn
  kwargs: {}                # Parameters for activation function call. {} to indicate PyTorch defaults
optimization:               # Optimization function
  choice: SGD               # Same name as function in torch.optim
  kwargs:                   # Parameters for activation function call
    lr: 0.001
    momentum: 0.9
loss:                       # Loss function
  choice: CrossEntropyLoss  # Same name as function in in torch.nn
  kwargs: {}                # Parameters for activation function call. {} to indicate PyTorch defaults
regularization:             # Regularization terms (currently only supports L2)
  L2: 0.001                 # Lagrange multiplier (i.e. lambda) value
batch_size: 4               # Mini-batch size for dataloaders
num_epochs: 5               # Number of training epochs
checkpoints:                # Checkpoint related configuration [OPTIONAL]
  dir:                      # Directory under which checkpoints are saved [DEFAULT]. None indicates experiment directory
  identifier: pth           # Identifier name of each checkpoint [DEFAULT]
  best_prefix: best         # Prefix of the best checkpoint [DEFAULT]. E.g. best.pth.tar
  last_prefix: last         # Prefix of the last checkpoint [DEFAULT]. E.g. last.pth.tar
  extension: tar            # File extension for checkpoint files [DEFAULT]
  stats_filename: stats.yml # Filename for reviewing best and last checkpoint statistics [DEFAULT]. It will be saved in the same directory as checkpoints
logs:                       # Logging related configuration [OPTIONAL]
  logger: log.log           # Filename which logger will log to [DEFAULT]. It sits in the same directory as the experiment
  tensorboard: TB_logdir    # The logdir for Tensorboard [DEFAULT]. It sits in the same directory as the experiment
remarks: This is a great experiment!

```
To read in this experiment configuration file, you'll need to first create a class that subclasses [`Experiment`](jinlib/Experiment.py). Your subclass *must* minimally override and implement the methods `_init_model`, `_init_dataset` and `_init_dataloaders`. Please refer to these methods to see what class attributes have to be defined within these methods. Every `Experiment` instance exposes the methods `.train()`, `.validation()` and `.test()` to correspond to the respective contexts of running the model. In addition, `analyze()` is meant for providing analytic outputs of a model's performance after training. Please refer to the [CIFAR10 classifier](example/CIFAR10_classifier.py) as a simple example. The directory structure of [`example`](example) also reflects the intention of how an experiment is organized for different settings.

 You may also override any of the methods in [`Experiment`](jinlib/Experiment.py) as you see fit. For instance, if you are using a custom loss function, you may simply override `_init_loss_fn` in your subclass. In the [CIFAR10 classifier](example/CIFAR10_classifier.py), we override `_update_iter_stats` to additionally keep track of a confusion matrix across the iterations in analyze context. There's also nothing stopping you from adding more custom configurations and overriding relevant methods to read from them.

Currently, only the best and last checkpoints will be saved for every experiment as this is sufficient for my line of work. Support for saving checkpoints every k iterations/epochs might be added in the future but is not a priority for now.

## Install instructions

### Dependencies
Too many to list manually for now, please trial and error and install neccessary dependencies yourself.

### Via pip
```sh
pip install git+https://github.com/jin-zhe/jinlib
```
### Via local repo
Clone this repo locally:
```sh
git clone git@github.com:jin-zhe/jinlib.git <dest>
```
Go to the directory you cloned the repo in:
```sh
cd <dest>
```
Install via pip:
```sh
pip install -e .
```
Alternatively, you may also do the following but you will lose the ability to uninstall it over pip:
```sh
python setup.py install develop
```

## Documentation
Currently there are no documentation support but most functions are well commented for easy understanding.
For a comprehensive overview of the main conveniences you get from this library, please see sample codes in [example](example).
