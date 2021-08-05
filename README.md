# jinlib

Library package for my line of PyTorch work. Includes convenience functions that are more semantic-oriented and improves readability. Also sports a lightweight framework that abstracts away standard routines and allows the user to quickly implement and compare various experiments.

To give some sense of just how handy the framework brought by the [Experiment class](jinlib/Experiment.py) and library is for whipping up quick experiments, you may wish to compare between [our code](example/CIFAR10_classifier.py) and [PyTorch's guide](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for the same CIFAR10 classifier experiment.

## Dependencies
Too many to list manually, please install and install neccessary dependencies yourself.

## Install instructions
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
Alternatively, you may also do
```sh
python setup.py install develop
```

## Documentation
Currently there are no documentation support but most functions are well commented for easy understanding.
For a comprehensive overview of the main conveniences you get from this library, please see sample codes in [example](example).
