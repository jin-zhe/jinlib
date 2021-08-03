from torch.utils.tensorboard import SummaryWriter
import torch
from torchsummary import summary
from jinlib.pytorch import choose_device, get_activation, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint, to_device
from jinlib.general import config_path, save_yaml, set_logger
from jinlib import Configuration, RunningAverage, Stopwatch
import logging

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from pathlib import Path
import shutil

class Experiment:
  __metaclass__ = ABCMeta
  def __init__(self, experiment_dir: Path, config_filename='config.yml'):
    # Experiment configs
    self.experiment_dir = experiment_dir
    self.experiment_config = Configuration(config_path(self.experiment_dir, filename=config_filename))

    # Attribute defaults before initializations
    self._metrics = None
    self._logger = None
    self._recorder = None
    self._model = None
    self._activation = None
    self._optimizer = None
    self._loss_fn = None
    self._dataset = None
    self._train_loader = None
    self._validation_loader = None
    self._test_loader = None
    self._input_dim = None
    self._batch_size = None
    self._num_epochs = None
    self._checkpoint_dir = None
    self._last_checkpoint_path = None
    self._best_checkpoint_path = None
    self.curr_epoch_stats = None
    self.best_epoch_stats = None
    self.curr_iter_stats = None

  ######## Getters/setters #############################################################################################
  @property
  def metrics(self):
    if self._metrics is None:
      metrics = set(self.experiment_config.metrics)
      if 'loss' not in metrics:
        metrics.add('loss') # loss is a compulsory metric!
      self.metrics = metrics
    return self._metrics

  @metrics.setter
  def metrics(self, value):
    self._metrics = value

  @property
  def logger(self):
    if self._logger is None:
      self._init_logger()
    return self._logger

  @logger.setter
  def logger(self, value):
    self._logger = value

  @property
  def recorder(self):
    if self._recorder is None:
      self._init_recorder()
    return self._recorder

  @recorder.setter
  def recorder(self, value):
    self._recorder = value

  @property
  def model(self):
    if self._model is None:
      self._init_model()
    return self._model

  @model.setter
  def model(self, value):
    self._model = value

  @property
  def dataset(self):
    if self._dataset is None:
      self._init_dataset()
    return self._dataset

  @dataset.setter
  def dataset(self, value):
    self._dataset = value

  @property
  def train_loader(self):
    if self._train_loader is None:
      self._init_dataloaders()
    return self._train_loader

  @train_loader.setter
  def train_loader(self, value):
    self._train_loader = value

  @property
  def validation_loader(self):
    if self._validation_loader is None:
      self._init_dataloaders()
    return self._validation_loader

  @validation_loader.setter
  def validation_loader(self, value):
    self._validation_loader = value

  @property
  def test_loader(self):
    if self._test_loader is None:
      self._init_dataloaders()
    return self._test_loader

  @test_loader.setter
  def test_loader(self, value):
    self._test_loader = value

  @property
  def input_dim(self):
    if self._input_dim is None:
      self._init_input_dim()
    return self._input_dim

  @input_dim.setter
  def input_dim(self, value):
    self._input_dim = value

  @property
  def batch_size(self):
    if self._batch_size is None:
      self.batch_size = self.experiment_config.hyperparams.batch_size
    return self._batch_size

  @batch_size.setter
  def batch_size(self, value):
    if value < 1:
      raise ValueError('Batch size may not be less than 1!')
    self._batch_size = value

  @property
  def num_epochs(self):
    if self._num_epochs is None:
      self.num_epochs = self.experiment_config.hyperparams.num_epochs
    return self._num_epochs

  @num_epochs.setter
  def num_epochs(self, value):
    if value < 1:
      raise ValueError('Number of epochs may not be less than 1!')
    self._num_epochs = value

  @property
  def activation(self):
    if self._activation is None:
      self._init_activation()
    return self._activation

  @activation.setter
  def activation(self, value):
    self._activation = value

  @property
  def optimizer(self):
    if self._optimizer is None:
      self._init_optimizer()
    return self._optimizer

  @optimizer.setter
  def optimizer(self, value):
    self._optimizer = value

  @property
  def loss_fn(self):
    if self._loss_fn is None:
      self._init_loss_fn()
    return self._loss_fn

  @loss_fn.setter
  def loss_fn(self, value):
    self._loss_fn = value

  @property
  def checkpoint_dir(self):
    if self._checkpoint_dir is None:
      if self.experiment_config.checkpoints.dir:
        self.checkpoint_dir = Path(self.experiment_config.checkpoints.dir)
      else:
        self.checkpoint_dir =  self.experiment_dir
    return self._checkpoint_dir

  @checkpoint_dir.setter
  def checkpoint_dir(self, value):
    self._checkpoint_dir = value

  @property
  def last_checkpoint_path(self):
    if self._last_checkpoint_path is None:
      self._last_checkpoint_path = self.resolve_checkpoint_path('last')
    return self._last_checkpoint_path

  @last_checkpoint_path.setter
  def last_checkpoint_path(self, value):
    self._last_checkpoint_path = value

  @property
  def best_checkpoint_path(self):
    if self._best_checkpoint_path is None:
      self._best_checkpoint_path = self.resolve_checkpoint_path('best')
    return self._best_checkpoint_path

  @best_checkpoint_path.setter
  def best_checkpoint_path(self, value):
    self._best_checkpoint_path = value

  ######## Initializations #############################################################################################

  def _init_logger(self):
    set_logger(self.experiment_dir, # omit this line in overriden function if logger is already set elsewhere
      log_filename=self.experiment_config.logs.logger)
    self.logger = logging.info

  def _init_recorder(self):
    self.recorder = SummaryWriter(str(self.experiment_dir / self.experiment_config.logs.tensorboard))

  def _init_epoch_stats(self):
    '''
    Initializes the stats for current *training* epoch (i.e. does not apply to validation/test etc)

    self.curr_epoch_stats = {
      'epoch': 0,
      'train': {
        'loss': None,
        # And other metrics in `self.metrics`
        },
      'validation': {
        'loss': None,
        # And other metrics in `self.metrics`
        }
    }
    '''
    self.curr_epoch_stats = {'epoch': 0, 'train': {}, 'validation': {}}
    for metric in self.metrics:
      self.curr_epoch_stats['train'][metric] = None
      self.curr_epoch_stats['validation'][metric] = None

  def _init_iter_stats(self):
    '''
    Initializes the stats for current training/evaluation iteration. To be used at the start of every epoch to reset.

    self.curr_iter_stats = {
      'loss': {
        'current': None,
        'running': RunningAverage(),
      },
      # And other metrics in `self.metrics`
    }
    '''
    self.curr_iter_stats = {}
    for metric in self.metrics:
      self.curr_iter_stats[metric] = {
        'current': None,              # Current metric value at each iteration
        'running': RunningAverage()   # Running average at each iteration
      }

  @abstractmethod
  def _init_model(self):
    '''
    Must override
    '''
    self.model = None

  @abstractmethod
  def _init_dataset(self):
    '''
    Must override
    '''
    self.dataset = None

  @abstractmethod
  def _init_dataloaders(self):
    '''
    Must override
    '''
    self.train_loader = None
    self.validation_loader = None
    self.test_loader = None

  def _init_activation(self):
    self.activation = get_activation(
      self.experiment_config.activation.choice,
      vars(self.experiment_config.activation.kwargs),
    )

  def _init_optimizer(self):
    self.optimizer = get_optimizer(
      self.experiment_config.optimization.choice,
      vars(self.experiment_config.optimization.kwargs),
      self.model
    )

  def _init_loss_fn(self):
    self.loss_fn = get_loss_fn(
      self.experiment_config.loss.choice,
      vars(self.experiment_config.loss.kwargs)
    )

  def _init_input_dim(self):
    images, _ = next(iter(self.train_loader))  
    self.input_dim = images[0].size()

  ######## Checkpoint related ##########################################################################################

  def _format_state(self): 
    return {
      'model_state': self.model.state_dict(),
      'optim_state': self.optimizer.state_dict(),
      'curr_epoch_stats': self.curr_epoch_stats,
      'best_epoch_stats': self.best_epoch_stats
    }

  def _format_stats(self):
    return {
      'last_epoch': self.curr_epoch_stats,
      'best_epoch': self.best_epoch_stats
    }

  def resolve_checkpoint_path(self, choice):
    basename = self.experiment_config.checkpoints.basename
    suffix = getattr(self.experiment_config.checkpoints, choice + '_suffix')
    extension = self.experiment_config.checkpoints.extension
    checkpoint_filename = '{}.{}.{}'.format(basename, suffix, extension)
    return self.checkpoint_dir / checkpoint_filename

  def save_stats(self):
    stats_path = self.checkpoint_dir / self.experiment_config.checkpoints.stats_filename
    save_yaml(self._format_stats(), stats_path)

  def save(self):
    self.save_stats()
    save_checkpoint(self.last_checkpoint_path, self._format_state())
    if self.curr_epoch_stats == self.best_epoch_stats:    # if currently at best epoch
      shutil.copyfile(                                    # copy last checkpoint as best
        str(self.last_checkpoint_path.resolve()),
        str(self.best_checkpoint_path.resolve())
      )

  def has_last_checkpoint(self):
    return self.last_checkpoint_path.is_file()

  def load(self, choice):
    if choice == 'last' or choice == 'best':
      checkpoint_path = getattr(self, '{}_checkpoint_path'.format(choice))
      checkpoint = load_checkpoint(
        checkpoint_path,
        self.model,
        self.optimizer
      )
      self.logger('Resuming epoch {} from checkpoint {}'.format(
        checkpoint['curr_epoch_stats']['epoch'], str(checkpoint_path.resolve())))
    else:
      raise ValueError('Unknown checkpoint type {}!'.format(choice))
    return checkpoint

  def resume(self):
    '''
    Override if more attributes are to be updated (using returned checkpoint)
    '''
    if self.has_last_checkpoint():
      checkpoint = self.load('last')
      self.curr_epoch_stats = checkpoint['curr_epoch_stats']
      self.best_epoch_stats = checkpoint['best_epoch_stats']
      self.curr_epoch_stats['epoch'] += 1
      return checkpoint
    else:
      raise ValueError('Resume indicated but no last checkpoint found in {}. Ignoring resume.'.format(str(self.experiment_dir)))

  ######## Metric computation and update ###############################################################################

  def compute_metric(self, metric, T_outputs, T_labels):
    '''
    Get the computation function for the given metric and compute based on it.
    Example: if metric is 'loss', returns `self.compute_loss(T_outputs, T_labels)` function
    It's the caller's responsibility of ensuring that method `self.compute_<metric>` exists in the class at run time
    '''
    return getattr(self, 'compute_'+metric)(T_outputs, T_labels)

  def compute_loss(self, T_outputs, T_labels):
    return self.loss_fn(T_outputs, T_labels)

  def compute_accuracy(self, T_outputs, T_labels):
    _, T_predictions = torch.max(T_outputs, 1)
    return torch.sum(T_predictions == T_labels.data)

  def compute_precision(self, T_outputs, T_labels):
    pass #TODO

  def compute_recall(self, T_outputs, T_labels):
    pass  #TODO

  def compute_F1(self, T_outputs, T_labels):
    pass  #TODO

  def is_new_best(self, metric='loss', context='validation'):
    '''
    Override if different criteria for selecting best epoch is used
    '''
    return self.best_epoch_stats is None or (
      self.curr_epoch_stats[context][metric] < self.best_epoch_stats[context][metric])

  def _check_and_update_stats(self):
    if self.is_new_best():
      self.best_epoch_stats = deepcopy(self.curr_epoch_stats)

  def _update_iter_stats(self, T_outputs, T_labels, batch_size):
    '''
    Note the batch_size here refers to the batch size pertaining to this iteration. It may not be the same as
    self.batch_size as in the case of the last batch if it were not dropped
    '''
    for metric in self.curr_iter_stats:
      T_value = self.compute_metric(metric, T_outputs, T_labels)
      self.curr_iter_stats[metric]['current'] = T_value
      self.curr_iter_stats[metric]['running'].update(T_value.item() * batch_size, batch_size)

  def _update_curr_epoch_stats(self, context):
    for metric in self.curr_iter_stats:
      self.curr_epoch_stats[context][metric] = self.curr_iter_stats[metric]['running']()

  def _update_train_stats(self):
    self._update_curr_epoch_stats('train')

  def _update_validation_stats(self):
    self._update_curr_epoch_stats('validation')

  ######## Logging  ####################################################################################################

  def record_progress(self):
    for context in ['train', 'validation']:
      for metric in self.metrics:
        self.recorder.add_scalar(
          '{}/{}'.format(metric.capitalize(),context),
          self.curr_epoch_stats[context][metric],
          self.curr_epoch_stats['epoch']
        )

  def record_hyperparams(self):
    self.recorder.add_hparams(
      {
        'activation': self.experiment_config.activation.choice,
        'optimization': self.experiment_config.optimization.choice,
        'loss_function': self.experiment_config.loss.choice,
        'batch_size': self.batch_size,
        'epochs': self.num_epochs
      },
      {
        'hparam/best_validation_loss': self.best_epoch_stats['validation']['loss'],
        'hparam/best_stats': self.best_epoch_stats['epoch']
      }
    )

  def log_progress(self):
    stmt = '[Epoch {:0>3d}] '.format(self.curr_epoch_stats['epoch'])
    stmt += 'Train Loss: {:.4f}'.format(self.curr_epoch_stats['train']['loss'])
    stmt += ' | '
    stmt += 'Validation Loss {:.4f}'.format(self.curr_epoch_stats['validation']['loss'])
    self.logger(stmt)

  def log_commencement(self):
    self.logger('Commencing training for experiment: {}'.format(self.experiment_dir))

  def log_completion(self, elapsed_time):
    self.logger('Experiment {} completed after {}. Epoch {} is the best with {:.4f} validation loss.'.format(
      self.experiment_dir,
      elapsed_time,
      self.best_epoch_stats['epoch'],
      self.best_epoch_stats['validation']['loss'])
    )

  ######## Train/Evaluation/Test  ######################################################################################
  def _train_epoch_end(self):
    '''
    Epoch end routines for train loop
    Recommended not to override
    '''
    self._update_train_stats()
    self.validation()
    self._update_validation_stats()
    self._check_and_update_stats()
    self.log_progress()
    self.record_progress()
    self.save()
    self.curr_epoch_stats['epoch'] += 1
  
  @abstractmethod
  def _validation_epoch_end(self):
    '''
    Must override
    '''
    pass

  @abstractmethod
  def _test_epoch_end(self):
    '''
    Must override
    '''
    pass

  @abstractmethod
  def _analyze_epoch_end(self):
    '''
    Must override
    '''
    pass

  def _epochal_subprocedure(self, context, epoch_end):
    if context == 'train':
      num_epochs = self.num_epochs
      dataloader = self.train_loader
    elif context == 'validation' or context == 'test':
      num_epochs = 1
      dataloader = self.validation_loader
    else:
      raise ValueError('Unknown epochal context {}!'.format(context))

    if context == 'validation':
      self.model.eval()
      torch.set_grad_enabled(False)
    for _ in range(num_epochs):
      self._init_iter_stats()   # reset stats at every epoch
      if context == 'train':
        self.model.train()
      for i, (T_samples, T_labels) in enumerate(dataloader, 0):
        batch_size = T_samples.size(0)
        T_samples, T_labels = to_device(choose_device(), T_samples, T_labels)
        if context == 'train':
          self.optimizer.zero_grad()  # zeroise parameter gradients
        T_outputs = self.model(T_samples)
        self._update_iter_stats(T_outputs, T_labels, batch_size)
        T_loss = self.curr_iter_stats['loss']['current']
        if context == 'train':
          T_loss.backward()           # calc gradients
          self.optimizer.step()       # update params
      epoch_end()               # run end epoch routines
    if context == 'validation':
      torch.set_grad_enabled(True)

  def train(self, resume=False):
    if resume:
      self.resume()
    else:
      self.model.to(choose_device())
      self._init_epoch_stats()
    
    self.logger('Model summary:')
    summary(self.model, self.input_dim)
    self.logger(self.model)
    self.log_commencement()
    elapsed = Stopwatch()
    self._epochal_subprocedure('train', self._train_epoch_end)
    self.log_completion(elapsed())
    self.record_hyperparams()

  def validation(self):
    self._epochal_subprocedure('validation', self._validation_epoch_end)
  
  @abstractmethod
  def test(self):
    '''
    Must override
    '''
    pass

  @abstractmethod
  def analyze(self):
    '''
    Must override
    '''
    pass
