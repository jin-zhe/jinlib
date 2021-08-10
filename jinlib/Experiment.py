from torch.utils.tensorboard import SummaryWriter
import torch
from torchsummary import summary
from tabulate import tabulate

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from pathlib import Path
import logging
import shutil

from .pytorch import choose_device, get_activation, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint, to_device
from .general import config_path, save_yaml, set_logger
from .Configuration import Configuration
from .RunningAverage import RunningAverage
from types import SimpleNamespace
from .Stopwatch import Stopwatch

class Experiment:
  __metaclass__ = ABCMeta
  def __init__(self, experiment_dir: Path, config_filename='config.yml'):
    self.experiment_dir = experiment_dir
    self.config = self._preprocess_config(config_filename)
    
    # Attribute defaults before initializations
    self._context = None
    self._evaluation_metrics = None
    self._criterion_metric = None
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
    self._analyze_loader = None
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
  def context(self):
    return self._context
  
  @context.setter
  def context(self, value):
    self._context = value

  def reset_context(self):
    self.context = None

  @property
  def evaluation_metrics(self):
    if self._evaluation_metrics is None:
      self._init_evaluation_metrics()
    return self._evaluation_metrics

  @evaluation_metrics.setter
  def evaluation_metrics(self, value):
    for metric in value:
      if not hasattr(self, 'compute_batch_'+metric):
        raise ValueError('"{m}" indicated as an evaluation metric but function `compute_batch_{m}` is not defined!'.format(m=metric))
    self._evaluation_metrics = value

  @property
  def criterion_metric(self):
    return self._criterion_metric

  @criterion_metric.setter
  def criterion_metric(self, value):
    if not hasattr(self, value + '_comparator'):
      raise ValueError('Criterion metric chosen as "{m}" but comparator function `{m}_comparator` is not defined!'.format(m=value))
    self._criterion_metric = value
    self.logger('Criterion metric chosen as "{}".'.format(value))

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
  def analyze_loader(self):
    if self._analyze_loader is None:
      self._init_dataloaders()
    return self._analyze_loader

  @analyze_loader.setter
  def analyze_loader(self, value):
    self._analyze_loader = value

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
      self.batch_size = self.config.batch_size
    return self._batch_size

  @batch_size.setter
  def batch_size(self, value):
    if value < 1:
      raise ValueError('Batch size may not be less than 1!')
    self._batch_size = value

  @property
  def num_epochs(self):
    if self._num_epochs is None:
      self.num_epochs = self.config.num_epochs
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
      self.checkpoint_dir = Path(self.config.checkpoints.dir)
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

  ######## Preprocessors ###############################################################################################

  def _preprocess_config(self, config_filename):
    '''
    Fill up optional configurations with default values
    '''
    config = Configuration(config_path(self.experiment_dir, filename=config_filename))

    # Default checkpoints configurations
    if not hasattr(config, 'checkpoints'):
      config.checkpoints = SimpleNamespace()
    if not hasattr(config.checkpoints, 'dir'):
      config.checkpoints.dir = str(self.experiment_dir.resolve())
    if not hasattr(config.checkpoints, 'identifier'):
      config.checkpoints.identifier = 'pth'
    if not hasattr(config.checkpoints, 'best_prefix'):
      config.checkpoints.best_prefix = 'best'
    if not hasattr(config.checkpoints, 'last_prefix'):
      config.checkpoints.last_prefix = 'last'
    if not hasattr(config.checkpoints, 'extension'):
      config.checkpoints.extension = 'tar'
    if not hasattr(config.checkpoints, 'stats_filename'):
      config.checkpoints.stats_filename = 'stats.yml'

    # Default logs configurations
    if not hasattr(config, 'logs'):
      config.logs = SimpleNamespace()
    if not hasattr(config.logs, 'logger'):
      config.logs.logger = 'log.log'
    if not hasattr(config.logs, 'tensorboard'):
      config.logs.tensorboard = 'TB_logdir'

    # Default regularisation configuration
    if not hasattr(config, 'regularization'):
      config.regularization = None

    return config
  ######## Initializations #############################################################################################

  def _init_evaluation_metrics(self):
    # preprocess
    evaluation_metrics = [m.lower() for m in self.config.evaluation_metrics]
    not_picked = set([m for m in evaluation_metrics if m[-1] != '*'])
    picked = set([m[:-1] for m in evaluation_metrics if m[-1] == '*'])
    not_picked -= picked # ensures mutual-exclusivity by eliminating cases such as ['accuracy','accuracy*'] where 'accuracy*' survives
    if len(picked) > 1:
      raise ValueError('Only 1 evaluation metric allowed but these were indicated: {}'.format(', '.join(list(picked))))
    elif len(picked) == 0: # if nothing was selected, defaults to 'loss'
      picked.add('loss')
      not_picked.discard('loss')

    # assignments
    self.criterion_metric = list(picked)[0]
    metrics = picked | not_picked
    metrics.remove('loss')
    self.evaluation_metrics = ['loss'] + list(metrics) # force loss to be the first metric in sequence

  def _init_logger(self):
    set_logger(self.experiment_dir, log_filename=self.config.logs.logger)
    self.logger = logging.info

  def _init_recorder(self):
    self.recorder = SummaryWriter(str(self.experiment_dir / self.config.logs.tensorboard))

  def _init_epoch_stats(self):
    '''
    Initializes the stats for current *training* epoch (i.e. does not apply to validation/test etc)

    self.curr_epoch_stats = {
      'epoch': 0,
      'train': {
        'loss': None,
        # And other metrics in `self.evaluation_metrics`
        },
      'validation': {
        'loss': None,
        # And other metrics in `self.evaluation_metrics`
        }
    }
    '''
    self.curr_epoch_stats = {'epoch': 0, 'train': {}, 'validation': {}}
    for metric in self.evaluation_metrics:
      self.curr_epoch_stats['train'][metric] = None
      self.curr_epoch_stats['validation'][metric] = None

  def _init_iter_stats(self):
    '''
    Initializes the stats for current iteration. To be called at the start of
    every epoch to reset.

    self.curr_iter_stats = {
      'loss': {
        'current': None,
        'running': RunningAverage(),
      },
      # And other metrics in `self.evaluation_metrics`
    }
    '''
    self.curr_iter_stats = {}
    for metric in self.evaluation_metrics:
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
    self.analyze_loader = None

  def _init_activation(self):
    self.activation = get_activation(
      self.config.activation.choice,
      vars(self.config.activation.kwargs),
    )

  def _init_optimizer(self):
    kwargs = vars(self.config.optimization.kwargs)
    if self.config.regularization:
      if not hasattr(self.config.regularization, 'L2'):
        raise ValueError('Only L2 regularization is currently supported!')
      else: # if L2
        if 'weight_decay' in kwargs:
          self.logger('You have indicated both L2 regularization and optimizer weight_decay. Your weight_decay of {} will take precedence.'.format(kwargs['weight_decay']))
        else:
          kwargs['weight_decay'] = self.config.regularization.L2
    self.optimizer = get_optimizer(
      self.config.optimization.choice,
      kwargs,
      self.model
    )

  def _init_loss_fn(self):
    self.loss_fn = get_loss_fn(
      self.config.loss.choice,
      vars(self.config.loss.kwargs)
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
    checkpoint_filename = '{}.{}.{}'.format(
      getattr(self.config.checkpoints, choice + '_prefix'),
      self.config.checkpoints.identifier,
      self.config.checkpoints.extension
    )
    return self.checkpoint_dir / checkpoint_filename

  def save_stats(self):
    stats_path = self.checkpoint_dir / self.config.checkpoints.stats_filename
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
      self.logger('Loaded checkpoint "{}"'.format(str(checkpoint_path.resolve())))
    else:
      raise ValueError('Unknown checkpoint type {}!'.format(choice))
    return checkpoint

  def resume(self):
    '''
    Extend if more attributes are to be updated (using returned checkpoint)
    '''
    if self.has_last_checkpoint():
      checkpoint = self.load('last')
      self.curr_epoch_stats = checkpoint['curr_epoch_stats']
      self.best_epoch_stats = checkpoint['best_epoch_stats']
      self.logger('Resuming from epoch {}'.format(self.curr_epoch_stats['epoch']))
      if self.context == 'train':
        self.curr_epoch_stats['epoch'] += 1
      return checkpoint
    else:
      raise ValueError('Resume indicated but no last checkpoint found in {}. Ignoring resume.'.format(str(self.experiment_dir)))

  ######## Metric computation and update ###############################################################################

  def compute_batch_loss(self, T_out, batch):
    return self.loss_fn(T_out, self.batch_y(batch))

  def compute_batch_accuracy(self, T_out, batch):
    batch_size = T_out.size(0)
    _, T_predictions = torch.max(T_out, 1)
    return torch.sum(T_predictions == self.batch_y(batch).data) / batch_size

  def loss_comparator(self, curr_loss, past_loss):
    '''
    Checks if `curr_loss` is lower than `past_loss`
    '''
    return curr_loss < past_loss

  def accuracy_comparator(self, curr_acc, past_acc):
    '''
    Checks if `curr_acc` is higher than `past_acc`
    '''
    return curr_acc > past_acc

  def is_new_best(self, context='validation'):
    '''
    Extend if different criteria for selecting best epoch is used
    '''
    metric = self.criterion_metric
    curr_val = self.curr_epoch_stats[context][metric]
    past_val = self.best_epoch_stats[context][metric]
    comparator = getattr(self, metric + '_comparator')
    return comparator(curr_val, past_val)

  def _check_and_update_stats(self):
    if self.best_epoch_stats is None or self.is_new_best():
      self.best_epoch_stats = deepcopy(self.curr_epoch_stats)

  def _update_iter_stats(self, T_out, batch):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    This method defines which calculations are to be done for every batch and
    what iteration states are to be updated/accumulated. It will compute
    every metric listed in `self.evaluation_metrics` and update
    `self.curr_iter_stats`. Extend for specifying additional batch-specific
    computations and data to withhold across iterations in an epoch.

    Args:
      T_out: (torch.tensor) model output from the given batch
      batch: (list) list of tensors returned by a `next()` call on dataloader
            iteratble
    Notes:
      - `batch_size` here refers to the batch size pertaining to this iteration.
      It *may not* be the same as `self.batch_size` such as in the case of the
      last batch if it has a smaller size and is not dropped.
      - For every metric to be computed, method `self._compute_batch_<metric>`
      has to be defined which measure the *averaged* performance over the batch
      - The return value for every metric computation method is a one-element
      tensor which returns its numerical value via the `.item()` call.
    '''
    batch_size = T_out.size(0)
    for metric in self.evaluation_metrics:
      T_value = getattr(self, 'compute_batch_'+metric)(T_out, batch)
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
      for metric in self.evaluation_metrics:
        self.recorder.add_scalar(
          '{}/{}'.format(metric.capitalize(),context),
          self.curr_epoch_stats[context][metric],
          self.curr_epoch_stats['epoch']
        )

  def record_hyperparams(self):
    self.recorder.add_hparams(
      {
        'activation': self.config.activation.choice,
        'optimization': self.config.optimization.choice,
        'loss_function': self.config.loss.choice,
        'batch_size': self.batch_size,
        'epochs': self.num_epochs
      },
      {
        'hparam/best_validation_loss': self.best_epoch_stats['validation']['loss'],
        'hparam/best_stats': self.best_epoch_stats['epoch']
      }
    )

  def format_performance(self, value, metric):
    unit = ''
    if metric == 'accuracy':
      unit = '%'
      value *= 100
    return '{:.4f}{}'.format(value, unit)

  def log_progress(self):
    is_curr_best = self.curr_epoch_stats == self.best_epoch_stats
    epoch_str = 'Epoch #{}{}'.format(self.curr_epoch_stats['epoch'], '*' if is_curr_best else '')
    headers = [epoch_str] + [m.capitalize() for m in self.evaluation_metrics]
    rows = [[c.capitalize()] + [self.format_performance(self.curr_epoch_stats[c][m],m) for m in self.evaluation_metrics] for c in ['train', 'validation']]
    stmt = tabulate(rows, headers=headers, tablefmt="fancy_grid")
    self.logger('\n' + stmt)

  def log_training_commencement(self, num_epochs):
    self.logger('Commencing training for experiment "{}" with {} epochs.'.format(self.experiment_dir, num_epochs))

  def log_training_completion(self, elapsed_time):
    self.logger('Training {} completed after {}. Epoch {} is the best with validation {}: {}'.format(
      self.experiment_dir,
      elapsed_time,
      self.best_epoch_stats['epoch'],
      self.criterion_metric,
      self.format_performance(self.best_epoch_stats['validation'][self.criterion_metric], self.criterion_metric)
      )
    )

  def print_model(self):
    input_dim = self.input_dim  # do this first as it'll trigger dataset load
    self.logger('Model summary:')
    summary(self.model, input_dim)

  ######## Train/Evaluation/Test  ######################################################################################
  def _train_epoch_begin(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for beginning each epoch under train context.
    Need to set context again as it is reset at every end epoch routine by
    default. This is a recommended pattern to follow.
    '''
    self.context = 'train'

  def _train_epoch_end(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for ending each epoch under train context.
    '''
    self._update_train_stats()
    self.validation()
    self._check_and_update_stats()
    self.log_progress()
    self.record_progress()
    self.save()
    self.curr_epoch_stats['epoch'] += 1
    self.reset_context()
  
  def _validation_epoch_begin(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for beginning each epoch under validation context.
    Need to set context again as it is reset at every end epoch routine by
    default. This is a recommended pattern to follow.
    '''
    self.context = 'validation'

  def _validation_epoch_end(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for ending each epoch under validation context.
    '''
    self._update_validation_stats()
    self.reset_context()

  def _test_epoch_begin(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for beginning each epoch under test context.
    Need to set context again as it is reset at every end epoch routine by
    default. This is a recommended pattern to follow.
    '''
    self.context = 'test'

  def _test_epoch_end(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for ending each epoch under test context.
    '''
    self.reset_context()

  def _analyze_epoch_begin(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for beginning each epoch under analyze context.
    Need to set context again as it is reset at every end epoch routine by
    default. This is a recommended pattern to follow.
    '''
    self.context = 'analyze'

  def _analyze_epoch_end(self):
    '''
    DO NOT OVERRIDE! EXTEND AS NEEDED!
    Routines for ending each epoch under analyze context.
    '''
    self.reset_context()

  def _output_batch(self, batch):
    return self.model(self.batch_x(batch))

  def _epochal_subprocedure(self, train_mode, dataloader, num_epochs, epoch_begin, epoch_end):
    '''
    Notes:
      `batch` is agnostic to your dataset which can return more than 2 for
      __getitem__ calls
    '''
    if not train_mode:
      self.model.eval()
      torch.set_grad_enabled(False)
    for _ in range(num_epochs):
      epoch_begin()
      self._init_iter_stats()   # reset stats at every epoch
      if train_mode:
        self.model.train()
      for i, batch in enumerate(dataloader, 0):
        batch = to_device(choose_device(), *batch)
        if train_mode:
          self.optimizer.zero_grad()  # zeroise parameter gradients
        T_out = self._output_batch(batch)
        self._update_iter_stats(T_out, batch)
        T_loss = self.curr_iter_stats['loss']['current']
        if train_mode:
          T_loss.backward()           # calc gradients
          self.optimizer.step()       # update params
      epoch_end()               # run end epoch routine
    if not train_mode:
      torch.set_grad_enabled(True)

  def train(self, resume=False):
    self.context = 'train'
    if resume:
      self.resume()
    else:
      self.model.to(choose_device())
      self._init_epoch_stats()
    
    self.print_model()
    self.logger(self.model)
    self.log_training_commencement(self.num_epochs)
    elapsed = Stopwatch()
    self._epochal_subprocedure(True, self.train_loader, self.num_epochs, self._train_epoch_begin, self._train_epoch_end)
    self.log_training_completion(elapsed())
    self.record_hyperparams()

  def validation(self, load_best=False):
    '''
    Note that we do not normally load the best checkpoint because validation is
    a subroutine at the end of every training epoch.
    '''
    self.context = 'validation'
    if load_best:
      self.load('best')
    self._epochal_subprocedure(False, self.validation_loader, 1, self._validation_epoch_begin, self._validation_epoch_end)
  
  def test(self):
    '''
    To override/extend as needed but ensure context is set as it affects
    checkpoint loading.
    '''
    self.context = 'test'
    self.load('best')
    self._epochal_subprocedure(False, self.test_loader, 1, self._test_epoch_begin, self._test_epoch_end)

  def analyze(self):
    '''
    To override/extend as needed but ensure context is set as it affects
    checkpoint loading.
    '''
    self.context = 'analyze'
    self.load('best')
    self._epochal_subprocedure(False, self.analyze_loader, 1, self._analyze_epoch_begin, self._analyze_epoch_end)

  ######## Static helper methods #######################################################################################

  @staticmethod
  def batch_x(batch):
    return batch[0]  # samples default to first item in batch (recommended)

  @staticmethod
  def batch_y(batch):
    return batch[1]  # labels default to second item in batch (recommended)
