from datetime import datetime
from pathlib import Path
from glob import glob
import urllib.request
import logging
import pickle
import json
import yaml
import csv

def config_path(experiment_dir: Path, filename='config.yml'):
  file_path = experiment_dir / filename
  if not file_path.exists():
    raise ValueError('{} does not exist!'.format(file_path))
  return file_path

def curr_datetime():
  return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def download(url, file_path: Path, overwrite_existing=False):
  if not overwrite_existing and file_path.is_file():
    return
  if not file_path.is_file():
    print('{} does not exist. Downloading file...'.format(file_path))
  else:
    print('{} arleady exist. Re-downloading file...'.format(file_path))
  urllib.request.urlretrieve(url, file_path)
  print('Download complete!')

def ls(dir_path: Path, pattern='*'):
  '''
  Lists everything in the given directory
  '''
  return dir_path.glob(pattern)

def list_files(dir_path: Path, extension):
  '''
  Lists all files with given extension under the given directory path
  '''
  return [x for x in ls(dir_path, '*' + extension)]

def list_folders(dir_path):
  '''
  Lists all the names of folders within the given directory path
  '''
  return [x for x in ls(dir_path) if x.is_dir()]

def load_csv(csv_path: Path, ignore_first_row=True, ignore_empty_rows=True, delimiter=','):
  '''
  Returns all the rows of a csv file
  '''
  rows = []
  with csv_path.open() as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=delimiter)
    if ignore_first_row:
      next(csv_reader)
    for row in csv_reader:
      if len(row) != 0 or not ignore_empty_rows:
        rows.append(row)
  return rows

def save_csv(rows, output_path: Path, titles=None, delimiter=','):
  '''
  Writes out rows to csv file given output path
  '''
  with output_path.open('w') as csvfile:
    out_writer = csv.writer(csvfile, delimiter=delimiter)
    if titles:
      out_writer.writerow(titles)
    for row in rows:
      out_writer.writerow(row)

def load_json(json_path: Path):
  '''
  Loads JSON from given path
  '''
  with json_path.open() as f:
    return json.load(f)

def save_json(obj, json_path: Path):
  '''
  Saves given object `obj` as JSON file
  '''
  with json_path.open('w') as f:
    json.dump(obj, f)

def load_yaml(yaml_path: Path, loader='FullLoader'):
  '''
  Loads YAML from given path
  Args:
    yaml_path: (Path) path of yaml file
    loader: (str) choice of yaml loader [SafeLoader | FullLoader | UnsafeLoader/Loader]
      - SafeLoader: Recommended for untrusted input by loading a subset of YAML.
      - FullLoader: For more trusted inputs with limitation prohibiting arbitrary code execution.
      - UnsafeLoader/Loader: Unsafe but has the full power of YAML
  '''
  with yaml_path.open() as f:
    return yaml.load(f, Loader=getattr(yaml, loader))

def save_yaml(obj, yaml_path: Path):
  '''
  Saves given object `obj` as YAML file
  '''
  with yaml_path.open('w') as f:
    yaml.dump(obj, f)

def load_pickle(pkl_path: Path):
  '''
  Loads a pickle file from the given path
  '''
  with pkl_path.open('rb') as f:
    return pickle.load(f)

def save_pickle(obj, pkl_path: Path):
  '''
  Saves a given pickle file to the given path
  '''
  with pkl_path.open('wb') as f:
    pickle.dump(obj, f)

def set_logger(directory: Path, log_filename='log.log', console_logging=True):
  '''
  Set the logger to log info in terminal and file `log_path`.
  Args:
    directory: (Path) directory path where log will reside
    log_filename: (str) filename for log
    console_logging: (bool) if also print log to console
  Example usage:  `logging.info("Starting training...")`
  Adapted from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
  '''
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  log_path = (directory / log_filename).resolve()
  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    if console_logging:
      stream_handler = logging.StreamHandler()
      stream_handler.setFormatter(logging.Formatter('%(message)s'))
      logger.addHandler(stream_handler)

def log_progress(e, train_loss, train_acc, val_loss, val_acc):
  stmt = '[Epoch {:0>3d}] '.format(e)
  stmt += 'Train Loss: {:.4f}'.format(train_loss)
  if train_acc:
    stmt += ', Train Acc: {:.4f}%'.format(train_acc*100)
  stmt += ' | '
  stmt += 'Val Loss {:.4f}'.format(val_loss)
  if val_acc:
    stmt += ', Val Acc {:.4f}%'.format(val_acc*100)
  logging.info(stmt)

