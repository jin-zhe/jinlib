from types import SimpleNamespace
from pathlib import Path
import yaml

class Configuration():
  '''
  Class that loads configuration settings from a yaml file.
  Example:
    ```
    config = Configuration(config_path)
    config.learning_rate         # get the learning rate
    config.learning_rate = 0.5   # set the value of learning_rate
    ```
  '''
  def __init__(self, config_path: Path):
    self.config_path = config_path
    self._load()

  def save(self, yaml_path: Path, indent=2):
    '''Saves current configuration to given path'''
    config_dict = self.__dict__.copy()
    del config_dict['config_path']
    payload = Configuration.to_dict(config_dict)
    with yaml_path.open('w') as f:
      yaml.dump(payload, f, indent=indent, sort_keys=True)
          
  def _load(self, loader='UnsafeLoader'):
    '''Loads yaml file as config'''
    with self.config_path.open() as f:
      config_dict = yaml.load(f, Loader=getattr(yaml, loader))
    obj = Configuration.to_obj(config_dict)
    self.__dict__.update(obj.__dict__)

  @staticmethod
  def ensure_default(config, key, value):
    key_hierarchy = key.split('.')
    # Base case
    if len(key_hierarchy) == 1:
      if not hasattr(config, key):
        config.key = value
    else:
      top_key = key_hierarchy[0]
      if not hasattr(config,top_key):
        setattr(config, top_key, SimpleNamespace())
      Configuration.ensure_default(getattr(config,top_key), '.'.join(key[1:]), value)

  @staticmethod
  def to_obj(value):
    '''Deep conversion of dict into obj using SimpleNamespace'''
    if type(value) is dict:         # if dict
      return SimpleNamespace(**{k: Configuration.to_obj(v) for k,v in value.items()})
    elif type(value) is list:
      return [Configuration.to_obj(item) for item in value]
    else:
      return value

  @staticmethod
  def to_dict(value):
    '''Deep conversion of object into dict'''
    if hasattr(value, '__dict__'):  # if object
      return Configuration.to_dict(value.__dict__)
    elif type(value) is dict:       # if dict
      return {k: Configuration.to_dict(v) for k,v in value.items()}
    else:
      return value
