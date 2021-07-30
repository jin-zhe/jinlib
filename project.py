from pathlib import Path

CURDIR = Path(__file__).resolve().parent

def get_experiment_dir(stage, exp_name, experiments_dirname='experiments'):
  return CURDIR / '..' / 'stage_{}'.format(stage) /experiments_dirname / exp_name