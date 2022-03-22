import matplotlib.pyplot as plt
from matplotlib import rcParams
from pandas.io.sql import DatabaseError
import seaborn as sns
import pandas as pd
import numpy as np

from pathlib import Path

from .general import save_pickle

def set_autolayout():
  rcParams.update({'figure.autolayout': True})

def unlabel_xy(axes):
  axes.set_xticks([])
  axes.set_yticks([])
  axes.set(xlabel=None, ylabel=None)

def annotate(x, y, annotation, axis, fontsize=10, offset=.02):
  a = pd.concat({'x': x, 'y': y, 'annotation': annotation}, axis=1)
  for _, point in a.iterrows():
    text = point['annotation']
    if type(text) is float:
      text = '{:.2f}'.format(text)
    axis.text(point['x']+offset, point['y'], text, fontsize=fontsize)

def propose_dpi(smallset_fontsize):
  '''
  Propose reasonable DPI based on smallest fontsize in the plot
  '''
  return -500 * smallset_fontsize/10 + 500

def plot_confusion_matrix(classnames, confusion_matrix, experiment_dir: Path, annot_size=None, dpi=None, figsize=None, annot=True, cmap='Greens', filename='confusion_matrix.png', save_data=True):
  '''
  Plot a confusion matrix
  '''
  outpath = experiment_dir / filename
  if save_data:
    save_pickle({'classnames': classnames, 'confusion_matrix': confusion_matrix}, outpath.with_suffix('.pkl'))
  
  fig, ax = plt.subplots()
  if figsize:
    fig.set_size_inches(*figsize)
  
  if annot_size is None:
    annot_size = -0.1 * len(classnames) + 8.15
  tick_size = annot_size * 1.5

  if dpi is None:
    dpi = propose_dpi(annot_size)
  
  print('Annotation size: {}, tick size: {}, DPI: {}'.format(annot_size, tick_size, dpi))
  data = pd.DataFrame(confusion_matrix*100, columns=classnames, index=classnames)
  hm = sns.heatmap(data, vmin=0, vmax=100, xticklabels=1, yticklabels=1, annot=annot, cmap=cmap, square=True, fmt='.1f', annot_kws={'fontsize': annot_size}, ax=ax)
  plt.setp(hm.get_xticklabels(), fontsize=tick_size)
  plt.setp(hm.get_yticklabels(), fontsize=tick_size)
  plt.title('Confusion Matrix (%)')
  plt.xlabel('Predicted')
  plt.ylabel('Ground Truth')
  plt.tight_layout()
  plt.savefig(str(outpath.resolve()), dpi=dpi, bbox_inches="tight")
  plt.close()
  print(f'Confusion matrix plot saved to {str(outpath.resolve())}')

def plot_classwise_accuracies(classnames, classwise_accuracies, experiment_dir: Path, figsize=None, tick_size=None, dpi=300, filename='classwise_accuracies.png', save_data=True):
  outpath = experiment_dir / filename
  if save_data:
    save_pickle({'classnames': classnames, 'classwise_accuracies': classwise_accuracies}, outpath.with_suffix('.pkl'))
  fig, ax = plt.subplots()
  if figsize:
    fig.set_size_inches(*figsize)

  ax.margins(x=0)
  
  if tick_size is None:
    tick_size = -.1 * len(classnames) + 10

  data = pd.DataFrame((classwise_accuracies*100), columns=['accuracy'])
  data['classname'] = classnames
  data.sort_values('accuracy', inplace=True)

  # Plot accuracy line
  lp = sns.lineplot(data=data, x='accuracy', y='classname', estimator=None, color='#00b894', ax=ax, sort=False)
  ax.set(xlabel='Accuracy (%)', ylabel='Classes')
  plt.setp(lp.get_yticklabels(), fontsize=tick_size)
  
  # Plot class lines
  colors = ['#81ecec','#74b9ff']
  x_min = data.iloc[0]['accuracy']
  for i, (_, row) in enumerate(data.iterrows()):
    ax.plot((x_min, row['accuracy']), (i, i), color=colors[i%2], linestyle='--', linewidth=.5)

  # Annotate accuracies
  annotate(x=data['accuracy'],y=data['classname'],annotation=data['accuracy'],fontsize=tick_size, axis=ax)

  plt.title('Classwise Accuracies')
  plt.savefig(str(outpath.resolve()), dpi=dpi, bbox_inches="tight")
  plt.close()
  print(f'Classwise accuracies plot saved to {str(outpath.resolve())}')
