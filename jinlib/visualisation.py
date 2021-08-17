import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path

from .general import save_pickle

def propose_dpi(smallset_fontsize):
  '''
  Propose reasonable DPI based on smallest fontsize in the plot
  '''
  return -500 * smallset_fontsize/10 + 500

def plot_confusion_matrix(classes, confusion_matrix, experiment_dir: Path, annot_size=None, dpi=None, figsize=None, annot=True, cmap='Greens', filename='confusion_matrix.png', save_data=True):
  '''
  Plot a confusion matrix
  '''
  outpath = experiment_dir / filename
  if save_data:
    save_pickle({'classes': classes, 'confusion_matrix': confusion_matrix}, outpath.with_suffix('.pkl'))
  
  if figsize:
    plt.figure(figsize=figsize)
  else:
    plt.figure()
  
  if annot_size is None:
    annot_size = -0.1 * len(classes) + 8.15
  tick_size = annot_size * 1.5

  if dpi is None:
    dpi = propose_dpi(annot_size)
  
  print('Annotation size: {}, tick size: {}, DPI: {}'.format(annot_size, tick_size, dpi))
  data = pd.DataFrame(confusion_matrix*100, columns=classes, index=classes)
  hm = sns.heatmap(data, vmin=0, vmax=100, xticklabels=1, yticklabels=1, annot=annot, cmap=cmap, square=True, fmt='.1f', annot_kws={'fontsize': annot_size})
  plt.setp(hm.get_xticklabels(), fontsize=tick_size)
  plt.setp(hm.get_yticklabels(), fontsize=tick_size)
  plt.title('Confusion matrix (%)')
  plt.xlabel('Predicted')
  plt.ylabel('Ground Truth')
  plt.tight_layout()
  plt.savefig(str(outpath.resolve()), dpi=dpi)
  plt.close()
  print(f'Confusion matrix plot saved to {str(outpath.resolve())}')

def plot_classwise_accuracies(classes, classwise_accuracies, experiment_dir: Path,
  figsize=None, color='b', dpi=300, filename='classwise_accuracies.png', save_data=True):
  outpath = experiment_dir / filename
  if save_data:
    save_pickle({'classes': classes, 'classwise_accuracies': classwise_accuracies}, outpath.with_suffix('.pkl'))
  if figsize:
    plt.figure(figsize=figsize)
  else:
    plt.figure()
  df = pd.DataFrame((classwise_accuracies*100), columns=['Accuracy'])
  df['Label'] = classes
  bp = sns.barplot(data=df, x='Accuracy', y='Label', order=df.sort_values('Accuracy').Label, edgecolor='w')
  for bar in bp.patches:
    width = bar.get_width()
    plt.text(width-0.55, bar.get_y() + 0.55 * bar.get_height(),
      '{:1.2f}'.format(width), ha='right', va='center', color='w')
  bp.tick_params()
  bp.set(xlim=(0, 100))
  bp.set_title('Classwise Accuracies (%)')
  sns.despine(left = True, bottom = True)
  plt.savefig(str(outpath.resolve()), dpi=dpi)
  plt.close()
  print(f'Classwise accuracies plot saved to {str(outpath.resolve())}')
