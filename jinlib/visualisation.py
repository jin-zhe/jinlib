from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(classnames, confusion_matrix, experiment_dir: Path,
  figsize=(100,64), annot=True, cmap='Greens', filename='confusion_matrix.png'):
  plt.figure(figsize=figsize)
  plt.rcParams['font.size'] = 42
  hm = sns.heatmap(confusion_matrix, vmin=0, vmax=1, annot=annot, cmap=cmap, square=True, annot_kws={"size": 16})
  hm.set_xticklabels(classnames, rotation=-90)
  hm.set_yticklabels(classnames, rotation=0)
  outpath = str(experiment_dir.resolve() / filename)
  plt.savefig(outpath)
  plt.close()
  print(f'Confusion matrix plot saved to {outpath}')

def plot_classwise_accuracies(classnames, classwise_accuracies,
  experiment_dir: Path, figsize=(128,64), color='b', filename='classwise_accuracies.png'):
  plt.figure(figsize=figsize)
  bp = sns.barplot(data=classwise_accuracies, color=color, edgecolor='w', orient='h')
  bp.tick_params(labelsize=64)
  bp.set_yticklabels(classnames, fontsize=30)
  bp.set(xlim=(0, 1))
  bp.set_title('Classwise Accuracies', fontsize=128)
  sns.despine(left = True, bottom = True)
  outpath = str(experiment_dir.resolve() / filename)
  plt.savefig(outpath)
  plt.close()
  print(f'Classwise accuracies plot saved to {outpath}')
