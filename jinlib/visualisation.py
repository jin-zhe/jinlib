import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path

def plot_confusion_matrix(classnames, confusion_matrix, experiment_dir: Path,
  figsize=None, annot=True, cmap='Greens', dpi=300, filename='confusion_matrix.png'):
  if figsize:
    plt.figure(figsize)
  else:
    plt.figure()
  confusion_matrix *= 100 # convert to percentage
  hm = sns.heatmap(confusion_matrix, vmin=0, vmax=100, annot=annot,
    cmap=cmap, square=True, fmt='.1f')
  hm.set_xticklabels(classnames, rotation=-90)
  hm.set_yticklabels(classnames, rotation=0)
  hm.set_title('Confusion matrix (%)')
  outpath = str(experiment_dir.resolve() / filename)
  plt.savefig(outpath, dpi=dpi)
  plt.close()
  print(f'Confusion matrix plot saved to {outpath}')

def plot_classwise_accuracies(classnames, classwise_accuracies, experiment_dir: Path,
  figsize=None, color='b', dpi=300, filename='classwise_accuracies.png'):
  if figsize:
    plt.figure(figsize)
  else:
    plt.figure()
  df = pd.DataFrame((classwise_accuracies.numpy()*100), columns=['Accuracy'])
  df['Label'] = classnames
  bp = sns.barplot(data=df, x='Accuracy', y='Label', order=df.sort_values('Accuracy').Label, edgecolor='w')
  for bar in bp.patches:
    width = bar.get_width()
    plt.text(width-0.55, bar.get_y() + 0.55 * bar.get_height(),
      '{:1.2f}'.format(width), ha='right', va='center', color='w')
  bp.tick_params()
  bp.set(xlim=(0, 100))
  bp.set_title('Classwise Accuracies (%)')
  sns.despine(left = True, bottom = True)
  outpath = str(experiment_dir.resolve() / filename)
  plt.savefig(outpath, dpi=dpi)
  plt.close()
  print(f'Classwise accuracies plot saved to {outpath}')
