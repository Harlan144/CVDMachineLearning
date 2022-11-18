import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig("Saves/plot_prc")



def plot_metrics(history, savePath):
  
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    plt.savefig(savePath)



def plot_cm(labels, predictions, p=0.5, savePath="Saves/PlotConfusionMatrix"):  
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  #plt.rcParams.update({'font.size': 16})
  sns.heatmap(cm, cmap=sns.color_palette("icefire", as_cmap=True), annot=True, annot_kws={'size': 15}, fmt="d")
  plt.title('CNN Confusion matrix')
  ax = plt.gca()
  ax.set_xticklabels(['Friendly', 'Unfriendly'])
  ax.set_yticklabels(['Friendly', 'Unfriendly'])
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.savefig("Saves/PlotConfusionMatrix")