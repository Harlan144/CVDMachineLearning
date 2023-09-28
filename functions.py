#Include functions to graph different metrics

#Imports
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#Create precision-recall curve plot
def plot_prc(name, labels, predictions, **kwargs):
    #Create precision-recall pairs for different probability
    #Takes in true binary labels, and probability estimates positive class
    #returns precision values of prediction and decreasing recall value
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    #create plot from above information with labels of y and x axis
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')

    #add a grid to plot
    plt.grid(True)
    #return current axes
    ax = plt.gca()
    #set axes scaling: equal means aspect=1
    ax.set_aspect('equal')
    #save figure at specific loction as 'plot_prc'
    plt.savefig("Saves/plot_prc")


#create subplots for 'loss', 'prc', 'precision' and 'recall'
def plot_metrics(history, savePath):
  #
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    #name variable used for y-axis label
    name = metric.replace("_"," ").capitalize()

    #layout of subplots that has # rows, # columns, # plot
    # below: 2 rows and 2 columns that be in n+1 plot
    plt.subplot(2,2,n+1)

    #create training data plot line with x-axis being each epoch, y-axis of metric
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    #create validating data plot line with x-axis being each epoch, y-axis of val_metric
    # with lines as dashes   
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    #labels of x and y axis
    plt.xlabel('Epoch')
    plt.ylabel(name)

    #create different y-axis limit/range depending on metric
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    #adjust the spacing between each subplots so labels don't overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.4) 
    #add a legend of the two different plot line
    plt.legend()

    #save figure/plot at specific loction as the 'savePath'   
    plt.savefig(savePath)


#create a confusion matrix
def plot_cm(labels, predictions, p=0.5, savePath="SavedModelOutputs/PlotConfusionMatrix"):
  #confusion maxtrix with labels from test images, 
  # predictions of model prediction of images that larger than p=0.5 
  cm = confusion_matrix(labels, predictions > p)
  #plot figure with specific size
  plt.figure(figsize=(5,5))
  ##plt.rcParams.update({'font.size': 16})

  #Plot rectangular data as a color-encoded matrix 
  # takes in Confusion_Matrix, mapping from data vaules to color space with specify color palette, 
  # annotate the data vaule in each cell, and the annotation font size
  sns.heatmap(cm, cmap=sns.color_palette("icefire", as_cmap=True), annot=True, annot_kws={'size': 15}, fmt="d")
  plt.title('CNN Confusion matrix') #title

  #create ticklabels and axis labels
  ax = plt.gca()
  ax.set_xticklabels(['Friendly', 'Unfriendly'])
  ax.set_yticklabels(['Friendly', 'Unfriendly'])
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  #save figure at specific loction as 'plot_prc'
  plt.savefig(savePath)

def plotPredictions(predictionsFile, outputFile):
  # libraries & dataset
  df = pd.read_csv(predictionsFile,sep="\t", header=None, names=["Type","Prediction"])
  ax = sns.boxplot(x='Type', y='Prediction', data=df)
  ax = sns.swarmplot(x='Type', y='Prediction', data=df, color="grey")
  plt.savefig(outputFile)