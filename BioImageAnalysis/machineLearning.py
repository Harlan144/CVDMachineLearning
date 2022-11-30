"""
Used Sklearn to attempt different simple machine learning algorithms on our metrics_with_classification.tsv.
Attempted to see if eucledian distance, contrast, number of high ratios, etc 
could predict whether an image was Colorblind Friendly or Unfriendly.
"""

import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Read and clean data. 
df = pd.read_csv("BioImageAnalysis/metrics_with_classification.tsv", delimiter="\t")
df.set_index("image_file_path",inplace=True)
df.drop(["firstPart","secondPart","article_id"], axis=1, inplace=True)

df = df.loc[(df["Conclusion"]==0) | (df["Conclusion"]==3)]
df.dropna(inplace=True)
df["Conclusion"] = df["Conclusion"].replace({0:False,3:True})
#y refers to df["Conclusion"]
y = df.iloc[:,-1]
#X refers to all other applicable columns.
X = df.iloc[:,:-1]
#print(df.head())
#print(df.shape)

with open("BioImageAnalysis Metrics", "w") as f:
    #Logistic regression
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',class_weight="balanced").fit(X, y)
    f.write(f"Logistic Regression accuracy is {round(LR.score(X,y), 4)}.\n")
    #print(LR.predict(X.iloc[4400:,:]))
    #print(y.iloc[4400:])
    f.write(f"Logistic Regression AUROC is {roc_auc_score(y, LR.predict_proba(X)[:, 1])}.\n\n")


    #Support vector classifier
    SVM = svm.SVC(class_weight="balanced")
    SVM.fit(X, y)
    svmPredict = SVM.predict(X)

    f.write(f"SVC accuracy is {round(SVM.score(X,y), 4)}.\n")
    f.write(f"SVC AUROC is {roc_auc_score(y, svmPredict)}.\n\n")
    
    #Random Foresst Classifier
    RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0,class_weight="balanced")
    RF.fit(X, y)
    rfPred = RF.predict(X)
    #print(y.iloc[4400:])

    f.write(f"Random Forest accuracy is {round(RF.score(X,y), 4)}.\n")
    f.write(f"Random Foresst AUROC is {roc_auc_score(y, RF.predict_proba(X)[:, 1])}.\n\n")


#Save our randomForest predictions as a confusion matrix. 
#ConfusionMatrixDisplay.from_predictions(y, rfPred, cmap=sns.color_palette("icefire", as_cmap=True))
#plt.savefig("RFConfusionMatrix")