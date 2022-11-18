import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("BioImageAnalysis/metrics_with_classification.tsv", delimiter="\t")
df.set_index("image_file_path",inplace=True)
df.drop(["firstPart","secondPart","article_id"], axis=1, inplace=True)

df = df.loc[(df["Conclusion"]==0) | (df["Conclusion"]==3)]
df.dropna(inplace=True)
df["Conclusion"] = df["Conclusion"].replace({0:False,3:True})
y = df.iloc[:,-1]
X = df.iloc[:,:-1]
print(df.head())
print(df.shape)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',class_weight="balanced").fit(X, y)
#print(LR.predict(X.iloc[4400:,:]))
#print(y.iloc[4400:])
print(round(LR.score(X,y), 4))
print("AUROC LR: ", roc_auc_score(y, LR.predict_proba(X)[:, 1]))

SVM = svm.SVC(class_weight="balanced") #Always predict False
SVM.fit(X, y)
svmPredict = SVM.predict(X)

print(round(SVM.score(X,y), 4))
print("AUROC SVM: ", roc_auc_score(y, svmPredict))

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0,class_weight="balanced")
RF.fit(X, y)
rfPred = RF.predict(X)
#print(y.iloc[4400:])
print(round(RF.score(X,y), 4))
print("AUROC RF: ", roc_auc_score(y, RF.predict_proba(X)[:, 1]))
ConfusionMatrixDisplay.from_predictions(y, rfPred)
plt.savefig("RFConfusionMatrix")