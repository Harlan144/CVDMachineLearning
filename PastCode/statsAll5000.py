import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

#read in the CSV file
tsv2 = pd.read_csv("BioImageAnalysis/EditedImage5000class.tsv", sep="\t")

#convert CSV file to a datafame
df = pd.DataFrame(tsv2)
#print(df)

#Define X and Y for the graph
#X = list(df.iloc[:, 5])
#print(X)

X = tsv2.iloc[:, :6]
print(X)
Y = tsv2['Conclusion']
print(Y)

graph = Y.value_counts().sort_index().plot.bar(X='Value', Y = 'Occurrences')
