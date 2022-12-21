import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("BioImageAnalysis/metrics_with_classification.tsv", delimiter="\t")
df.set_index("image_file_path",inplace=True)
df.drop(["firstPart","secondPart","article_id"], axis=1, inplace=True)

withoutGray = df.loc[df["Conclusion"]!=-1]
print(sum(withoutGray["is_rgb"] ==0))
#Why are there still some left that are "is_rgb"==0?

corrMatrix = withoutGray.corr()
corrMatrix.to_csv("BioImageAnalysis/corrMatrix")


cols =withoutGray.columns

fig = plt.figure(figsize=(12,12))  # figure so we can add axis
ax = fig.add_subplot(111)  # define axis, so we can modify
ax.matshow(corrMatrix)  # display the matrix
ax.set_xticks(np.arange(len(cols)))  # show them all!
ax.set_yticks(np.arange(len(cols)))  # show them all!
ax.set_xticklabels(cols)  # set to be the abbv (vs useless #)
ax.set_yticklabels(cols)  # set to be the abbv (vs useless #)

#plt.matshow(corrMatrix)

plt.savefig("corrMatrix")

plt.figure(figsize=(10, 10))
for i, label in enumerate(cols[:-1]):
    corr=df[label].corr(df['Conclusion'])
    ax = plt.subplot(3, 3, i + 1)
    plt.scatter(df["Conclusion"],df[label])
    plt.title(str(label)+" corr: "+str(round(corr,3)))
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.savefig("scatterPlots.png")



