import numpy as np
import pandas as pd


classifiedImages =pd.read_csv("BioImageAnalysis/EditedImageAll5000.csv")
imageNames= classifiedImages[["Image Names","Conclusion (5 types listed in drop down)"]] #classification
imageNames["Conclusion"]= imageNames["Conclusion (5 types listed in drop down)"]
imageNames["Conclusion"].replace({"Definitely problematic": 1, "Probably problematic":2, "Probably okay": 3 "Definitely okay":4, "Gray-scale":0})
imageNames = imageNames.sort_values(by=["Image Names"])
df= pd.read_csv("BioImageAnalysis/eLife_Metrics.csv", delimiter='\t')
print(df['image_file_path'].head)

df[['firstPart','secondPart', 'image_file_path']] = df['image_file_path'].str.split("/", expand=True)

newDF = df.loc[df['image_file_path'].isin(imageNames["Image Names"])]
newDF.join(imageNames["Conclusion"]) #does this work?
print(newDF.shape)
dfWithoutGray = newDF.loc(newDF["Conclusion"]!= 0)
#newDF.to_csv("BioImageAnalysis/metrics_with_classification.csv")
