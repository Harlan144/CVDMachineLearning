import numpy as np
import pandas as pd


classifiedImages =pd.read_csv("BioImageAnalysis/EditedImageAll5000.csv")

imageNames= classifiedImages[["Image Names","Conclusion (5 types listed in drop down)"]] #classification
imageNames["Conclusion"]= imageNames["Conclusion (5 types listed in drop down)"]
imageNames["Conclusion"] = imageNames["Conclusion"].replace({"Definitely problematic": 1, "Probably problematic":2, "Probably okay": 3, "Definitely okay":4, "Gray-scale":0})
#may want to change the ranking system (0-4 with 0 the NO cololr issue)?
imageNames = imageNames.sort_values(by=["Image Names"])
#print(imageNames.head(10))

df= pd.read_csv("BioImageAnalysis/mergeMetrics.csv")
#print(df['image_file_path'].head)

df[['firstPart','secondPart', 'image_file_path']] = df['image_file_path'].str.split("/", expand=True)
#what is the 'firstPart' & 'secondPart'?
print("Second:")
print(df.head)

newDF = df.loc[df['image_file_path'].isin(imageNames["Image Names"])]
print("Third:")
print(newDF.head)