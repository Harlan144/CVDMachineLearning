import numpy as np
import pandas as pd


classifiedImages =pd.read_csv("BioImageAnalysis/EditedImage5000class.tsv", delimiter="\t")

imageNames= classifiedImages[["Image Names","Conclusion"]] #classification

imageNames["Conclusion"] = imageNames["Conclusion"].replace({"Definitely problematic": 3, "Probably problematic":2, "Probably okay": 1, "Definitely okay":0, "Gray-scale":-1})
imageNames["Image Names"]= imageNames["Image Names"].str.replace(".jpg","")

df= pd.read_csv("BioImageAnalysis/eLife_Metrics.csv", delimiter="\t")

df[['firstPart','secondPart', 'image_file_path']] = df['image_file_path'].str.split("/", expand=True)
df['image_file_path'] = df['image_file_path'].str.replace(".jpg", "")

newDF = df.loc[df['image_file_path'].isin(imageNames["Image Names"])]

imageNames.set_index("Image Names", inplace=True)
newDF.set_index("image_file_path", inplace=True)

newDF = newDF.join(imageNames["Conclusion"])

newDF.to_csv("BioImageAnalysis/metrics_with_classification.tsv", sep="\t")