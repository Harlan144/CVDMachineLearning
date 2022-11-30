import numpy as np
import pandas as pd

#Add a column with the image classificaton (Definitely Okay, Definetly probematic, etc) onto eLife_Metrics.csv.

#Read in our file with image classifications
classifiedImages =pd.read_csv("BioImageAnalysis/EditedImage5000class.tsv", delimiter="\t")

imageNames= classifiedImages[["Image Names","Conclusion"]] #classification

#Format the df so that conclusions are numeric (easier to run regression on)
imageNames["Conclusion"] = imageNames["Conclusion"].replace({"Definitely problematic": 3, "Probably problematic":2, "Probably okay": 1, "Definitely okay":0, "Gray-scale":-1})
#Remove jpg ending from df so we can merge dfs.
imageNames["Image Names"]= imageNames["Image Names"].str.replace(".jpg","")

#Read in second df with Dr. Piccolo's calculated image metrics
df= pd.read_csv("BioImageAnalysis/eLife_Metrics.csv", delimiter="\t")


df[['firstPart','secondPart', 'image_file_path']] = df['image_file_path'].str.split("/", expand=True)
df['image_file_path'] = df['image_file_path'].str.replace(".jpg", "")

#create a newDF where the image_file_path is in our image classification.
newDF = df.loc[df['image_file_path'].isin(imageNames["Image Names"])]


imageNames.set_index("Image Names", inplace=True)
newDF.set_index("image_file_path", inplace=True)

#Add the conclusion column to our new DF, save it as a tsv: metrics_with_classification.tsv
newDF = newDF.join(imageNames["Conclusion"])

newDF.to_csv("BioImageAnalysis/metrics_with_classification.tsv", sep="\t")