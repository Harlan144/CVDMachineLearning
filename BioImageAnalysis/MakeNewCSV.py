import numpy as np
import pandas as pd


classifiedImages =pd.read_csv("")
#classifiedImages["ImageNames"] = classifiedImages["ImageNames"].str.split("/", expand=False)
imageNames= classifiedImages[["ImageNames","classification"]] #classification

df= pd.read_csv("eLife_Metrics.csv")
df['image_file_path']= df['image_file_path'].str.split("/", expand=False)
newDF = df.loc[df['image_file_path'].isin(classifiedImages["ImageNames"])]

newDF.join(imageNames["classification"]) #does this work?
newDF.to_csv("metrics_with_classification.csv")
