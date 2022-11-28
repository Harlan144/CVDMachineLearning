import pandas as pd

#read in TSV files
tsv1 = pd.read_csv("BioImageAnalysis/ImageSample1000_Metrics.tsv", sep='\t', )
tsv2 = pd.read_csv("BioImageAnalysis/ImageSample1001to5000_Metrics.tsv", sep='\t')

#combine two TSV files
Output_df = pd.concat([tsv1 , tsv2], )
Output_df.to_csv("BioImageAnalysis/mergeMetrics.csv") #save in right file
#print(Output_df.head(10))
