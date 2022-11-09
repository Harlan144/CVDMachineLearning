import pandas as pd

tsv1 = pd.read_csv("BioImageAnalysis/ImageSample1000_Metrics.tsv", sep='\t', )
tsv2 = pd.read_csv("BioImageAnalysis/ImageSample1001to5000_Metrics.tsv", sep='\t')

Output_df = pd.merge(tsv1, tsv2, on='article_id', how='inner')
Output_df.to_csv("BioImageAnalysis/Fix_eLife_metrics.csv")
