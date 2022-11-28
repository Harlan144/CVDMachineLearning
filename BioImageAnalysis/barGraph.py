import matplotlib.pyplot as plt
import pandas as pd

#read in TSV file
df = pd.read_csv("BioImageAnalysis/EditedImage5000class.tsv", sep='\t')

#Get total number from 'conclusion' column
total = len(df["Conclusion"])

#Percentage of each kind of 'conclusion' to be use in bar plot
counts = df["Conclusion"].value_counts()
percentage = round((counts/total)*100,1)
print(percentage)

#Create bar graph
fig,ax = plt.subplots()
percentage.plot(ax=ax, kind='bar')
ax.set_ylim(0,100) #change y-axis to go from 0-100
plt.xlabel("Conclusions")
plt.ylabel("Percentage of Total (" + str(total) + ")")
plt.title("Colorblind Images Percentages")
plt.tight_layout() #fix x-axis so can see label

#Code to label each bar on bar graph 
for p in ax.patches:
    ax.annotate(
        str(p.get_height()), xy=(p.get_x() + 0.1, p.get_height() + 0.45), fontsize=12 
    )

plt.savefig("barGraph.png") #save file within CVDMachineLearning
