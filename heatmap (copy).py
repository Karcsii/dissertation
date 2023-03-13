import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

name = "blk-seq-read"

df = pd.read_csv("/home/karoly/{name}/throughputnumber.csv".format(name = name), sep='\t', index_col=0)
#print(df.to_string()) 
axs = sns.heatmap(df, annot=True, fmt='.2f', linewidth=.5)
fig = axs.get_figure()
fig.savefig('./{name}/throughputnumber.png'.format(name = name))
plt.clf()

df = pd.read_csv("/home/karoly/{name}/latency.csv".format(name = name), sep='\t', index_col=0)
#print(df.to_string()) 
axs = sns.heatmap(df, annot=True, fmt='.2g', linewidth=.5)
fig = axs.get_figure()
fig.savefig('./{name}/latency.png'.format(name = name))
plt.clf()

df = pd.read_csv("/home/karoly/{name}/iops.csv".format(name = name), sep='\t', index_col=0)
print(df.to_string()) 
labels = df.applymap(lambda x: '{0:.2g}'.format(x))

print(labels.to_string()) 
axs = sns.heatmap(df, linewidth=.5, annot=labels, annot_kws={"useMathText": True})
fig = axs.get_figure()
fig.savefig('./{name}/iops.png'.format(name = name))

