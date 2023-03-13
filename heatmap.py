import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

name = "blk-seq-read"

def latex_float(float_str):
	if "e" in float_str:
		base, exponent = float_str.split("e")
		return '${base}e^{exp}$'.format(base=base, exp=int(exponent))
	else:
		return float_str

#throughput
df = pd.read_csv("/home/karoly/{name}/throughputnumber.csv".format(name = name), sep='\t', index_col=0)
#print(df.to_string()) 
labels = df.applymap(lambda x: '{0:.2f}'.format(x).replace('.',','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(df, linewidth=.5, fmt = '', annot=labels)
axs.collections[0].colorbar.set_label("Throughput (MiB/s)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/throughputnumber.png'.format(name = name))
plt.clf()
#latency
df = pd.read_csv("/home/karoly/{name}/latency.csv".format(name = name), sep='\t', index_col=0)
#print(df.to_string()) 
labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.',','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(df, linewidth=.5, fmt = '', annot=labels, cmap="crest")


labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
print(labels)
#axs.collections[0].colorbar.set_ticklabels([x for x in labels])
axs.collections[0].colorbar.set_label("Latency (μs)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/latency.png'.format(name = name))
plt.clf()
#iops
df = pd.read_csv("/home/karoly/{name}/iops.csv".format(name = name), sep='\t', index_col=0)
print(df.to_string()) 
labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.',','))
labels = labels.applymap(lambda x: latex_float(x))

print(labels.to_string()) 
axs = sns.heatmap(df, linewidth=.5, fmt = '', annot=labels, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True))
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
print(labels)
axs.collections[0].colorbar.set_ticklabels([int(x) for x in labels])
axs.collections[0].colorbar.set_label("IOPS")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/iops.png'.format(name = name))


