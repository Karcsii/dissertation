import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

name = "randwrite_hdd_backup"
folder = "C:/Users/Karcsi-Y720/Desktop/dissertation/"

block_size = [1, 2, 4, 8, 16, 32, 64]
parallel_tasks = [1, 2, 4, 8, 16, 32, 64]


def latex_float(float_str):
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return '${base}e^{exp}$'.format(base=base, exp=int(exponent))
    else:
        return float_str


# throughput
df = pd.read_csv("{folder}{name}/throughputnumber.csv".format(name=name, folder=folder), sep='\t', index_col=0)
labels = df.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(df, linewidth=.5, fmt='', annot=labels)
axs.collections[0].colorbar.set_label("Throughput (MiB/s)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/throughputnumber.png'.format(name=name))
plt.clf()

# latency
df = pd.read_csv("{folder}{name}/latency.csv".format(name=name, folder=folder), sep='\t', index_col=0)
labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(df, linewidth=.5, fmt='', annot=labels, cmap="crest")
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
print(labels)
# axs.collections[0].colorbar.set_ticklabels([x for x in labels])
axs.collections[0].colorbar.set_label("Latency (μs)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/latency.png'.format(name=name))
plt.clf()

# iops
df = pd.read_csv("{folder}{name}/iops.csv".format(name=name, folder=folder), sep='\t', index_col=0)
labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(df, linewidth=.5, fmt='', annot=labels, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True))
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
# axs.collections[0].colorbar.set_ticklabels([int(x) for x in labels])
axs.collections[0].colorbar.set_label("IOPS")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/iops.png'.format(name=name))
plt.clf()

powerssd = {}
powerhdd = {}
powermemory = {}
powertotal = {}
powerssdparallel = {}
powerhddparallel = {}
powermemoryparallel = {}
powertotalparallel = {}

# power
df = pd.read_csv("{folder}{name}/power.csv".format(name=name, folder=folder), sep='\t', header=None)
runs = df.index[df[0].str.startswith('run')].tolist()
runs.append(df[df.columns[0]].count())
for idx, x in enumerate(runs[:-1]):
    df2 = df[x + 1: runs[idx + 1]]
    df2.index = np.arange(len(df2))
    new_header = df2.iloc[0].tolist()
    df2 = df2[1:]  # take the data less the header row
    df2.columns = new_header  # set the header row as the df header
    df2 = df2.apply(pd.to_numeric)
    powerssdparallel[block_size[idx % 7]] = df2.loc[:, 'Power PCIe'].mean()
    powerhddparallel[block_size[idx % 7]] = df2.loc[:, 'Power SATA'].mean()
    powermemoryparallel[block_size[idx % 7]] = df2.loc[:, 'Power Memory'].mean()
    powertotalparallel[block_size[idx % 7]] = df2.loc[:, 'Power PSU'].mean()
    if (idx % 7) == 6:
        powerssd[block_size[idx // 7]] = powerssdparallel
        powerhdd[block_size[idx // 7]] = powerhddparallel
        powermemory[block_size[idx // 7]] = powermemoryparallel
        powertotal[block_size[idx // 7]] = powertotalparallel
        powerssdparallel = {}
        powerhddparallel = {}
        powermemoryparallel = {}
        powertotalparallel = {}
dftotal = pd.DataFrame(powertotal, columns=block_size, index=parallel_tasks)
dfssd = pd.DataFrame(powerssd, columns=block_size, index=parallel_tasks)
dfhdd = pd.DataFrame(powerhdd, columns=block_size, index=parallel_tasks)
dfmemory = pd.DataFrame(powermemory, columns=block_size, index=parallel_tasks)

labels = dftotal.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(dftotal, linewidth=.5, fmt='', annot=labels, cmap=sns.cubehelix_palette(rot=-0.4, as_cmap=True, reverse=False))
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
axs.collections[0].colorbar.set_label("Total Power (W)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/powertotal.png'.format(name=name))
plt.clf()

labels = dfssd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(dfssd, linewidth=.5, fmt='', annot=labels,
                  cmap=sns.cubehelix_palette(rot=-0.1, as_cmap=True, reverse=False))
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
axs.collections[0].colorbar.set_label("SSD Power (W)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/powerssd.png'.format(name=name))
plt.clf()

labels = dfhdd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(dfhdd, linewidth=.5, fmt='', annot=labels,
                  cmap=sns.cubehelix_palette(rot=-0.1, as_cmap=True, reverse=False))
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
axs.collections[0].colorbar.set_label("HDD Power (W)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/powerhdd.png'.format(name=name))
plt.clf()

labels = dfmemory.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
labels = labels.applymap(lambda x: latex_float(x))
axs = sns.heatmap(dfmemory, linewidth=.5, fmt='', annot=labels,
                  cmap=sns.cubehelix_palette(rot=0, start=1.3, as_cmap=True, reverse=False))
labels = [item.get_text() for item in axs.collections[0].colorbar.ax.get_yticklabels()]
axs.collections[0].colorbar.set_label("Memory Power (W)")
axs.set_ylabel("Parallel Tasks")
axs.set_xlabel("Block Size")
fig = axs.get_figure()
fig.savefig('./{name}/powermemory.png'.format(name=name))
plt.clf()
# power SSD

# power HDD

# power memory
