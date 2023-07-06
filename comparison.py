import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test = ["randwrite", "randread", "seqwrite", "seqread"]
folder = "C:/Users/Karcsi-Y720/Desktop/dissertation/"

block_size = [1, 2, 4, 8, 16, 32, 64]
parallel_tasks = [1, 2, 4, 8, 16, 32, 64]


def latex_float(float_str):
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return '${base}e^{exp}$'.format(base=base, exp=int(exponent))
    else:
        return float_str


for name in test:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches([12.8, 4.8])
    # throughput
    df = pd.read_csv("{folder}{name}_ssd_backup/throughputnumber.csv".format(name=name, folder=folder), sep='\t', index_col=0)
    labels = df.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(df, linewidth=.5, fmt='', annot=labels, ax=ax1)
    ax1.collections[0].colorbar.set_label("Throughput (MiB/s)")
    ax1.set_ylabel("Parallel Tasks")
    ax1.set_xlabel("Block Size")

    df = pd.read_csv("{folder}{name}_hdd_backup/throughputnumber.csv".format(name=name, folder=folder), sep='\t',
                     index_col=0)
    labels = df.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(df, linewidth=.5, fmt='', annot=labels, ax=ax2)
    ax2.collections[0].colorbar.set_label("Throughput (MiB/s)")
    ax2.set_ylabel("Parallel Tasks")
    ax2.set_xlabel("Block Size")

    fig.savefig('./comparison/{name}_throughputnumber.png'.format(name=name))
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches([12.8, 4.8])
    # latency
    df = pd.read_csv("{folder}{name}_ssd_backup/latency.csv".format(name=name, folder=folder), sep='\t', index_col=0)
    labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(df, linewidth=.5, fmt='', annot=labels, cmap="crest", ax=ax1)
    labels = [item.get_text() for item in ax1.collections[0].colorbar.ax.get_yticklabels()]
    print(labels)
    # axs.collections[0].colorbar.set_ticklabels([x for x in labels])
    ax1.collections[0].colorbar.set_label("Latency (μs)")
    ax1.set_ylabel("Parallel Tasks")
    ax1.set_xlabel("Block Size")

    df = pd.read_csv("{folder}{name}_hdd_backup/latency.csv".format(name=name, folder=folder), sep='\t', index_col=0)
    labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(df, linewidth=.5, fmt='', annot=labels, cmap="crest", ax=ax2)
    labels = [item.get_text() for item in ax2.collections[0].colorbar.ax.get_yticklabels()]
    print(labels)
    # axs.collections[0].colorbar.set_ticklabels([x for x in labels])
    ax2.collections[0].colorbar.set_label("Latency (μs)")
    ax2.set_ylabel("Parallel Tasks")
    ax2.set_xlabel("Block Size")

    fig.savefig('./comparison/{name}_latency.png'.format(name=name))
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches([12.8, 4.8])
    # iops
    df = pd.read_csv("{folder}{name}_ssd_backup/iops.csv".format(name=name, folder=folder), sep='\t', index_col=0)
    labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(df, linewidth=.5, fmt='', annot=labels, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True), ax=ax1)
    labels = [item.get_text() for item in ax1.collections[0].colorbar.ax.get_yticklabels()]
    # axs.collections[0].colorbar.set_ticklabels([int(x) for x in labels])
    ax1.collections[0].colorbar.set_label("IOPS")
    ax1.set_ylabel("Parallel Tasks")
    ax1.set_xlabel("Block Size")

    df = pd.read_csv("{folder}{name}_hdd_backup/iops.csv".format(name=name, folder=folder), sep='\t', index_col=0)
    labels = df.applymap(lambda x: '{0:.3g}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(df, linewidth=.5, fmt='', annot=labels, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True), ax=ax2)
    labels = [item.get_text() for item in ax2.collections[0].colorbar.ax.get_yticklabels()]
    # axs.collections[0].colorbar.set_ticklabels([int(x) for x in labels])
    ax2.collections[0].colorbar.set_label("IOPS")
    ax2.set_ylabel("Parallel Tasks")
    ax2.set_xlabel("Block Size")

    fig.savefig('./comparison/{name}_iops.png'.format(name=name))
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
    dfssd = pd.read_csv("{folder}{name}_ssd_backup/power.csv".format(name=name, folder=folder), sep='\t', header=None)
    runsssd = dfssd.index[dfssd[0].str.startswith('run')].tolist()
    runsssd.append(dfssd[dfssd.columns[0]].count())
    dfhdd = pd.read_csv("{folder}{name}_hdd_backup/power.csv".format(name=name, folder=folder), sep='\t', header=None)
    runshdd = dfhdd.index[dfhdd[0].str.startswith('run')].tolist()
    runshdd.append(dfhdd[dfhdd.columns[0]].count())
    for idx, x in enumerate(runsssd[:-1]):
        df2 = dfssd[x + 1: runsssd[idx + 1]]
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
    dftotalssd = pd.DataFrame(powertotal, columns=block_size, index=parallel_tasks)
    dfssdssd = pd.DataFrame(powerssd, columns=block_size, index=parallel_tasks)
    dfhddssd = pd.DataFrame(powerhdd, columns=block_size, index=parallel_tasks)
    dfmemoryssd = pd.DataFrame(powermemory, columns=block_size, index=parallel_tasks)
    for idx, x in enumerate(runshdd[:-1]):
        df2 = dfhdd[x + 1: runshdd[idx + 1]]
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
    dftotalhdd = pd.DataFrame(powertotal, columns=block_size, index=parallel_tasks)
    dfssdhdd = pd.DataFrame(powerssd, columns=block_size, index=parallel_tasks)
    dfhddhdd = pd.DataFrame(powerhdd, columns=block_size, index=parallel_tasks)
    dfmemoryhdd = pd.DataFrame(powermemory, columns=block_size, index=parallel_tasks)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches([12.8, 4.8])
    labels = dftotalssd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(dftotalssd, linewidth=.5, fmt='', annot=labels,
                      cmap=sns.cubehelix_palette(rot=-0.4, as_cmap=True, reverse=False), ax=ax1)
    labels = [item.get_text() for item in ax1.collections[0].colorbar.ax.get_yticklabels()]
    ax1.collections[0].colorbar.set_label("Total Power (W)")
    ax1.set_ylabel("Parallel Tasks")
    ax1.set_xlabel("Block Size")

    labels = dftotalhdd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(dftotalhdd, linewidth=.5, fmt='', annot=labels,
                cmap=sns.cubehelix_palette(rot=-0.4, as_cmap=True, reverse=False), ax=ax2)
    labels = [item.get_text() for item in ax2.collections[0].colorbar.ax.get_yticklabels()]
    ax2.collections[0].colorbar.set_label("Total Power (W)")
    ax2.set_ylabel("Parallel Tasks")
    ax2.set_xlabel("Block Size")

    fig.savefig('./comparison/{name}_powertotal.png'.format(name=name))
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches([12.8, 4.8])
    labels = dfssdssd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(dfssdssd, linewidth=.5, fmt='', annot=labels,
                      cmap=sns.cubehelix_palette(rot=-0.1, as_cmap=True, reverse=False), ax=ax1)
    labels = [item.get_text() for item in ax1.collections[0].colorbar.ax.get_yticklabels()]
    ax1.collections[0].colorbar.set_label("SSD Power (W)")
    ax1.set_ylabel("Parallel Tasks")
    ax1.set_xlabel("Block Size")

    labels = dfhddhdd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(dfhddhdd, linewidth=.5, fmt='', annot=labels,
                      cmap=sns.cubehelix_palette(rot=-0.1, as_cmap=True, reverse=False), ax=ax2)
    labels = [item.get_text() for item in ax2.collections[0].colorbar.ax.get_yticklabels()]
    ax2.collections[0].colorbar.set_label("HDD Power (W)")
    ax2.set_ylabel("Parallel Tasks")
    ax2.set_xlabel("Block Size")

    fig.savefig('./comparison/{name}_powerdrive.png'.format(name=name))
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches([12.8, 4.8])
    labels = dfmemoryssd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(dfmemoryssd, linewidth=.5, fmt='', annot=labels,
                      cmap=sns.cubehelix_palette(rot=0, start=1.3, as_cmap=True, reverse=False), ax=ax1)
    labels = [item.get_text() for item in ax1.collections[0].colorbar.ax.get_yticklabels()]
    ax1.collections[0].colorbar.set_label("Memory Power (W)")
    ax1.set_ylabel("Parallel Tasks")
    ax1.set_xlabel("Block Size")

    labels = dfmemoryhdd.applymap(lambda x: '{0:.2f}'.format(x).replace('.', ','))
    labels = labels.applymap(lambda x: latex_float(x))
    sns.heatmap(dfmemoryhdd, linewidth=.5, fmt='', annot=labels,
                      cmap=sns.cubehelix_palette(rot=0, start=1.3, as_cmap=True, reverse=False), ax=ax2)
    labels = [item.get_text() for item in ax2.collections[0].colorbar.ax.get_yticklabels()]
    ax2.collections[0].colorbar.set_label("Memory Power (W)")
    ax2.set_ylabel("Parallel Tasks")
    ax2.set_xlabel("Block Size")

    fig.savefig('./comparison/{name}_powermemory.png'.format(name=name))
    plt.clf()
    # power SSD

    # power HDD

    # power memory
