import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import numpy as np


def add_stats(dfs: list, ax, first_plot=True):
    colors1 = ['purple', 'orange']
    labels1 = ['Infected Cells', 'Virions']
    for df in dfs:
        df['tot_infected'] = df['incb'] + df['expr']
        plot_subplot(ax, df, ['tot_infected', 'virs'], labels=labels1,
                     colors=colors1, title='', first_plot=first_plot, alpha=0.6)
        first_plot = False
    return None


def main(dfs1: list, dfs2: list, dfs3: list, save_pth: str = None):
    # expects stats from 3 config files, each having multiple runs
    style.use('fivethirtyeight')

    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rcParams['figure.dpi'] = 300
    fig, axs = plt.subplots(1, 3, figsize=(8, 4),
                            constrained_layout=True, sharey='row')
    dfs = [dfs1, dfs2, dfs3]
    titles = ['stable', 'periodic', 'chaotic']
    for idx, ax in enumerate(axs):
        add_stats(dfs[idx], ax)
        ax.set_title(titles[idx])
    axs[0].set_ylabel("Count")
    axs[1].set_xlabel("Time (Days)")
    max_val = 10e6
    for ax in axs:
        ax.set_yscale('log')
        ax.set_ylim(0.5, 10 * np.max(max_val))
    if save_pth is not None:
        plt.savefig(save_pth)
        print("Successfully saved figure to:", save_pth)
    plt.show()
    return fig


def plot_subplot(ax, df, columns, title, colors, labels, alpha=1.0,
                 log_scale=True, first_plot=True):
    xs = df['time'].map(lambda x: x / 1440)  # convert from mins -> days
    for i, col in enumerate(columns):
        ys = df[col]
        if log_scale:
            ys = ys.map(lambda y: y + 1)  # include 0-vals
        if first_plot:
            ax.plot(xs, ys, label=labels[i], lw=1, alpha=alpha, color=colors[i])
        else:
            ax.plot(xs, ys, lw=1, alpha=alpha, color=colors[i])
    if not first_plot:
        return None
    ax.legend(loc='upper right')
    ax.set_title(title)
    xticks = ax.get_xticks()
    if xticks[1] - xticks[0] > 1 and len(xs) > 0:
        ax.set_xticks(range(0, round(max(xs)) + 1, 5))
    return None


def get_dfs(data_dir: str, cutoff: int = None):
    dfs = []
    for run_dir in os.listdir(data_dir):
        fpath = os.path.join(data_dir, run_dir, "simcov.stats")
        stats_df = pd.read_csv(fpath, sep='\t')
        stats_df = stats_df.rename(columns={'# time': 'time'})
        if cutoff is not None:
            # only include interesting part of graph for figure 4
            stats_df = stats_df.loc[stats_df['time'] > cutoff]
        dfs.append(stats_df)
    return dfs


if __name__:
    data_dir1 = "../data/simcov/stable"
    data_dir2 = "../data/simcov/periodic"
    data_dir3 = "../data/simcov/chaotic"
    save_pth = "../figures/figure2.png"
    # cutoff = 1440 * 15  # for fig 4, interesting dynamics start ~ day 15
    cutoff = None
    dfs_1 = get_dfs(data_dir1, cutoff)
    dfs_2 = get_dfs(data_dir2, cutoff)
    dfs_3 = get_dfs(data_dir3, cutoff)

    main(dfs_1, dfs_2, dfs_3, save_pth)
