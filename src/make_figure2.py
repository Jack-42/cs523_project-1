import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import numpy as np


def main(dfs: list):
    # expects stats from 3 config files
    assert len(dfs) == 3
    style.use('fivethirtyeight')

    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rcParams['figure.dpi'] = 200
    fig, axs = plt.subplots(1, len(dfs), figsize=(8, 4),
                            constrained_layout=True)
    colors1 = ['purple', 'pink']
    labels1 = ['Infected Cells', 'Virions']
    for idx, ax in enumerate(axs):
        df = dfs[idx]
        df['tot_infected'] = df['incb'] + df['expr']
        ax.set_ylabel("Count")
        plot_subplot(ax, df, ['tot_infected', 'virs'], labels=labels1,
                     colors=colors1, title='')

    plt.show()
    return fig


def plot_subplot(ax, df, columns, title, colors, labels, alpha=1.0,
                 log_scale=True):
    xs = df['time'].map(lambda x: x / 1440)  # convert from mins -> days
    for i, col in enumerate(columns):
        ys = df[col]
        if log_scale:
            ys = ys.map(lambda y: y + 1)  # include 0-vals
        ax.plot(xs, ys, label=labels[i], lw=2, alpha=alpha, color=colors[i])
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (days)')
    ax.set_title(title)
    xticks = ax.get_xticks()
    if xticks[1] - xticks[0] > 1 and len(xs) > 0:
        ax.set_xticks(range(0, round(max(xs)) + 1, 5))
    if log_scale:
        ax.set_yscale('log')
        max_val = df[columns].max(axis=1)
        ax.set_ylim(0.5, 10 * np.max(max_val))
    return None


if __name__:
    fpath = "../data/simcov/stats_files/simcov.stats"
    assert os.path.exists(fpath)
    stats_df = pd.read_csv(fpath, sep='\t')
    stats_df = stats_df.rename(columns={'# time': 'time'})
    # will use 3 configs once we have all of them
    main([stats_df, stats_df, stats_df])
