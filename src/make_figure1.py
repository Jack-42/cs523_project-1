import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import numpy as np


def add_stats(df: pd.DataFrame, axs, first_plot=True):
    ax_epicells = axs[0]
    ax_vload_tcell = axs[1]
    ax_epicells.set_ylabel('Epithelial Cell Count')
    ax_vload_tcell.set_ylabel('Count')
    colors1 = ['blue', 'red', 'orange', 'black']
    labels1 = ['Incubating', 'Expressing', 'Apoptotic', 'Dead']
    colors2 = ['green', 'orange']
    labels2 = ['T-Cells in Tissue', 'Virions']
    plot_subplot(ax_epicells, df, ['incb', 'expr', 'apop', 'dead'],
                 labels=labels1, colors=colors1, title='',
                 first_plot=first_plot)
    plot_subplot(ax_vload_tcell, df, ['ttis', 'virs'], labels=labels2,
                 colors=colors2, title='', first_plot=first_plot)
    return None


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


def main(data_dir: str, save_pth: str = None):
    assert os.path.exists(data_dir)
    style.use('fivethirtyeight')
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rcParams['figure.dpi'] = 200
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    first_plot = True
    for run_dir in os.listdir(data_dir):
        fpath = os.path.join(data_dir, run_dir, "simcov.stats")
        stats_df = pd.read_csv(fpath, sep='\t')
        stats_df = stats_df.rename(columns={'# time': 'time'})
        add_stats(stats_df, axs, first_plot)
        first_plot = False  # don't add additional keys, not necessary\

    if save_pth is not None:
        plt.savefig(save_pth)
    plt.show()
    return fig


if __name__:
    data_folder = "../data/simcov/stable"
    save_pth = "../figures/figure1.png"
    main(data_folder, save_pth)
