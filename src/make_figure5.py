import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import numpy as np


def add_stats(df: pd.DataFrame, axs, first_plot=True):
    ax_casesvdeaths = axs
    ax_casesvdeaths.set_ylabel('Count')
    colors1 = ['red', 'black']
    labels1 = ['Cases', 'Deaths']

    plot_subplot(ax_casesvdeaths, df, ['Cases', 'Deaths'], labels=labels1,
                 colors=colors1,
                 title='', first_plot=first_plot)
    return None


def plot_subplot(ax, df, columns, title, colors, labels, alpha=1.0,
                 log_scale=True, first_plot=True):
    xs = df['day']  # convert from mins -> days
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
    ax.set_xlabel('Time (days)', fontdict={"size": 12})
    ax.set_title(title)
    xticks = ax.get_xticks()
    if log_scale:
        ax.set_yscale('log')
        max_val = df[columns].max(axis=1)
        ax.set_ylim(0.5, 10 * np.max(max_val))
    return None


def main(data_dir: str, save_pth: str = None):
    assert os.path.exists(data_dir)
    style.use('fivethirtyeight')
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=12)
    plt.rcParams['figure.dpi'] = 300
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    first_plot = True

    fpath = os.path.join(data_dir, "new_cases.csv")
    fpath1 = os.path.join(data_dir, "new_deaths.csv")
    stats_df = pd.read_csv(fpath)
    stats_df1 = pd.read_csv(fpath1)

    df = pd.DataFrame(stats_df["World"])

    df1 = pd.DataFrame(stats_df1["World"])

    cases_df = df.rename(columns={'World': 'Cases'})['Cases']
    deaths_df = df1.rename(columns={'World': 'Deaths'})['Deaths']

    newdf = pd.DataFrame([cases_df, deaths_df]).T

    newdf.insert(0, "day", np.arange(0, len(newdf)))
    print(newdf.head())

    add_stats(newdf, axs, first_plot)

    if save_pth is not None:
        plt.savefig(save_pth)
    plt.show()
    return fig


if __name__:
    data_folder = "../data/real_world"
    save_pth = "../figures/figure5.png"
    main(data_folder, save_pth)
