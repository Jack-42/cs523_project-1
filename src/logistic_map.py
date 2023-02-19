import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style

"""
This file contains functions related to generating, plotting, and binning/saving
data from the logistic map.
"""


def gen_log_map(t_steps: int, r: float, y0: float = 0.5) -> np.array:
    """
    Generate array containing values for logistic map
    :param t_steps: int, time steps/points
    :param r: float, rate
    :param y0: float, initial value
    :return: t, y where t is an array of time-vals and y is the population-vals
    """
    t = np.zeros(t_steps)
    y = np.zeros(t_steps)
    t[0], y[0] = 0, y0
    for i in range(t_steps - 1):
        y[i + 1] = r * y[i] * (1 - y[i])
        t[i + 1] = i + 1
    return t, y


def create_dual_plot(t_steps: int, r: float, yi_1: float, yi_2: float, ax):
    """
    Create plot of logistic map for two different initial values
    :param t_steps: int, number of time steps/points
    :param r: float, rate
    :param yi_1: float, first initial value
    :param yi_2: float,  second initial value
    :param ax: matplotlib axis
    :return: None
    """
    t1, y1 = gen_log_map(t_steps, r, yi_1)
    t2, y2 = gen_log_map(t_steps, r, yi_2)
    ax.plot(t1, y1, 'r', lw=1, marker='.', alpha=0.5, label="$x_{0}$=%f" % yi_1)
    ax.plot(t2, y2, 'b', lw=1, marker='.', alpha=0.5, label="$x_{0}$=%f" % yi_2)
    return None


def _helper_save(df: pd.DataFrame, n: int, data_dir: str):
    pth1 = os.path.join(data_dir, "full_data.txt")
    df.to_csv(pth1, header=None, index=None, sep=' ', mode='a')
    pth2 = os.path.join(data_dir, "first_n=%d_data.txt" % n)
    df.iloc[0:n].to_csv(pth2, header=None, index=None, sep=' ', mode='a')
    pth3 = os.path.join(data_dir, "remaining_data.txt")
    df.iloc[n:].to_csv(pth3, header=None, index=None, sep=' ', mode='a')


def save_data(r_vals: list, y0_vals: list, t_steps: int, n: int,
              data_dir: str):
    """
    Method used to save data to .txt files for analysis from JIDT tool.
    Will save 3 .txt files:
        1) Full logistic maps generated from r_vals and y0_vals
        2) First n-values from each map
        3) Last n-values from each map
    :param r_vals: list(float), list of r-values to use for logistic map
    :param y0_vals: list(float), list of initial values to use for logistic map
    :param t_steps: int, number of time steps to use for each map
    :param n: int, cutoff-value for chaotic behavior (determined from graph)
    :param data_dir: str, path to data directory to save .txt files
    :return: None
    """
    assert len(r_vals) == len(y0_vals)
    data = []
    for r, y0 in zip(r_vals, y0_vals):
        _, series = gen_log_map(t_steps, r, y0)
        data.append(series)
    df = pd.DataFrame(data).T
    # save complete data
    _helper_save(df, n, os.path.join(data_dir, "original"))
    # bin data
    n_bins = 10
    binned_data = []
    labels = np.arange(n_bins)
    for series in data:
        binned_series = pd.cut(series, bins=n_bins, labels=labels)
        binned_data.append(binned_series)
    df_bin = pd.DataFrame(binned_data)
    _helper_save(df_bin, n, os.path.join(data_dir, "binned"))
    print("Successfully saved data to:", data_dir)
    return None


def main(save_pth: str = None):
    # r=3.800918828, x0=0.10, 0.12 is weird in WolframAlpha
    # periodic: r = 3.2
    # chaotic: r = 3.75, diverges around t=37
    style.use('fivethirtyeight')
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=12)
    plt.rcParams['figure.dpi'] = 300
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), constrained_layout=True,
                            sharex='all', sharey='all')
    axs[1].set_xlabel('timestep')
    for ax in axs:
        ax.set_ylabel('population')

    r1 = 3.2
    r2 = 3.75
    n = 100
    eps = 0.000001
    yi_1 = 0.10
    yi_2 = yi_1 + eps
    create_dual_plot(n, r1, yi_1, yi_2, axs[0])
    create_dual_plot(n, r2, yi_1, yi_2, axs[1])
    axs[0].legend(loc="lower right")
    if save_pth is not None:
        plt.savefig(save_pth)
    plt.show()


def save_main(data_dir: str):
    """
    Main to cache data used in plotting
    :return:
    """
    r1 = 3.2
    r2 = 3.75
    r_vals = [r1, r1, r2, r2]
    eps = 0.000001
    yi_1 = 0.10
    yi_2 = yi_1 + eps
    y0_vals = [yi_1, yi_2, yi_1, yi_2]
    n = 37  # determined by looking at graph of r=3.75 w/ these initial vals
    save_data(r_vals, y0_vals, t_steps=100, n=n, data_dir=data_dir)


if __name__ == "__main__":
    # data_folder = "../data/logistic_map"
    # assert os.path.exists(data_folder)
    # save_main(data_folder)
    save_pth = "../figures/figure3.png"
    main(save_pth)
