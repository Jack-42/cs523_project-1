import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import pandas as pd
import os


def gen_log_map(n: int, r: float, y0: float = 0.5) -> np.array:
    """
    Generate array containing values for logistic map
    :param n: int, number of points
    :param r: float, rate
    :param y0: float, initial value
    :return: t, y where t is an array of time-vals and y is the population-vals
    """
    t = np.zeros(n)
    y = np.zeros(n)
    t[0], y[0] = 0, y0
    for i in range(n - 1):
        y[i + 1] = r * y[i] * (1 - y[i])
        t[i + 1] = i + 1
    return t, y


def create_dual_plot(n: int, r: float, yi_1: float, yi_2: float):
    """
    Create plot of logistic map for two different initial values
    :param n: int, number of points
    :param r: float, rate
    :param yi_1: float, first initial value
    :param yi_2: float,  second initial value
    :return: None
    """
    t1, y1 = gen_log_map(n, r, yi_1)
    t2, y2 = gen_log_map(n, r, yi_2)
    figure(figsize=(10, 6), dpi=160)
    plt.xlabel("timestep")
    plt.ylabel("population")
    plt.plot(t1, y1, 'r', marker='.', alpha=0.5, label="y0=%f" % yi_1)
    plt.plot(t2, y2, 'b', marker='.', alpha=0.5, label="y0=%f" % yi_2)
    plt.legend(loc="lower right")
    plt.show()


def save_data(r_vals: list, y0_vals: list, n: int, save_path: str):
    data = []
    for r, y0 in zip(r_vals, y0_vals):
        _, series = gen_log_map(n, r, y0)
        data.append(series)
    df = pd.DataFrame(data).T
    df.to_csv(save_path, header=None, index=None, sep=' ', mode='a')
    return


def main():
    # r=3.800918828, x0=0.10, 0.12 is weird in WolframAlpha
    # periodic: r = 3.2
    # chaotic: r = 3.75
    r = 3.75
    n = 100
    eps = 0.000001
    yi_1 = 0.10
    yi_2 = yi_1 + eps
    create_dual_plot(n, r, yi_1, yi_2)


def save_main(save_path: str):
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
    save_data(r_vals, y0_vals, n=100, save_path=save_path)


if __name__ == "__main__":
    data_dir = "../data/logistic_map"
    assert os.path.exists(data_dir)
    save_pth = os.path.join(data_dir, "data.txt")
    save_main(save_pth)
