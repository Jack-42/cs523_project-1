import numpy as np
import matplotlib.pyplot as plt


def gen_log_map(n: int, r: float, y0: float = 0.5) -> np.array:
    """
    Generate array containing values for logistic map
    :param n: int, number of points
    :param r: float, rate
    :param y0: float, initial value
    :return: x, y where x is an array of time-vals and y is the population-vals
    """
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = 0, y0
    for i in range(n - 1):
        y[i + 1] = r * y[i] * (1 - y[i])
        x[i + 1] = i + 1
    return x, y


def main():
    # create plot of logistic map
    r = 4.0
    n = 100
    x0 = 0.8
    x, y = gen_log_map(n, r, x0)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
