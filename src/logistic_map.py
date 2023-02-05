import numpy as np


def gen_log_map(n: int, r: float, x0: float = 0.5) -> np.array:
    """
    Generate array containing values for logisitic map
    :param n: int, number of points
    :param r: float, rate
    :param x0: float, initial value
    :return: numpy array containing values
    """
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = r * x[i] * (1 - x[i])
    return x
