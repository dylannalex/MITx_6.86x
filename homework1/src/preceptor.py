import numpy as np


def preceptor(
    x: np.ndarray,
    y: np.ndarray,
    cycles: int,
    theta: np.ndarray = None,
    theta0: float = None,
):
    if theta is None:
        theta = np.zeros(x.shape[1])
    if theta0 is None:
        theta0 = 0
    for cycle in range(cycles):
        mistakes = 0
        for i in range(len(x)):
            if y[i] * (np.dot(theta, x[i]) + theta0) <= 0:
                mistakes += 1
                theta += y[i] * x[i]
                theta0 += y[i]
        if not mistakes:
            return theta, theta0
    return theta, theta0
