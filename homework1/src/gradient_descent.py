import numpy as np
from typing import Callable

xiType = np.ndarray
yiType = np.ndarray
thetaType = np.ndarray
xType = list[xiType]
yType = list[yiType]


def gradient_descent(
    dimension: int,
    x: xType,
    y: yType,
    gradient: Callable[[xType, yType, thetaType], thetaType],
    learning_rate: float,
    iterations: int,
    initial_theta: thetaType = None,
) -> np.ndarray:
    if initial_theta is None:
        initial_theta = np.zeros(dimension)
    theta = initial_theta
    for _ in range(iterations):
        theta -= learning_rate * gradient(x, y, theta)
    return theta


def hinge_loss_gradient(x_i: xiType, y_i: yiType, theta: thetaType) -> thetaType:
    z = y_i * (np.dot(x_i, theta))
    if z >= 1:
        return np.zeros(len(theta))
    return -y_i * x_i


def regularization_gradient(theta: thetaType, lambda_: float) -> thetaType:
    return lambda_ * theta


def j_gradient(x: xType, y: yType, theta: thetaType, lambda_: float) -> thetaType:
    result = regularization_gradient(theta, lambda_)
    for xi, yi in zip(x, y):
        result += hinge_loss_gradient(xi, yi, theta)
    result /= len(x)
    return result


theta = gradient_descent(
    dimension=2,
    x=np.array([np.array([1.0, 0.0])]),
    y=np.array([1.0]),
    gradient=lambda x, y, t: j_gradient(x, y, t, lambda_=0.5),
    learning_rate=0.00001,
    iterations=1000,
    initial_theta=np.array([1.0, 0.0]),
)
print(theta)
