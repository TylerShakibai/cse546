from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    errVec = ((X @ weight) + bias) - y
    errSum = np.sum(errVec)
    bias -= 2 * eta * errSum
    weight -= 2 * eta * (X.T @ errVec)
    regStep = 2 * eta * _lambda

    argZero = np.logical_and(weight > -regStep, weight < regStep)
    weight[weight < -regStep] = weight[weight < -regStep] + regStep
    weight[weight > regStep] = weight[weight > regStep] - regStep
    weight[argZero] = 0
    return weight, bias


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return np.sum((((X @ weight) + bias) - y)**2) + _lambda * np.sum(np.abs(weight))


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    currW: Optional[np.ndarray] = start_weight
    currB: Optional[np.ndarray] = start_bias
    
    flag = True
    while flag:
        nextW, nextB = step(X, y, currW, currB, _lambda, eta)
        flag = not convergence_criterion(nextW, currW, nextB, currB, convergence_delta=0.1)
        currW = nextW.copy()
        currB = nextW.copy()

    return nextW, nextB


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if np.max(np.abs(old_w - weight)) > convergence_delta:
        convergence = False
    else:
        convergence = True

    return convergence


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    x = np.random.chisquare(100, (n, d))
    w = np.zeros(d)
    w[:100] = np.linspace(1, 100, 100)/k
    
    y = (x @ w) + np.random.normal(0, 1, n)
    standX = (x - x.mean(axis=0))/x.std(axis=0)
    _lambda_max = np.max(2 * np.abs(standX.T @ (y - y.mean())))

    dense_weights = []
    truPosRate = []
    falDiscRate = []
    regVals = []
    for lam in np.geomspace(0.01, _lambda_max, 100):
        weight, bias = train(standX, y, lam)
        if np.count_nonzero(weight) != 0:
            dense_weights.append(np.count_nonzero(weight))
            truPosRate.append(np.count_nonzero(weight[:k+1])/k)
            falDiscRate.append(np.count_nonzero(weight[k+1:])/np.count_nonzero(weight))
            regVals.append(lam)

    fig, ax = plt.subplots()
    ax.semilogx(regVals, dense_weights)
    ax.set_xlabel('Regularization Value')
    ax.set_ylabel('Number of Non-Zero Weights')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(falDiscRate, truPosRate)
    ax.set_xlabel('False Discovery Rate')
    ax.set_ylabel('True Positive Rate')
    plt.show()


if __name__ == "__main__":
    main()