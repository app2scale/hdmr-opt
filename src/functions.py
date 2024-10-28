import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.functions_forecast import optimize_helper
from scipy.optimize import minimize, OptimizeResult


# ------- FUNCTION DEFINITIONS -------

def xgb_opt(X):
    """
    Applies optimization helper function row-wise using XGBoost.

    Parameters:
        X (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Optimized output.
    """
    return np.apply_along_axis(lambda row: optimize_helper(row[0], row[1]), 1, X)


def testfunc_2d(x):
    """
    Test function for evaluating a simple quadratic form.

    Parameters:
        x (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    if len(x.shape) == 2:
        axis = 1
    else:
        axis = 0
    return np.sum(x**2, axis=axis, keepdims=True)


def camel3_2d(X):
    """
    Evaluates the Camel3 function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (2 * X[:, 0]**2 - 1.05 * X[:, 0]**4 + (X[:, 0]**6) / 6 + X[:, 0] * X[:, 1] + X[:, 1]**2).reshape(-1, 1)


def camel16_2d(X):
    """
    Evaluates the Camel16 function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (4 * X[:, 0]**2 - 2.1 * X[:, 0]**4 + (X[:, 0]**6) / 3 + X[:, 0] * X[:, 1] - 4 * X[:, 1]**2 + 4 * X[:, 1]**4).reshape(-1, 1)


def treccani_2d(X):
    """
    Evaluates the Treccani function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (X[:, 0]**4 + 4 * X[:, 0]**3 + 4 * X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)


def goldstein_2d(X):
    """
    Evaluates the Goldstein-Price function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    
    term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    term2 = 30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    
    return (term1 * term2).reshape(-1, 1)


def branin_2d(X):
    """
    Evaluates the Branin-Hoo function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return ((x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2 +
            (10 * (1 - (1 / (8 * np.pi))) * np.cos(x1)) + 10).reshape(-1, 1)


def rosenbrock_2d(X):
    """
    Evaluates the Rosenbrock function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (0.5 * (X[:, 0]**2 - X[:, 1])**2 + (X[:, 0] - 1)**2).reshape(-1, 1)


def ackley_2d(X):
    """
    Evaluates the Ackley function in 2D.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (20 + np.exp(1) - 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))).reshape(-1, 1)


def rosenbrock_10d(X):
    """
    Evaluates the 10-dimensional Rosenbrock function.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 10) or (10,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + (X[:, :-1] - 1)**2, axis=1)).reshape(-1, 1)


def griewank_10d(X):
    """
    Evaluates the 10-dimensional Griewank function.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 10) or (10,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    sum_sq = np.sum((X[:, 1:] - 100)**2, axis=1)
    prod_cos = np.prod(np.cos((X - 100) / np.sqrt(np.arange(1, d + 1))), axis=1)
    return (sum_sq / 4000 - prod_cos + 1).reshape(-1, 1)


def rastrigin_10d(X):
    """
    Evaluates the 10-dimensional Rastrigin function.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 10) or (10,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)).reshape(-1, 1)


def rastrigin_2d(X):
    """
    Evaluates the 2-dimensional Rastrigin function.

    Parameters:
        X (numpy.ndarray): Input data of shape (N, 2) or (2,).

    Returns:
        numpy.ndarray: Evaluated function result.
    """
    X = _ensure_2d(X)
    return (np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)).reshape(-1, 1)


def _ensure_2d(X):
    """
    Ensures that the input is a 2D numpy array.

    Parameters:
        X (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: 2D reshaped input.
    """
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X

# ------- MAIN EXECUTION -------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print("Usage: python functions.py [function_name]")
        sys.exit(1)

    function_name = sys.argv[1]
    f = globals().get(function_name)

    if not f:
        print(f"Function '{function_name}' not found.")
        sys.exit(1)

    # Define axis ranges based on function type
    axis_ranges = {
        'camel3_2d': (-5, 5),
        'camel16_2d': (-5, 5),
        'treccani_2d': (-5, 5),
        'goldstein_2d': (-2, 2),
        'branin_2d': ((-5, 10), (0, 15)),
        'rosenbrock_2d': (-2.048, 2.048),
        'ackley_2d': (-30, 30),
        'griewank_10d': (-600, 600),
        'rastrigin_10d': (-5.12, 5.12),
        'rastrigin_2d': (-5.12, 5.12),
        'testfunc_2d': (-5, 5)
    }

    # Determine ranges for plotting
    x1_min, x1_max = axis_ranges.get(function_name, (-5, 5))
    x2_min, x2_max = axis_ranges.get(function_name, (-5, 5))

    # Generate mesh grid for plotting
    N = 100
    x1 = np.linspace(x1_min, x1_max, N)
    x2 = np.linspace(x2_min, x2_max, N)
    X1, X2 = np.meshgrid(x1, x2)
    Y = f(np.column_stack((X1.ravel(), X2.ravel()))).reshape(X1.shape)

    # Plotting
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap='jet')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()
