import numpy as np

# ------- BEGIN FUNCTIONS -------

# 2d Camel3
def testfunc_2d(x):
    if len(x.shape) == 2:
        N, n = x.shape
        axis = 1
    else:
        n = len(x)
        axis = 0
    y = np.sum(x**2, axis=axis, keepdims=True)
    return y

def camel3_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (2*X[:, 0]**2 - 1.05*X[:, 0]**4 + (X[:, 0]**6)/6 + X[:, 0]*X[:, 1] + X[:, 1]**2).reshape(-1, 1)

# 2d Camel16 
def camel16_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (4*X[:, 0]**2 - 2.1*X[:, 0]**4 + (X[:, 0]**6)/3 + X[:, 0]*X[:, 1] - 4*X[:, 1]**2 + 4*X[:, 1]**4).reshape(-1, 1)

# 2d Treccani
def treccani_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (X[:, 0]**4 + 4*X[:, 0]**3 + 4*X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)

# 2d Goldstein
def goldstein_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + (2*x1 - 3*x2)**2 * (16 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    
    return (term1 * term2).reshape(-1, 1)

# 2d Branin
def branin_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    x1 = X[:, 0]
    x2 = X[:, 1]
    return ((x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2 + (10 * (1 - (1 / (8 * np.pi))) * np.cos(x1)) + 10).reshape(-1, 1)

# 2d Rosenbrock
def rosenbrock_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (0.5 * (X[:, 0]**2 - X[:, 1])**2 + (X[:, 0] - 1)**2).reshape(-1, 1)

# 2d Ackley
def ackley_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    x1 = X[:, 0]
    x2 = X[:, 1]
    return (20 + np.exp(1) - 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))).reshape(-1, 1)

# 10d Rosenbrock
def rosenbrock_10d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + (X[:, :-1] - 1)**2, axis=1)).reshape(-1, 1)

# 10d Griewank
def griewank_10d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    d = X.shape[1]
    sum_sq = np.sum((X[:, 1:] - 100)**2, axis=1)
    prod_cos = np.prod(np.cos((X - 100) / np.sqrt(np.arange(1, d + 1))), axis=1)
    return (sum_sq / 4000 - prod_cos + 1).reshape(-1, 1)

# 10d Rastrigin
def rastrigin_10d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)).reshape(-1, 1)

# 2d Rastrigin
def rastrigin_2d(X):
    try:
        X.shape[1]
    except:
        X = np.array([X])
    return (np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)).reshape(-1, 1)

# ------- END FUNCTIONS -------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print("Usage: python functions.py [function_name]")
        sys.exit(1)

    function_name = sys.argv[1]
    # Evaluate the function based on the provided name
    f = globals().get(function_name)

    if f == camel3_2d or f == camel16_2d or f == treccani_2d:
        x1_min = x2_min = -5
        x1_max = x2_max = 5
    elif f == goldstein_2d:
        x1_min = x2_min = -2
        x1_max = x2_max = 2
    elif f == branin_2d:
        x1_min, x1_max = (-5, 10)
        x2_min, x2_max = (0, 15)
    elif f == rosenbrock_2d or f == rosenbrock_10d:
        x1_min = x2_min = -2.048
        x1_max = x2_max = 2.048
    elif f == ackley_2d:
        # normally it was (-15, 30) but the plot was bad in that case
        x1_min = x2_min = -30 
        x1_max = x2_max = 30
    elif f == griewank_10d:
        x1_min = x2_min = -600
        x1_max = x2_max = 600
    elif f == rastrigin_10d:
        x1_min = x2_min = -5.12
        x1_max = x2_max = 5.12
    elif f == rastrigin_2d:
        x1_min = x2_min = -5.12
        x1_max = x2_max = 5.12
    elif f == testfunc_2d:
        x1_min = x2_min = -5
        x1_max = x2_max = 5
    

    N = 100
    x1 = np.linspace(x1_min, x1_max, N)
    x2 = np.linspace(x2_min, x2_max, N)
    # x2 = np.random.uniform(0, 1, N) * x1

    X1, X2 = np.meshgrid(x1, x2)
    Y = f(np.column_stack((X1.ravel(), X2.ravel()))).reshape(X1.shape)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap='jet')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()
    