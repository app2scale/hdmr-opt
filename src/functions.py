"""
Benchmark Test Functions for Global Optimization

This module provides a comprehensive collection of benchmark test functions
commonly used in optimization literature. These functions are used to evaluate
the performance of optimization algorithms like HDMR.

Function Categories:
-------------------
1. 2D Functions: Simple test cases for visualization and debugging
2. Multi-modal Functions: Functions with multiple local minima
3. High-dimensional Functions: For testing scalability (10D)

Each function is implemented with:
- Proper vectorization for efficient computation
- Comprehensive docstrings with mathematical formulation
- Known global minimum information
- Input domain specifications

References:
----------
- Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation 
  Experiments: Test Functions and Datasets. 
  Retrieved from http://www.sfu.ca/~ssurjano/

Author: HDMR Optimization Research Group
Date: 2024
Version: 2.0.0
"""

import sys
from typing import Tuple, Optional, Callable
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _ensure_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Ensure input is a 2D array with shape (N, d).
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input array of shape (d,) or (N, d).
    
    Returns
    -------
    NDArray[np.float64]
        Reshaped array of shape (N, d).
    """
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X


# ============================================================================
# 2D BENCHMARK FUNCTIONS
# ============================================================================

def testfunc_2d(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Simple quadratic test function: f(x) = Σ xᵢ²
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5, 5]
    - Global minimum: f(0, 0) = 0
    - Convex, unimodal
    - Used for basic algorithm verification
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Examples
    --------
    >>> testfunc_2d(np.array([0.0, 0.0]))
    array([[0.]])
    >>> testfunc_2d(np.array([[1.0, 1.0], [2.0, 2.0]]))
    array([[2.],
           [8.]])
    """
    x = _ensure_2d(x)
    axis = 1 if len(x.shape) == 2 else 0
    return np.sum(x**2, axis=axis, keepdims=True)


def rastrigin_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    2D Rastrigin function: Highly multi-modal with many local minima.
    
    Mathematical Form:
    -----------------
    f(x) = 20 + Σᵢ[xᵢ² - 10cos(2πxᵢ)]
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5.12, 5.12]
    - Global minimum: f(0, 0) = 0
    - Highly multi-modal (many local minima)
    - Challenging for gradient-based methods
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Rastrigin function is a typical example of a non-linear multimodal
    function. It was first proposed by Rastrigin as a 2-dimensional function
    and has been generalized by Mühlenbein et al.
    
    References
    ----------
    Rastrigin, L. A. (1974). Systems of extremal control. Nauka, Moscow.
    """
    X = _ensure_2d(X)
    return (np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)).reshape(-1, 1)


def camel3_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Three-Hump Camel function.
    
    Mathematical Form:
    -----------------
    f(x₁, x₂) = 2x₁² - 1.05x₁⁴ + x₁⁶/6 + x₁x₂ + x₂²
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5, 5]
    - Global minimum: f(0, 0) = 0
    - Three local minima (hence the name)
    - Smooth, differentiable
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2).reshape(-1, 1)


def camel16_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Six-Hump Camel function.
    
    Mathematical Form:
    -----------------
    f(x₁, x₂) = 4x₁² - 2.1x₁⁴ + x₁⁶/3 + x₁x₂ - 4x₂² + 4x₂⁴
    
    Properties:
    ----------
    - Domain: x₁ ∈ [-3, 3], x₂ ∈ [-2, 2] (standard)
              xᵢ ∈ [-5, 5] (extended)
    - Global minima: f(±0.0898, ∓0.7126) ≈ -1.0316
    - Six local minima
    - Two global minima (symmetric)
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    This function has two global minima and four local minima.
    It is commonly used as a test problem for optimization algorithms.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (4 * x1**2 - 2.1 * x1**4 + (x1**6) / 3 + 
            x1 * x2 - 4 * x2**2 + 4 * x2**4).reshape(-1, 1)


def treccani_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Trecanni function.
    
    Mathematical Form:
    -----------------
    f(x₁, x₂) = x₁⁴ + 4x₁³ + 4x₁² + x₂²
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5, 5]
    - Global minima: f(-2, 0) = 0, f(0, 0) = 0
    - Two global minima
    - Asymmetric landscape
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (x1**4 + 4 * x1**3 + 4 * x1**2 + x2**2).reshape(-1, 1)


def goldstein_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Goldstein-Price function.
    
    Mathematical Form:
    -----------------
    f(x) = [1 + (x₁+x₂+1)²(19-14x₁+3x₁²-14x₂+6x₁x₂+3x₂²)] ×
           [30 + (2x₁-3x₂)²(18-32x₁+12x₁²+48x₂-36x₁x₂+27x₂²)]
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-2, 2]
    - Global minimum: f(0, -1) = 3
    - Several local minima
    - Highly non-linear
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    This function is difficult for optimization due to its steep valleys
    and narrow ridges near the global minimum.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    
    term1 = (1 + (x1 + x2 + 1)**2 * 
             (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2))
    term2 = (30 + (2 * x1 - 3 * x2)**2 * 
             (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2))
    
    return (term1 * term2).reshape(-1, 1)


def branin_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Branin-Hoo function (or Branin RCOS function).
    
    Mathematical Form:
    -----------------
    f(x₁, x₂) = (x₂ - 5.1x₁²/4π² + 5x₁/π - 6)² + 10(1 - 1/8π)cos(x₁) + 10
    
    Properties:
    ----------
    - Domain: x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
    - Global minima: f(-π, 12.275) = f(π, 2.275) = f(9.42478, 2.475) ≈ 0.397887
    - Three global minima
    - Commonly used in Bayesian optimization
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Branin function is widely used to test optimization algorithms,
    particularly in the context of response surface methodology.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    term3 = s
    
    return (term1 + term2 + term3).reshape(-1, 1)


def rosenbrock_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    2D Rosenbrock function (scaled version).
    
    Mathematical Form:
    -----------------
    f(x₁, x₂) = 0.5(x₁² - x₂)² + (x₁ - 1)²
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-2.048, 2.048]
    - Global minimum: f(1, 1) = 0
    - Narrow curved valley
    - Classic test function
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Rosenbrock function is also known as the "banana function" due to
    its characteristic shape. It is a classic test problem for optimization
    algorithms due to its narrow, curved valley leading to the global minimum.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (0.5 * (x1**2 - x2)**2 + (x1 - 1)**2).reshape(-1, 1)


def ackley_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    2D Ackley function.
    
    Mathematical Form:
    -----------------
    f(x) = 20 + e - 20exp(-0.2√(0.5Σxᵢ²)) - exp(0.5Σcos(2πxᵢ))
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-30, 30] (standard), xᵢ ∈ [-5, 5] (common)
    - Global minimum: f(0, 0) = 0
    - Highly multi-modal with exponential characteristics
    - Nearly flat outer region
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 2) or (2,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Ackley function is characterized by a nearly flat outer region and
    a large hole at the center. It is useful for testing algorithm ability
    to escape from local minima and find the global minimum.
    """
    X = _ensure_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    
    return (20 + np.exp(1) + term1 + term2).reshape(-1, 1)


# ============================================================================
# 10D BENCHMARK FUNCTIONS
# ============================================================================

def rosenbrock_10d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    10-dimensional Rosenbrock function.
    
    Mathematical Form:
    -----------------
    f(x) = Σᵢ₌₁⁹ [100(xᵢ² - xᵢ₊₁)² + (xᵢ - 1)²]
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5, 10] (or [-2.048, 2.048])
    - Global minimum: f(1, 1, ..., 1) = 0
    - Narrow curved valley in high dimensions
    - Scalable test function
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 10) or (10,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    """
    X = _ensure_2d(X)
    return (np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + 
                   (X[:, :-1] - 1)**2, axis=1)).reshape(-1, 1)


def griewank_10d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    10-dimensional Griewank function.
    
    Mathematical Form:
    -----------------
    f(x) = 1 + (1/4000)Σ(xᵢ-100)² - Π cos((xᵢ-100)/√i)
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-600, 600] (standard)
    - Global minimum: f(100, 100, ..., 100) = 0
    - Many regularly distributed local minima
    - Product term creates interdependence
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 10) or (10,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Griewank function has a product term that introduces interdependence
    among the variables, making it more difficult for optimization algorithms.
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    
    sum_term = np.sum((X - 100)**2, axis=1) / 4000
    prod_term = np.prod(np.cos((X - 100) / np.sqrt(np.arange(1, d + 1))), axis=1)
    
    return (sum_term - prod_term + 1).reshape(-1, 1)


def rastrigin_10d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    10-dimensional Rastrigin function.
    
    Mathematical Form:
    -----------------
    f(x) = 10d + Σᵢ₌₁ᵈ [xᵢ² - 10cos(2πxᵢ)]
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5.12, 5.12]
    - Global minimum: f(0, 0, ..., 0) = 0
    - Highly multi-modal (10^10 local minima approx.)
    - Scalable difficulty with dimension
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, 10) or (10,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    """
    X = _ensure_2d(X)
    return (np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)).reshape(-1, 1)


# ============================================================================
# FUNCTION REGISTRY AND UTILITIES
# ============================================================================

# Global registry of all benchmark functions
BENCHMARK_FUNCTIONS = {
    # 2D Functions
    'testfunc_2d': testfunc_2d,
    'rastrigin_2d': rastrigin_2d,
    'camel3_2d': camel3_2d,
    'camel16_2d': camel16_2d,
    'treccani_2d': treccani_2d,
    'goldstein_2d': goldstein_2d,
    'branin_2d': branin_2d,
    'rosenbrock_2d': rosenbrock_2d,
    'ackley_2d': ackley_2d,
    # 10D Functions
    'rosenbrock_10d': rosenbrock_10d,
    'griewank_10d': griewank_10d,
    'rastrigin_10d': rastrigin_10d,
}


def get_function_info(function_name: str) -> dict:
    """
    Get information about a benchmark function.
    
    Parameters
    ----------
    function_name : str
        Name of the benchmark function.
    
    Returns
    -------
    dict
        Dictionary containing function metadata:
        - 'function': callable
        - 'dimension': int
        - 'domain': tuple
        - 'global_minimum': float
        - 'global_minimizer': list
    
    Raises
    ------
    ValueError
        If function_name is not in BENCHMARK_FUNCTIONS.
    """
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Function '{function_name}' not found. "
                        f"Available: {list(BENCHMARK_FUNCTIONS.keys())}")
    
    # Extract dimension from function name
    dim_str = function_name.split('_')[-1]
    dimension = int(dim_str[:-1])  # Remove 'd' suffix
    
    # Function-specific metadata (simplified - extend as needed)
    metadata = {
        'testfunc_2d': {'domain': (-5, 5), 'min': 0, 'minimizer': [0, 0]},
        'rastrigin_2d': {'domain': (-5.12, 5.12), 'min': 0, 'minimizer': [0, 0]},
        'camel3_2d': {'domain': (-5, 5), 'min': 0, 'minimizer': [0, 0]},
        'camel16_2d': {'domain': (-5, 5), 'min': -1.0316, 
                       'minimizer': [0.0898, -0.7126]},
        'treccani_2d': {'domain': (-5, 5), 'min': 0, 'minimizer': [-2, 0]},
        'goldstein_2d': {'domain': (-2, 2), 'min': 3, 'minimizer': [0, -1]},
        'branin_2d': {'domain': (-5, 15), 'min': 0.397887, 
                      'minimizer': [-np.pi, 12.275]},
        'rosenbrock_2d': {'domain': (-2.048, 2.048), 'min': 0, 'minimizer': [1, 1]},
        'ackley_2d': {'domain': (-30, 30), 'min': 0, 'minimizer': [0, 0]},
        'rosenbrock_10d': {'domain': (-5, 10), 'min': 0, 'minimizer': [1] * 10},
        'griewank_10d': {'domain': (-600, 600), 'min': 0, 'minimizer': [100] * 10},
        'rastrigin_10d': {'domain': (-5.12, 5.12), 'min': 0, 'minimizer': [0] * 10},
    }
    
    info = metadata.get(function_name, {'domain': (-5, 5), 'min': 0, 'minimizer': [0] * dimension})
    
    return {
        'function': BENCHMARK_FUNCTIONS[function_name],
        'dimension': dimension,
        'domain': info['domain'],
        'global_minimum': info['min'],
        'global_minimizer': info['minimizer']
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_2d_function(
    function_name: str,
    resolution: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a 2D benchmark function.
    
    Parameters
    ----------
    function_name : str
        Name of the 2D function to visualize.
    resolution : int, optional
        Number of points in each dimension (default: 100).
    save_path : str, optional
        Path to save the figure (if None, displays instead).
    """
    if not function_name.endswith('_2d'):
        raise ValueError(f"Function '{function_name}' is not a 2D function")
    
    info = get_function_info(function_name)
    func = info['function']
    domain = info['domain']
    
    # Handle asymmetric domains (like branin)
    if isinstance(domain, tuple) and len(domain) == 2 and not isinstance(domain[0], tuple):
        x1_range = x2_range = domain
    else:
        x1_range, x2_range = domain if isinstance(domain[0], tuple) else (domain, domain)
    
    # Generate mesh
    x1 = np.linspace(x1_range[0], x1_range[1], resolution)
    x2 = np.linspace(x2_range[0], x2_range[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate function
    points = np.column_stack((X1.ravel(), X2.ravel()))
    Y = func(points).reshape(X1.shape)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_zlabel('f(x)', fontsize=11)
    ax1.set_title(f'{function_name} - 3D Surface', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X1, X2, Y, levels=20, cmap='viridis')
    ax2.contourf(X1, X2, Y, levels=20, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('x₁', fontsize=11)
    ax2.set_ylabel('x₂', fontsize=11)
    ax2.set_title(f'{function_name} - Contour Plot', fontsize=12, fontweight='bold')
    fig.colorbar(contour, ax=ax2)
    
    # Mark global minimum
    minimizer = info['global_minimizer']
    ax2.plot(minimizer[0], minimizer[1], 'r*', markersize=15, label='Global Min')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Command-line interface for visualizing benchmark functions.
    
    Usage:
        python functions.py <function_name>
    
    Example:
        python functions.py rastrigin_2d
    """
    if len(sys.argv) != 2:
        print("Usage: python functions.py [function_name]")
        print(f"\nAvailable functions:")
        for fname in sorted(BENCHMARK_FUNCTIONS.keys()):
            info = get_function_info(fname)
            print(f"  - {fname:20s} (dim={info['dimension']}, "
                  f"f*={info['global_minimum']})")
        sys.exit(1)
    
    function_name = sys.argv[1]
    
    if function_name not in BENCHMARK_FUNCTIONS:
        print(f"Error: Function '{function_name}' not found.")
        print(f"Available: {list(BENCHMARK_FUNCTIONS.keys())}")
        sys.exit(1)
    
    # Display function information
    info = get_function_info(function_name)
    print("\n" + "=" * 70)
    print(f"BENCHMARK FUNCTION: {function_name}")
    print("=" * 70)
    print(f"Dimension:        {info['dimension']}")
    print(f"Domain:           {info['domain']}")
    print(f"Global Minimum:   {info['global_minimum']}")
    print(f"Global Minimizer: {info['global_minimizer']}")
    print("=" * 70 + "\n")
    
    # Visualize if 2D
    if function_name.endswith('_2d'):
        visualize_2d_function(function_name)
    else:
        print(f"Visualization not available for {info['dimension']}D functions")
        print("Evaluating at origin:")
        func = info['function']
        x_origin = np.zeros(info['dimension'])
        f_origin = func(x_origin)
        print(f"f(0, 0, ..., 0) = {f_origin[0, 0]:.6f}")