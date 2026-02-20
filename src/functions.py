"""
Benchmark Test Functions for Global Optimization
=================================================

This module provides a comprehensive collection of benchmark test functions
commonly used in optimization literature (1980-2026). These functions are used
to evaluate the performance of optimization algorithms like HDMR.

Function Categories:
-------------------
1. 2D Functions: Simple test cases for visualization and debugging (9 functions)
2. Classical 10D Functions: Traditional high-dimensional benchmarks (3 functions)
3. Modern Scalable Functions: Contemporary benchmarks supporting arbitrary dimensions (8 functions)

Total: 20 unique functions × multiple dimensions = comprehensive test suite

Each function is implemented with:
- Proper vectorization for efficient computation
- Comprehensive docstrings with mathematical formulation
- Known global minimum information
- Input domain specifications
- Literature references

Dimension Support:
-----------------
- Fixed: 2D functions (visualization and algorithm verification)
- Scalable: All other functions support 10D, 20D, 30D, 50D, 100D, etc.

References:
----------
- Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation 
  Experiments: Test Functions and Datasets. 
  Retrieved from http://www.sfu.ca/~ssurjano/
- Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions 
  for global optimisation problems. International Journal of Mathematical 
  Modelling and Numerical Optimisation, 4(2), 150-194.
- CEC 2017 Competition on Real-Parameter Single Objective Optimization
- Hansen, N., et al. (2021). COCO: A platform for comparing continuous 
  optimizers in a black-box setting. Optimization Methods and Software, 36(1).

Author: HDMR Optimization Research Group
Date: 2025-02-16
Version: 3.0.0
"""

import sys
from typing import Tuple, Optional, Callable, Dict, List
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
    
    Examples
    --------
    >>> _ensure_2d(np.array([1, 2, 3]))
    array([[1, 2, 3]])
    >>> _ensure_2d(np.array([[1, 2], [3, 4]]))
    array([[1, 2],
           [3, 4]])
    """
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X


# ============================================================================
# 2D BENCHMARK FUNCTIONS (VISUALIZATION & DEBUGGING)
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
# CLASSICAL HIGH-DIMENSIONAL FUNCTIONS (10D)
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
# MODERN SCALABLE BENCHMARK FUNCTIONS (2020-2026)
# ============================================================================

def schwefel(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Schwefel function - Highly deceptive multi-modal function.
    
    Mathematical Form:
    -----------------
    f(x) = 418.9829d - Σᵢ xᵢ sin(√|xᵢ|)
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-500, 500]
    - Global minimum: f(420.9687, ..., 420.9687) = 0
    - Highly multi-modal and deceptive
    - Global optimum is geometrically distant from next best local optima
    - One of the most difficult benchmark functions
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Schwefel function is deceptive in that the global minimum is 
    geometrically distant, over the parameter space, from the next 
    best local minima. Therefore, search algorithms are potentially 
    prone to convergence in the wrong direction.
    
    Used extensively in CEC 2017, 2020, 2022 competitions.
    
    References
    ----------
    Schwefel, H. P. (1981). Numerical optimization of computer models. 
    John Wiley & Sons.
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    return (418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)).reshape(-1, 1)


def levy(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Levy function - Multi-modal function with many local minima.
    
    Mathematical Form:
    -----------------
    f(x) = sin²(πw₁) + Σᵢ₌₁ᵈ⁻¹[(wᵢ-1)²(1+10sin²(πwᵢ+1))] + (wᵈ-1)²(1+sin²(2πwᵈ))
    where wᵢ = 1 + (xᵢ-1)/4
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-10, 10]
    - Global minimum: f(1, 1, ..., 1) = 0
    - Multi-modal with many local minima
    - Scalable to any dimension
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Levy function is a difficult optimization problem with many 
    local minima. Commonly used in recent optimization literature (2020+).
    
    References
    ----------
    Levy, A. V., & Montalvo, A. (1985). The tunneling algorithm for the 
    global minimization of functions. SIAM Journal on Scientific and 
    Statistical Computing, 6(1), 15-29.
    """
    X = _ensure_2d(X)
    
    # Transform to w
    w = 1 + (X - 1) / 4
    
    # Compute terms
    term1 = np.sin(np.pi * w[:, 0])**2
    
    term2 = np.sum(
        (w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2),
        axis=1
    )
    
    term3 = (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)
    
    return (term1 + term2 + term3).reshape(-1, 1)


def zakharov(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Zakharov function - Unimodal plate-shaped function.
    
    Mathematical Form:
    -----------------
    f(x) = Σᵢ xᵢ² + (Σᵢ 0.5i·xᵢ)² + (Σᵢ 0.5i·xᵢ)⁴
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5, 10] (or [-10, 10])
    - Global minimum: f(0, 0, ..., 0) = 0
    - Unimodal (single minimum)
    - Plate-shaped, asymmetric
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Zakharov function has no local minima except the global minimum.
    Useful for testing convergence properties of optimization algorithms.
    
    References
    ----------
    Zakharov, V. D. (1975). A method of searching for a global extremum.
    USSR Computational Mathematics and Mathematical Physics, 15(3), 37-44.
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    
    # Create coefficient vector [0.5, 1.0, 1.5, 2.0, ...]
    coeffs = 0.5 * np.arange(1, d + 1)
    
    # Compute terms
    term1 = np.sum(X**2, axis=1)
    sum_term = np.sum(X * coeffs, axis=1)
    term2 = sum_term**2
    term3 = sum_term**4
    
    return (term1 + term2 + term3).reshape(-1, 1)


def sphere(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Sphere function - Simple unimodal baseline function.
    
    Mathematical Form:
    -----------------
    f(x) = Σᵢ xᵢ²
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5.12, 5.12] (or unbounded)
    - Global minimum: f(0, 0, ..., 0) = 0
    - Convex, unimodal
    - Easiest test function (baseline)
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Sphere function is the simplest benchmark function and serves
    as a baseline for comparing optimization algorithms. Any reasonable
    optimizer should solve this easily.
    
    Universal baseline in all optimization literature.
    """
    X = _ensure_2d(X)
    return np.sum(X**2, axis=1).reshape(-1, 1)


def sum_of_different_powers(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Sum of Different Powers function - Unimodal with varying steepness.
    
    Mathematical Form:
    -----------------
    f(x) = Σᵢ |xᵢ|^(i+1)
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-1, 1]
    - Global minimum: f(0, 0, ..., 0) = 0
    - Unimodal
    - Different steepness in different directions
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    This function has different powers for each dimension, creating
    varying steepness. Useful for testing algorithm behavior with
    non-uniform landscapes.
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    
    # Powers: [2, 3, 4, 5, ..., d+1]
    powers = np.arange(2, d + 2)
    
    return np.sum(np.abs(X)**powers, axis=1).reshape(-1, 1)


def styblinski_tang(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Styblinski-Tang function - Multiple global optima.
    
    Mathematical Form:
    -----------------
    f(x) = 0.5 Σᵢ (xᵢ⁴ - 16xᵢ² + 5xᵢ)
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-5, 5]
    - Global minimum: f(-2.903534, ..., -2.903534) = -39.16599d
    - Multiple local minima
    - All dimensions contribute equally
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Styblinski-Tang function is multi-modal and has a large number
    of local minima. Commonly used in CEC competitions.
    """
    X = _ensure_2d(X)
    return (0.5 * np.sum(X**4 - 16 * X**2 + 5 * X, axis=1)).reshape(-1, 1)


def dixon_price(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Dixon-Price function - Unimodal with steep ridges.
    
    Mathematical Form:
    -----------------
    f(x) = (x₁-1)² + Σᵢ₌₂ᵈ i(2xᵢ² - xᵢ₋₁)²
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [-10, 10]
    - Global minimum: f(2^(-2^(i-1)/2^i) for i=1..d) = 0
    - Unimodal but with steep ridges
    - Becomes more difficult with increasing dimension
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Dixon-Price function is continuous, differentiable, non-separable,
    and scalable. The global optimum lies in a steep ridge.
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    
    term1 = (X[:, 0] - 1)**2
    
    # Vectorized computation for i=2 to d
    i = np.arange(2, d + 1)
    term2 = np.sum(i * (2 * X[:, 1:]**2 - X[:, :-1])**2, axis=1)
    
    return (term1 + term2).reshape(-1, 1)


def michalewicz(X: NDArray[np.float64], m: int = 10) -> NDArray[np.float64]:
    """
    Michalewicz function - Multi-modal with steep ridges and valleys.
    
    Mathematical Form:
    -----------------
    f(x) = -Σᵢ sin(xᵢ) sin²ᵐ(i·xᵢ²/π)
    
    Properties:
    ----------
    - Domain: xᵢ ∈ [0, π]
    - Global minimum: depends on d and m
      - d=2: f(2.20, 1.57) ≈ -1.8013
      - d=5: ≈ -4.687658
      - d=10: ≈ -9.66015
    - Highly multi-modal
    - Steepness controlled by m (default m=10)
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Input of shape (N, d) or (d,).
    m : int
        Steepness parameter (default: 10).
    
    Returns
    -------
    NDArray[np.float64]
        Function values of shape (N, 1).
    
    Notes
    -----
    The Michalewicz function has d! local minima. The parameter m defines
    the steepness of the valleys/ridges. Larger m leads to more difficult
    searches. Recommended m = 10.
    
    Popular in recent optimization literature (2020+).
    """
    X = _ensure_2d(X)
    d = X.shape[1]
    
    # Create index array
    i = np.arange(1, d + 1)
    
    # Compute function
    result = -np.sum(
        np.sin(X) * np.sin((i * X**2) / np.pi)**(2 * m),
        axis=1
    )
    
    return result.reshape(-1, 1)


# ============================================================================
# FUNCTION REGISTRY AND UTILITIES
# ============================================================================

# Global registry of all benchmark functions
BENCHMARK_FUNCTIONS: Dict[str, Callable] = {
    # 2D Functions (9 functions)
    'testfunc_2d': testfunc_2d,
    'rastrigin_2d': rastrigin_2d,
    'camel3_2d': camel3_2d,
    'camel16_2d': camel16_2d,
    'treccani_2d': treccani_2d,
    'goldstein_2d': goldstein_2d,
    'branin_2d': branin_2d,
    'rosenbrock_2d': rosenbrock_2d,
    'ackley_2d': ackley_2d,
    
    # Classical 10D Functions (3 functions)
    'rosenbrock_10d': rosenbrock_10d,
    'griewank_10d': griewank_10d,
    'rastrigin_10d': rastrigin_10d,
    
    # Modern Scalable Functions (8 functions)
    'schwefel': schwefel,
    'levy': levy,
    'zakharov': zakharov,
    'sphere': sphere,
    'sum_of_different_powers': sum_of_different_powers,
    'styblinski_tang': styblinski_tang,
    'dixon_price': dixon_price,
    'michalewicz': michalewicz,
}


def get_function_info(function_name: str) -> Dict[str, any]:
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
    
    Examples
    --------
    >>> info = get_function_info('rastrigin_2d')
    >>> print(info['dimension'])
    2
    >>> print(info['global_minimum'])
    0.0
    """
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Function '{function_name}' not found. "
                        f"Available: {list(BENCHMARK_FUNCTIONS.keys())}")
    
    # Extract dimension from function name
    dim_str = function_name.split('_')[-1]
    if dim_str.endswith('d') and dim_str[:-1].isdigit():
        dimension = int(dim_str[:-1])
    else:
        # Modern scalable functions - default to 10D
        dimension = 10
    
    # Function-specific metadata
    metadata = {
        # 2D Functions
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
        
        # Classical 10D
        'rosenbrock_10d': {'domain': (-5, 10), 'min': 0, 'minimizer': [1] * 10},
        'griewank_10d': {'domain': (-600, 600), 'min': 0, 'minimizer': [100] * 10},
        'rastrigin_10d': {'domain': (-5.12, 5.12), 'min': 0, 'minimizer': [0] * 10},
        
        # Modern Scalable (defaults for dimension=10)
        'schwefel': {'domain': (-500, 500), 'min': 0, 'minimizer': [420.9687] * dimension},
        'levy': {'domain': (-10, 10), 'min': 0, 'minimizer': [1] * dimension},
        'zakharov': {'domain': (-5, 10), 'min': 0, 'minimizer': [0] * dimension},
        'sphere': {'domain': (-5.12, 5.12), 'min': 0, 'minimizer': [0] * dimension},
        'sum_of_different_powers': {'domain': (-1, 1), 'min': 0, 'minimizer': [0] * dimension},
        'styblinski_tang': {'domain': (-5, 5), 'min': -39.16599 * dimension, 
                           'minimizer': [-2.903534] * dimension},
        'dixon_price': {'domain': (-10, 10), 'min': 0, 
                       'minimizer': [2**(-((2**i - 2) / 2**i)) for i in range(1, dimension + 1)]},
        'michalewicz': {'domain': (0, np.pi), 
                       'min': {2: -1.8013, 5: -4.687658, 10: -9.66015}.get(dimension, -dimension),
                       'minimizer': [None] * dimension},
    }
    
    info = metadata.get(function_name, {'domain': (-5, 5), 'min': 0, 'minimizer': [0] * dimension})
    
    return {
        'function': BENCHMARK_FUNCTIONS[function_name],
        'dimension': dimension,
        'domain': info['domain'],
        'global_minimum': info['min'],
        'global_minimizer': info['minimizer']
    }


def list_functions(category: Optional[str] = None) -> List[str]:
    """
    List available benchmark functions, optionally filtered by category.
    
    Parameters
    ----------
    category : str, optional
        Filter by category: '2d', '10d', 'modern', or None for all.
    
    Returns
    -------
    List[str]
        List of function names.
    
    Examples
    --------
    >>> list_functions('2d')
    ['testfunc_2d', 'rastrigin_2d', ...]
    >>> list_functions('modern')
    ['schwefel', 'levy', 'zakharov', ...]
    """
    if category is None:
        return list(BENCHMARK_FUNCTIONS.keys())
    
    category = category.lower()
    
    if category == '2d':
        return [name for name in BENCHMARK_FUNCTIONS if name.endswith('_2d')]
    elif category == '10d':
        return [name for name in BENCHMARK_FUNCTIONS if name.endswith('_10d')]
    elif category == 'modern':
        modern = ['schwefel', 'levy', 'zakharov', 'sphere', 
                 'sum_of_different_powers', 'styblinski_tang', 
                 'dixon_price', 'michalewicz']
        return [name for name in modern if name in BENCHMARK_FUNCTIONS]
    else:
        raise ValueError(f"Unknown category: {category}. Use '2d', '10d', 'modern', or None.")


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
    
    Examples
    --------
    >>> visualize_2d_function('rastrigin_2d')
    >>> visualize_2d_function('rosenbrock_2d', save_path='rosenbrock.png')
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
    if minimizer[0] is not None:
        ax2.plot(minimizer[0], minimizer[1], 'r*', markersize=15, label='Global Min')
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# MAIN EXECUTION (CLI)
# ============================================================================

if __name__ == "__main__":
    """
    Command-line interface for benchmark functions.
    
    Usage:
        python functions.py                    # List all functions
        python functions.py --list [category]  # List by category
        python functions.py [function_name]    # Visualize or test function
    
    Examples:
        python functions.py --list modern
        python functions.py rastrigin_2d
        python functions.py schwefel
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HDMR Benchmark Functions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'function', nargs='?', default=None,
        help='Function name to test/visualize'
    )
    parser.add_argument(
        '--list', type=str, default=None, const='all', nargs='?',
        help='List functions (all, 2d, 10d, modern)'
    )
    parser.add_argument(
        '--dimension', type=int, default=10,
        help='Dimension for scalable functions (default: 10)'
    )
    
    args = parser.parse_args()
    
    # List functions
    if args.list is not None:
        category = None if args.list == 'all' else args.list
        functions_list = list_functions(category)
        
        print("\n" + "=" * 70)
        print(f"BENCHMARK FUNCTIONS{' - ' + category.upper() if category else ''}")
        print("=" * 70)
        
        for fname in sorted(functions_list):
            try:
                info = get_function_info(fname)
                print(f"  - {fname:25s} (dim={info['dimension']:3d}, "
                      f"f*={info['global_minimum']:>12.6f})")
            except:
                print(f"  - {fname:25s}")
        
        print("=" * 70)
        print(f"Total: {len(functions_list)} functions")
        print("=" * 70 + "\n")
        sys.exit(0)
    
    # Test/visualize specific function
    if args.function is None:
        print("Usage: python functions.py [function_name] or --list [category]")
        print("Run 'python functions.py --list' to see available functions")
        sys.exit(0)
    
    function_name = args.function
    
    if function_name not in BENCHMARK_FUNCTIONS:
        print(f"Error: Function '{function_name}' not found.")
        print(f"Run 'python functions.py --list' to see available functions")
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
        print("Generating visualization...")
        visualize_2d_function(function_name)
    else:
        # Test at origin
        print(f"Testing function at origin (dimension={info['dimension']})...")
        func = info['function']
        x_origin = np.zeros(info['dimension'])
        f_origin = func(x_origin)
        print(f"f(0, 0, ..., 0) = {f_origin[0, 0]:.6f}\n")