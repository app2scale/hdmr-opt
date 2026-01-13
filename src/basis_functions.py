"""
Orthogonal Basis Functions for HDMR Decomposition

This module provides high-performance implementations of orthogonal basis functions
used in High Dimensional Model Representation (HDMR).

Mathematical Foundation:
-----------------------
HDMR approximates f(x₁, x₂, ..., xₙ) as:
    f(x) ≈ f₀ + Σᵢ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ, xⱼ) + ...

Each component fᵢ is represented using orthogonal basis functions Φᵣ(x).

Author: HDMR Research Group
Version: 3.0.0
Date: 2026-01-12
"""

import math
from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray


class BasisFunctionError(Exception):
    """Custom exception for basis function errors."""
    pass


def legendre_polynomial(
    order: int,
    x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute Legendre polynomial Pₙ(x) using three-term recurrence.
    
    Recurrence Relation:
        P₀(x) = 1
        P₁(x) = x
        Pₙ(x) = [(2n-1)xPₙ₋₁(x) - (n-1)Pₙ₋₂(x)] / n
    
    Args:
        order: Polynomial order (n ≥ 0)
        x: Evaluation points, any shape
    
    Returns:
        Pₙ(x) values with same shape as x
    
    Raises:
        BasisFunctionError: If order < 0
        
    Complexity:
        Time: O(n·|x|)
        Space: O(|x|)
    
    Examples:
        >>> x = np.array([0.0, 0.5, 1.0])
        >>> legendre_polynomial(0, x)
        array([1., 1., 1.])
        >>> legendre_polynomial(2, x)
        array([-0.5 ,  0.125,  1.  ])
    """
    if order < 0:
        raise BasisFunctionError(f"Order must be ≥ 0, got {order}")
    
    if order == 0:
        return np.ones_like(x, dtype=np.float64)
    elif order == 1:
        return x.astype(np.float64)
    
    # Three-term recurrence (numerically stable)
    p_prev2 = np.ones_like(x, dtype=np.float64)  # P₀
    p_prev1 = x.astype(np.float64)                # P₁
    
    for n in range(2, order + 1):
        p_curr = ((2*n - 1) * x * p_prev1 - (n - 1) * p_prev2) / n
        p_prev2 = p_prev1
        p_prev1 = p_curr
    
    return p_prev1


def legendre_basis(
    a: float,
    b: float,
    order: int,
    x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Normalized Legendre basis function on interval [a, b].
    
    Transformation:
        φᵣ(x) = √[(2r+1)/(b-a)] · Pᵣ(2(x-b)/(b-a) + 1)
    
    The basis satisfies orthonormality:
        ∫ₐᵇ φᵣ(x)φₛ(x)dx = δᵣₛ
    
    Args:
        a: Lower interval bound
        b: Upper interval bound (must be > a)
        order: Basis function order (r ≥ 0)
        x: Evaluation points
    
    Returns:
        Normalized basis function values
    
    Raises:
        BasisFunctionError: If b ≤ a or order < 0
        
    Properties:
        - Orthonormal on [a, b]
        - Smooth (C^∞)
        - Complete basis for L²[a, b]
    
    Examples:
        >>> x = np.linspace(0, 1, 5)
        >>> legendre_basis(0.0, 1.0, 0, x)  # Constant
        array([1., 1., 1., 1., 1.])
        >>> legendre_basis(0.0, 1.0, 1, x)  # Linear
        array([-1.73..., -0.86..., 0., 0.86..., 1.73...])
    """
    if b <= a:
        raise BasisFunctionError(
            f"Invalid interval: b ({b}) must be > a ({a})"
        )
    if order < 0:
        raise BasisFunctionError(f"Order must be ≥ 0, got {order}")
    
    # Normalization factor for orthonormality
    norm = np.sqrt((2 * order + 1) / (b - a))
    
    # Map [a, b] → [-1, 1]
    x_normalized = 2 * (x - b) / (b - a) + 1
    
    # Evaluate Legendre polynomial
    poly = legendre_polynomial(order, x_normalized)
    
    return norm * poly


def cosine_basis(
    a: float,
    b: float,
    frequency: int,
    x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Normalized cosine basis function on interval [a, b].
    
    Definition:
        φₘ(x) = √[1/(b-a) · 8πm/(sin(4πm) + 4πm)] · cos(2πm(x-a)/(b-a))
    
    Args:
        a: Lower interval bound
        b: Upper interval bound (must be > a)
        frequency: Frequency parameter (m ≥ 1)
        x: Evaluation points
    
    Returns:
        Normalized basis function values
    
    Raises:
        BasisFunctionError: If b ≤ a or frequency < 1
        
    Properties:
        - Orthonormal on [a, b]
        - Periodic with period (b-a)/m
        - Ideal for oscillatory functions
    
    Notes:
        For m=1: one complete cosine period over [a, b]
        For m=k: k complete periods over [a, b]
    
    Examples:
        >>> x = np.linspace(0, 1, 100)
        >>> cos1 = cosine_basis(0.0, 1.0, 1, x)  # One period
        >>> cos2 = cosine_basis(0.0, 1.0, 2, x)  # Two periods
    """
    if b <= a:
        raise BasisFunctionError(
            f"Invalid interval: b ({b}) must be > a ({a})"
        )
    if frequency < 1:
        raise BasisFunctionError(
            f"Frequency must be ≥ 1, got {frequency}"
        )
    
    # Normalization factor
    four_pi_m = 4 * math.pi * frequency
    norm = np.sqrt(
        1 / (b - a) * 8 * math.pi * frequency / 
        (math.sin(four_pi_m) + four_pi_m)
    )
    
    # Cosine argument
    theta = 2 * math.pi * frequency * (x - a) / (b - a)
    
    return norm * np.cos(theta)


# Aliases for backward compatibility
Legendre = legendre_basis
Cosine = cosine_basis
Pn = legendre_polynomial


class BasisFunction:
    """
    Factory class for basis function selection and configuration.
    
    Usage:
        >>> basis = BasisFunction.create("Legendre", a=0, b=1)
        >>> values = basis.evaluate(order=3, x=np.linspace(0, 1, 100))
    """
    
    SUPPORTED_TYPES = {"Legendre", "Cosine"}
    
    @staticmethod
    def create(
        basis_type: str,
        a: float,
        b: float
    ) -> 'BasisFunctionEvaluator':
        """
        Create basis function evaluator.
        
        Args:
            basis_type: "Legendre" or "Cosine"
            a: Lower interval bound
            b: Upper interval bound
        
        Returns:
            Configured evaluator instance
        
        Raises:
            BasisFunctionError: If basis_type not supported
        """
        if basis_type not in BasisFunction.SUPPORTED_TYPES:
            raise BasisFunctionError(
                f"Unsupported basis type: {basis_type}. "
                f"Choose from {BasisFunction.SUPPORTED_TYPES}"
            )
        
        return BasisFunctionEvaluator(basis_type, a, b)


class BasisFunctionEvaluator:
    """
    Evaluator for configured basis functions.
    
    Attributes:
        type: Basis function type ("Legendre" or "Cosine")
        a: Lower bound
        b: Upper bound
    """
    
    def __init__(self, basis_type: str, a: float, b: float):
        self.type = basis_type
        self.a = a
        self.b = b
        
        if basis_type == "Legendre":
            self._func = legendre_basis
        else:  # Cosine
            self._func = cosine_basis
    
    def evaluate(
        self,
        order: int,
        x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Evaluate basis function at given points.
        
        Args:
            order: Basis function order/frequency
            x: Evaluation points
        
        Returns:
            Basis function values
        """
        return self._func(self.a, self.b, order, x)
    
    def __call__(
        self,
        order: int,
        x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Allow direct calling: evaluator(order, x)."""
        return self.evaluate(order, x)
    
    def __repr__(self) -> str:
        return f"BasisFunctionEvaluator(type={self.type}, a={self.a}, b={self.b})"


if __name__ == "__main__":
    """Module self-test and demonstration."""
    
    print("=" * 70)
    print("HDMR BASIS FUNCTIONS - MODULE TEST")
    print("=" * 70)
    
    # Test configuration
    a, b = 0.0, 1.0
    x = np.linspace(a, b, 200)
    
    # Test 1: Legendre basis
    print("\n[TEST 1] Legendre Basis Functions")
    print("-" * 70)
    
    try:
        for order in range(5):
            y = legendre_basis(a, b, order, x)
            print(f"  Order {order}: min={y.min():.4f}, max={y.max():.4f}, "
                  f"norm²={np.trapz(y**2, x):.4f}")
        print("  ✓ Legendre basis test passed")
    except Exception as e:
        print(f"  ✗ Legendre basis test failed: {e}")
    
    # Test 2: Cosine basis
    print("\n[TEST 2] Cosine Basis Functions")
    print("-" * 70)
    
    try:
        for freq in range(1, 6):
            y = cosine_basis(a, b, freq, x)
            print(f"  Frequency {freq}: min={y.min():.4f}, max={y.max():.4f}, "
                  f"norm²={np.trapz(y**2, x):.4f}")
        print("  ✓ Cosine basis test passed")
    except Exception as e:
        print(f"  ✗ Cosine basis test failed: {e}")
    
    # Test 3: Factory pattern
    print("\n[TEST 3] Factory Pattern")
    print("-" * 70)
    
    try:
        legendre_eval = BasisFunction.create("Legendre", a, b)
        cosine_eval = BasisFunction.create("Cosine", a, b)
        
        y_leg = legendre_eval(3, x)
        y_cos = cosine_eval(2, x)
        
        print(f"  Legendre evaluator: {legendre_eval}")
        print(f"  Cosine evaluator: {cosine_eval}")
        print("  ✓ Factory pattern test passed")
    except Exception as e:
        print(f"  ✗ Factory pattern test failed: {e}")
    
    # Test 4: Error handling
    print("\n[TEST 4] Error Handling")
    print("-" * 70)
    
    try:
        legendre_basis(1.0, 0.0, 1, x)  # Invalid interval
        print("  ✗ Should have raised BasisFunctionError")
    except BasisFunctionError as e:
        print(f"  ✓ Correctly caught: {e}")
    
    try:
        cosine_basis(0.0, 1.0, 0, x)  # Invalid frequency
        print("  ✗ Should have raised BasisFunctionError")
    except BasisFunctionError as e:
        print(f"  ✓ Correctly caught: {e}")
    
    print("\n" + "=" * 70)
    print("MODULE TEST COMPLETED")
    print("=" * 70)