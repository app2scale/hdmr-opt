"""
HDMR-Based Global Optimization Main Module (Refactored)

A robust and professional implementation of a first-order HDMR optimizer with
optional adaptive refinement. Designed to be used via CLI, imported as a module,
or executed in Streamlit safely (no reliance on brittle global state).

Key Fixes vs. previous version
------------------------------
- Always returns an OptimizeResult (no None results from minimize).
- Removes unsafe global/closure dependencies (e.g., alpha not defined).
- Correctly supports SciPy minimize contract: fun(x: (n,)) -> float.
- Eliminates the bug where one_dim objective depended on outer-loop "i".
- Visualization is optional and cannot crash the main optimization path.
- Adaptive refinement logic is consistent (bounds + resampling per iteration).

Author: HDMR Optimization Research Group
Refactor: Senior-style stabilization
Version: 3.0.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult, minimize

from src.basis_functions import Legendre, Cosine
import src.functions as functions

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


# =============================================================================
# Configuration & Utilities
# =============================================================================

@dataclass(frozen=True)
class HDMRConfig:
    # Core problem
    n: int
    a: Union[float, NDArray[np.float64]]
    b: Union[float, NDArray[np.float64]]

    # HDMR sampling / approximation
    N: int = 1000
    m: int = 7
    basis: str = "Cosine"  # "Legendre" or "Cosine"
    seed: Optional[int] = None

    # Optimization / refinement
    adaptive: bool = False
    maxiter: int = 25
    k: int = 100
    epsilon: float = 1e-2
    clip: float = 0.9

    # Runtime behavior
    disp: bool = False
    enable_plots: bool = True
    interactive_3d: bool = False  # if True, tries plotly for 2D

    def bounds_as_vectors(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return per-dimension bounds arrays."""
        if np.isscalar(self.a):
            a_vec = np.full(self.n, float(self.a), dtype=np.float64)
        else:
            a_vec = np.array(self.a, dtype=np.float64).reshape(self.n)

        if np.isscalar(self.b):
            b_vec = np.full(self.n, float(self.b), dtype=np.float64)
        else:
            b_vec = np.array(self.b, dtype=np.float64).reshape(self.n)

        if a_vec.shape != (self.n,) or b_vec.shape != (self.n,):
            raise ValueError("Bounds must be scalar or per-dimension arrays of shape (n,).")

        if np.any(b_vec <= a_vec):
            raise ValueError("Each bound must satisfy b[i] > a[i].")

        return a_vec, b_vec


def _ensure_2d(X: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Ensure input is (N, n). Accepts (n,), (N,n)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        if X.shape[0] != n:
            raise ValueError(f"Expected shape (n,) with n={n}, got {X.shape}.")
        return X.reshape(1, n)
    if X.ndim == 2:
        if X.shape[1] != n:
            raise ValueError(f"Expected shape (N,n) with n={n}, got {X.shape}.")
        return X
    raise ValueError(f"Unsupported input ndim={X.ndim}. Expected 1D or 2D.")


def _to_float(y: NDArray[np.float64]) -> float:
    """Convert various function outputs to scalar float."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 1:
        return float(y.reshape(-1)[0])
    # If user accidentally returns vector for single point, use first element.
    return float(y.reshape(-1)[0])


def _safe_call(fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
               X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Call objective safely and normalize output to shape (N,1)."""
    y = fun(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        y = y.reshape(1, 1)
    elif y.ndim == 1:
        # assume (N,)
        y = y.reshape(-1, 1)
    elif y.ndim == 2:
        # accept (N,1) or (N,k) -> use first column
        if y.shape[1] != 1:
            y = y[:, [0]]
    else:
        raise ValueError("Objective must return scalar, (N,), or (N,1)-like output.")
    return y

def parse_x0(x0_raw: Union[None, str, Sequence[float], NDArray[np.float64]],n: int,*,allow_broadcast: bool = True,allow_pattern_repeat: bool = True,max_repeat_seed: int = 4) -> NDArray[np.float64]:
    """
    Parse initial point input into shape (n,) float array.

    - None -> zeros(n)
    - list/tuple/np.ndarray -> validated/broadcasted
    - string -> parsed numbers -> validated/broadcasted

    Broadcasting rules (if enabled):
      - 1 value -> repeated to length n
      - 2..max_repeat_seed values -> repeated pattern to length n (optional)
    """
    def _finalize(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)

        if arr.size == n:
            return arr

        if not allow_broadcast:
            raise ValueError(f"x0 must have {n} values, got {arr.size}")

        # 1 value -> broadcast to all dims
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=np.float64)

        # k values -> repeat pattern if allowed
        if allow_pattern_repeat and 2 <= arr.size <= max_repeat_seed and arr.size < n:
            reps = int(np.ceil(n / arr.size))
            return np.tile(arr, reps)[:n].astype(np.float64)

        raise ValueError(
            f"x0 must have {n} values, got {arr.size}. "
            f"Tip: provide {n} comma-separated numbers (e.g., '0,0,...'), "
            f"or a single value to broadcast (e.g., '0')."
        )

    if x0_raw is None:
        return np.zeros(n, dtype=np.float64)

    # Numeric sequence / array
    if isinstance(x0_raw, (list, tuple, np.ndarray)):
        arr = np.asarray(x0_raw, dtype=np.float64)
        return _finalize(arr)

    # String
    if isinstance(x0_raw, str):
        s = x0_raw.strip().strip("[]()")
        parts = re.split(r"[,\s;]+", s)
        parts = [p for p in parts if p != ""]
        try:
            arr = np.array([float(p) for p in parts], dtype=np.float64)
        except ValueError as e:
            raise ValueError(
                f"Invalid x0 string: '{x0_raw}'. Examples: "
                f"'2 2', '2,2', '2.0, 2.0', '[2.0,2.0]', or single value '0'."
            ) from e
        return _finalize(arr)

    raise TypeError(f"Unsupported x0 type: {type(x0_raw)}")

# =============================================================================
# HDMR Optimizer
# =============================================================================

class HDMROptimizer:
    """
    First-order HDMR optimizer (sum of 1D components) with optional adaptive refinement.

    The surrogate is:
        f_hat(x) = f0 + sum_i sum_r alpha[r,i] * Phi_r(x_i)

    This implementation:
    - Estimates alpha via Monte Carlo projection.
    - Minimizes each 1D surrogate independently.
    - Recombines into a candidate vector x*.

    Adaptive mode:
    - Iteratively shrinks the bounds around the best candidate using k-nearest samples.
    - Resamples within new bounds and repeats until convergence.
    """

    def __init__(self, fun_batch: Callable[[NDArray[np.float64]], NDArray[np.float64]], config: HDMRConfig):
        self.fun_batch = fun_batch
        self.cfg = config
        self.a_vec, self.b_vec = config.bounds_as_vectors()

        if config.basis not in {"Legendre", "Cosine"}:
            raise ValueError("basis must be 'Legendre' or 'Cosine'")

        self.BasisFunction = Legendre if config.basis == "Legendre" else Cosine

        if config.seed is not None:
            np.random.seed(config.seed)

        # populated during run
        self.xs: Optional[NDArray[np.float64]] = None
        self.alpha: Optional[NDArray[np.float64]] = None
        self.f0: Optional[float] = None

        # figures (optional)
        self.fig_results: Optional[plt.Figure] = None
        self.fig_3d: Optional[plt.Figure] = None
        self.fig_alpha: Optional[plt.Figure] = None

    # --------------------------
    # Sampling
    # --------------------------

    def _sample_uniform(self, a_vec: NDArray[np.float64], b_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        N, n = self.cfg.N, self.cfg.n
        U = np.random.random((N, n))
        return (b_vec - a_vec) * U + a_vec

    # --------------------------
    # Basis evaluation (robust)
    # --------------------------

    def _basis_eval(self, r: int, i: int, x_col: NDArray[np.float64], a_i: float, b_i: float) -> NDArray[np.float64]:
        """
        Evaluate r-th basis on dimension i.
        Expects x_col shape (N,1). Returns shape (N,1).
        """
        # Your basis functions appear to be called as BasisFunction(a, b, degree, x)
        # where x may be (N,1) or scalar. Normalize to (N,1).
        x_col = np.asarray(x_col, dtype=np.float64).reshape(-1, 1)
        out = self.BasisFunction(a_i, b_i, r, x_col)
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        elif out.ndim == 2 and out.shape[1] != 1:
            out = out[:, [0]]
        return out

    # --------------------------
    # HDMR coefficient estimation
    # --------------------------

    def _fit_alpha(self, xs: NDArray[np.float64], a_vec: NDArray[np.float64], b_vec: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
        """
        Estimate (f0, alpha) for first-order HDMR on samples xs.

        alpha[r-1,i] = (b_i-a_i) * E[(f(xs)-f0)*Phi_r(xs_i)]
        """
        y = _safe_call(self.fun_batch, xs)  # (N,1)
        f0 = float(np.mean(y))
        yc = y - f0

        m, n = self.cfg.m, self.cfg.n
        alpha = np.zeros((m, n), dtype=np.float64)

        for i in range(n):
            x_col = xs[:, [i]]
            for r in range(1, m + 1):
                phi = self._basis_eval(r=r, i=i, x_col=x_col, a_i=a_vec[i], b_i=b_vec[i])
                alpha[r - 1, i] = (b_vec[i] - a_vec[i]) * float(np.mean(yc * phi))

        return f0, alpha

    # --------------------------
    # Surrogate evaluation
    # --------------------------

    def _surrogate_1d(self,x_scalar: float,i: int,f0: float,alpha: NDArray[np.float64],a_vec: NDArray[np.float64],b_vec: NDArray[np.float64],) -> float:
        """
        1D component objective used by SciPy for dimension i.
    
        Hardening:
        - Bounds guard (soft penalty outside [a_i, b_i])
        - Finite checks (avoid NaN/Inf propagating into SciPy numdiff)
        - Stable fallback penalty on numerical overflow/invalid basis eval
        """
        a_i = float(a_vec[i])
        b_i = float(b_vec[i])
    
        # Soft bounds handling: allow BFGS to probe, but penalize outside range
        penalty = 0.0
        x = float(x_scalar)
        if x < a_i:
            d = a_i - x
            penalty = 1e6 * d * d
            x = a_i
        elif x > b_i:
            d = x - b_i
            penalty = 1e6 * d * d
            x = b_i
    
        x_col = np.array([[x]], dtype=np.float64)
    
        val = 0.0
        try:
            for r in range(1, self.cfg.m + 1):
                phi = self._basis_eval(r=r, i=i, x_col=x_col, a_i=a_i, b_i=b_i)
    
                # basis eval safety
                if phi is None or phi.size == 0:
                    return 1e30
    
                phi00 = float(phi[0, 0])
                if not np.isfinite(phi00):
                    return 1e30
    
                val += float(alpha[r - 1, i]) * phi00
    
            val = float(val) + float(penalty)
    
            # final safety
            if not np.isfinite(val):
                return 1e30
    
            return val
    
        except FloatingPointError:
            # in case numpy is set to raise on overflow/invalid
            return 1e30
        except Exception:
            # never let SciPy receive NaN/Inf or crash on occasional basis issues
            return 1e30

    def _surrogate_full(self, X: NDArray[np.float64], f0: float, alpha: NDArray[np.float64],
                        a_vec: NDArray[np.float64], b_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate full surrogate f_hat(X) for X shape (N,n). Returns (N,1).
        """
        X = _ensure_2d(X, self.cfg.n)
        N = X.shape[0]
        yhat = np.full((N, 1), f0, dtype=np.float64)

        for i in range(self.cfg.n):
            x_col = X[:, [i]]
            for r in range(1, self.cfg.m + 1):
                phi = self._basis_eval(r=r, i=i, x_col=x_col, a_i=a_vec[i], b_i=b_vec[i])
                yhat += alpha[r - 1, i] * phi

        return yhat

    # --------------------------
    # Core solve step
    # --------------------------

    def _solve_once(self, x0: NDArray[np.float64], a_vec: NDArray[np.float64], b_vec: NDArray[np.float64]) -> OptimizeResult:
        """
        One HDMR pass: sample -> fit alpha -> minimize each dimension -> evaluate true objective.
        """
        xs = self._sample_uniform(a_vec, b_vec)
        f0, alpha = self._fit_alpha(xs, a_vec, b_vec)

        # minimize each 1D surrogate independently
        x_star = np.zeros(self.cfg.n, dtype=np.float64)
        for i in range(self.cfg.n):
            res_i = minimize(
                lambda z, idx=i: self._surrogate_1d(float(np.atleast_1d(z)[0]), idx, f0, alpha, a_vec, b_vec),
                x0=np.array([x0[i]], dtype=np.float64),
                method="BFGS",
                options={"disp": False}
            )
            x_star[i] = float(np.atleast_1d(res_i.x)[0])

        # clip to bounds (safety)
        x_star = np.minimum(np.maximum(x_star, a_vec), b_vec)

        fx = _safe_call(self.fun_batch, x_star.reshape(1, -1))[0, 0]

        # Persist last-fit state for optional plotting
        self.xs = xs
        self.alpha = alpha
        self.f0 = f0

        return OptimizeResult(
            x=x_star,
            fun=float(fx),
            success=True,
            message="HDMR optimization completed",
            nfev=int(self.cfg.N),  # surrogate fitting calls the batch objective N times effectively
            nit=1,
        )

    # --------------------------
    # Adaptive loop
    # --------------------------

    def solve(self, x0: ArrayLike) -> OptimizeResult:
        """
        Run optimization (adaptive or standard) and optionally generate plots.
        Always returns an OptimizeResult (never None).
        """
        x0 = np.asarray(x0, dtype=np.float64).reshape(self.cfg.n)
        a_vec, b_vec = self.a_vec.copy(), self.b_vec.copy()

        if not self.cfg.adaptive:
            result = self._solve_once(x0=x0, a_vec=a_vec, b_vec=b_vec)
            result.nit = 1
            if self.cfg.enable_plots:
                self._try_make_plots(a_vec=a_vec, b_vec=b_vec)
            return result

        # Adaptive mode
        if self.cfg.disp:
            print("\n" + "=" * 70)
            print("ADAPTIVE HDMR OPTIMIZATION")
            print("=" * 70)

        best = None
        prev_x = x0.copy()
        total_nfev = 0

        for it in range(1, self.cfg.maxiter + 1):
            res = self._solve_once(x0=prev_x, a_vec=a_vec, b_vec=b_vec)
            total_nfev += res.nfev

            if best is None or res.fun < best.fun:
                best = res

            if self.cfg.disp:
                print(f"Iteration {it}: x*={res.x}  f(x*)={res.fun:.6f}")
                print(f"  bounds a={a_vec}  b={b_vec}")

            # convergence on x movement
            if np.linalg.norm(res.x - prev_x) < self.cfg.epsilon:
                if self.cfg.disp:
                    print("✓ Convergence achieved.")
                best.nit = it
                best.nfev = total_nfev
                if self.cfg.enable_plots:
                    self._try_make_plots(a_vec=a_vec, b_vec=b_vec)
                return best

            # refine bounds using k nearest samples from last xs
            xs = self.xs
            if xs is None:
                # should not happen, but do not crash
                break

            new_a, new_b = self._refine_bounds(res.x, xs, a_vec, b_vec)
            a_vec, b_vec = new_a, new_b
            prev_x = res.x.copy()

        # maxiter reached
        if best is None:
            # ultimate fallback: return something safe
            fx = _safe_call(self.fun_batch, x0.reshape(1, -1))[0, 0]
            best = OptimizeResult(x=x0, fun=float(fx), success=False, message="HDMR failed", nfev=0, nit=0)

        best.success = True
        best.message = f"HDMR adaptive optimization completed (maxiter={self.cfg.maxiter})"
        best.nfev = total_nfev
        best.nit = self.cfg.maxiter

        if self.cfg.enable_plots:
            self._try_make_plots(a_vec=a_vec, b_vec=b_vec)

        return best

    def _refine_bounds(self, x_center: NDArray[np.float64], xs: NDArray[np.float64],
                       old_a: NDArray[np.float64], old_b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Shrink bounds based on k-nearest samples to x_center, with clip guard."""
        k = min(self.cfg.k, xs.shape[0])
        d = np.linalg.norm(xs - x_center.reshape(1, -1), axis=1)
        idx = np.argsort(d)[:k]
        near = xs[idx]

        new_a = np.min(near, axis=0)
        new_b = np.max(near, axis=0)

        # clip guard: do not shrink below clip * previous range
        for i in range(self.cfg.n):
            old_range = old_b[i] - old_a[i]
            min_range = self.cfg.clip * old_range
            if (new_b[i] - new_a[i]) < min_range:
                mid = 0.5 * (old_a[i] + old_b[i])
                new_a[i] = mid - 0.5 * min_range
                new_b[i] = mid + 0.5 * min_range

        # enforce original absolute bounds (safety)
        abs_a, abs_b = self.a_vec, self.b_vec
        new_a = np.maximum(new_a, abs_a)
        new_b = np.minimum(new_b, abs_b)

        # ensure valid
        new_b = np.maximum(new_b, new_a + 1e-12)

        return new_a, new_b

    # --------------------------
    # Plotting (safe / optional)
    # --------------------------

    def _try_make_plots(self, a_vec: NDArray[np.float64], b_vec: NDArray[np.float64]) -> None:
        """Try plotting; never raises to the caller."""
        try:
            if self.xs is None or self.alpha is None or self.f0 is None:
                return
            self.fig_results = self._plot_results()
            self.fig_alpha = self._plot_alpha_squared()
            if self.cfg.n == 2:
                self.fig_3d = self._plot_3d(a_vec=a_vec, b_vec=b_vec)
        except Exception as e:
            # Critical: do not break Streamlit or CLI execution.
            if self.cfg.disp:
                print(f"DEBUG: Visualization failed: {e}")
            self.fig_results = self.fig_alpha = self.fig_3d = None

    def _plot_results(self) -> plt.Figure:
        xs = self.xs
        alpha = self.alpha
        f0 = self.f0
        assert xs is not None and alpha is not None and f0 is not None

        y_true = _safe_call(self.fun_batch, xs)  # (N,1)
        y_hat = self._surrogate_full(xs, f0, alpha, self.a_vec, self.b_vec)

        n = self.cfg.n
        fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(14, 4 * n), squeeze=False)

        for i in range(n):
            axs[i, 0].scatter(xs[:, i], y_true[:, 0], alpha=0.35, s=12, label="Ground Truth")
            axs[i, 0].set_title(f"Dimension x{i+1}: Target Response")
            axs[i, 0].set_xlabel(f"x{i+1}")
            axs[i, 0].set_ylabel("f(x)")
            axs[i, 0].grid(True, linestyle="--", alpha=0.5)
            axs[i, 0].legend(loc="best")

            axs[i, 1].scatter(xs[:, i], y_hat[:, 0], alpha=0.35, s=12, label="HDMR Surrogate")
            axs[i, 1].set_title(f"Dimension x{i+1}: HDMR Approximation")
            axs[i, 1].set_xlabel(f"x{i+1}")
            axs[i, 1].set_ylabel("f̂(x)")
            axs[i, 1].grid(True, linestyle="--", alpha=0.5)
            axs[i, 1].legend(loc="best")

        fig.tight_layout(pad=2.0)
        return fig

    def _plot_alpha_squared(self) -> plt.Figure:
        alpha = self.alpha
        assert alpha is not None

        m, n = alpha.shape
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n):
            ax.plot(range(1, m + 1), (alpha[:, i] ** 2), marker="o", linewidth=2, label=f"x{i+1}")
        ax.set_xlabel("Basis degree")
        ax.set_ylabel("alpha^2")
        ax.set_title("HDMR coefficient magnitudes by variable")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_yscale("log")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _plot_3d(self, a_vec: NDArray[np.float64], b_vec: NDArray[np.float64]) -> plt.Figure:
        # Matplotlib-only safe default
        xs = self.xs
        alpha = self.alpha
        f0 = self.f0
        assert xs is not None and alpha is not None and f0 is not None

        y_hat = self._surrogate_full(xs, f0, alpha, a_vec, b_vec)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], y_hat[:, 0], alpha=0.35, s=6, label="HDMR")

        # True surface
        x1 = np.linspace(a_vec[0], b_vec[0], 100)
        x2 = np.linspace(a_vec[1], b_vec[1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        grid = np.column_stack((X1.ravel(), X2.ravel()))
        Y = _safe_call(self.fun_batch, grid).reshape(X1.shape)
        ax.plot_surface(X1, X2, Y, alpha=0.6)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        ax.set_title("HDMR vs True Function (2D)")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig


# =============================================================================
# Public API
# =============================================================================

def main_function(
    N_: int,
    n_: int,
    function_name_: str,
    basis_function_: str,
    m_: int,
    a_: float,
    b_: float,
    random_init_: bool,
    x0_: Optional[Union[str, Sequence[float], NDArray[np.float64]]] = None,
    is_adaptive_: bool = False,
    k_: int = 100,
    epsilon_: float = 0.1,
    clip_: float = 0.9,
    number_of_runs_: int = 1,
    maxiter_: int = 25,
    disp_: bool = False,
    enable_plots_: bool = True,
    seed_: Optional[int] = None,
) -> Tuple[List[OptimizeResult], float, Optional[plt.Figure], Optional[plt.Figure], Optional[plt.Figure], str]:
    """
    Configure and run HDMR optimization on a benchmark function from src.functions.

    Returns:
      (results, runtime, fig_results, fig_3d, fig_alpha, file_name_prefix)
    """
    if not hasattr(functions, function_name_):
        raise ValueError(f"Unknown function '{function_name_}' in src.functions")

    fun = getattr(functions, function_name_)

    # Batch wrapper: accepts (N,n) and returns (N,1)
    def fun_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = _ensure_2d(X, n_)
        return _safe_call(fun, X)

    cfg = HDMRConfig(
        n=n_,
        a=a_,
        b=b_,
        N=N_,
        m=m_,
        basis=basis_function_,
        seed=seed_,
        adaptive=is_adaptive_,
        maxiter=maxiter_,
        k=k_,
        epsilon=epsilon_,
        clip=clip_,
        disp=disp_,
        enable_plots=enable_plots_,
    )

    file_name = (
        f"results/adaptive_{function_name_}_a{a_}_b{b_}_N{N_}_m{m_}_k{k_}_c{clip_:.2f}"
        if is_adaptive_
        else f"results/{function_name_}_a{a_}_b{b_}_N{N_}_m{m_}"
    )

    results: List[OptimizeResult] = []
    start = time.time()
    
    for run_idx in range(number_of_runs_):
        opt = HDMROptimizer(fun_batch=fun_batch, config=cfg)
    
        if random_init_:
            a_vec, b_vec = cfg.bounds_as_vectors()
            x0 = (b_vec - a_vec) * np.random.random(cfg.n) + a_vec
        else:
            x0 = parse_x0(x0_, n_)
    
        res = opt.solve(x0)
        results.append(res)
    
        if disp_ and number_of_runs_ > 1:
            print(f"Run {run_idx+1}/{number_of_runs_}: x*={res.x} f(x*)={res.fun:.6f}")
    
        if number_of_runs_ == 1:
            fig_results = opt.fig_results
            fig_3d = opt.fig_3d
            fig_alpha = opt.fig_alpha
        else:
            fig_results = fig_3d = fig_alpha = None
    
    runtime = time.time() - start
    return results, runtime, fig_results, fig_3d, fig_alpha, file_name


# =============================================================================
# CLI
# =============================================================================

def _load_bounds_from_json(function_name: str, path: str = "src/function_ranges.json") -> Tuple[float, float]:
    with open(path, "r") as f:
        ranges = json.load(f)
    if function_name not in ranges:
        raise KeyError(function_name)
    a_, b_ = ranges[function_name]
    return float(a_), float(b_)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="HDMR",
        description="High Dimensional Model Representation optimization tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--numSamples", type=int, required=True, help="Number of samples (100-10000)")
    parser.add_argument("--numVariables", type=int, required=True, help="Number of dimensions/variables")
    parser.add_argument("--function", required=True, help="Test function name (e.g., rastrigin_2d)")

    parser.add_argument("--min", type=float, default=None, help="Lower bound")
    parser.add_argument("--max", type=float, default=None, help="Upper bound")

    parser.add_argument("--x0", nargs="+", type=float, default=None, help="Initial point (space-separated)")
    parser.add_argument("--randomInit", action="store_true", help="Use random initialization")

    parser.add_argument("--basisFunction", type=str, default="Cosine", choices=["Legendre", "Cosine"])
    parser.add_argument("--degree", type=int, default=7, help="Number of basis functions (default: 7)")

    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive refinement")
    parser.add_argument("--maxiter", type=int, default=25, help="Max adaptive iterations")
    parser.add_argument("--numClosestPoints", type=int, default=100, help="k for adaptive")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Convergence tolerance")
    parser.add_argument("--clip", type=float, default=0.9, help="Min shrink ratio (0<clip<=1)")

    parser.add_argument("--numberOfRuns", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noPlots", action="store_true", help="Disable plots")
    parser.add_argument("--disp", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Bounds
    if args.min is None or args.max is None:
        try:
            a_, b_ = _load_bounds_from_json(args.function)
        except Exception:
            print(f"Error: Could not load bounds for {args.function}. Provide --min and --max.")
            sys.exit(1)
    else:
        a_, b_ = float(args.min), float(args.max)

    print("\n" + "=" * 70)
    print("HDMR OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    print(f"Function:       {args.function}")
    print(f"Dimensions:     {args.numVariables}")
    print(f"Samples:        {args.numSamples}")
    print(f"Basis:          {args.basisFunction} (degree {args.degree})")
    print(f"Bounds:         [{a_}, {b_}]")
    print(f"Adaptive:       {args.adaptive}")
    if args.adaptive:
        print(f"  maxiter:      {args.maxiter}")
        print(f"  k:            {args.numClosestPoints}")
        print(f"  epsilon:      {args.epsilon}")
        print(f"  clip:         {args.clip}")
    print(f"Runs:           {args.numberOfRuns}")
    print("=" * 70 + "\n")

    results, runtime, fig1, fig2, fig3, file_name = main_function(
        N_=args.numSamples,
        n_=args.numVariables,
        function_name_=args.function,
        basis_function_=args.basisFunction,
        m_=args.degree,
        a_=a_,
        b_=b_,
        random_init_=args.randomInit,
        x0_=args.x0,
        is_adaptive_=args.adaptive,
        k_=args.numClosestPoints,
        epsilon_=args.epsilon,
        clip_=args.clip,
        number_of_runs_=args.numberOfRuns,
        maxiter_=args.maxiter,
        disp_=args.disp,
        enable_plots_=not args.noPlots,
        seed_=args.seed,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Runtime:        {runtime:.4f} seconds")

    if args.numberOfRuns == 1:
        r = results[0]
        print(f"Solution:       {r.x}")
        print(f"Objective:      {r.fun:.6f}")
        print(f"Evaluations:    {r.nfev}")
        print(f"Iterations:     {getattr(r, 'nit', 1)}")
    else:
        # optional: user can implement external optimum stats
        vals = np.array([r.fun for r in results], dtype=np.float64)
        print(f"Mean f(x*):     {vals.mean():.6f}")
        print(f"Std  f(x*):     {vals.std():.6f}")
        print(f"Best f(x*):     {vals.min():.6f}")
        print(f"Worst f(x*):    {vals.max():.6f}")

    print("=" * 70 + "\n")

    os.makedirs("results", exist_ok=True)

    with open(file_name + ".txt", "w") as f:
        f.write("HDMR OPTIMIZATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"Run {i}:\n{r}\n\n")

    print(f"✓ Results saved to: {file_name}.txt")

    if args.numberOfRuns == 1 and not args.noPlots and fig1 is not None:
        fig1.savefig(file_name + "_results.png", dpi=300, bbox_inches="tight")
        print(f"✓ Plots saved to: {file_name}_results.png")
        if fig3 is not None:
            fig3.savefig(file_name + "_alpha.png", dpi=300, bbox_inches="tight")
            print(f"✓ Alpha plot saved to: {file_name}_alpha.png")
        if fig2 is not None:
            fig2.savefig(file_name + "_3d.png", dpi=300, bbox_inches="tight")
            print(f"✓ 3D plot saved to: {file_name}_3d.png")
        plt.show()

    print("\n✓ Optimization completed successfully!\n")


if __name__ == "__main__":
    # Reduce noise for CLI users; Streamlit can configure warnings separately.
    warnings.filterwarnings("once", category=RuntimeWarning)
    cli_main()
