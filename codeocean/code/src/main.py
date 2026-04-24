"""
HDMR-Based Global Optimization — Version 4.0.0 (Enhanced Adaptive)

Mathematical Improvements over v3.0.0
--------------------------------------
1. Sensitivity-Guided Bound Refinement (_refine_bounds_sensitivity):
   - Combines elite-sample center with Sobol-index-based shrinkage rates
   - Smooth shrinkage: sensitive dims stay wide, insensitive dims collapse fast
   - Insensitive dims are frozen at best point (no wasted budget)

2. Grid + Local 1D Minimization (_minimize_surrogate_1d):
   - Dense grid scan first → escapes poor BFGS starting points
   - Local BFGS refinement from best grid candidate
   - Eliminates surrogate minimization failures that plagued v3

3. Sobol Quasi-Random Sampling (_sample_quasi_random):
   - Low-discrepancy sequences instead of pseudo-random uniform
   - Better space coverage with same N → improved alpha estimates
   - Falls back to uniform if scipy.stats.qmc unavailable

4. Sensitivity-Focused Resampling (_sample_focused):
   - After first iteration: concentrate extra samples in top-k sensitive dims
   - Insensitive dims fixed at current best → lower effective dimensionality
   - Reduces noise in alpha estimation for dominant dimensions

5. Multi-Start Surrogate Minimization:
   - Run 1D minimization from multiple starting points per dimension
   - Select best across starts → more robust x* recovery

6. Conservative Convergence:
   - Track best-ever result (not just last iteration)
   - Convergence only if f-improvement < relative tolerance
   - Prevents premature termination

API Compatibility
-----------------
Fully backward compatible with v3.0.0. New parameters are optional with
sensible defaults. Same CLI interface.

Author: HDMR Optimization Research Group
Version: 4.0.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult, minimize

from src.basis_functions import Legendre, Cosine
import src.functions as functions

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


# =============================================================================
# Configuration
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
    basis: str = "Cosine"
    seed: Optional[int] = None

    # Optimization / refinement
    adaptive: bool = False
    maxiter: int = 25
    k: int = 100          # elite samples for bound refinement
    epsilon: float = 1e-2  # convergence: relative f-improvement threshold
    clip: float = 0.9      # minimum bound retention fraction (insensitive dims)

    # v4 enhancements
    use_quasi_random: bool = True    # Sobol quasi-random sampling
    use_focused_resample: bool = True  # sensitivity-focused resampling
    n_grid_1d: int = 500             # grid points for 1D surrogate scan
    n_multistarts_1d: int = 5        # multi-starts per dim in 1D minimization
    sensitivity_exponent: float = 0.5  # shrinkage exponent (0.5 = sqrt, smooth)
    min_shrink: float = 0.15         # minimum shrink factor for sensitive dims
    top_k_sensitive: int = 3         # dims to focus resampling on

    # Runtime behavior
    disp: bool = False
    enable_plots: bool = True
    interactive_3d: bool = False

    def bounds_as_vectors(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.isscalar(self.a):
            a_vec = np.full(self.n, float(self.a), dtype=np.float64)
        else:
            a_vec = np.array(self.a, dtype=np.float64).reshape(self.n)
        if np.isscalar(self.b):
            b_vec = np.full(self.n, float(self.b), dtype=np.float64)
        else:
            b_vec = np.array(self.b, dtype=np.float64).reshape(self.n)
        if np.any(b_vec <= a_vec):
            raise ValueError("Each bound must satisfy b[i] > a[i].")
        return a_vec, b_vec


# =============================================================================
# Utilities
# =============================================================================

def _ensure_2d(X: NDArray, n: int) -> NDArray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return X.reshape(1, n)
    return X


def _safe_call(fun: Callable, X: NDArray) -> NDArray:
    y = fun(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        y = y.reshape(1, 1)
    elif y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] != 1:
        y = y[:, [0]]
    return y


def parse_x0(x0_raw, n: int) -> NDArray:
    if x0_raw is None:
        return np.zeros(n, dtype=np.float64)
    if isinstance(x0_raw, (list, tuple, np.ndarray)):
        arr = np.asarray(x0_raw, dtype=np.float64).reshape(-1)
    elif isinstance(x0_raw, str):
        s = x0_raw.strip().strip("[]()")
        parts = [p for p in re.split(r"[,\s;]+", s) if p]
        arr = np.array([float(p) for p in parts], dtype=np.float64)
    else:
        raise TypeError(f"Unsupported x0 type: {type(x0_raw)}")
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size == n:
        return arr
    raise ValueError(f"x0 must have {n} values, got {arr.size}")


# =============================================================================
# HDMR Optimizer v4
# =============================================================================

class HDMROptimizer:
    """
    First-order HDMR optimizer with sensitivity-guided adaptive refinement.

    Surrogate model:
        f̂(x) = f₀ + Σᵢ Σᵣ αᵣᵢ · Φᵣ(xᵢ)

    Sobol first-order sensitivity index for dimension i:
        Sᵢ = Σᵣ αᵣᵢ² / Σᵢ Σᵣ αᵣᵢ²

    Adaptive bound shrinkage (v4):
        shrink_i = min_shrink + (1 - min_shrink) · Sᵢ^exponent
        → sensitive dims: wide bounds retained
        → insensitive dims: collapsed to best-point neighborhood
    """

    def __init__(
        self,
        fun_batch: Callable[[NDArray], NDArray],
        config: HDMRConfig,
    ):
        self.fun_batch = fun_batch
        self.cfg = config
        self.a_vec, self.b_vec = config.bounds_as_vectors()

        if config.basis not in {"Legendre", "Cosine"}:
            raise ValueError("basis must be 'Legendre' or 'Cosine'")
        self.BasisFunction = Legendre if config.basis == "Legendre" else Cosine

        if config.seed is not None:
            np.random.seed(config.seed)

        # state populated during solve
        self.xs: Optional[NDArray] = None
        self.alpha: Optional[NDArray] = None
        self.f0: Optional[float] = None
        self.sensitivity_indices_: Optional[NDArray] = None

        self.fig_results = self.fig_3d = self.fig_alpha = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_uniform(self, a_vec: NDArray, b_vec: NDArray) -> NDArray:
        N, n = self.cfg.N, self.cfg.n
        U = np.random.random((N, n))
        return (b_vec - a_vec) * U + a_vec

    def _sample_quasi_random(self, a_vec: NDArray, b_vec: NDArray) -> NDArray:
        """
        Sobol low-discrepancy sequence for better space coverage.
        Falls back to uniform if scipy.stats.qmc is unavailable.

        Why: Sobol sequences have discrepancy O((log N)^d / N) vs uniform's
        O(1/√N), meaning better coverage of the input space with fewer samples.
        This leads to lower-variance alpha estimates.
        """
        N, n = self.cfg.N, self.cfg.n
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=n, scramble=True)
            # Sobol requires power-of-2 samples; round up
            m_pow = int(np.ceil(np.log2(max(N, 2))))
            U = sampler.random_base2(m=m_pow)[:N, :]
        except Exception:
            U = np.random.random((N, n))
        return (b_vec - a_vec) * U + a_vec

    def _sample_focused(
        self,
        a_vec: NDArray,
        b_vec: NDArray,
        x_best: NDArray,
        sensitivity: NDArray,
    ) -> NDArray:
        """
        Sensitivity-focused sampling (v4 enhancement).

        Strategy:
        - Top-k sensitive dimensions: sample uniformly within [a_i, b_i]
        - Remaining insensitive dimensions: fix at x_best[i] ± small noise

        Mathematical justification:
        If Sᵢ ≈ 0, the surrogate component fᵢ(xᵢ) is negligible.
        Sampling these dimensions adds noise to α estimation without benefit.
        Fixing them at the current best reduces effective dimensionality,
        improving alpha quality for the sensitive dimensions.
        """
        N, n = self.cfg.N, self.cfg.n
        k = min(self.cfg.top_k_sensitive, n)

        top_dims = np.argsort(sensitivity)[-k:]   # indices of top-k sensitive dims
        low_dims = np.argsort(sensitivity)[:-k]   # remaining dims

        X = np.zeros((N, n), dtype=np.float64)

        # Sensitive dims: quasi-random within current bounds
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=k, scramble=True)
            m_pow = int(np.ceil(np.log2(max(N, 2))))
            U_top = sampler.random_base2(m=m_pow)[:N, :]
        except Exception:
            U_top = np.random.random((N, k))

        for j, i in enumerate(top_dims):
            X[:, i] = (b_vec[i] - a_vec[i]) * U_top[:, j] + a_vec[i]

        # Insensitive dims: x_best + small perturbation (1% of range)
        for i in low_dims:
            noise_scale = 0.01 * (b_vec[i] - a_vec[i])
            X[:, i] = x_best[i] + np.random.normal(0, noise_scale, N)
            X[:, i] = np.clip(X[:, i], a_vec[i], b_vec[i])

        return X

    # ------------------------------------------------------------------
    # Basis evaluation
    # ------------------------------------------------------------------

    def _basis_eval(
        self, r: int, i: int, x_col: NDArray, a_i: float, b_i: float
    ) -> NDArray:
        x_col = np.asarray(x_col, dtype=np.float64).reshape(-1, 1)
        out = self.BasisFunction(a_i, b_i, r, x_col)
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        elif out.ndim == 2 and out.shape[1] != 1:
            out = out[:, [0]]
        return out

    # ------------------------------------------------------------------
    # Alpha estimation
    # ------------------------------------------------------------------

    def _fit_alpha(
        self, xs: NDArray, a_vec: NDArray, b_vec: NDArray
    ) -> Tuple[float, NDArray]:
        """
        Monte Carlo projection of HDMR coefficients.

        αᵣᵢ = (bᵢ - aᵢ) · E[(f(x) - f₀) · Φᵣ(xᵢ)]

        The factor (bᵢ - aᵢ) corrects for the normalization convention so that
        sensitivity indices computed from α² are properly scaled.
        """
        y = _safe_call(self.fun_batch, xs)
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

    def _compute_sobol_indices(self, alpha: NDArray) -> NDArray:
        """
        First-order Sobol sensitivity indices from HDMR coefficients.

        Sᵢ = Var(fᵢ) / Var(f̂)  =  Σᵣ αᵣᵢ² / Σᵢ Σᵣ αᵣᵢ²

        These are exact for the first-order HDMR surrogate (no interaction terms).
        Returns uniform distribution if total variance is near zero (flat landscape).
        """
        var_i = np.sum(alpha ** 2, axis=0)         # shape (n,)
        total_var = np.sum(var_i)
        if total_var < 1e-15:
            return np.ones(self.cfg.n) / self.cfg.n  # uniform: no information
        return var_i / total_var

    # ------------------------------------------------------------------
    # 1D Surrogate minimization (v4: grid + local)
    # ------------------------------------------------------------------

    def _surrogate_1d_val(
        self,
        x_scalar: float,
        i: int,
        alpha: NDArray,
        a_vec: NDArray,
        b_vec: NDArray,
    ) -> float:
        """Evaluate 1D surrogate component fᵢ(xᵢ) at a scalar point."""
        a_i, b_i = float(a_vec[i]), float(b_vec[i])
        x = np.clip(float(x_scalar), a_i, b_i)
        x_col = np.array([[x]], dtype=np.float64)
        val = 0.0
        try:
            for r in range(1, self.cfg.m + 1):
                phi = self._basis_eval(r=r, i=i, x_col=x_col, a_i=a_i, b_i=b_i)
                phi00 = float(phi[0, 0])
                if not np.isfinite(phi00):
                    return 1e30
                val += float(alpha[r - 1, i]) * phi00
            return val if np.isfinite(val) else 1e30
        except Exception:
            return 1e30

    def _minimize_surrogate_1d(
        self,
        i: int,
        x0_i: float,
        alpha: NDArray,
        a_vec: NDArray,
        b_vec: NDArray,
    ) -> float:
        """
        Minimize surrogate component fᵢ(xᵢ) over [aᵢ, bᵢ].

        v4 strategy: Grid scan + local refinement
        1. Evaluate surrogate on dense 1D grid → identify best region
        2. Multi-start local BFGS from top-k grid points
        3. Return global best across all starts

        This is O(n_grid) evaluations of a cheap polynomial, so negligible
        cost vs actual objective evaluations. Avoids BFGS getting stuck in
        suboptimal local minima of the surrogate.
        """
        a_i, b_i = float(a_vec[i]), float(b_vec[i])
        n_grid = self.cfg.n_grid_1d
        n_starts = self.cfg.n_multistarts_1d

        # Step 1: Dense grid scan
        grid = np.linspace(a_i, b_i, n_grid)
        grid_vals = np.array([
            self._surrogate_1d_val(x, i, alpha, a_vec, b_vec)
            for x in grid
        ])

        best_x = float(grid[np.argmin(grid_vals)])
        best_val = float(np.min(grid_vals))

        # Step 2: Multi-start BFGS from top-k grid points
        top_idx = np.argsort(grid_vals)[:n_starts]
        starts = list(grid[top_idx])
        if x0_i not in starts:
            starts.append(x0_i)  # always include warm-start point

        for x_start in starts:
            try:
                res = minimize(
                    lambda z: self._surrogate_1d_val(
                        float(np.atleast_1d(z)[0]), i, alpha, a_vec, b_vec
                    ),
                    x0=np.array([x_start], dtype=np.float64),
                    method="L-BFGS-B",
                    bounds=[(a_i, b_i)],
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 200},
                )
                if res.success and float(res.fun) < best_val:
                    best_val = float(res.fun)
                    best_x = float(np.clip(np.atleast_1d(res.x)[0], a_i, b_i))
            except Exception:
                continue

        return float(np.clip(best_x, a_i, b_i))

    def _surrogate_full(
        self, X: NDArray, f0: float, alpha: NDArray, a_vec: NDArray, b_vec: NDArray
    ) -> NDArray:
        X = _ensure_2d(X, self.cfg.n)
        N = X.shape[0]
        yhat = np.full((N, 1), f0, dtype=np.float64)
        for i in range(self.cfg.n):
            x_col = X[:, [i]]
            for r in range(1, self.cfg.m + 1):
                phi = self._basis_eval(r=r, i=i, x_col=x_col, a_i=a_vec[i], b_i=b_vec[i])
                yhat += alpha[r - 1, i] * phi
        return yhat

    # ------------------------------------------------------------------
    # Bound refinement (v4: sensitivity-guided hybrid)
    # ------------------------------------------------------------------

    def _refine_bounds_sensitivity(
        self,
        x_best: NDArray,
        xs: NDArray,
        old_a: NDArray,
        old_b: NDArray,
        sensitivity: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Sensitivity-guided hybrid bound refinement (v4).

        Mathematical formulation:
            shrink_i = min_shrink + (1 - min_shrink) · Sᵢ^exponent

        where:
            Sᵢ  = normalized first-order Sobol index for dim i  ∈ [0, 1]
            exponent = sensitivity_exponent (default 0.5 = sqrt)
            min_shrink = minimum retention even for insensitive dims

        Resulting half-width for dimension i:
            new_half_width_i = shrink_i · old_half_width_i

        Center is the elite-sample mean (weighted toward high-performing region),
        clipped to the best point neighborhood for insensitive dims.

        Properties:
        - Sensitive dims (Sᵢ → 1): shrink_i → 1.0 → bounds barely shrink
        - Insensitive dims (Sᵢ → 0): shrink_i → min_shrink → aggressive collapse
        - Continuity: smooth function of Sᵢ (no hard thresholds)
        - Safety: always clip to original absolute bounds
        """
        n = self.cfg.n
        exponent = self.cfg.sensitivity_exponent
        min_shrink = self.cfg.min_shrink

        # Elite center: mean of top-k samples by objective value
        k = min(self.cfg.k, xs.shape[0])
        y_xs = _safe_call(self.fun_batch, xs).reshape(-1)
        elite_idx = np.argsort(y_xs)[:k]       # k lowest (minimization)
        elite_samples = xs[elite_idx]
        mu_elite = np.mean(elite_samples, axis=0)

        # Per-dim shrinkage rate based on Sobol index
        # shrink ∈ [min_shrink, 1.0]
        shrink = min_shrink + (1.0 - min_shrink) * (sensitivity ** exponent)

        new_a = np.zeros(n, dtype=np.float64)
        new_b = np.zeros(n, dtype=np.float64)

        for i in range(n):
            old_width = old_b[i] - old_a[i]
            new_half = shrink[i] * old_width / 2.0

            # Center: elite mean for sensitive dims, best point for insensitive
            # Smooth blend: sensitive → mu_elite, insensitive → x_best
            center = sensitivity[i] * mu_elite[i] + (1.0 - sensitivity[i]) * x_best[i]
            center = float(np.clip(center, old_a[i], old_b[i]))

            new_a[i] = center - new_half
            new_b[i] = center + new_half

            # Clip to absolute bounds
            new_a[i] = max(new_a[i], self.a_vec[i])
            new_b[i] = min(new_b[i], self.b_vec[i])

            # Ensure valid interval
            if new_b[i] - new_a[i] < 1e-12:
                mid = 0.5 * (old_a[i] + old_b[i])
                new_a[i] = max(mid - 1e-6, self.a_vec[i])
                new_b[i] = min(mid + 1e-6, self.b_vec[i])

        if self.cfg.disp:
            print(f"  Sensitivity: {np.round(sensitivity, 3)}")
            print(f"  Shrink rates: {np.round(shrink, 3)}")

        return new_a, new_b

    # Backward-compatible original method (kept for reference)
    def _refine_bounds(
        self,
        x_center: NDArray,
        xs: NDArray,
        old_a: NDArray,
        old_b: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Original k-nearest elite refinement (v3, kept for compatibility)."""
        k = min(self.cfg.k, xs.shape[0])
        d = np.linalg.norm(xs - x_center.reshape(1, -1), axis=1)
        idx = np.argsort(d)[:k]
        near = xs[idx]
        new_a = np.min(near, axis=0)
        new_b = np.max(near, axis=0)
        for i in range(self.cfg.n):
            old_range = old_b[i] - old_a[i]
            min_range = self.cfg.clip * old_range
            if (new_b[i] - new_a[i]) < min_range:
                mid = 0.5 * (old_a[i] + old_b[i])
                new_a[i] = mid - 0.5 * min_range
                new_b[i] = mid + 0.5 * min_range
        new_a = np.maximum(new_a, self.a_vec)
        new_b = np.minimum(new_b, self.b_vec)
        new_b = np.maximum(new_b, new_a + 1e-12)
        return new_a, new_b

    # ------------------------------------------------------------------
    # Core solve step
    # ------------------------------------------------------------------

    def _solve_once(
        self,
        x0: NDArray,
        a_vec: NDArray,
        b_vec: NDArray,
        use_focused: bool = False,
        x_best_prev: Optional[NDArray] = None,
        sensitivity_prev: Optional[NDArray] = None,
    ) -> Tuple[OptimizeResult, NDArray]:
        """
        One HDMR pass: sample → fit α → minimize each 1D surrogate → evaluate.

        Returns (result, sensitivity_indices).
        """
        # Sampling strategy
        if (use_focused
                and x_best_prev is not None
                and sensitivity_prev is not None
                and self.cfg.use_focused_resample):
            xs = self._sample_focused(a_vec, b_vec, x_best_prev, sensitivity_prev)
        elif self.cfg.use_quasi_random:
            xs = self._sample_quasi_random(a_vec, b_vec)
        else:
            xs = self._sample_uniform(a_vec, b_vec)

        # Fit alpha
        f0, alpha = self._fit_alpha(xs, a_vec, b_vec)
        sensitivity = self._compute_sobol_indices(alpha)

        # Minimize each 1D surrogate (v4: grid + multi-start)
        x_star = np.zeros(self.cfg.n, dtype=np.float64)
        for i in range(self.cfg.n):
            x_star[i] = self._minimize_surrogate_1d(
                i=i, x0_i=x0[i], alpha=alpha, a_vec=a_vec, b_vec=b_vec
            )

        x_star = np.clip(x_star, a_vec, b_vec)

        # Evaluate true objective at surrogate optimum
        fx = float(_safe_call(self.fun_batch, x_star.reshape(1, -1))[0, 0])

        # Persist state for plotting and next iteration
        self.xs = xs
        self.alpha = alpha
        self.f0 = f0
        self.sensitivity_indices_ = sensitivity

        res = OptimizeResult(
            x=x_star,
            fun=fx,
            success=True,
            message="HDMR step completed",
            nfev=int(self.cfg.N),
            nit=1,
        )
        return res, sensitivity

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self, x0: ArrayLike) -> OptimizeResult:
        """
        Run HDMR optimization (standard or adaptive).

        Adaptive v4 improvements:
        - Sensitivity-guided bound refinement (wider for sensitive dims)
        - Focused resampling after first iteration
        - Track best-ever result with relative f-improvement convergence
        - Always returns valid OptimizeResult
        """
        x0 = np.asarray(x0, dtype=np.float64).reshape(self.cfg.n)
        a_vec, b_vec = self.a_vec.copy(), self.b_vec.copy()

        if not self.cfg.adaptive:
            res, _ = self._solve_once(x0=x0, a_vec=a_vec, b_vec=b_vec)
            res.nit = 1
            if self.cfg.enable_plots:
                self._try_make_plots(a_vec=a_vec, b_vec=b_vec)
            return res

        # ── Adaptive mode ──────────────────────────────────────────────
        if self.cfg.disp:
            print("\n" + "=" * 70)
            print("ADAPTIVE HDMR v4 — SENSITIVITY-GUIDED OPTIMIZATION")
            print("=" * 70)

        best: Optional[OptimizeResult] = None
        best_f: float = np.inf
        prev_x = x0.copy()
        prev_sensitivity: Optional[NDArray] = None
        total_nfev = 0

        for it in range(1, self.cfg.maxiter + 1):
            use_focused = (it > 1)  # only after first pass has sensitivity info

            res, sensitivity = self._solve_once(
                x0=prev_x,
                a_vec=a_vec,
                b_vec=b_vec,
                use_focused=use_focused,
                x_best_prev=prev_x,
                sensitivity_prev=prev_sensitivity,
            )
            total_nfev += res.nfev

            if self.cfg.disp:
                top2 = np.argsort(sensitivity)[-2:][::-1]
                print(
                    f"Iter {it:2d}: f(x*)={res.fun:.6f} "
                    f"| top-2 sensitive dims: {top2.tolist()} "
                    f"(S={sensitivity[top2].round(3)})"
                )

            # Track best-ever
            if res.fun < best_f:
                best_f = res.fun
                best = res

            # Convergence: relative f-improvement below threshold
            if it > 1 and best is not None:
                rel_improvement = (prev_f - best_f) / (abs(prev_f) + 1e-10)
                if rel_improvement < self.cfg.epsilon:
                    if self.cfg.disp:
                        print(f"  [CONVERGED] at iter {it} (rel_improvement={rel_improvement:.2e})")
                    best.nit = it
                    best.nfev = total_nfev
                    if self.cfg.enable_plots:
                        self._try_make_plots(a_vec=a_vec, b_vec=b_vec)
                    return best

            prev_f = best_f

            # Refine bounds using v4 sensitivity-guided method
            new_a, new_b = self._refine_bounds_sensitivity(
                x_best=res.x,
                xs=self.xs,
                old_a=a_vec,
                old_b=b_vec,
                sensitivity=sensitivity,
            )
            a_vec, b_vec = new_a, new_b
            prev_x = res.x.copy()
            prev_sensitivity = sensitivity

        # Max iterations reached
        if best is None:
            fx = float(_safe_call(self.fun_batch, x0.reshape(1, -1))[0, 0])
            best = OptimizeResult(
                x=x0, fun=fx, success=False,
                message="HDMR: maxiter reached with no improvement",
                nfev=total_nfev, nit=self.cfg.maxiter,
            )

        best.success = True
        best.message = f"HDMR v4 adaptive completed (maxiter={self.cfg.maxiter})"
        best.nfev = total_nfev
        best.nit = self.cfg.maxiter

        if self.cfg.enable_plots:
            self._try_make_plots(a_vec=a_vec, b_vec=b_vec)

        return best

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _try_make_plots(self, a_vec: NDArray, b_vec: NDArray) -> None:
        try:
            if self.xs is None or self.alpha is None or self.f0 is None:
                return
            self.fig_results = self._plot_results()
            self.fig_alpha = self._plot_alpha_squared()
            if self.cfg.n == 2:
                self.fig_3d = self._plot_3d(a_vec=a_vec, b_vec=b_vec)
        except Exception as e:
            if self.cfg.disp:
                print(f"DEBUG: Visualization failed: {e}")
            self.fig_results = self.fig_alpha = self.fig_3d = None

    def _plot_results(self) -> plt.Figure:
        xs, alpha, f0 = self.xs, self.alpha, self.f0
        y_true = _safe_call(self.fun_batch, xs)
        y_hat = self._surrogate_full(xs, f0, alpha, self.a_vec, self.b_vec)
        n = self.cfg.n
        fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(14, 4 * n), squeeze=False)
        for i in range(n):
            axs[i, 0].scatter(xs[:, i], y_true[:, 0], alpha=0.3, s=10, label="True")
            axs[i, 0].set_title(f"Dim x{i+1}: True Response")
            axs[i, 0].grid(True, linestyle="--", alpha=0.4)
            axs[i, 1].scatter(xs[:, i], y_hat[:, 0], alpha=0.3, s=10,
                               label="HDMR", color="orange")
            sens_str = (f"  [S={self.sensitivity_indices_[i]:.3f}]"
                        if self.sensitivity_indices_ is not None else "")
            axs[i, 1].set_title(f"Dim x{i+1}: HDMR Surrogate{sens_str}")
            axs[i, 1].grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout(pad=2.0)
        return fig

    def _plot_alpha_squared(self) -> plt.Figure:
        alpha = self.alpha
        m, n = alpha.shape
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: alpha² by degree
        ax = axes[0]
        for i in range(n):
            label = f"x{i+1}"
            if self.sensitivity_indices_ is not None:
                label += f" (S={self.sensitivity_indices_[i]:.3f})"
            ax.plot(range(1, m + 1), alpha[:, i] ** 2, marker="o", linewidth=2, label=label)
        ax.set_xlabel("Basis degree r")
        ax.set_ylabel("αᵣᵢ²")
        ax.set_title("HDMR Coefficient Magnitudes")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

        # Right: Sobol sensitivity bar chart
        if self.sensitivity_indices_ is not None:
            ax2 = axes[1]
            dims = [f"x{i+1}" for i in range(n)]
            colors = plt.cm.RdYlGn(self.sensitivity_indices_)
            ax2.bar(dims, self.sensitivity_indices_, color=colors, edgecolor="black", alpha=0.8)
            ax2.axhline(y=1.0 / n, color="red", linestyle="--", linewidth=1.5,
                        label=f"Uniform (1/{n}={1/n:.3f})")
            ax2.set_ylabel("First-order Sobol index Sᵢ")
            ax2.set_title("Hyperparameter Sensitivity (HDMR)")
            ax2.legend()
            ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        return fig

    def _plot_3d(self, a_vec: NDArray, b_vec: NDArray) -> plt.Figure:
        xs, alpha, f0 = self.xs, self.alpha, self.f0
        y_hat = self._surrogate_full(xs, f0, alpha, a_vec, b_vec)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], y_hat[:, 0], alpha=0.3, s=6, label="HDMR")
        x1 = np.linspace(a_vec[0], b_vec[0], 60)
        x2 = np.linspace(a_vec[1], b_vec[1], 60)
        X1, X2 = np.meshgrid(x1, x2)
        grid = np.column_stack((X1.ravel(), X2.ravel()))
        Y = _safe_call(self.fun_batch, grid).reshape(X1.shape)
        ax.plot_surface(X1, X2, Y, alpha=0.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        ax.set_title("HDMR v4 vs True Function (2D)")
        fig.tight_layout()
        return fig


# =============================================================================
# Public API (backward compatible)
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
    x0_=None,
    is_adaptive_: bool = False,
    k_: int = 100,
    epsilon_: float = 0.1,
    clip_: float = 0.9,
    number_of_runs_: int = 1,
    maxiter_: int = 25,
    disp_: bool = False,
    enable_plots_: bool = True,
    seed_: Optional[int] = None,
    # v4 new parameters (all optional with sensible defaults)
    use_quasi_random_: bool = True,
    use_focused_resample_: bool = True,
    n_grid_1d_: int = 500,
    n_multistarts_1d_: int = 5,
    sensitivity_exponent_: float = 0.5,
    min_shrink_: float = 0.15,
    top_k_sensitive_: int = 3,
):
    if not hasattr(functions, function_name_):
        raise ValueError(f"Unknown function '{function_name_}' in src.functions")

    fun = getattr(functions, function_name_)

    def fun_batch(X: NDArray) -> NDArray:
        X = _ensure_2d(X, n_)
        return _safe_call(fun, X)

    cfg = HDMRConfig(
        n=n_, a=a_, b=b_, N=N_, m=m_, basis=basis_function_, seed=seed_,
        adaptive=is_adaptive_, maxiter=maxiter_, k=k_,
        epsilon=epsilon_, clip=clip_, disp=disp_, enable_plots=enable_plots_,
        use_quasi_random=use_quasi_random_,
        use_focused_resample=use_focused_resample_,
        n_grid_1d=n_grid_1d_,
        n_multistarts_1d=n_multistarts_1d_,
        sensitivity_exponent=sensitivity_exponent_,
        min_shrink=min_shrink_,
        top_k_sensitive=top_k_sensitive_,
    )

    file_name = (
        f"results/adaptive_{function_name_}_a{a_}_b{b_}_N{N_}_m{m_}_k{k_}_c{clip_:.2f}"
        if is_adaptive_
        else f"results/{function_name_}_a{a_}_b{b_}_N{N_}_m{m_}"
    )

    results = []
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
            print(f"Run {run_idx+1}/{number_of_runs_}: f(x*)={res.fun:.6f}")
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

def _load_bounds_from_json(function_name: str, path: str = "src/function_ranges.json"):
    with open(path, "r") as f:
        ranges = json.load(f)
    if function_name not in ranges:
        raise KeyError(function_name)
    return float(ranges[function_name][0]), float(ranges[function_name][1])


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="HDMR v4",
        description="HDMR v4: Sensitivity-Guided Adaptive Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v4 Enhancements:
  --use_quasi_random      Sobol low-discrepancy sampling (default: on)
  --use_focused_resample  Sensitivity-focused resampling after iter 1 (default: on)
  --n_grid_1d             Grid points for 1D surrogate scan (default: 500)
  --n_multistarts_1d      Multi-start BFGS per dimension (default: 5)
  --sensitivity_exponent  Shrinkage exponent sqrt=0.5, linear=1.0 (default: 0.5)
  --min_shrink            Min bound retention for insensitive dims (default: 0.15)
  --top_k_sensitive       Dims to focus resampling on (default: 3)

Example (compare v3 vs v4):
  python main.py --numSamples 200 --numVariables 10 --function rastrigin_10d \\
    --adaptive --maxiter 3 --seed 42
        """,
    )
    # Core arguments (identical to v3)
    parser.add_argument("--numSamples", type=int, required=True)
    parser.add_argument("--numVariables", type=int, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--min", type=float, default=None)
    parser.add_argument("--max", type=float, default=None)
    parser.add_argument("--x0", nargs="+", type=float, default=None)
    parser.add_argument("--randomInit", action="store_true")
    parser.add_argument("--basisFunction", type=str, default="Cosine",
                        choices=["Legendre", "Cosine"])
    parser.add_argument("--degree", type=int, default=7)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--numClosestPoints", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=0.9)
    parser.add_argument("--numberOfRuns", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noPlots", action="store_true")
    parser.add_argument("--disp", action="store_true")

    # v4 new arguments
    parser.add_argument("--no_quasi_random", action="store_true",
                        help="Disable Sobol sampling (use uniform)")
    parser.add_argument("--no_focused_resample", action="store_true",
                        help="Disable sensitivity-focused resampling")
    parser.add_argument("--n_grid_1d", type=int, default=500,
                        help="Grid points for 1D surrogate scan (default: 500)")
    parser.add_argument("--n_multistarts_1d", type=int, default=5,
                        help="Multi-starts per dim (default: 5)")
    parser.add_argument("--sensitivity_exponent", type=float, default=0.5,
                        help="Shrinkage exponent: 0.5=sqrt, 1.0=linear (default: 0.5)")
    parser.add_argument("--min_shrink", type=float, default=0.15,
                        help="Min bound retention for insensitive dims (default: 0.15)")
    parser.add_argument("--top_k_sensitive", type=int, default=3,
                        help="Dims to focus resampling on (default: 3)")

    args = parser.parse_args()

    if args.min is None or args.max is None:
        try:
            a_, b_ = _load_bounds_from_json(args.function)
        except Exception:
            print(f"Error: Could not load bounds for {args.function}. "
                  f"Provide --min and --max.")
            sys.exit(1)
    else:
        a_, b_ = float(args.min), float(args.max)

    print("\n" + "=" * 70)
    print("HDMR v4 — SENSITIVITY-GUIDED ADAPTIVE OPTIMIZER")
    print("=" * 70)
    print(f"Function:              {args.function}")
    print(f"Dimensions:            {args.numVariables}")
    print(f"Samples:               {args.numSamples}")
    print(f"Basis:                 {args.basisFunction} (degree {args.degree})")
    print(f"Bounds:                [{a_}, {b_}]")
    print(f"Adaptive:              {args.adaptive}")
    if args.adaptive:
        print(f"  maxiter:             {args.maxiter}")
        print(f"  k (elite):           {args.numClosestPoints}")
        print(f"  epsilon:             {args.epsilon}")
        print(f"  Quasi-random:        {not args.no_quasi_random}")
        print(f"  Focused resample:    {not args.no_focused_resample}")
        print(f"  Grid 1D:             {args.n_grid_1d} points")
        print(f"  Multi-starts 1D:     {args.n_multistarts_1d}")
        print(f"  Sensitivity exponent:{args.sensitivity_exponent}")
        print(f"  Min shrink:          {args.min_shrink}")
        print(f"  Top-k sensitive:     {args.top_k_sensitive}")
    print(f"Runs:                  {args.numberOfRuns}")
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
        use_quasi_random_=not args.no_quasi_random,
        use_focused_resample_=not args.no_focused_resample,
        n_grid_1d_=args.n_grid_1d,
        n_multistarts_1d_=args.n_multistarts_1d,
        sensitivity_exponent_=args.sensitivity_exponent,
        min_shrink_=args.min_shrink,
        top_k_sensitive_=args.top_k_sensitive,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Runtime:    {runtime:.4f}s")

    if args.numberOfRuns == 1:
        r = results[0]
        print(f"Solution:   {np.round(r.x, 6)}")
        print(f"Objective:  {r.fun:.6f}")
        print(f"Evals:      {r.nfev}")
        print(f"Iterations: {getattr(r, 'nit', 1)}")
    else:
        vals = np.array([r.fun for r in results])
        print(f"Mean:  {vals.mean():.6f} ± {vals.std():.6f}")
        print(f"Best:  {vals.min():.6f}")
        print(f"Worst: {vals.max():.6f}")
    print("=" * 70)

    # Code Ocean standard: Save all outputs to /results
    output_base = "/results/" + file_name
    os.makedirs("/results", exist_ok=True)
    
    with open(output_base + ".txt", "w") as f:
        f.write("HDMR v4 OPTIMIZATION RESULTS\n" + "=" * 70 + "\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"Run {i}:\n{r}\n\n")
    print(f"\n[INFO] Results saved to → {output_base}.txt")

    if args.numberOfRuns == 1 and not args.noPlots and fig1 is not None:
        fig1.savefig(output_base + "_results.png", dpi=300, bbox_inches="tight")
        if fig3 is not None:
            fig3.savefig(output_base + "_alpha.png", dpi=300, bbox_inches="tight")
        if fig2 is not None:
            fig2.savefig(output_base + "_3d.png", dpi=300, bbox_inches="tight")

        plt.show()

    print("\n[SUCCESS] HDMR v4 optimization completed.\n")


if __name__ == "__main__":
    warnings.filterwarnings("once", category=RuntimeWarning)
    cli_main()
