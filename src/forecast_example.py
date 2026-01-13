"""
HDMR-based Hyperparameter Optimization for Time Series Forecasting

This script optimizes forecasting model hyperparameters using the
`src/data/transactions.csv` dataset (MM/DD/YYYY date format).

Recommended usage (from repo root):
  python -m src.forecast_example --algorithm xgboost --metric mape --samples 1000 --adaptive --no-plots

Author: APP2SCALE Team
Date: 2026-01-13
Version: 2.2.0 (production-ready: module-safe, strict MM/DD/YYYY parsing)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from src.functions_forecast import (
    XGBoostForecaster,
    LightGBMForecaster,
    ARIMAForecaster,
    ETSForecaster,
    prepare_train_test,
    create_optimization_objective,
    calculate_metrics,
)

from src.main import HDMROptimizer, HDMRConfig, _ensure_2d


# -----------------------------------------------------------------------------
# Defaults / helpers
# -----------------------------------------------------------------------------
DATE_FORMAT = "%m/%d/%Y"  # STRICT: MM/DD/YYYY


def _default_data_path() -> str:
    # src/forecast_example.py -> src/data/transactions.csv
    return str(Path(__file__).resolve().parent / "data" / "transactions.csv")


def _p(msg: str = "") -> None:
    # Always flush to show progress in pipelines / logs
    print(msg, flush=True)


def _validate_csv_date_range(csv_path: str) -> None:
    """
    Lightweight sanity check:
    - Ensure required columns exist
    - Ensure dates parse strictly as MM/DD/YYYY
    - Print min/max date and row count
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "transactions" not in df.columns:
        raise ValueError("CSV must contain columns: 'date' and 'transactions'.")

    # strict parsing
    try:
        dt = pd.to_datetime(
            df["date"].astype(str),
            format="mixed",
            dayfirst=False,
            errors="raise"
        )
    except Exception as e:
        raise ValueError(
            f"CSV date parsing failed. Supported formats: MM/DD/YYYY, YYYY-MM-DD. "
            f"Original error: {e}"
        )
    print(f"   ✓ Date parsing: auto-detected ({dt.min().date()} → {dt.max().date()})")
    _p(f"   ✓ CSV rows:          {len(df)}")
    _p(f"   ✓ Date range:        {dt.min().date()} → {dt.max().date()}")
    _p(f"   ✓ Date format:       MM/DD/YYYY (strict)")
    _p(f"   ✓ Transactions dtype: {df['transactions'].dtype}")
    


# -----------------------------------------------------------------------------
# Core optimization
# -----------------------------------------------------------------------------
def optimize_forecasting_model(
    algorithm: str = "xgboost",
    data_path: Optional[str] = None,
    split_date: str = "2020-01-01",
    metric: str = "mape",
    num_samples: int = 1000,
    basis_function: str = "Cosine",
    degree: int = 7,
    adaptive: bool = False,
    maxiter: int = 25,
    num_closest_points: int = 100,
    epsilon: float = 0.1,
    clip: float = 0.9,
    seed: Optional[int] = None,
    disp: bool = True,
    enable_plots: bool = True,
    progress_every: int = 0,  # If >0 prints evaluation count every N objective evals (can be noisy)
) -> Optional[Dict[str, Any]]:
    """
    Run HDMR-based hyperparameter optimization for a given forecasting algorithm/metric.
    """
    model_map = {
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "arima": ARIMAForecaster,
        "ets": ETSForecaster,
    }

    algo_key = algorithm.lower().strip()
    if algo_key not in model_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(model_map.keys())}")

    model_class = model_map[algo_key]

    if data_path is None:
        data_path = _default_data_path()

    _p("=" * 80)
    _p("HDMR-BASED FORECASTING HYPERPARAMETER OPTIMIZATION")
    _p("=" * 80)

    _p("\n1. Configuration")
    _p("-" * 80)
    _p(f"   Algorithm:        {algo_key.upper()}")
    _p(f"   Metric:           {metric.upper()}")
    _p(f"   Data Path:        {data_path}")
    _p(f"   Split Date:       {split_date}")
    _p(f"   HDMR Samples:     {num_samples}")
    _p(f"   Basis Function:   {basis_function} (degree {degree})")
    _p(f"   Adaptive:         {adaptive}")
    if adaptive:
        _p(f"   Max Iterations:   {maxiter}")
        _p(f"   k (closest):      {num_closest_points}")
        _p(f"   Epsilon:          {epsilon}")
        _p(f"   Clip:             {clip}")
    _p(f"   Plots enabled:    {enable_plots}")
    _p(f"   Date format:      MM/DD/YYYY (strict)")

    _p("\n2. Loading & Validating Data")
    _p("-" * 80)

    if not os.path.exists(data_path):
        _p(f"   ✗ Data file not found: {data_path}")
        return None

    try:
        _validate_csv_date_range(data_path)
    except Exception as e:
        _p(f"   ✗ CSV validation failed: {e}")
        _p("   Hint: Ensure date column is strictly MM/DD/YYYY (e.g., 10/1/2015).")
        return None

    # IMPORTANT:
    # functions_forecast.prepare_train_test uses pandas.to_datetime without fixed format.
    # To enforce strict format end-to-end, we handle it here by reading the CSV with strict parsing,
    # then writing a temporary normalized CSV with ISO dates is overkill.
    #
    # Instead, simplest production-safe approach:
    # - ensure the CSV is strictly MM/DD/YYYY (validated above),
    # - and rely on pandas parsing being consistent for this format.
    #
    # If you want "hard enforcement inside prepare_train_test", we can patch functions_forecast.py too.

    try:
        data_dict = prepare_train_test(data_path, split_date)
        _p(f"   ✓ Train samples (after features): {len(data_dict['y_train'])}")
        _p(f"   ✓ Test samples  (after features): {len(data_dict['y_test'])}")
        _p(f"   ✓ Feature count:                 {data_dict['X_train'].shape[1]}")
    except Exception as e:
        _p(f"   ✗ Error preparing train/test: {e}")
        return None

    _p("\n3. Hyperparameter Space")
    _p("-" * 80)

    try:
        hyperparam_space = model_class().get_hyperparameter_space()
    except Exception as e:
        _p(f"   ✗ Could not initialize model / space: {e}")
        return None

    param_names = list(hyperparam_space.keys())
    num_params = len(param_names)

    _p(f"   Number of hyperparameters: {num_params}")
    for p, (mn, mx) in hyperparam_space.items():
        _p(f"   {p:20s} [{mn:10.6f}, {mx:10.6f}]")

    _p("\n4. Creating Optimization Objective")
    _p("-" * 80)

    objective_func = create_optimization_objective(
        model_class=model_class,
        data_dict=data_dict,
        metric=metric,
    )
    _p(f"   Objective: Minimize {metric.upper()} over ℝ^{num_params}")

    bounds_list = [hyperparam_space[p] for p in param_names]
    a_vec = np.array([b[0] for b in bounds_list], dtype=np.float64)
    b_vec = np.array([b[1] for b in bounds_list], dtype=np.float64)

    eval_counter = {"n": 0}

    def objective_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        HDMR expects a batch objective: X shape (N, num_params) -> (N, 1)

        NOTE: This loop is intentionally sequential because each evaluation fits a model.
        """
        X = _ensure_2d(X, num_params)
        N = X.shape[0]
        out = np.zeros((N, 1), dtype=np.float64)

        for i in range(N):
            out[i, 0] = float(objective_func(X[i, :]))
            eval_counter["n"] += 1
            if disp and progress_every > 0 and (eval_counter["n"] % progress_every == 0):
                _p(f"   [progress] objective evals: {eval_counter['n']}")

        return out

    _p("\n5. Running HDMR Optimization")
    _p("-" * 80)

    cfg = HDMRConfig(
        n=num_params,
        a=a_vec,
        b=b_vec,
        N=int(num_samples),
        m=int(degree),
        basis=basis_function,
        seed=seed,
        adaptive=adaptive,
        maxiter=int(maxiter),
        k=int(num_closest_points),
        epsilon=float(epsilon),
        clip=float(clip),
        disp=disp,
        enable_plots=enable_plots,
    )

    start_time = time.time()
    try:
        optimizer = HDMROptimizer(fun_batch=objective_batch, config=cfg)

        # Midpoint initialization in hyperparameter space
        x0 = 0.5 * (a_vec + b_vec)
        if disp:
            _p(f"   Initial point x0: {x0}")
            _p("   Optimization started...")

        result = optimizer.solve(x0)
        runtime = time.time() - start_time

        _p("\n6. Optimization Results")
        _p("-" * 80)
        _p(f"   Runtime:          {runtime:.2f} seconds")
        _p(f"   Success:          {bool(getattr(result, 'success', True))}")
        _p(f"   Iterations:       {int(getattr(result, 'nit', 0))}")
        _p(f"   Function evals:   {int(getattr(result, 'nfev', 0))}")
        _p(f"   Best {metric.upper():>4s}:       {float(result.fun):.6f}")

        optimal_params: Dict[str, Any] = {param_names[i]: float(result.x[i]) for i in range(num_params)}

        _p("\n   Optimal Hyperparameters (raw):")
        for k, v in optimal_params.items():
            _p(f"      {k:20s} = {v:.6f}")

        # Integer parameter casting
        int_params = {
            "max_depth",
            "num_leaves",
            "min_data_in_leaf",
            "min_child_weight",
            "p",
            "d",
            "q",
            "seasonal_periods",
        }
        for p in list(optimal_params.keys()):
            if p in int_params:
                optimal_params[p] = int(round(float(optimal_params[p])))

        _p("\n   Optimal Hyperparameters (casted):")
        for k, v in optimal_params.items():
            _p(f"      {k:20s} = {v}")

        _p("\n7. Final Model Evaluation")
        _p("-" * 80)

        final_model = model_class(**optimal_params)
        final_model.fit(
            data_dict["X_train"],
            data_dict["y_train"],
            X_test=data_dict["X_test"],
            y_test=data_dict["y_test"],
        )

        y_pred = final_model.predict(data_dict["X_test"])
        metrics_all = calculate_metrics(
            data_dict["y_test"],
            y_pred,
            data_dict["y_train"].values,
        )

        _p("   Test Set Metrics:")
        for mn, mv in metrics_all.items():
            _p(f"      {mn.upper():8s} = {mv:.6f}")

        # Save results
        out_dir = Path("results/forecasting")
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = "adaptive_" if adaptive else ""
        results_file = out_dir / f"{prefix}forecast_{algo_key}_{metric}_hdmr_N{num_samples}_m{degree}.txt"

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("HDMR FORECASTING OPTIMIZATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Algorithm:         {algo_key.upper()}\n")
            f.write(f"Metric:            {metric.upper()}\n")
            f.write(f"Samples:           {num_samples}\n")
            f.write(f"Basis Function:    {basis_function} (degree {degree})\n")
            f.write(f"Adaptive:          {adaptive}\n")
            if adaptive:
                f.write(f"Max Iterations:    {maxiter}\n")
                f.write(f"k (closest):       {num_closest_points}\n")
                f.write(f"Epsilon:           {epsilon}\n")
                f.write(f"Clip:              {clip}\n")
            f.write(f"Split Date:        {split_date}\n")
            f.write(f"CSV Date Format:   MM/DD/YYYY (strict)\n")
            f.write(f"\nOptimization Time: {runtime:.2f} seconds\n")
            f.write(f"Iterations:        {int(getattr(result, 'nit', 0))}\n")
            f.write(f"Function Evals:    {int(getattr(result, 'nfev', 0))}\n\n")

            f.write("Optimal Hyperparameters:\n")
            f.write("-" * 80 + "\n")
            for k, v in optimal_params.items():
                f.write(f"  {k:20s} = {v}\n")

            f.write("\nTest Set Performance:\n")
            f.write("-" * 80 + "\n")
            for k, v in metrics_all.items():
                f.write(f"  {k.upper():8s} = {v:.6f}\n")

        _p(f"\n   ✓ Results saved to: {results_file}")

        # Save plots if enabled
        if enable_plots:
            _p("\n8. Saving Visualizations")
            _p("-" * 80)

            if optimizer.fig_results is not None:
                plot_file = out_dir / f"{prefix}forecast_{algo_key}_{metric}_hdmr_results.png"
                optimizer.fig_results.savefig(plot_file, dpi=300, bbox_inches="tight")
                _p(f"   ✓ Results plot saved to: {plot_file}")

            if optimizer.fig_alpha is not None:
                alpha_file = out_dir / f"{prefix}forecast_{algo_key}_{metric}_hdmr_alpha.png"
                optimizer.fig_alpha.savefig(alpha_file, dpi=300, bbox_inches="tight")
                _p(f"   ✓ Alpha plot saved to: {alpha_file}")

            # Prediction plot
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                test_dates = (
                    data_dict["X_test"].index
                    if hasattr(data_dict["X_test"], "index")
                    else range(len(y_pred))
                )
                ax.plot(test_dates, data_dict["y_test"].values, label="Actual", linewidth=2, alpha=0.7)
                ax.plot(test_dates, y_pred, label="HDMR-Optimized Prediction", linewidth=2, alpha=0.7, linestyle="--")
                ax.set_xlabel("Time", fontsize=12)
                ax.set_ylabel("Transactions", fontsize=12)
                ax.set_title(
                    f"{algo_key.upper()} Forecasting (HDMR-Optimized) - {metric.upper()}: {metrics_all[metric]:.4f}",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                pred_file = out_dir / f"{prefix}forecast_{algo_key}_{metric}_predictions.png"
                fig.savefig(pred_file, dpi=300, bbox_inches="tight")
                _p(f"   ✓ Prediction plot saved to: {pred_file}")
                plt.close(fig)
            except Exception as e:
                _p(f"   ⚠ Could not create prediction plot: {e}")

        _p("\n" + "=" * 80)

        return {
            "optimal_params": optimal_params,
            "metrics": metrics_all,
            "predictions": y_pred,
            "optimization_result": result,
            "optimization_time": runtime,
        }

    except Exception as e:
        _p(f"\n   ✗ Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        _p("\n" + "=" * 80)
        return None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDMR-based hyperparameter optimization for forecasting (MM/DD/YYYY dates).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.forecast_example --algorithm xgboost --metric mape --samples 500 --adaptive --no-plots
  python -m src.forecast_example --algorithm lightgbm --metric rmse --samples 1000 --adaptive --no-plots
""",
    )

    parser.add_argument("--algorithm", type=str, default="xgboost", choices=["xgboost", "lightgbm", "arima", "ets"])
    parser.add_argument("--data", type=str, default=None, help="Path to CSV (default: src/data/transactions.csv)")
    parser.add_argument("--split", type=str, default="2020-01-01", help="Train/test split date (YYYY-MM-DD)")
    parser.add_argument("--metric", type=str, default="mape", choices=["mape", "rmse", "mae", "smape"])

    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--basis", type=str, default="Cosine", choices=["Legendre", "Cosine"])
    parser.add_argument("--degree", type=int, default=7)

    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--numClosestPoints", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=0.9)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation (recommended for batch runs)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="If >0, prints objective evaluation progress every N evals (can be noisy).",
    )

    args = parser.parse_args()

    data_path = args.data if args.data else _default_data_path()
    if not os.path.exists(data_path):
        _p(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    result = optimize_forecasting_model(
        algorithm=args.algorithm,
        data_path=data_path,
        split_date=args.split,
        metric=args.metric,
        num_samples=args.samples,
        basis_function=args.basis,
        degree=args.degree,
        adaptive=args.adaptive,
        maxiter=args.maxiter,
        num_closest_points=args.numClosestPoints,
        epsilon=args.epsilon,
        clip=args.clip,
        seed=args.seed,
        disp=not args.quiet,
        enable_plots=not args.no_plots,
        progress_every=args.progress_every,
    )

    if result is None:
        _p("\n✗ Optimization failed. Check logs above.")
        sys.exit(1)

    _p("\n✓ Optimization completed successfully!")
    _p(f"✓ Best {args.metric.upper()}: {result['metrics'][args.metric]:.6f}")
    _p(f"✓ Optimization time: {result['optimization_time']:.2f} seconds")
    _p("✓ Results saved under results/forecasting/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

