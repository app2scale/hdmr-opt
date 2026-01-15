# Path fix for reorganized structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Comprehensive Forecasting Benchmark for HDMR Hyperparameter Optimization

This script runs systematic experiments to evaluate HDMR performance across:
- Multiple forecasting algorithms (XGBoost, LightGBM, ARIMA, ETS, LSTM, GRU, N-BEATS)
- Different HDMR configurations (basis functions, sample sizes, adaptive vs standard)
- Multiple random seeds for statistical robustness

Key improvements in this version (2026-01-14):
- Deterministic, unbuffered console output (Docker logs will show progress reliably)
- Robust exception handling + traceback capture per run
- Publication-friendly output format (flattened metric columns + JSON params)
- Incremental checkpoint saving (progress is not lost if the run is interrupted)

Usage:
------
# Run all models with default config
python benchmark_forecasting.py --models all --seeds 5

# Run specific models
python benchmark_forecasting.py --models xgboost lightgbm lstm --seeds 10

# Run with adaptive HDMR
python benchmark_forecasting.py --models all --adaptive --seeds 5

Author: HDMR Research Team
Date: 2026-01-13 (updated 2026-01-14)
"""

import argparse
import json
import os
import sys
import time
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Add repo root to path (script is at /workspace, src is /workspace/src)
sys.path.insert(0, str(Path(__file__).parent))

from src.functions_forecast import (
    XGBoostForecaster,
    LightGBMForecaster,
    ARIMAForecaster,
    ETSForecaster,
    LSTMForecaster,
    GRUForecaster,
    NBeatsForecaster,
    prepare_train_test,
    create_optimization_objective,
    calculate_metrics,
)

from src.main import HDMROptimizer, HDMRConfig, _ensure_2d

warnings.filterwarnings("ignore")


# ============================================================================
# UTIL
# ============================================================================

def log(msg: str) -> None:
    """Console log with flush (critical for Docker logs reliability)."""
    print(msg, flush=True)


def safe_json(obj: Any) -> str:
    """Serialize to JSON string safely."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ============================================================================
# MODEL REGISTRY
# ============================================================================

AVAILABLE_MODELS = {
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    "arima": ARIMAForecaster,
    "ets": ETSForecaster,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "nbeats": NBeatsForecaster,
}


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================

def run_single_experiment(
    model_name: str,
    model_class: type,
    data_dict: Dict,
    hdmr_config: Dict,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single HDMR optimization experiment.

    Returns dict containing:
      - model, seed
      - metrics_* columns (mape/rmse/mae/smape)
      - optimization_time_s, train_time_s, total_time_s
      - optimal_params_json
      - nfev, nit, success
      - error, traceback (if failed)
    """
    t0_total = time.time()

    def v(msg: str) -> None:
        if verbose:
            log(msg)

    v(f"  Seed {seed}: initializing objective...")

    # Create optimization objective
    objective_func = create_optimization_objective(
        model_class=model_class,
        data_dict=data_dict,
        metric="mape",
    )

    # Get hyperparameter space
    hyperparam_space = model_class().get_hyperparameter_space()
    param_names = list(hyperparam_space.keys())
    num_params = len(param_names)

    bounds_list = [hyperparam_space[param] for param in param_names]
    a_vec = np.array([b[0] for b in bounds_list], dtype=np.float64)
    b_vec = np.array([b[1] for b in bounds_list], dtype=np.float64)

    # Batch objective wrapper
    def objective_batch(X):
        X = _ensure_2d(X, num_params)
        N = X.shape[0]
        results = np.zeros((N, 1), dtype=np.float64)
        for i in range(N):
            results[i, 0] = objective_func(X[i, :])
        return results

    # Configure HDMR
    config = HDMRConfig(
        n=num_params,
        a=a_vec,
        b=b_vec,
        N=int(hdmr_config["N"]),
        m=int(hdmr_config["m"]),
        basis=str(hdmr_config["basis"]),
        seed=int(seed),
        adaptive=bool(hdmr_config["adaptive"]),
        maxiter=int(hdmr_config["maxiter"]),
        k=int(hdmr_config["k"]),
        epsilon=float(hdmr_config["epsilon"]),
        clip=float(hdmr_config["clip"]),
        disp=False,
        enable_plots=False,
    )

    # Run optimization
    v(f"  Seed {seed}: running HDMR (N={config.N}, m={config.m}, basis={config.basis}, adaptive={config.adaptive})...")
    t0_opt = time.time()
    optimizer = HDMROptimizer(fun_batch=objective_batch, config=config)
    x0 = 0.5 * (a_vec + b_vec)  # midpoint init
    result = optimizer.solve(x0)
    opt_time = time.time() - t0_opt
    v(f"  Seed {seed}: HDMR done in {opt_time:.1f}s (success={getattr(result, 'success', False)})")

    # Map x -> parameter dict
    optimal_params: Dict[str, Any] = {param_names[i]: float(result.x[i]) for i in range(len(param_names))}

    # Handle integer parameters (IMPORTANT: do NOT cast min_child_weight to int for XGBoost)
    # Keep this conservative.
    int_params = {
        "max_depth",
        "num_leaves",
        "min_data_in_leaf",
        "p",
        "d",
        "q",
        "seasonal_periods",
        "num_layers",
        "num_blocks",
    }
    for p in list(optimal_params.keys()):
        if p in int_params:
            optimal_params[p] = int(round(optimal_params[p]))

    # Train final model and evaluate
    metrics = {"mape": 999.0, "rmse": 999.0, "mae": 999.0, "smape": 999.0}
    train_time = 0.0
    err: Optional[str] = None
    tb: Optional[str] = None

    try:
        v(f"  Seed {seed}: training final model with optimized params...")
        t0_train = time.time()
        final_model = model_class(**optimal_params)
        final_model.fit(
            data_dict["X_train"],
            data_dict["y_train"],
            X_test=data_dict["X_test"],
            y_test=data_dict["y_test"],
        )
        y_pred = final_model.predict(data_dict["X_test"])
        train_time = time.time() - t0_train

        metrics = calculate_metrics(
            data_dict["y_test"],
            y_pred,
            data_dict["y_train"].values,
        )
        v(f"  Seed {seed}: MAPE={metrics['mape']:.2f}% | RMSE={metrics['rmse']:.4g} | train={train_time:.1f}s")

    except Exception as e:
        err = str(e)
        tb = traceback.format_exc()
        v(f"  Seed {seed}: WARNING training/eval failed: {err}")

    total_time = time.time() - t0_total

    return {
        "model": model_name,
        "seed": int(seed),
        "hdmr_N": int(hdmr_config["N"]),
        "hdmr_m": int(hdmr_config["m"]),
        "hdmr_basis": str(hdmr_config["basis"]),
        "hdmr_adaptive": bool(hdmr_config["adaptive"]),
        "optimization_time_s": float(opt_time),
        "train_time_s": float(train_time),
        "total_time_s": float(total_time),
        "metrics_mape": float(metrics.get("mape", 999.0)),
        "metrics_rmse": float(metrics.get("rmse", 999.0)),
        "metrics_mae": float(metrics.get("mae", 999.0)),
        "metrics_smape": float(metrics.get("smape", 999.0)),
        "optimal_params_json": safe_json(optimal_params),
        "nfev": int(getattr(result, "nfev", 0)),
        "nit": int(getattr(result, "nit", 0)),
        "success": bool(getattr(result, "success", False)) and err is None,
        "error": err,
        "traceback": tb,
    }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(
    models: List[str],
    data_path: str,
    split_date: str,
    hdmr_config: Dict,
    num_seeds: int = 5,
    base_seed: int = 42,
    output_dir: str = "results/benchmarks",
    verbose: bool = True,
    checkpoint_every: int = 1,
) -> pd.DataFrame:
    """
    Run benchmark across models and seeds with robust logging + checkpointing.

    Checkpointing:
      - Writes a running CSV checkpoint after each experiment (or every N experiments).
      - Final CSV/summary/config are written at the end.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = now_ts()
    config_str = f"N{hdmr_config['N']}_m{hdmr_config['m']}_{'adaptive' if hdmr_config['adaptive'] else 'standard'}"
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{config_str}_{timestamp}.csv")

    log("=" * 80)
    log("HDMR FORECASTING BENCHMARK")
    log("=" * 80)
    log("")
    log("Configuration:")
    log(f"  Data: {data_path}")
    log(f"  Split date: {split_date}")
    log(f"  Models: {', '.join(models)}")
    log(f"  Seeds: {num_seeds} (base_seed={base_seed})")
    log(f"  HDMR: N={hdmr_config['N']}, m={hdmr_config['m']}, basis={hdmr_config['basis']}, adaptive={hdmr_config['adaptive']}")
    log(f"  Output dir: {output_dir}")
    log(f"  Checkpoint: {checkpoint_file}")
    log("")

    # Load data once
    log("Loading data...")
    data_dict = prepare_train_test(data_path, split_date)
    log(f"  ✓ Train: {len(data_dict['y_train'])} samples")
    log(f"  ✓ Test:  {len(data_dict['y_test'])} samples")

    all_results: List[Dict[str, Any]] = []
    exp_count = 0

    for model_name in models:
        if model_name not in AVAILABLE_MODELS:
            log(f"\nSkipping unknown model: {model_name}")
            continue

        model_class = AVAILABLE_MODELS[model_name]

        log("\n" + "=" * 80)
        log(f"MODEL: {model_name.upper()}")
        log("=" * 80)

        for i in range(num_seeds):
            seed = base_seed + i
            exp_count += 1
            log(f"\nExperiment {exp_count} / {len(models) * num_seeds}: model={model_name}, seed={seed}")

            try:
                result = run_single_experiment(
                    model_name=model_name,
                    model_class=model_class,
                    data_dict=data_dict,
                    hdmr_config=hdmr_config,
                    seed=seed,
                    verbose=verbose,
                )
            except Exception as e:
                # Catch any unexpected top-level errors
                result = {
                    "model": model_name,
                    "seed": int(seed),
                    "hdmr_N": int(hdmr_config["N"]),
                    "hdmr_m": int(hdmr_config["m"]),
                    "hdmr_basis": str(hdmr_config["basis"]),
                    "hdmr_adaptive": bool(hdmr_config["adaptive"]),
                    "optimization_time_s": 0.0,
                    "train_time_s": 0.0,
                    "total_time_s": 0.0,
                    "metrics_mape": 999.0,
                    "metrics_rmse": 999.0,
                    "metrics_mae": 999.0,
                    "metrics_smape": 999.0,
                    "optimal_params_json": "{}",
                    "nfev": 0,
                    "nit": 0,
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                log(f"  Seed {seed}: FAILED (unexpected): {e}")

            all_results.append(result)

            # Incremental checkpoint
            if checkpoint_every > 0 and (len(all_results) % checkpoint_every == 0):
                pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                log(f"  ✓ Checkpoint updated: {checkpoint_file}")

    results_df = pd.DataFrame(all_results)

    # Summary statistics
    log("\n" + "=" * 80)
    log("SUMMARY STATISTICS")
    log("=" * 80)

    summary_rows = []
    for model_name in sorted(results_df["model"].unique()):
        model_df = results_df[results_df["model"] == model_name].copy()

        summary_rows.append(
            {
                "Model": model_name.upper(),
                "Mean_MAPE": float(model_df["metrics_mape"].mean()),
                "Std_MAPE": float(model_df["metrics_mape"].std(ddof=0)),
                "Best_MAPE": float(model_df["metrics_mape"].min()),
                "Worst_MAPE": float(model_df["metrics_mape"].max()),
                "Mean_Time_s": float(model_df["total_time_s"].mean()),
                "Success_Rate": float(model_df["success"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    log(summary_df.to_string(index=False))

    # Save final outputs
    results_file = os.path.join(output_dir, f"benchmark_{config_str}_{timestamp}.csv")
    summary_file = os.path.join(output_dir, f"summary_{config_str}_{timestamp}.csv")
    config_file = os.path.join(output_dir, f"config_{config_str}_{timestamp}.json")

    results_df.to_csv(results_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    with open(config_file, "w") as f:
        json.dump(
            {
                "data_path": data_path,
                "split_date": split_date,
                "models": models,
                "num_seeds": int(num_seeds),
                "base_seed": int(base_seed),
                "hdmr_config": hdmr_config,
                "timestamp": timestamp,
                "checkpoint_file": checkpoint_file,
            },
            f,
            indent=2,
        )

    log("")
    log(f"✓ Full results saved:   {results_file}")
    log(f"✓ Summary saved:        {summary_file}")
    log(f"✓ Configuration saved:  {config_file}")
    log(f"✓ Latest checkpoint:    {checkpoint_file}")

    log("\n" + "=" * 80)
    log("BENCHMARK COMPLETED")
    log("=" * 80 + "\n")

    return results_df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive forecasting benchmark for HDMR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_forecasting.py --models all --seeds 5
  python benchmark_forecasting.py --models xgboost lstm nbeats --seeds 10
  python benchmark_forecasting.py --models all --adaptive --seeds 5 --maxiter 50
  python benchmark_forecasting.py --models all --samples 2000 --degree 10
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to benchmark (all, xgboost, lightgbm, arima, ets, lstm, gru, nbeats)",
    )

    parser.add_argument("--data", type=str, default="src/data/transactions.csv", help="Path to data file")
    parser.add_argument("--split", type=str, default="2020-01-01", help="Train/test split date (YYYY-MM-DD)")

    # IMPORTANT: In this script, --seeds is the NUMBER of seeds (not a list).
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to test (default: 5)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed (default: 42)")

    parser.add_argument("--samples", type=int, default=1000, help="Number of HDMR samples (default: 1000)")
    parser.add_argument("--degree", type=int, default=7, help="Basis function degree (default: 7)")
    parser.add_argument("--basis", type=str, default="Cosine", choices=["Legendre", "Cosine"], help="Basis type")

    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive HDMR")
    parser.add_argument("--maxiter", type=int, default=25, help="Max adaptive iterations (default: 25)")
    parser.add_argument("--k", type=int, default=100, help="Number of closest points for adaptive (default: 100)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Convergence threshold (default: 0.1)")
    parser.add_argument("--clip", type=float, default=0.9, help="Minimum shrink ratio (default: 0.9)")

    parser.add_argument("--output-dir", type=str, default="results/benchmarks", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose per-seed output")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint every N experiments (default: 1). Use 0 to disable.",
    )

    args = parser.parse_args()

    # Process model list
    if "all" in [m.lower() for m in args.models]:
        models = list(AVAILABLE_MODELS.keys())
    else:
        models = [m.lower() for m in args.models]

    # HDMR configuration
    hdmr_config = {
        "N": int(args.samples),
        "m": int(args.degree),
        "basis": str(args.basis),
        "adaptive": bool(args.adaptive),
        "maxiter": int(args.maxiter),
        "k": int(args.k),
        "epsilon": float(args.epsilon),
        "clip": float(args.clip),
    }

    # Run benchmark
    _ = run_benchmark(
        models=models,
        data_path=args.data,
        split_date=args.split,
        hdmr_config=hdmr_config,
        num_seeds=int(args.seeds),
        base_seed=int(args.base_seed),
        output_dir=args.output_dir,
        verbose=not args.quiet,
        checkpoint_every=int(args.checkpoint_every),
    )

    log("✓ Benchmark completed successfully!")


if __name__ == "__main__":
    main()