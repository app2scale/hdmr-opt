"""
HDMR Sensitivity Analysis for Hyperparameter Importance

This script analyzes HDMR's alpha coefficients to determine which hyperparameters
have the most influence on forecasting performance. This provides interpretable
insights beyond just finding optimal values.

Key research contribution: HDMR not only optimizes but also explains which
hyperparameters matter most.

Usage:
------
Single seed:
  python sensitivity_analysis.py --model xgboost --samples 2000 --seed 42

Multi-seed (recommended for stable importance estimates):
  python sensitivity_analysis.py --model xgboost --samples 2000 --seeds 1 2 3

Author: HDMR Research Team
Date: 2026-01-13
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from src.functions_forecast import (
    XGBoostForecaster,
    LightGBMForecaster,
    LSTMForecaster,
    GRUForecaster,
    NBeatsForecaster,
    prepare_train_test,
    create_optimization_objective,
)

from src.main import HDMROptimizer, HDMRConfig, _ensure_2d

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

AVAILABLE_MODELS = {
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "nbeats": NBeatsForecaster,
}


def run_hdmr_and_extract_coefficients(
    model_class: type, data_dict: Dict, hdmr_config: Dict
) -> Tuple[np.ndarray, List[str]]:
    """
    Run HDMR optimization and extract alpha coefficients.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (alpha coefficients shape (m, n), parameter names)
    """
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
        seed=int(hdmr_config["seed"]),
        adaptive=False,  # Use standard for sensitivity analysis
        disp=False,
        enable_plots=False,
    )

    # Run HDMR
    optimizer = HDMROptimizer(fun_batch=objective_batch, config=config)
    x0 = 0.5 * (a_vec + b_vec)
    _ = optimizer.solve(x0)

    # Extract alpha coefficients
    alpha = optimizer.alpha  # Shape: (m, n)

    return alpha, param_names


def analyze_sensitivity(
    alpha: np.ndarray,
    param_names: List[str],
    output_dir: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Analyze HDMR coefficients to determine hyperparameter importance.

    Parameters
    ----------
    alpha : np.ndarray
        HDMR coefficients, shape (m, n)
    param_names : List[str]
        Names of hyperparameters
    output_dir : str
        Directory to save outputs
    model_name : str
        Model name for labeling (used in output filenames)

    Returns
    -------
    pd.DataFrame
        Sensitivity metrics for each hyperparameter
    """
    m, n = alpha.shape

    sensitivity_metrics = []

    for i, param_name in enumerate(param_names):
        alpha_i = alpha[:, i]

        # Metric 1: Total variance contribution (sum of squared alphas)
        total_variance = float(np.sum(alpha_i**2))

        # Metric 2: Mean absolute coefficient
        mean_abs = float(np.mean(np.abs(alpha_i)))

        # Metric 3: Max absolute coefficient
        max_abs = float(np.max(np.abs(alpha_i)))

        # Metric 4: Dominant degree (degree with largest |alpha|)
        dominant_degree = int(np.argmax(np.abs(alpha_i)) + 1)

        # Metric 5: Effective dimensionality (number of significant degrees)
        threshold = 0.05 * total_variance
        significant_degrees = int(np.sum(alpha_i**2 > threshold))

        sensitivity_metrics.append(
            {
                "Parameter": param_name,
                "Total_Variance": total_variance,
                "Mean_Abs_Coeff": mean_abs,
                "Max_Abs_Coeff": max_abs,
                "Dominant_Degree": dominant_degree,
                "Significant_Degrees": significant_degrees,
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_metrics)

    # Normalize total variance to get relative importance (%)
    total_var_sum = float(sensitivity_df["Total_Variance"].sum())
    if total_var_sum <= 0:
        # Avoid division by zero; fallback to zeros
        sensitivity_df["Relative_Importance"] = 0.0
    else:
        sensitivity_df["Relative_Importance"] = (
            sensitivity_df["Total_Variance"] / total_var_sum * 100.0
        )

    # Sort by importance
    sensitivity_df = sensitivity_df.sort_values(
        "Relative_Importance", ascending=False
    ).reset_index(drop=True)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"sensitivity_{model_name}.csv")
    sensitivity_df.to_csv(output_file, index=False)
    print(f"✓ Sensitivity metrics saved: {output_file}")

    return sensitivity_df


def aggregate_sensitivity_results(
    per_seed: List[Tuple[int, pd.DataFrame]],
    output_dir: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Aggregate per-seed sensitivity results into mean/std ranking.

    Produces columns:
      - Total_Variance_Mean / Std
      - Mean_Abs_Coeff_Mean / Std
      - Max_Abs_Coeff_Mean / Std
      - Dominant_Degree_Mean / Std
      - Significant_Degrees_Mean / Std
      - Relative_Importance_Mean / Std
    """
    if len(per_seed) == 0:
        raise ValueError("No per-seed results to aggregate.")

    dfs = []
    for seed, df in per_seed:
        tmp = df.copy()
        tmp["Seed"] = int(seed)
        dfs.append(tmp)

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    metric_cols = [
        "Total_Variance",
        "Mean_Abs_Coeff",
        "Max_Abs_Coeff",
        "Dominant_Degree",
        "Significant_Degrees",
        "Relative_Importance",
    ]

    grouped = all_df.groupby("Parameter", as_index=False)

    agg = grouped[metric_cols].agg(["mean", "std"])
    # Flatten columns
    agg.columns = ["Parameter"] + [f"{c[0]}_{c[1].capitalize()}" for c in agg.columns[1:]]
    agg = agg.sort_values("Relative_Importance_Mean", ascending=False).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"sensitivity_{model_name}_agg.csv")
    agg.to_csv(out_csv, index=False)
    print(f"✓ Aggregated sensitivity saved: {out_csv}")

    return agg


def create_visualizations(
    alpha: np.ndarray,
    param_names: List[str],
    sensitivity_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
):
    """Create comprehensive sensitivity visualizations."""
    m, n = alpha.shape
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Alpha coefficients heatmap (log10 magnitude)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    alpha_mag = np.abs(alpha)
    alpha_log = np.log10(alpha_mag + 1e-10)

    im = ax1.imshow(alpha_log.T, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax1.set_xlabel("Basis Function Degree", fontsize=12)
    ax1.set_ylabel("Hyperparameter", fontsize=12)
    ax1.set_title(
        f"HDMR Coefficient Magnitudes - {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(range(m))
    ax1.set_xticklabels(range(1, m + 1))
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(param_names)

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("log₁₀(|α|)", fontsize=11)

    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, f"sensitivity_heatmap_{model_name}.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    print(f"✓ Heatmap saved: {fig1_path}")
    plt.close(fig1)

    # Figure 2: Relative importance bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(sensitivity_df))
    bars = ax2.barh(
        sensitivity_df["Parameter"],
        sensitivity_df["Relative_Importance"],
        color=colors,
    )

    ax2.set_xlabel("Relative Importance (%)", fontsize=12)
    ax2.set_ylabel("Hyperparameter", fontsize=12)
    ax2.set_title(
        f"Hyperparameter Importance - {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, sensitivity_df["Relative_Importance"]):
        ax2.text(
            float(val) + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{float(val):.1f}%",
            va="center",
            fontsize=9,
        )

    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, f"sensitivity_importance_{model_name}.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    print(f"✓ Importance chart saved: {fig2_path}")
    plt.close(fig2)

    # Figure 3: Coefficient evolution by degree (variance contribution α^2)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for i, param_name in enumerate(param_names):
        ax3.plot(
            range(1, m + 1),
            alpha[:, i] ** 2,
            marker="o",
            linewidth=2,
            label=param_name,
            alpha=0.7,
        )

    ax3.set_xlabel("Basis Function Degree", fontsize=12)
    ax3.set_ylabel("α² (Variance Contribution)", fontsize=12)
    ax3.set_title(
        f"HDMR Coefficient Evolution - {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9)

    fig3.tight_layout()
    fig3_path = os.path.join(output_dir, f"sensitivity_evolution_{model_name}.png")
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    print(f"✓ Evolution plot saved: {fig3_path}")
    plt.close(fig3)

    # Figure 4: Cumulative variance explained
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    for i, param_name in enumerate(param_names):
        var = alpha[:, i] ** 2
        denom = float(np.sum(var))
        if denom <= 0:
            cumulative_pct = np.zeros_like(var)
        else:
            cumulative_pct = np.cumsum(var) / denom * 100.0

        ax4.plot(
            range(1, m + 1),
            cumulative_pct,
            marker="s",
            linewidth=2,
            label=param_name,
            alpha=0.7,
        )

    ax4.axhline(
        y=90, color="r", linestyle="--", linewidth=1, alpha=0.5, label="90% threshold"
    )
    ax4.set_xlabel("Basis Function Degree", fontsize=12)
    ax4.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax4.set_title(
        f"Cumulative Variance by Degree - {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best", fontsize=9)
    ax4.set_ylim([0, 105])

    fig4.tight_layout()
    fig4_path = os.path.join(output_dir, f"sensitivity_cumulative_{model_name}.png")
    fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
    print(f"✓ Cumulative variance plot saved: {fig4_path}")
    plt.close(fig4)


def generate_interpretation_report(
    sensitivity_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
):
    """Generate text interpretation of sensitivity results (single run)."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"HDMR SENSITIVITY ANALYSIS REPORT: {model_name.upper()}")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("HYPERPARAMETER IMPORTANCE RANKING")
    report_lines.append("-" * 80)

    for idx, row in sensitivity_df.iterrows():
        report_lines.append(f"\n{idx + 1}. {row['Parameter']}")
        report_lines.append(f"   Relative Importance: {row['Relative_Importance']:.2f}%")
        report_lines.append(f"   Dominant Degree: {int(row['Dominant_Degree'])}")
        report_lines.append(f"   Significant Degrees: {int(row['Significant_Degrees'])}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("INTERPRETATION GUIDELINES")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("High Relative Importance (>20%):")
    report_lines.append("  → Critical hyperparameter requiring careful tuning")
    report_lines.append("")
    report_lines.append("Medium Relative Importance (10-20%):")
    report_lines.append("  → Important hyperparameter with moderate impact")
    report_lines.append("")
    report_lines.append("Low Relative Importance (<10%):")
    report_lines.append("  → Secondary hyperparameter, less sensitive")
    report_lines.append("")
    report_lines.append("Dominant Degree:")
    report_lines.append("  - Degree 1: Linear relationship")
    report_lines.append("  - Degree 2-3: Quadratic/cubic non-linearity")
    report_lines.append("  - Degree 4+: Complex non-linear effects")
    report_lines.append("")
    report_lines.append("Significant Degrees:")
    report_lines.append("  - Few (1-2): Simple functional form")
    report_lines.append("  - Many (4+): Complex interaction with objective")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("KEY INSIGHTS")
    report_lines.append("=" * 80)
    report_lines.append("")

    top_param = sensitivity_df.iloc[0]
    report_lines.append(f"Most Influential: {top_param['Parameter']}")
    report_lines.append(
        f"  Contributes {top_param['Relative_Importance']:.1f}% of total sensitivity"
    )
    report_lines.append("")

    low_importance = sensitivity_df[sensitivity_df["Relative_Importance"] < 5]
    if len(low_importance) > 0:
        report_lines.append("Low-Impact Hyperparameters:")
        for _, row in low_importance.iterrows():
            report_lines.append(f"  - {row['Parameter']} ({row['Relative_Importance']:.1f}%)")
        report_lines.append(
            "  These can be set to reasonable defaults with minimal performance impact."
        )

    report_lines.append("")
    report_lines.append("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"sensitivity_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"✓ Interpretation report saved: {report_path}")
    print("\n" + "\n".join(report_lines))


def generate_interpretation_report_agg(
    agg_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
):
    """Generate interpretation report for aggregated (multi-seed) results."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"HDMR SENSITIVITY ANALYSIS (AGGREGATED): {model_name.upper()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("HYPERPARAMETER IMPORTANCE (MEAN ± STD over seeds)")
    lines.append("-" * 80)

    for i, row in agg_df.iterrows():
        lines.append(f"\n{i + 1}. {row['Parameter']}")
        lines.append(
            f"   Relative Importance: {row['Relative_Importance_Mean']:.2f}% ± {row['Relative_Importance_Std']:.2f}%"
        )
        lines.append(
            f"   Total Variance: {row['Total_Variance_Mean']:.4g} ± {row['Total_Variance_Std']:.4g}"
        )
        lines.append(
            f"   Dominant Degree (mean±std): {row['Dominant_Degree_Mean']:.2f} ± {row['Dominant_Degree_Std']:.2f}"
        )
        lines.append(
            f"   Significant Degrees (mean±std): {row['Significant_Degrees_Mean']:.2f} ± {row['Significant_Degrees_Std']:.2f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"sensitivity_report_{model_name}_agg.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Aggregated interpretation report saved: {path}")


def _make_pseudo_df_for_agg_visuals(
    agg_df: pd.DataFrame,
    template_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    create_visualizations() expects a full sensitivity_df with all columns, but for aggregated
    runs we primarily care about Relative_Importance. We therefore:
      - take template_df (from seed1) aligned by Parameter
      - overwrite Relative_Importance with Relative_Importance_Mean
      - keep other columns from template (they are not used in plotting except labels/order)
    """
    template = template_df.set_index("Parameter")
    ordered_params = agg_df["Parameter"].tolist()
    pseudo = template.loc[ordered_params].reset_index()
    pseudo["Relative_Importance"] = agg_df["Relative_Importance_Mean"].values
    # Ensure it's sorted exactly as agg_df
    pseudo = pseudo.sort_values("Relative_Importance", ascending=False).reset_index(drop=True)
    return pseudo


def main():
    parser = argparse.ArgumentParser(
        description="HDMR sensitivity analysis for hyperparameter importance"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to analyze",
    )
    parser.add_argument("--data", type=str, default="src/data/transactions.csv")
    parser.add_argument("--split", type=str, default="2020-01-01")
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of HDMR samples (more = better sensitivity estimates)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=10,
        help="Basis function degree (higher = capture more complexity)",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="Cosine",
        choices=["Legendre", "Cosine"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Single RNG seed (used if --seeds is not provided)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple seeds, e.g. --seeds 1 2 3. Overrides --seed.",
    )
    parser.add_argument("--output-dir", type=str, default="results/sensitivity")

    args = parser.parse_args()

    seed_list = args.seeds if args.seeds is not None else [args.seed]

    print("=" * 80)
    print("HDMR SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nModel: {args.model.upper()}")
    print(f"HDMR Samples: {args.samples}")
    print(f"Basis Degree: {args.degree}")
    print(f"Basis Function: {args.basis}")
    print(f"Seeds: {seed_list}")

    # Load data
    print("\nLoading data...")
    data_dict = prepare_train_test(args.data, args.split)
    print(f"✓ Train: {len(data_dict['y_train'])} samples")
    print(f"✓ Test: {len(data_dict['y_test'])} samples")

    # Base HDMR configuration (seed overridden per run)
    hdmr_config = {
        "N": int(args.samples),
        "m": int(args.degree),
        "basis": str(args.basis),
        "seed": int(seed_list[0]),
    }

    model_class = AVAILABLE_MODELS[args.model]

    per_seed_results: List[Tuple[int, pd.DataFrame]] = []
    last_alpha: Optional[np.ndarray] = None
    last_param_names: Optional[List[str]] = None
    seed1_df: Optional[pd.DataFrame] = None

    # Run per seed
    for s in seed_list:
        s = int(s)
        hdmr_config["seed"] = s

        print(f"\nRunning HDMR optimization (seed={s})...")
        alpha, param_names = run_hdmr_and_extract_coefficients(
            model_class, data_dict, hdmr_config
        )
        last_alpha = alpha
        last_param_names = param_names

        print(f"✓ Extracted HDMR coefficients: shape {alpha.shape}")
        print(f"  Parameters: {', '.join(param_names)}")

        print(f"\nAnalyzing sensitivity (seed={s})...")
        df = analyze_sensitivity(
            alpha=alpha,
            param_names=param_names,
            output_dir=args.output_dir,
            model_name=f"{args.model}_seed{s}",
        )
        per_seed_results.append((s, df))
        if seed1_df is None:
            seed1_df = df

        print(f"\nGenerating interpretation report (seed={s})...")
        generate_interpretation_report(
            sensitivity_df=df,
            model_name=f"{args.model}_seed{s}",
            output_dir=args.output_dir,
        )

    # Aggregation
    if len(seed_list) > 1:
        print("\nAggregating results across seeds...")
        agg_df = aggregate_sensitivity_results(
            per_seed=per_seed_results, output_dir=args.output_dir, model_name=args.model
        )

        print("\nGenerating aggregated interpretation report...")
        generate_interpretation_report_agg(
            agg_df=agg_df, model_name=args.model, output_dir=args.output_dir
        )

        # Aggregated visualizations (mean importance)
        if last_alpha is not None and last_param_names is not None and seed1_df is not None:
            print("\nGenerating aggregated visualizations (mean importance)...")
            pseudo_df = _make_pseudo_df_for_agg_visuals(agg_df=agg_df, template_df=seed1_df)
            create_visualizations(
                alpha=last_alpha,
                param_names=last_param_names,
                sensitivity_df=pseudo_df,
                output_dir=args.output_dir,
                model_name=f"{args.model}_agg",
            )
    else:
        # Single seed visuals (use the only run)
        if last_alpha is not None and last_param_names is not None and per_seed_results:
            print("\nGenerating visualizations...")
            only_df = per_seed_results[0][1]
            create_visualizations(
                alpha=last_alpha,
                param_names=last_param_names,
                sensitivity_df=only_df,
                output_dir=args.output_dir,
                model_name=args.model,
            )

    print(f"\n{'=' * 80}")
    print("SENSITIVITY ANALYSIS COMPLETED")
    print(f"{'=' * 80}")
    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
