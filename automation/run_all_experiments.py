#!/usr/bin/env python3
"""
Master Experiment Runner for HDMR Forecasting Research

This script orchestrates all experiments needed for academic publication:
1. Comprehensive forecasting benchmark (all models, multiple seeds)
2. Optimizer comparison (HDMR vs baselines)
3. Sensitivity analysis (hyperparameter importance)
4. Ablation studies (HDMR configuration variations)

Usage:
------
# Run full experimental suite
python run_all_experiments.py --full

# Run specific experiment types
python run_all_experiments.py --benchmark --comparison --sensitivity

# Quick test run (fewer seeds/samples)
python run_all_experiments.py --quick

Author: HDMR Research Team
Date: 2026-01-13
"""

# Path fix for reorganized structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: str, description: str):
    """Run shell command and report status."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully ({elapsed:.1f}s)")
    else:
        print(f"\n✗ {description} failed (exit code: {result.returncode})")
    
    return result.returncode == 0


def run_forecasting_benchmark(quick=False):
    """Run comprehensive forecasting benchmark."""
    seeds = 3 if quick else 10
    samples = 500 if quick else 1000
    
    # Standard HDMR
    cmd = (
        f"python benchmark_forecasting.py "
        f"--models all "
        f"--seeds {seeds} "
        f"--samples {samples} "
        f"--degree 7 "
        f"--basis Cosine"
    )
    success = run_command(cmd, "Forecasting Benchmark (Standard HDMR)")
    
    if not success:
        return False
    
    # Adaptive HDMR
    if not quick:
        cmd_adaptive = (
            f"python benchmark_forecasting.py "
            f"--models all "
            f"--seeds {seeds} "
            f"--samples {samples} "
            f"--degree 7 "
            f"--basis Cosine "
            f"--adaptive "
            f"--maxiter 25"
        )
        success = run_command(cmd_adaptive, "Forecasting Benchmark (Adaptive HDMR)")
    
    return success


def run_optimizer_comparison(quick=False):
    """Run HDMR vs baseline optimizers comparison."""
    seeds = 3 if quick else 5
    trials = 50 if quick else 100
    hdmr_samples = 500 if quick else 1000
    
    # Compare for XGBoost
    cmd = (
        f"python compare_optimizers.py "
        f"--model xgboost "
        f"--trials {trials} "
        f"--hdmr-samples {hdmr_samples} "
        f"--seeds {seeds} "
        f"--methods default random_search optuna hdmr_standard hdmr_adaptive"
    )
    success = run_command(cmd, "Optimizer Comparison (XGBoost)")
    
    if not success or quick:
        return success
    
    # Compare for LSTM (deep learning representative)
    cmd_lstm = (
        f"python compare_optimizers.py "
        f"--model lstm "
        f"--trials {trials} "
        f"--hdmr-samples {hdmr_samples} "
        f"--seeds {seeds} "
        f"--methods default random_search hdmr_standard"
    )
    success = run_command(cmd_lstm, "Optimizer Comparison (LSTM)")
    
    return success


def run_sensitivity_analysis(quick=False):
    """Run HDMR sensitivity analysis for hyperparameter importance."""
    samples = 1000 if quick else 2000
    degree = 7 if quick else 10
    
    models = ['xgboost'] if quick else ['xgboost', 'lightgbm', 'lstm']
    
    for model in models:
        cmd = (
            f"python sensitivity_analysis.py "
            f"--model {model} "
            f"--samples {samples} "
            f"--degree {degree} "
            f"--basis Cosine"
        )
        success = run_command(cmd, f"Sensitivity Analysis ({model.upper()})")
        
        if not success:
            return False
    
    return True


def run_ablation_study(quick=False):
    """Run ablation study on HDMR configuration."""
    if quick:
        print("\n⊙ Skipping ablation study in quick mode")
        return True
    
    seeds = 3
    
    # Vary sample size
    for N in [500, 1000, 2000]:
        cmd = (
            f"python benchmark_forecasting.py "
            f"--models xgboost lightgbm "
            f"--seeds {seeds} "
            f"--samples {N} "
            f"--degree 7 "
            f"--output-dir results/ablation/samples"
        )
        run_command(cmd, f"Ablation Study (N={N})")
    
    # Vary basis degree
    for m in [3, 5, 7, 10]:
        cmd = (
            f"python benchmark_forecasting.py "
            f"--models xgboost lightgbm "
            f"--seeds {seeds} "
            f"--samples 1000 "
            f"--degree {m} "
            f"--output-dir results/ablation/degree"
        )
        run_command(cmd, f"Ablation Study (m={m})")
    
    # Compare basis functions
    for basis in ['Legendre', 'Cosine']:
        cmd = (
            f"python benchmark_forecasting.py "
            f"--models xgboost lightgbm "
            f"--seeds {seeds} "
            f"--samples 1000 "
            f"--degree 7 "
            f"--basis {basis} "
            f"--output-dir results/ablation/basis"
        )
        run_command(cmd, f"Ablation Study (basis={basis})")
    
    return True


def create_summary_report():
    """Create summary report of all experiments."""
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("HDMR FORECASTING OPTIMIZATION - EXPERIMENTAL RESULTS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Check which result files exist
    results_dir = Path("results")
    
    if results_dir.exists():
        report_lines.append("AVAILABLE RESULT FILES:")
        report_lines.append("-" * 80)
        
        for subdir in ['benchmarks', 'comparisons', 'sensitivity', 'ablation']:
            subdir_path = results_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.csv")) + list(subdir_path.glob("*.txt"))
                if files:
                    report_lines.append(f"\n{subdir.upper()}/ ({len(files)} files)")
                    for f in sorted(files)[:5]:  # Show first 5
                        report_lines.append(f"  - {f.name}")
                    if len(files) > 5:
                        report_lines.append(f"  ... and {len(files) - 5} more")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("NEXT STEPS FOR PUBLICATION")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("1. Review result files in results/ directory")
    report_lines.append("2. Generate publication-quality figures from CSV data")
    report_lines.append("3. Perform statistical significance tests (Wilcoxon, Friedman)")
    report_lines.append("4. Write paper sections using results/")
    report_lines.append("   - Methodology: Describe HDMR approach")
    report_lines.append("   - Experiments: Reference benchmark results")
    report_lines.append("   - Results: Create tables/figures from CSVs")
    report_lines.append("   - Discussion: Interpret sensitivity analysis")
    report_lines.append("5. Create supplementary materials:")
    report_lines.append("   - Full hyperparameter ranges")
    report_lines.append("   - Complete experimental results")
    report_lines.append("   - Code repository link")
    report_lines.append("")
    
    # Save report
    report_path = results_dir / "EXPERIMENTS_SUMMARY.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\n✓ Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Master script for running all publication experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (long runtime: ~2-4 hours)
  python run_all_experiments.py --full
  
  # Run specific experiment types
  python run_all_experiments.py --benchmark --comparison
  
  # Quick test run (for debugging)
  python run_all_experiments.py --quick --benchmark
  
  # Custom experiment sequence
  python run_all_experiments.py --benchmark --sensitivity
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run all experiments (benchmark, comparison, sensitivity, ablation)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer seeds/samples for testing')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run forecasting benchmark')
    parser.add_argument('--comparison', action='store_true',
                       help='Run optimizer comparison')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run sensitivity analysis')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study')
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.full:
        run_benchmark = run_comparison = run_sensitivity = run_ablation = True
    else:
        run_benchmark = args.benchmark
        run_comparison = args.comparison
        run_sensitivity = args.sensitivity
        run_ablation = args.ablation
    
    # If no specific experiments selected, show help
    if not any([run_benchmark, run_comparison, run_sensitivity, run_ablation]):
        parser.print_help()
        sys.exit(0)
    
    print("=" * 80)
    print("HDMR FORECASTING RESEARCH - MASTER EXPERIMENT RUNNER")
    print("=" * 80)
    print("\nExperiment Plan:")
    print(f"  Benchmark: {'✓' if run_benchmark else '✗'}")
    print(f"  Comparison: {'✓' if run_comparison else '✗'}")
    print(f"  Sensitivity: {'✓' if run_sensitivity else '✗'}")
    print(f"  Ablation: {'✓' if run_ablation else '✗'}")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print("")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Track overall success
    all_success = True
    start_time = time.time()
    
    # Run experiments
    if run_benchmark:
        success = run_forecasting_benchmark(quick=args.quick)
        all_success = all_success and success
    
    if run_comparison:
        success = run_optimizer_comparison(quick=args.quick)
        all_success = all_success and success
    
    if run_sensitivity:
        success = run_sensitivity_analysis(quick=args.quick)
        all_success = all_success and success
    
    if run_ablation:
        success = run_ablation_study(quick=args.quick)
        all_success = all_success and success
    
    # Create summary report
    create_summary_report()
    
    # Final summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE COMPLETED")
    print("=" * 80)
    print(f"Total runtime: {hours}h {minutes}m")
    print(f"Status: {'✓ All experiments successful' if all_success else '✗ Some experiments failed'}")
    print("\nResults saved in: results/")
    print("  - benchmarks/    : Model performance comparisons")
    print("  - comparisons/   : HDMR vs baseline optimizers")
    print("  - sensitivity/   : Hyperparameter importance analysis")
    if run_ablation:
        print("  - ablation/      : HDMR configuration studies")
    print("\n✓ Ready for publication analysis!")


if __name__ == "__main__":
    main()