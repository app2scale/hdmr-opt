#!/usr/bin/env python3
"""
Systematic HDMR Benchmark Runner
=================================

Comprehensive benchmark testing for HDMR optimization with:
- Multiple basis functions (Legendre, Cosine)
- Multiple orders/degrees (1, 2, 3, 5, 7)
- Standard vs Adaptive HDMR
- Multiple runs for statistical significance (10 runs per config)

Output:
    results/systematic_benchmark_YYYYMMDD_HHMMSS/
        raw_results.csv         - All individual runs
        summary_statistics.csv  - Aggregated statistics
        config.json             - Experimental configuration
        logs/                   - Detailed logs per run

Usage:
    python benchmark_systematic.py [--quick]      # Quick test (12 exp)
    python benchmark_systematic.py                # Standard 2D (1,000 exp)
    python benchmark_systematic.py --comprehensive # Full suite (3,200 exp)

Author: HDMR Research Group
Date: 2025-02-16
Version: 2.0
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import HDMR components
from src.main import HDMRConfig, HDMROptimizer, _ensure_2d, _safe_call
import src.functions as functions
from src.functions import get_function_info


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    # Functions to test
    functions: List[str]
    
    # HDMR parameters
    basis_functions: List[str]
    orders: List[int]
    modes: List[str]  # ['standard', 'adaptive']
    
    # Sampling
    num_samples: int
    num_runs: int
    seed_start: int
    
    # Adaptive-specific
    adaptive_maxiter: int
    adaptive_k: int
    adaptive_epsilon: float
    adaptive_clip: float
    
    def total_experiments(self) -> int:
        """Calculate total number of experiments."""
        return (len(self.functions) * len(self.basis_functions) * 
                len(self.orders) * len(self.modes) * self.num_runs)


@dataclass
class ExperimentResult:
    """Single experiment result."""
    function: str
    basis: str
    order: int
    mode: str
    run: int
    seed: int
    
    # Results
    best_f: float
    best_x: List[float]
    n_iterations: int
    wall_time: float
    success: bool
    
    # Metadata
    n_variables: int
    bounds: Tuple[float, float]
    true_optimum: float


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / "benchmark.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_function_and_info(func_name: str) -> Tuple[callable, dict]:
    """Get function and its metadata."""
    if not hasattr(functions, func_name):
        raise ValueError(f"Unknown function: {func_name}")
    
    func = getattr(functions, func_name)
    info = get_function_info(func_name)
    
    return func, info


def run_single_experiment(
    func_name: str,
    basis: str,
    order: int,
    mode: str,
    run_idx: int,
    seed: int,
    config: BenchmarkConfig
) -> ExperimentResult:
    """
    Run a single HDMR optimization experiment.
    
    Parameters
    ----------
    func_name : str
        Benchmark function name
    basis : str
        Basis function type ('Legendre' or 'Cosine')
    order : int
        Polynomial/basis order
    mode : str
        'standard' or 'adaptive'
    run_idx : int
        Run index (for tracking)
    seed : int
        Random seed
    config : BenchmarkConfig
        Benchmark configuration
    
    Returns
    -------
    ExperimentResult
        Experiment results and metadata
    """
    # Get function and metadata
    func, info = get_function_and_info(func_name)
    
    n = info['dimension']
    bounds = info['domain']
    true_opt = info['global_minimum']
    
    # Create batch wrapper function
    def fun_batch(X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X, n)
        return _safe_call(func, X)
    
    # Setup HDMR configuration
    hdmr_cfg = HDMRConfig(
        n=n,
        a=bounds[0],
        b=bounds[1],
        N=config.num_samples,
        m=order,
        basis=basis,
        seed=seed,
        adaptive=(mode == 'adaptive'),
        maxiter=config.adaptive_maxiter if mode == 'adaptive' else 1,
        k=config.adaptive_k,
        epsilon=config.adaptive_epsilon,
        clip=config.adaptive_clip,
        disp=False,
        enable_plots=False
    )
    
    # Run optimization
    start_time = time.time()
    try:
        optimizer = HDMROptimizer(fun_batch=fun_batch, config=hdmr_cfg)
        
        # Random initialization
        a_vec, b_vec = hdmr_cfg.bounds_as_vectors()
        x0 = (b_vec - a_vec) * np.random.random(n) + a_vec
        
        result = optimizer.solve(x0)
        wall_time = time.time() - start_time
        
        return ExperimentResult(
            function=func_name,
            basis=basis,
            order=order,
            mode=mode,
            run=run_idx,
            seed=seed,
            best_f=float(result.fun),
            best_x=result.x.tolist() if hasattr(result.x, 'tolist') else list(result.x),
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            wall_time=wall_time,
            success=result.success if hasattr(result, 'success') else True,
            n_variables=n,
            bounds=bounds,
            true_optimum=true_opt
        )
    
    except Exception as e:
        wall_time = time.time() - start_time
        logging.error(f"Experiment failed: {func_name}, {basis}, order={order}, "
                     f"mode={mode}, seed={seed}. Error: {str(e)}")
        
        return ExperimentResult(
            function=func_name,
            basis=basis,
            order=order,
            mode=mode,
            run=run_idx,
            seed=seed,
            best_f=float('inf'),
            best_x=[float('nan')] * n,
            n_iterations=0,
            wall_time=wall_time,
            success=False,
            n_variables=n,
            bounds=bounds,
            true_optimum=true_opt
        )


def run_benchmark(config: BenchmarkConfig, output_dir: Path) -> pd.DataFrame:
    """
    Run complete benchmark suite.
    
    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration
    output_dir : Path
        Output directory for results
    
    Returns
    -------
    pd.DataFrame
        All experiment results
    """
    logger = logging.getLogger(__name__)
    
    total_exp = config.total_experiments()
    logger.info(f"Starting benchmark suite: {total_exp} experiments")
    logger.info(f"Configuration: {asdict(config)}")
    
    results = []
    
    # Progress bar
    pbar = tqdm(total=total_exp, desc="Running experiments")
    
    for func_name in config.functions:
        for basis in config.basis_functions:
            for order in config.orders:
                for mode in config.modes:
                    for run_idx in range(config.num_runs):
                        seed = config.seed_start + run_idx
                        
                        # Update progress bar
                        pbar.set_description(
                            f"{func_name} | {basis} | ord={order} | "
                            f"{mode} | run={run_idx+1}/{config.num_runs}"
                        )
                        
                        # Run experiment
                        result = run_single_experiment(
                            func_name, basis, order, mode, run_idx, seed, config
                        )
                        results.append(asdict(result))
                        
                        pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    logger.info(f"Benchmark completed: {len(df)} experiments")
    logger.info(f"Successful runs: {df['success'].sum()} / {len(df)}")
    
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each configuration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment results
    
    Returns
    -------
    pd.DataFrame
        Summary statistics (mean, std, min, max, success rate)
    """
    # Group by configuration (excluding run and seed)
    group_cols = ['function', 'basis', 'order', 'mode']
    
    stats = df.groupby(group_cols).agg({
        'best_f': ['mean', 'std', 'min', 'max', 'median'],
        'wall_time': ['mean', 'std'],
        'success': 'mean',  # Success rate
        'n_iterations': 'mean'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                     for col in stats.columns.values]
    
    # Rename for clarity
    stats = stats.rename(columns={
        'success_mean': 'success_rate',
        'n_iterations_mean': 'avg_iterations'
    })
    
    # Add optimality gap (best_f - true_optimum)
    def get_true_opt(row):
        _, info = get_function_and_info(row['function'])
        return info['global_minimum']
    
    stats['true_optimum'] = stats.apply(get_true_opt, axis=1)
    stats['optimality_gap_mean'] = stats['best_f_mean'] - stats['true_optimum']
    stats['optimality_gap_median'] = stats['best_f_median'] - stats['true_optimum']
    
    # Sort by function and performance
    stats = stats.sort_values(['function', 'best_f_mean'])
    
    return stats


def save_results(df: pd.DataFrame, stats: pd.DataFrame, 
                config: BenchmarkConfig, output_dir: Path):
    """Save all results and configuration."""
    logger = logging.getLogger(__name__)
    
    # Save raw results
    raw_csv = output_dir / "raw_results.csv"
    df.to_csv(raw_csv, index=False)
    logger.info(f"Saved raw results: {raw_csv}")
    
    # Save summary statistics
    stats_csv = output_dir / "summary_statistics.csv"
    stats.to_csv(stats_csv, index=False)
    logger.info(f"Saved summary statistics: {stats_csv}")
    
    # Save configuration
    config_json = output_dir / "config.json"
    with open(config_json, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved configuration: {config_json}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    logger.info(f"Total experiments: {len(df)}")
    logger.info(f"Successful: {df['success'].sum()} ({100*df['success'].mean():.1f}%)")
    logger.info(f"Total runtime: {df['wall_time'].sum():.1f}s")
    logger.info(f"\nBest configuration per function:")
    
    for func in config.functions:
        best = stats[stats['function'] == func].nsmallest(1, 'best_f_mean')
        if not best.empty:
            row = best.iloc[0]
            logger.info(f"  {func}:")
            logger.info(f"    Config: {row['basis']}, order={row['order']}, {row['mode']}")
            logger.info(f"    Mean f(x*): {row['best_f_mean']:.6f} (±{row['best_f_std']:.6f})")
            logger.info(f"    Optimality gap: {row['optimality_gap_mean']:.6e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Systematic HDMR Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_systematic.py --quick              # Quick test (12 exp, ~1 min)
  python benchmark_systematic.py                      # Standard 2D (1,000 exp, ~30-60 min)
  python benchmark_systematic.py --comprehensive      # Full suite (3,200 exp, ~6-8 hours)
        """
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (12 experiments, ~1 minute)'
    )
    parser.add_argument(
        '--comprehensive', action='store_true',
        help='Comprehensive mode with modern functions (3,200 experiments, ~6-8 hours)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/systematic_benchmark_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Configure benchmark
    if args.quick:
        logger.info("Running in QUICK mode (reduced configurations)")
        logger.info("Purpose: System verification and testing")
        config = BenchmarkConfig(
            functions=['rastrigin_2d', 'rosenbrock_2d'],
            basis_functions=['Cosine'],
            orders=[3, 7],
            modes=['standard'],
            num_samples=500,
            num_runs=3,
            seed_start=42,
            adaptive_maxiter=15,
            adaptive_k=50,
            adaptive_epsilon=0.1,
            adaptive_clip=0.9
        )
    
    elif args.comprehensive:
        logger.info("Running COMPREHENSIVE benchmark suite")
        logger.info("Purpose: Full evaluation with all function categories")
        logger.info("⚠️  This will take 6-8 hours to complete!")
        config = BenchmarkConfig(
            functions=[
                # 2D functions (5)
                'rastrigin_2d', 'rosenbrock_2d', 'ackley_2d', 
                'branin_2d', 'camel16_2d',
                
                # Classical 10D (3)
                'rosenbrock_10d', 'rastrigin_10d', 'griewank_10d',
                
                # Modern scalable 10D (8)
                'schwefel', 'levy', 'zakharov', 'sphere',
                'styblinski_tang', 'dixon_price', 'michalewicz',
                'sum_of_different_powers'
            ],
            basis_functions=['Legendre', 'Cosine'],
            orders=[1, 2, 3, 5, 7],
            modes=['standard', 'adaptive'],
            num_samples=1000,
            num_runs=10,
            seed_start=42,
            adaptive_maxiter=25,
            adaptive_k=100,
            adaptive_epsilon=0.1,
            adaptive_clip=0.9
        )
    
    else:
        logger.info("Running STANDARD 2D benchmark suite")
        logger.info("Purpose: 2D functions with comprehensive parameter sweep")
        config = BenchmarkConfig(
            functions=[
                # 2D functions only
                'rastrigin_2d', 'rosenbrock_2d', 'ackley_2d', 
                'branin_2d', 'camel16_2d'
            ],
            basis_functions=['Legendre', 'Cosine'],
            orders=[1, 2, 3, 5, 7],
            modes=['standard', 'adaptive'],
            num_samples=1000,
            num_runs=10,
            seed_start=42,
            adaptive_maxiter=25,
            adaptive_k=100,
            adaptive_epsilon=0.1,
            adaptive_clip=0.9
        )
    
    logger.info(f"Total experiments: {config.total_experiments()}")
    logger.info(f"Estimated time: {config.total_experiments() * 0.15:.0f}-{config.total_experiments() * 0.25:.0f} seconds")
    
    # Run benchmark
    start_time = time.time()
    df_results = run_benchmark(config, output_dir)
    total_time = time.time() - start_time
    
    logger.info(f"\nBenchmark completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Compute statistics
    logger.info("Computing summary statistics...")
    df_stats = compute_statistics(df_results)
    
    # Save results
    save_results(df_results, df_stats, config, output_dir)
    
    logger.info(f"\n✓ All results saved to: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Analyze results: python analyze_benchmark.py {output_dir}")
    logger.info(f"  2. View raw data: cat {output_dir}/raw_results.csv")
    logger.info(f"  3. View summary: cat {output_dir}/summary_statistics.csv")


if __name__ == "__main__":
    main()