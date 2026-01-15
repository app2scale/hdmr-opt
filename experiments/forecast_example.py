# Path fix for reorganized structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
HDMR-based Hyperparameter Optimization for Time Series Forecasting

This script demonstrates how to use HDMR to optimize forecasting model hyperparameters
using the transactions.csv dataset. It integrates with the refactored main.py module
to leverage the HDMROptimizer class.

Example Usage:
-------------
python forecast_example.py --algorithm xgboost --metric mape --samples 1000

Author: APP2SCALE Team
Date: 2026-01-13
Version: 2.0.0
"""

import argparse
import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

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
    calculate_metrics
)

# Import HDMR optimizer components from main module
from src.main import HDMROptimizer, HDMRConfig, _ensure_2d, _safe_call


def optimize_forecasting_model(
    algorithm: str = 'xgboost',
    data_path: str = 'src/data/transactions.csv',
    split_date: str = '2020-01-01',
    metric: str = 'mape',
    num_samples: int = 1000,
    basis_function: str = 'Cosine',
    legendre_degree: int = 7,
    adaptive: bool = False,
    maxiter: int = 25,
    num_closest_points: int = 100,
    epsilon: float = 0.1,
    clip: float = 0.9,
    seed: Optional[int] = None,
    disp: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Optimize forecasting model hyperparameters using HDMR.
    
    Parameters
    ----------
    algorithm : str
        Forecasting algorithm ('xgboost', 'lightgbm', 'arima', 'ets')
    data_path : str
        Path to transactions.csv file
    split_date : str
        Date to split train/test (format: 'YYYY-MM-DD' or 'MM-DD-YYYY')
    metric : str
        Metric to optimize ('mape', 'rmse', 'mae', 'smape')
    num_samples : int
        Number of samples for HDMR
    basis_function : str
        'Legendre' or 'Cosine'
    legendre_degree : int
        Degree for basis functions
    adaptive : bool
        Use adaptive HDMR
    maxiter : int
        Maximum adaptive iterations
    num_closest_points : int
        k for adaptive refinement
    epsilon : float
        Convergence threshold for adaptive mode
    clip : float
        Minimum shrink ratio for adaptive bounds
    seed : Optional[int]
        Random seed for reproducibility
    disp : bool
        Print progress information
    
    Returns
    -------
    Dict[str, Any] or None
        Dictionary containing optimization results, or None if failed
    """
    
    print("=" * 80)
    print("HDMR-BASED FORECASTING HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Model selection
    model_map = {
        'xgboost': XGBoostForecaster,
        'lightgbm': LightGBMForecaster,
        'arima': ARIMAForecaster,
        'ets': ETSForecaster,
        'lstm': LSTMForecaster,
        'gru': GRUForecaster,
        'nbeats': NBeatsForecaster
    }
    
    if algorithm.lower() not in model_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(model_map.keys())}")
    
    model_class = model_map[algorithm.lower()]
    
    print(f"\n1. Configuration")
    print("-" * 80)
    print(f"   Algorithm:        {algorithm.upper()}")
    print(f"   Data Path:        {data_path}")
    print(f"   Split Date:       {split_date}")
    print(f"   Metric:           {metric.upper()}")
    print(f"   HDMR Samples:     {num_samples}")
    print(f"   Basis Function:   {basis_function} (degree {legendre_degree})")
    print(f"   Adaptive Mode:    {adaptive}")
    if adaptive:
        print(f"   Max Iterations:   {maxiter}")
        print(f"   k (closest):      {num_closest_points}")
        print(f"   Epsilon:          {epsilon}")
        print(f"   Clip:             {clip}")
    
    # Load and prepare data
    print(f"\n2. Loading Data")
    print("-" * 80)
    
    try:
        data_dict = prepare_train_test(data_path, split_date)
        print(f"   âœ“ Training samples:   {len(data_dict['y_train'])}")
        print(f"   âœ“ Test samples:       {len(data_dict['y_test'])}")
        print(f"   âœ“ Features:           {data_dict['X_train'].shape[1]}")
    except Exception as e:
        print(f"   âœ— Error loading data: {e}")
        return None
    
    # Get hyperparameter space
    print(f"\n3. Hyperparameter Space")
    print("-" * 80)
    
    hyperparam_space = model_class().get_hyperparameter_space()
    param_names = list(hyperparam_space.keys())
    num_params = len(param_names)
    
    print(f"   Number of hyperparameters: {num_params}")
    for param, (min_val, max_val) in hyperparam_space.items():
        print(f"   {param:20s} [{min_val:8.4f}, {max_val:8.4f}]")
    
    # Create optimization objective
    print(f"\n4. Creating Optimization Objective")
    print("-" * 80)
    
    objective_func = create_optimization_objective(
        model_class=model_class,
        data_dict=data_dict,
        metric=metric
    )
    
    print(f"   Objective: Minimize {metric.upper()}")
    print(f"   Function: f(x) where x âˆˆ â„^{num_params}")
    
    # Prepare bounds for HDMR
    bounds_list = [hyperparam_space[param] for param in param_names]
    a_vec = np.array([b[0] for b in bounds_list], dtype=np.float64)
    b_vec = np.array([b[1] for b in bounds_list], dtype=np.float64)
    
    # Wrap objective for HDMR (expects batch input)
    def objective_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Batch objective wrapper for HDMR.
        X: (N, num_params) array
        Returns: (N, 1) array of metric values
        """
        X = _ensure_2d(X, num_params)
        N = X.shape[0]
        results = np.zeros((N, 1), dtype=np.float64)
        
        for i in range(N):
            results[i, 0] = objective_func(X[i, :])
        
        return results
    
    # Configure HDMR
    hdmr_config = HDMRConfig(
        n=num_params,
        a=a_vec,
        b=b_vec,
        N=num_samples,
        m=legendre_degree,
        basis=basis_function,
        seed=seed,
        adaptive=adaptive,
        maxiter=maxiter,
        k=num_closest_points,
        epsilon=epsilon,
        clip=clip,
        disp=disp,
        enable_plots=True
    )
    
    # Run HDMR optimization
    print(f"\n5. Running HDMR Optimization")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        optimizer = HDMROptimizer(fun_batch=objective_batch, config=hdmr_config)
        
        # Initial point: middle of hyperparameter space
        x0 = 0.5 * (a_vec + b_vec)
        
        if disp:
            print(f"   Initial point: {x0}")
            print(f"   Starting optimization...")
        
        result = optimizer.solve(x0)
        
        optimization_time = time.time() - start_time
        
        print(f"\n6. Optimization Results")
        print("-" * 80)
        print(f"   Runtime:          {optimization_time:.2f} seconds")
        print(f"   Success:          {result.success}")
        print(f"   Iterations:       {result.nit}")
        print(f"   Function evals:   {result.nfev}")
        print(f"   Best {metric.upper()}:        {result.fun:.4f}")
        
        # Map optimized hyperparameters back to names
        optimal_params = {}
        for i, param_name in enumerate(param_names):
            optimal_params[param_name] = float(result.x[i])
        
        print(f"\n   Optimal Hyperparameters:")
        for param_name, value in optimal_params.items():
            print(f"      {param_name:20s} = {value:.6f}")
        
        # Handle integer parameters
        int_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 
                     'min_child_weight', 'p', 'd', 'q', 'seasonal_periods']
        for param in int_params:
            if param in optimal_params:
                optimal_params[param] = int(optimal_params[param])
        
        # Train final model with best parameters
        print(f"\n7. Final Model Evaluation")
        print("-" * 80)
        
        final_model = model_class(**optimal_params)
        final_model.fit(
            data_dict['X_train'],
            data_dict['y_train'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test']
        )
        
        y_pred = final_model.predict(data_dict['X_test'])
        all_metrics = calculate_metrics(
            data_dict['y_test'],
            y_pred,
            data_dict['y_train'].values
        )
        
        print("   Test Set Metrics:")
        for metric_name, value in all_metrics.items():
            print(f"      {metric_name.upper():8s} = {value:.4f}")
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        prefix = "adaptive_" if adaptive else ""
        results_file = results_dir / f"{prefix}forecast_{algorithm}_{metric}_hdmr_N{num_samples}_m{legendre_degree}.txt"
        
        with open(results_file, 'w') as f:
            f.write("HDMR FORECASTING OPTIMIZATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Algorithm:         {algorithm.upper()}\n")
            f.write(f"Metric:            {metric.upper()}\n")
            f.write(f"Samples:           {num_samples}\n")
            f.write(f"Basis Function:    {basis_function} (degree {legendre_degree})\n")
            f.write(f"Adaptive:          {adaptive}\n")
            if adaptive:
                f.write(f"Max Iterations:    {maxiter}\n")
                f.write(f"k (closest):       {num_closest_points}\n")
                f.write(f"Epsilon:           {epsilon}\n")
                f.write(f"Clip:              {clip}\n")
            f.write(f"\nOptimization Time: {optimization_time:.2f} seconds\n")
            f.write(f"Iterations:        {result.nit}\n")
            f.write(f"Function Evals:    {result.nfev}\n\n")
            f.write("Optimal Hyperparameters:\n")
            f.write("-" * 80 + "\n")
            for param_name, value in optimal_params.items():
                f.write(f"  {param_name:20s} = {value}\n")
            f.write(f"\nTest Set Performance:\n")
            f.write("-" * 80 + "\n")
            for metric_name, value in all_metrics.items():
                f.write(f"  {metric_name.upper():8s} = {value:.4f}\n")
        
        print(f"\n   âœ“ Results saved to: {results_file}")
        
        # Save plots if available
        if optimizer.fig_results is not None:
            plot_file = results_dir / f"{prefix}forecast_{algorithm}_{metric}_hdmr_results.png"
            optimizer.fig_results.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"   âœ“ Results plot saved to: {plot_file}")
        
        if optimizer.fig_alpha is not None:
            alpha_file = results_dir / f"{prefix}forecast_{algorithm}_{metric}_hdmr_alpha.png"
            optimizer.fig_alpha.savefig(alpha_file, dpi=300, bbox_inches='tight')
            print(f"   âœ“ Alpha plot saved to: {alpha_file}")
        
        # Create prediction plot
        print(f"\n8. Creating Prediction Visualization")
        print("-" * 80)
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot predictions
            test_dates = data_dict['X_test'].index if hasattr(data_dict['X_test'], 'index') else range(len(y_pred))
            ax.plot(test_dates, data_dict['y_test'].values, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(test_dates, y_pred, label='HDMR-Optimized Prediction', linewidth=2, alpha=0.7, linestyle='--')
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Transactions', fontsize=12)
            ax.set_title(f'{algorithm.upper()} Forecasting (HDMR-Optimized) - {metric.upper()}: {all_metrics[metric]:.4f}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            pred_file = results_dir / f"{prefix}forecast_{algorithm}_{metric}_predictions.png"
            fig.savefig(pred_file, dpi=300, bbox_inches='tight')
            print(f"   âœ“ Prediction plot saved to: {pred_file}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"   âš  Could not create prediction plot: {e}")
        
        return {
            'optimal_params': optimal_params,
            'metrics': all_metrics,
            'predictions': y_pred,
            'optimization_result': result,
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        print(f"\n   âœ— Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        print("\n" + "=" * 80)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='HDMR-based hyperparameter optimization for forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic XGBoost optimization
  python forecast_example.py --algorithm xgboost --samples 1000
  
  # LightGBM with RMSE metric
  python forecast_example.py --algorithm lightgbm --metric rmse --samples 2000
  
  # Adaptive HDMR with custom parameters
  python forecast_example.py --algorithm xgboost --adaptive --maxiter 50 --epsilon 0.05
  
  # Using custom data and split date
  python forecast_example.py --data data/my_data.csv --split 2021-01-01
        """
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='xgboost',
        choices=['xgboost', 'lightgbm', 'arima', 'ets', 'lstm', 'gru', 'nbeats'],
        help='Forecasting algorithm to optimize (default: xgboost)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='src/data/transactions.csv',
        help='Path to transactions CSV file (default: src/data/transactions.csv)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='2020-01-01',
        help='Train/test split date in YYYY-MM-DD format (default: 2020-01-01)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='mape',
        choices=['mape', 'rmse', 'mae', 'smape'],
        help='Metric to optimize (default: mape)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of HDMR samples (default: 1000)'
    )
    
    parser.add_argument(
        '--basis',
        type=str,
        default='Cosine',
        choices=['Legendre', 'Cosine'],
        help='HDMR basis function (default: Cosine)'
    )
    
    parser.add_argument(
        '--degree',
        type=int,
        default=7,
        help='Basis function degree/order (default: 7)'
    )
    
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Enable adaptive HDMR with iterative refinement'
    )
    
    parser.add_argument(
        '--maxiter',
        type=int,
        default=25,
        help='Maximum adaptive iterations (default: 25)'
    )
    
    parser.add_argument(
        '--numClosestPoints',
        type=int,
        default=100,
        help='Number of closest points for adaptive refinement (default: 100)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help='Convergence threshold for adaptive mode (default: 0.1)'
    )
    
    parser.add_argument(
        '--clip',
        type=float,
        default=0.9,
        help='Minimum shrink ratio for adaptive bounds (default: 0.9)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        print(f"Please ensure the file exists or specify a valid path with --data")
        sys.exit(1)
    
    # Run optimization
    result = optimize_forecasting_model(
        algorithm=args.algorithm,
        data_path=args.data,
        split_date=args.split,
        metric=args.metric,
        num_samples=args.samples,
        basis_function=args.basis,
        legendre_degree=args.degree,
        adaptive=args.adaptive,
        maxiter=args.maxiter,
        num_closest_points=args.numClosestPoints,
        epsilon=args.epsilon,
        clip=args.clip,
        seed=args.seed,
        disp=not args.quiet
    )
    
    if result is None:
        print("\nâœ— Optimization failed. Check error messages above.")
        sys.exit(1)
    
    print("\nâœ“ Optimization completed successfully!")
    print(f"âœ“ Best {args.metric.upper()}: {result['metrics'][args.metric]:.4f}")
    print(f"âœ“ Optimization time: {result['optimization_time']:.2f} seconds")
    print(f"âœ“ Results saved to results/ directory")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()