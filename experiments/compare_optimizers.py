"""
Optimizer Comparison Study: HDMR vs Baselines

Compares HDMR hyperparameter optimization against:
1. Random Search
2. Grid Search (feasible subset)
3. Default hyperparameters
4. Optuna (Bayesian Optimization) - if available

This provides evidence for academic claims about HDMR's effectiveness.

Usage:
------
python compare_optimizers.py --model xgboost --trials 100 --seeds 5

Author: HDMR Research Team
Date: 2026-01-13
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.functions_forecast import (
    XGBoostForecaster,
    LightGBMForecaster,
    LSTMForecaster,
    GRUForecaster,
    NBeatsForecaster,
    prepare_train_test,
    calculate_metrics
)

from src.main import HDMROptimizer, HDMRConfig, _ensure_2d

warnings.filterwarnings('ignore')

# Try to import optuna for Bayesian optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")


# ============================================================================
# MODEL REGISTRY
# ============================================================================

AVAILABLE_MODELS = {
    'xgboost': XGBoostForecaster,
    'lightgbm': LightGBMForecaster,
    'lstm': LSTMForecaster,
    'gru': GRUForecaster,
    'nbeats': NBeatsForecaster
}


# ============================================================================
# DEFAULT HYPERPARAMETERS
# ============================================================================

DEFAULT_PARAMS = {
    'xgboost': {
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0
    },
    'lightgbm': {
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8
    },
    'lstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001
    },
    'gru': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001
    },
    'nbeats': {
        'stack_width': 256,
        'num_blocks': 3,
        'num_layers': 4,
        'learning_rate': 0.001
    }
}


# ============================================================================
# OPTIMIZATION METHODS
# ============================================================================

def optimize_with_defaults(
    model_class: type,
    model_name: str,
    data_dict: Dict,
    seed: int
) -> Dict[str, Any]:
    """Test with default hyperparameters."""
    start_time = time.time()
    
    params = DEFAULT_PARAMS.get(model_name, {})
    
    try:
        model = model_class(**params)
        model.fit(
            data_dict['X_train'],
            data_dict['y_train'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test']
        )
        
        y_pred = model.predict(data_dict['X_test'])
        metrics = calculate_metrics(
            data_dict['y_test'],
            y_pred,
            data_dict['y_train'].values
        )
        
        optimization_time = time.time() - start_time
        
        return {
            'method': 'default',
            'seed': seed,
            'metrics': metrics,
            'params': params,
            'time': optimization_time,
            'evaluations': 1
        }
    
    except Exception as e:
        return {
            'method': 'default',
            'seed': seed,
            'metrics': {'mape': 999.0, 'rmse': 999.0, 'mae': 999.0},
            'params': params,
            'time': time.time() - start_time,
            'evaluations': 1,
            'error': str(e)
        }


def optimize_with_random_search(
    model_class: type,
    model_name: str,
    data_dict: Dict,
    num_trials: int,
    seed: int
) -> Dict[str, Any]:
    """Random search over hyperparameter space."""
    np.random.seed(seed)
    
    hyperparam_space = model_class().get_hyperparameter_space()
    
    start_time = time.time()
    best_mape = float('inf')
    best_params = None
    
    for trial in range(num_trials):
        # Sample random hyperparameters
        params = {}
        for param_name, (min_val, max_val) in hyperparam_space.items():
            # Log-scale for learning rate
            if 'learning_rate' in param_name:
                params[param_name] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
            else:
                params[param_name] = np.random.uniform(min_val, max_val)
        
        # Handle integer parameters
        int_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 
                     'min_child_weight', 'num_layers', 'num_blocks']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        try:
            model = model_class(**params)
            model.fit(
                data_dict['X_train'],
                data_dict['y_train'],
                X_test=data_dict['X_test'],
                y_test=data_dict['y_test']
            )
            
            y_pred = model.predict(data_dict['X_test'])
            metrics = calculate_metrics(
                data_dict['y_test'],
                y_pred,
                data_dict['y_train'].values
            )
            
            if metrics['mape'] < best_mape:
                best_mape = metrics['mape']
                best_params = params
                best_metrics = metrics
        
        except Exception:
            continue
    
    optimization_time = time.time() - start_time
    
    if best_params is None:
        return {
            'method': 'random_search',
            'seed': seed,
            'metrics': {'mape': 999.0, 'rmse': 999.0, 'mae': 999.0},
            'params': {},
            'time': optimization_time,
            'evaluations': num_trials,
            'error': 'All trials failed'
        }
    
    return {
        'method': 'random_search',
        'seed': seed,
        'metrics': best_metrics,
        'params': best_params,
        'time': optimization_time,
        'evaluations': num_trials
    }


def optimize_with_grid_search(
    model_class: type,
    model_name: str,
    data_dict: Dict,
    seed: int
) -> Dict[str, Any]:
    """Coarse grid search (feasible for small spaces only)."""
    hyperparam_space = model_class().get_hyperparameter_space()
    
    # Define coarse grid (3 points per dimension)
    grid = {}
    for param_name, (min_val, max_val) in hyperparam_space.items():
        if 'learning_rate' in param_name:
            # Log-scale for learning rate
            grid[param_name] = [
                10 ** x for x in np.linspace(np.log10(min_val), np.log10(max_val), 3)
            ]
        else:
            grid[param_name] = list(np.linspace(min_val, max_val, 3))
    
    # Generate all combinations
    import itertools
    param_names = list(grid.keys())
    param_values = [grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    if len(combinations) > 100:
        # Too many combinations, sample randomly
        np.random.seed(seed)
        indices = np.random.choice(len(combinations), size=100, replace=False)
        combinations = [combinations[i] for i in indices]
    
    start_time = time.time()
    best_mape = float('inf')
    best_params = None
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        # Handle integer parameters
        int_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 
                     'min_child_weight', 'num_layers', 'num_blocks']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        try:
            model = model_class(**params)
            model.fit(
                data_dict['X_train'],
                data_dict['y_train'],
                X_test=data_dict['X_test'],
                y_test=data_dict['y_test']
            )
            
            y_pred = model.predict(data_dict['X_test'])
            metrics = calculate_metrics(
                data_dict['y_test'],
                y_pred,
                data_dict['y_train'].values
            )
            
            if metrics['mape'] < best_mape:
                best_mape = metrics['mape']
                best_params = params
                best_metrics = metrics
        
        except Exception:
            continue
    
    optimization_time = time.time() - start_time
    
    return {
        'method': 'grid_search',
        'seed': seed,
        'metrics': best_metrics if best_params else {'mape': 999.0},
        'params': best_params if best_params else {},
        'time': optimization_time,
        'evaluations': len(combinations)
    }


def optimize_with_optuna(
    model_class: type,
    model_name: str,
    data_dict: Dict,
    num_trials: int,
    seed: int
) -> Dict[str, Any]:
    """Bayesian optimization using Optuna."""
    if not OPTUNA_AVAILABLE:
        return {
            'method': 'optuna',
            'seed': seed,
            'metrics': {'mape': 999.0},
            'params': {},
            'time': 0.0,
            'evaluations': 0,
            'error': 'Optuna not available'
        }
    
    hyperparam_space = model_class().get_hyperparameter_space()
    
    def objective(trial):
        params = {}
        for param_name, (min_val, max_val) in hyperparam_space.items():
            if 'learning_rate' in param_name:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
            elif param_name in ['max_depth', 'num_leaves', 'min_data_in_leaf', 
                                'min_child_weight', 'num_layers', 'num_blocks']:
                params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
            else:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
        
        try:
            model = model_class(**params)
            model.fit(
                data_dict['X_train'],
                data_dict['y_train'],
                X_test=data_dict['X_test'],
                y_test=data_dict['y_test']
            )
            
            y_pred = model.predict(data_dict['X_test'])
            metrics = calculate_metrics(
                data_dict['y_test'],
                y_pred,
                data_dict['y_train'].values
            )
            
            return metrics['mape']
        
        except Exception:
            return 999.0
    
    start_time = time.time()
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=num_trials, show_progress_bar=False)
    
    optimization_time = time.time() - start_time
    
    return {
        'method': 'optuna',
        'seed': seed,
        'metrics': {'mape': study.best_value} if study.best_trial else {'mape': 999.0},
        'params': study.best_params if study.best_trial else {},
        'time': optimization_time,
        'evaluations': len(study.trials)
    }


def optimize_with_hdmr(
    model_class: type,
    model_name: str,
    data_dict: Dict,
    hdmr_samples: int,
    seed: int,
    adaptive: bool = False
) -> Dict[str, Any]:
    """HDMR hyperparameter optimization."""
    from src.functions_forecast import create_optimization_objective
    
    objective_func = create_optimization_objective(
        model_class=model_class,
        data_dict=data_dict,
        metric='mape'
    )
    
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
    
    config = HDMRConfig(
        n=num_params,
        a=a_vec,
        b=b_vec,
        N=hdmr_samples,
        m=7,
        basis='Cosine',
        seed=seed,
        adaptive=adaptive,
        maxiter=25 if adaptive else 1,
        disp=False,
        enable_plots=False
    )
    
    start_time = time.time()
    optimizer = HDMROptimizer(fun_batch=objective_batch, config=config)
    np.random.seed(seed)
    x0 = np.random.uniform(a_vec, b_vec)
    result = optimizer.solve(x0)
    optimization_time = time.time() - start_time
    
    optimal_params = {}
    for i, param_name in enumerate(param_names):
        optimal_params[param_name] = float(result.x[i])
    
    # Handle integer parameters
    int_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 
                 'min_child_weight', 'num_layers', 'num_blocks']
    for param in int_params:
        if param in optimal_params:
            optimal_params[param] = int(optimal_params[param])
    
    # Final evaluation
    try:
        model = model_class(**optimal_params)
        model.fit(
            data_dict['X_train'],
            data_dict['y_train'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test']
        )
        
        y_pred = model.predict(data_dict['X_test'])
        final_metrics = calculate_metrics(
            data_dict['y_test'],
            y_pred,
            data_dict['y_train'].values
        )
    except Exception as e:
        final_metrics = {'mape': 999.0, 'rmse': 999.0, 'mae': 999.0}
    
    method_name = 'hdmr_adaptive' if adaptive else 'hdmr_standard'
    
    return {
        'method': method_name,
        'seed': seed,
        'metrics': final_metrics,
        'params': optimal_params,
        'time': optimization_time,
        'evaluations': result.nfev
    }


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

def run_comparison(
    model_name: str,
    data_path: str,
    split_date: str,
    num_trials: int,
    num_seeds: int,
    hdmr_samples: int,
    methods: List[str],
    output_dir: str
) -> pd.DataFrame:
    """Run complete comparison study."""
    
    print("=" * 80)
    print("OPTIMIZER COMPARISON STUDY")
    print("=" * 80)
    print(f"\nModel: {model_name.upper()}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Seeds: {num_seeds}")
    print(f"Trials: {num_trials} (for random search, Optuna)")
    print(f"HDMR samples: {hdmr_samples}")
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    data_dict = prepare_train_test(data_path, split_date)
    print(f"✓ Train: {len(data_dict['y_train'])} samples")
    print(f"✓ Test: {len(data_dict['y_test'])} samples")
    
    model_class = AVAILABLE_MODELS[model_name]
    all_results = []
    
    # Run each method
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print(f"{'='*80}")
        
        for seed_idx in range(num_seeds):
            seed = 42 + seed_idx
            print(f"  Seed {seed}: ", end='', flush=True)
            
            try:
                if method == 'default':
                    result = optimize_with_defaults(model_class, model_name, data_dict, seed)
                elif method == 'random_search':
                    result = optimize_with_random_search(model_class, model_name, data_dict, num_trials, seed)
                elif method == 'grid_search':
                    result = optimize_with_grid_search(model_class, model_name, data_dict, seed)
                elif method == 'optuna':
                    result = optimize_with_optuna(model_class, model_name, data_dict, num_trials, seed)
                elif method == 'hdmr_standard':
                    result = optimize_with_hdmr(model_class, model_name, data_dict, hdmr_samples, seed, adaptive=False)
                elif method == 'hdmr_adaptive':
                    result = optimize_with_hdmr(model_class, model_name, data_dict, hdmr_samples, seed, adaptive=True)
                else:
                    print(f"Unknown method: {method}")
                    continue
                
                all_results.append(result)
                print(f"MAPE={result['metrics']['mape']:.2f}%, Time={result['time']:.1f}s")
            
            except Exception as e:
                print(f"Failed - {e}")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    summary_stats = []
    for method in results_df['method'].unique():
        method_results = results_df[results_df['method'] == method]
        mapes = [r['mape'] for r in method_results['metrics']]
        times = method_results['time'].values
        evals = method_results['evaluations'].values
        
        summary_stats.append({
            'Method': method,
            'Mean MAPE': np.mean(mapes),
            'Std MAPE': np.std(mapes),
            'Best MAPE': np.min(mapes),
            'Median Time (s)': np.median(times),
            'Mean Evaluations': np.mean(evals)
        })
    
    summary_df = pd.DataFrame(summary_stats).sort_values('Mean MAPE')
    print(summary_df.to_string(index=False))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(output_dir, f"comparison_{model_name}_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    
    summary_file = os.path.join(output_dir, f"comparison_summary_{model_name}_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved: {summary_file}")
    
    return results_df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare HDMR with baseline optimizers')
    
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Model to optimize')
    parser.add_argument('--data', type=str, default='src/data/transactions.csv')
    parser.add_argument('--split', type=str, default='2020-01-01')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials for random search and Optuna')
    parser.add_argument('--hdmr-samples', type=int, default=1000,
                       help='Number of HDMR samples')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--methods', nargs='+',
                       default=['default', 'random_search', 'optuna', 'hdmr_standard', 'hdmr_adaptive'],
                       help='Methods to compare')
    parser.add_argument('--output-dir', type=str, default='results/comparisons')
    
    args = parser.parse_args()
    
    run_comparison(
        model_name=args.model,
        data_path=args.data,
        split_date=args.split,
        num_trials=args.trials,
        num_seeds=args.seeds,
        hdmr_samples=args.hdmr_samples,
        methods=args.methods,
        output_dir=args.output_dir
    )
    
    print("\n✓ Comparison completed successfully!")


if __name__ == "__main__":
    main()