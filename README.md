# HDMR Optimization

🚀 **High Dimensional Model Representation (HDMR) Optimization** is a research-oriented repository developed by the **APP2SCALE team** for computing global minimum points of mathematical functions using advanced optimization techniques.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-red.svg)](https://streamlit.io)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Main Modules](#main-modules)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Streamlit Web UI](#streamlit-web-ui)
  - [Benchmark Scripts](#benchmark-scripts)
  - [Forecasting Pipeline](#forecasting-pipeline)
- [Benchmark Functions](#benchmark-functions)
- [Time Series Forecasting](#time-series-forecasting)
- [Numerical Stability](#numerical-stability)
- [Production Deployment](#production-deployment)
- [About APP2SCALE](#about-app2scale)
- [License](#license)
- [References](#references)

---

## 🎯 Overview

HDMR Optimization reduces the complexity of high-dimensional optimization problems by decomposing a multivariate function into a sum of low-dimensional component functions. This repository provides code to compute the global minimum points of specified functions using two primary optimization techniques:

1. **HDMR-based optimization** - Extracts the one-dimensional form of a given function
2. **BFGS (Broyden–Fletcher–Goldfarb–Shanno)** - Traditional gradient-based method that directly optimizes the target function

### Mathematical Foundation

HDMR approximates f(x₁, x₂, ..., xₙ) as:

```
f(x) ≈ f₀ + Σᵢ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ, xⱼ) + ...
```

Each component fᵢ is represented using orthogonal basis functions (Legendre polynomials or Cosine functions).

---

## ✨ Features

- **HDMR-based global optimization** with orthogonal basis functions
- **Classical BFGS optimization** for comparison
- **Adaptive HDMR** with iterative refinement
- **Forecasting-based optimization** using XGBoost/LightGBM regression
- **Automated benchmark pipelines** for systematic testing
- **Interactive Streamlit Web UI** for easy experimentation
- **Command Line Interface** for automation and batch processing
- **Multiple benchmark test functions** (Rastrigin, Rosenbrock, Ackley, Griewank, etc.)
- **Visualization tools** for function landscapes and optimization paths
- **Production-ready deployment** with pinned dependencies
- **Comprehensive numerical stability** features

---

## 🔧 Installation

### Requirements

- **Python 3.9+** (Required, tested on Python 3.10)
- Virtual environment usage is **strongly recommended**
- Linux (CentOS 7+) or compatible OS

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/app2scale/hdmr-opt.git
   cd hdmr-opt
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using conda
   conda create -n hdmr-opt python=3.10
   conda activate hdmr-opt

   # Or using venv
   python -m venv hdmr-env
   source hdmr-env/bin/activate  # On Windows: hdmr-env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**Note**: The `requirements.txt` file contains pinned versions for production stability. All dependencies are tested for compatibility with Python 3.10 on CentOS 7.

---

## 📁 Repository Structure

```
hdmr-opt/
│
├── src/
│   ├── basis_functions.py      # Legendre & Cosine basis functions
│   ├── functions.py            # Benchmark test functions
│   ├── functions_forecast.py   # Forecasting-based optimization helpers
│   ├── main.py                 # HDMR optimization core
│   ├── forecast_example.py     # Time series forecasting example
│   ├── function_ranges.json    # Function domains
│   ├── optimum_points.json     # Known global minima
│   ├── __init__.py
│   └── data/
│       └── transactions.csv    # Time series data for forecasting
│
├── results/                    # Output files (reports & plots)
├── benchmark_2d.sh             # Automated 2D benchmark runner
├── forecast_pipeline.py        # Forecasting hyperparameter optimization pipeline
├── high_dim_test.py            # 10D benchmark runner
├── app.py                      # Streamlit Web UI
├── app_utils.py                # UI helper functions
├── requirements.txt            # Python dependencies (pinned versions)
├── README.md                   # This file
├── .gitignore
└── .gitattributes
```

---

## 📚 Main Modules

### `basis_functions.py`

Provides high-performance implementations of orthogonal basis functions used in HDMR decomposition:

- **Legendre polynomials** - Computed using three-term recurrence for numerical stability
- **Cosine basis functions** - Ideal for oscillatory functions
- **Factory pattern** for easy basis function selection

**Key Features**:
- Orthonormal on specified intervals
- Numerically stable implementations
- Complete basis for L²[a, b]

### `functions.py`

Contains benchmark optimization test functions used to evaluate algorithm performance.

**Can be run directly** to visualize functions:

```bash
python src/functions.py <function_name>
```

**Example**:
```bash
python src/functions.py camel3_2d
```

**Available Functions**:
- testfunc, camel3, camel16, treccani
- goldstein, branin, rosenbrock
- ackley, griewank, rastrigin

### `functions_forecast.py`

Contains forecasting-based optimization helpers with **production-ready date parsing**:

- **Multiple forecasting models**: XGBoost, LightGBM, ARIMA, ETS
- **Robust date parsing**: Supports MM/DD/YYYY and YYYY-MM-DD formats
- **Feature engineering**: Automatic lag features, rolling statistics, calendar features
- **Multiple metrics**: MAPE, RMSE, MAE, SMAPE, MASE
- **Safe error handling**: Graceful failure with informative messages

**Key Improvements in v2.2.0**:
- Strict MM/DD/YYYY date parsing with fallback to auto-detection
- Better error messages for date parsing failures
- Safer defaults (no mutable default arguments)
- Improved base class naming with backward compatibility

### `main.py`

Core optimization engine that:

- Implements HDMR and BFGS optimization methods
- Supports both standard and **adaptive HDMR**
- **Robust x0 parsing** - Supports single values, comma-separated lists, and broadcasting
- **Numerical stability guards** - NaN/Inf handling, soft bounds, basis overflow protection
- Generates comprehensive status reports
- Creates visualization plots
- Saves results to `results/` directory

**Version 3.0.0 Key Improvements**:
- Always returns OptimizeResult (never None)
- Eliminates unsafe global dependencies
- Correct SciPy minimize contract (fun: (n,) → float)
- Fixed adaptive refinement logic (consistent bounds + resampling)
- Safe visualization (never crashes main optimization)

---

## 🚀 Usage

### Command Line Interface

View all available options:

```bash
python src/main.py --help
```

#### Key Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--numSamples` | int | Number of samples to calculate alpha coefficients |
| `--numVariables` | int | Number of variables in the test function |
| `--function` | str | Test function name (e.g., `camel16_2d`) |
| `--min` | float | Lower range of the test function |
| `--max` | float | Upper range of the test function |
| `--x0` | float(s) | Starting point coordinates (e.g., `--x0 2.5 1.5`) |
| `--randomInit` | flag | Initialize x0 with random values in range |
| `--basisFunction` | str | `Legendre` or `Cosine` (default: Cosine) |
| `--degree` | int | Number of basis functions (default: 7) |
| `--adaptive` | flag | Enable iterative adaptive HDMR |
| `--maxiter` | int | Max adaptive iterations (default: 25) |
| `--numClosestPoints` | int | k for adaptive refinement (default: 100) |
| `--epsilon` | float | Convergence threshold (default: 0.1) |
| `--clip` | float | Clipping value for interval updates (default: 0.9) |
| `--numberOfRuns` | int | Number of test runs to calculate average error |
| `--seed` | int | Random seed for reproducibility |
| `--noPlots` | flag | Disable plot generation |
| `--disp` | flag | Verbose output |

### Examples

#### Standard HDMR Optimization

```bash
python -m src.main \
  --numSamples 1000 \
  --numVariables 2 \
  --function camel16_2d \
  --min -5 \
  --max 5
```

#### Adaptive HDMR with Custom Starting Point

```bash
python -m src.main \
  --numSamples 1000 \
  --numVariables 2 \
  --function camel16_2d \
  --min -5 \
  --max 5 \
  --adaptive \
  --epsilon 0.2 \
  --x0 2.5 1.5
```

#### Multiple Runs for Statistical Analysis

```bash
python -m src.main \
  --numSamples 1000 \
  --numVariables 2 \
  --function rosenbrock_2d \
  --numberOfRuns 10 \
  --seed 42
```

#### High-Dimensional Optimization (10D)

```bash
python -m src.main \
  --numSamples 2000 \
  --numVariables 10 \
  --function rastrigin_10d \
  --x0 0 \
  --adaptive \
  --noPlots
```

**Note**: For 10D functions, `--x0 0` broadcasts the single value to all 10 dimensions.

---

## 🌐 Streamlit Web UI

The repository includes an interactive web interface built with Streamlit.

### Running the Web UI

```bash
streamlit run app.py
```

### Accessing the UI

By default, the app runs at: **http://localhost:8501/**

### Features

- **Function selection** - Choose from available benchmark functions
- **Parameter tuning** - Adjust HDMR parameters interactively
- **Adaptive mode** - Toggle adaptive HDMR on/off
- **Real-time visualization** - Interactive plots using Plotly
- **Results display** - View optimization results immediately
- **No coding required** - User-friendly interface for experimentation
- **Initial point parser** - Supports various x0 input formats

---

## 🔬 Benchmark Scripts

The repository includes automated benchmark scripts for systematic testing.

### 2D Benchmark Runner (`benchmark_2d.sh`)

Runs all 2D benchmark functions with both standard and adaptive HDMR.

```bash
bash benchmark_2d.sh
```

**Features**:
- Tests 5 major 2D functions: rastrigin, rosenbrock, ackley, camel16, branin
- Runs both standard and adaptive HDMR for comparison
- Captures full logs for each run
- Generates summary report with key metrics
- Uses `python -m src.main` to avoid import issues

**Output**:
```
results/benchmark_2d/<timestamp>/
  ├── logs/
  │   ├── rastrigin_2d_standard.log
  │   ├── rastrigin_2d_adaptive.log
  │   └── ...
  └── summary.txt
```

**Environment Variables** (optional overrides):
```bash
PYTHON_BIN=python3.10 \
NUM_SAMPLES=2000 \
BASIS=Legendre \
DEGREE=10 \
ADAPTIVE_MAXITER=50 \
bash benchmark_2d.sh
```

### 10D High-Dimensional Runner (`high_dim_test.py`)

Runs 10-dimensional benchmark functions with automatic x0 initialization.

```bash
python high_dim_test.py
```

**Features**:
- Tests rosenbrock_10d, rastrigin_10d, griewank_10d
- **Automatic x0 injection** based on function optimal points:
  - rosenbrock_10d: ones(10)
  - griewank_10d: 100s(10)
  - rastrigin_10d: zeros(10)
- Runs both standard and adaptive modes
- JSON summary output with detailed results

**Output**:
```
results/high_dim_tests/
  └── summary.json
```

---

## 📈 Forecasting Pipeline

### Automated Hyperparameter Optimization (`forecast_pipeline.py`)

Systematically optimizes forecasting models across multiple algorithms and metrics.

```bash
python forecast_pipeline.py
```

**Features**:
- Tests 4 algorithms: xgboost, lightgbm, arima, ets
- Optimizes 3 metrics: mape, rmse, mae
- Total: 12 optimization runs (4 algorithms × 3 metrics)
- Captures stdout/stderr for each run
- Writes per-run logs and JSON summary
- Timeout protection (default: 20 minutes per run)

**Output**:
```
results/forecasting/<timestamp>/
  ├── logs/
  │   ├── xgboost_mape.log
  │   ├── xgboost_rmse.log
  │   └── ...
  └── optimization_summary.json
```

**Environment Variables**:
```bash
PYTHON_BIN=python
RUN_ID=my_experiment
SAMPLES=1000
BASIS=Cosine
DEGREE=7
ADAPTIVE=1
MAXITER=25
TIMEOUT_SEC=1200
python forecast_pipeline.py
```

### Single Forecasting Example (`forecast_example.py`)

Optimizes hyperparameters for a single forecasting model.

```bash
python -m src.forecast_example \
  --algorithm xgboost \
  --metric mape \
  --samples 1000 \
  --adaptive \
  --no-plots
```

**Supported Algorithms**:
- `xgboost` - Gradient boosting (hyperparams: learning_rate, max_depth, subsample, etc.)
- `lightgbm` - Fast gradient boosting (hyperparams: learning_rate, num_leaves, etc.)
- `arima` - AutoRegressive Integrated Moving Average (hyperparams: p, d, q)
- `ets` - Exponential Smoothing (hyperparams: seasonal_periods)

**Command-Line Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | xgboost | Algorithm: xgboost, lightgbm, arima, ets |
| `--data` | str | src/data/transactions.csv | Path to CSV file |
| `--split` | str | 2020-01-01 | Train/test split date (YYYY-MM-DD) |
| `--metric` | str | mape | Metric: mape, rmse, mae, smape |
| `--samples` | int | 1000 | Number of HDMR samples |
| `--basis` | str | Cosine | Basis function: Legendre or Cosine |
| `--degree` | int | 7 | Basis function degree |
| `--adaptive` | flag | False | Enable adaptive HDMR |
| `--maxiter` | int | 25 | Maximum adaptive iterations |
| `--numClosestPoints` | int | 100 | k for adaptive refinement |
| `--epsilon` | float | 0.1 | Convergence threshold |
| `--clip` | float | 0.9 | Minimum shrink ratio |
| `--seed` | int | None | Random seed |
| `--quiet` | flag | False | Suppress progress output |
| `--no-plots` | flag | False | Disable plot generation |

**Example: Adaptive Optimization with LightGBM**

```bash
python -m src.forecast_example \
  --algorithm lightgbm \
  --metric rmse \
  --samples 2000 \
  --basis Legendre \
  --degree 10 \
  --adaptive \
  --maxiter 50 \
  --epsilon 0.05 \
  --no-plots
```

---

## 📊 Benchmark Functions

The repository includes the following benchmark test functions:

### 2D Functions

| Function | Domain | Global Minimum | Value |
|----------|--------|----------------|-------|
| testfunc | [-5, 5] | (0, 0) | 0 |
| camel3 | [-5, 5] | (0, 0) | 0 |
| camel16 | [-5, 5] | (±0.0898, ∓0.7126) | -1.0316 |
| treccani | [-5, 5] | (-2, 0) or (0, 0) | 0 |
| goldstein | [-2, 2] | (0, -1) | 3 |
| branin | [-5, 15] | Multiple optima | 0.397887 |
| rosenbrock | [-2.048, 2.048] | (1, 1) | 0 |
| ackley | [-30, 30] | (0, 0) | 0 |

### High-Dimensional Functions

| Function | Dimensions | Domain | Global Minimum |
|----------|-----------|--------|----------------|
| rastrigin | 2D, 10D | [-5.12, 5.12] | All zeros |
| rosenbrock | 10D | [-5, 10] | All ones |
| griewank | 10D | [-600, 600] | (100, 100, ...) |

All function definitions, domains, and known global minima are stored in:
- `src/function_ranges.json` - Domain definitions
- `src/optimum_points.json` - Known optimal solutions

---

## 🔮 Time Series Forecasting

### Data Format

The forecasting module expects a CSV file with:
- `date`: Date column (format: **MM/DD/YYYY** or YYYY-MM-DD)
- `transactions`: Numeric transaction count or value

**Example** (`src/data/transactions.csv`):
```csv
date,transactions
10/1/2015,4.004739185
10/2/2015,4.139078221
10/3/2015,2.540515455
```

**Date Parsing Policy** (v2.2.0):
- Primary format: **MM/DD/YYYY** (e.g., 10/1/2015)
- Fallback: Auto-detection for YYYY-MM-DD and mixed formats
- Strict validation with informative error messages

### Evaluation Metrics

| Metric | Description | Best Value |
|--------|-------------|------------|
| **MAPE** | Mean Absolute Percentage Error | Lower |
| **SMAPE** | Symmetric MAPE | Lower |
| **MAE** | Mean Absolute Error | Lower |
| **RMSE** | Root Mean Squared Error | Lower |
| **MASE** | Mean Absolute Scaled Error | < 1 |

### Programmatic Usage

```python
from src.functions_forecast import (
    XGBoostForecaster,
    prepare_train_test,
    create_optimization_objective,
    calculate_metrics
)

# Load and prepare data (with date format validation)
data = prepare_train_test(
    'src/data/transactions.csv',
    split_date='2020-01-01',
    strict_dates=True,  # Enforce MM/DD/YYYY format
    date_format='%m/%d/%Y'
)

# Create optimization objective
objective = create_optimization_objective(
    model_class=XGBoostForecaster,
    data_dict=data,
    metric='mape'
)

# Use HDMR optimizer from main.py
from src.main import HDMROptimizer, HDMRConfig
import numpy as np

# Get hyperparameter space
hyperparam_space = XGBoostForecaster().get_hyperparameter_space()
param_names = list(hyperparam_space.keys())
a_vec = np.array([hyperparam_space[p][0] for p in param_names])
b_vec = np.array([hyperparam_space[p][1] for p in param_names])

# Configure HDMR
config = HDMRConfig(
    n=len(param_names),
    a=a_vec,
    b=b_vec,
    N=1000,
    m=7,
    basis='Cosine',
    adaptive=True,
    maxiter=25
)

# Batch objective wrapper
def objective_batch(X):
    X = X.reshape(-1, len(param_names))
    return np.array([[objective(x)] for x in X])

# Optimize
optimizer = HDMROptimizer(fun_batch=objective_batch, config=config)
result = optimizer.solve(x0=0.5 * (a_vec + b_vec))

# Extract optimal parameters
optimal_params = {param_names[i]: result.x[i] for i in range(len(param_names))}

# Train final model
model = XGBoostForecaster(**optimal_params)
model.fit(data['X_train'], data['y_train'])
predictions = model.predict(data['X_test'])

# Evaluate
metrics = calculate_metrics(data['y_test'], predictions, data['y_train'])
print(f"Test MAPE: {metrics['mape']:.2f}%")
```

---

## 🔒 Numerical Stability

The implementation includes several features to ensure numerical stability:

### Basis Function Evaluation
- **Input scaling and clipping** - Prevents overflow/underflow
- **Three-term recurrence** for Legendre polynomials - Numerically stable
- **Normalization factors** - Ensures orthonormality

### Optimization
- **NaN/Inf guards** - Detects and handles numerical issues
- **Soft penalties** for out-of-bound steps - Guides optimization away from invalid regions
- **Finite checks** in surrogate evaluation - Returns safe fallback (1e30) on overflow
- **Bounds enforcement** - Clips solution to valid domain

### Adaptive Refinement
- **Clip guard** - Prevents bounds from shrinking below minimum range
- **Absolute bounds enforcement** - Never exceeds original domain
- **Valid interval guarantee** - Ensures b > a + epsilon

### Error Handling
- **Try-catch wrappers** - Visualization errors never crash optimization
- **Graceful degradation** - Returns valid OptimizeResult even on failure
- **Informative error messages** - Helps debug issues quickly

---

## 🚀 Production Deployment

### Dependency Management

The repository uses **pinned versions** in `requirements.txt` for production stability:

```txt
numpy==1.26.4
scipy==1.15.3
pandas==2.2.3
matplotlib==3.10.0
xgboost==3.0.5
lightgbm==3.3.5
scikit-learn==1.3.2
statsmodels==0.14.4
streamlit==1.41.1
```

**Tested on**:
- Python 3.10
- CentOS 7 (Linux)
- Compatible with Python 3.9+

### Best Practices

1. **Use virtual environments**:
   ```bash
   python -m venv hdmr-env
   source hdmr-env/bin/activate
   pip install -r requirements.txt
   ```

2. **Run as module** (avoids import issues):
   ```bash
   python -m src.main [args]
   python -m src.forecast_example [args]
   ```

3. **Disable plots for batch jobs**:
   ```bash
   python -m src.main --function rastrigin_10d --noPlots
   ```

4. **Set random seed for reproducibility**:
   ```bash
   python -m src.main --seed 42 [other args]
   ```

5. **Use adaptive mode for difficult functions**:
   ```bash
   python -m src.main --adaptive --maxiter 50 --epsilon 0.05
   ```

### Continuous Integration

For CI/CD pipelines, use the automated benchmark scripts:

```bash
# Test all 2D functions
bash benchmark_2d.sh

# Test 10D functions
python high_dim_test.py

# Test forecasting pipeline
python forecast_pipeline.py
```

All scripts generate JSON summaries for automated result parsing.

---

## 👥 About APP2SCALE

This repository is developed and maintained by the **APP2SCALE team** as part of scalable optimization research. The project focuses on:

- High-dimensional optimization techniques
- Efficient decomposition methods
- Practical applications of HDMR
- Time series forecasting with ML/statistical models
- Benchmarking and comparison of optimization algorithms
- Production-ready scientific computing tools

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📖 References

1. **Sobol, I. M., et al. (2003)** - High Dimensional Model Representation and its Application Variants
2. **Surjanovic, S. & Bingham, D. (2013)** - Virtual Library of Simulation Experiments: Test Functions and Datasets
3. **Nocedal, J. & Wright, S. J. (2006)** - Numerical Optimization, Springer Series in Operations Research
4. **Chen, T. & Guestrin, C. (2016)** - XGBoost: A Scalable Tree Boosting System
5. **Ke, G., et al. (2017)** - LightGBM: A Highly Efficient Gradient Boosting Decision Tree

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration opportunities, please contact the APP2SCALE team through the repository issues page.

---

## 📝 Changelog

### Version 3.0.0 (2026-01-13)

**Major Refactoring - Production Ready**

- ✅ **Core Engine (`main.py`)**:
  - Robust x0 parsing (supports broadcasting, pattern repeat)
  - Always returns OptimizeResult (never None)
  - Fixed surrogate evaluation (correct 1D optimization per dimension)
  - Numerical stability hardening (NaN/Inf guards, soft bounds)
  - Safe visualization (never crashes optimization)

- ✅ **Forecasting Module (`functions_forecast.py`)**:
  - Strict MM/DD/YYYY date parsing with auto-detection fallback
  - Better error messages for date parsing failures
  - Safer defaults (no mutable default arguments)
  - BaseForecaster class with backward compatibility

- ✅ **Automation**:
  - Added `benchmark_2d.sh` for 2D function testing
  - Added `forecast_pipeline.py` for forecasting hyperparameter optimization
  - Added `high_dim_test.py` for 10D function testing
  - All scripts use `python -m` for import safety

- ✅ **Documentation**:
  - Complete README overhaul
  - Added usage examples for all scripts
  - Documented production deployment best practices
  - Added troubleshooting guide

### Version 2.0.0 (2024-11-30)

- Added `--numberOfRuns` parameter for statistical analysis
- Improved basis functions module with factory pattern
- Enhanced numerical stability

### Version 1.0.0 (2023-09-11)

- Initial release
- Added `--x0` command-line parameter
- Core HDMR and BFGS implementation
- Streamlit web interface

---

**Note**: This project requires **Python 3.9+** and is optimized for production deployment on Linux (CentOS 7+). Always use a virtual environment to avoid dependency conflicts.
