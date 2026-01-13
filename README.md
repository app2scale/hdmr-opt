# HDMR Optimization

🚀 **High Dimensional Model Representation (HDMR) Optimization** is a research-oriented repository developed by the **APP2SCALE team** for computing global minimum points of mathematical functions using advanced optimization techniques.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

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
  - [Examples](#examples)
- [Benchmark Functions](#benchmark-functions)
- [Numerical Stability](#numerical-stability)
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
- **Forecasting-based optimization** using XGBoost regression
- **Interactive Streamlit Web UI** for easy experimentation
- **Command Line Interface** for automation and batch processing
- **Multiple benchmark test functions** (Rastrigin, Rosenbrock, Ackley, Griewank, etc.)
- **Visualization tools** for function landscapes and optimization paths
- **Comprehensive numerical stability** features

---

## 🔧 Installation

### Requirements

- **Python 3.9+** (Required)
- Virtual environment usage is **strongly recommended**

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/app2scale/hdmr-opt.git
   cd hdmr-opt
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using conda
   conda create -n hdmr-opt python=3.9
   conda activate hdmr-opt

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

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
│   ├── function_ranges.json    # Function domains
│   ├── optimum_points.json     # Known global minima
│   ├── __init__.py
│   └── data/
│       └── transactions.csv    # Time series data for forecasting
│
├── results/                    # Output files (reports & plots)
├── forecast_example.py         # Example script for forecasting
├── app.py                      # Streamlit Web UI
├── app_utils.py                # UI helper functions
├── requirements.txt            # Python dependencies
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

This plots the specified function for visual inspection.

**Available Functions**:
- testfunc, camel3, camel16, treccani
- goldstein, branin, rosenbrock
- ackley, griewank, rastrigin

### `functions_forecast.py`

Contains forecasting-based optimization helpers:

- **`optimize_helper` function** - Uses XGBoost regression for time-series optimization
- **MAPE metric** - Mean Absolute Percentage Error for evaluation
- Optimizes predicted transactions on future dates
- Configurable learning rate and subsample parameters

### `main.py`

Core optimization engine that:

- Implements HDMR and BFGS optimization methods
- Supports both standard and **adaptive HDMR**
- Generates comprehensive status reports
- Creates visualization plots
- Saves results to `results/` directory

**Outputs**:
- `<parameters>.txt` - Status reports for both BFGS and HDMR methods
- `<parameters>.png` - HDMR component function plots

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
| `--legendreDegree` | int | Number of Legendre polynomials (default: 7) |
| `--adaptive` | flag | Enable iterative adaptive HDMR |
| `--numClosestPoints` | int | Number of closest points to x0 (default: 1000) |
| `--epsilon` | float | Convergence threshold (default: 0.1) |
| `--clip` | float | Clipping value for interval updates (default: 0.9) |
| `--numberOfRuns` | int | Number of test runs to calculate average error |

### Examples

#### Standard HDMR Optimization

```bash
python src/main.py \
  --numSamples 1000 \
  --numVariables 2 \
  --function camel16_2d \
  --min -5 \
  --max 5
```

#### Adaptive HDMR with Custom Starting Point

```bash
python src/main.py \
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
python src/main.py \
  --numSamples 1000 \
  --numVariables 2 \
  --function rosenbrock_2d \
  --min -2.048 \
  --max 2.048 \
  --numberOfRuns 10
```

**Note**: When using `--numberOfRuns` with `--adaptive`, be careful as some extreme parameter combinations may lead to very high errors.

#### Using Legendre Basis Functions

```bash
python src/main.py \
  --numSamples 1000 \
  --numVariables 2 \
  --function ackley_2d \
  --min -30 \
  --max 30 \
  --basisFunction Legendre \
  --legendreDegree 10
```

---

## 🌐 Streamlit Web UI

The repository includes an interactive web interface built with Streamlit.

### Running the Web UI

1. Ensure you're in the main project directory (where you see `results/`, `src/`, etc.)
2. Open terminal in this folder
3. Run:
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

---

## 🔮 HDMR for Time Series Forecasting

In addition to benchmark function optimization, this repository supports **HDMR-based hyperparameter optimization for time series forecasting models**. This feature allows you to optimize forecasting algorithms using real-world transaction data.

### Supported Forecasting Algorithms

| Algorithm | Description | Key Hyperparameters |
|-----------|-------------|---------------------|
| **XGBoost** | Gradient boosting regression | learning_rate, max_depth, subsample |
| **LightGBM** | Fast gradient boosting | learning_rate, num_leaves, min_data_in_leaf |
| **ARIMA** | AutoRegressive Integrated Moving Average | p, d, q (order parameters) |
| **ETS** | Exponential Smoothing | trend, seasonal, seasonal_periods |

### Data Format

The forecasting module expects a CSV file with two columns:
- `date`: Date of transaction (format: MM/DD/YYYY)
- `transactions`: Transaction count or value

**Example** (`src/data/transactions.csv`):
```csv
date,transactions
10/1/2015,4.004739185
10/2/2015,4.139078221
10/3/2015,2.540515455
10/4/2015,2.265059112
10/5/2015,4.001959428
```

### Forecasting Example Usage

#### Quick Start with XGBoost

```bash
python forecast_example.py \
  --algorithm xgboost \
  --data src/data/transactions.csv \
  --split 2020-01-01 \
  --metric mape \
  --samples 1000
```

#### Adaptive HDMR with LightGBM

```bash
python forecast_example.py \
  --algorithm lightgbm \
  --metric rmse \
  --samples 2000 \
  --basis Legendre \
  --degree 10 \
  --adaptive \
  --maxiter 50 \
  --epsilon 0.05
```

#### ARIMA Model Optimization

```bash
python forecast_example.py \
  --algorithm arima \
  --metric mae \
  --samples 500
```

#### Available Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | xgboost | Algorithm: xgboost, lightgbm, arima, ets |
| `--data` | str | src/data/transactions.csv | Path to CSV file |
| `--split` | str | 2020-01-01 | Train/test split date (YYYY-MM-DD) |
| `--metric` | str | mape | Metric to optimize: mape, rmse, mae, smape |
| `--samples` | int | 1000 | Number of HDMR samples |
| `--basis` | str | Cosine | Basis function: Legendre or Cosine |
| `--degree` | int | 7 | Basis function degree |
| `--adaptive` | flag | False | Enable adaptive HDMR |
| `--maxiter` | int | 25 | Maximum adaptive iterations |
| `--numClosestPoints` | int | 100 | k for adaptive refinement |
| `--epsilon` | float | 0.1 | Convergence threshold |
| `--clip` | float | 0.9 | Minimum shrink ratio |
| `--seed` | int | None | Random seed for reproducibility |
| `--quiet` | flag | False | Suppress progress output |

### Evaluation Metrics

The following metrics are available for forecasting evaluation:

| Metric | Description | Best Value |
|--------|-------------|------------|
| **MAPE** | Mean Absolute Percentage Error | Lower |
| **SMAPE** | Symmetric MAPE | Lower |
| **MAE** | Mean Absolute Error | Lower |
| **RMSE** | Root Mean Squared Error | Lower |
| **MASE** | Mean Absolute Scaled Error | < 1 |

### Programmatic Usage

You can also use the forecasting module programmatically:

```python
from src.functions_forecast import (
    XGBoostForecaster,
    prepare_train_test,
    create_optimization_objective,
    calculate_metrics
)

# Load and prepare data
data = prepare_train_test('src/data/transactions.csv', '2020-01-01')

# Create optimization objective
objective = create_optimization_objective(
    model_class=XGBoostForecaster,
    data_dict=data,
    metric='mape'
)

# Optimize using HDMR (integrate with main.py)
# optimal_params = hdmr_optimizer(objective, ...)

# Train final model
model = XGBoostForecaster(learning_rate=0.1, max_depth=3)
model.fit(data['X_train'], data['y_train'])
predictions = model.predict(data['X_test'])

# Evaluate
metrics = calculate_metrics(data['y_test'], predictions)
print(f"Test MAPE: {metrics['mape']:.2f}%")
```

### Legacy `optimize_helper` Function

For backward compatibility, the simple `optimize_helper` function is still available:

```python
from src.functions_forecast import optimize_helper

# Optimize 2 XGBoost parameters
mape = optimize_helper(learning_rate=0.1, subsample=0.8)
print(f"MAPE: {mape:.2f}%")
```

This function uses hardcoded paths:
- Data: `./src/data/transactions.csv` or `./transactions.csv`
- Split: `01-01-2020`

**Note**: For production use, prefer `create_optimization_objective()` for flexibility.

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

## 🔒 Numerical Stability

The implementation includes several features to ensure numerical stability:

- **Basis input scaling and clipping** - Prevents overflow/underflow
- **NaN/Inf guards** - Detects and handles numerical issues
- **Soft penalties** for out-of-bound steps - Guides optimization away from invalid regions
- **Three-term recurrence** for Legendre polynomials - Numerically stable computation
- **Normalization factors** - Ensures orthonormality of basis functions

---

## 👥 About APP2SCALE

This repository is developed and maintained by the **APP2SCALE team** as part of scalable optimization research. The project focuses on:

- High-dimensional optimization techniques
- Efficient decomposition methods
- Practical applications of HDMR
- Benchmarking and comparison of optimization algorithms

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📖 References

1. **Sobol, I. M., et al. (2003)** - High Dimensional Model Representation and its Application Variants
2. **Surjanovic, S. & Bingham, D. (2013)** - Virtual Library of Simulation Experiments: Test Functions and Datasets
3. **Nocedal, J. & Wright, S. J. (2006)** - Numerical Optimization, Springer Series in Operations Research

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or collaboration opportunities, please contact the APP2SCALE team through the repository issues page.

---

## 🔄 Updates

- **2023-09-11**: Added `--x0` command-line parameter for custom starting points
- **2023-11-30**: Added `--numberOfRuns` parameter for statistical analysis across multiple trials
- **2026-01-12**: Major update with improved documentation, basis functions module refactoring, and enhanced numerical stability

---

**Note**: This project is compatible with **Python 3.9+**. It is strongly recommended to set up a dedicated Conda or virtual environment to avoid compatibility issues due to library versions.
