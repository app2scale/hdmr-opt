# HDMR Optimization

🚀 **High Dimensional Model Representation (HDMR) Optimization** - Research framework for hyperparameter optimization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Compute%20Capsule-blue.svg)](https://codeocean.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 📁 Repository Structure
```
hdmr-opt/
│
├── codeocean/                     # Professional Code Ocean Compute Capsule
│   ├── code/                      # Capsule entry points (main.py, run.sh, scripts/)
│   ├── data/                      # Industrial datasets (Payten, Medianova)
│   └── README_CodeOcean.md        # Platform-specific instructions
│
├── src/                           # Core HDMR library (v4.0)
│   ├── main.py                    # HDMR optimizer engine
│   ├── basis_functions.py         # Orthogonal basis functions
│   ├── functions.py               # Benchmark test functions
│   ├── functions_forecast.py      # Forecasting models logic
│   ├── function_ranges.json       # Function domains
│   └── optimum_points.json        # Known global minima
│
├── experiments/                   # Main research scripts
│   ├── compare_optimizers.py      # Compare HDMR vs Optuna vs Random Search
│   ├── sensitivity_analysis.py    # Hyperparameter importance analysis
│   └── benchmark_forecasting.py   # Deep learning benchmarks (LSTM, GRU, N-BEATS)
│
├── analysis/                      # Visualization & reporting tools
│   ├── analyze_results_v2.py      # Advanced analysis (Pareto fronts, etc.)
│   ├── create_final_visualization.py  # Publication-ready plots
│   └── create_summary_report.py   # Text-based summary reports
│
├── scripts/                       # Benchmarking & Automation
│   ├── tabarena_hdmr_lgb.py       # Tabular benchmark script
│   ├── forecast_hpo.py            # Forecasting optimization pipeline
│   ├── run_all_experiments.py     # Full experimental pipeline runner
│   └── run_all_tabarena.sh        # Batch runner for tabular tests
│
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/app2scale/hdmr-opt.git
cd hdmr-opt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Compare Optimization Methods
```bash
python experiments/compare_optimizers.py \
  --model xgboost \
  --trials 20 \
  --seeds 3
```

### 3. Sensitivity Analysis
```bash
python experiments/sensitivity_analysis.py \
  --model xgboost \
  --samples 200 \
  --seeds 3
```

### 4. Mathematical Benchmark (10D)
```bash
python src/main.py --function Rastrigin --numVariables 10 --adaptive --seed 42
```

---

## 📊 Key Scripts

### Experiments

| Script | Purpose | Usage |
|--------|---------|-------|
| `experiments/compare_optimizers.py` | Compare HDMR vs baselines | `--model xgboost --trials 20` |
| `experiments/sensitivity_analysis.py` | Hyperparameter importance | `--model xgboost --samples 200` |
| `experiments/benchmark_forecasting.py` | Deep learning benchmarks | `--models lstm gru nbeats` |
| `scripts/forecast_hpo.py` | Forecasting optimization | `--dataset payten --methods hdmr` |

### Analysis

| Script | Purpose |
|--------|---------|
| `analysis/analyze_results_v2.py` | Advanced analysis (Pareto, trade-offs) |
| `analysis/create_final_visualization.py` | Publication-ready figures |
| `analysis/create_summary_report.py` | Text summaries |

---

## 📚 Core API
```python
from src.main import HDMROptimizer, HDMRConfig

# Configure HDMR
config = HDMRConfig(n=5, a=[0.01, 1], b=[0.3, 10], N=1000)

# Optimize
optimizer = HDMROptimizer(objective_function, config)
result = optimizer.solve(x0=[0.1, 5])
```

---

## 🌐 Web Interface
```bash
streamlit run app.py
```

Access at: http://localhost:8501

---

## 📄 License

MIT License - see LICENSE file

---

## 📖 References

1. Sobol, I. M., et al. (2003) - High Dimensional Model Representation
2. Chen, T. & Guestrin, C. (2016) - XGBoost: Scalable Tree Boosting
3. Akiba, T., et al. (2019) - Optuna: Hyperparameter Optimization Framework
4. Oreshkin, B. N., et al. (2020) - N-BEATS: Neural Basis Expansion

---

## 📝 Changelog

### v4.0.0 (2026-04-24) - Code Ocean & Production Ready

**Repository Reorganization:**
- Standardized directory layout (experiments/, analysis/, scripts/, codeocean/)
- Integrated Code Ocean Compute Capsule for full reproducibility
- Unified entry point via main.py dispatcher in capsule
- Refactored all scripts to use relative pathing and absolute results volumes

**Features:**
- Professional orchestration via run.sh
- Enhanced industrial forecasting data loaders
- Standardized TabArena benchmark suite
- Deterministic seeding (SEED=42) across all optimization modules

**Developed by APP2SCALE Team**

---

### v3.0.0 (2026-01-13) - Core Hardening

**Core Engine Improvements:**
- Robust x0 parsing (supports broadcasting, pattern repeat)
- Always returns OptimizeResult (never None)
- Fixed surrogate evaluation (correct 1D optimization per dimension)
- Numerical stability hardening (NaN/Inf guards, soft bounds)
- Safe visualization (never crashes optimization)

**Forecasting Module:**
- Strict MM/DD/YYYY date parsing with auto-detection fallback
- Better error messages for date parsing failures
- Safer defaults (no mutable default arguments)
- BaseForecaster class with backward compatibility

**Developed by APP2SCALE Team**

---

### v2.0.0 (2024-11-30)

**Features:**
- Added --numberOfRuns parameter for statistical analysis
- Improved basis functions module with factory pattern
- Enhanced numerical stability

**Developed by APP2SCALE Team**

---

### v1.0.0 (2023-09-11)

**Initial Release:**
- Core HDMR + BFGS implementation
- Streamlit web interface
- Command-line interface
- Basic benchmark functions
- Added --x0 parameter for custom starting points

---

**Developed by APP2SCALE Team**
