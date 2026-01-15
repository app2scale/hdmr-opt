# HDMR Optimization

🚀 **High Dimensional Model Representation (HDMR) Optimization** - Research framework for hyperparameter optimization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 📁 Repository Structure
```
hdmr-opt/
│
├── src/                           # Core HDMR library
│   ├── main.py                    # HDMR optimizer engine
│   ├── basis_functions.py         # Orthogonal basis functions
│   ├── functions.py               # Benchmark test functions
│   ├── functions_forecast.py      # Forecasting models (XGBoost, LSTM, etc.)
│   ├── function_ranges.json       # Function domains
│   ├── optimum_points.json        # Known global minima
│   └── data/transactions.csv      # Example time series data
│
├── experiments/                   # Main research scripts
│   ├── compare_optimizers.py      # Compare HDMR vs Optuna vs Random Search
│   ├── sensitivity_analysis.py    # Hyperparameter importance analysis
│   ├── benchmark_forecasting.py   # Deep learning benchmarks (LSTM, GRU, N-BEATS)
│   └── forecast_example.py        # Single model optimization
│
├── analysis/                      # Visualization & reporting tools
│   ├── analyze_results.py         # Basic result visualization
│   ├── analyze_results_v2.py      # Advanced analysis (Pareto fronts, etc.)
│   ├── create_final_visualization.py  # Publication-ready plots
│   └── create_summary_report.py   # Text-based summary reports
│
├── automation/                    # Batch processing scripts
│   ├── run_all_experiments.py     # Full experimental pipeline
│   └── run_hdmr_clean.sh          # Shell-based batch runner
│
├── legacy/                        # Older benchmark scripts
│   ├── benchmark_2d.sh            # 2D function benchmarks
│   ├── forecast_pipeline.py       # Legacy forecasting pipeline
│   └── high_dim_test.py           # 10D benchmark runner
│
├── docker/                        # Containerization files
│   ├── Dockerfile                 # GPU-accelerated container
│   └── docker-compose.yml         # Multi-experiment orchestration
│
├── app.py                         # Streamlit web interface
├── app_utils.py                   # UI helper functions
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
python -m venv hdmr-env
source hdmr-env/bin/activate

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

### 4. Single Model Optimization
```bash
python experiments/forecast_example.py \
  --algorithm xgboost \
  --metric mape \
  --samples 1000
```

### 5. Deep Learning Benchmarks
```bash
python experiments/benchmark_forecasting.py \
  --models lstm gru nbeats \
  --seeds 3
```

---

## 🐳 Docker Usage

### Build and Run
```bash
# Build image
docker build -t hdmr-opt -f docker/Dockerfile .

# Run single experiment
docker run --gpus all \
  -v $(pwd)/results:/workspace/results \
  hdmr-opt \
  python3 experiments/compare_optimizers.py --model xgboost
```

### Docker Compose (Multiple Experiments)
```bash
cd docker
docker-compose up
```

---

## 📊 Key Scripts

### Experiments

| Script | Purpose | Usage |
|--------|---------|-------|
| `experiments/compare_optimizers.py` | Compare HDMR vs baselines | `--model xgboost --trials 20` |
| `experiments/sensitivity_analysis.py` | Hyperparameter importance | `--model xgboost --samples 200` |
| `experiments/benchmark_forecasting.py` | Deep learning benchmarks | `--models lstm gru nbeats` |
| `experiments/forecast_example.py` | Single optimization | `--algorithm xgboost --metric mape` |

### Analysis

| Script | Purpose |
|--------|---------|
| `analysis/analyze_results.py` | Basic visualization |
| `analysis/analyze_results_v2.py` | Advanced analysis (Pareto, trade-offs) |
| `analysis/create_final_visualization.py` | Publication-ready figures |
| `analysis/create_summary_report.py` | Text summaries |

### Automation

| Script | Purpose |
|--------|---------|
| `automation/run_all_experiments.py` | Run full pipeline |
| `automation/run_hdmr_clean.sh` | Shell-based batch runs |

---

## 📚 Core API
```python
from src.main import HDMROptimizer, HDMRConfig
from src.functions_forecast import XGBoostForecaster, prepare_train_test

# Prepare data
data = prepare_train_test('src/data/transactions.csv', '2020-01-01')

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

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

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

### v4.0.0 (2026-01-15) - Organized Structure

**Repository Reorganization:**
- Structured directory layout (experiments/, analysis/, automation/)
- Fixed import paths for all scripts
- Updated Docker configuration
- Improved documentation

**Features:**
- Optimizer comparison framework
- Sensitivity analysis tool
- Deep learning benchmarks
- Docker containerization

---

**Developed by APP2SCALE Team**

### v3.0.0 (2026-01-13) - Production Ready

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

**Automation:**
- Added benchmark_2d.sh for 2D function testing
- Added forecast_pipeline.py for forecasting optimization
- Added high_dim_test.py for 10D function testing
- All scripts use python -m for import safety

**Documentation:**
- Complete README overhaul
- Added usage examples for all scripts
- Documented production deployment best practices

### v2.0.0 (2024-11-30)

**Features:**
- Added --numberOfRuns parameter for statistical analysis
- Improved basis functions module with factory pattern
- Enhanced numerical stability

**Infrastructure:**
- Better error handling and logging
- Improved test coverage

### v1.0.0 (2023-09-11)

**Initial Release:**
- Core HDMR + BFGS implementation
- Streamlit web interface
- Command-line interface
- Basic benchmark functions
- Added --x0 parameter for custom starting points

---

**Developed by APP2SCALE Team**
