# HDMR-opt: Sensitivity-Guided Adaptive HDMR for Optimization

🚀 **High Dimensional Model Representation (HDMR)** - A professional research framework for hyperparameter optimization and global sensitivity analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Compute%20Capsule-blue.svg)](https://codeocean.com/)
[![IEEE](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieee.org)

---

## 🔬 Overview

HDMR-opt is an advanced optimization engine that leverages **High Dimensional Model Representation** to escape local minima and provide global sensitivity insights. It is particularly effective for high-dimensional hyperparameter optimization (HPO) in machine learning and industrial time-series forecasting.

### Key Features
- **Sensitivity-Guided Refinement:** Adaptive bound shrinkage based on Sobol sensitivity indices.
- **Quasi-Random Sampling:** Utilizes Sobol sequences for superior space-filling coverage.
- **Multi-Study Support:** Benchmarked on TabArena (Tabular Data), Industrial Forecasting, and Mathematical Test Functions.
- **Reproducible Research:** Fully compatible with Code Ocean Compute Capsules.

---

## 📁 Repository Structure

```
hdmr-opt/
├── codeocean/             # Professional Code Ocean Compute Capsule
│   ├── code/              # Capsule entry points (main.py, run.sh, scripts/)
│   ├── data/              # Industrial datasets (Payten, Medianova)
│   └── README_CodeOcean.md# Platform-specific instructions
│
├── src/                   # Core HDMR Library (v4.0)
│   ├── main.py            # Optimizer engine
│   ├── basis_functions.py # Basis implementations
│   ├── functions.py       # Math test functions
│   └── functions_forecast.py # Forecasting logic
│
├── experiments/           # Research experiments
│   ├── compare_optimizers.py
│   ├── sensitivity_analysis.py
│   └── benchmark_forecasting.py
│
├── analysis/              # Visualization tools
│   ├── analyze_results_v2.py
│   └── create_final_visualization.py
│
├── scripts/               # Benchmarking & Automation
│   ├── tabarena_hdmr_lgb.py
│   ├── forecast_hpo.py
│   └── run_all_experiments.py
│
├── data/                  # Local datasets (CSV/ARFF)
├── app.py                 # Streamlit interface
└── requirements.txt       # Dependencies
```


---

## 🚀 Reproducibility (Code Ocean)

The most reliable way to replicate the results presented in our research is via the **Code Ocean Compute Capsule**.

1. **Import** this repository into Code Ocean.
2. Ensure you have the datasets in the `/data` directory.
3. Use the environment variable `STUDY` to switch between modes:
   - `STUDY=tabular` (TabArena benchmarks)
   - `STUDY=forecasting` (Industrial cases)
   - `STUDY=benchmark` (Mathematical test functions)
4. Click **Run** to execute the master `run.sh` script.

---

## 💻 Local Installation

```bash
# Clone and enter
git clone https://github.com/app2scale/hdmr-opt.git
cd hdmr-opt

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick Example: Mathematical Benchmark
```bash
# Run 10D Rastrigin function optimization
python src/main.py --function Rastrigin --numVariables 10 --adaptive --seed 42
```

---

## 🌐 Web Interface

Experience HDMR-opt interactively using the built-in Streamlit application:

```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

---

## 📝 Citation

If you use this framework in your research, please cite our paper:

```bibtex
@article{hdmr_opt_2026,
  title={Sensitivity-Guided Adaptive HDMR for Hyperparameter Optimization},
  author={Erdem, Y. and APP2SCALE Team},
  journal={IEEE Conference on ...},
  year={2026}
}
```

---
**Developed by APP2SCALE Team** | [Konya Technical University](https://www.ktun.edu.tr)
