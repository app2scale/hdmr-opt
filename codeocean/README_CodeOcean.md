# HDMR-opt: High Dimensional Model Representation for HPO

This Compute Capsule contains the implementation and benchmark scripts for **HDMR-opt**, a sensitivity-guided optimization framework based on High Dimensional Model Representation (HDMR).

## Directory Structure

- `/code`: Contains the core optimization library (`src/`) and the main benchmark script (`main.py`).
- `/data`: (Read-Only) Standard location for input datasets. If using OpenML datasets, the code will download them automatically; however, local datasets can be linked here.
- `/results`: (Writable) All logs, CSV result files, and plots are saved here.

## How to Run

Click the **Run** button. By default, it executes a "Smoke Test" for the **Tabular Benchmark**.

### Study Modes

You can switch between different studies using the `STUDY` environment variable:

1.  **`tabular`** (Default): Runs the TabArena benchmark (OpenML datasets).
    - Uses `main.py`.
    - Key params: `DATASETS`, `N_FOLDS`, `HDMR_SAMPLES`.
2.  **`forecasting`**: Runs the industrial forecasting study (Payten & Medianova).
    - Uses `scripts/forecast_hpo.py`.
    - Key params: `HORIZON`, `N_FOLDS`, `HDMR_SAMPLES`.
3.  **`benchmark`**: Runs mathematical test functions (Table XIV in the paper).
    - Uses `src/main.py`.
    - Example: Rastrigin, Ackley, Rosenbrock.

### Customizing the Run

You can control the execution via environment variables in the Code Ocean interface:
- `STUDY`: `tabular`, `forecasting`, or `benchmark`.
- `DATASETS`: Comma-separated OpenML DIDs (for tabular mode).
- `N_FOLDS`: Number of outer CV folds (default: 8).
- `HDMR_SAMPLES`: Samples per iteration (default: 200).
- `SEED`: Random seed for reproducibility (default: 42).



## Outputs

After execution, check the `/results` folder:
- `/results/logs/`: Detailed execution logs.
- `/results/tabarena/`: CSV files containing performance metrics (RMSE, AUC, etc.) and Sobol sensitivity indices for each hyperparameter.

## Dependencies

The environment is pre-configured with the necessary Python packages, including `lightgbm`, `xgboost`, `openml`, and `scipy`.
