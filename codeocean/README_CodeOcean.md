# HDMR-opt: High Dimensional Model Representation for HPO

This Compute Capsule contains the implementation and benchmark scripts for **HDMR-opt**, a sensitivity-guided optimization framework based on High Dimensional Model Representation (HDMR).

## Directory Structure

- `/code`: Contains the universal entry point (`main.py`), the core library (`src/`), and specialized benchmarks (`scripts/`).
- `/data`: (Read-Only) Contains industrial datasets (Payten, Medianova).
- `/results`: (Writable) All logs and CSV result files are persisted here.

## How to Run

Click the **Run** button. By default, the `run.sh` entry point executes the **Universal Dispatcher** (`main.py`) in the selected mode.

### Study Modes

You can switch between research studies using the `STUDY` environment variable:

1.  **`tabular`** (Default): Executes the TabArena benchmark (OpenML datasets).
    - Runs: `python main.py --study tabular`
2.  **`forecasting`**: Executes the industrial time-series forecasting study.
    - Runs: `python main.py --study forecasting`
3.  **`benchmark`**: Executes mathematical optimization test functions (e.g., Rastrigin, Ackley).
    - Runs: `python main.py --study benchmark`


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
