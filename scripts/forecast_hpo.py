#!/usr/bin/env python3
"""
forecast_hpo.py — Forecasting Hyperparameter Optimization Pipeline
====================================================================
Compares HDMR-200, A-HDMR-200, A-HDMR-600, Random Search-200, Optuna-200
on two time-series datasets (Payten, Medianova) using four models:
XGBoost, LightGBM, N-BEATS, LSTM.

Usage:
    python scripts/forecast_hpo.py --dataset payten
    python scripts/forecast_hpo.py --dataset both --models xgb,lgb
    python scripts/forecast_hpo.py --dataset payten --methods hdmr,rs

Environment variables:
    HORIZON        forecast horizon in days          [30]
    N_FOLDS        walk-forward folds               [3]
    HDMR_SAMPLES   HDMR evaluations (per iteration) [200]
    OPTUNA_TRIALS  Optuna trials per fold            [200]
    RS_SAMPLES     Random Search samples             [200]
    SEED           global random seed               [42]
    LOG_LEVEL      logging verbosity                [INFO]
    USE_GPU        auto | true | false              [auto]

Nohup examples:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    nohup python scripts/forecast_hpo.py --dataset payten \\
    > logs/forecast_payten_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    # Smoke test
    HORIZON=7 N_FOLDS=2 HDMR_SAMPLES=20 OPTUNA_TRIALS=20 RS_SAMPLES=20 \\
    python scripts/forecast_hpo.py --dataset payten --models xgb --methods hdmr
"""

# =============================================================================
# 0.  Imports & global configuration
# =============================================================================
import os
import sys
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Optional: statsmodels for ARIMA ──────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=ConvergenceWarning)
    HAS_STATSMODELS = True
except ImportError:
    ARIMAModel = None       # type: ignore[assignment,misc]
    HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Paths  (script lives in scripts/ → parent.parent is project root)
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent.parent
DATA_DIR    = _HERE / "src" / "data"
RESULTS_DIR = _HERE / "results" / "forecasting"
LOG_DIR     = _HERE / "logs"

# ---------------------------------------------------------------------------
# Environment-variable configuration
# ---------------------------------------------------------------------------
RUN_ID        = datetime.now().strftime("%Y%m%d_%H%M%S")
HORIZON       = int(os.environ.get("HORIZON",       "30"))
N_FOLDS       = int(os.environ.get("N_FOLDS",       "3"))
HDMR_SAMPLES  = int(os.environ.get("HDMR_SAMPLES",  "200"))
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "200"))
RS_SAMPLES    = int(os.environ.get("RS_SAMPLES",    "200"))
SEED          = int(os.environ.get("SEED",          "42"))
LOG_LEVEL     = os.environ.get("LOG_LEVEL",         "INFO")
USE_GPU       = os.environ.get("USE_GPU",           "auto")

# HDMR surrogate settings (fixed)
_HDMR_DEGREE  = 7
_HDMR_BASIS   = "Legendre"
_HDMR_K       = 100
_HDMR_EPSILON = 1e-2
_HDMR_CLIP    = 0.9

LOOKBACK = HORIZON * 2  # lookback window for sequence models

# ---------------------------------------------------------------------------
# HDMR import from src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_HERE))
from src.main import HDMRConfig, HDMROptimizer  # noqa: E402


# =============================================================================
# 1.  Logging
# =============================================================================
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"forecast_hpo_{RUN_ID}.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("hdmr.forecast")


# =============================================================================
# 2.  GPU detection
# =============================================================================
def get_device(logger: logging.Logger) -> torch.device:
    if USE_GPU == "false":
        logger.info("GPU disabled via USE_GPU=false")
        return torch.device("cpu")
    if USE_GPU == "true" or (USE_GPU == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        return device
    logger.info("Using CPU (no CUDA available or USE_GPU=%s)", USE_GPU)
    return torch.device("cpu")


# =============================================================================
# 3.  Dataset configuration
# =============================================================================
DATASET_CONFIGS: Dict[str, Dict] = {
    "payten": {
        "data":         DATA_DIR / "pt_transactions.csv",
        "special_days": DATA_DIR / "pt_special_days.csv",
        "target":       "transactions",
        "date_col":     "date",
        "name":         "Payten (Transactions)",
    },
    "medianova": {
        "data":         DATA_DIR / "mn_traffic_data.csv",
        "special_days": DATA_DIR / "mn_special_days.csv",
        "target":       None,   # auto-detect non-date column
        "date_col":     None,   # auto-detect
        "name":         "Medianova (CDN Traffic)",
    },
}


def load_dataset(
    name: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, str, Optional[pd.DataFrame]]:
    """Return (df_with_datetime_index, target_col, special_days_df_or_None)."""
    cfg = DATASET_CONFIGS[name]
    df  = pd.read_csv(cfg["data"])

    # Detect date column
    date_col = cfg["date_col"]
    if date_col is None:
        candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        date_col   = candidates[0] if candidates else df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)

    # Detect target column
    target_col = cfg["target"]
    if target_col is None:
        target_col = df.columns[0]

    # Load special days
    special_df: Optional[pd.DataFrame] = None
    sp_path = cfg["special_days"]
    if sp_path.exists():
        sp     = pd.read_csv(sp_path)
        dc     = [c for c in sp.columns if "date" in c.lower()][0]
        sp[dc] = pd.to_datetime(sp[dc])
        special_df = sp[[dc]].rename(columns={dc: "date"})

    logger.info(
        "Loaded '%s': %d rows, target='%s', special_days=%s",
        cfg["name"], len(df), target_col,
        f"{len(special_df)} entries" if special_df is not None else "none",
    )
    return df, target_col, special_df


# =============================================================================
# 4.  Feature engineering
# =============================================================================
def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    special_days_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build tabular (X, y) with no look-ahead leakage."""
    df = df.copy()
    t  = df[target_col].copy()

    # Lag features (shift ensures no leakage)
    for lag in (1, 7, 14, 30):
        df[f"lag_{lag}"] = t.shift(lag)

    # Rolling statistics on past values only
    df["rolling_mean_7"]  = t.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = t.shift(1).rolling(14).mean()
    df["rolling_std_7"]   = t.shift(1).rolling(7).std()

    # Calendar features
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df["day_of_week"]  = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"]        = idx.month
    df["week_of_year"] = idx.isocalendar().week.astype(int).values
    df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)
    df["quarter"]      = idx.quarter

    # Special-day features
    if special_days_df is not None:
        sd_set   = set(pd.to_datetime(special_days_df["date"]).dt.date)
        sd_sorted = sorted(sd_set)
        dates_arr = np.array([d.date() if hasattr(d, "date") else d for d in idx])

        df["is_special_day"] = [1 if d in sd_set else 0 for d in dates_arr]

        def _to_next(d):
            future = [s for s in sd_sorted if s >= d]
            return (future[0] - d).days if future else 30

        def _since_last(d):
            past = [s for s in sd_sorted if s <= d]
            return (d - past[-1]).days if past else 30

        df["days_to_next_special"]    = [min(_to_next(d),   30) for d in dates_arr]
        df["days_since_last_special"] = [min(_since_last(d), 30) for d in dates_arr]
    else:
        df["is_special_day"]          = 0
        df["days_to_next_special"]    = 30
        df["days_since_last_special"] = 30

    df.dropna(inplace=True)
    y = df.pop(target_col)
    X = df.copy()
    return X, y


# =============================================================================
# 5.  Walk-forward CV splits  (expanding window)
# =============================================================================
def walk_forward_splits(
    n_samples: int,
    horizon: int,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward splits.

    Example (n_samples=2100, horizon=30, n_folds=3):
        Fold 1: train=[0:1800], test=[1800:1830]
        Fold 2: train=[0:1900], test=[1900:1930]
        Fold 3: train=[0:2000], test=[2000:2030]
    """
    min_train = int(n_samples * 0.5)
    max_end   = n_samples
    # Evenly space fold ends between (min_train + horizon) and max_end
    ends = np.linspace(min_train + horizon, max_end, n_folds + 1, dtype=int)[1:]
    splits = []
    for end in ends:
        ts = end - horizon
        if ts < min_train:
            continue
        splits.append((np.arange(ts), np.arange(ts, end)))
    return splits[-n_folds:]


# =============================================================================
# 6.  Metrics
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mask   = np.abs(y_true) > 1e-8
    mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae    = float(mean_absolute_error(y_true, y_pred))
    return {"mape": round(mape, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}


# =============================================================================
# 7.  Search space definitions
# =============================================================================
XGB_SPACE: Dict[str, Tuple] = {
    "learning_rate":     ("log_float", 0.005, 0.1),
    "max_depth":         ("log_int",   4,     10),
    "min_child_weight":  ("log_float", 0.001, 5.0),
    "subsample":         ("float",     0.6,   1.0),
    "colsample_bylevel": ("float",     0.6,   1.0),
    "colsample_bynode":  ("float",     0.6,   1.0),
    "reg_alpha":         ("float",     1e-4,  5.0),
    "reg_lambda":        ("float",     1e-4,  5.0),
    "grow_policy":       ("choice",    ["depthwise", "lossguide"]),
    "max_leaves":        ("log_int",   8,     1024),
}

LGB_SPACE: Dict[str, Tuple] = {
    "learning_rate":      ("log_float", 0.005,  0.1),
    "feature_fraction":   ("float",     0.4,    1.0),
    "bagging_fraction":   ("float",     0.7,    1.0),
    "bagging_freq":       ("fixed",     1),
    "num_leaves":         ("log_int",   2,      200),
    "min_data_in_leaf":   ("log_int",   1,      64),
    "extra_trees":        ("choice",    [False, True]),
    "min_data_per_group": ("log_int",   2,      100),
    "cat_l2":             ("log_float", 0.005,  2.0),
    "cat_smooth":         ("log_float", 0.001,  100.0),
    "max_cat_to_onehot":  ("log_int",   8,      100),
    "lambda_l1":          ("float",     1e-4,   1.0),
    "lambda_l2":          ("float",     1e-4,   2.0),
}

NBEATS_SPACE: Dict[str, Tuple] = {
    "hidden_size":   ("log_int",   32,   512),
    "n_blocks":      ("int",       2,    6),
    "n_layers":      ("int",       2,    5),
    "theta_size":    ("log_int",   8,    128),
    "learning_rate": ("log_float", 1e-4, 1e-2),
    "batch_size":    ("log_int",   16,   128),
}

LSTM_SPACE: Dict[str, Tuple] = {
    "hidden_size":   ("log_int",   32,   256),
    "n_layers":      ("int",       1,    3),
    "dropout":       ("float",     0.0,  0.5),
    "learning_rate": ("log_float", 1e-4, 1e-2),
    "batch_size":    ("log_int",   16,   128),
}

ARIMA_SPACE: Dict[str, Tuple] = {
    "p": ("int", 0, 4),   # AR order
    "d": ("int", 0, 2),   # differencing
    "q": ("int", 0, 4),   # MA order
    "P": ("int", 0, 2),   # seasonal AR
    "Q": ("int", 0, 2),   # seasonal MA
}
ARIMA_S = 7   # fixed seasonal period (weekly)
ARIMA_D = 1   # fixed seasonal differencing order

GRU_SPACE: Dict[str, Tuple] = {
    "hidden_size":   ("log_int",   32,   256),
    "n_layers":      ("int",       1,    3),
    "dropout":       ("float",     0.0,  0.5),
    "learning_rate": ("log_float", 1e-4, 1e-2),
    "batch_size":    ("log_int",   16,   128),
}

SPACES: Dict[str, Dict[str, Tuple]] = {
    "XGBoost":  XGB_SPACE,
    "LightGBM": LGB_SPACE,
    "N-BEATS":  NBEATS_SPACE,
    "LSTM":     LSTM_SPACE,
    "ARIMA":    ARIMA_SPACE,
    "GRU":      GRU_SPACE,
}


# =============================================================================
# 8.  Search-space decoder & Optuna helper
# =============================================================================
def decode_params(space_def: Dict[str, Tuple], x_unit: np.ndarray) -> Dict[str, Any]:
    """Map x_unit ∈ [0,1]^d  →  actual hyperparameter dict."""
    params: Dict[str, Any] = {}
    idx = 0
    for name, spec in space_def.items():
        if spec[0] == "fixed":
            params[name] = spec[1]
            continue
        t = float(np.clip(x_unit[idx], 0.0, 1.0))
        if spec[0] == "log_float":
            lo, hi = spec[1], spec[2]
            params[name] = float(np.exp(np.log(lo) + t * np.log(hi / lo)))
        elif spec[0] == "float":
            lo, hi = spec[1], spec[2]
            params[name] = float(lo + t * (hi - lo))
        elif spec[0] == "log_int":
            lo, hi = spec[1], spec[2]
            params[name] = max(lo, int(round(np.exp(np.log(lo) + t * np.log(hi / lo)))))
        elif spec[0] == "int":
            lo, hi = spec[1], spec[2]
            params[name] = int(round(lo + t * (hi - lo)))
        elif spec[0] == "choice":
            choices = spec[1]
            params[name] = choices[int(t * len(choices)) % len(choices)]
        idx += 1
    return params


def suggest_optuna_params(trial: optuna.Trial, space_def: Dict[str, Tuple]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space_def.items():
        if spec[0] == "fixed":
            params[name] = spec[1]
        elif spec[0] in ("log_float", "float"):
            params[name] = trial.suggest_float(
                name, spec[1], spec[2], log=(spec[0] == "log_float"))
        elif spec[0] in ("log_int", "int"):
            params[name] = trial.suggest_int(
                name, spec[1], spec[2], log=(spec[0] == "log_int"))
        elif spec[0] == "choice":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def n_free_dims(space_def: Dict[str, Tuple]) -> int:
    """Number of non-fixed dimensions in the search space."""
    return sum(1 for spec in space_def.values() if spec[0] != "fixed")


# =============================================================================
# 9.  PyTorch model definitions
# =============================================================================
class NBeatsBlock(nn.Module):
    """Generic N-BEATS block: FC stack → theta → backcast + forecast projections."""

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        n_layers: int,
        forecast_size: int,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.fc_stack    = nn.Sequential(*layers)
        self.theta_b     = nn.Linear(hidden_size, theta_size)
        self.theta_f     = nn.Linear(hidden_size, theta_size)
        self.backcast    = nn.Linear(theta_size, input_size)
        self.forecast    = nn.Linear(theta_size, forecast_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(x)
        return self.backcast(self.theta_b(h)), self.forecast(self.theta_f(h))


class NBeatsModel(nn.Module):
    """Stack of N-BEATS blocks with residual connections."""

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        n_blocks: int = 3,
        hidden_size: int = 256,
        n_layers: int = 4,
        theta_size: int = 32,
    ) -> None:
        super().__init__()
        self.forecast_size = forecast_size
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, hidden_size, n_layers, forecast_size)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        total    = torch.zeros(x.shape[0], self.forecast_size, device=x.device)
        for block in self.blocks:
            b, f   = block(residual)
            residual = residual - b
            total    = total + f
        return total


class LSTMModel(nn.Module):
    """LSTM with a single linear output head for multi-step forecasting."""

    def __init__(
        self,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
        forecast_size: int = 30,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=(dropout if n_layers > 1 else 0.0),
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, forecast_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (batch, forecast_size)


class GRUModel(nn.Module):
    """Sequence-to-one GRU for multivariate multi-step forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        forecast_size: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, forecast_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])   # last timestep → (batch, forecast_size)


# =============================================================================
# 10.  Sequence helpers
# =============================================================================
def make_sequences(series: np.ndarray, lookback: int, horizon: int):
    """Create (X, y) pairs from a 1-D float32 series."""
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback: i + lookback + horizon])
    if not X:
        return np.zeros((0, lookback), dtype=np.float32), np.zeros((0, horizon), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_sequences_mv(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multivariate sequences for GRU.

    Returns:
        X_seq: (N, lookback, n_features)  float32
        y_seq: (N, horizon)               float32
    """
    n      = len(X_arr)
    n_feat = X_arr.shape[1] if X_arr.ndim == 2 else 1
    Xs, ys = [], []
    for i in range(n - lookback - horizon + 1):
        Xs.append(X_arr[i: i + lookback])
        ys.append(y_arr[i + lookback: i + lookback + horizon])
    if not Xs:
        return (
            np.zeros((0, lookback, n_feat), dtype=np.float32),
            np.zeros((0, horizon),          dtype=np.float32),
        )
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# =============================================================================
# 11.  Training helpers for PyTorch models
# =============================================================================
def _train_torch(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    lr: float,
    batch_size: int,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
) -> float:
    """Train model; return best validation RMSE."""
    if len(X_tr) < 2 or len(X_va) < 1:
        return 1e6

    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(device)

    # Reshape for LSTM (needs 3-D input)
    if X_tr.ndim == 2 and isinstance(model, LSTMModel):
        X_tr = X_tr[:, :, None]
        X_va = X_va[:, :, None]

    Xt = _to_tensor(X_tr); yt = _to_tensor(y_tr)
    Xv = _to_tensor(X_va); yv = _to_tensor(y_va)

    bs      = max(1, min(int(batch_size), len(Xt)))
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    stagnant = 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xv), yv).item())
        if vl < best_val - 1e-7:
            best_val = vl
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= patience:
            break
    return float(np.sqrt(max(best_val, 0.0)))


def _prepare_seq_splits(y_sc: np.ndarray):
    """Split normalised series into train/val sequence arrays."""
    val_sz  = max(HORIZON, int(len(y_sc) * 0.15))
    train_s = y_sc[:-val_sz]
    val_s   = y_sc[-(val_sz + LOOKBACK):]   # include lookback context
    X_s, y_s = make_sequences(train_s, LOOKBACK, HORIZON)
    X_v, y_v = make_sequences(val_s,   LOOKBACK, HORIZON)
    return X_s, y_s, X_v, y_v


def train_torch_model(
    model: nn.Module,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> Tuple[Optional[nn.Module], float]:
    """Train a PyTorch model with early stopping.

    Returns (trained_model, best_val_rmse).
    Returns (None, 1e6) on any failure.
    """
    try:
        best_rmse = _train_torch(
            model, X_seq, y_seq, X_val_seq, y_val_seq,
            lr=lr, batch_size=batch_size, device=device,
            max_epochs=max_epochs, patience=patience,
        )
        return model, best_rmse
    except Exception:
        return None, 1e6


# =============================================================================
# 12.  Objective factories
# =============================================================================

# ── XGBoost ──────────────────────────────────────────────────────────────────
def build_xgb_objective(
    space, X_tr, y_tr, X_te, y_te, fold_idx, device, logger, eval_log
):
    val_sz = max(HORIZON, int(len(X_tr) * 0.20))
    sx = StandardScaler(); sy = StandardScaler()
    X_tr_sc = sx.fit_transform(X_tr.iloc[:-val_sz])
    X_va_sc = sx.transform(X_tr.iloc[-val_sz:])
    y_tr_sc = sy.fit_transform(y_tr.iloc[:-val_sz].values.reshape(-1, 1)).ravel()
    y_va_sc = sy.transform(y_tr.iloc[-val_sz:].values.reshape(-1, 1)).ravel()

    use_cuda = device.type == "cuda"
    fixed = dict(
        n_estimators=10000,
        early_stopping_rounds=50,
        eval_metric="rmse",
        verbosity=0,
        random_state=SEED + fold_idx,
    )
    if use_cuda:
        fixed.update({"device": "cuda", "tree_method": "hist"})

    def _eval(params: Dict[str, Any]) -> float:
        try:
            m = xgb.XGBRegressor(**{**fixed, **params})
            m.fit(X_tr_sc, y_tr_sc,
                  eval_set=[(X_va_sc, y_va_sc)],
                  verbose=False)
            return float(np.sqrt(mean_squared_error(y_va_sc, m.predict(X_va_sc))))
        except Exception as exc:
            logger.warning("XGB eval failed [fold=%d]: %s", fold_idx, exc)
            return 1e6

    def fun_batch(X_unit: np.ndarray) -> np.ndarray:
        out = []
        for x in X_unit:
            v = _eval(decode_params(space, x))
            eval_log.append(v)
            if len(eval_log) % 10 == 0:
                logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
            out.append(v)
        return np.array(out)

    def optuna_obj(trial: optuna.Trial) -> float:
        v = _eval(suggest_optuna_params(trial, space))
        eval_log.append(v)
        if len(eval_log) % 10 == 0:
            logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
        return v

    def evaluate_test(best_params: Dict[str, Any]) -> Dict[str, float]:
        val_sz2 = max(HORIZON, int(len(X_tr) * 0.15))
        sx2 = StandardScaler(); sy2 = StandardScaler()
        X_full_tr = sx2.fit_transform(X_tr)
        X_full_te = sx2.transform(X_te)
        y_full_tr = sy2.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
        m = xgb.XGBRegressor(**{**fixed, **best_params})
        m.fit(
            X_full_tr[:-val_sz2], y_full_tr[:-val_sz2],
            eval_set=[(X_full_tr[-val_sz2:], y_full_tr[-val_sz2:])],
            verbose=False,
        )
        pred = sy2.inverse_transform(m.predict(X_full_te).reshape(-1, 1)).ravel()
        return compute_metrics(y_te.values, pred)

    return fun_batch, optuna_obj, evaluate_test


# ── LightGBM ─────────────────────────────────────────────────────────────────
def build_lgb_objective(
    space, X_tr, y_tr, X_te, y_te, fold_idx, device, logger, eval_log
):
    val_sz = max(HORIZON, int(len(X_tr) * 0.20))
    sx = StandardScaler(); sy = StandardScaler()
    X_tr_sc = sx.fit_transform(X_tr.iloc[:-val_sz])
    X_va_sc = sx.transform(X_tr.iloc[-val_sz:])
    y_tr_sc = sy.fit_transform(y_tr.iloc[:-val_sz].values.reshape(-1, 1)).ravel()
    y_va_sc = sy.transform(y_tr.iloc[-val_sz:].values.reshape(-1, 1)).ravel()

    fixed = dict(
        n_estimators=10000,
        verbosity=-1,
        num_threads=4,
        feature_pre_filter=False,
        seed=SEED + fold_idx,
    )
    _es = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

    def _eval(params: Dict[str, Any]) -> float:
        try:
            m = lgb.LGBMRegressor(**{**fixed, **params})
            m.fit(X_tr_sc, y_tr_sc,
                  eval_set=[(X_va_sc, y_va_sc)],
                  callbacks=_es)
            return float(np.sqrt(mean_squared_error(y_va_sc, m.predict(X_va_sc))))
        except Exception as exc:
            logger.warning("LGB eval failed [fold=%d]: %s", fold_idx, exc)
            return 1e6

    def fun_batch(X_unit: np.ndarray) -> np.ndarray:
        out = []
        for x in X_unit:
            v = _eval(decode_params(space, x))
            eval_log.append(v)
            if len(eval_log) % 10 == 0:
                logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
            out.append(v)
        return np.array(out)

    def optuna_obj(trial: optuna.Trial) -> float:
        v = _eval(suggest_optuna_params(trial, space))
        eval_log.append(v)
        if len(eval_log) % 10 == 0:
            logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
        return v

    def evaluate_test(best_params: Dict[str, Any]) -> Dict[str, float]:
        val_sz2 = max(HORIZON, int(len(X_tr) * 0.15))
        sx2 = StandardScaler(); sy2 = StandardScaler()
        X_full_tr = sx2.fit_transform(X_tr)
        X_full_te = sx2.transform(X_te)
        y_full_tr = sy2.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
        m = lgb.LGBMRegressor(**{**fixed, **best_params})
        m.fit(
            X_full_tr[:-val_sz2], y_full_tr[:-val_sz2],
            eval_set=[(X_full_tr[-val_sz2:], y_full_tr[-val_sz2:])],
            callbacks=_es,
        )
        pred = sy2.inverse_transform(m.predict(X_full_te).reshape(-1, 1)).ravel()
        return compute_metrics(y_te.values, pred)

    return fun_batch, optuna_obj, evaluate_test


# ── N-BEATS ───────────────────────────────────────────────────────────────────
def build_nbeats_objective(
    space, X_tr, y_tr, X_te, y_te, fold_idx, device, logger, eval_log
):
    sy = StandardScaler()
    y_tr_sc = sy.fit_transform(y_tr.values.reshape(-1, 1)).ravel().astype(np.float32)
    X_s, y_s, X_v, y_v = _prepare_seq_splits(y_tr_sc)

    def _eval(params: Dict[str, Any]) -> float:
        try:
            model = NBeatsModel(
                input_size=LOOKBACK,
                forecast_size=HORIZON,
                n_blocks=int(params["n_blocks"]),
                hidden_size=int(params["hidden_size"]),
                n_layers=int(params["n_layers"]),
                theta_size=int(params["theta_size"]),
            ).to(device)
            return _train_torch(
                model, X_s, y_s, X_v, y_v,
                lr=float(params["learning_rate"]),
                batch_size=int(params["batch_size"]),
                device=device,
            )
        except Exception as exc:
            logger.warning("N-BEATS eval failed [fold=%d]: %s", fold_idx, exc)
            return 1e6
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def fun_batch(X_unit: np.ndarray) -> np.ndarray:
        out = []
        for x in X_unit:
            v = _eval(decode_params(space, x))
            eval_log.append(v)
            if len(eval_log) % 10 == 0:
                logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
            out.append(v)
        return np.array(out)

    def optuna_obj(trial: optuna.Trial) -> float:
        v = _eval(suggest_optuna_params(trial, space))
        eval_log.append(v)
        if len(eval_log) % 10 == 0:
            logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
        return v

    def evaluate_test(best_params: Dict[str, Any]) -> Dict[str, float]:
        try:
            sy2 = StandardScaler()
            y_full_sc = sy2.fit_transform(y_tr.values.reshape(-1, 1)).ravel().astype(np.float32)
            Xs2, ys2, Xv2, yv2 = _prepare_seq_splits(y_full_sc)
            model = NBeatsModel(
                input_size=LOOKBACK,
                forecast_size=HORIZON,
                n_blocks=int(best_params["n_blocks"]),
                hidden_size=int(best_params["hidden_size"]),
                n_layers=int(best_params["n_layers"]),
                theta_size=int(best_params["theta_size"]),
            ).to(device)
            _train_torch(model, Xs2, ys2, Xv2, yv2,
                         lr=float(best_params["learning_rate"]),
                         batch_size=int(best_params["batch_size"]),
                         device=device, max_epochs=200, patience=15)
            model.eval()
            inp = torch.from_numpy(y_full_sc[-LOOKBACK:][None]).to(device)
            with torch.no_grad():
                pred_sc = model(inp).cpu().numpy().ravel()
            pred = sy2.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
            n = min(len(pred), len(y_te))
            return compute_metrics(y_te.values[:n], pred[:n])
        except Exception as exc:
            logger.warning("N-BEATS test eval failed: %s", exc)
            return {"mape": float("nan"), "rmse": float("nan"), "mae": float("nan")}
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return fun_batch, optuna_obj, evaluate_test


# ── LSTM ──────────────────────────────────────────────────────────────────────
def build_lstm_objective(
    space, X_tr, y_tr, X_te, y_te, fold_idx, device, logger, eval_log
):
    sy = StandardScaler()
    y_tr_sc = sy.fit_transform(y_tr.values.reshape(-1, 1)).ravel().astype(np.float32)
    X_s, y_s, X_v, y_v = _prepare_seq_splits(y_tr_sc)

    def _eval(params: Dict[str, Any]) -> float:
        try:
            model = LSTMModel(
                hidden_size=int(params["hidden_size"]),
                n_layers=int(params["n_layers"]),
                dropout=float(params["dropout"]),
                forecast_size=HORIZON,
            ).to(device)
            return _train_torch(
                model, X_s, y_s, X_v, y_v,
                lr=float(params["learning_rate"]),
                batch_size=int(params["batch_size"]),
                device=device,
            )
        except Exception as exc:
            logger.warning("LSTM eval failed [fold=%d]: %s", fold_idx, exc)
            return 1e6
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def fun_batch(X_unit: np.ndarray) -> np.ndarray:
        out = []
        for x in X_unit:
            v = _eval(decode_params(space, x))
            eval_log.append(v)
            if len(eval_log) % 10 == 0:
                logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
            out.append(v)
        return np.array(out)

    def optuna_obj(trial: optuna.Trial) -> float:
        v = _eval(suggest_optuna_params(trial, space))
        eval_log.append(v)
        if len(eval_log) % 10 == 0:
            logger.debug("  fold=%d eval=%d best=%.4f", fold_idx, len(eval_log), min(eval_log))
        return v

    def evaluate_test(best_params: Dict[str, Any]) -> Dict[str, float]:
        try:
            sy2 = StandardScaler()
            y_full_sc = sy2.fit_transform(y_tr.values.reshape(-1, 1)).ravel().astype(np.float32)
            Xs2, ys2, Xv2, yv2 = _prepare_seq_splits(y_full_sc)
            model = LSTMModel(
                hidden_size=int(best_params["hidden_size"]),
                n_layers=int(best_params["n_layers"]),
                dropout=float(best_params["dropout"]),
                forecast_size=HORIZON,
            ).to(device)
            _train_torch(model, Xs2, ys2, Xv2, yv2,
                         lr=float(best_params["learning_rate"]),
                         batch_size=int(best_params["batch_size"]),
                         device=device, max_epochs=200, patience=15)
            model.eval()
            inp = torch.from_numpy(y_full_sc[-LOOKBACK:, None][None]).to(device)
            with torch.no_grad():
                pred_sc = model(inp).cpu().numpy().ravel()
            pred = sy2.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
            n = min(len(pred), len(y_te))
            return compute_metrics(y_te.values[:n], pred[:n])
        except Exception as exc:
            logger.warning("LSTM test eval failed: %s", exc)
            return {"mape": float("nan"), "rmse": float("nan"), "mae": float("nan")}
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return fun_batch, optuna_obj, evaluate_test


# ── ARIMA ─────────────────────────────────────────────────────────────────────
def build_arima_objective(
    space, X_tr, y_tr, X_te, y_te, fold_idx, device, logger, eval_log
):
    """ARIMA uses raw unscaled target series; ignores X features and device."""
    y_raw = y_tr.values.astype(float)
    # Internal val: last HORIZON steps of the training window (no shuffling)
    val_sz       = HORIZON
    train_series = y_raw[:-val_sz]
    val_series   = y_raw[-val_sz:]

    def _eval(params: Dict[str, Any]) -> float:
        if not HAS_STATSMODELS:
            return 1e6
        try:
            p = int(params.get("p", 1)); d = int(params.get("d", 1))
            q = int(params.get("q", 0)); P = int(params.get("P", 1))
            Q = int(params.get("Q", 1))
            import warnings as _w
            mdl = ARIMAModel(
                train_series,
                order=(p, d, q),
                seasonal_order=(P, ARIMA_D, Q, ARIMA_S),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = mdl.fit(method_kwargs={"warn_convergence": False})
            fc   = res.forecast(steps=val_sz)
            rmse = float(np.sqrt(np.mean((fc - val_series) ** 2)))
            return rmse if np.isfinite(rmse) else 1e6
        except Exception as exc:
            logger.debug("ARIMA eval failed [fold=%d]: %s", fold_idx, exc)
            return 1e6

    def fun_batch(X_unit: np.ndarray) -> np.ndarray:
        out = []
        for x in X_unit:
            v = _eval(decode_params(space, x))
            eval_log.append(v)
            if len(eval_log) % 10 == 0:
                logger.debug("  fold=%d eval=%d best=%.4f",
                             fold_idx, len(eval_log), min(eval_log))
            out.append(v)
        return np.array(out)

    def optuna_obj(trial: optuna.Trial) -> float:
        v = _eval(suggest_optuna_params(trial, space))
        eval_log.append(v)
        if len(eval_log) % 10 == 0:
            logger.debug("  fold=%d eval=%d best=%.4f",
                         fold_idx, len(eval_log), min(eval_log))
        return v

    def evaluate_test(best_params: Dict[str, Any]) -> Dict[str, float]:
        if not HAS_STATSMODELS:
            return {"mape": float("nan"), "rmse": float("nan"), "mae": float("nan")}
        try:
            p = int(best_params.get("p", 1)); d = int(best_params.get("d", 1))
            q = int(best_params.get("q", 0)); P = int(best_params.get("P", 1))
            Q = int(best_params.get("Q", 1))
            import warnings as _w
            mdl = ARIMAModel(
                y_raw,   # fit on the full training window
                order=(p, d, q),
                seasonal_order=(P, ARIMA_D, Q, ARIMA_S),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = mdl.fit(method_kwargs={"warn_convergence": False})
            fc = res.forecast(steps=HORIZON)
            n  = min(len(fc), len(y_te))
            return compute_metrics(y_te.values[:n], fc[:n])
        except Exception as exc:
            logger.warning("ARIMA test eval failed: %s", exc)
            return {"mape": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    return fun_batch, optuna_obj, evaluate_test


# ── GRU ───────────────────────────────────────────────────────────────────────
def build_gru_objective(
    space, X_tr, y_tr, X_te, y_te, fold_idx, device, logger, eval_log
):
    """GRU uses multivariate engineered features; GPU-aware."""
    val_sz  = max(HORIZON, int(len(X_tr) * 0.20))
    n_feats = X_tr.shape[1]

    X_arr = X_tr.values.astype(np.float32)
    y_arr = y_tr.values.astype(np.float32)

    sx = StandardScaler(); sy = StandardScaler()
    X_tr_sc = sx.fit_transform(X_arr[:-val_sz])
    y_tr_sc = sy.fit_transform(y_arr[:-val_sz].reshape(-1, 1)).ravel()

    # Include LOOKBACK context in validation window (no leakage)
    X_va_sc = sx.transform(X_arr[-(val_sz + LOOKBACK):])
    y_va_sc = sy.transform(y_arr[-(val_sz + LOOKBACK):].reshape(-1, 1)).ravel()

    X_s, y_s = make_sequences_mv(X_tr_sc, y_tr_sc, LOOKBACK, HORIZON)
    X_v, y_v = make_sequences_mv(X_va_sc, y_va_sc, LOOKBACK, HORIZON)

    def _eval(params: Dict[str, Any]) -> float:
        try:
            model = GRUModel(
                input_size=n_feats,
                hidden_size=int(params["hidden_size"]),
                n_layers=int(params["n_layers"]),
                dropout=float(params["dropout"]),
                forecast_size=HORIZON,
            ).to(device)
            return _train_torch(
                model, X_s, y_s, X_v, y_v,
                lr=float(params["learning_rate"]),
                batch_size=int(params["batch_size"]),
                device=device,
            )
        except Exception as exc:
            logger.warning("GRU eval failed [fold=%d]: %s", fold_idx, exc)
            return 1e6
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def fun_batch(X_unit: np.ndarray) -> np.ndarray:
        out = []
        for x in X_unit:
            v = _eval(decode_params(space, x))
            eval_log.append(v)
            if len(eval_log) % 10 == 0:
                logger.debug("  fold=%d eval=%d best=%.4f",
                             fold_idx, len(eval_log), min(eval_log))
            out.append(v)
        return np.array(out)

    def optuna_obj(trial: optuna.Trial) -> float:
        v = _eval(suggest_optuna_params(trial, space))
        eval_log.append(v)
        if len(eval_log) % 10 == 0:
            logger.debug("  fold=%d eval=%d best=%.4f",
                         fold_idx, len(eval_log), min(eval_log))
        return v

    def evaluate_test(best_params: Dict[str, Any]) -> Dict[str, float]:
        try:
            val_sz2 = max(HORIZON, int(len(X_arr) * 0.15))
            sx2 = StandardScaler(); sy2 = StandardScaler()
            X_full_sc = sx2.fit_transform(X_arr)
            y_full_sc = sy2.fit_transform(y_arr.reshape(-1, 1)).ravel()

            X_va2 = sx2.transform(X_arr[-(val_sz2 + LOOKBACK):])
            y_va2 = sy2.transform(y_arr[-(val_sz2 + LOOKBACK):].reshape(-1, 1)).ravel()
            Xs2, ys2 = make_sequences_mv(
                X_full_sc[:-val_sz2], y_full_sc[:-val_sz2], LOOKBACK, HORIZON)
            Xv2, yv2 = make_sequences_mv(X_va2, y_va2, LOOKBACK, HORIZON)

            model = GRUModel(
                input_size=n_feats,
                hidden_size=int(best_params["hidden_size"]),
                n_layers=int(best_params["n_layers"]),
                dropout=float(best_params["dropout"]),
                forecast_size=HORIZON,
            ).to(device)
            _train_torch(model, Xs2, ys2, Xv2, yv2,
                         lr=float(best_params["learning_rate"]),
                         batch_size=int(best_params["batch_size"]),
                         device=device, max_epochs=200, patience=15)
            model.eval()
            # Context: last LOOKBACK rows of training features
            inp = torch.from_numpy(X_full_sc[-LOOKBACK:][None]).to(device)
            with torch.no_grad():
                pred_sc = model(inp).cpu().numpy().ravel()
            pred = sy2.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
            n = min(len(pred), len(y_te))
            return compute_metrics(y_te.values[:n], pred[:n])
        except Exception as exc:
            logger.warning("GRU test eval failed: %s", exc)
            return {"mape": float("nan"), "rmse": float("nan"), "mae": float("nan")}
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return fun_batch, optuna_obj, evaluate_test


OBJECTIVE_FACTORIES: Dict[str, Callable] = {
    "XGBoost":  build_xgb_objective,
    "LightGBM": build_lgb_objective,
    "N-BEATS":  build_nbeats_objective,
    "LSTM":     build_lstm_objective,
    "ARIMA":    build_arima_objective,
    "GRU":      build_gru_objective,
}


# =============================================================================
# 13.  HPO drivers
# =============================================================================
def _hdmr_samples(method: str) -> Tuple[int, bool, int]:
    """Return (N_per_iter, adaptive, maxiter) for a given method tag."""
    if method == "HDMR-200":
        return HDMR_SAMPLES, False, 1
    if method == "A-HDMR-200":
        return max(5, HDMR_SAMPLES // 2), True, 2
    # A-HDMR-600: 3 iterations × N each = 3× budget
    return HDMR_SAMPLES, True, 3


def _run_hdmr(
    method: str,
    fun_batch: Callable,
    n_dims: int,
    fold_idx: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, float, Any]:
    N, adaptive, maxiter = _hdmr_samples(method)
    cfg = HDMRConfig(
        n=n_dims, a=0.0, b=1.0,
        N=N, m=_HDMR_DEGREE, basis=_HDMR_BASIS,
        seed=SEED + fold_idx,
        adaptive=adaptive, maxiter=maxiter,
        k=_HDMR_K, epsilon=_HDMR_EPSILON, clip=_HDMR_CLIP,
        disp=False, enable_plots=False,
    )
    optimizer = HDMROptimizer(fun_batch=fun_batch, config=cfg)
    result    = optimizer.solve(np.full(n_dims, 0.5))
    return result.x, float(result.fun), optimizer


def _run_rs(
    fun_batch: Callable,
    n_dims: int,
    fold_idx: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, float]:
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=n_dims, scramble=True, seed=SEED + fold_idx)
    configs = sampler.random(RS_SAMPLES)
    vals    = fun_batch(configs)
    best_i  = int(np.argmin(vals))
    return configs[best_i], float(vals[best_i])


def _run_optuna(
    optuna_obj: Callable,
    n_dims: int,
    fold_idx: int,
    space: Dict[str, Tuple],
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], float]:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED + fold_idx),
    )
    study.optimize(optuna_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# =============================================================================
# 14.  Unified HPO runner — one (fold × model × method) cell
# =============================================================================
def run_hpo(
    method: str,
    model_name: str,
    space: Dict[str, Tuple],
    X_train, y_train,
    X_test,  y_test,
    fold_idx: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[Dict, float, Dict, float, int, Dict]:
    """
    Returns:
        best_params    — decoded hyperparameter dict
        best_val       — best validation objective value
        test_metrics   — {"mape", "rmse", "mae"} on test set
        duration_sec   — wall time
        n_evals        — total objective evaluations
        sensitivity    — {param: Sobol-index} (HDMR only, else {})
    """
    eval_log: List[float] = []
    n_dims   = n_free_dims(space)
    factory  = OBJECTIVE_FACTORIES[model_name]

    fun_batch, optuna_obj, evaluate_test = factory(
        space, X_train, y_train, X_test, y_test,
        fold_idx, device, logger, eval_log,
    )

    t0             = time.perf_counter()
    hdmr_optimizer = None
    best_params    : Dict[str, Any] = {}
    best_val       = 1e6

    try:
        if "HDMR" in method:
            best_x, best_val, hdmr_optimizer = _run_hdmr(
                method, fun_batch, n_dims, fold_idx, logger)
            best_params = decode_params(space, best_x)

        elif method == "RS-200":
            best_x, best_val = _run_rs(fun_batch, n_dims, fold_idx, logger)
            best_params = decode_params(space, best_x)

        elif method == "Optuna-200":
            best_params, best_val = _run_optuna(
                optuna_obj, n_dims, fold_idx, space, logger)

        else:
            raise ValueError(f"Unknown method: {method!r}")

        test_metrics = evaluate_test(best_params)

    except Exception as exc:
        logger.error(
            "run_hpo failed [%s / %s / %s / fold=%d]: %s",
            method, model_name, "train→test", fold_idx + 1, exc,
        )
        test_metrics = {"mape": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    duration = time.perf_counter() - t0
    n_evals  = len(eval_log)

    # Sensitivity indices from HDMR alpha coefficients
    sensitivity: Dict[str, float] = {}
    if hdmr_optimizer is not None and hdmr_optimizer.alpha is not None:
        V_i = np.sum(hdmr_optimizer.alpha ** 2, axis=0)
        tot = V_i.sum()
        if tot > 1e-12:
            free_names = [k for k, v in space.items() if v[0] != "fixed"]
            sensitivity = {n: float(V_i[i] / tot) for i, n in enumerate(free_names)}

    return best_params, best_val, test_metrics, duration, n_evals, sensitivity


# =============================================================================
# 15.  Results I/O
# =============================================================================
def save_results(all_records: List[Dict], logger: logging.Logger) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"forecast_results_{RUN_ID}.csv"
    rows = []
    for r in all_records:
        row: Dict[str, Any] = {
            "run_id":       RUN_ID,
            "dataset":      r["dataset"],
            "model":        r["model"],
            "method":       r["method"],
            "fold":         r["fold"],
            "mape":         r["mape"],
            "rmse":         r["rmse"],
            "mae":          r["mae"],
            "best_val":     r["best_val"],
            "duration_sec": r["duration_sec"],
            "n_evals":      r["n_evals"],
        }
        for k, v in r.get("best_params", {}).items():
            row[f"param_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.debug("Saved → %s  (%d rows)", path, len(rows))
    return path


# =============================================================================
# 16.  Summary & sensitivity reports
# =============================================================================
_ALL_METHODS = ["HDMR-200", "A-HDMR-200", "A-HDMR-600", "RS-200", "Optuna-200"]
_ALL_MODELS  = ["XGBoost", "LightGBM", "N-BEATS", "LSTM", "ARIMA", "GRU"]


def print_summary(all_records: List[Dict], logger: logging.Logger) -> None:
    datasets = sorted({r["dataset"] for r in all_records})
    for ds in datasets:
        recs = [r for r in all_records if r["dataset"] == ds]
        sep  = "=" * 74
        logger.info("\n%s", sep)
        logger.info("FORECASTING HPO RESULTS SUMMARY — %s",
                    DATASET_CONFIGS[ds]["name"])
        logger.info("%s", sep)
        col_w = 12
        header = f"{'Model':<12} {'Metric':<8}" + "".join(
            f"  {m[:col_w]:>{col_w}}" for m in _ALL_METHODS
        )
        logger.info(header)
        logger.info("-" * len(header))
        for mdl in _ALL_MODELS:
            for metric in ("mape", "rmse"):
                line = f"{mdl:<12} {metric:<8}"
                for mth in _ALL_METHODS:
                    vals = [
                        r[metric] for r in recs
                        if r["model"] == mdl and r["method"] == mth
                        and not np.isnan(r.get(metric, float("nan")))
                    ]
                    if vals:
                        line += f"  {np.mean(vals):>7.2f}±{np.std(vals):.2f}"
                    else:
                        line += f"  {'N/A':>{col_w}}"
                logger.info(line)
        # Best MAPE across all models × methods
        candidates = [
            (r["mape"], r["method"], r["model"]) for r in recs
            if not np.isnan(r.get("mape", float("nan")))
        ]
        if candidates:
            best_mape, best_mth, best_mdl = min(candidates)
            logger.info("Best MAPE: %s with %s = %.2f%%", best_mth, best_mdl, best_mape)
        logger.info("%s", sep)


def print_sensitivity_report(all_records: List[Dict], logger: logging.Logger) -> None:
    hdmr_methods = {"HDMR-200", "A-HDMR-200", "A-HDMR-600"}
    by_model: Dict[str, List[Dict[str, float]]] = {}
    for r in all_records:
        if r["method"] in hdmr_methods and r.get("sensitivity"):
            by_model.setdefault(r["model"], []).append(r["sensitivity"])

    if not by_model:
        return
    logger.info("\n%s", "=" * 60)
    logger.info("SENSITIVITY SUMMARY  (first-order Sobol indices, HDMR)")
    logger.info("%s", "=" * 60)
    for mdl, sens_list in by_model.items():
        agg: Dict[str, float] = {}
        n = len(sens_list)
        for s in sens_list:
            for k, v in s.items():
                agg[k] = agg.get(k, 0.0) + v / n
        top3  = sorted(agg.items(), key=lambda kv: -kv[1])[:3]
        parts = " > ".join(f"{name}({val:.2f})" for name, val in top3)
        logger.info("[%s] Top sensitivity: %s", mdl, parts)


# =============================================================================
# 17.  ETA tracker
# =============================================================================
class ETATracker:
    def __init__(self, total: int, window: int = 3) -> None:
        self.total = total
        self.done  = 0
        self._q: deque = deque(maxlen=window)
        self._t = time.perf_counter()

    def tick(self) -> str:
        self._q.append(time.perf_counter() - self._t)
        self._t = time.perf_counter()
        self.done += 1
        remaining = self.total - self.done
        if remaining == 0 or not self._q:
            return "done"
        avg = sum(self._q) / len(self._q)
        return str(timedelta(seconds=int(avg * remaining)))


# =============================================================================
# 18.  Main
# =============================================================================
def main(args) -> None:
    logger = setup_logging()
    device = get_device(logger)

    # ── Resolve datasets ──────────────────────────────────────────────────
    datasets_to_run: List[str] = (
        ["payten", "medianova"] if args.dataset == "both" else [args.dataset]
    )

    # ── Resolve models ────────────────────────────────────────────────────
    _tag2model = {
        "xgb":    "XGBoost",
        "lgb":    "LightGBM",
        "nbeats": "N-BEATS",
        "lstm":   "LSTM",
        "arima":  "ARIMA",
        "gru":    "GRU",
    }
    models_to_run: List[str] = (
        _ALL_MODELS if args.models == "all"
        else [_tag2model[t.strip()] for t in args.models.split(",")
              if t.strip() in _tag2model]
    )

    # ── ARIMA availability check ──────────────────────────────────────────
    if "ARIMA" in models_to_run and not HAS_STATSMODELS:
        logger.warning(
            "ARIMA requested but statsmodels is not installed — skipping ARIMA."
            "  Install with: pip install statsmodels"
        )
        models_to_run = [m for m in models_to_run if m != "ARIMA"]

    # ── Resolve methods ───────────────────────────────────────────────────
    _tag2method = {
        "hdmr": "HDMR-200", "ahdmr200": "A-HDMR-200",
        "ahdmr600": "A-HDMR-600", "rs": "RS-200", "optuna": "Optuna-200",
    }
    methods_to_run: List[str] = (
        _ALL_METHODS if args.methods == "all"
        else [_tag2method[t.strip()] for t in args.methods.split(",")
              if t.strip() in _tag2method]
    )

    logger.info("=" * 70)
    logger.info("FORECAST HPO  run_id=%s", RUN_ID)
    logger.info("  Datasets  : %s", datasets_to_run)
    logger.info("  Models    : %s", models_to_run)
    logger.info("  Methods   : %s", methods_to_run)
    logger.info("  HORIZON=%d  N_FOLDS=%d  HDMR_SAMPLES=%d  "
                "OPTUNA_TRIALS=%d  RS_SAMPLES=%d  SEED=%d",
                HORIZON, N_FOLDS, HDMR_SAMPLES, OPTUNA_TRIALS, RS_SAMPLES, SEED)
    logger.info("=" * 70)

    all_records: List[Dict] = []

    for ds_name in datasets_to_run:
        df, target_col, special_df = load_dataset(ds_name, logger)
        X, y   = engineer_features(df, target_col, special_df)
        splits = walk_forward_splits(len(X), HORIZON, N_FOLDS)
        ds_display = DATASET_CONFIGS[ds_name]["name"]

        logger.info(
            "\n[%s]  %d rows after engineering  %d features  %d folds",
            ds_display, len(X), X.shape[1], len(splits),
        )

        n_total = len(models_to_run) * len(methods_to_run) * len(splits)
        eta     = ETATracker(n_total)

        for model_name in models_to_run:
            space = SPACES[model_name]

            for method in methods_to_run:
                fold_results: List[Dict] = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    X_test  = X.iloc[test_idx]
                    y_test  = y.iloc[test_idx]

                    logger.info(
                        "  ┌─ [%s][%s][%s] fold %d/%d  "
                        "(train=%d  test=%d)",
                        ds_name, model_name, method,
                        fold_idx + 1, len(splits),
                        len(train_idx), len(test_idx),
                    )

                    best_params, best_val, metrics, duration, n_evals, sensitivity = run_hpo(
                        method, model_name, space,
                        X_train, y_train,
                        X_test,  y_test,
                        fold_idx, device, logger,
                    )

                    eta_str = eta.tick()
                    logger.info(
                        "  └─ MAPE=%.2f%%  RMSE=%.4f  MAE=%.4f  "
                        "best_val=%.4f  evals=%d  %.1fs  ETA: %s",
                        metrics["mape"], metrics["rmse"], metrics["mae"],
                        best_val, n_evals, duration, eta_str,
                    )

                    record: Dict[str, Any] = {
                        "dataset":      ds_name,
                        "model":        model_name,
                        "method":       method,
                        "fold":         fold_idx + 1,
                        "mape":         metrics["mape"],
                        "rmse":         metrics["rmse"],
                        "mae":          metrics["mae"],
                        "best_val":     best_val,
                        "duration_sec": round(duration, 2),
                        "n_evals":      n_evals,
                        "best_params":  best_params,
                        "sensitivity":  sensitivity,
                    }
                    fold_results.append(record)
                    all_records.append(record)
                    save_results(all_records, logger)   # incremental

                # Per-method aggregate log
                mapes = [r["mape"] for r in fold_results
                         if not np.isnan(r.get("mape", float("nan")))]
                if mapes:
                    logger.info(
                        "  [%s][%s][%s]  MAPE=%.2f±%.2f  "
                        "RMSE=%.4f  MAE=%.4f",
                        ds_name, model_name, method,
                        np.mean(mapes), np.std(mapes),
                        np.nanmean([r["rmse"] for r in fold_results]),
                        np.nanmean([r["mae"]  for r in fold_results]),
                    )

    # ── Final reports ──────────────────────────────────────────────────────
    print_summary(all_records, logger)
    print_sensitivity_report(all_records, logger)

    out_path = RESULTS_DIR / f"forecast_results_{RUN_ID}.csv"
    logger.info("\nAll experiments complete.")
    logger.info("Results  : %s", out_path)
    logger.info("Log      : %s", LOG_DIR / f"forecast_hpo_{RUN_ID}.log")


# =============================================================================
# 19.  Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Forecasting HPO Pipeline — HDMR vs. Baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="payten",
        choices=["payten", "medianova", "both"],
        help="Dataset to run on  (default: payten)",
    )
    parser.add_argument(
        "--models", default="all",
        help=(
            "Comma-separated model tags: xgb,lgb,nbeats,lstm,arima,gru  "
            "or 'all'  (default: all)"
        ),
    )
    parser.add_argument(
        "--methods", default="all",
        help=(
            "Comma-separated method tags: "
            "hdmr,ahdmr200,ahdmr600,rs,optuna  or 'all'  (default: all)"
        ),
    )
    main(parser.parse_args())
