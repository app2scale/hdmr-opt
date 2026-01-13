"""
Time Series Forecasting Module for HDMR Hyperparameter Optimization

Production-ready version with:
- Strict MM/DD/YYYY date parsing (configurable)
- Safer defaults (no mutable default args)
- Improved base class naming (BaseForecaster) + backward-compatible alias
- More robust feature engineering and train/test preparation
- Controlled warnings (no blanket ignore at import time)

Author: APP2SCALE Team
Date: 2026-01-13
Version: 2.2.0
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, Callable, List
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----------------------------------------------------------------------------
# Optional ML libraries
# -----------------------------------------------------------------------------
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Keep as warning (optional dependency)
    warnings.warn("PyTorch not available. Install with: pip install torch")


# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------
DEFAULT_DATE_FORMAT = "%m/%d/%Y"  # STRICT: MM/DD/YYYY


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def mean_absolute_percentage_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        raise ValueError("MAPE undefined: y_true contains only zeros")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def symmetric_mean_absolute_percentage_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def mean_absolute_scaled_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_train: NDArray[np.float64],
    seasonality: int = 1,
) -> float:
    mae_test = mean_absolute_error(y_true, y_pred)

    naive_forecast = y_train[:-seasonality] if seasonality > 1 else y_train[:-1]
    naive_actual = y_train[seasonality:] if seasonality > 1 else y_train[1:]
    mae_naive = mean_absolute_error(naive_actual, naive_forecast)

    if mae_naive == 0:
        return float(np.inf if mae_test > 0 else 0.0)
    return float(mae_test / mae_naive)


def calculate_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_train: Optional[NDArray[np.float64]] = None,
) -> Dict[str, float]:
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError:
        metrics["mape"] = float(np.inf)

    metrics["smape"] = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    if y_train is not None:
        metrics["mase"] = mean_absolute_scaled_error(y_true, y_pred, y_train)

    return metrics


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_time_features(
    df: pd.DataFrame,
    target_col: str = "transactions",
    include_lags: bool = True,
    lag_periods: Optional[List[int]] = None,
    include_rolling: bool = True,
    rolling_windows: Optional[List[int]] = None,
    include_date_features: bool = True,
) -> pd.DataFrame:
    """
    Create time-series features.

    Note: This function expects a DatetimeIndex. If not provided, it will
    attempt to convert the 'date' column to datetime (non-strict).
    For strict parsing, use prepare_train_test(..., date_format=...).
    """
    df = df.copy()

    if lag_periods is None:
        lag_periods = [1, 2, 3, 7, 14, 28]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]

    # Ensure date index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            # non-strict here; strict parsing is enforced in prepare_train_test
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df.set_index("date")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'date' column.")

    # Lags
    if include_lags:
        for lag in lag_periods:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling stats
    if include_rolling:
        for window in rolling_windows:
            roll = df[target_col].rolling(window=window)
            df[f"rolling_mean_{window}"] = roll.mean()
            df[f"rolling_std_{window}"] = roll.std()
            df[f"rolling_min_{window}"] = roll.min()
            df[f"rolling_max_{window}"] = roll.max()

    # Calendar features
    if include_date_features:
        idx = df.index
        df["day_of_week"] = idx.dayofweek
        df["day_of_month"] = idx.day
        df["day_of_year"] = idx.dayofyear
        df["week_of_year"] = idx.isocalendar().week.astype(int)
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["year"] = idx.year
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["is_month_start"] = idx.is_month_start.astype(int)
        df["is_month_end"] = idx.is_month_end.astype(int)
        df["is_quarter_start"] = idx.is_quarter_start.astype(int)
        df["is_quarter_end"] = idx.is_quarter_end.astype(int)
        df["is_year_start"] = idx.is_year_start.astype(int)
        df["is_year_end"] = idx.is_year_end.astype(int)

    return df.dropna()


def prepare_train_test(
    file_path: str,
    split_date: str,
    target_col: str = "transactions",
    date_col: str = "date",
    date_format: str = DEFAULT_DATE_FORMAT,  # preferred format if strict_dates=True
    strict_dates: bool = True,
    dayfirst: bool = False,                  # for auto parsing; keep False for US-style
    **feature_kwargs,
) -> Dict[str, Any]:
    """
    Load data and prepare train/test splits with features.

    Date parsing policy:
    - If strict_dates=True:
        1) Try parsing with `date_format` (errors='raise')
        2) If that fails, fallback to robust auto-detection (format='mixed')
    - If strict_dates=False:
        Auto-detect directly (format='mixed')

    This makes the pipeline resilient when the CSV arrives as:
      - MM/DD/YYYY (e.g., 10/1/2015)
      - YYYY-MM-DD (e.g., 2015-10-01)
      - or mixed (still parseable)

    Returns
    -------
    Dict with:
      - X_train, y_train, X_test, y_test
      - feature_names
      - meta: dict (date_range, n_rows_raw, n_rows_features, split_date, parsing_mode)
    """
    df_raw = pd.read_csv(file_path)

    if date_col not in df_raw.columns or target_col not in df_raw.columns:
        raise ValueError(f"CSV must include columns '{date_col}' and '{target_col}'.")

    # --- Robust datetime parsing ---
    raw_series = df_raw[date_col].astype(str)

    parsing_mode = "auto"
    if strict_dates:
        try:
            df_raw[date_col] = pd.to_datetime(raw_series, format=date_format, errors="raise")
            parsing_mode = f"strict:{date_format}"
        except Exception:
            # Fallback to auto-detection
            df_raw[date_col] = pd.to_datetime(
                raw_series,
                format="mixed",
                dayfirst=dayfirst,
                errors="raise",
            )
            parsing_mode = "auto:fallback"
    else:
        df_raw[date_col] = pd.to_datetime(
            raw_series,
            format="mixed",
            dayfirst=dayfirst,
            errors="raise",
        )
        parsing_mode = "auto"

    # Safety: ensure no NaT
    if df_raw[date_col].isna().any():
        bad = df_raw[df_raw[date_col].isna()].head(5)[date_col]
        raise ValueError(
            "Date parsing produced NaT values. "
            f"Examples of problematic rows: {bad.tolist()}"
        )

    df_raw = df_raw.set_index(date_col).sort_index()

    # Feature engineering (drops NaNs created by lag/rolling)
    df_feat = create_time_features(df_raw, target_col=target_col, **feature_kwargs)

    # Split by date (split_date usually ISO)
    split_dt = pd.to_datetime(split_date, errors="raise")

    train_df = df_feat[df_feat.index < split_dt]
    test_df = df_feat[df_feat.index >= split_dt]

    # Guard: empty splits are common failure sources downstream
    if len(train_df) == 0:
        raise ValueError(
            f"Train split is empty after feature engineering. "
            f"split_date={split_date}, feature_rows={len(df_feat)}, "
            f"date_min={df_feat.index.min() if len(df_feat) else None}, "
            f"date_max={df_feat.index.max() if len(df_feat) else None}"
        )
    if len(test_df) == 0:
        raise ValueError(
            f"Test split is empty after feature engineering. "
            f"split_date={split_date}, feature_rows={len(df_feat)}, "
            f"date_min={df_feat.index.min() if len(df_feat) else None}, "
            f"date_max={df_feat.index.max() if len(df_feat) else None}"
        )

    feature_cols = [c for c in df_feat.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    meta = {
        "n_rows_raw": int(len(df_raw)),
        "n_rows_features": int(len(df_feat)),
        "date_min": str(df_raw.index.min().date()) if len(df_raw) else None,
        "date_max": str(df_raw.index.max().date()) if len(df_raw) else None,
        "split_date": str(split_dt.date()),
        "strict_dates": bool(strict_dates),
        "preferred_date_format": date_format,
        "parsing_mode": parsing_mode,
        "dayfirst": bool(dayfirst),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols,
        "meta": meta,
    }


# =============================================================================
# BASE FORECASTER CLASS
# =============================================================================
class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.
    """

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        pass


# Backward-compatible alias (old typo in codebase)
BaseForcaster = BaseForecaster


# =============================================================================
# XGBOOST FORECASTER
# =============================================================================
class XGBoostForecaster(BaseForecaster):
    def __init__(
        self,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42,
        **kwargs,
    ):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available")

        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = int(min_child_weight)
        self.gamma = float(gamma)
        self.n_estimators = int(n_estimators)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.random_state = int(random_state)
        self.model = None

    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        eval_set = [(X_test, y_test)] if X_test is not None and y_test is not None else None

        self.model = xgb.XGBRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            random_state=self.random_state,
            verbosity=0,
        )

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X_test)

    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        return {
            "learning_rate": (0.001, 0.3),
            "max_depth": (1, 10),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "min_child_weight": (1, 10),
            "gamma": (0, 5),
        }


# =============================================================================
# LIGHTGBM FORECASTER
# =============================================================================
class LightGBMForecaster(BaseForecaster):
    def __init__(
        self,
        learning_rate=0.1,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42,
        **kwargs,
    ):
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM not available")

        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.feature_fraction = float(feature_fraction)
        self.bagging_fraction = float(bagging_fraction)
        self.n_estimators = int(n_estimators)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.random_state = int(random_state)
        self.model = None

    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        eval_set = [(X_test, y_test)] if X_test is not None and y_test is not None else None

        self.model = lgb.LGBMRegressor(
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbosity=-1,
        )

        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)] if eval_set else []
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X_test)

    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        return {
            "learning_rate": (0.001, 0.3),
            "num_leaves": (2, 256),
            "min_data_in_leaf": (1, 100),
            "feature_fraction": (0.5, 1.0),
            "bagging_fraction": (0.5, 1.0),
        }


# =============================================================================
# ARIMA FORECASTER
# =============================================================================
class ARIMAForecaster(BaseForecaster):
    def __init__(self, p=1, d=1, q=1):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not available")
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.model = None

    def fit(self, X_train, y_train, **kwargs):
        self.model = ARIMA(y_train, order=(self.p, self.d, self.q)).fit()
        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        forecast = self.model.forecast(steps=len(X_test))
        return np.array(forecast)

    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        return {"p": (0, 5), "d": (0, 2), "q": (0, 5)}


# =============================================================================
# ETS FORECASTER
# =============================================================================
class ETSForecaster(BaseForecaster):
    def __init__(self, trend="add", seasonal="add", seasonal_periods=7):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not available")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = int(seasonal_periods)
        self.model = None

    def fit(self, X_train, y_train, **kwargs):
        self.model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit()
        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        forecast = self.model.forecast(steps=len(X_test))
        return np.array(forecast)

    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        return {"seasonal_periods": (1, 52)}


# =============================================================================
# OPTIMIZATION OBJECTIVE
# =============================================================================
def create_optimization_objective(
    model_class: type,
    data_dict: Dict[str, Any],
    metric: str = "mape",
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Callable[[NDArray[np.float64]], float]:
    """
    Create objective f(x)->float where x is hyperparameter vector.

    Notes:
    - This trains a model per evaluation (expensive but expected).
    - Exceptions return a large penalty (1e6) to keep optimization running.
    """
    fixed_params = fixed_params or {}

    def objective(hyperparams: NDArray[np.float64]) -> float:
        try:
            param_space = model_class().get_hyperparameter_space()
            param_names = list(param_space.keys())

            if len(hyperparams) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} hyperparameters, got {len(hyperparams)}")

            params = dict(fixed_params)
            for i, name in enumerate(param_names):
                params[name] = float(hyperparams[i])

            # Integer params
            int_params = {
                "max_depth",
                "num_leaves",
                "min_data_in_leaf",
                "min_child_weight",
                "p",
                "d",
                "q",
                "seasonal_periods",
            }
            for p in list(params.keys()):
                if p in int_params:
                    params[p] = int(round(float(params[p])))

            model = model_class(**params)
            model.fit(
                data_dict["X_train"],
                data_dict["y_train"],
                X_test=data_dict["X_test"],
                y_test=data_dict["y_test"],
            )

            y_pred = model.predict(data_dict["X_test"])
            metrics = calculate_metrics(
                data_dict["y_test"].values if hasattr(data_dict["y_test"], "values") else data_dict["y_test"],
                y_pred,
                data_dict["y_train"].values if hasattr(data_dict["y_train"], "values") else data_dict["y_train"],
            )

            if metric not in metrics:
                raise KeyError(f"Metric '{metric}' not computed. Available: {list(metrics.keys())}")

            return float(metrics[metric])

        except Exception as e:
            # Keep warnings short; pipeline logs will capture details as needed
            warnings.warn(f"Objective evaluation failed: {e}")
            return 1e6

    return objective


# =============================================================================
# BACKWARD-COMPAT HELPER
# =============================================================================
def optimize_helper(learning_rate: float, subsample: float) -> float:
    """
    Backward-compatible helper (kept for legacy code).
    """
    import os

    if os.path.exists("./src/data/transactions.csv"):
        file_path = "./src/data/transactions.csv"
    elif os.path.exists("./transactions.csv"):
        file_path = "./transactions.csv"
    else:
        warnings.warn("transactions.csv not found in ./src/data/ or ./")
        return 1e6

    split_date = "2020-01-01"

    params = {
        "n_estimators": 1000,
        "early_stopping_rounds": 100,
        "max_depth": 2,
        "subsample": subsample,
        "learning_rate": learning_rate,
        "random_state": 42,
    }

    try:
        data = prepare_train_test(file_path, split_date, strict_dates=True, date_format=DEFAULT_DATE_FORMAT)
        model = XGBoostForecaster(**params)
        model.fit(data["X_train"], data["y_train"], X_test=data["X_test"], y_test=data["y_test"])
        y_pred = model.predict(data["X_test"])
        metrics = calculate_metrics(data["y_test"].values, y_pred, data["y_train"].values)
        return float(metrics["mape"])
    except Exception as e:
        warnings.warn(f"optimize_helper failed: {e}")
        return 1e6

