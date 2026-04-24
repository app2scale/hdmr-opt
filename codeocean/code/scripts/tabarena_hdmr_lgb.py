"""
TabArena HDMR Benchmark: LightGBM HPO — 16 Datasets
=====================================================

Protocol mirrors tabarena_hdmr_xgb.py exactly, adapted for LightGBM.
Compares HDMR-200, A-HDMR-200, A-HDMR-600 on 16 TabArena v0.1 datasets
(9 regression + 7 classification; website_phishing excluded — metric mismatch).

Search space: TabArena Table C.3 — LightGBM (12 active dims)
  learning_rate       LogUniform    [0.005, 0.1]
  feature_fraction    Uniform       [0.4,   1.0]
  bagging_fraction    Uniform       [0.7,   1.0]
  num_leaves          LogUniformInt [2,     200]
  min_data_in_leaf    LogUniformInt [1,      64]
  extra_trees         Binary        {False, True}
  min_data_per_group  LogUniformInt [2,     100]
  cat_l2              LogUniform    [0.005,   2]
  cat_smooth          LogUniform    [0.001, 100]
  max_cat_to_onehot   LogUniformInt [8,     100]
  lambda_l1           Uniform       [1e-4,  1.0]
  lambda_l2           Uniform       [1e-4,  2.0]

  bagging_freq is fixed at 1 (not tuned), reducing active dims to 12.

Env vars:
  DATASETS        comma-separated DID list (default: all 16)
  TASK_FILTER     regression | classification | all (default: all)
  HDMR_SAMPLES    samples per iteration (default: 200)
  HDMR_BASIS      Legendre | Cosine (default: Legendre)
  HDMR_DEGREE     polynomial degree (default: 3)
  HDMR_K          adaptive elite count (default: 50)
  HDMR_EPSILON    convergence tolerance (default: 0.01)
  HDMR_CLIP       bound shrink ratio (default: 0.90)
  N_FOLDS         outer CV folds (default: 8)
  SEED            random seed (default: 42)
  LOG_LEVEL       DEBUG | INFO | WARNING (default: INFO)

Usage:
  python tabarena_hdmr_lgb.py                           # all 16 datasets
  TASK_FILTER=regression python tabarena_hdmr_lgb.py    # regression only
  DATASETS=46954,46917 N_FOLDS=2 HDMR_SAMPLES=20 python tabarena_hdmr_lgb.py  # smoke

Author : HDMR Research
Version: 1.0.0  (TabArena LightGBM, mirrors tabarena_hdmr_xgb.py v7.0)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from src.main import HDMRConfig, HDMROptimizer  # noqa: E402

# =============================================================================
# Dataset Registry — TabArena v0.1  (website_phishing excluded)
# =============================================================================

REGRESSION_DATASETS: Dict[int, str] = {
    46954: "QSAR_fish_toxicity",
    46917: "concrete_compressive_strength",
    46931: "healthcare_insurance_expenses",
    46904: "airfoil_self_noise",
    46907: "Fiat500_used",
    46934: "houses",
    46928: "Food_Delivery_Time",
    46923: "diamonds",
    46949: "physiochemical_protein",
}

CLASSIFICATION_DATASETS: Dict[int, str] = {
    46952: "qsar-biodeg",
    46927: "Fitness_Club",
    46938: "Is-this-a-good-customer",
    46940: "Marketing_Campaign",
    46930: "hazelnut-contaminant-detection",
    46956: "seismic-bumps",
    46918: "credit-g",
}

ALL_DATASETS: Dict[int, str] = {**REGRESSION_DATASETS, **CLASSIFICATION_DATASETS}

# =============================================================================
# TabArena LightGBM Baselines (Erickson et al., 2025) — Appendix C.2
# =============================================================================
# Format: (default_mean, default_std, tuned_mean, tuned_std, ens_mean, ens_std)
# Regression: RMSE ↓    Classification: AUC ↑ (except website_phishing: Logloss ↓)

TABARENA_LGB_BASELINES: Dict[str, Dict] = {
    # Regression
    "QSAR_fish_toxicity":             {"default": (0.894, 0.043), "tuned": (0.889, 0.045), "ens": (0.883, 0.044), "metric": "rmse"},
    "concrete_compressive_strength":  {"default": (4.484, 0.388), "tuned": (4.235, 0.395), "ens": (4.212, 0.396), "metric": "rmse"},
    "healthcare_insurance_expenses":  {"default": (4610.4, 313.8), "tuned": (4525.1, 329.2), "ens": (4511.9, 325.5), "metric": "rmse"},
    "airfoil_self_noise":             {"default": (1.554, 0.093), "tuned": (1.480, 0.108), "ens": (1.451, 0.108), "metric": "rmse"},
    "Fiat500_used":                   {"default": (746.0, 22.4),  "tuned": (740.4, 24.6),  "ens": (729.4, 22.7),  "metric": "rmse"},
    "houses":                         {"default": (0.217, 0.002), "tuned": (0.212, 0.002), "ens": (0.211, 0.002), "metric": "rmse"},
    "Food_Delivery_Time":             {"default": (7.616, 0.053), "tuned": (7.378, 0.054), "ens": (7.374, 0.053), "metric": "rmse"},
    "diamonds":                       {"default": (532.1, 9.1),   "tuned": (524.9, 9.7),   "ens": (519.0, 9.4),   "metric": "rmse"},
    "physiochemical_protein":         {"default": (3.477, 0.026), "tuned": (3.381, 0.027), "ens": (3.384, 0.027), "metric": "rmse"},
    # Classification (AUC ↑)
    "qsar-biodeg":                    {"default": (0.927, 0.012), "tuned": (0.933, 0.012), "ens": (0.933, 0.012), "metric": "auc"},
    "Fitness_Club":                   {"default": (0.795, 0.015), "tuned": (0.815, 0.015), "ens": (0.814, 0.015), "metric": "auc"},
    "Is-this-a-good-customer":        {"default": (0.724, 0.020), "tuned": (0.741, 0.022), "ens": (0.746, 0.020), "metric": "auc"},
    "Marketing_Campaign":             {"default": (0.901, 0.014), "tuned": (0.911, 0.015), "ens": (0.911, 0.014), "metric": "auc"},
    "hazelnut-contaminant-detection": {"default": (0.973, 0.005), "tuned": (0.978, 0.004), "ens": (0.978, 0.004), "metric": "auc"},
    "seismic-bumps":                  {"default": (0.752, 0.027), "tuned": (0.770, 0.027), "ens": (0.771, 0.026), "metric": "auc"},
    "website_phishing":               {"default": (0.255, 0.021), "tuned": (0.249, 0.021), "ens": (0.247, 0.021), "metric": "logloss"},
    "credit-g":                       {"default": (0.771, 0.019), "tuned": (0.792, 0.020), "ens": (0.796, 0.020), "metric": "auc"},
}

# =============================================================================
# Configuration
# =============================================================================

RUN_ID       : str   = datetime.now().strftime("%Y%m%d_%H%M%S")
HDMR_SAMPLES : int   = int(os.environ.get("HDMR_SAMPLES",   "200"))
HDMR_BASIS   : str   = os.environ.get("HDMR_BASIS",          "Legendre")
HDMR_DEGREE  : int   = int(os.environ.get("HDMR_DEGREE",     "3"))
HDMR_K       : int   = int(os.environ.get("HDMR_K",          "50"))
HDMR_EPSILON : float = float(os.environ.get("HDMR_EPSILON",  "0.01"))
HDMR_CLIP    : float = float(os.environ.get("HDMR_CLIP",     "0.90"))
N_OUTER_FOLDS: int   = int(os.environ.get("N_FOLDS",          "8"))
SEED         : int   = int(os.environ.get("SEED",            "42"))
LOG_LEVEL    : str   = os.environ.get("LOG_LEVEL",            "INFO")
TASK_FILTER  : str   = os.environ.get("TASK_FILTER",          "all").lower()

_DS_ENV = os.environ.get("DATASETS", "").strip()
if _DS_ENV:
    TARGET_DIDS = [int(x.strip()) for x in _DS_ENV.split(",") if x.strip()]
elif TASK_FILTER == "regression":
    TARGET_DIDS = list(REGRESSION_DATASETS.keys())
elif TASK_FILTER == "classification":
    TARGET_DIDS = list(CLASSIFICATION_DATASETS.keys())
else:
    TARGET_DIDS = list(ALL_DATASETS.keys())

N_ESTIMATORS_FIXED    : int = 10_000
EARLY_STOPPING_ROUNDS : int = 50

# =============================================================================
# Search Space — TabArena Table C.3 LightGBM (12D active)
# =============================================================================

PARAM_NAMES: List[str] = [
    "learning_rate",      # 0  LogUniform    [0.005, 0.1]
    "feature_fraction",   # 1  Uniform       [0.4,   1.0]
    "bagging_fraction",   # 2  Uniform       [0.7,   1.0]
    "num_leaves",         # 3  LogUniformInt [2,     200]
    "min_data_in_leaf",   # 4  LogUniformInt [1,      64]
    "extra_trees",        # 5  Binary        {0, 1}
    "min_data_per_group", # 6  LogUniformInt [2,     100]
    "cat_l2",             # 7  LogUniform    [0.005,   2]
    "cat_smooth",         # 8  LogUniform    [0.001, 100]
    "max_cat_to_onehot",  # 9  LogUniformInt [8,     100]
    "lambda_l1",          # 10 Uniform       [1e-4,  1.0]
    "lambda_l2",          # 11 Uniform       [1e-4,  2.0]
]
N_DIMS: int = len(PARAM_NAMES)  # 12


def decode_params(x: NDArray) -> Dict[str, Any]:
    """Map unit hypercube [0,1]^12 → LightGBM param dict (TabArena Table C.3)."""
    x = np.clip(x, 0.0, 1.0)

    def log_f  (t, lo, hi): return float(np.exp(np.log(lo) + t * np.log(hi / lo)))
    def lin_f  (t, lo, hi): return float(lo + t * (hi - lo))
    def log_int(t, lo, hi): return max(lo, int(round(
                                np.exp(np.log(lo) + t * np.log(hi / lo)))))

    return {
        "learning_rate":      log_f  (x[0],  0.005,   0.1),
        "feature_fraction":   lin_f  (x[1],  0.4,     1.0),
        "bagging_fraction":   lin_f  (x[2],  0.7,     1.0),
        "num_leaves":         log_int(x[3],  2,       200),
        "min_data_in_leaf":   log_int(x[4],  1,        64),
        "extra_trees":        bool(round(float(x[5]))),
        "min_data_per_group": log_int(x[6],  2,       100),
        "cat_l2":             log_f  (x[7],  0.005,   2.0),
        "cat_smooth":         log_f  (x[8],  0.001, 100.0),
        "max_cat_to_onehot":  log_int(x[9],  8,       100),
        "lambda_l1":          lin_f  (x[10], 1e-4,    1.0),
        "lambda_l2":          lin_f  (x[11], 1e-4,    2.0),
        # fixed
        "bagging_freq":       1,
        "feature_pre_filter": False,
    }

# =============================================================================
# Logging
# =============================================================================

# Code Ocean Paths
DATA_DIR    = Path('/data')
RESULTS_DIR = Path('/results')
LOG_DIR     = RESULTS_DIR / "logs"
TAB_RESULTS = RESULTS_DIR / "tabarena"

def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"hdmr_lgb_tabarena_{run_id}.log"


    fmt  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(fmt, dfmt))
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, dfmt))
    root.addHandler(fh)

    logger = logging.getLogger("hdmr.benchmark")
    logger.info("=" * 70)
    logger.info("HDMR vs Adaptive HDMR  |  LightGBM HPO  |  16 Datasets  v1.0")
    logger.info("Run ID   : %s", run_id)
    logger.info("Log file : %s", log_file)
    logger.info("=" * 70)
    return logger

# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(did: int, logger: logging.Logger
                 ) -> Tuple[NDArray, NDArray, str, str]:
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        import openml
        openml.config.ssl_verify = False
    except ImportError:
        raise ImportError("pip install openml")

    task = "regression" if did in REGRESSION_DATASETS else "classification"
    name = ALL_DATASETS.get(did, f"openml_{did}")

    logger.info("Loading: %s  (DID=%d, task=%s) ...", name, did, task)
    ds = openml.datasets.get_dataset(
        did,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X_df, y_s, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )

    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = X_df[col].astype("category").cat.codes

    X = X_df.fillna(0).values.astype(float)

    if task == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y_s.values.astype(str)).astype(int)
        n_classes = len(le.classes_)
        logger.info("  %s | %d x %d | classes=%d  %s",
                    name, X.shape[0], X.shape[1], n_classes, list(le.classes_[:5]))
    else:
        y = y_s.values.astype(float)
        logger.info("  %s | %d x %d | y mean=%.3f  std=%.3f",
                    name, X.shape[0], X.shape[1], y.mean(), y.std())

    return X, y, name, task

# =============================================================================
# Objectives
# =============================================================================

def make_regression_objective(
    X_train: NDArray, y_train: NDArray,
    seed: int, pbar=None, logger=None,
) -> Tuple[Callable, List]:
    """
    Returns (fun_batch, eval_log).
    fun_batch accepts (N, 12) and returns (N, 1) val RMSE (scaled).
    Minimises val RMSE on 80/20 inner holdout.
    """
    import lightgbm as lgb

    eval_log: List[Tuple[NDArray, float]] = []
    _log = logger or logging.getLogger("hdmr.benchmark")

    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(X_train) * 0.2))
    idx   = rng.permutation(len(X_train))
    X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
    X_htr, y_htr = X_train[idx[n_val:]], y_train[idx[n_val:]]

    train_ds = lgb.Dataset(X_htr, label=y_htr)
    val_ds   = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    def fun_batch(X: NDArray) -> NDArray:
        X = np.atleast_2d(X)
        results = np.zeros((len(X), 1), dtype=np.float64)

        for i, xi in enumerate(X):
            params = decode_params(xi)
            lgb_params = {
                **params,
                "objective":   "regression",
                "metric":      "rmse",
                "verbosity":   -1,
                "seed":         seed,
                "num_threads":  2,
            }
            try:
                callbacks = [
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(-1),
                ]
                booster = lgb.train(
                    lgb_params, train_ds,
                    num_boost_round=N_ESTIMATORS_FIXED,
                    valid_sets=[val_ds],
                    callbacks=callbacks,
                )
                preds    = booster.predict(X_val)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            except Exception as exc:
                _log.debug("LGB regression eval error: %s", exc)
                val_rmse = 1e6

            results[i, 0] = val_rmse
            eval_log.append((xi.copy(), val_rmse))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"val": f"{val_rmse:.4f}",
                                  "best": f"{min(r for _,r in eval_log):.4f}"})

        return results

    return fun_batch, eval_log


def make_classification_objective(
    X_train: NDArray, y_train: NDArray,
    n_classes: int, seed: int, pbar=None, logger=None,
) -> Tuple[Callable, List]:
    """
    Minimises val log-loss on 80/20 inner holdout (lower = better, consistent
    with HDMR minimisation convention).
    """
    import lightgbm as lgb

    eval_log: List[Tuple[NDArray, float]] = []
    _log = logger or logging.getLogger("hdmr.benchmark")

    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(X_train) * 0.2))
    idx   = rng.permutation(len(X_train))
    X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
    X_htr, y_htr = X_train[idx[n_val:]], y_train[idx[n_val:]]

    objective  = "binary"       if n_classes == 2 else "multiclass"
    metric     = "binary_logloss" if n_classes == 2 else "multi_logloss"
    num_class  = {"num_class": n_classes} if n_classes > 2 else {}

    train_ds = lgb.Dataset(X_htr, label=y_htr)
    val_ds   = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    def fun_batch(X: NDArray) -> NDArray:
        X = np.atleast_2d(X)
        results = np.zeros((len(X), 1), dtype=np.float64)

        for i, xi in enumerate(X):
            params = decode_params(xi)
            lgb_params = {
                **params,
                "objective":   objective,
                "metric":      metric,
                "verbosity":   -1,
                "seed":         seed,
                "num_threads":  2,
                **num_class,
            }
            try:
                callbacks = [
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(-1),
                ]
                booster = lgb.train(
                    lgb_params, train_ds,
                    num_boost_round=N_ESTIMATORS_FIXED,
                    valid_sets=[val_ds],
                    callbacks=callbacks,
                )
                if n_classes == 2:
                    proba    = booster.predict(X_val).reshape(-1, 1)
                    proba    = np.hstack([1 - proba, proba])
                else:
                    proba    = booster.predict(X_val)
                val_loss = float(log_loss(y_val, proba))
            except Exception as exc:
                _log.debug("LGB classification eval error: %s", exc)
                val_loss = 1e6

            results[i, 0] = val_loss
            eval_log.append((xi.copy(), val_loss))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"val": f"{val_loss:.4f}",
                                  "best": f"{min(r for _,r in eval_log):.4f}"})

        return results

    return fun_batch, eval_log

# =============================================================================
# Test Evaluation
# =============================================================================

def evaluate_regression(
    params: Dict[str, Any],
    X_tr: NDArray, y_tr: NDArray,
    X_te: NDArray, y_te_raw: NDArray,
    sy: Optional[StandardScaler],
) -> Dict[str, float]:
    import lightgbm as lgb

    rng_es = np.random.default_rng(SEED)
    n_es   = max(1, int(len(X_tr) * 0.1))
    idx_es = rng_es.permutation(len(X_tr))

    train_ds = lgb.Dataset(X_tr[idx_es[n_es:]], label=y_tr[idx_es[n_es:]])
    val_ds   = lgb.Dataset(X_tr[idx_es[:n_es]],  label=y_tr[idx_es[:n_es]],
                           reference=train_ds)

    lgb_params = {
        **params,
        "objective":   "regression",
        "metric":      "rmse",
        "verbosity":   -1,
        "seed":         SEED,
        "num_threads":  2,
    }
    try:
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                     lgb.log_evaluation(-1)]
        booster = lgb.train(lgb_params, train_ds,
                            num_boost_round=N_ESTIMATORS_FIXED,
                            valid_sets=[val_ds], callbacks=callbacks)
        preds_s = booster.predict(X_te)
        preds   = sy.inverse_transform(preds_s.reshape(-1, 1)).ravel() if sy else preds_s
        rmse    = float(np.sqrt(mean_squared_error(y_te_raw, preds)))
        mae     = float(mean_absolute_error(y_te_raw, preds))
        r2      = float(r2_score(y_te_raw, preds))
    except Exception as exc:
        logging.getLogger("hdmr.benchmark").error("Test regression eval failed: %s", exc)
        rmse = mae = r2 = float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_classification(
    params: Dict[str, Any],
    X_tr: NDArray, y_tr: NDArray,
    X_te: NDArray, y_te: NDArray,
    n_classes: int,
) -> Dict[str, float]:
    import lightgbm as lgb

    objective = "binary"        if n_classes == 2 else "multiclass"
    metric    = "binary_logloss" if n_classes == 2 else "multi_logloss"
    num_class = {"num_class": n_classes} if n_classes > 2 else {}

    rng_es = np.random.default_rng(SEED)
    n_es   = max(1, int(len(X_tr) * 0.1))
    idx_es = rng_es.permutation(len(X_tr))

    train_ds = lgb.Dataset(X_tr[idx_es[n_es:]], label=y_tr[idx_es[n_es:]])
    val_ds   = lgb.Dataset(X_tr[idx_es[:n_es]],  label=y_tr[idx_es[:n_es]],
                           reference=train_ds)

    lgb_params = {
        **params,
        "objective":   objective,
        "metric":      metric,
        "verbosity":   -1,
        "seed":         SEED,
        "num_threads":  2,
        **num_class,
    }
    try:
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                     lgb.log_evaluation(-1)]
        booster = lgb.train(lgb_params, train_ds,
                            num_boost_round=N_ESTIMATORS_FIXED,
                            valid_sets=[val_ds], callbacks=callbacks)

        if n_classes == 2:
            proba = booster.predict(X_te).reshape(-1, 1)
            proba = np.hstack([1 - proba, proba])
        else:
            proba = booster.predict(X_te)

        preds   = np.argmax(proba, axis=1)
        auc     = float(roc_auc_score(y_te, proba[:, 1] if n_classes == 2 else proba,
                                      multi_class="ovr", average="weighted"))
        acc     = float(accuracy_score(y_te, preds))
        logloss = float(log_loss(y_te, proba))
    except Exception as exc:
        logging.getLogger("hdmr.benchmark").error("Test clf eval failed: %s", exc)
        auc = acc = logloss = float("nan")

    return {"auc": auc, "accuracy": acc, "logloss": logloss}

# =============================================================================
# Sensitivity Analysis
# =============================================================================

def compute_sensitivity(optimizer) -> Dict[str, float]:
    """First-order Sobol indices from HDMR alpha coefficients."""
    if optimizer.alpha is None:
        return {n: 0.0 for n in PARAM_NAMES}
    V_i = np.sum(optimizer.alpha ** 2, axis=0)
    tot = V_i.sum()
    if tot < 1e-12:
        return {n: 0.0 for n in PARAM_NAMES}
    return {n: float(V_i[i] / tot) for i, n in enumerate(PARAM_NAMES)}

# =============================================================================
# ETA Tracker
# =============================================================================

class ETATracker:
    def __init__(self, total: int, window: int = 3):
        self.total = total
        self.done  = 0
        self._q: deque = deque(maxlen=window)
        self._t = time.perf_counter()

    def tick(self) -> str:
        self._q.append(time.perf_counter() - self._t)
        self._t = time.perf_counter()
        self.done += 1
        rem = self.total - self.done
        if rem == 0 or not self._q:
            return "done"
        avg = sum(self._q) / len(self._q)
        return str(timedelta(seconds=int(avg * rem)))

# =============================================================================
# Single Method × Dataset Run
# =============================================================================

def run_method_on_dataset(
    X: NDArray, y: NDArray, name: str, task: str,
    adaptive: bool, maxiter: int, samples: int,
    logger: logging.Logger,
) -> Dict:
    iters       = maxiter if adaptive else 1
    total_per_f = samples * iters
    tag         = f"A-HDMR-{total_per_f}" if adaptive else f"HDMR-{samples}"
    is_clf      = (task == "classification")
    n_classes   = int(np.max(y) + 1) if is_clf else 0
    primary_metric = "auc" if is_clf else "rmse"

    logger.info("")
    logger.info("  ┌─ %s  [%s]", name, tag)
    logger.info("  │  task=%s  samples=%d  maxiter=%d  → %d evals/fold",
                task, samples, iters, total_per_f)

    splitter = (StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
                if is_clf
                else KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED))

    fold_results: List[Dict] = []
    all_sens    : List[Dict] = []
    t_wall      = time.perf_counter()
    eta         = ETATracker(N_OUTER_FOLDS)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        t_fold = time.perf_counter()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sx    = StandardScaler()
        X_tr  = sx.fit_transform(X_train)
        X_te  = sx.transform(X_test)
        sy    = None
        y_tr  = y_train.copy()
        if not is_clf:
            sy   = StandardScaler()
            y_tr = sy.fit_transform(y_train.reshape(-1, 1)).ravel()

        with tqdm(
            total=total_per_f,
            desc=f"  [{tag}] {name[:22]} Fold {fold_idx}/{N_OUTER_FOLDS}",
            unit="eval", leave=True, ncols=95,
            bar_format=("{desc} |{bar}| {n_fmt}/{total_fmt}"
                        " [{elapsed}<{remaining}]{postfix}"),
        ) as pbar:
            if is_clf:
                obj_fn, eval_log = make_classification_objective(
                    X_tr, y_tr, n_classes=n_classes,
                    seed=SEED + fold_idx, pbar=pbar, logger=logger)
            else:
                obj_fn, eval_log = make_regression_objective(
                    X_tr, y_tr,
                    seed=SEED + fold_idx, pbar=pbar, logger=logger)

            cfg = HDMRConfig(
                n=N_DIMS, a=0.0, b=1.0,
                N=samples, m=HDMR_DEGREE, basis=HDMR_BASIS,
                seed=SEED + fold_idx,
                adaptive=adaptive, maxiter=maxiter,
                k=HDMR_K, epsilon=HDMR_EPSILON, clip=HDMR_CLIP,
                disp=False, enable_plots=False,
            )
            optimizer = HDMROptimizer(fun_batch=obj_fn, config=cfg)
            result    = optimizer.solve(np.full(N_DIMS, 0.5))

        best_val    = min(r for _, r in eval_log)
        best_params = decode_params(result.x)

        if is_clf:
            metrics = evaluate_classification(
                best_params, X_tr, y_tr, X_te, y_test, n_classes)
        else:
            metrics = evaluate_regression(
                best_params, X_tr, y_tr, X_te, y_test, sy)

        sens     = compute_sensitivity(optimizer)
        fold_sec = time.perf_counter() - t_fold
        eta_str  = eta.tick()
        all_sens.append(sens)

        conv_vals = [r for _, r in eval_log]
        conv_best = [min(conv_vals[:i+1]) for i in range(len(conv_vals))]

        fold_rec = {
            "fold": fold_idx, "best_val": best_val,
            "n_evals": len(eval_log), "duration_sec": round(fold_sec, 2),
            "best_params": best_params, "sensitivity": sens,
            "convergence": conv_best, **metrics,
        }
        fold_results.append(fold_rec)

        pm_val = metrics.get(primary_metric, float("nan"))
        logger.info("  │  [%s] Fold %d/%d | %s=%.4f | best_val=%.4f | "
                    "evals=%d | %.1fs | ETA: %s",
                    tag, fold_idx, N_OUTER_FOLDS,
                    primary_metric.upper(), pm_val,
                    best_val, len(eval_log), fold_sec, eta_str)

    total_sec = time.perf_counter() - t_wall

    def _agg(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in fold_results if not np.isnan(r.get(key, float("nan")))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    if is_clf:
        auc_mean, auc_std  = _agg("auc")
        acc_mean, acc_std  = _agg("accuracy")
        ll_mean,  ll_std   = _agg("logloss")
        metrics_summary    = {"auc_mean": auc_mean, "auc_std": auc_std,
                               "accuracy_mean": acc_mean, "accuracy_std": acc_std,
                               "logloss_mean": ll_mean, "logloss_std": ll_std}
        pm_mean, pm_std    = auc_mean, auc_std
    else:
        rmse_mean, rmse_std = _agg("rmse")
        mae_mean,  mae_std  = _agg("mae")
        r2_mean,   r2_std   = _agg("r2")
        metrics_summary     = {"rmse_mean": rmse_mean, "rmse_std": rmse_std,
                                "mae_mean": mae_mean,   "mae_std": mae_std,
                                "r2_mean":  r2_mean,    "r2_std":  r2_std}
        pm_mean, pm_std     = rmse_mean, rmse_std

    mean_sens = {n: float(np.mean([s.get(n, 0) for s in all_sens]))
                 for n in PARAM_NAMES}
    sorted_s  = sorted(mean_sens.items(), key=lambda kv: -kv[1])
    cumul = 0.0; k80 = 0
    for rank, (_, si) in enumerate(sorted_s, 1):
        cumul += si
        if k80 == 0 and cumul >= 0.80:
            k80 = rank

    summary = {
        "dataset":      name,
        "task":         task,
        "method":       tag,
        "adaptive":     adaptive,
        "samples":      samples,
        "maxiter":      iters,
        "total_evals":  total_per_f,
        "wall_sec":     round(total_sec, 1),
        "eff_dim":      k80 or N_DIMS,
        "top1_param":   sorted_s[0][0]  if sorted_s        else "",
        "top1_si":      sorted_s[0][1]  if sorted_s        else 0.0,
        "top2_param":   sorted_s[1][0]  if len(sorted_s)>1 else "",
        "top2_si":      sorted_s[1][1]  if len(sorted_s)>1 else 0.0,
        "top3_param":   sorted_s[2][0]  if len(sorted_s)>2 else "",
        "top3_si":      sorted_s[2][1]  if len(sorted_s)>2 else 0.0,
        "fold_results": fold_results,
        "sensitivity":  mean_sens,
        **metrics_summary,
    }

    metric_label = ("AUC" if is_clf else "RMSE")
    logger.info("  └─ [%s] %s | %s=%.4f±%.4f | eff_dim=%d/%d | %.1f min",
                tag, name, metric_label, pm_mean, pm_std,
                k80 or N_DIMS, N_DIMS, total_sec / 60)

    return summary

# =============================================================================
# Results Persistence
# =============================================================================

def save_results(all_summaries: List[Dict], logger: logging.Logger) -> Path:
    """Incremental CSV save — safe to call after each method completes."""
    TAB_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = TAB_RESULTS / f"lgb_all16_{RUN_ID}.csv"


    rows = []
    for s in all_summaries:
        for r in s["fold_results"]:
            row: Dict[str, Any] = {
                "run_id":      RUN_ID,
                "dataset":     s["dataset"],
                "task":        s["task"],
                "method":      s["method"],
                "adaptive":    s["adaptive"],
                "total_evals": s["total_evals"],
                "fold":        r["fold"],
                "seed":        SEED,
                "best_val":    r["best_val"],
                "n_evals":     r["n_evals"],
                "duration_sec":r["duration_sec"],
            }
            for key in ("rmse", "mae", "r2", "auc", "accuracy", "logloss"):
                if key in r:
                    row[key] = r[key]
            row.update({f"sens_{k}": v for k, v in r["sensitivity"].items()})
            row.update({f"param_{k}": v for k, v in r["best_params"].items()})
            for cp in [10, 25, 50, 100, 150, 200]:
                c = r["convergence"]
                row[f"conv_{cp}"] = c[cp-1] if cp <= len(c) else float("nan")
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.debug("Results saved → %s  (%d rows)", out_path, len(rows))
    return out_path

# =============================================================================
# Comparison Report
# =============================================================================

def print_comparison_report(all_summaries: List[Dict],
                             logger: logging.Logger) -> None:
    by_ds: Dict[str, Dict[str, Dict]] = {}
    for s in all_summaries:
        by_ds.setdefault(s["dataset"], {})[s["method"]] = s

    methods = sorted({s["method"] for s in all_summaries})

    for task in ("regression", "classification"):
        task_datasets = {ds: ms for ds, ms in by_ds.items()
                         if next(iter(ms.values()))["task"] == task}
        if not task_datasets:
            continue

        primary = "rmse_mean" if task == "regression" else "auc_mean"
        pm_lbl  = "RMSE ↓"   if task == "regression" else "AUC ↑"
        best_fn = (min if task == "regression" else max)

        logger.info("")
        logger.info("=" * 110)
        logger.info("  %s  ─  %s (mean±std over %d folds)  |  TabArena LightGBM baselines: Erickson et al., 2025",
                    task.upper(), pm_lbl, N_OUTER_FOLDS)
        logger.info("=" * 110)

        # Header: our methods + TabArena columns
        hdr = f"  {'Dataset':<38}"
        for m in methods:
            hdr += f"  {m:<18}"
        hdr += f"  {'LGB-Default':>14}  {'LGB-Tuned':>14}  {'LGB-Tuned+Ens':>14}"
        logger.info(hdr)
        logger.info("  " + "─" * (38 + len(methods) * 20 + 48))

        wins = {m: 0 for m in methods}
        for ds_name, ms in sorted(task_datasets.items()):
            avail = {m: ms[m] for m in methods if m in ms}
            if not avail:
                continue
            best_val = best_fn(avail[m][primary] for m in avail)
            row = f"  {ds_name:<38}"

            for m in methods:
                if m not in ms:
                    row += f"  {'—':<18}"; continue
                pm  = ms[m][primary]
                std = ms[m][primary.replace("_mean", "_std")]
                mark = " ★" if abs(pm - best_val) < 1e-9 else "  "
                row += f"  {pm:.4f}±{std:.4f}{mark}"

            winner = best_fn(avail, key=lambda m: avail[m][primary])
            wins[winner] += 1

            # TabArena LightGBM baselines
            bl = TABARENA_LGB_BASELINES.get(ds_name)
            if bl:
                row += f"  {bl['default'][0]:>6.4f}±{bl['default'][1]:.4f}"
                row += f"  {bl['tuned'][0]:>6.4f}±{bl['tuned'][1]:.4f}"
                row += f"  {bl['ens'][0]:>6.4f}±{bl['ens'][1]:.4f}"
            else:
                row += f"  {'—':>14}  {'—':>14}  {'—':>14}"

            logger.info(row)

        logger.info("  " + "─" * (38 + len(methods) * 20 + 48))
        wr = f"  {'Wins (our methods)':<38}"
        for m in methods:
            wr += f"  {wins[m]:<18}"
        logger.info(wr)

    # ── Comparison vs TabArena Tuned ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("  COMPARISON vs TabArena LightGBM-Tuned  (A-HDMR-600 = best budget)")
    logger.info("  + = better than Tuned,  - = worse,  ≈ = within 1σ")
    logger.info("=" * 70)
    logger.info("  %-38s  %-14s  %-14s  %-8s", "Dataset", "A-HDMR-600", "LGB-Tuned", "Delta%")
    logger.info("  " + "─" * 78)
    ahdmr600_key = next((m for m in sorted(
        {s["method"] for s in all_summaries}) if "600" in m), None)
    if ahdmr600_key:
        for task in ("regression", "classification"):
            for s in sorted(all_summaries, key=lambda x: x["dataset"]):
                if s["method"] != ahdmr600_key or s["task"] != task:
                    continue
                bl = TABARENA_LGB_BASELINES.get(s["dataset"])
                if not bl:
                    continue
                is_reg = (task == "regression")
                our_m   = s["rmse_mean"] if is_reg else s["auc_mean"]
                our_s   = s["rmse_std"]  if is_reg else s["auc_std"]
                ref_m, ref_s = bl["tuned"]
                delta_pct = (our_m - ref_m) / (abs(ref_m) + 1e-10) * 100
                # positive delta = worse for regression, better for classification
                if is_reg:
                    sign = "+" if delta_pct > 0 else ("≈" if abs(delta_pct) < 1.0 else "-")
                    verdict = f"{sign} ({delta_pct:+.1f}%)"
                else:
                    sign = "+" if delta_pct > 0 else ("≈" if abs(delta_pct) < 0.5 else "-")
                    verdict = f"{sign} ({delta_pct:+.1f}%)"
                logger.info("  %-38s  %.4f±%.4f  %.4f±%.4f  %s",
                            s["dataset"], our_m, our_s, ref_m, ref_s, verdict)

    # ── Sensitivity ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SENSITIVITY SUMMARY  (HDMR-200, top-3 params per dataset)")
    logger.info("=" * 70)
    logger.info("  %-38s  %-22s  %-22s  %-22s  %s",
                "Dataset", "Top-1", "Top-2", "Top-3", "EffDim")
    logger.info("  " + "─" * 118)
    hdmr_key = next((m for m in methods if not m.startswith("A-")), None)
    if hdmr_key:
        for s in sorted(all_summaries, key=lambda x: (x["task"], x["dataset"])):
            if s["method"] != hdmr_key:
                continue
            logger.info("  %-38s  %-22s  %-22s  %-22s  %d/%d",
                        s["dataset"],
                        f"{s['top1_param']}({s['top1_si']:.3f})",
                        f"{s['top2_param']}({s['top2_si']:.3f})",
                        f"{s['top3_param']}({s['top3_si']:.3f})",
                        s["eff_dim"], N_DIMS)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger = setup_logging(RUN_ID)

    logger.info("Configuration:")
    logger.info("  basis=%s  degree=%d  samples=%d",
                HDMR_BASIS, HDMR_DEGREE, HDMR_SAMPLES)
    logger.info("  k=%d  eps=%.3f  clip=%.2f", HDMR_K, HDMR_EPSILON, HDMR_CLIP)
    logger.info("  outer_folds=%d  seed=%d  search_space=TabArena C.3 LightGBM (%dD)",
                N_OUTER_FOLDS, SEED, N_DIMS)
    logger.info("  task_filter=%s  datasets=%s (%d total)",
                TASK_FILTER, TARGET_DIDS, len(TARGET_DIDS))
    logger.info("")
    logger.info("Methods:")
    logger.info("  1) HDMR-200     — standard,   200 evals/fold")
    logger.info("  2) A-HDMR-200   — adaptive,   T=2×100 = 200 evals/fold (equal budget)")
    logger.info("  3) A-HDMR-600   — adaptive,   T=3×200 = 600 evals/fold (full budget)")

    METHODS = [
        {"adaptive": False, "samples": HDMR_SAMPLES,      "maxiter": 1},  # HDMR-200
        {"adaptive": True,  "samples": HDMR_SAMPLES // 2, "maxiter": 2},  # A-HDMR-200
        {"adaptive": True,  "samples": HDMR_SAMPLES,      "maxiter": 3},  # A-HDMR-600
    ]

    all_summaries : List[Dict] = []
    failed        : List[str]  = []
    t_total       = time.perf_counter()

    for did in TARGET_DIDS:
        ds_name = ALL_DATASETS.get(did, f"openml_{did}")
        logger.info("")
        logger.info("▓" * 70)
        logger.info("  DATASET: %s  (DID=%d)", ds_name, did)
        logger.info("▓" * 70)

        try:
            X, y, name, task = load_dataset(did, logger)
        except Exception as exc:
            logger.error("Failed to load DID=%d (%s): %s — SKIPPING", did, ds_name, exc)
            failed.append(f"load:{ds_name}")
            continue

        for method in METHODS:
            m_tag = (f"A-HDMR-{method['samples']*method['maxiter']}"
                     if method["adaptive"]
                     else f"HDMR-{method['samples']}")
            try:
                summary = run_method_on_dataset(
                    X, y, name, task,
                    adaptive=method["adaptive"],
                    maxiter=method["maxiter"],
                    samples=method["samples"],
                    logger=logger,
                )
                all_summaries.append(summary)
                save_results(all_summaries, logger)  # incremental save
            except Exception as exc:
                logger.error("Method %s failed on %s: %s", m_tag, name, exc)
                import traceback; traceback.print_exc()
                failed.append(f"{m_tag}:{name}")

    total_wall = time.perf_counter() - t_total

    if all_summaries:
        print_comparison_report(all_summaries, logger)
        out_path = save_results(all_summaries, logger)
    else:
        logger.error("No results to save.")
        out_path = None

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("Datasets completed : %d / %d",
                len({s["dataset"] for s in all_summaries}), len(TARGET_DIDS))
    logger.info("Total summaries    : %d  (dataset × method)", len(all_summaries))
    logger.info("Failed             : %d  %s", len(failed), failed or "")
    logger.info("Total wall time    : %s  (%.1f hr)",
                str(timedelta(seconds=int(total_wall))), total_wall / 3600)
    logger.info("Results            : %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
