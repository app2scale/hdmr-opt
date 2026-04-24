"""
TabArena Optuna (TPE) Baseline: XGBoost HPO — All 18 Datasets
==============================================================

Runs the Optuna TPE baseline on all 18 TabArena datasets (10 regression + 8 classification).
Uses exactly the same protocol and search space as tabarena_hdmr_xgb.py for a fair comparison.

Regression     → XGBRegressor  + RMSE / MAE / R²    (minimize val RMSE)
Classification → XGBClassifier + ROC-AUC / Acc / LL (minimize val log-loss)

Protocol:
  - Search space : TabArena Table C.3 (10D)
  - Budget       : 200 trials per fold (OPTUNA_TRIALS)
  - Outer CV     : 8-fold stratified (classification) / KFold (regression)
  - Inner split  : 80/20 holdout, fixed seed per fold
  - Early stop   : n_estimators=10_000, early_stopping_rounds=50
  - Sampler      : TPE (Tree-structured Parzen Estimator)
  - Pruner       : None

Datasets (TabArena v0.1, OpenML suite 457):
  Regression  (10): 46954 46917 46931 46904 46907 46942 46934 46961 46928 46923
  Classification(8): 46952 46927 46938 46940 46930 46956 46963 46918

Env vars:
  DATASETS        comma-separated DID list (default: all 18)
  TASK_FILTER     regression | classification | all (default: all)
  OPTUNA_TRIALS   number of trials per fold (default: 200)
  N_FOLDS         number of outer CV folds (default: 8)
  SEED            random seed (default: 42)
  LOG_LEVEL       DEBUG | INFO | WARNING (default: INFO)

Usage:
  python tabarena_optuna_xgb.py
  TASK_FILTER=regression python tabarena_optuna_xgb.py
  DATASETS=46917,46904 N_FOLDS=2 OPTUNA_TRIALS=20 python tabarena_optuna_xgb.py  # smoke test

Author : HDMR Research
Version: 2.0.0
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
from sklearn.metrics import (
    accuracy_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("pip install optuna")

# =============================================================================
# Dataset Registry — TabArena v0.1
# =============================================================================

REGRESSION_DATASETS: Dict[int, str] = {
    46954: "QSAR_fish_toxicity",
    46917: "concrete_compressive_strength",
    46931: "healthcare_insurance_expenses",
    46904: "airfoil_self_noise",
    46907: "Fiat500_used",
    46942: "miami_housing",
    46949: "physiochemical_protein",
    46934: "houses",
    46961: "superconductivity",
    46928: "Food_Delivery_Time",
    46923: "diamonds",
}

CLASSIFICATION_DATASETS: Dict[int, str] = {
    46952: "qsar-biodeg",
    46927: "Fitness_Club",
    46938: "Is-this-a-good-customer",
    46940: "Marketing_Campaign",
    46930: "hazelnut-contaminant-detection",
    46956: "seismic-bumps",
    46963: "website_phishing",
    46918: "credit-g",
}

ALL_DATASETS: Dict[int, str] = {**REGRESSION_DATASETS, **CLASSIFICATION_DATASETS}

# =============================================================================
# Configuration
# =============================================================================

_HERE         = Path(__file__).resolve().parent
RUN_ID        = datetime.now().strftime("%Y%m%d_%H%M%S")
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "200"))
N_OUTER_FOLDS = int(os.environ.get("N_FOLDS",       "8"))
SEED          = int(os.environ.get("SEED",           "42"))
LOG_LEVEL     = os.environ.get("LOG_LEVEL",           "INFO")
TASK_FILTER   = os.environ.get("TASK_FILTER",         "all").lower()

_DS_ENV = os.environ.get("DATASETS", "").strip()
if _DS_ENV:
    TARGET_DIDS = [int(x.strip()) for x in _DS_ENV.split(",") if x.strip()]
elif TASK_FILTER == "regression":
    TARGET_DIDS = list(REGRESSION_DATASETS.keys())
elif TASK_FILTER == "classification":
    TARGET_DIDS = list(CLASSIFICATION_DATASETS.keys())
else:
    TARGET_DIDS = list(ALL_DATASETS.keys())

N_ESTIMATORS_FIXED    = 10_000
EARLY_STOPPING_ROUNDS = 50

# =============================================================================
# Search Space — TabArena Table C.3 (10D)
# =============================================================================

PARAM_NAMES: List[str] = [
    "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bylevel", "colsample_bynode",
    "reg_alpha", "reg_lambda", "grow_policy", "max_leaves",
]


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Identical search space to tabarena_hdmr_xgb.py (TabArena Table C.3)."""
    return {
        "learning_rate":     trial.suggest_float("learning_rate",     0.005, 0.1,   log=True),
        "max_depth":         trial.suggest_int(  "max_depth",         4,     10,    log=True),
        "min_child_weight":  trial.suggest_float("min_child_weight",  0.001, 5.0,   log=True),
        "subsample":         trial.suggest_float("subsample",         0.6,   1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6,   1.0),
        "colsample_bynode":  trial.suggest_float("colsample_bynode",  0.6,   1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",         1e-4,  5.0),
        "reg_lambda":        trial.suggest_float("reg_lambda",        1e-4,  5.0),
        "grow_policy":       trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "max_leaves":        trial.suggest_int(  "max_leaves",        8,     1024,  log=True),
    }

# =============================================================================
# Logging
# =============================================================================

# Code Ocean Paths
# Output directory: /results is standard for Code Ocean
# Fallback to local ./results if /results is not writable
RESULTS_DIR = Path('/results')
if not os.access("/", os.W_OK) and not RESULTS_DIR.exists():
    RESULTS_DIR = Path('./results')
LOG_DIR     = RESULTS_DIR / "logs"
TAB_RESULTS = RESULTS_DIR / "tabarena"

def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"optuna_tabarena_{run_id}.log"


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

    logger = logging.getLogger("tpe.benchmark")
    logger.info("=" * 70)
    logger.info("Optuna TPE Baseline  |  XGBoost HPO  |  All 18 Datasets  v2.0")
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
        did, download_data=True,
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
        logger.info("  %s | %d x %d | classes=%d",
                    name, X.shape[0], X.shape[1], len(le.classes_))
    else:
        y = y_s.values.astype(float)
        logger.info("  %s | %d x %d | y mean=%.3f  std=%.3f",
                    name, X.shape[0], X.shape[1], y.mean(), y.std())

    return X, y, name, task

# =============================================================================
# ETA Tracker
# =============================================================================

class ETATracker:
    def __init__(self, total: int, window: int = 3):
        self.total = total; self.done = 0
        self._q: deque = deque(maxlen=window)
        self._t = time.perf_counter()

    def tick(self) -> str:
        self._q.append(time.perf_counter() - self._t)
        self._t = time.perf_counter(); self.done += 1
        rem = self.total - self.done
        if rem == 0 or not self._q: return "done"
        return str(timedelta(seconds=int(sum(self._q) / len(self._q) * rem)))

# =============================================================================
# Optuna Objectives
# =============================================================================

def run_optuna_on_dataset(
    X: NDArray, y: NDArray, name: str, task: str,
    logger: logging.Logger,
) -> Dict:
    import xgboost as xgb

    is_clf    = (task == "classification")
    n_classes = int(np.max(y) + 1) if is_clf else 0
    is_binary = (n_classes == 2)
    primary   = "auc" if is_clf else "rmse"

    logger.info("")
    logger.info("  ┌─ %s  [Optuna-TPE-%d]", name, OPTUNA_TRIALS)
    logger.info("  │  task=%s  trials=%d/fold", task, OPTUNA_TRIALS)

    splitter = (StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
                if is_clf
                else KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED))

    fold_results: List[Dict] = []
    t_wall = time.perf_counter()
    eta    = ETATracker(N_OUTER_FOLDS)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        t_fold = time.perf_counter()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sx   = StandardScaler()
        X_tr = sx.fit_transform(X_train)
        X_te = sx.transform(X_test)
        sy   = None
        y_tr = y_train.copy()
        if not is_clf:
            sy   = StandardScaler()
            y_tr = sy.fit_transform(y_train.reshape(-1, 1)).ravel()

        rng   = np.random.default_rng(SEED + fold_idx)
        n_val = max(1, int(len(y_tr) * 0.2))
        idx   = rng.permutation(len(y_tr))
        X_htr, y_htr = X_tr[idx[n_val:]], y_tr[idx[n_val:]]
        X_hvl, y_hvl = X_tr[idx[:n_val]], y_tr[idx[:n_val]]

        # Convergence tracking
        conv_best: List[float] = []

        with tqdm(total=OPTUNA_TRIALS,
                  desc=f"  [Optuna] {name[:22]} Fold {fold_idx}/{N_OUTER_FOLDS}",
                  unit="trial", leave=True, ncols=95,
                  bar_format=("{desc} |{bar}| {n_fmt}/{total_fmt}"
                              " [{elapsed}<{remaining}]{postfix}")) as pbar:

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial)
                try:
                    if is_clf:
                        obj_m  = "binary:logistic" if is_binary else "multi:softprob"
                        ev_m   = "logloss"         if is_binary else "mlogloss"
                        xp     = {**params, "objective": obj_m, "eval_metric": ev_m}
                        if not is_binary: xp["num_class"] = n_classes
                        model  = xgb.XGBClassifier(
                            **xp, n_estimators=N_ESTIMATORS_FIXED,
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            tree_method="hist", device="cpu",
                            use_label_encoder=False,
                            random_state=SEED + fold_idx, verbosity=0, n_jobs=2,
                        )
                        model.fit(X_htr, y_htr, eval_set=[(X_hvl, y_hvl)], verbose=False)
                        proba = model.predict_proba(X_hvl)
                        score = float(log_loss(y_hvl, proba))
                    else:
                        model = xgb.XGBRegressor(
                            **params, n_estimators=N_ESTIMATORS_FIXED,
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            tree_method="hist", device="cpu",
                            random_state=SEED + fold_idx, verbosity=0, n_jobs=2,
                        )
                        model.fit(X_htr, y_htr, eval_set=[(X_hvl, y_hvl)], verbose=False)
                        score = float(np.sqrt(mean_squared_error(
                            y_hvl, model.predict(X_hvl))))
                except Exception:
                    score = 1e6

                curr_best = min(score, min(conv_best) if conv_best else score)
                conv_best.append(curr_best)
                pbar.update(1)
                pbar.set_postfix({"val": f"{score:.4f}", "best": f"{curr_best:.4f}"})
                return score

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED + fold_idx),
            )
            study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best_trial  = study.best_trial
        best_params = best_trial.params
        best_val    = best_trial.value

        # Test evaluation
        try:
            rng_es = np.random.default_rng(SEED)
            n_es   = max(1, int(len(X_tr) * 0.1))
            idx_es = rng_es.permutation(len(X_tr))
            if is_clf:
                obj_m  = "binary:logistic" if is_binary else "multi:softprob"
                ev_m   = "logloss"         if is_binary else "mlogloss"
                xp     = {**best_params, "objective": obj_m, "eval_metric": ev_m}
                if not is_binary: xp["num_class"] = n_classes
                model  = xgb.XGBClassifier(
                    **xp, n_estimators=N_ESTIMATORS_FIXED,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    tree_method="hist", device="cpu",
                    use_label_encoder=False,
                    random_state=SEED, verbosity=0, n_jobs=2,
                )
                model.fit(X_tr[idx_es[n_es:]], y_tr[idx_es[n_es:]],
                          eval_set=[(X_tr[idx_es[:n_es]], y_tr[idx_es[:n_es]])],
                          verbose=False)
                proba   = model.predict_proba(X_te)
                preds   = model.predict(X_te)
                auc     = (float(roc_auc_score(y_test, proba[:, 1])) if is_binary
                           else float(roc_auc_score(y_test, proba, multi_class="ovr",
                                                    average="macro")))
                metrics = {"auc": auc,
                           "accuracy": float(accuracy_score(y_test, preds)),
                           "logloss":  float(log_loss(y_test, proba))}
            else:
                model = xgb.XGBRegressor(
                    **best_params, n_estimators=N_ESTIMATORS_FIXED,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    tree_method="hist", device="cpu",
                    random_state=SEED, verbosity=0, n_jobs=2,
                )
                model.fit(X_tr[idx_es[n_es:]], y_tr[idx_es[n_es:]],
                          eval_set=[(X_tr[idx_es[:n_es]], y_tr[idx_es[:n_es]])],
                          verbose=False)
                preds = sy.inverse_transform(
                    model.predict(X_te).reshape(-1, 1)).ravel()
                metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                    "mae":  float(mean_absolute_error(y_test, preds)),
                    "r2":   float(r2_score(y_test, preds)),
                }
        except Exception as exc:
            logger.error("Test eval failed fold %d: %s", fold_idx, exc)
            metrics = {}

        fold_sec = time.perf_counter() - t_fold
        eta_str  = eta.tick()

        fold_rec = {
            "fold": fold_idx, "best_val": best_val,
            "n_trials": OPTUNA_TRIALS, "duration_sec": round(fold_sec, 2),
            "best_params": best_params, "convergence": conv_best,
            **metrics,
        }
        fold_results.append(fold_rec)

        pm_val = metrics.get("auc" if is_clf else "rmse", float("nan"))
        logger.info("  │  [Optuna] Fold %d/%d | %s=%.4f | best_val=%.4f | "
                    "trials=%d | %.1fs | ETA: %s",
                    fold_idx, N_OUTER_FOLDS,
                    "AUC" if is_clf else "RMSE", pm_val,
                    best_val, OPTUNA_TRIALS, fold_sec, eta_str)

    total_sec = time.perf_counter() - t_wall

    def _agg(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in fold_results if not np.isnan(r.get(key, float("nan")))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    if is_clf:
        auc_m, auc_s = _agg("auc");   acc_m, acc_s = _agg("accuracy"); ll_m, ll_s = _agg("logloss")
        summary_metrics = {"auc_mean": auc_m, "auc_std": auc_s,
                           "accuracy_mean": acc_m, "accuracy_std": acc_s,
                           "logloss_mean": ll_m, "logloss_std": ll_s}
        pm_mean, pm_std = auc_m, auc_s
    else:
        rm_m, rm_s = _agg("rmse"); ma_m, ma_s = _agg("mae"); r2_m, r2_s = _agg("r2")
        summary_metrics = {"rmse_mean": rm_m, "rmse_std": rm_s,
                           "mae_mean": ma_m, "mae_std": ma_s,
                           "r2_mean": r2_m, "r2_std": r2_s}
        pm_mean, pm_std = rm_m, rm_s

    pm_lbl = "AUC" if is_clf else "RMSE"
    logger.info("  └─ [Optuna] %s | %s=%.4f±%.4f | %.1f min",
                name, pm_lbl, pm_mean, pm_std, total_sec / 60)

    return {
        "dataset": name, "task": task,
        "method": f"Optuna-{OPTUNA_TRIALS}",
        "total_evals": OPTUNA_TRIALS,
        "wall_sec": round(total_sec, 1),
        "fold_results": fold_results,
        **summary_metrics,
    }

# =============================================================================
# Results Persistence
# =============================================================================

def save_results(all_summaries: List[Dict], logger: logging.Logger) -> Path:
    TAB_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = TAB_RESULTS / f"optuna_all18_{RUN_ID}.csv"


    rows = []
    for s in all_summaries:
        for r in s["fold_results"]:
            row: Dict[str, Any] = {
                "run_id":      RUN_ID,
                "dataset":     s["dataset"],
                "task":        s["task"],
                "method":      s["method"],
                "total_evals": s["total_evals"],
                "fold":        r["fold"],
                "seed":        SEED,
                "best_val":    r["best_val"],
                "n_trials":    r["n_trials"],
                "duration_sec":r["duration_sec"],
            }
            for key in ("rmse", "mae", "r2", "auc", "accuracy", "logloss"):
                if key in r: row[key] = r[key]
            row.update({f"param_{k}": v for k, v in r["best_params"].items()})
            for cp in [10, 25, 50, 100, 150, 200]:
                c = r["convergence"]
                row[f"conv_{cp}"] = c[cp-1] if cp <= len(c) else float("nan")
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.debug("Results saved → %s  (%d rows)", out_path, len(rows))
    return out_path

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger = setup_logging(RUN_ID)

    logger.info("Configuration:")
    logger.info("  sampler=TPE  trials=%d/fold  outer_folds=%d  seed=%d",
                OPTUNA_TRIALS, N_OUTER_FOLDS, SEED)
    logger.info("  task_filter=%s  datasets=%s (%d total)",
                TASK_FILTER, TARGET_DIDS, len(TARGET_DIDS))

    all_summaries: List[Dict] = []
    failed       : List[str]  = []
    t_total      = time.perf_counter()

    for did in TARGET_DIDS:
        ds_name = ALL_DATASETS.get(did, f"openml_{did}")
        logger.info("")
        logger.info("▓" * 70)
        logger.info("  DATASET: %s  (DID=%d)", ds_name, did)
        logger.info("▓" * 70)

        try:
            X, y, name, task = load_dataset(did, logger)
        except Exception as exc:
            logger.error("Failed to load DID=%d: %s — SKIPPING", did, exc)
            failed.append(f"load:{ds_name}"); continue

        try:
            summary = run_optuna_on_dataset(X, y, name, task, logger)
            all_summaries.append(summary)
            save_results(all_summaries, logger)
        except Exception as exc:
            logger.error("Optuna failed on %s: %s", name, exc)
            import traceback; traceback.print_exc()
            failed.append(name)

    total_wall = time.perf_counter() - t_total

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("Datasets   : %d / %d",
                len({s["dataset"] for s in all_summaries}), len(TARGET_DIDS))
    logger.info("Failed     : %d  %s", len(failed), failed or "")
    logger.info("Wall time  : %s  (%.1f hr)",
                str(timedelta(seconds=int(total_wall))), total_wall / 3600)
    out = save_results(all_summaries, logger) if all_summaries else None
    logger.info("Results    : %s", out)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()