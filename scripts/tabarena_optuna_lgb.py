"""
TabArena Optuna (TPE) Baseline: LightGBM HPO — 16 Datasets
===========================================================

Exact baseline counterpart to tabarena_hdmr_lgb.py.
Uses Optuna TPE sampler — same budget (200 trials/fold) as HDMR-200.
16 TabArena v0.1 datasets (9 regression + 7 classification).

Search space: TabArena Table C.3 — LightGBM (12 active dims)

Env vars:
  DATASETS        comma-separated DID list (default: all 16)
  TASK_FILTER     regression | classification | all (default: all)
  OPTUNA_TRIALS   trials per fold (default: 200)
  N_FOLDS         outer CV folds (default: 8)
  SEED            random seed (default: 42)
  LOG_LEVEL       DEBUG | INFO | WARNING (default: INFO)

Usage:
  python tabarena_optuna_lgb.py
  DATASETS=46954,46917 N_FOLDS=2 OPTUNA_TRIALS=20 python tabarena_optuna_lgb.py  # smoke

Author : HDMR Research
Version: 1.0.0
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("pip install optuna")

_HERE = Path(__file__).resolve().parent.parent

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

TABARENA_LGB_BASELINES: Dict[str, Dict] = {
    "QSAR_fish_toxicity":             {"default": (0.894, 0.043), "tuned": (0.889, 0.045), "ens": (0.883, 0.044), "metric": "rmse"},
    "concrete_compressive_strength":  {"default": (4.484, 0.388), "tuned": (4.235, 0.395), "ens": (4.212, 0.396), "metric": "rmse"},
    "healthcare_insurance_expenses":  {"default": (4610.4, 313.8), "tuned": (4525.1, 329.2), "ens": (4511.9, 325.5), "metric": "rmse"},
    "airfoil_self_noise":             {"default": (1.554, 0.093), "tuned": (1.480, 0.108), "ens": (1.451, 0.108), "metric": "rmse"},
    "Fiat500_used":                   {"default": (746.0, 22.4),  "tuned": (740.4, 24.6),  "ens": (729.4, 22.7),  "metric": "rmse"},
    "houses":                         {"default": (0.217, 0.002), "tuned": (0.212, 0.002), "ens": (0.211, 0.002), "metric": "rmse"},
    "Food_Delivery_Time":             {"default": (7.616, 0.053), "tuned": (7.378, 0.054), "ens": (7.374, 0.053), "metric": "rmse"},
    "diamonds":                       {"default": (532.1, 9.1),   "tuned": (524.9, 9.7),   "ens": (519.0, 9.4),   "metric": "rmse"},
    "physiochemical_protein":         {"default": (3.477, 0.026), "tuned": (3.381, 0.027), "ens": (3.384, 0.027), "metric": "rmse"},
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

RUN_ID        : str  = datetime.now().strftime("%Y%m%d_%H%M%S")
OPTUNA_TRIALS : int  = int(os.environ.get("OPTUNA_TRIALS", "200"))
N_OUTER_FOLDS : int  = int(os.environ.get("N_FOLDS",        "8"))
SEED          : int  = int(os.environ.get("SEED",           "42"))
LOG_LEVEL     : str  = os.environ.get("LOG_LEVEL",          "INFO")
TASK_FILTER   : str  = os.environ.get("TASK_FILTER",        "all").lower()

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

def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Optuna trial → LightGBM param dict (TabArena Table C.3)."""
    return {
        "learning_rate":      trial.suggest_float("learning_rate",      0.005,   0.1,   log=True),
        "feature_fraction":   trial.suggest_float("feature_fraction",   0.4,     1.0),
        "bagging_fraction":   trial.suggest_float("bagging_fraction",   0.7,     1.0),
        "num_leaves":         trial.suggest_int(  "num_leaves",         2,       200,   log=True),
        "min_data_in_leaf":   trial.suggest_int(  "min_data_in_leaf",   1,       64,    log=True),
        "extra_trees":        trial.suggest_categorical("extra_trees",  [False, True]),
        "min_data_per_group": trial.suggest_int(  "min_data_per_group", 2,       100,   log=True),
        "cat_l2":             trial.suggest_float("cat_l2",             0.005,   2.0,   log=True),
        "cat_smooth":         trial.suggest_float("cat_smooth",         0.001, 100.0,   log=True),
        "max_cat_to_onehot":  trial.suggest_int(  "max_cat_to_onehot",  8,       100,   log=True),
        "lambda_l1":          trial.suggest_float("lambda_l1",          1e-4,    1.0),
        "lambda_l2":          trial.suggest_float("lambda_l2",          1e-4,    2.0),
        "bagging_freq":       1,
        "feature_pre_filter": False,
    }

# =============================================================================
# Logging
# =============================================================================

def setup_logging(run_id: str) -> logging.Logger:
    log_dir  = _HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"optuna_lgb_tabarena_{run_id}.log"

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

    logger = logging.getLogger("optuna.benchmark")
    logger.info("=" * 70)
    logger.info("Optuna TPE  |  LightGBM HPO  |  16 Datasets  v1.0")
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
        download_qualities=False, download_features_meta_data=False,
    )
    X_df, y_s, _, _ = ds.get_data(
        dataset_format="dataframe", target=ds.default_target_attribute)

    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = X_df[col].astype("category").cat.codes
    X = X_df.fillna(0).values.astype(float)

    if task == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y_s.values.astype(str)).astype(int)
        logger.info("  %s | %d x %d | classes=%d", name, *X.shape, len(le.classes_))
    else:
        y = y_s.values.astype(float)
        logger.info("  %s | %d x %d | y mean=%.3f  std=%.3f",
                    name, X.shape[0], X.shape[1], y.mean(), y.std())

    return X, y, name, task

# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_lgb(params, X_tr, y_tr, X_te, y_te, task, n_classes,
                 sy=None) -> Dict[str, float]:
    import lightgbm as lgb

    is_clf    = (task == "classification")
    objective = "binary" if n_classes == 2 else ("multiclass" if is_clf else "regression")
    metric    = "binary_logloss" if n_classes == 2 else ("multi_logloss" if is_clf else "rmse")
    num_class = {"num_class": n_classes} if (is_clf and n_classes > 2) else {}

    rng_es = np.random.default_rng(SEED)
    n_es   = max(1, int(len(X_tr) * 0.1))
    idx_es = rng_es.permutation(len(X_tr))

    train_ds = lgb.Dataset(X_tr[idx_es[n_es:]], label=y_tr[idx_es[n_es:]])
    val_ds   = lgb.Dataset(X_tr[idx_es[:n_es]],  label=y_tr[idx_es[:n_es]],
                           reference=train_ds)
    lgb_params = {**params, "objective": objective, "metric": metric,
                  "verbosity": -1, "seed": SEED, "num_threads": 2, **num_class}
    try:
        cb = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
              lgb.log_evaluation(-1)]
        booster = lgb.train(lgb_params, train_ds,
                            num_boost_round=N_ESTIMATORS_FIXED,
                            valid_sets=[val_ds], callbacks=cb)
        if is_clf:
            proba = (booster.predict(X_te).reshape(-1, 1) if n_classes == 2
                     else booster.predict(X_te))
            if n_classes == 2:
                proba = np.hstack([1 - proba, proba])
            preds = np.argmax(proba, axis=1)
            return {"auc": float(roc_auc_score(y_te, proba[:, 1] if n_classes == 2 else proba,
                                               multi_class="ovr", average="weighted")),
                    "accuracy": float(accuracy_score(y_te, preds)),
                    "logloss":  float(log_loss(y_te, proba))}
        else:
            ps = booster.predict(X_te)
            p  = sy.inverse_transform(ps.reshape(-1, 1)).ravel() if sy else ps
            return {"rmse": float(np.sqrt(mean_squared_error(y_te, p))),
                    "mae":  float(mean_absolute_error(y_te, p)),
                    "r2":   float(r2_score(y_te, p))}
    except Exception as exc:
        logging.getLogger("optuna.benchmark").error("Test eval failed: %s", exc)
        if is_clf: return {"auc": float("nan"), "accuracy": float("nan"), "logloss": float("nan")}
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

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
# Run Optuna on one dataset
# =============================================================================

def run_optuna_on_dataset(X: NDArray, y: NDArray, name: str, task: str,
                          logger: logging.Logger) -> Dict:
    import lightgbm as lgb

    is_clf    = (task == "classification")
    n_classes = int(np.max(y) + 1) if is_clf else 0
    tag       = f"Optuna-{OPTUNA_TRIALS}"

    logger.info("")
    logger.info("  ┌─ %s  [%s]", name, tag)
    logger.info("  │  task=%s  trials=%d/fold  sampler=TPE", task, OPTUNA_TRIALS)

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
        sy   = None; y_tr = y_train.copy()
        if not is_clf:
            sy   = StandardScaler()
            y_tr = sy.fit_transform(y_train.reshape(-1, 1)).ravel()

        objective_lgb = "binary" if n_classes == 2 else ("multiclass" if is_clf else "regression")
        metric_lgb    = "binary_logloss" if n_classes == 2 else ("multi_logloss" if is_clf else "rmse")
        num_class     = {"num_class": n_classes} if (is_clf and n_classes > 2) else {}

        rng_val = np.random.default_rng(SEED + fold_idx)
        n_val   = max(1, int(len(X_tr) * 0.2))
        idx_val = rng_val.permutation(len(X_tr))
        Xv, yv  = X_tr[idx_val[:n_val]], y_tr[idx_val[:n_val]]
        Xh, yh  = X_tr[idx_val[n_val:]], y_tr[idx_val[n_val:]]

        train_ds = lgb.Dataset(Xh, label=yh)
        val_ds   = lgb.Dataset(Xv, label=yv, reference=train_ds)

        eval_log : List[float] = []
        conv_best: List[float] = []

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            lgb_params = {**params, "objective": objective_lgb, "metric": metric_lgb,
                          "verbosity": -1, "seed": SEED + fold_idx,
                          "num_threads": 2, **num_class}
            try:
                cb = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                      lgb.log_evaluation(-1)]
                booster = lgb.train(lgb_params, train_ds,
                                    num_boost_round=N_ESTIMATORS_FIXED,
                                    valid_sets=[val_ds], callbacks=cb)
                if is_clf:
                    if n_classes == 2:
                        p = booster.predict(Xv).reshape(-1, 1)
                        p = np.hstack([1 - p, p])
                    else:
                        p = booster.predict(Xv)
                    score = float(log_loss(yv, p))
                else:
                    score = float(np.sqrt(mean_squared_error(yv, booster.predict(Xv))))
            except Exception:
                score = 1e6

            eval_log.append(score)
            conv_best.append(min(eval_log))
            return score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED + fold_idx),
        )
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best_trial  = study.best_trial
        best_val    = best_trial.value
        best_params = suggest_params(best_trial)

        metrics  = evaluate_lgb(best_params, X_tr, y_tr, X_te, y_test,
                                 task, n_classes, sy)
        fold_sec = time.perf_counter() - t_fold
        eta_str  = eta.tick()

        pm_key = "auc" if is_clf else "rmse"
        logger.info("  │  [%s] Fold %d/%d | %s=%.4f | best_val=%.4f | trials=%d | %.1fs | ETA: %s",
                    tag, fold_idx, N_OUTER_FOLDS,
                    pm_key.upper(), metrics.get(pm_key, float("nan")),
                    best_val, OPTUNA_TRIALS, fold_sec, eta_str)

        fold_results.append({
            "fold": fold_idx, "best_val": best_val,
            "n_trials": OPTUNA_TRIALS, "duration_sec": round(fold_sec, 2),
            "best_params": best_params, "convergence": conv_best, **metrics,
        })

    total_sec = time.perf_counter() - t_wall

    def _agg(k):
        v = [r[k] for r in fold_results if not np.isnan(r.get(k, float("nan")))]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    if is_clf:
        am, as_ = _agg("auc");  acm, acs = _agg("accuracy"); llm, lls = _agg("logloss")
        ms = {"auc_mean": am, "auc_std": as_, "accuracy_mean": acm, "accuracy_std": acs,
              "logloss_mean": llm, "logloss_std": lls}
        pm_mean, pm_std = am, as_
    else:
        rm, rs = _agg("rmse"); mm, ms_ = _agg("mae"); r2m, r2s = _agg("r2")
        ms = {"rmse_mean": rm, "rmse_std": rs, "mae_mean": mm, "mae_std": ms_,
              "r2_mean": r2m, "r2_std": r2s}
        pm_mean, pm_std = rm, rs

    ml = "AUC" if is_clf else "RMSE"
    logger.info("  └─ [%s] %s | %s=%.4f±%.4f | %.1f min",
                tag, name, ml, pm_mean, pm_std, total_sec / 60)

    return {"dataset": name, "task": task, "method": tag,
            "total_evals": OPTUNA_TRIALS, "wall_sec": round(total_sec, 1),
            "fold_results": fold_results, **ms}

# =============================================================================
# Results Persistence
# =============================================================================

def save_results(all_summaries: List[Dict], logger: logging.Logger) -> Path:
    out_dir  = _HERE / "results" / "tabarena"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"optuna_lgb_all16_{RUN_ID}.csv"

    rows = []
    for s in all_summaries:
        for r in s["fold_results"]:
            row: Dict[str, Any] = {
                "run_id": RUN_ID, "dataset": s["dataset"], "task": s["task"],
                "method": s["method"], "total_evals": s["total_evals"],
                "fold": r["fold"], "seed": SEED,
                "best_val": r["best_val"],
                "n_trials": r.get("n_trials", OPTUNA_TRIALS),
                "duration_sec": r["duration_sec"],
            }
            for key in ("rmse", "mae", "r2", "auc", "accuracy", "logloss"):
                if key in r: row[key] = r[key]
            row.update({f"param_{k}": v for k, v in r["best_params"].items()
                        if k not in ("bagging_freq", "feature_pre_filter")})
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
    for task in ("regression", "classification"):
        subs = [s for s in all_summaries if s["task"] == task]
        if not subs: continue
        primary = "rmse_mean" if task == "regression" else "auc_mean"
        pm_lbl  = "RMSE ↓"   if task == "regression" else "AUC ↑"

        logger.info("")
        logger.info("=" * 100)
        logger.info("  %s  ─  Optuna-200 vs TabArena LightGBM Baselines  |  %s",
                    task.upper(), pm_lbl)
        logger.info("=" * 100)
        logger.info("  %-38s  %-18s  %-16s  %-16s  %-16s",
                    "Dataset", "Optuna-200", "LGB-Default", "LGB-Tuned", "LGB-Ens")
        logger.info("  " + "─" * 90)

        for s in sorted(subs, key=lambda x: x["dataset"]):
            bl  = TABARENA_LGB_BASELINES.get(s["dataset"])
            pm  = s[primary]; std = s[primary.replace("_mean", "_std")]
            row = f"  {s['dataset']:<38}  {pm:.4f}±{std:.4f}  "
            if bl:
                row += (f"{bl['default'][0]:.4f}±{bl['default'][1]:.4f}  "
                        f"{bl['tuned'][0]:.4f}±{bl['tuned'][1]:.4f}  "
                        f"{bl['ens'][0]:.4f}±{bl['ens'][1]:.4f}")
                ref   = bl["tuned"][0]
                delta = (pm - ref) / (abs(ref) + 1e-10) * 100
                sign  = ("+" if (delta > 0) == (task == "regression") else
                         ("≈" if abs(delta) < (1.0 if task == "regression" else 0.5) else "-"))
                row += f"  {sign}({delta:+.1f}%)"
            logger.info(row)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger = setup_logging(RUN_ID)
    logger.info("Configuration:")
    logger.info("  sampler=TPE  trials=%d/fold  outer_folds=%d  seed=%d",
                OPTUNA_TRIALS, N_OUTER_FOLDS, SEED)
    logger.info("  search_space=TabArena C.3 LightGBM (12D)")
    logger.info("  task_filter=%s  datasets=%s (%d total)",
                TASK_FILTER, TARGET_DIDS, len(TARGET_DIDS))

    all_summaries: List[Dict] = []
    failed       : List[str]  = []
    t_total = time.perf_counter()

    for did in TARGET_DIDS:
        ds_name = ALL_DATASETS.get(did, f"openml_{did}")
        logger.info("")
        logger.info("▓" * 70)
        logger.info("  DATASET: %s  (DID=%d)", ds_name, did)
        logger.info("▓" * 70)

        try:
            X, y, name, task = load_dataset(did, logger)
        except Exception as exc:
            logger.error("Load failed DID=%d: %s — SKIPPING", did, exc)
            failed.append(f"load:{ds_name}"); continue

        try:
            summary = run_optuna_on_dataset(X, y, name, task, logger)
            all_summaries.append(summary)
            save_results(all_summaries, logger)
        except Exception as exc:
            logger.error("Optuna failed on %s: %s", ds_name, exc)
            import traceback; traceback.print_exc()
            failed.append(ds_name)

    total_wall = time.perf_counter() - t_total

    if all_summaries:
        print_comparison_report(all_summaries, logger)
        out_path = save_results(all_summaries, logger)
    else:
        out_path = None

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("Datasets completed : %d / %d", len(all_summaries), len(TARGET_DIDS))
    logger.info("Failed             : %d  %s", len(failed), failed or "")
    logger.info("Total wall time    : %s  (%.1f hr)",
                str(timedelta(seconds=int(total_wall))), total_wall / 3600)
    logger.info("Results            : %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
