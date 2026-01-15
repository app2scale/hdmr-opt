"""
Forecasting Hyperparameter Optimization Pipeline (HDMR) - Production Ready

This pipeline runs `src/forecast_example.py` (as a module) across a grid of:
- algorithms: xgboost, lightgbm, arima, ets
- metrics: mape, rmse, mae

It captures stdout/stderr for each run, writes per-run logs, and produces a
JSON summary + console table.

Repo layout assumed:
  ./src/forecast_example.py

Usage:
  python forecast_pipeline.py

Environment overrides (optional):
  PYTHON_BIN=python
  RUN_ID=20260113_120000
  SAMPLES=1000
  BASIS=Cosine
  DEGREE=7
  ADAPTIVE=1            # 1 or 0
  MAXITER=25
  NUM_CLOSEST=100
  EPSILON=0.1
  CLIP=0.9
  SEED=42
  TIMEOUT_SEC=1200
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")

ALGORITHMS = ["xgboost", "lightgbm", "arima", "ets"]
METRICS = ["mape", "rmse", "mae"]

RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d_%H%M%S"))

SAMPLES = os.environ.get("SAMPLES", "1000")
BASIS = os.environ.get("BASIS", "Cosine")
DEGREE = os.environ.get("DEGREE", "7")

ADAPTIVE = os.environ.get("ADAPTIVE", "1") == "1"
MAXITER = os.environ.get("MAXITER", "25")
NUM_CLOSEST = os.environ.get("NUM_CLOSEST", "100")
EPSILON = os.environ.get("EPSILON", "0.1")
CLIP = os.environ.get("CLIP", "0.9")

SEED = os.environ.get("SEED", "42")
TIMEOUT_SEC = int(os.environ.get("TIMEOUT_SEC", "1200"))  # 20 minutes

# Output directories
BASE_DIR = Path("results/forecasting") / RUN_ID
LOG_DIR = BASE_DIR / "logs"
BASE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = BASE_DIR / "optimization_summary.json"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class RunResult:
    algorithm: str
    metric: str
    status: str  # success | failed | timeout | error
    returncode: Optional[int]
    runtime_sec: Optional[float]
    log_file: str
    cmd: List[str]
    error_preview: Optional[str] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_repo_root() -> None:
    """
    Ensure the script is executed from repository root.
    """
    if not Path("src").is_dir():
        raise RuntimeError("Missing ./src directory. Run this script from repository root.")
    if not Path("src/forecast_example.py").is_file():
        raise RuntimeError("Missing ./src/forecast_example.py. Repo layout mismatch.")


def build_cmd(algo: str, metric: str) -> List[str]:
    """
    Build the command to execute forecasting optimization for a given algorithm and metric.
    We run the example as a module to avoid import/path issues.
    """
    cmd = [
        PYTHON_BIN, "-m", "src.forecast_example",
        "--algorithm", algo,
        "--metric", metric,
        "--samples", str(SAMPLES),
        "--basis", str(BASIS),
        "--degree", str(DEGREE),
        "--split", "2020-01-01",
        "--seed", str(SEED),
    ]

    if ADAPTIVE:
        cmd += [
            "--adaptive",
            "--maxiter", str(MAXITER),
            "--numClosestPoints", str(NUM_CLOSEST),
            "--epsilon", str(EPSILON),
            "--clip", str(CLIP),
        ]

    # quiet output is optional; keep it OFF by default for debugging
    # cmd += ["--quiet"]

    return cmd


def write_log(log_file: Path, stdout: str, stderr: str, cmd: List[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(stdout or "")
        f.write("\n\nSTDERR:\n")
        f.write(stderr or "")


def run_one(algo: str, metric: str) -> RunResult:
    cmd = build_cmd(algo, metric)
    log_file = LOG_DIR / f"{algo}_{metric}.log"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
        dt = time.time() - t0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        write_log(log_file, stdout, stderr, cmd)

        if proc.returncode == 0:
            return RunResult(
                algorithm=algo,
                metric=metric,
                status="success",
                returncode=proc.returncode,
                runtime_sec=round(dt, 4),
                log_file=str(log_file),
                cmd=cmd,
            )

        # Failed
        combined = (stderr or stdout).strip().splitlines()
        preview = "\n".join(combined[:10]) if combined else "Unknown error"

        return RunResult(
            algorithm=algo,
            metric=metric,
            status="failed",
            returncode=proc.returncode,
            runtime_sec=round(dt, 4),
            log_file=str(log_file),
            cmd=cmd,
            error_preview=preview,
        )

    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        write_log(log_file, "", f"TIMEOUT after {TIMEOUT_SEC} seconds", cmd)
        return RunResult(
            algorithm=algo,
            metric=metric,
            status="timeout",
            returncode=None,
            runtime_sec=round(dt, 4),
            log_file=str(log_file),
            cmd=cmd,
            error_preview=f"Timeout after {TIMEOUT_SEC}s",
        )

    except Exception as e:
        dt = time.time() - t0
        write_log(log_file, "", f"ERROR: {e}", cmd)
        return RunResult(
            algorithm=algo,
            metric=metric,
            status="error",
            returncode=None,
            runtime_sec=round(dt, 4),
            log_file=str(log_file),
            cmd=cmd,
            error_preview=str(e),
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_repo_root()

    print("=" * 70)
    print("FORECASTING HYPERPARAMETER OPTIMIZATION PIPELINE (HDMR)")
    print("=" * 70)
    print(f"Run ID:   {RUN_ID}")
    print(f"Output:   {BASE_DIR}")
    print(f"Samples:  {SAMPLES}")
    print(f"Basis:    {BASIS} (degree={DEGREE})")
    print(f"Adaptive: {ADAPTIVE}")
    if ADAPTIVE:
        print(f"  maxiter={MAXITER}  k={NUM_CLOSEST}  eps={EPSILON}  clip={CLIP}")
    print(f"Timeout:  {TIMEOUT_SEC}s")
    print()

    results: List[RunResult] = []

    for algo in ALGORITHMS:
        for metric in METRICS:
            print(f"{algo.upper():8s} - Optimizing {metric.upper():4s}", end=" ... ", flush=True)

            r = run_one(algo, metric)
            results.append(r)

            if r.status == "success":
                print(f"OK ({r.runtime_sec}s)")
            else:
                print(f"{r.status.upper()} ({r.runtime_sec}s)")
                if r.error_preview:
                    print("  Error preview:")
                    for line in r.error_preview.splitlines()[:6]:
                        print(f"    {line}")
                print(f"  Log: {r.log_file}")

    # Write JSON summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in results], f, indent=2, ensure_ascii=False)

    # Console summary table
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    df = pd.DataFrame([asdict(x) for x in results])
    cols = ["algorithm", "metric", "status", "returncode", "runtime_sec", "log_file"]
    print(df[cols].to_string(index=False))

    print("\nSaved:")
    print(f"  Summary JSON: {SUMMARY_FILE}")
    print(f"  Logs dir    : {LOG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

