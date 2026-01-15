"""
High-dimensional (10D) HDMR benchmark runner (production-ready).

Key features:
- Uses `python -m src.main` to avoid import/path issues.
- Automatically injects an appropriate `--x0` based on the function:
    - rosenbrock_10d -> ones(10)
    - griewank_10d   -> 100s(10)
    - rastrigin_10d  -> zeros(10)
- Captures stdout + stderr.
- Records returncode, runtime, key result lines, and the exact command used.
- Writes JSON summary to results/high_dim_tests/summary.json

Usage:
  python high_dim_test.py

Optional env overrides:
  PYTHON_BIN=python
"""

# Path fix for reorganized structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Configuration
# -----------------------------
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")

CONFIGS: List[Dict[str, Any]] = [
    {"name": "Rosenbrock 10D - Standard", "function": "rosenbrock_10d", "samples": 2000, "adaptive": False},
    {"name": "Rosenbrock 10D - Adaptive", "function": "rosenbrock_10d", "samples": 2000, "adaptive": True},
    {"name": "Rastrigin 10D - Standard", "function": "rastrigin_10d", "samples": 2000, "adaptive": False},
    {"name": "Rastrigin 10D - Adaptive", "function": "rastrigin_10d", "samples": 2000, "adaptive": True},
    {"name": "Griewank 10D - Standard", "function": "griewank_10d", "samples": 3000, "adaptive": False},
    {"name": "Griewank 10D - Adaptive", "function": "griewank_10d", "samples": 3000, "adaptive": True},
]

RESULTS_DIR = Path("results/high_dim_tests")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "summary.json"


# -----------------------------
# x0 policy (function -> x0)
# -----------------------------
def get_default_x0(function_name: str, n: int = 10) -> Optional[List[float]]:
    """
    Return an appropriate default x0 for a given function, or None if not defined.
    """
    fn = function_name.lower().strip()

    if fn == "rosenbrock_10d":
        return [1.0] * n

    if fn == "griewank_10d":
        return [100.0] * n

    if fn == "rastrigin_10d":
        return [0.0] * n

    # Unknown function -> do not inject x0 automatically
    return None


# -----------------------------
# Helpers
# -----------------------------
def ensure_repo_root() -> None:
    if not Path("src").is_dir() or not Path("src/main.py").is_file():
        raise RuntimeError("Run this script from repository root (missing ./src or ./src/main.py).")


def extract_key_lines(stdout: str) -> List[str]:
    """
    Best-effort extraction of useful lines from CLI output.
    Adjust patterns if your CLI output changes.
    """
    patterns = [
        r"^Runtime:\s+.*$",
        r"^Solution:\s+.*$",
        r"^Objective:\s+.*$",
        r"^Mean f\(x\*\):\s+.*$",
        r"^Std\s+f\(x\*\):\s+.*$",
        r"^Best f\(x\*\):\s+.*$",
        r"^Worst f\(x\*\):\s+.*$",
        r"^Run \d+/\d+:\s+x\*.*$",
    ]
    rx = re.compile("|".join(patterns), re.MULTILINE)
    return [m.group(0).strip() for m in rx.finditer(stdout)]


def run_one(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one benchmark configuration via subprocess and capture outputs.
    Inject --x0 automatically based on function name.
    """
    function_name = config["function"]
    n = 10

    x0 = get_default_x0(function_name, n=n)

    cmd: List[str] = [
        PYTHON_BIN, "-m", "src.main",
        "--numSamples", str(config["samples"]),
        "--numVariables", str(n),
        "--function", function_name,
        "--basisFunction", "Cosine",
        "--degree", "7",
        "--numberOfRuns", "3",
        "--seed", "42",
        "--disp",
        "--noPlots",
    ]

    # Inject x0 if available (space-separated values)
    if x0 is not None:
        cmd.extend(["--x0", *[str(v) for v in x0]])

    if config["adaptive"]:
        cmd.extend([
            "--adaptive",
            "--maxiter", "30",
            "--numClosestPoints", "200",
            "--epsilon", "0.05",
            "--clip", "0.9",
        ])

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(".").resolve()),
        env={**os.environ},
    )
    dt = time.time() - t0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    success = (proc.returncode == 0)

    error_hint: Optional[str] = None
    if not success:
        error_hint = (stderr.strip().splitlines()[-1] if stderr.strip() else "Unknown error")

    return {
        "config": config,
        "auto_x0": x0,                 # <-- important: record what we injected
        "success": success,
        "returncode": proc.returncode,
        "runtime_sec": round(dt, 4),
        "stdout": stdout,
        "stderr": stderr,
        "key_lines": extract_key_lines(stdout),
        "error_hint": error_hint,
        "cmd": cmd,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_repo_root()

    print("=" * 70)
    print("HIGH-DIMENSIONAL HDMR OPTIMIZATION TESTS")
    print("=" * 70)
    print()

    all_results: List[Dict[str, Any]] = []

    for cfg in CONFIGS:
        print(f"\n{cfg['name']}")
        print("-" * 70)

        r = run_one(cfg)
        all_results.append(r)

        if r["success"]:
            x0_str = "None" if r["auto_x0"] is None else f"len={len(r['auto_x0'])}, first={r['auto_x0'][0]}"
            print(f"✓ OK (rc=0, {r['runtime_sec']}s, x0={x0_str})")
            if r["key_lines"]:
                for line in r["key_lines"][-6:]:
                    print(f"  {line}")
        else:
            print(f"✗ FAIL (rc={r['returncode']}, {r['runtime_sec']}s)")
            if r["error_hint"]:
                print(f"  Last error: {r['error_hint']}")
            if r["stderr"].strip():
                preview = "\n".join(r["stderr"].strip().splitlines()[:12])
                print("  stderr preview:")
                print("  " + preview.replace("\n", "\n  "))

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"All tests completed. Summary saved to: {SUMMARY_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

