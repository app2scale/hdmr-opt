#!/usr/bin/env python3
"""
results_compiler.py — Publication Results Compiler for HDMR Optimization
=========================================================================
v2.0  |  2026-02-03  |  APP2SCALE Research Group

Düzeltilen kritik sorunlar (v1 → v2)
-------------------------------------
1. Improvement hesabında abs() kullanılıyordu.
   Six-Hump Camel (f* = -1.0316) gibi negatif optimalarda bu
   tamamen yanlış sonuç üretiyordu.  → Düzeltme: signed gap-to-optimum.

2. Six-Hump Camel f≈0 "optimal" olarak raporlandı.
   f(0,0)=0 bir saddle point; gerçek minimum = -1.0316.
   → Her fonksiyon için analitik f* ile otomatik validation.

3. Branin "89.46% improvement" aldatıcıydı.
   İki kötü sonuç birbirine karşı karşılaştırılıyordu.
   → Gap-to-optimum ile mutlak mesafe raporlanır.

4. Rosenbrock f=1.0 bağlam olmadan verilmişti.
   f(0,0)=1.0; gerçek min f(1,1)=0.  → Otomatik flag mekanizması.

5. 10D sonuçlar (Rastrigin, Rosenbrock, Griewank) hiç dahil edilmemişti.
   → Tam pipeline: 2D + 10D + forecasting + comparison + sensitivity.

Mimare
------
  GroundTruth          – Analitik f* değerleri, domain sınırları
  BenchmarkResult      – Per-function dataclass: runs, gaps, flags
  Parsers              – txt / csv / json → structured data
  ResultsCompiler      – Orchestrator: compile → validate → export
  LaTeX               – booktabs-uyumlu, bold-best, footnote-flag tablo generatörü
  SummaryReport        – Validation audit log + publication-ready summary

Kullanım
--------
    python results_compiler.py                          # ./results/ varsayılan
    python results_compiler.py --results /path/to/results --output ./pub_out
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# 1. GROUND TRUTH — analitik f* ve domain
# ============================================================================

# f(x*) — bilinen global minimumlar.  Kaynaklar: optimum_points.json + litaratür.
TRUE_F_STAR: Dict[str, float] = {
    "testfunc_2d":     0.0,
    "ackley_2d":       0.0,
    "rastrigin_2d":    0.0,
    "rastrigin_10d":   0.0,
    "camel3_2d":       0.0,
    "camel16_2d":     -1.0316,        # ← negatif; eskiden abs() ile maskeleniyordu
    "treccani_2d":     0.0,
    "goldstein_2d":    3.0,
    "branin_2d":       0.397887,
    "rosenbrock_2d":   0.0,           # scaled: 0.5*(x1²−x2)² + (x1−1)²
    "rosenbrock_10d":  0.0,           # standard Rosenbrock
    "griewank_10d":    0.0,
}

# Yayın tablolarında kullanılacak insan-okunur isimler
DISPLAY_NAMES: Dict[str, str] = {
    "ackley_2d":      "Ackley",
    "branin_2d":      "Branin",
    "camel16_2d":     "Six-Hump Camel",
    "rastrigin_2d":   "Rastrigin",
    "rosenbrock_2d":  "Rosenbrock",
    "rastrigin_10d":  "Rastrigin-10D",
    "rosenbrock_10d": "Rosenbrock-10D",
    "griewank_10d":   "Griewank-10D",
}

# Optimum noktalarında interaction mi var?  First-order HDMR limitation flaggı.
INTERACTION_FLAG: Dict[str, Optional[str]] = {
    "ackley_2d":      None,                          # separable → OK
    "rastrigin_2d":   None,                          # separable → OK
    "rastrigin_10d":  None,                          # separable → OK
    "camel16_2d":     "x₁x₂ cross-term",
    "rosenbrock_2d":  "x₁²−x₂ coupling",
    "rosenbrock_10d": "xᵢ²−xᵢ₊₁ coupling",
    "branin_2d":      "x₁²−x₂ interaction",
    "griewank_10d":   "∏cos(…) product term",
}

# Tolerance: |f_found − f*| < TOL → "optimal" kabul edilir
OPTIMALITY_TOL = 0.05


# ============================================================================
# 2. DATACLASSES
# ============================================================================

@dataclass
class SingleRun:
    fun:    float
    nfev:   int
    nit:    int
    x:      Optional[List[float]] = None
    success: bool = True


@dataclass
class BenchmarkResult:
    """Tek bir fonksiyon + mode (standard / adaptive) için tüm runs."""
    function_key: str
    mode:         str                          # "standard" | "adaptive"
    runs:         List[SingleRun] = field(default_factory=list)

    # ---------- aggregate stats ----------
    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def fun_values(self) -> List[float]:
        return [r.fun for r in self.runs]

    @property
    def best_fun(self) -> Optional[float]:
        return min(self.fun_values) if self.runs else None

    @property
    def mean_fun(self) -> Optional[float]:
        return float(np.mean(self.fun_values)) if self.runs else None

    @property
    def std_fun(self) -> Optional[float]:
        return float(np.std(self.fun_values, ddof=0)) if self.runs else None

    @property
    def mean_nfev(self) -> Optional[float]:
        return float(np.mean([r.nfev for r in self.runs])) if self.runs else None

    # ---------- gap-to-optimum (SIGNED) ----------
    @property
    def f_star(self) -> float:
        return TRUE_F_STAR.get(self.function_key, 0.0)

    @property
    def best_gap(self) -> Optional[float]:
        """best_fun − f*   ( ≥ 0 ideally; negatif → numerical noise )"""
        return (self.best_fun - self.f_star) if self.best_fun is not None else None

    @property
    def mean_gap(self) -> Optional[float]:
        return (self.mean_fun - self.f_star) if self.mean_fun is not None else None

    @property
    def reached_optimum(self) -> bool:
        if self.best_gap is None:
            return False
        return abs(self.best_gap) < OPTIMALITY_TOL

    @property
    def interaction_note(self) -> Optional[str]:
        return INTERACTION_FLAG.get(self.function_key)


# ============================================================================
# 3. PARSERS
# ============================================================================

def _parse_x_array(text: str) -> Optional[List[float]]:
    """array([1.2, 3.4, ...]) → [1.2, 3.4]  (newline-safe)"""
    m = re.search(r"array\(\[\s*(.*?)\s*\]\)", text, re.DOTALL)
    if not m:
        return None
    try:
        return [float(v.strip()) for v in m.group(1).split(",") if v.strip()]
    except ValueError:
        return None


def parse_optimizeresult_block(block: str) -> Optional[SingleRun]:
    """Bir OptimizeResult str → SingleRun."""
    fun_m  = re.search(r"fun:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", block)
    nfev_m = re.search(r"nfev:\s*(\d+)", block)
    nit_m  = re.search(r"nit:\s*(\d+)", block)
    suc_m  = re.search(r"success:\s*(True|False)", block)

    if not fun_m:
        return None

    return SingleRun(
        fun=float(fun_m.group(1)),
        nfev=int(nfev_m.group(1)) if nfev_m else 0,
        nit=int(nit_m.group(1))   if nit_m  else 1,
        x=_parse_x_array(block),
        success=(suc_m.group(1) == "True") if suc_m else True,
    )


def parse_benchmark_txt(filepath: Path) -> List[SingleRun]:
    """results/*.txt  →  List[SingleRun]  (her Run N: bloğu ayrı)"""
    runs: List[SingleRun] = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        # "Run 1:" ... "Run 2:" ... ile böl
        blocks = re.split(r"Run\s+\d+\s*:", content)
        for blk in blocks[1:]:          # ilk parça header
            sr = parse_optimizeresult_block(blk)
            if sr:
                runs.append(sr)
    except Exception as exc:
        print(f"    [WARN] {filepath.name}: {exc}")
    return runs


def parse_summary_txt(filepath: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """benchmark_2d/*/summary.txt  →  {func: {mode: {mean, best, std, …}}}

    Format assumption (flexible): bloklar '>>>' ile ayrılmış,
    her blokta 'funcname [mode]' header + key: value satırları.
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r">>>", content)
        for sec in sections[1:]:
            hdr = re.match(r"\s*([\w]+)\s*\[([\w]+)\]", sec)
            if not hdr:
                continue
            func, mode = hdr.group(1), hdr.group(2)
            data: Dict[str, float] = {}
            for tag, key in [
                (r"Mean\s+f\(x\*\):\s*", "mean"),
                (r"Best\s+f\(x\*\):\s*", "best"),
                (r"Std\s+f\(x\*\):\s*",  "std"),
                (r"nfev\s+mean:\s*",      "nfev_mean"),
            ]:
                m = re.search(tag + r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", sec)
                if m:
                    data[key] = float(m.group(1))
            result.setdefault(func, {})[mode] = data
    except Exception as exc:
        print(f"    [WARN] summary.txt: {exc}")
    return result


def parse_forecasting_txt(filepath: Path) -> Dict[str, Any]:
    """forecasting/forecast_*_hdmr_*.txt  →  structured dict."""
    out: Dict[str, Any] = {}
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")

        # --- metadata ---
        for key, pat in [
            ("algorithm",         r"Algorithm:\s*(\w+)"),
            ("metric",            r"Metric:\s*(\w+)"),
            ("optimization_time", r"Optimization Time:\s*([\d.]+)"),
            ("iterations",        r"Iterations:\s*(\d+)"),
            ("function_evals",    r"Function Evals:\s*(\d+)"),
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                val = m.group(1)
                if key == "optimization_time":
                    out[key] = float(val)
                elif key in ("iterations", "function_evals"):
                    out[key] = int(val)
                else:
                    out[key] = val

        # --- hyperparameters ---
        hp_sec = re.search(
            r"Optimal Hyperparameters:(.*?)Test Set Performance:",
            content, re.DOTALL
        )
        if hp_sec:
            out["hyperparameters"] = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r"(\w+)\s*=\s*([\d.]+)", hp_sec.group(1))
            }

        # --- performance metrics ---
        out["performance"] = {}
        for metric in ("MSE", "RMSE", "MAE", "MAPE", "SMAPE", "MASE"):
            m = re.search(rf"{metric}\s*=\s*([\d.]+)", content)
            if m:
                out["performance"][metric] = float(m.group(1))

    except Exception as exc:
        print(f"    [WARN] {filepath.name}: {exc}")
    return out


def load_json(filepath: Path) -> Dict:
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"    [WARN] {filepath}: {exc}")
        return {}


# ============================================================================
# 4. RESULTS COMPILER
# ============================================================================

# Dosya adları (results/ altında)
_BENCHMARK_REGISTRY: Dict[str, Tuple[str, str]] = {
    # function_key : (standard_txt, adaptive_txt)
    "ackley_2d":      ("ackley_2d_a-30.0_b30.0_N1000_m7.txt",
                       "adaptive_ackley_2d_a-30.0_b30.0_N1000_m7_k100_c0.90.txt"),
    "branin_2d":      ("branin_2d_a-5.0_b15.0_N1000_m7.txt",
                       "adaptive_branin_2d_a-5.0_b15.0_N1000_m7_k100_c0.90.txt"),
    "camel16_2d":     ("camel16_2d_a-5.0_b5.0_N1000_m7.txt",
                       "adaptive_camel16_2d_a-5.0_b5.0_N1000_m7_k100_c0.90.txt"),
    "rastrigin_2d":   ("rastrigin_2d_a-5.12_b5.12_N1000_m7.txt",
                       "adaptive_rastrigin_2d_a-5.12_b5.12_N1000_m7_k100_c0.90.txt"),
    "rosenbrock_2d":  ("rosenbrock_2d_a-2.048_b2.048_N1000_m7.txt",
                       "adaptive_rosenbrock_2d_a-2.048_b2.048_N1000_m7_k100_c0.90.txt"),
    # --- 10D ---
    "rastrigin_10d":  ("rastrigin_10d_a-5.12_b5.12_N2000_m7.txt",
                       "adaptive_rastrigin_10d_a-5.12_b5.12_N2000_m7_k200_c0.90.txt"),
    "rosenbrock_10d": ("rosenbrock_10d_a-5.0_b10.0_N2000_m7.txt",
                       "adaptive_rosenbrock_10d_a-5.0_b10.0_N2000_m7_k200_c0.90.txt"),
    "griewank_10d":   ("griewank_10d_a-600.0_b600.0_N3000_m7.txt",
                       "adaptive_griewank_10d_a-600.0_b600.0_N3000_m7_k200_c0.90.txt"),
}

_FORECAST_MODELS  = ("arima", "ets", "lightgbm", "xgboost")
_FORECAST_METRICS = ("mae", "mape", "rmse")


class ResultsCompiler:
    """Tüm sonuç kaynaklarını yükler, doğrular ve export ediler."""

    def __init__(self, results_dir: str):
        self.root = Path(results_dir)
        # İç storage
        self._benchmarks: Dict[str, Dict[str, BenchmarkResult]] = {}   # func → mode → BR
        self._forecasts:  Dict[Tuple[str, str], Dict[str, Any]]  = {}  # (model, metric) → parsed
        self._comparison: pd.DataFrame = pd.DataFrame()
        self._sensitivity_df: pd.DataFrame = pd.DataFrame()
        self._sensitivity_report: str = ""

    # ------------------------------------------------------------------
    # 4a.  BENCHMARK compilation
    # ------------------------------------------------------------------

    def compile_benchmarks(self) -> None:
        print("\n  [benchmark] individual txt files …")
        for func_key, (std_file, adp_file) in _BENCHMARK_REGISTRY.items():
            self._benchmarks[func_key] = {}
            for mode, fname in (("standard", std_file), ("adaptive", adp_file)):
                fpath = self.root / fname
                if not fpath.exists():
                    print(f"    [MISS] {fname}")
                    continue
                runs = parse_benchmark_txt(fpath)
                br = BenchmarkResult(function_key=func_key, mode=mode, runs=runs)
                self._benchmarks[func_key][mode] = br
                print(f"    [OK]   {fname}  →  {br.n_runs} run(s), best f = {br.best_fun}")

        # summary.txt cross-check (bilgi amacı)
        summaries = sorted(self.root.glob("benchmark_2d/*/summary.txt"))
        if summaries:
            summary_data = parse_summary_txt(summaries[-1])
            print(f"  [benchmark] summary.txt cross-ref:  {list(summary_data.keys())}")

        # benchmarks/summary_*.csv (newer format — varsa ek cross-check)
        sum_csvs = sorted(self.root.glob("benchmarks/summary_*.csv"))
        if sum_csvs:
            try:
                latest_csv = pd.read_csv(sum_csvs[-1])
                print(f"  [benchmark] benchmarks/summary csv:  {latest_csv.shape}")
            except Exception:
                pass

    def benchmark_table(self) -> pd.DataFrame:
        """Publication-ready DataFrame üretir.

        Sütunlar:
            Function | Dim | f* (true) | Interaction? |
            Std Best | Std Gap | Std Runs |
            Adp Best | Adp Gap | Adp Runs |
            Gap Reduction | Status
        """
        rows = []
        # İstenen sıra
        order = [
            "ackley_2d", "rastrigin_2d", "camel16_2d",
            "branin_2d", "rosenbrock_2d",
            "rastrigin_10d", "rosenbrock_10d", "griewank_10d",
        ]
        for fk in order:
            if fk not in self._benchmarks:
                continue
            dim = int(re.search(r"(\d+)d", fk).group(1))  # type: ignore[union-attr]
            f_star = TRUE_F_STAR.get(fk, 0.0)
            std = self._benchmarks[fk].get("standard")
            adp = self._benchmarks[fk].get("adaptive")
            interaction = INTERACTION_FLAG.get(fk)

            row: Dict[str, Any] = {
                "Function":      DISPLAY_NAMES.get(fk, fk),
                "Dim":           dim,
                "f* (true)":     f_star,
                "Interaction":   interaction if interaction else "—",
            }

            # --- standard ---
            if std and std.runs:
                row["Std Best"]  = std.best_fun
                row["Std Gap"]   = std.best_gap
                row["Std Runs"]  = std.n_runs
            # --- adaptive ---
            if adp and adp.runs:
                row["Adp Best"]  = adp.best_fun
                row["Adp Gap"]   = adp.best_gap
                row["Adp Runs"]  = adp.n_runs

            # --- improvement = gap_std − gap_adp  (positif → adaptive daha iyi) ---
            if std and adp and std.best_gap is not None and adp.best_gap is not None:
                row["Gap Δ"] = round(std.best_gap - adp.best_gap, 6)

            # --- status: combined flag ---
            parts = []
            if std and not std.reached_optimum:
                parts.append("Std✗")
            if adp and not adp.reached_optimum:
                parts.append("Adp✗")
            row["Status"] = ", ".join(parts) if parts else "✓ Optimal"

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4b.  FORECASTING compilation
    # ------------------------------------------------------------------

    def compile_forecasting(self) -> None:
        print("\n  [forecast] parsing txt files …")
        fdir = self.root / "forecasting"
        for model in _FORECAST_MODELS:
            for metric in _FORECAST_METRICS:
                fname = f"forecast_{model}_{metric}_hdmr_N200_m7.txt"
                fpath = fdir / fname
                if not fpath.exists():
                    continue
                parsed = parse_forecasting_txt(fpath)
                self._forecasts[(model.upper(), metric.upper())] = parsed
                print(f"    [OK]   {fname}")

        # optimization_summary.json (üst seviye)
        json_candidates = sorted(fdir.glob("*/optimization_summary.json"))
        if json_candidates:
            self._forecast_summary_json = load_json(json_candidates[-1])
            print(f"  [forecast] optimization_summary.json loaded")

    def forecasting_table(self) -> pd.DataFrame:
        """Her model için en iyi MAE / MAPE / RMSE satırı.

        Yayın için: MAPE üzerinden optimize edilen config referans alınır;
        diğer metrikler tüm optimization-metric variants üzerinden min'ed.
        """
        rows = []
        for model in ("ARIMA", "ETS", "LIGHTGBM", "XGBOOST"):
            best: Dict[str, Optional[float]] = {"MAE": None, "MAPE": None, "RMSE": None}
            opt_time: Optional[float] = None
            n_evals: Optional[int] = None

            for metric in ("MAE", "MAPE", "RMSE"):
                key = (model, metric)
                if key not in self._forecasts:
                    continue
                perf = self._forecasts[key].get("performance", {})
                for k in best:
                    v = perf.get(k)
                    if v is not None and (best[k] is None or v < best[k]):
                        best[k] = v
                # MAPE-optimize config → zaman bilgisi
                if metric == "MAPE":
                    opt_time = self._forecasts[key].get("optimization_time")
                    n_evals = self._forecasts[key].get("function_evals")

            rows.append({
                "Model":         model,
                "MAE":           best["MAE"],
                "MAPE (%)":      best["MAPE"],
                "RMSE":          best["RMSE"],
                "Opt Time (s)":  opt_time,
                "Evals":         n_evals,
            })
        return pd.DataFrame(rows)

    def forecasting_hyperparams_table(self) -> pd.DataFrame:
        """MAPE-optimize config → hyperparameter tablosu."""
        rows = []
        for model in ("ARIMA", "ETS", "LIGHTGBM", "XGBOOST"):
            key = (model, "MAPE")
            if key not in self._forecasts:
                continue
            hp = self._forecasts[key].get("hyperparameters", {})
            if hp:
                row = {"Model": model}
                row.update(hp)
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # 4c.  COMPARISON compilation (Optuna vs HDMR)
    # ------------------------------------------------------------------

    def compile_comparison(self) -> None:
        print("\n  [comparison] searching CSV files …")
        comp_dir = self.root / "comparisons"
        csvs = sorted(comp_dir.glob("comparison_summary_xgboost_*.csv"))
        if csvs:
            try:
                self._comparison = pd.read_csv(csvs[-1])
                print(f"    [OK]   {csvs[-1].name}  →  {self._comparison.shape}")
            except Exception as exc:
                print(f"    [WARN] {exc}")
        else:
            print("    [MISS] comparison_summary CSV not found")

    # ------------------------------------------------------------------
    # 4d.  SENSITIVITY compilation
    # ------------------------------------------------------------------

    def compile_sensitivity(self) -> None:
        print("\n  [sensitivity] parsing aggregated data …")
        sens_dir = self.root / "sensitivity"

        # Aggregated CSV
        agg_csv = sens_dir / "sensitivity_xgboost_agg.csv"
        if agg_csv.exists():
            try:
                self._sensitivity_df = pd.read_csv(agg_csv)
                print(f"    [OK]   agg.csv  →  {self._sensitivity_df.shape}")
            except Exception as exc:
                print(f"    [WARN] {exc}")

        # Aggregated text report
        agg_txt = sens_dir / "sensitivity_report_xgboost_agg.txt"
        if agg_txt.exists():
            self._sensitivity_report = agg_txt.read_text(encoding="utf-8", errors="replace")
            print(f"    [OK]   agg report ({len(self._sensitivity_report)} chars)")

        # Per-seed CSVs (cross-check)
        seed_csvs = sorted(sens_dir.glob("sensitivity_xgboost_seed*.csv"))
        if seed_csvs:
            print(f"    [INFO] {len(seed_csvs)} per-seed CSV(s) found")

    # ------------------------------------------------------------------
    # 4e.  HIGH-DIM extra (summary.json)
    # ------------------------------------------------------------------

    def compile_high_dim(self) -> Dict:
        json_file = self.root / "high_dim_tests" / "summary.json"
        data = load_json(json_file) if json_file.exists() else {}
        if data:
            print(f"    [OK]   high_dim_tests/summary.json")
        return data


# ============================================================================
# 5. LaTeX TABLE GENERATORS
# ============================================================================

def _fmt(val: Any, decimals: int = 4) -> str:
    """Sayı → LaTeX-safe string."""
    if val is None:
        return "—"
    if isinstance(val, float):
        if np.isnan(val):
            return "—"
        if abs(val) < 1e-4 and val != 0.0:
            return f"${val:.2e}$"
        return f"{val:.{decimals}f}"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return str(val)


def latex_benchmark_table(df: pd.DataFrame) -> str:
    """Table 1 — Benchmark.

    Layout:
      Function | Dim | f*(true) | Interaction |
        Std Best | Std Gap | Adp Best | Adp Gap | Gap Δ | Status
    """
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Standard vs.\ Adaptive HDMR on benchmark functions. "
        r"``Gap'' $= f_{\mathrm{found}} - f^{*}$; lower is better. "
        r"``Gap $\Delta$'' $= \mathrm{Gap}_{\mathrm{Std}} - \mathrm{Gap}_{\mathrm{Adp}}$; "
        r"positive means Adaptive is superior. "
        r"Status ``$\checkmark$ Optimal'': $|\mathrm{Gap}| < 0.05$.}"
    )
    lines.append(r"\label{tab:benchmark}")
    lines.append(r"\begin{tabular}{llcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Function & Int.\footnotesize{$^{\dagger}$} "
        r"& $f^{*}$ "
        r"& \multicolumn{2}{c}{Standard} "
        r"& \multicolumn{2}{c}{Adaptive} "
        r"& Gap $\Delta$ & Status \\"
    )
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r" &  & & Best $f$ & Gap & Best $f$ & Gap &  & \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        std_gap = row.get("Std Gap")
        adp_gap = row.get("Adp Gap")

        # Bold: daha küçük gapa sahip olan
        std_bold = (std_gap is not None and adp_gap is not None
                    and not np.isnan(std_gap) and not np.isnan(adp_gap)
                    and std_gap <= adp_gap)
        adp_bold = (std_gap is not None and adp_gap is not None
                    and not np.isnan(std_gap) and not np.isnan(adp_gap)
                    and adp_gap < std_gap)

        def bf(val: Any, bold: bool) -> str:
            s = _fmt(val)
            return rf"\textbf{{{s}}}" if bold else s

        # Interaction sütunu (kısa)
        interaction_raw = row.get("Interaction", "—")
        interaction = "✓" if interaction_raw != "—" else "—"

        line = (
            f"{row['Function']} & "
            f"{interaction} & "
            f"{_fmt(row.get('f* (true)'), 4)} & "
            f"{bf(row.get('Std Best'), std_bold)} & "
            f"{_fmt(std_gap)} & "
            f"{bf(row.get('Adp Best'), adp_bold)} & "
            f"{_fmt(adp_gap)} & "
            f"{_fmt(row.get('Gap Δ'))} & "
            f"{row.get('Status', '—')} \\\\"
        )
        lines.append(line)

    # --- summary footer ---
    adp_gaps = df["Adp Gap"].dropna()
    n_opt = int((adp_gaps.abs() < OPTIMALITY_TOL).sum())
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{2}}{{l}}{{Adaptive optimal}} & "
        rf"\multicolumn{{7}}{{c}}{{{n_opt} / {len(adp_gaps)}}} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(
        r"\end{tabular}"
        "\n"
        r"\footnotesize"
        r"$^{\dagger}$ Functions with cross-variable interaction terms "
        r"(see Sec.~\ref{sec:limitation}): Six-Hump Camel ($x_1 x_2$), "
        r"Rosenbrock ($x_i^2 - x_{i+1}$), Branin ($x_1^2 - x_2$), "
        r"Griewank ($\prod\cos$). First-order HDMR cannot capture these; "
        r"second-order decomposition required."
    )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_forecasting_table(df: pd.DataFrame) -> str:
    """Table 2 — Forecasting performance."""
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{HDMR hyperparameter optimization results "
        r"for time-series forecasting models ($N{=}200$, $m{=}7$). "
        r"Best value per column is \textbf{bold}.}"
    )
    lines.append(r"\label{tab:forecasting}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & MAE & MAPE (\%) & RMSE & Opt.~Time (s) & Evals \\")
    lines.append(r"\midrule")

    # column-wise min (lower-is-better)
    metric_cols = ["MAE", "MAPE (%)", "RMSE"]
    col_mins: Dict[str, float] = {}
    for c in metric_cols:
        valid = df[c].dropna()
        if len(valid):
            col_mins[c] = float(valid.min())

    for _, row in df.iterrows():
        parts = [str(row["Model"])]
        for c in ["MAE", "MAPE (%)", "RMSE", "Opt Time (s)", "Evals"]:
            val = row.get(c)
            s = _fmt(val)
            if c in col_mins and val is not None and not np.isnan(float(val)):
                if abs(float(val) - col_mins[c]) < 1e-8:
                    s = rf"\textbf{{{s}}}"
            parts.append(s)
        lines.append(" & ".join(parts) + " \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_comparison_table(df: pd.DataFrame) -> str:
    """Table 3 — Method comparison (generic)."""
    if df.empty:
        return "% [comparison table: no data]\n"

    cols = list(df.columns)
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Comparison of HDMR variants vs.\ Optuna on XGBoost "
        r"hyperparameter optimization. Best value per column is \textbf{bold}.}"
    )
    lines.append(r"\label{tab:comparison}")
    spec = "l" + "c" * (len(cols) - 1)
    lines.append(rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append(r"\midrule")

    # min per numeric column (lower-is-better heuristic)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col_mins_comp: Dict[str, float] = {}
    for c in num_cols:
        valid = df[c].dropna()
        if len(valid):
            col_mins_comp[c] = float(valid.min())

    for _, row in df.iterrows():
        parts: List[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(float(v)):
                s = _fmt(v)
                if c in col_mins_comp and abs(float(v) - col_mins_comp[c]) < 1e-8:
                    s = rf"\textbf{{{s}}}"
                parts.append(s)
            else:
                parts.append(str(v))
        lines.append(" & ".join(parts) + " \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_sensitivity_table(df: pd.DataFrame) -> str:
    """Table 5 — Sensitivity analysis (hyperparameter importance).

    Flexible: sütun isimleri CSV'den okunur; ilk sütun label, diğerleri numeric.
    Cumulative % hesaplanır eğer eksikse.  Rank sütunu otomatik eklenir.
    """
    if df.empty:
        return "% [sensitivity table: no data]\n"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Sensitivity analysis: hyperparameter importance ranking "
        r"(XGBoost, aggregated over seeds). ``Importance'' is the variance "
        r"contribution of each parameter to the objective. "
        r"Top-$k$ parameters capture $>90\%$ of total sensitivity.}"
    )
    lines.append(r"\label{tab:sensitivity}")

    # ---------- normalise columns ----------
    # Heuristic: ilk sütun = isim sütunu (string), geri kalanlar numeric
    label_col = df.columns[0]
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()

    # Importance sütununu bul (case-insensitive)
    imp_col: Optional[str] = None
    for c in num_cols:
        if "importance" in c.lower() or "variance" in c.lower() or "sensitivity" in c.lower():
            imp_col = c
            break
    if imp_col is None and len(num_cols) > 0:
        imp_col = num_cols[0]  # fallback: ilk sayısal sütun

    # ---------- cumulative % hesaplama ----------
    if imp_col and "cumulative" not in " ".join(df.columns).lower():
        total = df[imp_col].sum()
        if total > 0:
            df = df.copy()
            df["Cumulative (%)"] = (df[imp_col].cumsum() / total * 100).round(2)
            num_cols.append("Cumulative (%)")

    # ---------- rank sütunu ----------
    if imp_col:
        df = df.copy()
        df["Rank"] = range(1, len(df) + 1)

    # ---------- tablo sütun sırası ----------
    display_cols = [label_col]
    if imp_col:
        display_cols.append(imp_col)
    for c in num_cols:
        if c not in display_cols:
            display_cols.append(c)
    if "Rank" in df.columns and "Rank" not in display_cols:
        display_cols.append("Rank")

    spec = "l" + "c" * (len(display_cols) - 1)
    lines.append(rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")
    # header: replace underscores
    lines.append(" & ".join(c.replace("_", " ") for c in display_cols) + " \\\\")
    lines.append(r"\midrule")

    # ---------- satırlar ----------
    for _, row in df.iterrows():
        parts: List[str] = []
        for c in display_cols:
            val = row.get(c)
            if c == "Rank":
                parts.append(str(int(val)) if val is not None else "—")
            else:
                parts.append(_fmt(val, 4))
        lines.append(" & ".join(parts) + " \\\\")

    # ---------- footer ----------
    if imp_col and "Cumulative (%)" in df.columns:
        # 90%-threshold satırı
        over_90 = df[df["Cumulative (%)"] >= 90.0]
        if not over_90.empty:
            k = int(over_90.index[0]) + 1          # 0-indexed → 1-based
            lines.append(r"\midrule")
            lines.append(
                rf"\multicolumn{{2}}{{l}}{{\textit{{Top-{k} parameters: "
                rf"$\geq 90\%$ cumulative}}}} "
                rf"& \multicolumn{{{len(display_cols)-2}}}{{c}}{{}} \\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_hyperparams_table(df: pd.DataFrame) -> str:
    """Table 4 — Optimal hyperparameters per model."""
    if df.empty:
        return "% [hyperparams table: no data]\n"

    cols = list(df.columns)
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Optimal hyperparameters discovered by HDMR "
        r"(MAPE-optimized configuration).}"
    )
    lines.append(r"\label{tab:hyperparams}")
    spec = "l" + "c" * (len(cols) - 1)
    lines.append(rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(v, 4) for v in row) + " \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# 6. VALIDATION & SUMMARY REPORT
# ============================================================================

def build_validation_report(compiler: ResultsCompiler) -> str:
    """Satırsal validation audit log — doğruluk kontrolü."""
    lines: List[str] = []
    sep = "=" * 78

    lines.append(sep)
    lines.append(" HDMR PUBLICATION — VALIDATION & SUMMARY REPORT  (v2.0)")
    lines.append(f" Generated: 2026-02-03  |  Compiler: results_compiler.py v2.0")
    lines.append(sep)

    # ---- A. Benchmark correctness ----
    lines.append("")
    lines.append("A. BENCHMARK RESULTS — GROUND-TRUTH VALIDATION")
    lines.append("-" * 78)

    for fk in _BENCHMARK_REGISTRY:
        name  = DISPLAY_NAMES.get(fk, fk)
        f_star = TRUE_F_STAR.get(fk, 0.0)
        inter = INTERACTION_FLAG.get(fk)
        std   = compiler._benchmarks.get(fk, {}).get("standard")
        adp   = compiler._benchmarks.get(fk, {}).get("adaptive")

        lines.append(f"\n  ┌─ {name}  │  f* = {f_star}  │  interaction: {inter or 'none'}")

        for label, br in (("Standard", std), ("Adaptive", adp)):
            if br is None or not br.runs:
                lines.append(f"  │  {label:10s}: [no data]")
                continue
            gap   = br.best_gap
            flag  = "✓ OPTIMAL" if br.reached_optimum else f"✗ GAP = {gap:+.4f}"
            lines.append(
                f"  │  {label:10s}: best={br.best_fun:>12.6f}  "
                f"mean={br.mean_fun:>12.6f}  std={br.std_fun:>10.6f}  "
                f"[{flag}]  ({br.n_runs} run{'s' if br.n_runs != 1 else ''})"
            )
        lines.append(f"  └{'─' * 74}")

    # ---- B. Summary counters ----
    lines.append("")
    lines.append("B. AGGREGATE STATISTICS")
    lines.append("-" * 78)

    std_optimal = adp_optimal = 0
    total = 0
    for fk in _BENCHMARK_REGISTRY:
        std = compiler._benchmarks.get(fk, {}).get("standard")
        adp = compiler._benchmarks.get(fk, {}).get("adaptive")
        if std and std.runs:
            total += 1
            if std.reached_optimum:
                std_optimal += 1
        if adp and adp.runs:
            if adp.reached_optimum:
                adp_optimal += 1

    lines.append(f"  Standard HDMR — optimal reached:   {std_optimal} / {total}")
    lines.append(f"  Adaptive HDMR — optimal reached:   {adp_optimal} / {total}")

    # Separable vs interaction
    sep_ok = inter_ok = 0
    sep_total = inter_total = 0
    for fk in _BENCHMARK_REGISTRY:
        adp = compiler._benchmarks.get(fk, {}).get("adaptive")
        if not adp or not adp.runs:
            continue
        if INTERACTION_FLAG.get(fk):
            inter_total += 1
            if adp.reached_optimum:
                inter_ok += 1
        else:
            sep_total += 1
            if adp.reached_optimum:
                sep_ok += 1

    lines.append(f"  Separable functions:               {sep_ok} / {sep_total} optimal")
    lines.append(f"  Interaction functions:             {inter_ok} / {inter_total} optimal")

    # ---- C. Known issues ----
    lines.append("")
    lines.append("C. KNOWN ISSUES — PUBLICATION NOTES")
    lines.append("-" * 78)
    issues = [
        ("First-order HDMR limitation",
         "Additive decomposition f₀ + Σ gᵢ(xᵢ) cannot represent cross-variable\n"
         "        interactions. Affected: Six-Hump Camel, Rosenbrock, Branin, Griewank.\n"
         "        Mitigation: second-order HDMR with interaction terms (future work)."),
        ("Branin domain asymmetry",
         "function_ranges.json uses [-5, 15] symmetric.  Standard domain is\n"
         "        x₁ ∈ [-5, 10], x₂ ∈ [0, 15].  All 3 global minima lie within\n"
         "        [-5, 15]² so results are valid, but search space is 2× larger."),
        ("Six-Hump Camel — f(0,0) = 0 ≠ f*",
         "f* = −1.0316.  Any report claiming f ≈ 0 as optimal is incorrect;\n"
         "        (0, 0) is a saddle point.  v1 compiler reported this as 'success'."),
        ("Rosenbrock — f(0,0) = 1.0 ≠ f*",
         "f* = 0.  Optimizer stuck at origin due to narrow curved valley\n"
         "        that first-order HDMR cannot resolve."),
        ("Previous Improvement % bug",
         "v1 used abs(f_found) in denominator, completely masking negative\n"
         "        optima.  v2 uses signed gap-to-optimum throughout."),
    ]
    for title, body in issues:
        lines.append(f"  • {title}")
        lines.append(f"        {body}")
        lines.append("")

    # ---- D. Forecasting summary ----
    lines.append("D. FORECASTING RESULTS")
    lines.append("-" * 78)
    try:
        ftbl = compiler.forecasting_table()
        lines.append(ftbl.to_string(index=False))
    except Exception:
        lines.append("  [no forecasting data]")

    # ---- E. Comparison summary ----
    lines.append("")
    lines.append("E. METHOD COMPARISON (HDMR vs OPTUNA)")
    lines.append("-" * 78)
    if not compiler._comparison.empty:
        lines.append(compiler._comparison.to_string(index=False))
    else:
        lines.append("  [no comparison data]")

    # ---- F. Sensitivity ----
    lines.append("")
    lines.append("F. SENSITIVITY ANALYSIS")
    lines.append("-" * 78)
    if not compiler._sensitivity_df.empty:
        lines.append(compiler._sensitivity_df.to_string(index=False))
    else:
        lines.append("  [no sensitivity CSV data]")

    lines.append("")
    lines.append(sep)
    lines.append(" END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def run(results_dir: str = "./results", output_dir: str = "./publication_results") -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    compiler = ResultsCompiler(results_dir)

    print("\n╔══════════════════════════════════════════════════════════════════════════╗")
    print("║        HDMR PUBLICATION RESULTS COMPILER  v2.0                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    # --- compile phases ---
    compiler.compile_benchmarks()
    compiler.compile_forecasting()
    compiler.compile_comparison()
    compiler.compile_sensitivity()
    high_dim = compiler.compile_high_dim()

    # --- DataFrames ---
    benchmark_df     = compiler.benchmark_table()
    forecast_df      = compiler.forecasting_table()
    forecast_hp_df   = compiler.forecasting_hyperparams_table()
    comparison_df    = compiler._comparison
    sensitivity_df   = compiler._sensitivity_df

    # ── CSV exports ──
    benchmark_df.to_csv(out / "benchmark_results.csv", index=False)
    forecast_df.to_csv(out / "forecasting_summary.csv", index=False)
    if not forecast_hp_df.empty:
        forecast_hp_df.to_csv(out / "forecasting_hyperparams.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(out / "comparison_results.csv", index=False)
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(out / "sensitivity_results.csv", index=False)
    if high_dim:
        (out / "high_dim_summary.json").write_text(json.dumps(high_dim, indent=2))

    # ── LaTeX export ──
    print("\n  [export] Generating LaTeX …")
    tex_parts: List[str] = [
        "% ============================================================",
        "% HDMR Optimization — Publication Tables",
        "% Generated by results_compiler.py  v2.0  (2026-02-03)",
        "% Requires: \\usepackage{booktabs}",
        "% ============================================================\n",
    ]
    tex_parts.append("% --- Table 1: Benchmark Functions ---")
    tex_parts.append(latex_benchmark_table(benchmark_df))
    tex_parts.append("")
    tex_parts.append("% --- Table 2: Forecasting Performance ---")
    tex_parts.append(latex_forecasting_table(forecast_df))
    tex_parts.append("")
    tex_parts.append("% --- Table 3: Method Comparison ---")
    tex_parts.append(latex_comparison_table(comparison_df))
    tex_parts.append("")
    tex_parts.append("% --- Table 4: Optimal Hyperparameters ---")
    tex_parts.append(latex_hyperparams_table(forecast_hp_df))
    tex_parts.append("")
    tex_parts.append("% --- Table 5: Sensitivity Analysis ---")
    tex_parts.append(latex_sensitivity_table(sensitivity_df))

    (out / "latex_tables.tex").write_text("\n".join(tex_parts), encoding="utf-8")

    # ── Validation report ──
    print("  [export] Generating validation report …")
    report = build_validation_report(compiler)
    (out / "validation_report.txt").write_text(report, encoding="utf-8")

    # ── Terminal preview ──
    print("\n" + "─" * 78)
    print(" BENCHMARK TABLE PREVIEW")
    print("─" * 78)
    print(benchmark_df.to_string(index=False))

    print("\n" + "─" * 78)
    print(" FORECASTING TABLE PREVIEW")
    print("─" * 78)
    print(forecast_df.to_string(index=False))

    if not comparison_df.empty:
        print("\n" + "─" * 78)
        print(" COMPARISON TABLE PREVIEW")
        print("─" * 78)
        print(comparison_df.to_string(index=False))

    # ── Validation block ──
    print("\n" + "─" * 78)
    print(" VALIDATION SUMMARY")
    print("─" * 78)
    for _, row in benchmark_df.iterrows():
        status = row.get("Status", "?")
        gap_a  = row.get("Adp Gap")
        print(f"  {row['Function']:20s}  Adp Gap = {_fmt(gap_a):>10s}   {status}")

    # ── Output manifest ──
    print("\n" + "─" * 78)
    print(" OUTPUT FILES")
    print("─" * 78)
    for f in sorted(out.iterdir()):
        print(f"  {f.name:40s}  ({f.stat().st_size:,} bytes)")

    print("\n✓ Compilation complete.\n")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDMR Publication Results Compiler v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python results_compiler.py\n"
            "  python results_compiler.py --results ./results --output ./pub\n"
        ),
    )
    parser.add_argument("--results", default="./results",
                        help="Path to results/ directory  (default: ./results)")
    parser.add_argument("--output",  default="./publication_results",
                        help="Output directory  (default: ./publication_results)")
    args = parser.parse_args()

    run(results_dir=args.results, output_dir=args.output)


if __name__ == "__main__":
    main()