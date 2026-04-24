"""
download_datasets.py
====================
Downloads and caches the 18 TabArena datasets from OpenML (~/.openml/).
Reports N, d, target, and task information for each dataset.

Usage:
    python download_datasets.py           # all 18 datasets
    python download_datasets.py --verify  # metadata only, don't download
    python download_datasets.py --ids 46917 46954  # specific IDs
"""


from __future__ import annotations
import argparse
import sys
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

try:
    import openml
    openml.config.ssl_verify = False
except ImportError:
    print("ERROR: pip install openml")
    sys.exit(1)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset registry — TabArena v0.1
# ---------------------------------------------------------------------------

REGRESSION_DATASETS = {
    46954: "QSAR_fish_toxicity",
    46917: "concrete_compressive_strength",
    46931: "healthcare_insurance_expenses",
    46904: "airfoil_self_noise",
    46907: "Fiat500_used",
    46942: "miami_housing",
    46934: "houses",
    46961: "superconductivity",
    46928: "Food_Delivery_Time",
    46923: "diamonds",
}

CLASSIFICATION_DATASETS = {
    46952: "qsar-biodeg",
    46927: "Fitness_Club",
    46938: "Is-this-a-good-customer",
    46940: "Marketing_Campaign",
    46930: "hazelnut-spread-contaminant-detection",
    46956: "seismic-bumps",
    46963: "website_phishing",
    46918: "credit-g",
}

ALL_DATASETS = {**REGRESSION_DATASETS, **CLASSIFICATION_DATASETS}

# ---------------------------------------------------------------------------
# Download + verify
# ---------------------------------------------------------------------------

def download_dataset(did: int, verify_only: bool = False) -> dict:
    name = ALL_DATASETS.get(did, f"unknown_{did}")
    task = "regression" if did in REGRESSION_DATASETS else "classification"
    result = {
        "did": did, "name": name, "task": task,
        "status": "unknown", "N": None, "d": None,
        "target": None, "missing": None, "error": None,
    }

    try:
        t0 = time.time()

        if verify_only:
            ds = openml.datasets.get_dataset(
                did, download_data=False,
                download_qualities=True,
                download_features_meta_data=False,
            )
            q = ds.qualities or {}
            result.update({
                "status":  "ok_meta",
                "N":       int(q.get("NumberOfInstances", 0)),
                "d":       int(q.get("NumberOfFeatures", 0)) - 1,
                "target":  ds.default_target_attribute,
                "elapsed": round(time.time() - t0, 2),
            })
        else:
            ds = openml.datasets.get_dataset(
                did, download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            X_df, y_s, _, _ = ds.get_data(
                dataset_format="dataframe",
                target=ds.default_target_attribute,
            )
            # Encode categoricals
            for col in X_df.select_dtypes(include=["object", "category"]).columns:
                X_df[col] = X_df[col].astype("category").cat.codes

            X       = X_df.fillna(0).values.astype(float)
            y       = y_s.values
            missing = int(X_df.isnull().sum().sum())

            result.update({
                "status":  "ok",
                "N":       X.shape[0],
                "d":       X.shape[1],
                "target":  ds.default_target_attribute,
                "missing": missing,
                "y_mean":  round(float(np.mean(y.astype(float))), 4) if task == "regression" else None,
                "y_std":   round(float(np.std(y.astype(float))), 4)  if task == "regression" else None,
                "elapsed": round(time.time() - t0, 2),
            })

    except Exception as e:
        result["status"] = "ERROR"
        result["error"]  = str(e)[:120]

    return result


def print_table(results: list[dict]) -> None:
    reg   = [r for r in results if r["task"] == "regression"]
    clf   = [r for r in results if r["task"] == "classification"]

    def fmt_row(r):
        status = "[OK]" if r["status"].startswith("ok") else "[FAIL]"
        n   = str(r["N"])  if r["N"]  else "—"
        d   = str(r["d"])  if r["d"]  else "—"
        err = f"  ← {r['error'][:50]}" if r["error"] else ""
        return f"  {status} {r['did']:6d}  {r['name']:<42}  N={n:<7} d={d:<4}{err}"

    print("\n" + "=" * 70)
    print(f"  REGRESSION DATASETS ({len(reg)}/10)")
    print("=" * 70)
    for r in reg:
        print(fmt_row(r))

    print("\n" + "=" * 70)
    print(f"  CLASSIFICATION DATASETS ({len(clf)}/8)")
    print("=" * 70)
    for r in clf:
        print(fmt_row(r))

    ok_count  = sum(1 for r in results if r["status"].startswith("ok"))
    err_count = sum(1 for r in results if r["status"] == "ERROR")
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {ok_count}/18 OK  |  {err_count} errors")
    print("=" * 70)

    if err_count > 0:
        print("\nFailed datasets:")
        for r in results:
            if r["status"] == "ERROR":
                print(f"  DID {r['did']} ({r['name']}): {r['error']}")


def generate_latex_table(results: list[dict]) -> str:
    """Table 1 — Dataset Summary (LaTeX)"""
    reg = [r for r in results if r["task"] == "regression" and r["status"].startswith("ok")]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Dataset summary. All datasets are from TabArena-v0.1 (Erickson et al., 2025),",
        r"accessed via OpenML suite ID 457. $N$ = number of instances, $d$ = number of features.}",
        r"\label{tab:datasets}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{OpenML ID} & $N$ & $d$ & \textbf{Target} \\",
        r"\midrule",
    ]
    for r in reg:
        target = (r["target"] or "—")[:20]
        lines.append(
            f"  {r['name'].replace('_', ' ')} & {r['did']} "
            f"& {r['N']:,} & {r['d']} & {target} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Download TabArena datasets from OpenML")
    parser.add_argument("--verify", action="store_true",
                        help="Only check metadata, don't download data")
    parser.add_argument("--ids", nargs="+", type=int,
                        help="Specific dataset IDs to download")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX Table 1 after download")
    parser.add_argument("--regression-only", action="store_true",
                        help="Only download regression datasets")
    args = parser.parse_args()

    target_ids = args.ids or list(ALL_DATASETS.keys())
    if args.regression_only:
        target_ids = [d for d in target_ids if d in REGRESSION_DATASETS]

    mode = "VERIFY (metadata only)" if args.verify else "DOWNLOAD + CACHE"
    print(f"\nTabArena Dataset Downloader")
    print(f"Mode    : {mode}")
    print(f"Datasets: {len(target_ids)}")
    print(f"Cache   : ~/.openml/  (auto-managed by openml-python)\n")

    results = []
    for i, did in enumerate(target_ids, 1):
        name = ALL_DATASETS.get(did, f"unknown_{did}")
        task = "regression" if did in REGRESSION_DATASETS else "classification"
        print(f"[{i:2d}/{len(target_ids)}] {did} {name} ...", end=" ", flush=True)

        r = download_dataset(did, verify_only=args.verify)
        results.append(r)

        if r["status"].startswith("ok"):
            elapsed = r.get("elapsed", "?")
            n = r["N"] or "?"
            d = r["d"] or "?"
            print(f"OK  N={n}  d={d}  ({elapsed}s)")
        else:
            print(f"ERROR: {r['error']}")

        # Be polite to OpenML API
        time.sleep(0.5)

    print_table(results)

    if args.latex:
        print("\n\n% ── LaTeX Table 1 ─────────────────────────────────────")
        print(generate_latex_table(results))

    # Save summary CSV
    df = pd.DataFrame(results)
    df.to_csv("dataset_info.csv", index=False)
    print(f"\nDataset info saved → dataset_info.csv")

    # Return exit code based on errors
    n_err = sum(1 for r in results if r["status"] == "ERROR")
    sys.exit(1 if n_err > 0 else 0)


if __name__ == "__main__":
    main()