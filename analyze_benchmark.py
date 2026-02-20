#!/usr/bin/env python3
"""
Systematic HDMR Benchmark Analyzer
====================================

Comprehensive analysis tool for HDMR benchmark results including:
- Descriptive statistics (mean, std, median, IQR)
- Statistical significance tests (Wilcoxon, Friedman, Kruskal-Wallis)
- Effect size calculations (Cohen's d, rank-biserial r)
- Performance visualizations (boxplots, heatmaps, convergence)
- LaTeX table generation for academic publication
- Automated text report generation

Usage:
    python analyze_benchmark.py <results_dir>
    python analyze_benchmark.py results/systematic_benchmark_20260216_151558
    python analyze_benchmark.py results/systematic_benchmark_20260216_151558 --no-plots
    python analyze_benchmark.py results/systematic_benchmark_20260216_151558 --latex

Output:
    <results_dir>/analysis/
        figures/               - All visualization plots (8 figures)
        statistical_report.txt - Full statistical report
        latex_tables.tex       - Ready-to-use LaTeX tables
        basis_comparison_tests.csv
        mode_comparison_tests.csv
        order_effect_tests.csv

Author: HDMR Research Group
Date: 2025-02-16
Version: 2.0.0
"""

import sys
import json
import warnings
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import wilcoxon, kruskal

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

COLORS = {
    'Legendre': '#2196F3',
    'Cosine':   '#FF5722',
    'standard': '#4CAF50',
    'adaptive': '#9C27B0',
}

DIFFICULTY_ORDER = [
    'sphere', 'testfunc_2d', 'zakharov', 'rosenbrock_2d', 'sum_of_different_powers',
    'dixon_price', 'camel16_2d', 'branin_2d', 'levy', 'ackley_2d',
    'griewank_10d', 'styblinski_tang', 'rastrigin_2d', 'rosenbrock_10d',
    'rastrigin_10d', 'michalewicz', 'schwefel',
]

FUNCTION_CATEGORIES = {
    '2d':       ['testfunc_2d', 'rosenbrock_2d', 'ackley_2d', 'branin_2d',
                 'camel16_2d', 'camel3_2d', 'rastrigin_2d', 'treccani_2d', 'goldstein_2d'],
    'classical': ['rosenbrock_10d', 'rastrigin_10d', 'griewank_10d'],
    'modern':   ['sphere', 'zakharov', 'levy', 'dixon_price', 'styblinski_tang',
                 'michalewicz', 'schwefel', 'sum_of_different_powers'],
}

CATEGORY_COLORS = {'2d': '#2196F3', 'classical': '#FF5722', 'modern': '#4CAF50'}

SIGNIFICANCE_LEVELS = [(0.001, '***'), (0.01, '**'), (0.05, '*'), (1.0, 'ns')]


# ============================================================================
# HELPERS
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def setup_plot_style():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    })


def stars(p: float) -> str:
    for thresh, s in SIGNIFICANCE_LEVELS:
        if p < thresh:
            return s
    return 'ns'


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return 'negligible'
    if d < 0.5:  return 'small'
    if d < 0.8:  return 'medium'
    return 'large'


def func_category(name: str) -> str:
    for cat, members in FUNCTION_CATEGORIES.items():
        if name in members:
            return cat
    return 'modern'


def load_results(results_dir: Path) -> Tuple[pd.DataFrame, dict]:
    raw_path = results_dir / "raw_results.csv"
    config_path = results_dir / "config.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.csv not found: {results_dir}")

    df = pd.read_csv(raw_path)
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    df['optimality_gap'] = df['best_f'] - df['true_optimum']
    df['best_f'] = df['best_f'].replace([np.inf, -np.inf], np.nan)
    df['optimality_gap'] = df['optimality_gap'].replace([np.inf, -np.inf], np.nan)

    return df, config


def sorted_funcs(df: pd.DataFrame) -> List[str]:
    present = df['function'].unique()
    ordered = [f for f in DIFFICULTY_ORDER if f in present]
    ordered += [f for f in present if f not in ordered]
    return ordered


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def _wilcoxon_pair(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 5 or np.all(a == b):
        return 0.0, 1.0
    try:
        return wilcoxon(a, b)
    except Exception:
        return np.nan, np.nan


def test_basis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for func in df['function'].unique():
        d = df[df['function'] == func]
        leg = d[d['basis']=='Legendre']['optimality_gap'].dropna().values
        cos = d[d['basis']=='Cosine']['optimality_gap'].dropna().values
        if len(leg) < 5 or len(cos) < 5:
            continue
        stat, p = _wilcoxon_pair(leg, cos)
        cd = cohen_d(leg, cos)
        rows.append(dict(
            function=func,
            legendre_median=np.median(leg), cosine_median=np.median(cos),
            p_value=p, significance=stars(p) if not np.isnan(p) else 'N/A',
            cohens_d=cd, effect_size=interpret_d(cd),
            winner='Legendre' if np.median(leg) < np.median(cos) else 'Cosine',
        ))
    return pd.DataFrame(rows)


def test_mode(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for func in df['function'].unique():
        d = df[df['function'] == func]
        std = d[d['mode']=='standard']['optimality_gap'].dropna().values
        adp = d[d['mode']=='adaptive']['optimality_gap'].dropna().values
        if len(std) < 5 or len(adp) < 5:
            continue
        stat, p = _wilcoxon_pair(std, adp)
        cd = cohen_d(std, adp)
        rows.append(dict(
            function=func,
            standard_median=np.median(std), adaptive_median=np.median(adp),
            p_value=p, significance=stars(p) if not np.isnan(p) else 'N/A',
            cohens_d=cd, effect_size=interpret_d(cd),
            winner='standard' if np.median(std) < np.median(adp) else 'adaptive',
        ))
    return pd.DataFrame(rows)


def test_order(df: pd.DataFrame) -> pd.DataFrame:
    orders = sorted(df['order'].unique())
    rows = []
    for func in df['function'].unique():
        d = df[df['function'] == func]
        groups = []
        medians = {}
        for o in orders:
            g = d[d['order']==o]['optimality_gap'].dropna().values
            g = np.clip(g[np.isfinite(g)], 1e-12, None)
            if len(g) >= 3:
                groups.append(g)
                medians[f'order_{o}_median'] = np.median(g)
        if len(groups) < 2:
            continue
        try:
            H, p = kruskal(*groups)
        except Exception:
            H, p = np.nan, np.nan
        best_o = orders[np.argmin([np.median(g) for g in groups])]
        row = dict(function=func, kruskal_H=H, p_value=p,
                   significance=stars(p) if not np.isnan(p) else 'N/A',
                   best_order=best_o)
        row.update(medians)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# FIGURES
# ============================================================================

def fig1_overview(df: pd.DataFrame, out: Path):
    """Optimality gap distribution per function — log boxplot."""
    funcs = sorted_funcs(df)
    fig, ax = plt.subplots(figsize=(max(14, len(funcs)), 7))

    data, colors_list = [], []
    for func in funcs:
        vals = df[df['function']==func]['optimality_gap'].dropna()
        vals = np.clip(vals[np.isfinite(vals)], 1e-12, None)
        data.append(vals.values)
        cat = func_category(func)
        colors_list.append(CATEGORY_COLORS.get(cat, '#888888'))

    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, c in zip(bp['boxes'], colors_list):
        patch.set(facecolor=c, alpha=0.7)

    ax.set_yscale('log')
    ax.set_xticklabels([f.replace('_','\n') for f in funcs], fontsize=8.5)
    ax.set_xlabel('Benchmark Function', fontsize=12)
    ax.set_ylabel('Optimality Gap (log scale)', fontsize=12)
    ax.set_title('HDMR Performance Overview — All Functions & Configurations',
                 fontsize=13, fontweight='bold')
    ax.legend(handles=[Patch(facecolor=CATEGORY_COLORS[c], alpha=0.7, label=c.title())
                        for c in CATEGORY_COLORS], loc='upper left')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig1_overview.{ext}', bbox_inches='tight')
    plt.close()


def fig2_basis(df: pd.DataFrame, out: Path):
    """Legendre vs Cosine — side-by-side boxplots per function."""
    funcs = sorted_funcs(df)
    ncols = 4
    nrows = int(np.ceil(len(funcs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, func in enumerate(funcs):
        ax = axes[i]
        d = df[df['function']==func]
        vals = [np.clip(d[d['basis']==b]['optimality_gap'].dropna()
                         .pipe(lambda s: s[np.isfinite(s)]), 1e-12, None).values
                for b in ('Legendre', 'Cosine')]
        bp = ax.boxplot(vals, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        bp['boxes'][0].set(facecolor=COLORS['Legendre'], alpha=0.7)
        bp['boxes'][1].set(facecolor=COLORS['Cosine'], alpha=0.7)
        ax.set_yscale('log')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Legendre', 'Cosine'], fontsize=9)
        m0 = np.median(vals[0]) if len(vals[0]) else np.inf
        m1 = np.median(vals[1]) if len(vals[1]) else np.inf
        tag = '◀ L' if m0 < m1 else 'C ▶'
        ax.set_title(f'{func}\n{tag}', fontsize=9, fontweight='bold')
        ax.set_ylabel('Gap', fontsize=8)

    for i in range(len(funcs), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Basis Comparison: Legendre vs Cosine', fontsize=14, fontweight='bold', y=1.01)
    fig.legend(handles=[Patch(facecolor=COLORS[b], alpha=0.7, label=b) for b in ('Legendre','Cosine')],
               loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig2_basis.{ext}', bbox_inches='tight')
    plt.close()


def fig3_mode(df: pd.DataFrame, out: Path):
    """Standard vs Adaptive — side-by-side boxplots per function."""
    funcs = sorted_funcs(df)
    ncols = 4
    nrows = int(np.ceil(len(funcs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, func in enumerate(funcs):
        ax = axes[i]
        d = df[df['function']==func]
        vals = [np.clip(d[d['mode']==m]['optimality_gap'].dropna()
                         .pipe(lambda s: s[np.isfinite(s)]), 1e-12, None).values
                for m in ('standard', 'adaptive')]
        bp = ax.boxplot(vals, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        bp['boxes'][0].set(facecolor=COLORS['standard'], alpha=0.7)
        bp['boxes'][1].set(facecolor=COLORS['adaptive'], alpha=0.7)
        ax.set_yscale('log')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Std', 'Adp'], fontsize=9)
        m0 = np.median(vals[0]) if len(vals[0]) else np.inf
        m1 = np.median(vals[1]) if len(vals[1]) else np.inf
        tag = '◀ Std' if m0 < m1 else 'Adp ▶'
        ax.set_title(f'{func}\n{tag}', fontsize=9, fontweight='bold')
        ax.set_ylabel('Gap', fontsize=8)

    for i in range(len(funcs), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Mode Comparison: Standard vs Adaptive HDMR', fontsize=14, fontweight='bold', y=1.01)
    fig.legend(handles=[Patch(facecolor=COLORS[m], alpha=0.7, label=m.title())
                        for m in ('standard','adaptive')], loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig3_mode.{ext}', bbox_inches='tight')
    plt.close()


def fig4_order(df: pd.DataFrame, out: Path):
    """Order effect — line plot with IQR band per function."""
    orders = sorted(df['order'].unique())
    funcs = sorted_funcs(df)
    ncols = 4
    nrows = int(np.ceil(len(funcs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, func in enumerate(funcs):
        ax = axes[i]
        d = df[df['function']==func]
        for mode, color in (('standard', COLORS['standard']), ('adaptive', COLORS['adaptive'])):
            med, q25, q75 = [], [], []
            for o in orders:
                g = np.clip(d[(d['order']==o)&(d['mode']==mode)]['optimality_gap']
                             .dropna().pipe(lambda s: s[np.isfinite(s)]), 1e-12, None)
                if len(g) > 0:
                    med.append(float(np.median(g)))
                    q25.append(float(np.percentile(g, 25)))
                    q75.append(float(np.percentile(g, 75)))
                else:
                    med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
            ax.semilogy(orders, med, 'o-', color=color, linewidth=2, markersize=5,
                       label=mode.capitalize())
            ax.fill_between(orders, q25, q75, alpha=0.15, color=color)
        ax.set_xticks(orders)
        ax.set_xlabel('Order', fontsize=8)
        ax.set_ylabel('Gap', fontsize=8)
        ax.set_title(func, fontsize=9, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(len(funcs), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Effect of Polynomial Order on Optimality Gap (Median ± IQR)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig4_order.{ext}', bbox_inches='tight')
    plt.close()


def fig5_heatmap(df: pd.DataFrame, out: Path):
    """Heatmap: function × order median gap, separate per mode."""
    orders = sorted(df['order'].unique())
    funcs = sorted_funcs(df)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(funcs) * 0.65)))

    for ax, mode in zip(axes, ('standard', 'adaptive')):
        dm = df[df['mode']==mode]
        mat = np.full((len(funcs), len(orders)), np.nan)
        for i, func in enumerate(funcs):
            for j, o in enumerate(orders):
                g = dm[(dm['function']==func)&(dm['order']==o)]['optimality_gap'].dropna()
                g = np.clip(g[np.isfinite(g)], 1e-12, None)
                if len(g) > 0:
                    mat[i, j] = np.median(g)

        log_mat = np.log10(np.where(mat > 0, mat, np.nan))
        im = ax.imshow(log_mat, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels([f'Ord {o}' for o in orders], rotation=45, ha='right')
        ax.set_yticks(range(len(funcs)))
        ax.set_yticklabels(funcs, fontsize=9)
        ax.set_title(f'{mode.title()} HDMR — Median Gap (log₁₀)', fontsize=12, fontweight='bold')

        for i in range(len(funcs)):
            for j in range(len(orders)):
                if not np.isnan(log_mat[i, j]):
                    txt = f'{mat[i,j]:.1e}'
                    color = 'white' if log_mat[i,j] > 1 else 'black'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color=color)

        plt.colorbar(im, ax=ax, label='log₁₀(Gap)', shrink=0.8)

    plt.suptitle('Performance Heatmap: Function × Order', fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig5_heatmap.{ext}', bbox_inches='tight')
    plt.close()


def fig6_runtime(df: pd.DataFrame, out: Path):
    """Runtime: distribution + speed-accuracy scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: boxplot by mode
    ax = axes[0]
    std_t = df[df['mode']=='standard']['wall_time'].dropna()
    adp_t = df[df['mode']=='adaptive']['wall_time'].dropna()
    bp = ax.boxplot([std_t, adp_t], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set(facecolor=COLORS['standard'], alpha=0.7)
    bp['boxes'][1].set(facecolor=COLORS['adaptive'], alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Standard', 'Adaptive'])
    ax.set_ylabel('Wall Time (s)')
    ax.set_title('Runtime Distribution by Mode', fontweight='bold')
    for pos, vals in ((1, std_t), (2, adp_t)):
        ax.text(pos, np.median(vals), f'  {np.median(vals):.2f}s', va='center', fontsize=10)

    # Right: speed vs accuracy scatter
    ax = axes[1]
    for mode, color, marker in (('standard', COLORS['standard'], 'o'),
                                  ('adaptive', COLORS['adaptive'], 's')):
        agg = (df[df['mode']==mode]
               .groupby('function')
               .agg(wall_time=('wall_time','median'), optimality_gap=('optimality_gap','median'))
               .reset_index())
        agg = agg[np.isfinite(agg['optimality_gap']) & (agg['optimality_gap'] > 0)]
        ax.scatter(agg['wall_time'], agg['optimality_gap'],
                  color=color, marker=marker, s=80, alpha=0.8, label=mode.title())
        for _, row in agg.iterrows():
            ax.annotate(row['function'], (row['wall_time'], row['optimality_gap']),
                       fontsize=6, alpha=0.7)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Median Wall Time (s, log)'); ax.set_ylabel('Median Gap (log)')
    ax.set_title('Speed vs Accuracy', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig6_runtime.{ext}', bbox_inches='tight')
    plt.close()


def fig7_significance(basis_tests: pd.DataFrame, mode_tests: pd.DataFrame, out: Path):
    """Statistical significance summary — bar chart of -log10(p)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, max(len(basis_tests), len(mode_tests)) * 0.45)))

    sig_colors = [(0.001,'#1B5E20'),(0.01,'#388E3C'),(0.05,'#81C784'),(1.0,'#FFCDD2')]

    for ax, df_t, title in (
        (axes[0], basis_tests,  'Legendre vs Cosine'),
        (axes[1], mode_tests,   'Standard vs Adaptive'),
    ):
        if df_t.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        funcs = df_t['function'].values
        p_vals = df_t['p_value'].values

        bar_colors = []
        for p in p_vals:
            if np.isnan(p):
                bar_colors.append('#CCCCCC')
            else:
                for thresh, c in sig_colors:
                    if p < thresh:
                        bar_colors.append(c)
                        break

        log_p = -np.log10(np.clip(p_vals, 1e-10, 1))
        ax.barh(range(len(funcs)), log_p, color=bar_colors, alpha=0.85, edgecolor='white')
        ax.axvline(-np.log10(0.05), color='orange', linestyle='--', linewidth=1.5, label='p=0.05')
        ax.axvline(-np.log10(0.01), color='red',    linestyle='--', linewidth=1.5, label='p=0.01')
        ax.set_yticks(range(len(funcs)))
        ax.set_yticklabels(funcs, fontsize=9)
        ax.set_xlabel('-log₁₀(p-value)')
        ax.set_title(f'{title}\n(Wilcoxon Signed-Rank)', fontweight='bold')
        ax.legend(fontsize=9)

        winner_col = 'winner'
        for j, (p, func) in enumerate(zip(p_vals, funcs)):
            s = stars(p) if not np.isnan(p) else 'N/A'
            winner = df_t[df_t['function']==func].iloc[0].get('winner', '')
            ax.text(log_p[j] + 0.05, j, f' {s} ({winner})', va='center', fontsize=8)

    sig_legend = [Patch(facecolor=c, alpha=0.85, label=f'p < {t}' if t < 1 else 'ns')
                  for t, c in sig_colors]
    fig.legend(handles=sig_legend, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5,-0.05))
    plt.suptitle('Statistical Significance of HDMR Hyperparameter Effects',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig7_significance.{ext}', bbox_inches='tight')
    plt.close()


def fig8_best_config(df: pd.DataFrame, out: Path):
    """Best achieved result per function — annotated bar chart."""
    funcs = sorted_funcs(df)
    rows = []
    for func in funcs:
        d = df[df['function']==func].dropna(subset=['optimality_gap'])
        d = d[np.isfinite(d['optimality_gap'])]
        if d.empty:
            continue
        best = d.loc[d['optimality_gap'].idxmin()]
        rows.append(dict(
            function=func,
            gap=best['optimality_gap'],
            mode=best['mode'],
            label=f"{best['basis'][0]}{best['order']}-{best['mode'][:3].title()}"
        ))

    df_best = pd.DataFrame(rows)
    colors_bar = [COLORS.get(r['mode'], '#888') for _, r in df_best.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 6))
    log_gaps = np.log10(np.clip(df_best['gap'], 1e-12, None))
    ax.bar(range(len(df_best)), log_gaps, color=colors_bar, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(df_best)))
    ax.set_xticklabels(df_best['function'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Best Optimality Gap (log₁₀)')
    ax.set_title('Best Configuration Achieved per Function', fontsize=13, fontweight='bold')

    for i, (_, row) in enumerate(df_best.iterrows()):
        ax.text(i, log_gaps.iloc[i] + 0.1, row['label'],
               ha='center', va='bottom', fontsize=7, rotation=45)

    ax.legend(handles=[Patch(facecolor=COLORS[m], alpha=0.8, label=m.title())
                        for m in ('standard','adaptive')])
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out / f'fig8_best_config.{ext}', bbox_inches='tight')
    plt.close()


# ============================================================================
# TEXT REPORT
# ============================================================================

def generate_report(df, basis_tests, mode_tests, order_tests, config, out_dir):
    sep = "=" * 80
    sub = "-" * 80
    lines = [sep, "HDMR BENCHMARK — STATISTICAL ANALYSIS REPORT",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", sep]

    # Overview
    lines += ["\n1. EXPERIMENT OVERVIEW", sub,
              f"  Total experiments:  {len(df)}",
              f"  Successful runs:    {df['success'].sum()} ({100*df['success'].mean():.1f}%)",
              f"  Functions tested:   {df['function'].nunique()}",
              f"  Basis functions:    {df['basis'].unique().tolist()}",
              f"  Orders:             {sorted(df['order'].unique().tolist())}",
              f"  Modes:              {df['mode'].unique().tolist()}",
              f"  Runs per config:    {config.get('num_runs','N/A')}",
              f"  Samples per run:    {config.get('num_samples','N/A')}",
              f"  Total wall time:    {df['wall_time'].sum():.1f}s "
              f"({df['wall_time'].sum()/3600:.2f}h)"]

    # Descriptive stats
    lines += ["\n\n2. DESCRIPTIVE STATISTICS PER FUNCTION", sub,
              f"  {'Function':<28} {'Median':>12} {'Mean':>12} {'Std':>12} "
              f"{'Min':>12} {'Success%':>10}", "  " + "-"*90]
    for func in df['function'].unique():
        g = df[df['function']==func]['optimality_gap'].dropna()
        g = g[np.isfinite(g)]
        s = df[df['function']==func]['success'].mean() * 100
        if len(g):
            lines.append(f"  {func:<28} {np.median(g):>12.4e} {np.mean(g):>12.4e} "
                        f"{np.std(g):>12.4e} {np.min(g):>12.4e} {s:>9.1f}%")

    # Basis comparison
    leg_wins = (basis_tests['winner']=='Legendre').sum() if not basis_tests.empty else 0
    cos_wins = (basis_tests['winner']=='Cosine').sum()   if not basis_tests.empty else 0
    sig_basis = (basis_tests['p_value'] < 0.05).sum()   if not basis_tests.empty else 0
    lines += ["\n\n3. BASIS FUNCTION COMPARISON (Legendre vs Cosine)", sub,
              f"  {'Function':<28} {'Leg Med':>12} {'Cos Med':>12} "
              f"{'p-value':>10} {'Sig':>5} {'Effect':>12} {'Winner':>12}",
              "  " + "-"*95]
    for _, r in basis_tests.iterrows():
        lines.append(f"  {r['function']:<28} {r['legendre_median']:>12.4e} "
                    f"{r['cosine_median']:>12.4e} {r['p_value']:>10.4f} "
                    f"{r['significance']:>5} {r['effect_size']:>12} {r['winner']:>12}")
    lines += [f"\n  Legendre wins: {leg_wins}/{len(basis_tests)}, "
              f"Cosine wins: {cos_wins}/{len(basis_tests)}, "
              f"Significant: {sig_basis}/{len(basis_tests)}"]

    # Mode comparison
    std_wins = (mode_tests['winner']=='standard').sum() if not mode_tests.empty else 0
    adp_wins = (mode_tests['winner']=='adaptive').sum() if not mode_tests.empty else 0
    sig_mode = (mode_tests['p_value'] < 0.05).sum()    if not mode_tests.empty else 0
    lines += ["\n\n4. MODE COMPARISON (Standard vs Adaptive)", sub,
              f"  {'Function':<28} {'Std Med':>12} {'Adp Med':>12} "
              f"{'p-value':>10} {'Sig':>5} {'Effect':>12} {'Winner':>12}",
              "  " + "-"*95]
    for _, r in mode_tests.iterrows():
        lines.append(f"  {r['function']:<28} {r['standard_median']:>12.4e} "
                    f"{r['adaptive_median']:>12.4e} {r['p_value']:>10.4f} "
                    f"{r['significance']:>5} {r['effect_size']:>12} {r['winner']:>12}")
    lines += [f"\n  Standard wins: {std_wins}/{len(mode_tests)}, "
              f"Adaptive wins: {adp_wins}/{len(mode_tests)}, "
              f"Significant: {sig_mode}/{len(mode_tests)}"]

    # Order effect
    sig_order = (order_tests['p_value'] < 0.05).sum() if not order_tests.empty else 0
    lines += ["\n\n5. POLYNOMIAL ORDER EFFECT (Kruskal-Wallis)", sub,
              f"  {'Function':<28} {'H-stat':>12} {'p-value':>10} {'Sig':>5} {'Best Order':>12}",
              "  " + "-"*75]
    for _, r in order_tests.iterrows():
        lines.append(f"  {r['function']:<28} {r['kruskal_H']:>12.4f} "
                    f"{r['p_value']:>10.4f} {r['significance']:>5} {int(r['best_order']):>12}")
    if not order_tests.empty:
        top_o = int(order_tests['best_order'].value_counts().index[0])
        lines += [f"\n  Order {top_o} is optimal most frequently. "
                  f"Significant in {sig_order}/{len(order_tests)} cases."]

    # Best configs
    lines += ["\n\n6. BEST CONFIGURATION PER FUNCTION", sub,
              f"  {'Function':<28} {'Best Gap':>12} {'Basis':>12} "
              f"{'Order':>8} {'Mode':>12} {'Time':>8}", "  " + "-"*85]
    for func in df['function'].unique():
        d = df[df['function']==func].dropna(subset=['optimality_gap'])
        d = d[np.isfinite(d['optimality_gap'])]
        if d.empty: continue
        b = d.loc[d['optimality_gap'].idxmin()]
        lines.append(f"  {func:<28} {b['optimality_gap']:>12.4e} "
                    f"{b['basis']:>12} {int(b['order']):>8} "
                    f"{b['mode']:>12} {b['wall_time']:>7.2f}s")

    # Findings
    lines += ["\n\n7. KEY FINDINGS", sub]
    if not basis_tests.empty:
        winner = 'Legendre' if leg_wins >= cos_wins else 'Cosine'
        lines.append(f"  [Basis] {winner} wins on {max(leg_wins,cos_wins)}/{len(basis_tests)} "
                    f"functions. Statistically significant: {sig_basis} cases.")
    if not mode_tests.empty:
        winner = 'Adaptive' if adp_wins >= std_wins else 'Standard'
        lines.append(f"  [Mode]  {winner} wins on {max(std_wins,adp_wins)}/{len(mode_tests)} "
                    f"functions. Statistically significant: {sig_mode} cases.")
    if not order_tests.empty:
        lines.append(f"  [Order] Order {top_o} is most frequently optimal. "
                    f"Significant order effect: {sig_order}/{len(order_tests)} functions.")
    lines += ["\n  Caveats:",
              "  - p-values without effect sizes may be misleading — check Cohen's d.",
              f"  - Only {config.get('num_runs','N/A')} runs per config. Consider 30+ for robust statistics.",
              "  - Bonferroni correction recommended for multiple comparisons in publication.",
              f"\n{sep}", "END OF REPORT", sep]

    txt = "\n".join(lines)
    path = out_dir / "statistical_report.txt"
    path.write_text(txt, encoding='utf-8')
    return txt


def generate_latex(df, basis_tests, mode_tests, out_dir):
    lines = [
        "% HDMR Benchmark — Auto-generated LaTeX Tables",
        f"% {datetime.now():%Y-%m-%d %H:%M:%S}", "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{HDMR Benchmark: Descriptive Statistics (Optimality Gap)}",
        r"\label{tab:descriptive}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Function & Median & Mean & Std & Best \\",
        r"\midrule",
    ]
    for func in df['function'].unique():
        g = df[df['function']==func]['optimality_gap'].dropna()
        g = g[np.isfinite(g)]
        if not len(g): continue
        ft = func.replace('_', r'\_')
        lines.append(f"{ft} & {np.median(g):.3e} & {np.mean(g):.3e} & "
                    f"{np.std(g):.3e} & {np.min(g):.3e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    if not basis_tests.empty:
        lines += [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Basis Comparison: Legendre vs Cosine (Wilcoxon Test)}",
            r"\label{tab:basis}", r"\small",
            r"\begin{tabular}{lrrrllr}", r"\toprule",
            r"Function & Leg.\ Med. & Cos.\ Med. & $p$ & Sig. & Effect & Winner \\",
            r"\midrule",
        ]
        for _, r in basis_tests.iterrows():
            ft = r['function'].replace('_', r'\_')
            lines.append(f"{ft} & {r['legendre_median']:.3e} & {r['cosine_median']:.3e} & "
                        f"{r['p_value']:.4f} & {r['significance']} & "
                        f"{r['effect_size']} & {r['winner']} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    if not mode_tests.empty:
        lines += [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Mode Comparison: Standard vs Adaptive HDMR (Wilcoxon Test)}",
            r"\label{tab:mode}", r"\small",
            r"\begin{tabular}{lrrrllr}", r"\toprule",
            r"Function & Std.\ Med. & Adp.\ Med. & $p$ & Sig. & Effect & Winner \\",
            r"\midrule",
        ]
        for _, r in mode_tests.iterrows():
            ft = r['function'].replace('_', r'\_')
            lines.append(f"{ft} & {r['standard_median']:.3e} & {r['adaptive_median']:.3e} & "
                        f"{r['p_value']:.4f} & {r['significance']} & "
                        f"{r['effect_size']} & {r['winner']} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = out_dir / "latex_tables.tex"
    path.write_text("\n".join(lines), encoding='utf-8')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HDMR Benchmark Analyzer — Statistical analysis & visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_benchmark.py results/systematic_benchmark_20260216_151558
  python analyze_benchmark.py results/systematic_benchmark_20260216_151558 --no-plots
  python analyze_benchmark.py results/systematic_benchmark_20260216_151558 --latex
        """
    )
    parser.add_argument('results_dir', help='Path to benchmark results directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip figure generation')
    parser.add_argument('--latex',    action='store_true', help='Generate LaTeX tables')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Not found: {results_dir}")
        sys.exit(1)

    analysis_dir = results_dir / "analysis"
    figures_dir  = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(analysis_dir)
    setup_plot_style()

    logger.info("=" * 70)
    logger.info("HDMR BENCHMARK ANALYZER  v2.0")
    logger.info("=" * 70)
    logger.info(f"Input : {results_dir}")
    logger.info(f"Output: {analysis_dir}")

    # 1. Load
    logger.info("\n[1/5] Loading data...")
    df, config = load_results(results_dir)
    logger.info(f"  {len(df)} experiments | {df['function'].nunique()} functions | "
               f"{100*df['success'].mean():.1f}% success")

    # 2. Tests
    logger.info("\n[2/5] Statistical tests...")
    basis_tests = test_basis(df)
    mode_tests  = test_mode(df)
    order_tests = test_order(df)
    basis_tests.to_csv(analysis_dir / "basis_tests.csv",  index=False)
    mode_tests.to_csv( analysis_dir / "mode_tests.csv",   index=False)
    order_tests.to_csv(analysis_dir / "order_tests.csv",  index=False)
    logger.info(f"  Basis: {len(basis_tests)} | Mode: {len(mode_tests)} | Order: {len(order_tests)}")

    # 3. Report
    logger.info("\n[3/5] Generating report...")
    generate_report(df, basis_tests, mode_tests, order_tests, config, analysis_dir)
    logger.info(f"  ✓ statistical_report.txt")

    if args.latex:
        generate_latex(df, basis_tests, mode_tests, analysis_dir)
        logger.info(f"  ✓ latex_tables.tex")

    # 4. Figures
    if not args.no_plots:
        logger.info("\n[4/5] Generating figures...")
        plot_tasks = [
            ('fig1_overview',      fig1_overview,    (df, figures_dir)),
            ('fig2_basis',         fig2_basis,       (df, figures_dir)),
            ('fig3_mode',          fig3_mode,        (df, figures_dir)),
            ('fig4_order',         fig4_order,       (df, figures_dir)),
            ('fig5_heatmap',       fig5_heatmap,     (df, figures_dir)),
            ('fig6_runtime',       fig6_runtime,     (df, figures_dir)),
            ('fig7_significance',  fig7_significance,(basis_tests, mode_tests, figures_dir)),
            ('fig8_best_config',   fig8_best_config, (df, figures_dir)),
        ]
        for name, fn, args_fn in plot_tasks:
            try:
                fn(*args_fn)
                logger.info(f"  ✓ {name}")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")

    # 5. Summary
    logger.info("\n[5/5] Summary")
    logger.info("-" * 50)
    if not basis_tests.empty:
        lw = (basis_tests['winner']=='Legendre').sum()
        cw = (basis_tests['winner']=='Cosine').sum()
        sb = (basis_tests['p_value'] < 0.05).sum()
        logger.info(f"  Basis  → Legendre {lw}W, Cosine {cw}W | Sig: {sb}/{len(basis_tests)}")
    if not mode_tests.empty:
        sw = (mode_tests['winner']=='standard').sum()
        aw = (mode_tests['winner']=='adaptive').sum()
        sm = (mode_tests['p_value'] < 0.05).sum()
        logger.info(f"  Mode   → Standard {sw}W, Adaptive {aw}W | Sig: {sm}/{len(mode_tests)}")
    if not order_tests.empty:
        to = int(order_tests['best_order'].value_counts().index[0])
        so = (order_tests['p_value'] < 0.05).sum()
        logger.info(f"  Order  → Best order = {to} | Sig: {so}/{len(order_tests)}")

    logger.info(f"\n✓ Analysis complete!")
    logger.info(f"  Report : {analysis_dir / 'statistical_report.txt'}")
    if not args.no_plots:
        logger.info(f"  Figures: {figures_dir}/ (8 plots × PNG+PDF)")
    if args.latex:
        logger.info(f"  LaTeX  : {analysis_dir / 'latex_tables.tex'}")


if __name__ == "__main__":
    main()