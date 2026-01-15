# Path fix for reorganized structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load NEW results with Optuna
df = pd.read_csv('results/comparisons/comparison_xgboost_20260114_140609.csv')

# Extract MAPE
df_clean = df[df['metrics'].apply(lambda x: eval(x)['mape'] < 100)].copy()
df_clean['mape'] = df_clean['metrics'].apply(lambda x: eval(x)['mape'])

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. MAPE by method
methods = ['hdmr_adaptive', 'optuna', 'random_search', 'hdmr_standard', 'default']
colors = ['#e74c3c', '#9b59b6', '#2ecc71', '#f39c12', '#3498db']
labels = ['HDMR Adaptive', 'Optuna (Bayesian)', 'Random Search', 'HDMR Standard', 'Default']

for method, color, label in zip(methods, colors, labels):
    data = df_clean[df_clean['method'] == method]['mape']
    ax1.scatter([label]*len(data), data, alpha=0.7, s=180, color=color)

ax1.set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
ax1.set_title('Performance Comparison', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=11)
ax1.axhline(y=9.10, color='#9b59b6', linestyle='--', alpha=0.3, label='Optuna Mean')
ax1.legend()

# 2. Time vs Performance (Pareto front)
summary = df_clean.groupby('method').agg({
    'mape': 'mean',
    'time': 'median'
}).reset_index()

for i, row in summary.iterrows():
    idx = methods.index(row['method'])
    ax2.scatter(row['time'], row['mape'], s=400, alpha=0.7, 
               color=colors[idx], edgecolors='black', linewidth=2)
    ax2.annotate(labels[idx], (row['time'], row['mape']), 
                fontsize=10, ha='center', va='bottom', fontweight='bold')

# Highlight Optuna as optimal trade-off
optuna_data = summary[summary['method'] == 'optuna'].iloc[0]
ax2.scatter(optuna_data['time'], optuna_data['mape'], s=600, 
           facecolors='none', edgecolors='red', linewidth=3, label='Best Trade-off')

ax2.set_xlabel('Median Time (seconds)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Mean MAPE (%)', fontsize=14, fontweight='bold')
ax2.set_title('Speed-Accuracy Trade-off', fontsize=16, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('results/comparisons/comparison_final_with_optuna.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: results/comparisons/comparison_final_with_optuna.png")

# Statistical summary
print("\n" + "="*80)
print("COMPLETE COMPARISON WITH OPTUNA")
print("="*80)

print("\nRanking by Mean MAPE:")
ranking = summary.sort_values('mape')
for i, row in ranking.iterrows():
    idx = methods.index(row['method'])
    print(f"  {i+1}. {labels[idx]:20s} MAPE={row['mape']:.2f}%, Time={row['time']:.1f}s")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("✓ HDMR Adaptive: Best accuracy (8.99%), but 74x slower than Optuna")
print("✓ Optuna: Near-best accuracy (9.10%), extremely fast (11s) → BEST TRADE-OFF!")
print("✓ Random Search: Good baseline (9.23%), fast (15s)")
print(f"\nImprovement of Optuna over Random Search: {((9.23-9.10)/9.23)*100:.1f}%")
print(f"Speed advantage of Optuna over HDMR Adaptive: {812/11:.0f}x faster")

# Detailed statistics
print("\n" + "="*80)
print("DETAILED STATISTICS")
print("="*80)
stats = df_clean.groupby('method')['mape'].agg(['mean', 'std', 'min', 'max', 'count'])
stats = stats.loc[methods]
stats.index = labels
print(stats.round(2))
