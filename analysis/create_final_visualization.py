import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Comparison: MAPE by method
ax1 = fig.add_subplot(gs[0, 0])
comp = pd.read_csv('results/comparisons/comparison_summary_xgboost_20260114_140609.csv')
comp_sorted = comp.sort_values('Mean MAPE')
colors = ['#e74c3c', '#9b59b6', '#2ecc71', '#f39c12', '#3498db']
ax1.barh(comp_sorted['Method'], comp_sorted['Mean MAPE'], color=colors)
ax1.set_xlabel('Mean MAPE (%)', fontweight='bold')
ax1.set_title('Optimizer Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(comp_sorted['Mean MAPE']):
    ax1.text(v + 0.1, i, f'{v:.2f}%', va='center', fontweight='bold')

# 2. Time-Performance Trade-off
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(comp['Median Time (s)'], comp['Mean MAPE'], s=300, alpha=0.7, c=colors)
for i, row in comp.iterrows():
    ax2.annotate(row['Method'], (row['Median Time (s)'], row['Mean MAPE']),
                fontsize=9, ha='center', va='bottom')
ax2.set_xlabel('Median Time (seconds)', fontweight='bold')
ax2.set_ylabel('Mean MAPE (%)', fontweight='bold')
ax2.set_title('Speed-Accuracy Trade-off', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# 3. Sensitivity: Hyperparameter Importance
ax3 = fig.add_subplot(gs[1, :])
sens = pd.read_csv('results/sensitivity/sensitivity_xgboost_agg.csv')
sens_sorted = sens.sort_values('Relative_Importance_Mean', ascending=True)
colors_sens = ['#27ae60' if x < 5 else '#f39c12' if x < 20 else '#e74c3c' 
               for x in sens_sorted['Relative_Importance_Mean']]
ax3.barh(sens_sorted['Parameter'], sens_sorted['Relative_Importance_Mean'], 
         xerr=sens_sorted['Relative_Importance_Std'], color=colors_sens, alpha=0.8)
ax3.set_xlabel('Relative Importance (%)', fontweight='bold', fontsize=12)
ax3.set_title('Hyperparameter Sensitivity Analysis', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(sens_sorted.iterrows()):
    val = row['Relative_Importance_Mean']
    ax3.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Critical (>20%)'),
    Patch(facecolor='#f39c12', label='Moderate (5-20%)'),
    Patch(facecolor='#27ae60', label='Low (<5%)')
]
ax3.legend(handles=legend_elements, loc='lower right')

# 4. Key Statistics Table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

stats_text = """
KEY FINDINGS

Performance Rankings:
  1. HDMR Adaptive: 8.99% MAPE (best accuracy, 74x slower)
  2. Optuna: 9.10% MAPE (best trade-off: near-best + 74x faster!)
  3. Random Search: 9.23% MAPE (good baseline)

Hyperparameter Insights:
  • Critical (82% total): min_child_weight (47%) + max_depth (35%)
  • Moderate (14%): gamma
  • Negligible (<2% each): learning_rate, subsample, colsample_bytree

Practical Implications:
  ✓ Focus tuning on top 2-3 parameters
  ✓ Use Optuna for production (fast + accurate)
  ✓ Use HDMR Adaptive for research (best accuracy)
  ✓ Reduce search space by ~50% (6 → 3 parameters)
"""

ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
        verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('HDMR Optimization: Complete Analysis', 
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig('results/FINAL_SUMMARY_REPORT.png', dpi=300, bbox_inches='tight')
print("✓ Final visualization saved: results/FINAL_SUMMARY_REPORT.png")
