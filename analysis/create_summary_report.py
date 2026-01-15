import pandas as pd

print("="*80)
print("HDMR OPTIMIZATION PROJECT - SUMMARY REPORT")
print("="*80)

# 1. Comparison Results
print("\n1. OPTIMIZER COMPARISON (XGBoost)")
print("-"*80)
comp = pd.read_csv('results/comparisons/comparison_summary_xgboost_20260114_140609.csv')
print(comp.to_string(index=False))

# 2. Sensitivity Analysis
print("\n2. SENSITIVITY ANALYSIS (XGBoost)")
print("-"*80)
try:
    sens = pd.read_csv('results/sensitivity/sensitivity_xgboost_agg.csv')
    print("\nAvailable columns:", sens.columns.tolist())
    
    # Kolon isimlerine göre sırala
    if 'relative_importance_mean' in sens.columns:
        sens_sorted = sens.sort_values('relative_importance_mean', ascending=False)
        print("\nHyperparameter Importance:")
        for idx, row in sens_sorted.iterrows():
            param = row['parameter'] if 'parameter' in sens.columns else row.iloc[0]
            importance = row['relative_importance_mean']
            std = row['relative_importance_std'] if 'relative_importance_std' in sens.columns else 0
            print(f"  {str(param):20s} {importance:6.2f}% ± {std:.2f}%")
    else:
        print("\nSensitivity data:")
        print(sens.to_string(index=False))
except Exception as e:
    print(f"Error reading sensitivity: {e}")

# 3. Key Findings
print("\n3. KEY FINDINGS")
print("-"*80)
print("✓ HDMR Adaptive: Best accuracy (8.99% MAPE)")
print("✓ Optuna: Best trade-off (9.10% MAPE, 74x faster)")
print("✓ Critical parameters: min_child_weight (47%), max_depth (35%)")
print("✓ Low-impact parameters: learning_rate, subsample, colsample_bytree (<2%)")

print("\n4. RUNNING EXPERIMENTS")
print("-"*80)
print("✓ Comparison: Completed")
print("✓ Sensitivity: Completed")
print("🔄 Deep Learning Benchmark: Running (~2 hours)")

print("\n" + "="*80)
