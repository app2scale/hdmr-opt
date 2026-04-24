#!/bin/bash
# =============================================================================
# test_capsule.sh — Rapid Verification Script
# =============================================================================
# Purpose: Verifies that the Code Ocean capsule is configured correctly by
#          running minimal "smoke tests" for each research mode.
# =============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[TEST]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Environment Verification
# -----------------------------------------------------------------------------
log "Checking Python environment..."
python3 --version
python3 -c "import numpy; import scipy; import pandas" || { error "Missing core scientific libraries"; exit 1; }
python3 -c "import optuna; import xgboost; import lightgbm" 2>/dev/null || warn "Some ML libraries (Optuna/XGB/LGB) are missing locally. Full smoke tests might fail but basic logic can still be verified."
log "Scientific base found."

# 2. Path & Import Verification
# -----------------------------------------------------------------------------
log "Verifying internal paths and imports..."
cd "$(dirname "$0")"
python3 -c "from src.main import HDMROptimizer; print('  - src.main: OK')"
python3 -c "from src.functions import testfunc_2d; print('  - src.functions: OK')"
python3 -c "from src.functions_forecast import calculate_metrics; print('  - src.functions_forecast: OK')"

# 3. Mode Smoke Tests
# -----------------------------------------------------------------------------
# We run minimal versions of each study (few samples, few folds)

# A. Benchmark Mode (Math functions)
log "Testing Benchmark Mode (Math)..."
STUDY=benchmark SEED=42 python3 main.py --study benchmark --function rastrigin_2d --numVariables 2 --maxiter 2 --numSamples 50
log "Benchmark test passed."

# B. Tabular Mode (XGBoost Smoke)
log "Testing Tabular Mode (Smoke)..."
# DID 46917 = Concrete, N_FOLDS=2, HDMR_SAMPLES=20
DATASETS=46917 N_FOLDS=2 HDMR_SAMPLES=20 python3 main.py --study tabular --task_filter regression
log "Tabular test passed."

# C. Forecasting Mode (XGBoost Smoke)
log "Testing Forecasting Mode (Smoke)..."
# Horizon=7, folds=1, samples=20
HORIZON=7 N_FOLDS=1 HDMR_SAMPLES=20 python3 main.py --study forecasting --dataset payten --models xgb --methods hdmr
log "Forecasting test passed."

echo ""
echo "=========================================================="
log "ALL SMOKE TESTS COMPLETED SUCCESSFULLY!"
echo "The capsule is ready for production-scale experiments."
echo "=========================================================="
