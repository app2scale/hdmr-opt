#!/bin/bash
# =============================================================================
# HDMR-opt: Code Ocean Master Entry Point
# =============================================================================
# Author: HDMR Research Group
# Purpose: Orchestrates experiments via the universal main.py dispatcher.
# =============================================================================

set -euo pipefail

# Ensure we are in the code directory
cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Default Environment Variables
# -----------------------------------------------------------------------------
export STUDY=${STUDY:-tabular}
export SEED=${SEED:-42}
export N_FOLDS=${N_FOLDS:-2}
export HDMR_SAMPLES=${HDMR_SAMPLES:-200}

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------
echo "[INFO] Initializing environment..."
mkdir -p /results/logs /results/tabarena /results/forecasting /results/benchmarks

echo "----------------------------------------------------------"
echo "  HDMR-opt Execution Started: $(date)"
echo "  Study Mode : $STUDY"
echo "  Global Seed: $SEED"
echo "----------------------------------------------------------"

# -----------------------------------------------------------------------------
# Main Dispatcher Call
# -----------------------------------------------------------------------------
# We use the universal main.py to handle different study modes.
# Additional arguments can be passed through environment variables.

case "$STUDY" in
    "forecasting")
        python main.py --study forecasting --dataset both --models xgb,lgb --methods hdmr
        ;;

    "benchmark")
        # Run core mathematical functions from Table XIV
        python main.py --study benchmark --function Rastrigin --numVariables 10 --adaptive --maxiter 10 --seed "$SEED"
        python main.py --study benchmark --function Ackley --numVariables 10 --adaptive --maxiter 10 --seed "$SEED"
        ;;

    "tabular")
        python main.py --study tabular
        ;;

    *)
        echo "[ERROR] Unknown STUDY mode: $STUDY"
        exit 1
        ;;
esac

echo ""
echo "----------------------------------------------------------"
echo "✓ SUCCESS: All tasks completed successfully."
echo "Results Location: /results/"
echo "End Time: $(date)"
echo "----------------------------------------------------------"
