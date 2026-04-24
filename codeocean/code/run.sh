#!/bin/bash
# HDMR-opt: Code Ocean Entry Point Script
set -e

echo "=========================================================="
echo "      HDMR-opt: High Dimensional Model Representation     "
echo "=========================================================="
echo ""

# Ensure output directories exist
mkdir -p /results/logs
mkdir -p /results/tabarena
mkdir -p /results/forecasting
mkdir -p /results/benchmarks

# Study modes: tabular | forecasting | benchmark
STUDY=${STUDY:-tabular}
# Global seed for reproducibility
export SEED=${SEED:-42}

echo "Using SEED: $SEED"

if [ "$STUDY" == "forecasting" ]; then
    echo "Running Industrial Forecasting Study (Payten & Medianova)..."
    export HORIZON=7
    export N_FOLDS=2
    export HDMR_SAMPLES=20
    python scripts/forecast_hpo.py --dataset both --models xgb,lgb --methods hdmr
    
elif [ "$STUDY" == "benchmark" ]; then
    echo "Running Mathematical Benchmark Functions (Table XIV)..."
    echo "Testing Rastrigin-10D..."
    python src/main.py --function Rastrigin --numVariables 10 --adaptive --maxiter 10 --numClosestPoints 50 --seed $SEED
    echo "Testing Ackley-10D..."
    python src/main.py --function Ackley --numVariables 10 --adaptive --maxiter 10 --numClosestPoints 50 --seed $SEED

else
    echo "Running Tabular Benchmark (TabArena)..."
    export DATASETS="46954,46917"
    export N_FOLDS=2
    export HDMR_SAMPLES=50
    python main.py
fi

echo ""
echo "=========================================================="
echo "✓ Task Completed!"
echo "Outputs are saved in /results/"
echo "=========================================================="
