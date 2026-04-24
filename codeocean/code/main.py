#!/usr/bin/env python3
"""
HDMR-opt: High-Dimensional Model Representation for Optimization
================================================================
Universal Entry Point for the Code Ocean Compute Capsule.

This script acts as a dispatcher for the three main research studies:
1. Tabular Benchmark (TabArena datasets)
2. Industrial Forecasting (Payten & Medianova)
3. Mathematical Benchmarks (Test functions)

Usage:
    python main.py --study tabular
    python main.py --study forecasting
    python main.py --study benchmark
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_script(script_path, args=None):
    """Run a sub-script with appropriate environment and arguments."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n>>> Executing: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Pass current environment to subprocess
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Script failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="HDMR-opt Research Capsule Entry Point")
    parser.add_argument(
        "--study", 
        type=str, 
        default="tabular",
        choices=["tabular", "forecasting", "benchmark"],
        help="Select the research study to execute."
    )
    
    # Allow passing unknown args to the sub-scripts
    args, unknown = parser.parse_known_args()
    
    root = Path(__file__).resolve().parent
    scripts_dir = root / "scripts"
    
    if args.study == "tabular":
        # Default to LightGBM study as main tabular result
        script = scripts_dir / "tabarena_hdmr_lgb.py"
        run_script(script, unknown)
        
    elif args.study == "forecasting":
        script = scripts_dir / "forecast_hpo.py"
        run_script(script, unknown)
        
    elif args.study == "benchmark":
        # Use the core library CLI for mathematical benchmarks
        script = root / "src" / "main.py"
        run_script(script, unknown)

if __name__ == "__main__":
    main()
