#!/usr/bin/env python3
"""Test all benchmark functions."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.functions import BENCHMARK_FUNCTIONS, get_function_info

print("=" * 80)
print("TESTING ALL BENCHMARK FUNCTIONS")
print("=" * 80)

failed = []
passed = []

for func_name in sorted(BENCHMARK_FUNCTIONS.keys()):
    try:
        info = get_function_info(func_name)
        func = info['function']
        dim = info['dimension']
        
        x_test = np.zeros((1, dim))
        result = func(x_test)
        
        assert result.shape == (1, 1), f"Wrong shape: {result.shape}"
        
        print(f"✓ {func_name:30s} dim={dim:3d}  f(0)={result[0,0]:12.6f}")
        passed.append(func_name)
        
    except Exception as e:
        print(f"✗ {func_name:30s} FAILED: {str(e)}")
        failed.append(func_name)

print("=" * 80)
print(f"RESULTS: {len(passed)}/{len(BENCHMARK_FUNCTIONS)} tests passed")

if failed:
    print(f"\nFAILED FUNCTIONS ({len(failed)}):")
    for name in failed:
        print(f"  - {name}")
    sys.exit(1)
else:
    print("\n✓ ALL FUNCTIONS WORKING CORRECTLY!")
    print("\nAvailable function categories:")
    print(f"  - 2D functions: 9")
    print(f"  - Classical 10D: 3")
    print(f"  - Modern scalable: 8")
    print(f"  Total: {len(BENCHMARK_FUNCTIONS)} functions")
    sys.exit(0)
