#!/usr/bin/env python3
"""
setup_model.py  –  Master orchestrator

Downloads the BitNet b1.58-2B-4T model from Hugging Face and extracts
all weight assets needed by the WebGPU kernel into the weights/ directory.

Run from the repository root:
    python scripts/setup_model.py

This sequentially invokes the main() function of each extraction script
in the correct order.
"""

import os
import sys
import time

# Ensure the scripts/ directory is importable and that file paths
# resolve relative to the repository root (not the scripts/ folder).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)


def run():
    steps = [
        ("extract_sparse_embeddings", "Sparse Embeddings (FP16)"),
        ("extract_lm_head",           "LM Head (FP16, Vocab-Sliced)"),
        ("extract_all_layers",        "All 30 Transformer Layer Weights"),
        ("extract_rmsnorm",           "RMSNorm Weights"),
        ("extract_sub_norms",         "Sub-LayerNorm Weights"),
        ("extract_weight_scales",     "Per-Layer Weight Scales"),
    ]

    print("=" * 70)
    print("  BitNet WebGPU – Full Model Setup")
    print("=" * 70)
    print(f"  Working directory : {os.getcwd()}")
    print(f"  Steps             : {len(steps)}")
    print("=" * 70)
    print()

    t0 = time.time()

    for i, (module_name, description) in enumerate(steps, 1):
        print(f"\n{'━' * 70}")
        print(f"  Step {i}/{len(steps)}: {description}")
        print(f"{'━' * 70}\n")

        step_t0 = time.time()
        module = __import__(module_name)
        module.main()
        step_elapsed = time.time() - step_t0

        print(f"\n  ✔ Step {i} complete ({step_elapsed:.1f}s)")

    elapsed = time.time() - t0

    print(f"\n\n{'═' * 70}")
    print(f"  ✅ ALL {len(steps)} STEPS COMPLETE  ({elapsed:.1f}s total)")
    print(f"  Weights directory: {os.path.abspath('weights')}/")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    run()
