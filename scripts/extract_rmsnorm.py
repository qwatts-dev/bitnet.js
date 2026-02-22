#!/usr/bin/env python3
"""
extract_rmsnorm.py

Extracts the learned RMSNorm weights from the BitNet 1.58B-2B-4T model.

These are the "gamma" scaling vectors that the model learned during training.
Without them, the activations get scrambled as they pass through the 26 layers,
resulting in gibberish output.

We extract:
  1. model.layers.{i}.input_layernorm.weight        (26 files)
  2. model.layers.{i}.post_attention_layernorm.weight (26 files)
  3. model.norm.weight                                (1 file — final norm before LM Head)

All vectors are 1D float32 of dimension 2560 (~10 KB each).

Output:
  weights/bitnet_layer_{i}_attn_norm.bin   (26 files)
  weights/bitnet_layer_{i}_mlp_norm.bin    (26 files)
  weights/bitnet_final_norm.bin            (1 file)

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_rmsnorm.py
"""

import os
import sys
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

NUM_LAYERS  = 30
HIDDEN_DIM  = 2560
OUTPUT_DIR  = "weights"
MODEL_NAME  = "microsoft/bitnet-b1.58-2B-4T"


def main():
    print("=" * 70)
    print("  BitNet RMSNorm Weight Extractor")
    print("=" * 70)
    total_files = NUM_LAYERS * 2 + 1  # 26 attn_norm + 26 mlp_norm + 1 final
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Hidden dim   : {HIDDEN_DIM}")
    print(f"  Layers       : {NUM_LAYERS}")
    print(f"  Total files  : {total_files}")
    print("=" * 70)

    # ── 1. Create output directory ─────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[1/3] Output directory: {os.path.abspath(OUTPUT_DIR)}/")

    # ── 2. Download & load safetensors ─────────────────────────
    print(f"\n[2/3] Downloading {MODEL_NAME}/model.safetensors ...")
    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    print(f"       Cached at: {sf_path}")

    print("       Loading safetensors into memory ...")
    t0 = time.time()
    tensors = load_file(sf_path)
    print(f"       Loaded {len(tensors)} tensors in {time.time() - t0:.1f}s")

    # ── 3. Extract RMSNorm weights ─────────────────────────────
    print(f"\n[3/3] Extracting {total_files} RMSNorm weight vectors ...\n")

    t_start = time.time()
    total_bytes = 0
    file_count = 0

    for i in range(NUM_LAYERS):
        # ── input_layernorm.weight → attn_norm ──
        attn_norm_name = f"model.layers.{i}.input_layernorm.weight"
        attn_norm_path = os.path.join(OUTPUT_DIR, f"bitnet_layer_{i}_attn_norm.bin")

        if attn_norm_name not in tensors:
            print(f"  ❌ Missing tensor: {attn_norm_name}")
            sys.exit(1)

        attn_norm = tensors[attn_norm_name].float().numpy().astype(np.float32).flatten()
        assert attn_norm.shape[0] == HIDDEN_DIM, \
            f"Expected {HIDDEN_DIM} dims, got {attn_norm.shape[0]} for {attn_norm_name}"
        attn_norm.tofile(attn_norm_path)
        attn_bytes = os.path.getsize(attn_norm_path)
        total_bytes += attn_bytes
        file_count += 1

        # ── post_attention_layernorm.weight → mlp_norm ──
        mlp_norm_name = f"model.layers.{i}.post_attention_layernorm.weight"
        mlp_norm_path = os.path.join(OUTPUT_DIR, f"bitnet_layer_{i}_mlp_norm.bin")

        if mlp_norm_name not in tensors:
            print(f"  ❌ Missing tensor: {mlp_norm_name}")
            sys.exit(1)

        mlp_norm = tensors[mlp_norm_name].float().numpy().astype(np.float32).flatten()
        assert mlp_norm.shape[0] == HIDDEN_DIM, \
            f"Expected {HIDDEN_DIM} dims, got {mlp_norm.shape[0]} for {mlp_norm_name}"
        mlp_norm.tofile(mlp_norm_path)
        mlp_bytes = os.path.getsize(mlp_norm_path)
        total_bytes += mlp_bytes
        file_count += 1

        print(f"  Layer {i:>2d}/25: attn_norm ({attn_bytes:,} B) + "
              f"mlp_norm ({mlp_bytes:,} B)  ✔")

    # ── Final norm (model.norm.weight) ──
    final_norm_name = "model.norm.weight"
    final_norm_path = os.path.join(OUTPUT_DIR, "bitnet_final_norm.bin")

    if final_norm_name not in tensors:
        print(f"\n  ❌ Missing tensor: {final_norm_name}")
        sys.exit(1)

    final_norm = tensors[final_norm_name].float().numpy().astype(np.float32).flatten()
    assert final_norm.shape[0] == HIDDEN_DIM, \
        f"Expected {HIDDEN_DIM} dims, got {final_norm.shape[0]} for {final_norm_name}"
    final_norm.tofile(final_norm_path)
    final_bytes = os.path.getsize(final_norm_path)
    total_bytes += final_bytes
    file_count += 1

    print(f"\n  Final norm : {final_bytes:,} B  ✔")

    elapsed = time.time() - t_start

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Files written    : {file_count}")
    print(f"  Total size       : {total_bytes:,} bytes  "
          f"({total_bytes / 1024:.1f} KB)")
    print(f"  Per-vector size  : {HIDDEN_DIM * 4:,} bytes  "
          f"({HIDDEN_DIM} × float32)")
    print(f"  Elapsed time     : {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  Files saved to {os.path.abspath(OUTPUT_DIR)}/")
    print(f"    bitnet_layer_{{0..25}}_attn_norm.bin  (26 files)")
    print(f"    bitnet_layer_{{0..25}}_mlp_norm.bin   (26 files)")
    print(f"    bitnet_final_norm.bin                 (1 file)")
    print("=" * 70)


if __name__ == "__main__":
    main()
