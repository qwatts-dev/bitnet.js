#!/usr/bin/env python3
"""
extract_sub_norms.py

Extracts the BitNet-specific sub-layer RMSNorm weights that sit INSIDE
the attention and MLP blocks:

  1. model.layers.{i}.self_attn.attn_sub_norm.weight   (26 files, dim 2560)
     Applied between GQA attention output and o_proj.

  2. model.layers.{i}.mlp.ffn_sub_norm.weight           (26 files, dim 6912)
     Applied between ReLU²·mul activation and down_proj.

These intermediate norms are unique to the BitNet 1.58-bit architecture
and keep activation magnitudes controlled between sub-blocks.  Without
them the model produces garbled BPE fragments.

Output:
  weights/bitnet_layer_{i}_attn_sub_norm.bin   (26 files, 2560 × f32 = 10 KB each)
  weights/bitnet_layer_{i}_ffn_sub_norm.bin    (26 files, 6912 × f32 = 27 KB each)

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_sub_norms.py
"""

import os
import sys
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

NUM_LAYERS  = 30
ATTN_DIM    = 2560   # Q_DIM = NUM_Q_HEADS * HEAD_DIM
MLP_DIM     = 6912   # intermediate_size
OUTPUT_DIR  = "weights"
MODEL_NAME  = "microsoft/bitnet-b1.58-2B-4T"


def main():
    print("=" * 70)
    print("  BitNet Sub-Norm Weight Extractor")
    print("=" * 70)
    total_files = NUM_LAYERS * 2
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Attn sub-norm: dim {ATTN_DIM}  ({NUM_LAYERS} files)")
    print(f"  FFN sub-norm : dim {MLP_DIM}  ({NUM_LAYERS} files)")
    print(f"  Total files  : {total_files}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download & load safetensors
    print(f"\n[1/3] Downloading {MODEL_NAME}/model.safetensors ...")
    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    print(f"       Cached at: {sf_path}")

    print("       Loading safetensors into memory ...")
    t0 = time.time()
    tensors = load_file(sf_path)
    print(f"       Loaded {len(tensors)} tensors in {time.time() - t0:.1f}s")

    # Extract sub-norms
    print(f"\n[2/3] Extracting {total_files} sub-norm weight vectors ...\n")

    t_start = time.time()
    total_bytes = 0
    file_count = 0

    for i in range(NUM_LAYERS):
        # ── attn_sub_norm ──
        attn_name = f"model.layers.{i}.self_attn.attn_sub_norm.weight"
        attn_path = os.path.join(OUTPUT_DIR, f"bitnet_layer_{i}_attn_sub_norm.bin")

        if attn_name not in tensors:
            print(f"  ❌ Missing tensor: {attn_name}")
            sys.exit(1)

        attn_w = tensors[attn_name].float().numpy().astype(np.float32).flatten()
        assert attn_w.shape[0] == ATTN_DIM, \
            f"Expected {ATTN_DIM} dims, got {attn_w.shape[0]} for {attn_name}"
        attn_w.tofile(attn_path)
        attn_bytes = os.path.getsize(attn_path)
        total_bytes += attn_bytes
        file_count += 1

        # ── ffn_sub_norm ──
        ffn_name = f"model.layers.{i}.mlp.ffn_sub_norm.weight"
        ffn_path = os.path.join(OUTPUT_DIR, f"bitnet_layer_{i}_ffn_sub_norm.bin")

        if ffn_name not in tensors:
            print(f"  ❌ Missing tensor: {ffn_name}")
            sys.exit(1)

        ffn_w = tensors[ffn_name].float().numpy().astype(np.float32).flatten()
        assert ffn_w.shape[0] == MLP_DIM, \
            f"Expected {MLP_DIM} dims, got {ffn_w.shape[0]} for {ffn_name}"
        ffn_w.tofile(ffn_path)
        ffn_bytes = os.path.getsize(ffn_path)
        total_bytes += ffn_bytes
        file_count += 1

        print(f"  Layer {i:2d}  attn_sub_norm: {attn_bytes:,} bytes  "
              f"ffn_sub_norm: {ffn_bytes:,} bytes")

    elapsed = time.time() - t_start

    # Summary
    print(f"\n[3/3] Done!")
    print(f"  Files written : {file_count}")
    print(f"  Total bytes   : {total_bytes:,} ({total_bytes / 1024:.1f} KB)")
    print(f"  Time          : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
