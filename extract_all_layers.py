#!/usr/bin/env python3
"""
extract_all_layers.py

Extracts ALL 30 transformer layers (layers 0–29) of the BitNet 1.58B-2B-4T
model from Hugging Face safetensors. For each layer, the following 7 weight
matrices are unpacked and re-packed:

  Self-Attention:
    q_proj, k_proj, v_proj, o_proj

  MLP (SwiGLU):
    gate_proj, up_proj, down_proj

HF on-disk format
─────────────────
  Shape: (M/4, K)  uint8  — 4 ternary values per byte, packed along rows.
  Packed in contiguous blocks (NOT interleaved):
    bits [1:0] → rows 0..M/4-1,  bits [3:2] → rows M/4..M/2-1,
    bits [5:4] → rows M/2..3M/4-1,  bits [7:6] → rows 3M/4..M-1.
  Values are stored as (ternary + 1): 0→-1, 1→0, 2→+1.

Our JS kernel format
────────────────────
  16 ternary weights per uint32, packed along columns (K dimension).
  Weight encoding (2 bits per weight):
    0b00 → 0   (zero)
    0b01 → +1
    0b10 → −1

Output:
  weights/bitnet_layer_{i}_{proj_name}.bin   (182 files total)

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_all_layers.py
"""

import os
import sys
import time
import math
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

NUM_LAYERS = 30
OUTPUT_DIR = "weights"

# The 7 projection names per layer (tensor suffix → output file suffix)
PROJECTIONS = [
    {"suffix": "self_attn.q_proj",  "desc": "q_proj  (Query)"},
    {"suffix": "self_attn.k_proj",  "desc": "k_proj  (Key)"},
    {"suffix": "self_attn.v_proj",  "desc": "v_proj  (Value)"},
    {"suffix": "self_attn.o_proj",  "desc": "o_proj  (Output)"},
    {"suffix": "mlp.gate_proj",     "desc": "gate_proj (SwiGLU gate)"},
    {"suffix": "mlp.up_proj",       "desc": "up_proj   (SwiGLU up)"},
    {"suffix": "mlp.down_proj",     "desc": "down_proj (projection back)"},
]


# ═══════════════════════════════════════════════════════════════
# Packing / unpacking helpers
# ═══════════════════════════════════════════════════════════════

def unpack_hf_weights(packed_bytes: np.ndarray, M: int, K: int) -> np.ndarray:
    """
    Unpack HF's row-packed uint8 tensor into a full (M, K) ternary int8 matrix.

    HF stores (M/4, K) uint8 with 4 weights per byte along the row dimension.
    Packed in contiguous blocks: bits[1:0] → rows 0..M/4-1,
    bits[3:2] → rows M/4..M/2-1, etc.
    2-bit codes after subtracting 1: 0→-1, 1→0, 2→+1.

    Returns np.ndarray of shape (M, K) with values in {-1, 0, +1}.
    """
    packed_rows = packed_bytes.shape[0]  # M/4
    assert packed_rows * 4 == M, f"Expected {M // 4} packed rows, got {packed_rows}"
    assert packed_bytes.shape[1] == K

    weights = np.zeros((M, K), dtype=np.int8)
    b = packed_bytes.astype(np.uint8)

    for i in range(4):
        codes = (b >> (i * 2)) & 0x03
        vals = codes.astype(np.int8) - np.int8(1)  # 0→-1, 1→0, 2→+1
        start_row = i * packed_rows
        weights[start_row:start_row + packed_rows, :] = vals

    return weights


def pack_weight_matrix(weights: np.ndarray, M: int, K: int):
    """
    Packs 16 ternary weights per uint32 along the column dimension.
      0b00 → 0,  0b01 → +1,  0b10 → −1.

    Returns (flat_packed, packed_stride).
    """
    packed_stride = math.ceil(K / 16)
    packed = np.zeros((M, packed_stride), dtype=np.uint32)

    K_padded = packed_stride * 16
    if K_padded != K:
        padded = np.zeros((M, K_padded), dtype=np.int8)
        padded[:, :K] = weights
        weights = padded

    for g in range(packed_stride):
        col_start = g * 16
        block = weights[:, col_start:col_start + 16]

        word = np.zeros(M, dtype=np.uint32)
        for i in range(16):
            col_vals = block[:, i]
            codes = np.where(col_vals == 1, np.uint32(0b01),
                    np.where(col_vals == -1, np.uint32(0b10),
                             np.uint32(0b00)))
            word |= (codes << np.uint32(i * 2))

        packed[:, g] = word

    return packed.flatten(), packed_stride


# ═══════════════════════════════════════════════════════════════
# Single-tensor extraction
# ═══════════════════════════════════════════════════════════════

def extract_and_pack(tensors: dict, tensor_name: str, out_path: str,
                     desc: str, progress_label: str) -> dict:
    """
    Extract one tensor, unpack HF format, repack to JS kernel format, save.
    Derives M and K automatically from the on-disk tensor shape.
    """
    packed_tensor = tensors[tensor_name]
    packed_np = packed_tensor.numpy()  # uint8, shape (M/4, K)

    M = packed_np.shape[0] * 4
    K = packed_np.shape[1]

    print(f"    {progress_label}  {desc:<30s}  ({M:>5d} × {K:>5d})", end="", flush=True)

    # Unpack
    weights = unpack_hf_weights(packed_np, M, K)

    # Validate
    unique_vals = set(np.unique(weights).tolist())
    assert unique_vals.issubset({-1, 0, 1}), f"Unexpected values in {tensor_name}"

    # Repack
    packed, packed_stride = pack_weight_matrix(weights, M, K)

    # Save
    packed.tofile(out_path)
    size_kb = packed.nbytes / 1024
    print(f"  →  {size_kb:>8.1f} KB  ✓")

    return {
        "name": desc,
        "file": out_path,
        "M": M,
        "K": K,
        "packed_stride": packed_stride,
        "size_bytes": packed.nbytes,
    }


# ═══════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════

def main():
    total_matrices = NUM_LAYERS * len(PROJECTIONS)  # 26 × 7 = 182

    print("=" * 70)
    print("  BitNet 1.58B-2B-4T  —  Full 26-Layer Weight Extraction")
    print(f"  {NUM_LAYERS} layers × {len(PROJECTIONS)} projections = "
          f"{total_matrices} weight matrices")
    print("=" * 70)

    # ── 1. Create output directory ─────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[1/3] Output directory: {os.path.abspath(OUTPUT_DIR)}/")

    # ── 2. Download & load safetensors ─────────────────────────
    model_name = "microsoft/bitnet-b1.58-2B-4T"
    print(f"\n[2/3] Downloading {model_name}/model.safetensors ...")
    sf_path = hf_hub_download(model_name, "model.safetensors")
    print(f"       Cached at: {sf_path}")

    print(f"       Loading safetensors into memory ...")
    t0 = time.time()
    tensors = load_file(sf_path)
    print(f"       Loaded {len(tensors)} tensors in {time.time() - t0:.1f}s")

    # ── 3. Extract all layers ──────────────────────────────────
    print(f"\n[3/3] Extracting & packing {total_matrices} matrices ...\n")

    all_results = []
    global_idx = 0
    t_start = time.time()

    for layer_i in range(NUM_LAYERS):
        layer_t0 = time.time()
        print(f"  ╔══ Layer {layer_i:>2d} / {NUM_LAYERS - 1} "
              f"{'═' * 46}╗")

        layer_results = []
        for proj in PROJECTIONS:
            global_idx += 1
            suffix = proj["suffix"]
            desc = proj["desc"]

            tensor_name = f"model.layers.{layer_i}.{suffix}.weight"

            # Build a short filename: e.g. "self_attn.q_proj" → "q_proj"
            short_name = suffix.split(".")[-1]
            out_filename = f"bitnet_layer_{layer_i}_{short_name}.bin"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            progress = f"[{global_idx:>3d}/{total_matrices}]"
            info = extract_and_pack(tensors, tensor_name, out_path,
                                    desc, progress)
            layer_results.append(info)

        layer_bytes = sum(r["size_bytes"] for r in layer_results)
        layer_elapsed = time.time() - layer_t0
        print(f"  ╚══ Layer {layer_i:>2d} done: "
              f"{layer_bytes / (1024 * 1024):.2f} MB  "
              f"({layer_elapsed:.1f}s)")
        print()

        all_results.extend(layer_results)

    elapsed = time.time() - t_start

    # ── Summary ────────────────────────────────────────────────
    total_bytes = sum(r["size_bytes"] for r in all_results)
    total_mb = total_bytes / (1024 * 1024)

    print("=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Layers extracted : {NUM_LAYERS}")
    print(f"  Matrices written : {len(all_results)}")
    print(f"  Output directory : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Total weight data: {total_bytes:,} bytes  "
          f"({total_mb:.2f} MB)")
    print(f"  Elapsed time     : {elapsed:.1f}s")
    print("=" * 70)

    # Per-layer breakdown
    print("\n  Per-layer size breakdown:")
    for layer_i in range(NUM_LAYERS):
        layer_slice = all_results[layer_i * 7 : (layer_i + 1) * 7]
        layer_bytes = sum(r["size_bytes"] for r in layer_slice)
        bar_len = int(layer_bytes / (1024 * 1024) * 2)  # ~2 chars per MB
        bar = "█" * bar_len
        print(f"    Layer {layer_i:>2d}: {layer_bytes / (1024 * 1024):>6.2f} MB  {bar}")

    print(f"\n  Grand total: {total_mb:.2f} MB across {len(all_results)} files")
    print("  All files saved to weights/bitnet_layer_{{i}}_{{proj}}.bin")
    print("=" * 70)


if __name__ == "__main__":
    main()
