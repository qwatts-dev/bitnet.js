#!/usr/bin/env python3
"""
extract_weights.py

Downloads microsoft/bitnet-b1.58-2B-4T from Hugging Face (safetensors),
extracts the weight matrix from layer 0's MLP down_proj, unpacks the
HF 2-bit row-packed uint8 encoding, then re-packs into our JS kernel's
column-packed uint32 format (16 ternary weights per u32).

HF on-disk format
─────────────────
  Shape: (M/4, K)  uint8  — 4 ternary values per byte, packed along rows.
  Each byte encodes 4 consecutive row weights at a fixed column:
    bits [1:0] → row i*4+0,  bits [3:2] → row i*4+1,
    bits [5:4] → row i*4+2,  bits [7:6] → row i*4+3.

Our JS kernel format
────────────────────
  16 ternary weights per uint32, packed along columns (K dimension).
  Weight encoding (2 bits per weight):
    0b00 → 0   (zero)
    0b01 → +1
    0b10 → −1

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_weights.py
"""

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import math


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
    assert packed_rows * 4 == M, f"Expected {M/4} packed rows, got {packed_rows}"
    assert packed_bytes.shape[1] == K

    weights = np.zeros((M, K), dtype=np.int8)
    b = packed_bytes.astype(np.uint8)

    for i in range(4):
        codes = (b >> (i * 2)) & 0x03  # shape (packed_rows, K)
        vals = codes.astype(np.int8) - np.int8(1)  # 0→-1, 1→0, 2→+1
        start_row = i * packed_rows
        weights[start_row:start_row + packed_rows, :] = vals

    return weights


def pack_weight_matrix(weights: np.ndarray, M: int, K: int):
    """
    Replicates the JavaScript packWeightMatrix function exactly.

    Packs 16 ternary weights per uint32 along the column dimension.
      0b00 → 0,  0b01 → +1,  0b10 → −1.

    Vectorised implementation (processes 16 columns at a time).
    """
    packed_stride = math.ceil(K / 16)
    packed = np.zeros((M, packed_stride), dtype=np.uint32)

    # Pad K to multiple of 16 if needed
    K_padded = packed_stride * 16
    if K_padded != K:
        padded = np.zeros((M, K_padded), dtype=np.int8)
        padded[:, :K] = weights
        weights = padded

    # Process 16 columns at a time → one uint32
    for g in range(packed_stride):
        col_start = g * 16
        block = weights[:, col_start:col_start + 16]  # (M, 16) int8

        word = np.zeros(M, dtype=np.uint32)
        for i in range(16):
            col_vals = block[:, i]  # (M,) int8
            # Encode: +1 → 0b01, -1 → 0b10, 0 → 0b00
            codes = np.where(col_vals == 1, np.uint32(0b01),
                    np.where(col_vals == -1, np.uint32(0b10),
                             np.uint32(0b00)))
            word |= (codes << np.uint32(i * 2))

        packed[:, g] = word

    return packed.flatten(), packed_stride


def main():
    print("=" * 60)
    print("BitNet Weight Extraction Script")
    print("=" * 60)

    # ── 1. Download safetensors ────────────────────────────────
    model_name = "microsoft/bitnet-b1.58-2B-4T"
    print(f"\n[1/5] Downloading {model_name}/model.safetensors ...")
    sf_path = hf_hub_download(model_name, "model.safetensors")
    print(f"      Cached at: {sf_path}")

    # ── 2. Extract packed weight tensor ────────────────────────
    tensor_name = "model.layers.0.mlp.down_proj.weight"
    print(f"\n[2/5] Loading tensor: {tensor_name}")
    tensors = load_file(sf_path)
    packed_tensor = tensors[tensor_name]
    packed_np = packed_tensor.numpy()  # uint8, shape (M/4, K)
    print(f"      Stored shape : {packed_np.shape} (uint8, row-packed)")

    # Real dimensions from config
    M = packed_np.shape[0] * 4  # 4 ternary values per byte
    K = packed_np.shape[1]
    print(f"      Real shape   : ({M}, {K})")
    print(f"      M (rows/out) : {M}")
    print(f"      K (cols/in)  : {K}")

    # ── 3. Unpack HF format → full ternary matrix ─────────────
    print(f"\n[3/5] Unpacking HF row-packed uint8 → ({M}, {K}) int8 ...")
    weights = unpack_hf_weights(packed_np, M, K)
    print(f"      Unpacked shape: {weights.shape}")

    unique_vals = np.unique(weights)
    print(f"      Unique values : {unique_vals}")
    assert set(unique_vals.tolist()).issubset({-1, 0, 1}), \
        f"ERROR: unexpected values {unique_vals}"

    # Distribution
    n_neg  = int(np.sum(weights == -1))
    n_zero = int(np.sum(weights == 0))
    n_pos  = int(np.sum(weights == 1))
    total  = weights.size
    print(f"      Distribution  : -1={n_neg} ({100*n_neg/total:.1f}%)  "
          f"0={n_zero} ({100*n_zero/total:.1f}%)  "
          f"+1={n_pos} ({100*n_pos/total:.1f}%)")

    # ── 4. Re-pack for JS kernel (16 per uint32, along cols) ──
    print(f"\n[4/5] Bit-packing for JS kernel (16 weights/u32, column-packed) ...")
    packed, packed_stride = pack_weight_matrix(weights, M, K)
    print(f"      Packed array length : {len(packed)} uint32 words")
    print(f"      Packed stride (K)   : {packed_stride} words/row")
    print(f"      Packed size         : {packed.nbytes:,} bytes "
          f"({packed.nbytes / 1024:.1f} KB)")

    # ── 5. Save to binary file ─────────────────────────────────
    out_file = "bitnet_layer_0_down_proj.bin"
    print(f"\n[5/5] Saving to {out_file} ...")
    packed.tofile(out_file)
    print(f"      Saved successfully.")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Feed these values to the JS kernel:")
    print(f"  M = {M}")
    print(f"  K = {K}")
    print(f"  packedStride = {packed_stride}")
    print(f"  File: {out_file} ({packed.nbytes:,} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
