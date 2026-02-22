#!/usr/bin/env python3
"""
extract_attention.py

Downloads microsoft/bitnet-b1.58-2B-4T from Hugging Face (safetensors)
and extracts the four Self-Attention weight matrices for Layer 0:

    q_proj  — Query projection
    k_proj  — Key projection   (may be smaller if GQA is used)
    v_proj  — Value projection  (may be smaller if GQA is used)
    o_proj  — Output projection

Each matrix is unpacked from HF's row-packed uint8 encoding, then
re-packed into our JS kernel's column-packed uint32 format (16 ternary
weights per u32).

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

Output files:
  bitnet_layer_0_q_proj.bin
  bitnet_layer_0_k_proj.bin
  bitnet_layer_0_v_proj.bin
  bitnet_layer_0_o_proj.bin

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_attention.py
"""

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import math


# ═══════════════════════════════════════════════════════════════
# Packing / unpacking helpers  (identical to extract_full_mlp.py)
# ═══════════════════════════════════════════════════════════════

def unpack_hf_weights(packed_bytes: np.ndarray, M: int, K: int) -> np.ndarray:
    """
    Unpack HF's row-packed uint8 tensor into a full (M, K) ternary int8 matrix.

    HF stores (M/4, K) uint8 with 4 weights per byte along the row dimension.
    2-bit codes: 0b00→0, 0b01→+1, 0b10→-1.

    Returns np.ndarray of shape (M, K) with values in {-1, 0, +1}.
    """
    packed_rows = packed_bytes.shape[0]  # M/4
    assert packed_rows * 4 == M, f"Expected {M // 4} packed rows, got {packed_rows}"
    assert packed_bytes.shape[1] == K

    # Vectorised extraction of 4 weights per byte
    weights = np.zeros((M, K), dtype=np.int8)
    b = packed_bytes.astype(np.uint8)

    for i in range(4):
        codes = (b >> (i * 2)) & 0x03  # shape (packed_rows, K)
        # 0b00→0, 0b01→+1, 0b10→-1
        vals = np.where(codes == 1, np.int8(1),
               np.where(codes == 2, np.int8(-1), np.int8(0)))
        weights[i::4, :] = vals

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


# ═══════════════════════════════════════════════════════════════
# Layer 0 Self-Attention tensors to extract
# ═══════════════════════════════════════════════════════════════
#
# NOTE: We read M and K directly from the safetensors file so that
#       Grouped-Query Attention (GQA) dimensions are handled
#       automatically — K_proj and V_proj may be narrower than Q_proj.

LAYER_0_ATTN_TENSORS = [
    {
        "tensor_name": "model.layers.0.self_attn.q_proj.weight",
        "out_file":    "bitnet_layer_0_q_proj.bin",
        "desc":        "q_proj (Query projection)",
    },
    {
        "tensor_name": "model.layers.0.self_attn.k_proj.weight",
        "out_file":    "bitnet_layer_0_k_proj.bin",
        "desc":        "k_proj (Key projection)",
    },
    {
        "tensor_name": "model.layers.0.self_attn.v_proj.weight",
        "out_file":    "bitnet_layer_0_v_proj.bin",
        "desc":        "v_proj (Value projection)",
    },
    {
        "tensor_name": "model.layers.0.self_attn.o_proj.weight",
        "out_file":    "bitnet_layer_0_o_proj.bin",
        "desc":        "o_proj (Output projection)",
    },
]


def extract_and_pack(tensors: dict, spec: dict, index: int, total: int):
    """
    Extract a single tensor from the safetensors dict, unpack HF format,
    re-pack to JS kernel format, and save to disk.

    Reads M and K from the on-disk tensor shape so GQA dimensions are
    discovered automatically.
    """
    tensor_name = spec["tensor_name"]
    out_file    = spec["out_file"]
    desc        = spec["desc"]

    # ── Load packed tensor & derive real dimensions ───────────
    packed_tensor = tensors[tensor_name]
    packed_np = packed_tensor.numpy()  # uint8, shape (M/4, K)

    # HF packs 4 ternary weights per byte along the row dimension
    M = packed_np.shape[0] * 4
    K = packed_np.shape[1]

    print(f"\n{'─' * 60}")
    print(f"  [{index}/{total}]  {desc}")
    print(f"  Tensor : {tensor_name}")
    print(f"  Stored shape : {packed_np.shape} (uint8, row-packed)")
    print(f"  Real shape   : ({M}, {K})  →  M={M}  K={K}")
    print(f"{'─' * 60}")

    # ── Unpack HF format → full ternary matrix ────────────────
    print(f"  Unpacking HF row-packed uint8 → ({M}, {K}) int8 ...")
    weights = unpack_hf_weights(packed_np, M, K)
    print(f"  Unpacked shape: {weights.shape}")

    unique_vals = np.unique(weights)
    print(f"  Unique values : {unique_vals}")
    assert set(unique_vals.tolist()).issubset({-1, 0, 1}), \
        f"ERROR: unexpected values {unique_vals}"

    # Distribution
    n_neg  = int(np.sum(weights == -1))
    n_zero = int(np.sum(weights == 0))
    n_pos  = int(np.sum(weights == 1))
    total_w = weights.size
    print(f"  Distribution  : -1={n_neg} ({100*n_neg/total_w:.1f}%)  "
          f"0={n_zero} ({100*n_zero/total_w:.1f}%)  "
          f"+1={n_pos} ({100*n_pos/total_w:.1f}%)")

    # ── Re-pack for JS kernel (16 per uint32, along cols) ─────
    print(f"  Bit-packing for JS kernel (16 weights/u32, column-packed) ...")
    packed, packed_stride = pack_weight_matrix(weights, M, K)
    print(f"  Packed array length : {len(packed)} uint32 words")
    print(f"  Packed stride (K)   : {packed_stride} words/row")
    print(f"  Packed size         : {packed.nbytes:,} bytes "
          f"({packed.nbytes / 1024:.1f} KB)")

    # ── Save to binary file ───────────────────────────────────
    print(f"  Saving to {out_file} ...")
    packed.tofile(out_file)
    print(f"  ✓ Saved successfully.")

    return {
        "name": desc,
        "file": out_file,
        "M": M,
        "K": K,
        "packed_stride": packed_stride,
        "size_bytes": packed.nbytes,
    }


def main():
    print("=" * 60)
    print("BitNet Self-Attention Weight Extraction (Layer 0)")
    print("  q_proj + k_proj + v_proj + o_proj")
    print("=" * 60)

    # ── 1. Download safetensors ────────────────────────────────
    model_name = "microsoft/bitnet-b1.58-2B-4T"
    print(f"\n[Step 1] Downloading {model_name}/model.safetensors ...")
    sf_path = hf_hub_download(model_name, "model.safetensors")
    print(f"         Cached at: {sf_path}")

    # ── 2. Load all tensors once ───────────────────────────────
    print(f"\n[Step 2] Loading safetensors file ...")
    tensors = load_file(sf_path)
    print(f"         Loaded {len(tensors)} tensors.")

    # ── 3. Extract each attention weight ───────────────────────
    print(f"\n[Step 3] Extracting & packing {len(LAYER_0_ATTN_TENSORS)} "
          f"attention weight matrices ...")

    results = []
    for idx, spec in enumerate(LAYER_0_ATTN_TENSORS, start=1):
        info = extract_and_pack(tensors, spec, idx, len(LAYER_0_ATTN_TENSORS))
        results.append(info)

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Summary for JS kernel / WGSL shader integration:")
    print("=" * 60)
    total_bytes = 0
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    File         : {r['file']}")
        print(f"    M (rows/out) : {r['M']}")
        print(f"    K (cols/in)  : {r['K']}")
        print(f"    packedStride : {r['packed_stride']}")
        print(f"    Size         : {r['size_bytes']:,} bytes ({r['size_bytes']/1024:.1f} KB)")
        total_bytes += r["size_bytes"]

    print(f"\n  Total attention weight data: {total_bytes:,} bytes "
          f"({total_bytes/1024:.1f} KB, {total_bytes/(1024*1024):.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
