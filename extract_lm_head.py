#!/usr/bin/env python3
"""
extract_lm_head.py

Extracts the Language Model Head (`lm_head.weight`) from the BitNet
safetensors, applies the EXACT same vocab-slice used for the sparse
embeddings (first 16,384 rows + row 50991 for "WebGPU"), converts
to float16, and saves as `sparse_lm_head.bin`.

The LM Head is a dense (non-ternary) matrix that maps the hidden
representation (2560 dims) to logits over the vocabulary.

Output:
  sparse_lm_head.bin  – flat row-major float16, (16385 × 2560)

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_lm_head.py
"""

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch


CONTIGUOUS_SLICE = 16384       # first N tokens (same as embeddings)
EXTRA_TOKEN_IDS  = [50991]     # "WebGPU" token
MODEL_NAME       = "microsoft/bitnet-b1.58-2B-4T"
EXPECTED_HIDDEN  = 2560        # hidden_size


def main():
    print("=" * 60)
    print("LM Head Extraction (FP16, Vocab-Sliced)")
    print("=" * 60)

    # ── 1. Download safetensors ────────────────────────────────
    print(f"\n[1/5] Downloading {MODEL_NAME}/model.safetensors ...")
    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    print(f"      Cached at: {sf_path}")

    # ── 2. Load lm_head weights ──────────────────────────────────
    # BitNet uses tied embeddings: embed_tokens.weight IS the LM head.
    # There is no separate "lm_head.weight" tensor in the safetensors.
    tensor_name = "model.embed_tokens.weight"
    print(f"\n[2/5] Loading tensor: {tensor_name}  (tied LM head)")
    tensors = load_file(sf_path)
    lm_head = tensors[tensor_name]
    print(f"      Shape: {tuple(lm_head.shape)}, dtype: {lm_head.dtype}")
    print(f"      (Model uses weight-tying: embed_tokens == lm_head)")

    full_vocab_size, hidden_dim = lm_head.shape
    print(f"      Vocab size  : {full_vocab_size:,}")
    print(f"      Hidden dim  : {hidden_dim}")

    assert hidden_dim == EXPECTED_HIDDEN, \
        f"Expected hidden_dim={EXPECTED_HIDDEN}, got {hidden_dim}"

    # ── 3. Apply the EXACT same vocab-slice as embeddings ──────
    print(f"\n[3/5] Applying vocab-slice ...")
    print(f"      Contiguous rows: 0–{CONTIGUOUS_SLICE - 1}")
    print(f"      Extra token IDs: {EXTRA_TOKEN_IDS}")

    # Build token ID list (same ordering as sparse_embeddings)
    token_ids = list(range(CONTIGUOUS_SLICE))
    for tid in EXTRA_TOKEN_IDS:
        if tid not in token_ids:
            token_ids.append(tid)

    total_rows = len(token_ids)
    print(f"      Total rows     : {total_rows}")

    # Extract sliced rows and convert to float16
    indices = torch.tensor(token_ids, dtype=torch.long)
    subset = lm_head[indices].to(torch.float16).numpy()

    print(f"      Subset shape   : {subset.shape}")
    print(f"      Subset dtype   : {subset.dtype}")
    print(f"      Subset size    : {subset.nbytes:,} bytes "
          f"({subset.nbytes / 1024 / 1024:.1f} MB)")

    # ── 4. Sanity checks ──────────────────────────────────────
    print(f"\n[4/5] Sanity checks ...")

    # Row 0 (token ID 0)
    row0 = subset[0]
    print(f"      Row 0 first 5 values : {row0[:5]}")
    print(f"      Row 0 non-zero       : {np.count_nonzero(row0)} / {hidden_dim}")

    # Last row (token ID 50991 "WebGPU")
    webgpu_row = subset[-1]
    print(f'      Row {total_rows-1} ("WebGPU") first 5: {webgpu_row[:5]}')
    print(f"      Row {total_rows-1} non-zero   : {np.count_nonzero(webgpu_row)} / {hidden_dim}")

    # Check for NaN / Inf
    nan_count = np.count_nonzero(np.isnan(subset))
    inf_count = np.count_nonzero(np.isinf(subset))
    print(f"      NaN count      : {nan_count}")
    print(f"      Inf count      : {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("      WARNING: NaN/Inf detected in LM head weights!")

    # ── 5. Save output ─────────────────────────────────────────
    out_file = "sparse_lm_head.bin"
    print(f"\n[5/5] Saving to {out_file} ...")
    subset.tofile(out_file)

    import os
    file_size = os.path.getsize(out_file)
    print(f"      File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — LM Head weights ready for browser:")
    print(f"  {out_file}  ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Rows: {total_rows}  |  Cols: {hidden_dim}  |  Precision: float16")
    print(f"  Expected shape: ({total_rows}, {hidden_dim})")
    print(f"  Expected bytes: {total_rows * hidden_dim * 2:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
