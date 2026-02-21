#!/usr/bin/env python3
"""
extract_sparse_embeddings.py

Extracts a sparse subset of the BitNet embedding layer in Float16
for efficient browser-side loading (~84 MB instead of 1.3 GB).

Strategy:
  - Take the first 16,384 BPE token rows (IDs 0–16383).
    Since BPE assigns lower IDs to more frequent tokens, this
    covers ~99% of standard English text.
  - Explicitly append the row for token ID 50991 ("WebGPU")
    so we can test domain-specific vocabulary.
  - Convert to float16 (halves memory vs float32).
  - Save a vocab_map.json that maps original token IDs → dense
    row indices in the binary file.

Output files:
  sparse_embeddings.bin  – flat row-major float16, (16385 × 2560)
  vocab_map.json         – { "9906": 9906, "50991": 16384, ... }

Usage:
  pip install torch safetensors huggingface_hub numpy
  python extract_sparse_embeddings.py
"""

import json
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


CONTIGUOUS_SLICE = 16384       # first N tokens (frequency-ordered by BPE)
EXTRA_TOKEN_IDS  = [50991]     # "WebGPU" and any other specific tokens
MODEL_NAME       = "microsoft/bitnet-b1.58-2B-4T"


def main():
    print("=" * 60)
    print("Sparse Embedding Extraction (FP16)")
    print("=" * 60)

    # ── 1. Download safetensors ────────────────────────────────
    print(f"\n[1/5] Downloading {MODEL_NAME}/model.safetensors ...")
    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    print(f"      Cached at: {sf_path}")

    # ── 2. Load embed_tokens ───────────────────────────────────
    tensor_name = "model.embed_tokens.weight"
    print(f"\n[2/5] Loading tensor: {tensor_name}")
    tensors = load_file(sf_path)
    emb = tensors[tensor_name]
    print(f"      Shape: {tuple(emb.shape)}, dtype: {emb.dtype}")

    full_vocab_size, embed_dim = emb.shape
    print(f"      Vocab size : {full_vocab_size:,}")
    print(f"      Embed dim  : {embed_dim}")

    # ── 3. Build the sparse subset ─────────────────────────────
    print(f"\n[3/5] Building sparse subset ...")
    print(f"      Contiguous slice: IDs 0–{CONTIGUOUS_SLICE - 1}")
    print(f"      Extra tokens    : {EXTRA_TOKEN_IDS}")

    # Collect all token IDs we want (deduplicated, ordered)
    token_ids = list(range(CONTIGUOUS_SLICE))
    for tid in EXTRA_TOKEN_IDS:
        if tid not in token_ids:
            token_ids.append(tid)

    total_rows = len(token_ids)
    print(f"      Total rows      : {total_rows}")

    # Build vocab map: original token ID → dense row index
    vocab_map = {str(tid): idx for idx, tid in enumerate(token_ids)}

    # Extract rows and convert to float16
    # Use torch indexing for efficiency, then convert
    import torch
    indices = torch.tensor(token_ids, dtype=torch.long)
    subset = emb[indices].to(torch.float16).numpy()

    print(f"      Subset shape    : {subset.shape}")
    print(f"      Subset dtype    : {subset.dtype}")
    print(f"      Subset size     : {subset.nbytes:,} bytes "
          f"({subset.nbytes / 1024 / 1024:.1f} MB)")

    # ── 4. Sanity checks ──────────────────────────────────────
    print(f"\n[4/5] Sanity checks ...")

    # Check token 9906 ("Hello")
    hello_idx = vocab_map.get("9906")
    if hello_idx is not None:
        hello_vec = subset[hello_idx]
        print(f'      Token 9906 ("Hello") → row {hello_idx}')
        print(f"        First 5 values: {hello_vec[:5]}")
        print(f"        Non-zero: {np.count_nonzero(hello_vec)} / {embed_dim}")
    else:
        print("      WARNING: Token 9906 not in subset!")

    # Check token 50991 ("WebGPU")
    webgpu_idx = vocab_map.get("50991")
    if webgpu_idx is not None:
        webgpu_vec = subset[webgpu_idx]
        print(f'      Token 50991 ("WebGPU") → row {webgpu_idx}')
        print(f"        First 5 values: {webgpu_vec[:5]}")
        print(f"        Non-zero: {np.count_nonzero(webgpu_vec)} / {embed_dim}")
    else:
        print("      WARNING: Token 50991 not in subset!")

    # ── 5. Save output files ──────────────────────────────────
    print(f"\n[5/5] Saving output files ...")

    bin_file = "sparse_embeddings.bin"
    subset.tofile(bin_file)
    print(f"      {bin_file}: {subset.nbytes:,} bytes ({subset.nbytes / 1024 / 1024:.1f} MB)")

    map_file = "vocab_map.json"
    with open(map_file, "w") as f:
        json.dump(vocab_map, f, separators=(",", ":"))
    import os
    map_size = os.path.getsize(map_file)
    print(f"      {map_file}: {map_size:,} bytes ({map_size / 1024:.1f} KB)")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Files ready for browser:")
    print(f"  {bin_file}  ({subset.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  {map_file}  ({map_size / 1024:.1f} KB)")
    print(f"  Rows: {total_rows}  |  Dims: {embed_dim}  |  Precision: float16")
    print("=" * 60)


if __name__ == "__main__":
    main()
