#!/usr/bin/env python3
"""
extract_weight_scales.py

Extracts the BitLinear `weight_scale` scalar from every ternary projection
in the BitNet b1.58-2B-4T model.  Each projection has a learned scalar
(shape [1], BFloat16) that must be applied after the ternary mat-vec to
restore proper output magnitudes.

Output:
  weights/bitnet_layer_scales.json

  {
    "0": {
      "q_proj": 1.21875,
      "k_proj": 1.796875,
      ...
    },
    "1": { ... },
    ...
    "25": { ... }
  }

Usage:
  pip install torch safetensors huggingface_hub
  python extract_weight_scales.py
"""

import os
import json
import time
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


NUM_LAYERS = 30
OUTPUT_DIR = "weights"
MODEL_NAME = "microsoft/bitnet-b1.58-2B-4T"

PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def main():
    print("=" * 70)
    print("  BitNet Weight Scale Extractor")
    print("=" * 70)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Layers      : {NUM_LAYERS}")
    print(f"  Projections : {len(PROJECTIONS)} per layer")
    print(f"  Total scales: {NUM_LAYERS * len(PROJECTIONS)}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[1/3] Downloading {MODEL_NAME}/model.safetensors ...")
    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    print(f"       Cached at: {sf_path}")

    print("       Loading safetensors into memory ...")
    t0 = time.time()
    tensors = load_file(sf_path)
    print(f"       Loaded {len(tensors)} tensors in {time.time() - t0:.1f}s")

    print(f"\n[2/3] Extracting weight scales ...\n")

    scales = {}
    for i in range(NUM_LAYERS):
        layer_scales = {}
        for proj in PROJECTIONS:
            tensor_name = f"model.layers.{i}.{proj}.weight_scale"
            if tensor_name not in tensors:
                print(f"  ❌ Missing: {tensor_name}")
                continue
            # Convert BFloat16 → Python float
            val = tensors[tensor_name].float().item()
            # Use short name (e.g. "self_attn.q_proj" → "q_proj")
            short = proj.split(".")[-1]
            layer_scales[short] = val
        scales[str(i)] = layer_scales
        print(f"  Layer {i:>2d}: {layer_scales}")

    out_path = os.path.join(OUTPUT_DIR, "bitnet_layer_scales.json")
    with open(out_path, "w") as f:
        json.dump(scales, f, indent=2)

    print(f"\n[3/3] Saved to {os.path.abspath(out_path)}")
    print(f"       {os.path.getsize(out_path):,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
