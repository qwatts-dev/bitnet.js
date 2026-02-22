# Testing & Validation

This document describes the automated test suite built into the WebGPU kernel. The tests run automatically every time the engine boots in the browser — open the DevTools console or scroll to the bottom of the page to see results.

## Test Suite Overview

| Test | What It Validates |
|------|-------------------|
| **Test 1 — 1D Branchless Kernel** | Element-wise ternary ops (packed u32) match CPU reference |
| **Test 2 — 2D Tiled Mat-Vec** | Shared-memory tiled matrix–vector multiply matches CPU reference |
| **Test 3 — Real AI Weights** | Layer 0 `down_proj` from the actual model produces non-trivial output |

All three tests run a **CPU reference implementation** alongside the GPU kernel and compare results with strict epsilon thresholds.

---

## Test 1 — 1D Bit-Packed Branchless Kernel

**Purpose:** Verify that the core ternary arithmetic (branchless bitmask approach) produces correct results when weights are packed 16-per-u32.

- **N** = 1,024 elements
- **Epsilon** = 1e-3
- **Input:** Random f32 values in [-10, 10]
- **Weights:** Random ternary {-1, 0, +1}

Compares GPU output against `cpuReference1D()` — a simple loop that applies `+input`, `-input`, or `0` per weight.

### Expected Output
```
✅ PASS – 1D kernel: GPU output matches CPU reference!
   Max |err| ≈ 0.00e+0
```

---

## Test 2 — 2D Tiled Matrix–Vector Multiply

**Purpose:** Verify the tiled mat-vec kernel with `var<workgroup>` shared memory produces correct row-dot-products across a large matrix.

- **Dimensions:** 4,096 × 4,096
- **Tile size:** 256 (TILE_K)
- **Workgroup:** 64 threads
- **Epsilon:** 0.1 (f32 accumulation tolerance at this scale)

Compares GPU output against `cpuReferenceMatVec()` — a double-loop with `Math.fround` accumulation.

### Expected Output
```
✅ PASS – 2D tiled kernel: GPU output matches CPU reference!
   Max |err| ≈ O(1e-2)
   GPU is ~N× faster than CPU (compute only)
```

---

## Test 3 — Real AI Weights Integration

**Purpose:** Confirm that actual Hugging Face model weights (layer 0 `down_proj`, 2560 × 6912) load correctly and produce non-zero, non-trivial output on the GPU.

- **Matrix:** `weights/bitnet_layer_0_down_proj.bin` (pre-packed ternary u32)
- **Input:** Random f32 vector (6,912 dims)
- **Validation:** Output should have a high percentage of non-zero values

### Expected Output
```
✅ PASS – Real AI weights: GPU produced non-trivial output!
   Output stats: ~2,560 / 2,560 non-zero values (100.0%)
```

---

## Running the Tests

Tests execute automatically when the page loads. No extra setup needed beyond the standard [Quick Start](../README.md#quick-start).

To view results:
1. Open `http://localhost:8080` in a WebGPU-capable browser
2. Scroll to the test log at the bottom of the page, or open DevTools → Console

All three tests must show **✅ PASS** for the engine to be considered functional.

---

## Benchmarks

See the [Benchmarks section](../README.md#benchmarks) in the main README for real-world generation performance numbers.
