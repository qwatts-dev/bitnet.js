# BitNet WebGPU PoC

An experimental proof-of-concept aimed at running 1-bit AI models (like Microsoft's BitNet b1.58) natively in JavaScript using WebGPU.

## The Goal
Standard AI relies on heavy floating-point (`f32`) matrix multiplication, which creates a massive memory bottleneck. This PoC explores rewriting the core mathematical kernels in WGSL (WebGPU Shading Language) to use ternary weights (-1, 0, 1). 

By replacing complex multiplication with simple addition and subtraction directly on the GPU, we aim to drastically reduce memory bandwidth and make running massive Large Language Models in the browser a reality.

## Current Status
- [x] Initial repository setup
- [x] WebGPU environment configuration
- [x] WGSL compute shader for ternary math
- [x] Browser-based execution via local web server
- [x] Bit-packed weights (16 ternary values per u32)
- [x] Branchless ternary arithmetic (no if/else, no select)
- [x] 2D tiled mat-vec kernel using `var<workgroup>`
- [x] Automated CPU vs GPU validation (PASS/FAIL)
- [x] Stress-tested at 4096×4096 (~16.7M parameters)
- [x] Isolated GPU setup vs compute timing
- [x] Real AI weights from Hugging Face (`microsoft/bitnet-b1.58-2B-4T`)
- [x] Tokenizer integration via Hugging Face `transformers.js` v4 (CDN, no bundler)
- [x] Interactive UI — type text, click Compute, see real GPU output
- [x] Cross-device determinism verified (iPhone / iPad / MacBook — bit-exact match)
- [x] Migrated to standalone [`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers) (~8.3 kB gzipped)
- [x] Browser Cache API (`fetchWithCache`) for instant repeat tokenizer loads
- [x] Real embedding layer — sparse FP16 vocab slice (16,385 tokens, 80 MB)
- [x] Mobile-optimised: iPhone / iPad / MacBook all load and run successfully
- [x] Full SwiGLU MLP block — gate_proj + up_proj + SiLU·mul + down_proj (52.7M ternary params)
- [x] SiLU activation WGSL compute shader for SwiGLU fusion
- [x] Unified GPU orchestration — single command encoder, zero CPU round-trips
- [x] Cross-device bit-exact determinism verified (M2 Max / M3 iPad / A16 iPhone)
- [x] LM Head — dense FP32 mat-vec maps MLP output to vocabulary logits (16,385 × 2,560)
- [x] RMSNorm — CPU-side normalization prevents FP32 overflow in LM Head
- [x] End-to-end text-in → word-out pipeline (Embed → MLP → RMSNorm → LM Head → Argmax → Decode)
- [x] Adapter-aware WebGPU limits (requests hardware max `maxStorageBufferBindingSize`)
- [ ] Real RMSNorm weights from model
- [ ] Attention block
- [ ] Multi-layer inference pipeline

## How It Works

The WGSL compute shader receives an input vector and a ternary weight vector (`{-1, 0, +1}`). Instead of multiplying, it branches on each weight:

| Weight | Operation | Cost |
|--------|-----------|------|
| `+1`   | Copy the input value | One add |
| `-1`   | Negate the input value | One subtract |
| `0`    | Output zero (skip) | Nothing |

This completely eliminates floating-point multiplication, which is the core insight behind BitNet b1.58.

### Bit-packing (2 bits per weight)

Weights are packed into `u32` buffers to reduce memory bandwidth:

| 2-bit code | Weight | Meaning |
|------------|--------|---------|
| `00`       | 0      | skip    |
| `01`       | +1     | add     |
| `10`       | -1     | subtract |

The WGSL kernel unpacks 16 weights per `u32` and applies a branchless
bitmask to include or exclude the input value.

### Branchless ternary math

Instead of `if/else`, the kernel uses full-width bitmasks to select
`+input`, `-input`, or `0` without warp divergence.

### 2D tiled mat-vec kernel

For matrix-vector multiplication, input tiles are cached in
`var<workgroup>` shared memory. Each workgroup computes one output row,
and a reduction across the workgroup produces the final dot product.

### Tokenizer integration (v0.3.2)

The Llama 3 tokenizer is loaded at runtime using the standalone
[`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers)
package (~8.3 kB gzipped) via CDN as a native ES module — no bundler
required. This replaces the full `@huggingface/transformers` library
(~1.2 MB) with a purpose-built tokenizer that is **~150× smaller**.

Tokenizer config files (`tokenizer.json` and `tokenizer_config.json`)
are fetched from the Hugging Face Hub and cached using the browser's
standard Cache API (`caches.open('hf-tokenizer-cache')`), making repeat
page loads instant.

### Real semantic embeddings (v0.4.0)

The `extract_sparse_embeddings.py` script extracts the actual
`model.embed_tokens.weight` tensor from
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T).
Since BPE tokenizers assign lower IDs to more frequent tokens, the first
16,384 tokens cover ~99% of standard English. An extra row for token
50991 ("WebGPU") is appended for domain-specific testing.

The embedding vectors are stored in **Float16** to halve memory versus
Float32, producing an **80 MB** binary file (down from 1.3 GB for the
full Float32 vocabulary). A `vocab_map.json` maps original token IDs to
dense row indices in the binary file.

At runtime, the browser fetches both files in parallel. A lightweight
`fp16ToNumber()` function converts each half-precision value to a
standard JavaScript number. The raw 2,560-dimensional embedding is
passed directly into the MLP pipeline.

Tokens outside the 16K subset gracefully fall back to row 0 (OOV).

**Pipeline:** text → tokenizer → vocab map lookup → FP16→F32 conversion →
GPU SwiGLU MLP → output

### Full SwiGLU MLP block (v0.5.0)

The complete SwiGLU Multi-Layer Perceptron block for Layer 0:

```
embedding(2560) → gate_proj(6912) ─→ SiLU·mul(6912) → down_proj(2560)
                → up_proj(6912)   ─┘
```

Three ternary weight matrices are extracted via `extract_full_mlp.py`,
bit-packed (16 weights per `u32`), and served as static `.bin` files:
- **gate_proj** — 6912×2560 (4,320 KB)
- **up_proj** — 6912×2560 (4,320 KB)
- **down_proj** — 2560×6912 (4,320 KB)

### Unified GPU orchestration (v0.5.0)

All four compute passes (gate matmul, up matmul, SiLU·multiply, down
matmul) execute in a **single WebGPU command encoder submission**.
Intermediate tensors (`gate_out`, `up_out`, `silu_out`) are GPU-only
buffers that never leave VRAM — eliminating the GPU↔CPU "ping-pong"
that previously added ~35ms of transfer latency per step.

**Before (separate submissions):** 55.1ms
**After (unified):** 7.2ms on M2 Max

### Real AI weight integration

The `extract_full_mlp.py` script downloads pre-trained weights from
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
on Hugging Face. The model stores weights as row-packed `uint8` tensors
(4 ternary values per byte). The script unpacks all three MLP matrices
(gate_proj, up_proj, down_proj), then re-packs into our kernel's
column-packed `uint32` format (16 weights per `u32`) and saves `.bin`
files that the browser can `fetch()` directly into GPU buffers.

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/qwatts-dev/bitnet-webgpu-poc.git
   cd bitnet-webgpu-poc
   ```

2. **Start a local web server**
   ```bash
   npx serve . -l 8080
   ```

3. **Open in a WebGPU-capable browser** (Chrome 113+, Edge 113+, Safari 18+)
   
   Navigate to `http://localhost:8080` — the page will automatically run the ternary-weight compute shader on your GPU and display the results. Tests 1 and 2 run CPU-vs-GPU validation with synthetic data. Test 3 runs a real AI weight matrix through the WebGPU kernel.

## Testing on Mobile Devices (iPhone / iPad)

WebGPU requires a **secure context** (`https://`). Browsers make a special exception for `localhost`, but if you try to access your MacBook via its local IP address (e.g. `http://192.168.1.X:8080`), mobile Safari/Chrome will block WebGPU.

The fastest solution is **ngrok**, which creates a temporary `https://` tunnel from the internet to your local server.

1. **Keep your local server running** in one terminal:
   ```bash
   npx serve . -l 8080
   ```

2. **Open a second terminal** and start ngrok:
   ```bash
   npx ngrok http 8080
   ```

3. **Copy the secure URL** from the ngrok output. Look for the **Forwarding** line:
   ```
   Forwarding    https://a1b2-c3d4.ngrok-free.app -> http://localhost:8080
   ```

4. **Open that `https://` URL** on your iPhone or iPad.
   - WebGPU will be fully enabled because the connection is `https`.
   - The `.bin` weight files are served directly from your MacBook over your local network — nothing is uploaded to the internet.
   - First load will take a few seconds as the ~175 MB of weight files transfer over Wi-Fi.

> **Note:** You can install ngrok globally (`npm i -g ngrok`) or use `npx` for zero-install one-off usage. A free ngrok account is sufficient — the only limitation is a random subdomain that changes each session.

## Extracting Real AI Weights (Optional)

To reproduce the real-weight integration (Test 3), you need Python 3.10+ and (optionally) a Hugging Face account:

1. **Install Python dependencies**
   ```bash
   pip install torch safetensors huggingface-hub numpy
   ```

2. **Run the full MLP weight extraction script**
   ```bash
   python extract_full_mlp.py
   ```
   This will:
   - Download `microsoft/bitnet-b1.58-2B-4T` (~1.1 GB safetensors file)
   - Extract all 3 Layer 0 MLP matrices: gate_proj (6912×2560), up_proj (6912×2560), down_proj (2560×6912)
   - Unpack the HF row-packed `uint8` format (4 weights/byte)
   - Re-pack into our JS kernel's column-packed `uint32` format (16 weights/u32)
   - Save `bitnet_layer_0_gate_proj.bin`, `bitnet_layer_0_up_proj.bin`, `bitnet_layer_0_down_proj.bin` (~4.3 MB each)

3. **Run the sparse embedding extraction script**
   ```bash
   python extract_sparse_embeddings.py
   ```
   This will:
   - Extract `model.embed_tokens.weight` (128,256 × 2,560, bfloat16)
   - Slice the first 16,384 tokens (BPE frequency-ordered) + token 50991 ("WebGPU")
   - Convert to Float16
   - Save `sparse_embeddings.bin` (80 MB) and `vocab_map.json` (202 KB)

4. **Run the LM head extraction script**
   ```bash
   python extract_lm_head.py
   ```
   This will:
   - Extract `model.embed_tokens.weight` (the model uses tied embeddings — the embedding matrix doubles as the LM head)
   - Apply the exact same vocab-slice as the embeddings (rows 0–16,383 + row 50,991)
   - Convert to Float16
   - Save `sparse_lm_head.bin` (80 MB)

5. **Serve and test** — all `.bin` and `.json` files must be in the same directory as `index.html`:
   ```bash
   npx serve . -l 8080
   ```

## Latest Benchmark Results

### Test 2: Synthetic 4096×4096 mat-vec (CPU vs GPU validation)

| Metric | iPhone 14 Pro Max | iPad Air M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| CPU mat-vec | 78 ms | 71 ms | 93 ms |
| GPU setup | 84 ms | 73 ms | 98 ms |
| GPU compute | **50 ms** | **24 ms** | **3.1 ms** |
| Speedup | **1.6×** | **3.0×** | **30.1×** |
| Max error | 2.08e-3 | 2.08e-3 | 2.08e-3 |

### Test 3: Real AI weights — `microsoft/bitnet-b1.58-2B-4T`

Layer: `model.layers.0.mlp.down_proj` (2560 × 6912 = 17.7M ternary params)

| Metric | iPhone 14 Pro Max | iPad Air M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| GPU setup | 2 ms | 1 ms | 1.4 ms |
| GPU compute | **13 ms** | **7 ms** | **4.8 ms** |
| Non-zero outputs | 2560/2560 (100%) | 2560/2560 (100%) | 2560/2560 (100%) |
| Result | ✅ PASS | ✅ PASS | ✅ PASS |

### Interactive mode: Full SwiGLU MLP pipeline (v0.5.0)

Input run through the complete SwiGLU pipeline: `@huggingface/tokenizers` encode → sparse FP16 vocab lookup → FP16→F32 conversion → unified GPU SwiGLU MLP (gate_proj → up_proj → SiLU·mul → down_proj).

**"Hello"** (1 token, ID 9906) — 52.7M ternary parameters, 4 compute passes

| Metric | iPhone 14 Pro Max | iPad 13" M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| 1st run (cold JIT) | 378 ms | 361 ms | 21.2 ms |
| 2nd run | 68 ms | 30 ms | 16.2 ms |
| Warmed (3rd/4th) | **38 ms** | **18 ms** | **7.2 ms** |
| output[2559] | -106371.250000 | -106371.250000 | -106371.250000 |
| Deterministic | ✅ Bit-exact | ✅ Bit-exact | ✅ Bit-exact |

**Warmed cache breakdown:**

| | iPhone 14 Pro Max | iPad 13" M3 | MacBook M2 Max |
|---|---|---|---|
| Setup (buffers + pipelines) | 9.0 ms | 2.0 ms | 0.8 ms |
| Compute (submit + readback) | 29.0 ms | 16.0 ms | 6.4 ms |

**Asset load times** (sparse_embeddings.bin 80 MB + 3× MLP .bin files ~13 MB):

| | iPhone 14 Pro Max | iPad 13" M3 | MacBook M2 Max |
|---|---|---|---|
| First load | ~12 s | ~10 s | ~10 s |
| Cached (304) | < 2 s | < 2 s | < 1 s |

**Key takeaways:**

- **Complete SwiGLU MLP block.** The full gate_proj + up_proj + SiLU·mul + down_proj pipeline runs with real ternary weights from `microsoft/bitnet-b1.58-2B-4T`.
- **Unified GPU orchestration.** All 4 compute passes run in a single command encoder submission — intermediate tensors never leave VRAM. This eliminated ~35ms of GPU↔CPU transfer latency.
- **Metal JIT compilation.** First-run times (350ms+ on mobile) are entirely shader compilation. Apple's Metal backend caches the compiled GPU code — subsequent runs drop to steady-state speeds.
- **Cross-device determinism.** All output values match to 6 decimal places across iPhone (A16), iPad (M3), and MacBook (M2 Max). The branchless, bit-packed kernel produces identical IEEE 754 results regardless of GPU architecture.
- **Mobile-friendly.** FP16 sparse vocab slice (80 MB) loads cleanly on all devices — previously the 1.3 GB Float32 file crashed iOS Safari.
- **M2 Max processes 52.7M ternary parameters in 7.2ms** — ~140 MLP blocks/second.
- **iPhone processes the same in 38ms** — well within interactive latency for a phone.

### End-to-end prediction: Embed → MLP → RMSNorm → LM Head → Decode (v0.6.0)

Full pipeline: `@huggingface/tokenizers` encode → sparse FP16 embedding → unified GPU SwiGLU MLP → CPU RMSNorm → GPU dense LM Head (16,385 × 2,560) → argmax → reverse vocab map → tokenizer decode.

**"Hello"** (1 token, ID 9906) — first word prediction

| Metric | iPhone 14 Pro Max | iPad 13" M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| maxStorageBufferBindingSize | 1024 MB | 1024 MB | 4096 MB |
| MLP Compute | 37 ms | 23 ms | 34 ms |
| LM Head Compute | 338 ms | 255 ms | 74.5 ms |
| **Total Compute** | **375 ms** | **278 ms** | **108.5 ms** |
| Predicted word | `" volume"` | `" volume"` | `" volume"` |
| Token ID | 8286 | 8286 | 8286 |
| Logit | 207.9285 | 207.9285 | 207.9285 |
| Deterministic | ✅ Bit-exact | ✅ Bit-exact | ✅ Bit-exact |

**Top 5 predictions (all devices identical):**

| Rank | Word | Token ID | Logit |
|------|------|----------|-------|
| 1 | volume | 8286 | 207.93 |
| 2 | Count | 4605 | — |
| 3 | mass | 3148 | — |
| 4 | Mass | 9346 | — |
| 5 | Ma | 11583 | — |

**Key takeaways:**

- **End-to-end text → word.** The complete pipeline — embedding lookup, ternary SwiGLU MLP, RMSNorm, dense LM Head, argmax decode — produces a real English word on all three devices.
- **Semantic coherence.** All top-5 predictions (volume, Count, mass, Mass, Ma) cluster around measurement/quantity concepts, demonstrating the network's learned weight structure produces meaningful semantic groupings even through a single MLP layer without attention.
- **Adapter-aware limits.** By requesting `adapter.limits.maxStorageBufferBindingSize`, all devices successfully allocated the 160 MB LM Head buffer (default spec limit is 128 MB).
- **RMSNorm prevents overflow.** The MLP outputs ~564,000-scale values. Without normalization, these overflow FP32 dot products in the LM Head → NaN → 0.0. The CPU-side RMSNorm squishes to ~4.09 max, enabling safe computation.
- **Cross-device bit-exact determinism.** Token ID 8286, logit 207.9285 — identical across A16, M3, and M2 Max.
- **iPhone handles 160 MB GPU buffer.** The A16 Bionic granted 1024 MB of storage buffer space — no OOM crashes.
- **M2 Max wider memory bus shows on dense mat-vec.** The M3 iPad beats the M2 Max on ternary MLP (23ms vs 34ms) but the M2 Max dominates the dense FP32 LM Head (74ms vs 255ms) thanks to its wider memory bandwidth.

## Project Structure

| File | Description |
|------|-------------|
| `index.html` | Page with interactive text input panel and automated test log |
| `bitnet-kernel.js` | WebGPU setup, WGSL shaders, tokenizer integration, FP16 embedding loader, interactive handler, and validation |
| `extract_weights.py` | Python script to extract and bit-pack ternary weights from Hugging Face (single layer) |
| `extract_full_mlp.py` | Python script to extract & pack all 3 MLP weight matrices (gate, up, down) |
| `extract_sparse_embeddings.py` | Python script to extract sparse FP16 embedding vocab slice + vocab map |
| `extract_lm_head.py` | Python script to extract vocab-sliced FP16 LM head weights (tied embeddings) |
| `bitnet_layer_0_gate_proj.bin` | Pre-packed ternary weight binary for gate_proj (generated by `extract_full_mlp.py`) |
| `bitnet_layer_0_up_proj.bin` | Pre-packed ternary weight binary for up_proj (generated by `extract_full_mlp.py`) |
| `bitnet_layer_0_down_proj.bin` | Pre-packed ternary weight binary for down_proj (generated by `extract_full_mlp.py`) |
| `sparse_embeddings.bin` | FP16 embedding dictionary — 16,385 rows × 2,560 dims (generated by `extract_sparse_embeddings.py`) |
| `vocab_map.json` | Token ID → dense row index mapping (generated by `extract_sparse_embeddings.py`) |
| `sparse_lm_head.bin` | FP16 LM head weights — 16,385 rows × 2,560 dims (generated by `extract_lm_head.py`) |
| `package.json` | Project metadata |

## Dependencies

| Dependency | How it's used | Loaded via |
|---|---|---|
| [`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers) v0.1.1 | Llama 3 tokenizer (`Tokenizer`) — ~8.3 kB gzipped | CDN ES module import (no install needed) |
| [`Xenova/llama3-tokenizer`](https://huggingface.co/Xenova/llama3-tokenizer) | `tokenizer.json` + `tokenizer_config.json` | Fetched at runtime from Hugging Face Hub, cached via Cache API |

## License
This project is licensed under the MIT License - free and open for anyone to contribute, fork, and hack on!
