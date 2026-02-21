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
standard JavaScript number. The 2,560-dimensional embedding is then
zero-padded to 6,912 (the GPU kernel's input width) before dispatch.

Tokens outside the 16K subset gracefully fall back to row 0 (OOV).

**Pipeline:** text → tokenizer → vocab map lookup → FP16→F32 conversion →
zero-pad → GPU ternary mat-vec → output

### Real AI weight integration

The `extract_weights.py` script downloads pre-trained weights from
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
on Hugging Face. The model stores weights as row-packed `uint8` tensors
(4 ternary values per byte). The script unpacks these, then re-packs into
our kernel's column-packed `uint32` format (16 weights per `u32`) and saves
a binary `.bin` file that the browser can `fetch()` directly into GPU buffers.

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

## Extracting Real AI Weights (Optional)

To reproduce the real-weight integration (Test 3), you need Python 3.10+ and (optionally) a Hugging Face account:

1. **Install Python dependencies**
   ```bash
   pip install torch safetensors huggingface-hub numpy
   ```

2. **Run the weight extraction script**
   ```bash
   python extract_weights.py
   ```
   This will:
   - Download `microsoft/bitnet-b1.58-2B-4T` (~1.1 GB safetensors file)
   - Extract `model.layers.0.mlp.down_proj.weight` (2560 × 6912 ternary matrix)
   - Unpack the HF row-packed `uint8` format (4 weights/byte)
   - Re-pack into our JS kernel's column-packed `uint32` format (16 weights/u32)
   - Save `bitnet_layer_0_down_proj.bin` (4.2 MB)

3. **Run the sparse embedding extraction script**
   ```bash
   python extract_sparse_embeddings.py
   ```
   This will:
   - Extract `model.embed_tokens.weight` (128,256 × 2,560, bfloat16)
   - Slice the first 16,384 tokens (BPE frequency-ordered) + token 50991 ("WebGPU")
   - Convert to Float16
   - Save `sparse_embeddings.bin` (80 MB) and `vocab_map.json` (202 KB)

4. **Serve and test** — all `.bin` and `.json` files must be in the same directory as `index.html`:
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

### Interactive mode: real embeddings → GPU mat-vec (v0.4.0)

Input run through the full pipeline: `@huggingface/tokenizers` encode → sparse FP16 vocab lookup → FP16→F32 conversion → zero-pad to 6912 → real BitNet weight matrix on GPU.

**"Hello"** (1 token, ID 9906)

| Metric | iPhone 14 Pro Max | iPad 13" M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| GPU compute | 27.0 ms | 10.0 ms | 6.3 ms |
| output[0] | 22.507725 | 22.507725 | 22.507725 |
| output[1] | 33.629837 | 33.629837 | 33.629837 |
| Deterministic | ✅ Bit-exact | ✅ Bit-exact | ✅ Bit-exact |

**Asset load times** (sparse_embeddings.bin 80 MB + bitnet_layer_0_down_proj.bin 4.2 MB):

| | iPhone 14 Pro Max | iPad 13" M3 | MacBook M2 Max |
|---|---|---|---|
| Load time | ~15 s | ~12 s | ~12 s |

**Key takeaways:**

- **Real semantic embeddings.** The system looks up the actual 2,560-dimensional vector that Microsoft's AI uses to represent "Hello", not random numbers.
- **Cross-device determinism.** All output values match to 6 decimal places across iPhone (A16), iPad (M3), and MacBook (M2 Max). The branchless, bit-packed kernel produces identical IEEE 754 results regardless of GPU architecture.
- **Mobile-friendly.** FP16 sparse vocab slice (80 MB) loads cleanly on all devices — previously the 1.3 GB Float32 file crashed iOS Safari.
- **FP16 precision is lossless for embeddings.** Output values are identical between the F32 and FP16 approaches, confirming the BitNet spec's guidance that boundary layers tolerate 16-bit precision.
- **M2 Max dominates on compute** at 6.3 ms, benefiting from its 30-core GPU and 400 GB/s memory bandwidth.
- **iPhone processes 17.7M ternary parameters in 27 ms** — well within interactive latency.

## Project Structure

| File | Description |
|------|-------------|
| `index.html` | Page with interactive text input panel and automated test log |
| `bitnet-kernel.js` | WebGPU setup, WGSL shaders, tokenizer integration, FP16 embedding loader, interactive handler, and validation |
| `extract_weights.py` | Python script to extract and bit-pack ternary weights from Hugging Face |
| `extract_sparse_embeddings.py` | Python script to extract sparse FP16 embedding vocab slice + vocab map |
| `bitnet_layer_0_down_proj.bin` | Pre-packed ternary weight binary for down_proj layer (generated by `extract_weights.py`) |
| `sparse_embeddings.bin` | FP16 embedding dictionary — 16,385 rows × 2,560 dims (generated by `extract_sparse_embeddings.py`) |
| `vocab_map.json` | Token ID → dense row index mapping (generated by `extract_sparse_embeddings.py`) |
| `package.json` | Project metadata |

## Dependencies

| Dependency | How it's used | Loaded via |
|---|---|---|
| [`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers) v0.1.1 | Llama 3 tokenizer (`Tokenizer`) — ~8.3 kB gzipped | CDN ES module import (no install needed) |
| [`Xenova/llama3-tokenizer`](https://huggingface.co/Xenova/llama3-tokenizer) | `tokenizer.json` + `tokenizer_config.json` | Fetched at runtime from Hugging Face Hub, cached via Cache API |

## License
This project is licensed under the MIT License - free and open for anyone to contribute, fork, and hack on!
