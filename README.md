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
- [ ] Load real weights from Hugging Face

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
   
   Navigate to `http://localhost:8080` — the page will automatically run the ternary-weight compute shader on your GPU and display the results.

## Latest Benchmark Results

2D tiled mat-vec at **4096×4096** (~16.7M ternary parameters), bit-packed to 4 MB (93.8% smaller than unpacked).

| Metric | iPad Air M3 | MacBook M2 Max |
|--------|-------------|----------------|
| CPU mat-vec | 71 ms | 94 ms |
| GPU setup | 73 ms | 98.2 ms |
| GPU compute | **7 ms** | **4.1 ms** |
| Speedup | **10.1×** | **22.9×** |
| Max error | 2.08e-3 | 2.08e-3 |

**Key takeaways:**

- **The timing split paid off.** Setup cost (buffers + pipeline compile) dominates total GPU time at ~73–98 ms, but the raw compute is only 4–7 ms for 16.7M ternary parameters. Without the split, the GPU would have looked slower than CPU.
- **M2 Max is ~1.7× faster on compute** than M3 (4.1 ms vs 7 ms), which tracks with its 30-core vs 10-core GPU advantage and higher memory bandwidth (400 GB/s vs ~150 GB/s).
- **M3 has faster CPU single-thread** (71 ms vs 94 ms), consistent with its newer core architecture despite fewer cores.
- **Numerical precision is identical** across both devices — max error of 2.08e-3 at the same row (496), confirming deterministic f32 accumulation behavior across Apple GPU generations.
- **Setup cost is a one-time expense** in a real inference pipeline — the pipeline and buffers would be reused across tokens, so the 4–7 ms compute time is the number that matters for throughput.

## Project Structure

| File | Description |
|------|-------------|
| `index.html` | Minimal page that loads the kernel as an ES module |
| `bitnet-kernel.js` | WebGPU setup, WGSL shader, buffer management, and result display |
| `package.json` | Project metadata |

## License
This project is licensed under the MIT License - free and open for anyone to contribute, fork, and hack on!
