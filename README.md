# BitNet WebGPU

Run Microsoft's [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (2 billion parameter, 1.58-bit ternary) large language model **entirely in the browser** using WebGPU — no server, no Python runtime, no WASM. Just JavaScript, WGSL shaders, and your GPU.

## Why?

Standard AI relies on heavy floating-point (`f32`) matrix multiplication, which creates a massive memory bottleneck. BitNet replaces those multiplications with ternary weights (-1, 0, +1), turning every matrix–vector product into pure addition and subtraction. This project implements that idea end-to-end in WGSL compute shaders, making a 2B-parameter LLM runnable in a single browser tab.

## Current Status — v0.11.0

The engine produces **coherent, grammatically correct English** through all 30 transformer layers. This is a full autoregressive inference engine — not a demo, not a single-layer proof.

**Sample outputs:**

> **Prompt:** *The capital city of France is*
> **Output:** *Paris, and the capital of Italy is... 3. What are some factors to consider when looking for*

> **Prompt:** *Hello*
> **Output:** *Hello, my name is [Your Name], and I am a student here at your school. We are*

### What's in the box

- `export class BitNetEngine` — clean OOP API: `init()`, `generate()`, `reset()`
- Bit-packed ternary weights (16 values per `u32`) with branchless arithmetic
- 2D tiled mat-vec kernel using `var<workgroup>` shared memory
- Full 30-layer transformer: RoPE + GQA attention, SwiGLU MLP with ReLU², SubLN, BitLinear weight scales
- Streaming autoregressive generation with Temperature + Top-K + Top-P + repetition penalty
- One-command model setup: `python scripts/setup_model.py`

## Architecture

### Model: `microsoft/bitnet-b1.58-2B-4T`

| Parameter | Value |
|-----------|-------|
| Layers | 30 |
| Hidden dim | 2,560 |
| MLP intermediate dim | 6,912 |
| Attention heads (Q) | 20 |
| KV heads | 5 (GQA group size 4) |
| Head dim | 128 |
| Vocabulary | 128,256 (16,385 extracted) |
| RoPE θ | 500,000 |
| RMSNorm ε | 1e-5 |
| Activation | ReLU² (squared ReLU) |
| Weight format | 1.58-bit ternary {-1, 0, +1} |
| Tie embeddings | Yes (embed_tokens = lm_head) |

### Inference Pipeline

```
Token IDs → Embedding Lookup (FP16→F32)
  ↓
For each of 30 layers:
  │  RMSNorm (input_layernorm, learned γ)
  │  ↓
  │  Self-Attention:
  │    Q/K/V projections (ternary mat-vec × weight_scale)
  │    → RoPE (Llama rotate_half, θ=500k)
  │    → KV Cache update
  │    → Grouped-Query Attention (20Q / 5KV)
  │    → SubLN (attn_sub_norm, RMSNorm)
  │    → O projection (ternary mat-vec × weight_scale)
  │  ↓
  │  Residual connection (+ input)
  │  ↓
  │  RMSNorm (post_attention_layernorm, learned γ)
  │  ↓
  │  MLP:
  │    gate_proj + up_proj (ternary mat-vec × weight_scale)
  │    → ReLU²(gate) ⊙ up
  │    → SubLN (ffn_sub_norm, RMSNorm)
  │    → down_proj (ternary mat-vec × weight_scale)
  │  ↓
  │  Residual connection (+ post-attn)
  ↓
Final RMSNorm (learned γ)
  ↓
LM Head (dense FP16 mat-vec, tied weights)
  ↓
Sampling (temp → top-k → top-p → repetition penalty)
  ↓
Decoded Token
```

### How Ternary Math Works

The WGSL compute shader receives an input vector and ternary weights (`{-1, 0, +1}`). Instead of multiplying, it branches on each weight:

| Weight | Operation | Cost |
|--------|-----------|------|
| `+1` | Copy the input value | One add |
| `-1` | Negate the input value | One subtract |
| `0` | Output zero (skip) | Nothing |

This completely eliminates floating-point multiplication — the core insight behind BitNet b1.58.

### Bit-Packing (2 bits per weight)

Weights are packed into `u32` buffers (16 values per word) to minimise memory bandwidth:

| 2-bit code | Weight | Meaning |
|------------|--------|---------|
| `00` | 0 | skip |
| `01` | +1 | add |
| `10` | -1 | subtract |

The WGSL kernel unpacks weights using branchless bitmasks — no warp divergence.

### HuggingFace Bit-Packing Format

The model stores weights as row-packed `uint8` tensors (4 ternary values per byte). The packing uses **contiguous blocks** (NOT interleaved rows) with a `+1` offset encoding:

| Stored code | Ternary value |
|-------------|---------------|
| `0` | -1 |
| `1` | 0 |
| `2` | +1 |

Bits `[1:0]` → rows `0..M/4-1`, bits `[3:2]` → rows `M/4..M/2-1`, bits `[5:4]` → rows `M/2..3M/4-1`, bits `[7:6]` → rows `3M/4..M-1`.

Our extraction scripts unpack from this HF format and re-pack into column-packed `uint32` (16 weights per word) for the WebGPU kernel.

### SubLN (Sub-Layer Normalization)

BitNet uses additional RMSNorm layers *within* the attention and MLP blocks (not present in standard Llama):

- **attn_sub_norm** — applied to the concatenated attention output *before* the O projection
- **ffn_sub_norm** — applied to ReLU²(gate)⊙up *before* the down projection

These sub-norms are critical for training stability with ternary weights.

### RoPE (Rotary Position Embeddings)

Uses the **Llama-style `rotate_half`** convention: dimension `d` is paired with `d + head_dim/2` within each head (NOT the GPT-NeoX interleaved `d, d+1` pairing). With `θ = 500,000` for the extended context window.

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/qwatts-dev/bitnet-webgpu-poc.git
   cd bitnet-webgpu-poc
   ```

2. **Extract model weights** (requires Python 3.10+, ~660 MB disk space)
   ```bash
   pip install torch safetensors huggingface-hub numpy accelerate transformers
   npm run setup
   ```

   This single command downloads the model from Hugging Face and writes everything into `weights/`:
   - `vocab_map.json` — token ID → dense row index mapping
   - `sparse_embeddings.bin` — FP16 embedding slice (16,385 × 2,560)
   - `sparse_lm_head.bin` — FP16 LM head (tied to embeddings)
   - `bitnet_layer_{i}_{proj}.bin` — 7 ternary weight files × 30 layers
   - `bitnet_layer_{i}_attn_norm.bin` + `_mlp_norm.bin` — learned RMSNorm γ
   - `bitnet_layer_{i}_attn_sub_norm.bin` + `_ffn_sub_norm.bin` — SubLN γ
   - `bitnet_layer_scales.json` — per-projection weight_scale values
   - `bitnet_final_norm.bin` — final layer RMSNorm γ

3. **Start a local web server**
   ```bash
   npm start
   ```

4. **Open in a WebGPU-capable browser** (Chrome 113+, Edge 113+, Safari 18+)

   Navigate to `http://localhost:8080` — type a prompt and generate text.

## Testing on Mobile Devices (iPhone / iPad)

WebGPU requires a **secure context** (`https://`). Use [Cloudflare Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) for quick HTTPS access:

```bash
# Terminal 1: local server
npx serve . -l 8080

# Terminal 2: HTTPS tunnel
npx cloudflared tunnel --url http://localhost:8080
```

Open the `https://...trycloudflare.com` URL on your mobile device.

## Benchmarks

| Device | Tokens/sec | Latency/token | Notes |
|--------|-----------|---------------|-------|
| MacBook Pro M2 Max | ~5 tok/s | ~200 ms | 20 tokens in ~4–5 s, 300 GPU dispatches/token |

> Benchmarks are from autoregressive generation (not prefill). Each token requires 300 GPU compute dispatches across 30 transformer layers.

## Testing

The engine ships with 3 automated GPU validation tests (1D ternary kernel, 2D tiled mat-vec, real AI weight forward pass). See [docs/TESTING.md](docs/TESTING.md) for full details and expected outputs.

## Project Structure

| Path | Description |
|------|-------------|
| `index.html` | Interactive UI — prompt input, streaming token output, sampling controls |
| `bitnet.js` | Complete WebGPU inference engine (`BitNetEngine` class) — WGSL shaders, 30-layer transformer, tokenizer, generation loop |
| `docs/TESTING.md` | Test suite documentation — 3 automated GPU validation tests |
| **`scripts/`** | **Python extraction scripts** |
| `scripts/setup_model.py` | Master orchestrator — runs all extractors in sequence |
| `scripts/extract_all_layers.py` | All 7 ternary weight matrices × 30 layers (210 files) |
| `scripts/extract_rmsnorm.py` | RMSNorm learned γ (input_layernorm, post_attention_layernorm, final_norm) |
| `scripts/extract_sub_norms.py` | SubLN γ (attn_sub_norm, ffn_sub_norm) |
| `scripts/extract_weight_scales.py` | Per-projection weight_scale scalars from BitLinear layers |
| `scripts/extract_sparse_embeddings.py` | Sparse FP16 embedding slice (16,385 tokens) + vocab map |
| `scripts/extract_lm_head.py` | FP16 LM head (tied to embed_tokens) |
| `scripts/extract_attention.py` | Single-layer attention extractor (legacy) |
| `scripts/extract_full_mlp.py` | Single-layer MLP extractor (legacy) |
| `scripts/extract_weights.py` | Single-matrix extractor (legacy) |
| **`weights/`** | **Runtime assets** *(gitignored — generated by `setup_model.py`)* |
| `weights/sparse_embeddings.bin` | FP16 embeddings — 16,385 × 2,560 (80 MB) |
| `weights/vocab_map.json` | Token ID → dense row index mapping |
| `weights/sparse_lm_head.bin` | FP16 LM head — 16,385 × 2,560 (80 MB) |
| `weights/bitnet_layer_*.bin` | 331 binary files — ternary weights, norms, sub-norms, scales (~500 MB) |

## Key Technical Details

### Weight Files Per Layer (11 files × 30 layers = 330 + 1 final norm = 331)

| File | Shape | Size | Type |
|------|-------|------|------|
| `q_proj` | 2560×2560 | 819 KB | Ternary (packed u32) |
| `k_proj` | 640×2560 | 205 KB | Ternary (packed u32) |
| `v_proj` | 640×2560 | 205 KB | Ternary (packed u32) |
| `o_proj` | 2560×2560 | 819 KB | Ternary (packed u32) |
| `gate_proj` | 6912×2560 | 2,211 KB | Ternary (packed u32) |
| `up_proj` | 6912×2560 | 2,211 KB | Ternary (packed u32) |
| `down_proj` | 2560×6912 | 2,211 KB | Ternary (packed u32) |
| `attn_norm` | 2560 | 10 KB | FP16 |
| `mlp_norm` | 2560 | 10 KB | FP16 |
| `attn_sub_norm` | 2560 | 10 KB | FP16 |
| `ffn_sub_norm` | 6912 | 27 KB | FP16 |

### GPU Passes Per Token Per Layer

| Pass | Operation | Dimensions |
|------|-----------|------------|
| 1–3 | Q/K/V projections (ternary mat-vec) | 2560/640/640 × 2560 |
| 4 | RoPE + KV cache + GQA attention | 20 heads × 128 dim |
| 5 | SubLN (attn_sub_norm) | 2560 |
| 6 | O projection (ternary mat-vec) | 2560 × 2560 |
| 7–8 | Gate + Up projections (ternary mat-vec) | 6912 × 2560 |
| 9 | ReLU² · element-wise multiply + SubLN | 6912 |
| 10 | Down projection (ternary mat-vec) | 2560 × 6912 |

**Total: 10 GPU passes × 30 layers = 300 GPU dispatches per token**

## Dependencies

| Dependency | Purpose | Loaded via |
|---|---|---|
| [`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers) v0.1.1 | Llama 3 BPE tokenizer (~8.3 kB gzipped) | CDN ES module import |
| [`Xenova/llama3-tokenizer`](https://huggingface.co/Xenova/llama3-tokenizer) | `tokenizer.json` + `tokenizer_config.json` | Fetched at runtime, cached via Cache API |

## Known Limitations

- **Sparse vocabulary** — only 16,385 of 128,256 tokens are extracted (covers ~99% of standard English via BPE frequency ordering). Token 128000 (BOS) falls back to row 0.
- **No BOS token** — the BOS token ID 128000 is outside the extracted vocabulary slice. Generation starts without it.
- **FP16 precision** — embeddings and LM head use FP16 (vs BF16 in the original model). Maximum error is negligible (~3e-8).
- **Sequence length** — KV cache is limited to 128 tokens.
- **No batching** — single-sequence inference only.

## License
This project is licensed under the MIT License — free and open for anyone to contribute, fork, and hack on!
