/**
 * bitnet-kernel.js
 *
 * Hardware-optimised BitNet 1.58-bit WebGPU kernel with:
 *
 *   1. Bit-Packed Weights  – 16 ternary values per u32 (2 bits each),
 *      reducing weight-buffer memory by 16×.
 *
 *   2. Branchless Arithmetic – pure bitwise masking replaces all
 *      if/else and even select().  Zero warp divergence.
 *
 *   3. 2D Tiled Mat-Vec (draft) – workgroup shared memory
 *      (var<workgroup>) caches input tiles so threads avoid
 *      redundant global reads during matrix–vector multiply.
 *
 *   4. Automated Validation – CPU reference runs the same ternary
 *      math; results are strictly compared with ✅ PASS / ❌ FAIL.
 *
 *   5. W1.58A8 Activation Quantization – dynamic INT8 quantization
 *      of activation vectors on the fly in the shader.  Ternary
 *      weights stay packed at 1.58-bit; activations are quantized
 *      to i8 in workgroup shared memory, dot products use pure
 *      integer arithmetic, and results are dequantized to f32.
 *
 * Weight encoding (2 bits per weight):
 *   0b00  →  0   (skip)
 *   0b01  → +1   (add input)
 *   0b10  → −1   (subtract input)
 *
 * Branchless core (no if • no select • no f32 multiply):
 *   code     = 2-bit value extracted from packed u32
 *   bit0     = code & 1           →  1 when weight == +1
 *   bit1     = (code >> 1) & 1    →  1 when weight == −1
 *   mask_pos = 0u − bit0          →  0xFFFFFFFF or 0x00000000
 *   mask_neg = 0u − bit1          →  0xFFFFFFFF or 0x00000000
 *   pos_val  = bitcast<f32>(bitcast<u32>(inp) & mask_pos)
 *   neg_val  = bitcast<f32>(bitcast<u32>(inp) & mask_neg)
 *   result   = pos_val − neg_val
 *
 * Usage
 * ─────
 *   Browser  : <script type="module" src="bitnet-kernel.js">
 *   Node ≥ 22: node --experimental-webgpu bitnet-kernel.js
 */

// ════════════════════════════════════════════════
// Tokenizer (@huggingface/tokenizers – standalone, ~8.3 kB)
// ════════════════════════════════════════════════

import { Tokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers';

let tokenizer      = null;
let gpuDevice       = null;   // kept alive for interactive use
let realWeights     = null;   // Uint32Array – packed down_proj (kept for Test 3 compat)
let embeddingData   = null;   // Uint16Array – FP16 sparse embed_tokens
let vocabMap        = null;   // Object – original token ID → dense row index
let lmHeadWeights   = null;   // Float32Array – dense LM head (vocab-sliced)
let reverseVocabMap = null;   // Object – dense row index → original token ID
let finalNormWeights = null;  // Float32Array – learned RMSNorm weights for final norm
let layerScales     = null;  // Object – { "0": { q_proj: 1.2, ... }, "1": {...}, ... }

// ── 26 Transformer Layers ──
// Each entry: { qW, kW, vW, oW, gateW, upW, downW, attnNorm, mlpNorm } (Uint32Array / Float32Array)
const layers      = [];       // layers[0..25]
const kCacheBufs  = [];       // GPUBuffer per layer – persistent KV cache (K)
const vCacheBufs  = [];       // GPUBuffer per layer – persistent KV cache (V)

let seqPos          = 0;      // current token position in sequence
const HIDDEN_DIM    = 2560;   // hidden_size of bitnet-b1.58-2B-4T
const MLP_DIM       = 6912;   // intermediate_size (SwiGLU)
const REAL_M        = 2560;   // rows  (down_proj output dim) — kept for Test 3
const REAL_K        = 6912;   // cols  (down_proj input  dim) — kept for Test 3
const EMBED_DIM     = 2560;   // hidden_size of bitnet-b1.58-2B-4T
const LM_HEAD_ROWS  = 16385;  // vocab-sliced rows (16384 + 1 for "WebGPU")
const NUM_LAYERS    = 30;     // transformer layers

// ── Self-Attention architecture constants ──
const NUM_Q_HEADS    = 20;           // query heads
const NUM_KV_HEADS   = 5;            // key/value heads (GQA 4:1)
const HEAD_DIM       = 128;          // dimension per head
const GQA_GROUP_SIZE = NUM_Q_HEADS / NUM_KV_HEADS;  // 4
const Q_DIM          = NUM_Q_HEADS  * HEAD_DIM;      // 2560
const KV_DIM         = NUM_KV_HEADS * HEAD_DIM;      // 640
const MAX_SEQ_LEN    = 128;          // max tokens in KV cache
const ROPE_THETA     = 500000.0;     // RoPE base frequency

/**
 * Fetch a URL using the browser Cache API so repeated loads are instant.
 * On cache miss: fetches, stores a clone in the cache, returns parsed JSON.
 * On cache hit: returns the cached response as parsed JSON.
 */
async function fetchWithCache(url) {
  const cache = await caches.open('hf-tokenizer-cache');
  const cached = await cache.match(url);
  if (cached) return cached.json();
  const res = await fetch(url);
  await cache.put(url, res.clone());
  return res.json();
}

/**
 * Convert a single IEEE 754 half-precision (FP16) value stored as a
 * 16-bit unsigned integer into a standard JavaScript number (f64).
 *
 * Bit layout:  [15] sign  |  [14:10] exponent (5-bit)  |  [9:0] mantissa
 */
function fp16ToNumber(h) {
  const sign = (h >>> 15) & 1;
  const exp  = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;

  let val;
  if (exp === 0) {
    // Sub-normal or zero
    val = (mant / 1024) * Math.pow(2, -14);
  } else if (exp === 31) {
    // Inf / NaN
    val = mant === 0 ? Infinity : NaN;
  } else {
    // Normal
    val = (1 + mant / 1024) * Math.pow(2, exp - 15);
  }
  return sign ? -val : val;
}

/**
 * Fetch and store the sparse FP16 embedding dictionary plus the
 * vocab map that translates original token IDs → dense row indices.
 *
 * Files:
 *   sparse_embeddings.bin – flat row-major float16 (rows × EMBED_DIM)
 *   vocab_map.json        – { "<tokenId>": <rowIndex>, … }
 */
async function loadEmbeddings() {
  log('Loading sparse embeddings (FP16) …', 'info');

  // Fetch vocab map and binary embeddings in parallel
  const [mapResp, binResp] = await Promise.all([
    fetch('vocab_map.json'),
    fetch('sparse_embeddings.bin'),
  ]);
  if (!mapResp.ok) throw new Error(`vocab_map.json fetch failed: HTTP ${mapResp.status}`);
  if (!binResp.ok) throw new Error(`sparse_embeddings.bin fetch failed: HTTP ${binResp.status}`);

  vocabMap = await mapResp.json();
  const buf = await binResp.arrayBuffer();
  embeddingData = new Uint16Array(buf);

  const totalRows = embeddingData.length / EMBED_DIM;
  const mapEntries = Object.keys(vocabMap).length;
  log(`✔ Embeddings loaded: ${totalRows.toLocaleString()} rows × ${EMBED_DIM} dims ` +
      `(${(buf.byteLength / 1024 / 1024).toFixed(1)} MB, FP16)`, 'info');
  log(`✔ Vocab map loaded: ${mapEntries.toLocaleString()} token ID entries ` +
      `(${(JSON.stringify(vocabMap).length / 1024).toFixed(0)} KB)`, 'info');
  log('');
}

/**
 * Fetch the sparse FP16 LM head weights and decode to Float32.
 * Also builds the reverse vocab map (dense index → original token ID).
 *
 * File: sparse_lm_head.bin – flat row-major float16 (LM_HEAD_ROWS × HIDDEN_DIM)
 */
async function loadLMHead() {
  log('Loading sparse LM head (FP16) …', 'info');

  const resp = await fetch('sparse_lm_head.bin');
  if (!resp.ok) throw new Error(`sparse_lm_head.bin fetch failed: HTTP ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const fp16 = new Uint16Array(buf);

  const expectedLen = LM_HEAD_ROWS * HIDDEN_DIM;
  if (fp16.length !== expectedLen) {
    throw new Error(`LM head size mismatch: got ${fp16.length}, expected ${expectedLen}`);
  }

  // Decode FP16 → Float32 (reuse the existing fp16ToNumber helper)
  lmHeadWeights = new Float32Array(fp16.length);
  for (let i = 0; i < fp16.length; i++) {
    lmHeadWeights[i] = fp16ToNumber(fp16[i]);
  }

  log(`✔ LM head loaded: ${LM_HEAD_ROWS.toLocaleString()} rows × ${HIDDEN_DIM} dims ` +
      `(${(buf.byteLength / 1024 / 1024).toFixed(1)} MB FP16 → ` +
      `${(lmHeadWeights.byteLength / 1024 / 1024).toFixed(1)} MB F32)`, 'info');

  // Build reverse vocab map: dense row index → original token ID string
  if (vocabMap) {
    reverseVocabMap = {};
    for (const [tokenId, rowIdx] of Object.entries(vocabMap)) {
      reverseVocabMap[String(rowIdx)] = tokenId;
    }
    log(`✔ Reverse vocab map built: ${Object.keys(reverseVocabMap).length.toLocaleString()} entries`, 'info');
  }
  log('');
}

/**
 * Load all 26 transformer layers from the weights/ directory.
 * Each layer has 7 packed binary matrices:
 *   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
 *
 * Loads layers sequentially (2 at a time) to avoid swamping the
 * browser's maximum concurrent connection limit.
 *
 * Populates the global `layers` array with objects:
 *   { qW, kW, vW, oW, gateW, upW, downW }
 */
async function loadAllLayers() {
  log(`━━━ Loading All ${NUM_LAYERS} Transformer Layers (${NUM_LAYERS * 7} matrices) ━━━`);
  log('');

  const PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
  const t0 = performance.now();
  let totalBytes = 0;

  // Fetch weight scales JSON in parallel with first layer
  const scalesResp = await fetch('weights/bitnet_layer_scales.json');
  if (scalesResp.ok) {
    layerScales = await scalesResp.json();
    log(`  ✔ Weight scales loaded (${Object.keys(layerScales).length} layers)`, 'pass');
  } else {
    log(`  ⚠ bitnet_layer_scales.json not found – scales will default to 1.0`, 'fail');
    layerScales = {};
  }

  for (let i = 0; i < NUM_LAYERS; i++) {
    const layerT0 = performance.now();

    // Fetch all 7 ternary matrices + 2 RMSNorm vectors + 2 sub-norms for this layer in parallel
    const projUrls = PROJ_NAMES.map(p => `weights/bitnet_layer_${i}_${p}.bin`);
    const normUrls = [
      `weights/bitnet_layer_${i}_attn_norm.bin`,
      `weights/bitnet_layer_${i}_mlp_norm.bin`,
      `weights/bitnet_layer_${i}_attn_sub_norm.bin`,
      `weights/bitnet_layer_${i}_ffn_sub_norm.bin`,
    ];
    const urls = [...projUrls, ...normUrls];
    const responses = await Promise.all(urls.map(u => fetch(u)));

    // Validate all responses
    for (let j = 0; j < responses.length; j++) {
      if (!responses[j].ok) {
        throw new Error(`${urls[j]}: HTTP ${responses[j].status}`);
      }
    }

    // Parse all ArrayBuffers in parallel
    const buffers = await Promise.all(responses.map(r => r.arrayBuffer()));

    // Per-projection weight scales (default 1.0 if JSON missing)
    const ls = (layerScales && layerScales[String(i)]) || {};

    const layerWeights = {
      qW:      new Uint32Array(buffers[0]),
      kW:      new Uint32Array(buffers[1]),
      vW:      new Uint32Array(buffers[2]),
      oW:      new Uint32Array(buffers[3]),
      gateW:   new Uint32Array(buffers[4]),
      upW:     new Uint32Array(buffers[5]),
      downW:   new Uint32Array(buffers[6]),
      attnNorm:    new Float32Array(buffers[7]),   // learned input_layernorm γ
      mlpNorm:     new Float32Array(buffers[8]),   // learned post_attention_layernorm γ
      attnSubNorm: new Float32Array(buffers[9]),   // sub-norm between GQA out & o_proj (dim 2560)
      ffnSubNorm:  new Float32Array(buffers[10]),  // sub-norm between ReLU²·mul & down_proj (dim 6912)
      // BitLinear weight_scale per projection
      qScale:    ls.q_proj    ?? 1.0,
      kScale:    ls.k_proj    ?? 1.0,
      vScale:    ls.v_proj    ?? 1.0,
      oScale:    ls.o_proj    ?? 1.0,
      gateScale: ls.gate_proj ?? 1.0,
      upScale:   ls.up_proj   ?? 1.0,
      downScale: ls.down_proj ?? 1.0,
    };

    let layerBytes = 0;
    for (const buf of buffers) layerBytes += buf.byteLength;
    totalBytes += layerBytes;

    layers.push(layerWeights);

    const layerMs = performance.now() - layerT0;
    log(`  Layer ${String(i).padStart(2)}/25 loaded: ` +
        `${(layerBytes / 1024 / 1024).toFixed(2)} MB  ` +
        `(${layerMs.toFixed(0)} ms)`, 'info');
  }

  const elapsed = performance.now() - t0;
  log('');
  log(`  ✅ All ${NUM_LAYERS} layers loaded: ${layers.length * 7} matrices, ` +
      `${(totalBytes / 1024 / 1024).toFixed(2)} MB total ` +
      `(${(elapsed / 1000).toFixed(1)}s)`, 'pass');
  log('');
}

/**
 * Initialise the Llama-3 tokenizer via the standalone
 * @huggingface/tokenizers package (~8.3 kB).
 * Downloads tokenizer.json + tokenizer_config.json from the
 * public Xenova/llama3-tokenizer repo on Hugging Face Hub.
 */
async function initTokenizer() {
  const BASE = 'https://huggingface.co/Xenova/llama3-tokenizer/resolve/main';
  log('Loading tokenizer (Xenova/llama3-tokenizer) …', 'info');

  const [tokenizerJson, tokenizerConfig] = await Promise.all([
    fetchWithCache(`${BASE}/tokenizer.json`),
    fetchWithCache(`${BASE}/tokenizer_config.json`),
  ]);

  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  log('✔ Tokenizer ready', 'info');
  log('');
}

// ════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════

const WORKGROUP_SIZE   = 64;
const TILE_K           = 256;   // tile width for 2D kernel
const ELEMS_PER_THREAD = TILE_K / WORKGROUP_SIZE; // 4

// ════════════════════════════════════════════════
// 1. WGSL – Bit-Packed Branchless 1D Kernel
// ════════════════════════════════════════════════

const SHADER_1D = /* wgsl */ `

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read>       inputs:         array<f32>;
@group(0) @binding(1) var<storage, read>       packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> result:         array<f32>;
@group(0) @binding(3) var<uniform>             params:         Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;

  // Bounds guard (only trailing threads in the last workgroup)
  if (idx >= params.n) { return; }

  let inp = inputs[idx];

  // ── Unpack: extract 2-bit weight code from packed u32 ──
  //    16 weights per u32, weight[i] lives at bits [(i%16)*2 +: 2]
  let pack_idx = idx / 16u;
  let bit_pos  = (idx % 16u) * 2u;
  let code     = (packed_weights[pack_idx] >> bit_pos) & 3u;

  // ── Branchless ternary arithmetic ──
  //    No if/else, no select(), no f32 multiply.
  //    Uses unsigned underflow (0u - 1u = 0xFFFFFFFF) to create
  //    full-width bitmasks, then AND-masks the IEEE 754 bits of
  //    the input.  Masked-out values become +0.0 (all-zeros).
  let bit0     = code & 1u;           // 1 when weight == +1
  let bit1     = (code >> 1u) & 1u;   // 1 when weight == -1
  let mask_pos = 0u - bit0;           // 0xFFFFFFFF or 0x00000000
  let mask_neg = 0u - bit1;           // 0xFFFFFFFF or 0x00000000

  let pos_val = bitcast<f32>(bitcast<u32>(inp) & mask_pos);
  let neg_val = bitcast<f32>(bitcast<u32>(inp) & mask_neg);

  result[idx] = pos_val - neg_val;
}
`;

// ════════════════════════════════════════════════
// 2. WGSL – 2D Tiled Matrix–Vector Multiply
// ════════════════════════════════════════════════
//
// Architecture
// ────────────
//   • Computes  output[row] = Σ_col  weight[row,col] · input[col]
//   • One workgroup per output row  (M workgroups dispatched)
//   • Inner dimension K is tiled into chunks of TILE_K
//   • Each tile is cooperatively loaded into var<workgroup>
//     shared memory so every thread reads from fast on-chip
//     SRAM instead of global VRAM
//   • After all tiles, a parallel binary reduction across the
//     workgroup produces the final scalar for that row
//
// Memory flow
// ───────────
//   global input_vec ──► shared tile[] ──► registers
//                                              │
//                           shared shared_acc[] ◄─┘
//                                  │
//                            output_vec[row]
// ════════════════════════════════════════════════

const SHADER_2D_TILED = /* wgsl */ `

const TILE_K_C: u32         = ${TILE_K}u;
const WG_SIZE_C: u32        = ${WORKGROUP_SIZE}u;
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD}u;

struct MatParams {
  M: u32,              // rows   (output length)
  K: u32,              // cols   (input  length)
  packed_stride: u32,  // u32s per packed row = ceil(K / 16)
  weight_scale: f32,   // BitLinear per-projection scale factor
}

@group(0) @binding(0) var<storage, read>       input_vec:      array<f32>;
@group(0) @binding(1) var<storage, read>       packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_vec:     array<f32>;
@group(0) @binding(3) var<uniform>             params:         MatParams;

// ── Workgroup shared memory ──
// tile[]       – cached slice of the input vector
// shared_acc[] – per-thread partial sums for reduction
var<workgroup> tile:       array<f32, ${TILE_K}>;
var<workgroup> shared_acc: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id)        wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wid.x;
  let tid = lid.x;

  if (row >= params.M) { return; }

  var acc: f32 = 0.0;
  let num_tiles = (params.K + TILE_K_C - 1u) / TILE_K_C;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {

    // ── Phase 1: Cooperative tile load ──────────────────────
    // All threads collaborate to pull a TILE_K-element slice
    // of the input vector into fast workgroup shared memory.
    // Each thread loads ELEMS_PER_THREAD contiguous elements.
    for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {
      let local_idx  = tid * ELEMS_PER_THREAD + e;
      let global_col = t * TILE_K_C + local_idx;
      // WebGPU robust buffer access guarantees OOB reads are
      // safe (return 0 or clamped), so select is well-defined.
      tile[local_idx] = select(0.0, input_vec[global_col],
                               global_col < params.K);
    }

    // Barrier: all threads must finish writing tile[] before
    // any thread reads from it.
    workgroupBarrier();

    // ── Phase 2: Accumulate from shared memory ─────────────
    // Each thread processes its slice of the tile, unpacking
    // the packed weight and applying branchless ternary math.
    for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {
      let local_idx  = tid * ELEMS_PER_THREAD + e;
      let global_col = t * TILE_K_C + local_idx;

      // Branchless bounds: create a mask that zeroes out
      // contributions from columns beyond K.
      let in_bounds  = u32(global_col < params.K);
      let bound_mask = 0u - in_bounds;

      // Unpack 2-bit weight for (row, col)
      let pack_idx = row * params.packed_stride + global_col / 16u;
      let bit_pos  = (global_col % 16u) * 2u;
      let code     = (packed_weights[pack_idx] >> bit_pos) & 3u;

      // Branchless ternary (same as 1D kernel)
      let bit0     = code & 1u;
      let bit1     = (code >> 1u) & 1u;
      let mask_pos = (0u - bit0) & bound_mask;
      let mask_neg = (0u - bit1) & bound_mask;

      let inp     = tile[local_idx];
      let pos_val = bitcast<f32>(bitcast<u32>(inp) & mask_pos);
      let neg_val = bitcast<f32>(bitcast<u32>(inp) & mask_neg);
      acc = acc + pos_val - neg_val;
    }

    // Barrier before next tile overwrites shared memory.
    workgroupBarrier();
  }

  // ── Phase 3: Workgroup parallel reduction ────────────────
  // Sum the partial accumulators from all threads into
  // shared_acc[0] using a binary reduction tree.
  shared_acc[tid] = acc;
  workgroupBarrier();

  for (var s: u32 = WG_SIZE_C / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_acc[tid] = shared_acc[tid] + shared_acc[tid + s];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the final dot-product to the output,
  // scaled by the BitLinear weight_scale factor.
  if (tid == 0u) {
    output_vec[row] = shared_acc[0] * params.weight_scale;
  }
}
`;

// ════════════════════════════════════════════════
// 2b. WGSL – W1.58A8 Tiled Mat-Vec (Activation Quantization)
//     [PARKED] Reverted to FP32 – kept as reference for when
//     WebGPU gains packed_4x8_integer_dot_product (DP4a).
// ════════════════════════════════════════════════
/*
// Architecture: W1.58A8
// ─────────────────────
//   Weights stay 1.58-bit packed (2 bits per u32, 16 per word).
//   Activations (input_vec) are dynamically quantized to INT8
//   on the fly inside the shader, enabling pure integer dot products.
//
// Pipeline per workgroup (one row of output):
//   Phase 0 – Quantize activations:
//     • Cooperatively scan input_vec to find abs_max
//     • scale = 127.0 / abs_max
//     • inv_scale = abs_max / 127.0
//   Phase 1 – Tiled integer mat-vec:
//     • Load tile of input_vec → quantize to i32 in shared memory
//     • Unpack 2-bit ternary weights
//     • Integer multiply-accumulate: acc += weight_ternary * act_int
//   Phase 2 – Reduce & dequantize:
//     • Binary tree reduction of i32 partial sums
//     • output = f32(int_sum) * inv_scale

const SHADER_2D_TILED_A8 = ` // wgsl

const TILE_K_C: u32         = ${TILE_K}u;
const WG_SIZE_C: u32        = ${WORKGROUP_SIZE}u;
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD}u;

struct MatParams {
  M: u32,              // rows   (output length)
  K: u32,              // cols   (input  length)
  packed_stride: u32,  // u32s per packed row = ceil(K / 16)
}

@group(0) @binding(0) var<storage, read>       input_vec:      array<f32>;
@group(0) @binding(1) var<storage, read>       packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_vec:     array<f32>;
@group(0) @binding(3) var<uniform>             params:         MatParams;

// ── Workgroup shared memory ──
// tile_q[]      – quantised INT8 activations (stored as i32)
// shared_max[]  – per-thread local maximums for abs_max reduction
// shared_iacc[] – per-thread integer partial sums for final reduction
var<workgroup> tile_q:      array<i32, ${TILE_K}>;
var<workgroup> shared_max:  array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> shared_iacc: array<i32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id)        wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wid.x;
  let tid = lid.x;

  if (row >= params.M) { return; }

  // ═══════════════════════════════════════════════════════
  // Phase 0: Dynamic Activation Quantization
  //   Cooperatively find abs_max of the entire input_vec,
  //   then compute scale factors for INT8 quantization.
  // ═══════════════════════════════════════════════════════
  var local_max: f32 = 0.0;
  for (var i: u32 = tid; i < params.K; i = i + WG_SIZE_C) {
    local_max = max(local_max, abs(input_vec[i]));
  }
  shared_max[tid] = local_max;
  workgroupBarrier();

  // Binary reduction → shared_max[0] = global abs_max
  for (var s: u32 = WG_SIZE_C / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
    }
    workgroupBarrier();
  }

  let abs_max   = shared_max[0];
  let scale     = select(1.0, 127.0 / abs_max, abs_max > 0.0);
  let inv_scale = select(0.0, abs_max / 127.0, abs_max > 0.0);

  // ═══════════════════════════════════════════════════════
  // Phase 1 & 2: Tiled Integer Mat-Vec
  //   Activations are quantised to i32 (INT8 range) in
  //   shared memory.  Ternary weights are unpacked from
  //   packed u32s.  Dot product uses pure integer math.
  // ═══════════════════════════════════════════════════════
  var int_acc: i32 = 0;
  let num_tiles = (params.K + TILE_K_C - 1u) / TILE_K_C;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {

    // ── Tile load: quantise f32 activations → i32 ──────
    for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {
      let local_idx  = tid * ELEMS_PER_THREAD + e;
      let global_col = t * TILE_K_C + local_idx;
      if (global_col < params.K) {
        tile_q[local_idx] = i32(round(input_vec[global_col] * scale));
      } else {
        tile_q[local_idx] = 0i;
      }
    }
    workgroupBarrier();

    // ── Integer dot product with ternary weights ───────
    for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {
      let local_idx  = tid * ELEMS_PER_THREAD + e;
      let global_col = t * TILE_K_C + local_idx;

      if (global_col < params.K) {
        // Unpack 2-bit ternary weight for (row, global_col)
        let pack_idx = row * params.packed_stride + global_col / 16u;
        let bit_pos  = (global_col % 16u) * 2u;
        let code     = (packed_weights[pack_idx] >> bit_pos) & 3u;

        // Ternary integer multiply (branchless):
        //   code 0b01 → bit0=1, bit1=0 → +1 * act
        //   code 0b10 → bit0=0, bit1=1 → -1 * act
        //   code 0b00 → bit0=0, bit1=0 →  0 * act
        let bit0 = i32(code & 1u);
        let bit1 = i32((code >> 1u) & 1u);
        int_acc = int_acc + tile_q[local_idx] * (bit0 - bit1);
      }
    }
    workgroupBarrier();
  }

  // ═══════════════════════════════════════════════════════
  // Phase 3: Integer Reduction + Dequantization
  //   Sum partial i32 accumulators across all threads,
  //   then convert back to f32 and multiply by inv_scale.
  // ═══════════════════════════════════════════════════════
  shared_iacc[tid] = int_acc;
  workgroupBarrier();

  for (var s: u32 = WG_SIZE_C / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_iacc[tid] = shared_iacc[tid] + shared_iacc[tid + s];
    }
    workgroupBarrier();
  }

  // Thread 0 dequantises and writes the final output.
  if (tid == 0u) {
    output_vec[row] = f32(shared_iacc[0]) * inv_scale;
  }
}
`;
*/  // END of parked SHADER_2D_TILED_A8

// ════════════════════════════════════════════════
// 3. JS – Pack Ternary Weights
// ════════════════════════════════════════════════

/**
 * Pack a 1-D array of ternary weights {-1, 0, +1} into a Uint32Array.
 * 16 weights per u32, 2 bits each.
 *   +1 → 0b01,  0 → 0b00,  -1 → 0b10
 */
function packWeights(weights) {
  const packedLen = Math.ceil(weights.length / 16);
  const packed = new Uint32Array(packedLen);
  for (let i = 0; i < weights.length; i++) {
    const packIdx = (i / 16) | 0;
    const bitPos  = (i % 16) * 2;
    let code = 0;
    if (weights[i] === 1)  code = 0b01;
    if (weights[i] === -1) code = 0b10;
    packed[packIdx] |= (code << bitPos);
  }
  return packed;
}

/**
 * Pack a 2-D weight matrix (flat row-major, M × K) into a Uint32Array.
 * Returns { packed, packedStride } where packedStride = ceil(K/16).
 */
function packWeightMatrix(weights, M, K) {
  const packedStride = Math.ceil(K / 16);
  const packed = new Uint32Array(M * packedStride);
  for (let row = 0; row < M; row++) {
    for (let col = 0; col < K; col++) {
      const w = weights[row * K + col];
      const packIdx = row * packedStride + ((col / 16) | 0);
      const bitPos  = (col % 16) * 2;
      let code = 0;
      if (w === 1)  code = 0b01;
      if (w === -1) code = 0b10;
      packed[packIdx] |= (code << bitPos);
    }
  }
  return { packed, packedStride };
}

// ════════════════════════════════════════════════
// 4. CPU Reference Implementations
// ════════════════════════════════════════════════

/** CPU element-wise ternary op (reference for 1D kernel). */
function cpuReference1D(inputs, weights) {
  const out = new Float32Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    if (weights[i] === 1)       out[i] =  inputs[i];
    else if (weights[i] === -1) out[i] = -inputs[i];
    else                        out[i] =  0;
  }
  return out;
}

/** CPU mat-vec multiply with ternary weights (f32 accumulation). */
function cpuReferenceMatVec(weightMatrix, inputVec, M, K) {
  const out = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    // Use Math.fround to match GPU f32 accumulation precision
    let acc = 0;
    for (let col = 0; col < K; col++) {
      const w = weightMatrix[row * K + col];
      if (w === 1)       acc = Math.fround(acc + inputVec[col]);
      else if (w === -1) acc = Math.fround(acc - inputVec[col]);
    }
    out[row] = acc;
  }
  return out;
}

// ════════════════════════════════════════════════
// 5. Deterministic PRNG (reproducible test data)
// ════════════════════════════════════════════════

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed  = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateTestData(n, seed = 42) {
  const rng     = mulberry32(seed);
  const inputs  = new Float32Array(n);
  const weights = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    inputs[i] = rng() * 20 - 10;               // f32 in [-10, 10]
    const r = rng();
    weights[i] = r < 0.333 ? -1 : r < 0.666 ? 0 : 1;
  }
  return { inputs, weights };
}

function generateMatrixTestData(M, K, seed = 123) {
  const rng          = mulberry32(seed);
  const inputVec     = new Float32Array(K);
  const weightMatrix = new Int32Array(M * K);
  for (let i = 0; i < K; i++) inputVec[i] = rng() * 20 - 10;
  for (let i = 0; i < M * K; i++) {
    const r = rng();
    weightMatrix[i] = r < 0.333 ? -1 : r < 0.666 ? 0 : 1;
  }
  return { inputVec, weightMatrix };
}

// ════════════════════════════════════════════════
// 5b. Real Embedding Lookup + CPU-Side Pad (Sparse FP16)
// ════════════════════════════════════════════════

/**
 * Look up the real high-precision embedding for `text` via the
 * sparse FP16 dictionary.
 *
 * Steps:
 *   1. Tokenize `text` with the loaded tokenizer.
 *   2. Take the FIRST token ID (index 0) — the primary semantic token.
 *   3. Look up the token ID in `vocabMap` to get the dense row index.
 *      If the token is out-of-vocabulary, fall back to row 0.
 *   4. Read EMBED_DIM FP16 values from `embeddingData`, convert to f32.
 *
 * Returns the raw Float32Array of size EMBED_DIM (2560).
 *
 * @param {string} text – input text to tokenize
 * @returns {Float32Array} embedding of length EMBED_DIM
 */
function getRealEmbedding(text) {
  if (!tokenizer) {
    throw new Error('Tokenizer not initialised – call initTokenizer() first.');
  }
  if (!embeddingData || !vocabMap) {
    throw new Error('Embeddings not loaded – call loadEmbeddings() first.');
  }

  const encoded = tokenizer.encode(text);
  const ids = encoded.ids;

  if (ids.length === 0) {
    throw new Error('Tokenizer produced an empty sequence.');
  }

  // Use the FIRST token ID as the primary semantic representative
  const tokenId = ids[0];

  // Look up dense row index via vocab map; fall back to row 0 (OOV)
  let rowIndex = vocabMap[String(tokenId)];
  let oov = false;
  if (rowIndex === undefined) {
    rowIndex = 0;
    oov = true;
  }

  // Read EMBED_DIM FP16 values and convert to Float32
  const fp16Start = rowIndex * EMBED_DIM;
  const embedding = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) {
    embedding[i] = fp16ToNumber(embeddingData[fp16Start + i]);
  }

  const oovTag = oov ? ' [OOV → fallback row 0]' : '';
  log(`  Tokenized "${text.length > 40 ? text.slice(0, 37) + '…' : text}"` +
      ` → ${ids.length} token(s), first ID = ${tokenId}${oovTag}`, 'info');
  log(`  Real embedding (FP16→F32): ${EMBED_DIM} dims`, 'info');

  return embedding;
}

// ════════════════════════════════════════════════
// 5b′. Embedding Lookup by Raw Token ID
// ════════════════════════════════════════════════

/**
 * Look up the embedding for a raw token ID (no tokenization step).
 * Used by the autoregressive loop where we already know the token ID.
 *
 * @param {number} tokenId – original Llama 3 token ID
 * @returns {Float32Array} embedding of length EMBED_DIM (2560)
 */
function getEmbeddingByTokenId(tokenId) {
  if (!embeddingData || !vocabMap) {
    throw new Error('Embeddings not loaded – call loadEmbeddings() first.');
  }

  let rowIndex = vocabMap[String(tokenId)];
  if (rowIndex === undefined) {
    rowIndex = 0;   // OOV fallback
  }

  const fp16Start = rowIndex * EMBED_DIM;
  const embedding = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) {
    embedding[i] = fp16ToNumber(embeddingData[fp16Start + i]);
  }
  return embedding;
}

// ════════════════════════════════════════════════
// 5c-LM. WGSL – Dense LM Head Mat-Vec (Float32, NOT ternary)
// ════════════════════════════════════════════════
//
// Architecture
// ────────────
//   Dense (non-bit-packed) matrix–vector multiply for the LM head.
//   The LM head is stored as full-precision Float32 (decoded from FP16).
//   Computes  logits[row] = Σ_col  lm_head[row, col] · input[col]
//   One workgroup per output row (M workgroups dispatched).
//

const SHADER_LM_HEAD = /* wgsl */ `

const WG_SIZE: u32 = ${WORKGROUP_SIZE}u;

struct Params {
  M: u32,   // rows (vocab size = 16385)
  K: u32,   // cols (hidden_dim = 2560)
}

@group(0) @binding(0) var<storage, read>       input_vec:  array<f32>;
@group(0) @binding(1) var<storage, read>       weight_mat: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(3) var<uniform>             params:     Params;

var<workgroup> shared_acc: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id)        wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wid.x;
  let tid = lid.x;

  if (row >= params.M) { return; }

  // Each thread accumulates a strided portion of the dot-product
  var acc: f32 = 0.0;
  let row_offset = row * params.K;
  for (var col: u32 = tid; col < params.K; col = col + WG_SIZE) {
    acc = acc + weight_mat[row_offset + col] * input_vec[col];
  }

  // Parallel reduction across workgroup
  shared_acc[tid] = acc;
  workgroupBarrier();

  for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_acc[tid] = shared_acc[tid] + shared_acc[tid + s];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output_vec[row] = shared_acc[0];
  }
}
`;

// ════════════════════════════════════════════════
// 5c. WGSL – SiLU · Element-wise Multiply (SwiGLU fusion)
// ════════════════════════════════════════════════
//
//  SwiGLU(x) = SiLU(gate_proj(x)) ⊙ up_proj(x)
//  ReLU²(g) = max(0, g)²
//
//  BitNet b1.58 uses Squared ReLU instead of SiLU for stable
//  1-bit training.  This shader takes two f32 vectors and writes:
//    result[i] = ReLU²(gate[i]) * up[i]

const SHADER_RELU2_MUL = /* wgsl */ `

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read>       gate_vec:   array<f32>;
@group(0) @binding(1) var<storage, read>       up_vec:     array<f32>;
@group(0) @binding(2) var<storage, read_write> result_vec: array<f32>;
@group(0) @binding(3) var<uniform>             params:     Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }

  // Squared ReLU: max(0, g)² — stable activation for 1-bit weights
  let gate = max(0.0, gate_vec[idx]);
  result_vec[idx] = (gate * gate) * up_vec[idx];
}
`;

// ════════════════════════════════════════════════
// 5c′. WGSL – In-Place RMSNorm (GPU)
// ════════════════════════════════════════════════
//
//  Normalises a vector in-place using RMSNorm with learned γ weights:
//    rms = sqrt( mean(x²) + ε )
//    x[i] = (x[i] / rms) * γ[i]
//
//  Dispatched as a SINGLE workgroup.  Each thread handles a strided
//  slice of the vector for both the reduction and the write-back.
//
//  Bindings:
//    @binding(0) vec (read_write) – the vector to normalise in-place
//    @binding(1) gamma (read)     – learned γ weights
//    @binding(2) params (uniform) – { n: u32 }

const SHADER_RMSNORM = /* wgsl */ `

const WG_SIZE: u32 = ${WORKGROUP_SIZE}u;
const EPS: f32     = 1e-5;

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read_write> vec:   array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

var<workgroup> shared_sum: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let n   = params.n;

  // Phase 1: Partial sum of squares (strided)
  var partial: f32 = 0.0;
  for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
    let val = vec[i];
    partial = partial + val * val;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();

  // Phase 2: Binary reduction tree → shared_sum[0] = total sum
  for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
    }
    workgroupBarrier();
  }

  // Phase 3: Compute inv_rms (all threads read shared_sum[0])
  let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(n) + EPS);

  // Phase 4: Normalise + scale by γ (strided, non-overlapping)
  for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
    vec[i] = vec[i] * inv_rms * gamma[i];
  }
}
`;

// ════════════════════════════════════════════════
// 5d. WGSL – RoPE + KV Cache Write
// ════════════════════════════════════════════════
//
//  Applies Rotary Positional Embeddings to Q and K vectors
//  using the Llama/HuggingFace "rotate_half" convention:
//    pair(d, d + head_dim/2) for d = 0..head_dim/2-1
//  NOT the GPT-NeoX interleaved convention (2d, 2d+1).
//
//  Then writes the rotated K and raw V into the persistent
//  KV cache at the current sequence position.
//
//  Dispatch: ceil(Q_DIM / 2 / WORKGROUP_SIZE) workgroups.
//  Each thread handles one (d, d+half_head) pair within a head.

const SHADER_ROPE_CACHE = /* wgsl */ `

struct RoPEParams {
  seq_pos:  u32,     // current position in sequence
  q_dim:    u32,     // total Q dimensions (2560)
  kv_dim:   u32,     // total KV dimensions (640)
  head_dim: u32,     // dimension per head (128)
}

@group(0) @binding(0) var<storage, read_write> q_vec:   array<f32>;
@group(0) @binding(1) var<storage, read_write> k_vec:   array<f32>;
@group(0) @binding(2) var<storage, read>       v_vec:   array<f32>;
@group(0) @binding(3) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(5) var<uniform>             params:  RoPEParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pair_idx = gid.x;
  let half_head = params.head_dim / 2u;          // 64
  let q_pairs   = params.q_dim  / 2u;            // 1280
  let kv_pairs  = params.kv_dim / 2u;            // 320

  if (pair_idx >= q_pairs) { return; }

  let pos = f32(params.seq_pos);

  // ── Llama-style rotate_half RoPE pairing ──
  // pair_idx encodes (head, d) where d ∈ [0, half_head).
  // The pair is (head*head_dim + d, head*head_dim + d + half_head).
  let head = pair_idx / half_head;
  let d    = pair_idx % half_head;

  // inv_freq[d] = 1 / (θ ^ (2d / head_dim))
  let freq  = 1.0 / pow(${ROPE_THETA}, f32(d * 2u) / f32(params.head_dim));
  let theta = pos * freq;
  let cos_t = cos(theta);
  let sin_t = sin(theta);

  // ── Apply RoPE to Q ──
  {
    let i0 = head * params.head_dim + d;              // first-half dim
    let i1 = i0 + half_head;                          // second-half dim
    let q0 = q_vec[i0];
    let q1 = q_vec[i1];
    // rotate_half: embed[i] = x[i]*cos - x[i+half]*sin
    //              embed[i+half] = x[i+half]*cos + x[i]*sin
    q_vec[i0] = q0 * cos_t - q1 * sin_t;
    q_vec[i1] = q1 * cos_t + q0 * sin_t;
  }

  // ── Apply RoPE to K + write KV cache ──
  if (pair_idx < kv_pairs) {
    let i0 = head * params.head_dim + d;
    let i1 = i0 + half_head;
    let k0 = k_vec[i0];
    let k1 = k_vec[i1];
    let k0r = k0 * cos_t - k1 * sin_t;
    let k1r = k1 * cos_t + k0 * sin_t;

    // Write rotated K into cache at current position
    let cache_off = params.seq_pos * params.kv_dim;
    k_cache[cache_off + i0] = k0r;
    k_cache[cache_off + i1] = k1r;

    // Write raw V into cache (no RoPE on V)
    v_cache[cache_off + i0] = v_vec[i0];
    v_cache[cache_off + i1] = v_vec[i1];
  }
}
`;

// ════════════════════════════════════════════════
// 5e. WGSL – Grouped-Query Attention (GQA)
// ════════════════════════════════════════════════
//
//  20 Query heads share 5 Key/Value heads (4:1 GQA ratio).
//  Computes scaled dot-product attention across the KV cache:
//    softmax((Q @ K^T) / sqrt(head_dim)) @ V
//
//  Dispatch: NUM_Q_HEADS (20) workgroups, WG_SIZE = HEAD_DIM (128).
//  Each workgroup handles one query head with 1 thread per dimension.

const SHADER_GQA_ATTENTION = /* wgsl */ `

const WG_ATTN: u32   = ${HEAD_DIM}u;
const MAX_SEQ: u32   = ${MAX_SEQ_LEN}u;
const GQA_GROUP: u32 = ${GQA_GROUP_SIZE}u;

struct AttnParams {
  seq_len:     u32,   // tokens in cache (seq_pos + 1)
  kv_dim:      u32,   // 640
  head_dim:    u32,   // 128
  num_q_heads: u32,   // 20
}

@group(0) @binding(0) var<storage, read>       q_vec:    array<f32>;
@group(0) @binding(1) var<storage, read>       k_cache:  array<f32>;
@group(0) @binding(2) var<storage, read>       v_cache:  array<f32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;
@group(0) @binding(4) var<uniform>             params:   AttnParams;

var<workgroup> scores: array<f32, ${MAX_SEQ_LEN}>;
var<workgroup> temp:   array<f32, ${HEAD_DIM}>;

@compute @workgroup_size(${HEAD_DIM})
fn main(
  @builtin(workgroup_id)        wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let head_id = wid.x;
  let tid     = lid.x;

  if (head_id >= params.num_q_heads) { return; }

  let kv_head   = head_id / GQA_GROUP;
  let q_offset  = head_id * params.head_dim;
  let kv_offset = kv_head * params.head_dim;
  let scale     = 1.0 / sqrt(f32(params.head_dim));

  // ── Phase 1: Compute attention scores for each cached position ──
  for (var p: u32 = 0u; p < params.seq_len; p = p + 1u) {
    let q_val = q_vec[q_offset + tid];
    let k_val = k_cache[p * params.kv_dim + kv_offset + tid];
    temp[tid] = q_val * k_val;
    workgroupBarrier();

    // Binary reduction (sum 128 → 1)
    for (var s: u32 = WG_ATTN / 2u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        temp[tid] = temp[tid] + temp[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      scores[p] = temp[0] * scale;
    }
    workgroupBarrier();
  }

  // ── Phase 2: Softmax over scores[0 .. seq_len-1] ──
  // 2a: Find max (numerical stability)
  if (tid < params.seq_len) {
    temp[tid] = scores[tid];
  } else {
    temp[tid] = -3.402823e+38;
  }
  workgroupBarrier();

  for (var s: u32 = WG_ATTN / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      temp[tid] = max(temp[tid], temp[tid + s]);
    }
    workgroupBarrier();
  }
  let max_score = temp[0];
  workgroupBarrier();

  // 2b: exp(score - max) and sum
  if (tid < params.seq_len) {
    temp[tid] = exp(scores[tid] - max_score);
    scores[tid] = temp[tid];
  } else {
    temp[tid] = 0.0;
  }
  workgroupBarrier();

  for (var s: u32 = WG_ATTN / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      temp[tid] = temp[tid] + temp[tid + s];
    }
    workgroupBarrier();
  }
  let sum_exp = temp[0];
  workgroupBarrier();

  // 2c: Normalize
  if (tid < params.seq_len) {
    scores[tid] = scores[tid] / sum_exp;
  }
  workgroupBarrier();

  // ── Phase 3: Weighted V sum ──
  var out_val: f32 = 0.0;
  for (var p: u32 = 0u; p < params.seq_len; p = p + 1u) {
    let v_val = v_cache[p * params.kv_dim + kv_offset + tid];
    out_val = out_val + scores[p] * v_val;
  }

  attn_out[q_offset + tid] = out_val;
}
`;

// ════════════════════════════════════════════════
// 6. WebGPU Initialisation
// ════════════════════════════════════════════════

async function initWebGPU() {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error(
      "WebGPU is not available.\n" +
      "• Browser: use a recent Chromium-based browser.\n" +
      "• Node ≥ 22: run with  node --experimental-webgpu"
    );
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found.");

  // Request the adapter's actual hardware limits instead of spec
  // defaults.  The LM Head weight buffer is ~160 MB (F32), which
  // exceeds the default maxStorageBufferBindingSize of 128 MB.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize:               adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) =>
    console.error(`WebGPU device lost: ${info.message}`),
  );

  log(`  maxStorageBufferBindingSize: ${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`, 'info');
  log(`  maxBufferSize:               ${(device.limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`, 'info');

  return device;
}

// ════════════════════════════════════════════════
// 7. 1D Kernel – Pipeline, Buffers, Dispatch
// ════════════════════════════════════════════════

async function run1DKernel(device, inputData, weightData) {
  const n             = inputData.length;
  const packedWeights = packWeights(weightData);

  // ── Buffers ──
  const inputBuf = device.createBuffer({
    label: "1d-input",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputData));

  const weightBuf = device.createBuffer({
    label: "1d-packed-weights",
    size:  packedWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, packedWeights);

  const resultBuf = device.createBuffer({
    label: "1d-result",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "1d-staging",
    size:  n * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "1d-params",
    size:  4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([n]));

  // ── Pipeline ──
  const module   = device.createShaderModule({ code: SHADER_1D });
  const bgl      = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: weightBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  // ── Dispatch ──
  const t0      = performance.now();
  const encoder = device.createCommandEncoder();
  const pass    = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
  pass.end();
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, n * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const gpuMs = performance.now() - t0;

  [inputBuf, weightBuf, resultBuf, stagingBuf, uniformBuf].forEach((b) =>
    b.destroy(),
  );
  return { results, gpuMs, packedWeights };
}

// ════════════════════════════════════════════════
// 8. 2D Tiled Kernel – Pipeline, Buffers, Dispatch
// ════════════════════════════════════════════════

async function run2DKernel(device, weightMatrix, inputVec, M, K) {
  const setupT0 = performance.now();

  const { packed, packedStride } = packWeightMatrix(weightMatrix, M, K);

  // ── Buffers ──
  const inputBuf = device.createBuffer({
    label: "2d-input",
    size:  K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const weightBuf = device.createBuffer({
    label: "2d-packed-weights",
    size:  packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, packed);

  const resultBuf = device.createBuffer({
    label: "2d-result",
    size:  M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "2d-staging",
    size:  M * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // M, K, packed_stride  (3 × u32 = 12 bytes; padded to 16)
  const uniformBuf = device.createBuffer({
    label: "2d-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Pack uniform: M(u32), K(u32), packed_stride(u32), weight_scale(f32)
  const uniformData = new ArrayBuffer(16);
  const uniformU32 = new Uint32Array(uniformData);
  const uniformF32 = new Float32Array(uniformData);
  uniformU32[0] = M;
  uniformU32[1] = K;
  uniformU32[2] = packedStride;
  uniformF32[3] = 1.0;  // no scale for standalone test kernel
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array(uniformData));

  // ── Pipeline ──
  const module   = device.createShaderModule({ code: SHADER_2D_TILED });
  const bgl      = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: weightBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const setupMs = performance.now() - setupT0;

  // ── Dispatch: one workgroup per row ──
  const computeT0 = performance.now();
  const encoder   = device.createCommandEncoder();
  const pass      = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(M);
  pass.end();
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, M * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  [inputBuf, weightBuf, resultBuf, stagingBuf, uniformBuf].forEach((b) =>
    b.destroy(),
  );
  return { results, setupMs, computeMs, packed, packedStride };
}

/**
 * run2DKernelPacked – same as run2DKernel but accepts pre-packed
 * Uint32Array weights directly (no JS-side packing step).
 */
async function run2DKernelPacked(device, packedWeights, inputVec, M, K) {
  const packedStride = Math.ceil(K / 16);
  const setupT0 = performance.now();

  // ── Buffers ──
  const inputBuf = device.createBuffer({
    label: "2d-input",
    size:  K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const weightBuf = device.createBuffer({
    label: "2d-packed-weights",
    size:  packedWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, packedWeights);

  const resultBuf = device.createBuffer({
    label: "2d-result",
    size:  M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "2d-staging",
    size:  M * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "2d-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Pack uniform: M(u32), K(u32), packed_stride(u32), weight_scale(f32)
  {
    const uData = new ArrayBuffer(16);
    const uU32  = new Uint32Array(uData);
    const uF32  = new Float32Array(uData);
    uU32[0] = M;
    uU32[1] = K;
    uU32[2] = packedStride;
    uF32[3] = 1.0;  // no scale for standalone packed kernel
    device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uData));
  }

  // ── Pipeline ──
  const module   = device.createShaderModule({ code: SHADER_2D_TILED });
  const bgl      = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: weightBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const setupMs = performance.now() - setupT0;

  // ── Dispatch: one workgroup per row ──
  const computeT0 = performance.now();
  const encoder   = device.createCommandEncoder();
  const pass      = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(M);
  pass.end();
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, M * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  [inputBuf, weightBuf, resultBuf, stagingBuf, uniformBuf].forEach((b) =>
    b.destroy(),
  );
  return { results, setupMs, computeMs, packedStride };
}

// ════════════════════════════════════════════════
// 8b. ReLU² · Multiply Kernel – Pipeline, Buffers, Dispatch
// ════════════════════════════════════════════════

/**
 * Run the ReLU²·Mul element-wise kernel on the GPU.
 *   result[i] = ReLU²(gateData[i]) * upData[i]
 *
 * @param {GPUDevice}    device   – WebGPU device
 * @param {Float32Array} gateData – gate_proj output (length n)
 * @param {Float32Array} upData   – up_proj output   (length n)
 * @param {number}       n        – vector length
 * @returns {{ results: Float32Array, computeMs: number }}
 */
async function runReLU2MulKernel(device, gateData, upData, n) {
  const setupT0 = performance.now();

  // ── Buffers ──
  const gateBuf = device.createBuffer({
    label: "relu2-gate",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gateBuf, 0, new Float32Array(gateData));

  const upBuf = device.createBuffer({
    label: "relu2-up",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(upBuf, 0, new Float32Array(upData));

  const resultBuf = device.createBuffer({
    label: "relu2-result",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "relu2-staging",
    size:  n * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "relu2-params",
    size:  4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([n]));

  // ── Pipeline ──
  const module = device.createShaderModule({ code: SHADER_RELU2_MUL });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: gateBuf } },
      { binding: 1, resource: { buffer: upBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const setupMs = performance.now() - setupT0;

  // ── Dispatch ──
  const computeT0 = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
  pass.end();
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, n * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  [gateBuf, upBuf, resultBuf, stagingBuf, uniformBuf].forEach((b) =>
    b.destroy(),
  );
  return { results, setupMs, computeMs };
}

// ════════════════════════════════════════════════
// 8b-LM. LM Head Kernel – Dense Float32 Mat-Vec
// ════════════════════════════════════════════════

/**
 * Run the dense LM head matrix–vector multiply on the GPU.
 *   logits[row] = Σ_col  lmHeadWeights[row, col] · inputVec[col]
 *
 * @param {GPUDevice}    device      – WebGPU device
 * @param {Float32Array} lmWeights   – flat row-major (M × K) float32
 * @param {Float32Array} inputVec    – hidden state vector (length K)
 * @param {number}       M           – number of rows (vocab size = 16385)
 * @param {number}       K           – number of cols (hidden_dim = 2560)
 * @returns {{ results: Float32Array, setupMs: number, computeMs: number }}
 */
async function runLMHeadKernel(device, lmWeights, inputVec, M, K) {
  const setupT0 = performance.now();

  // ── Buffers ──
  const inputBuf = device.createBuffer({
    label: "lm-head-input",
    size:  K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const weightBuf = device.createBuffer({
    label: "lm-head-weights",
    size:  lmWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, lmWeights);

  const resultBuf = device.createBuffer({
    label: "lm-head-result",
    size:  M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "lm-head-staging",
    size:  M * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "lm-head-params",
    size:  16,   // M + K = 2 × u32, padded to 16 for WebGPU alignment
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([M, K, 0, 0]));

  // ── Pipeline ──
  const module = device.createShaderModule({ code: SHADER_LM_HEAD });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: weightBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const setupMs = performance.now() - setupT0;

  // ── Dispatch: one workgroup per vocab row ──
  const computeT0 = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(M);
  pass.end();
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, M * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  [inputBuf, weightBuf, resultBuf, stagingBuf, uniformBuf].forEach((b) =>
    b.destroy(),
  );
  return { results, setupMs, computeMs };
}

/**
 * Return the index of the maximum value in a typed array.
 */
function argmax(arr) {
  let maxIdx = 0;
  let maxVal = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Sample a token from logits using Repetition Penalty + Temperature +
 * Top-K + Top-P (nucleus) sampling.
 *
 * Pipeline:
 *   0. Apply repetition penalty to previously generated tokens
 *   1. Apply temperature scaling:  logits[i] /= temperature
 *   2. Softmax → probabilities
 *   3. Sort descending by probability
 *   4. Top-K: keep only the K highest-probability tokens
 *   5. Top-P: keep tokens until cumulative probability >= P
 *   6. Renormalize the surviving probabilities
 *   7. Weighted random sample from the remaining distribution
 *
 * @param {Float32Array} logits           – raw logits from LM head
 * @param {number}       temperature      – sharpness control (default 1.0)
 * @param {number}       topK             – max tokens to consider (default 50)
 * @param {number}       topP             – cumulative probability cutoff (default 0.9)
 * @param {Set|Array}    generatedTokens  – sparse indices already emitted (for penalty)
 * @param {number}       repetitionPenalty – divisor for repeated positive logits (default 1.2)
 * @returns {number} selected index into logits array
 */
function sampleToken(logits, temperature = 1.0, topK = 50, topP = 0.9,
                     generatedTokens = null, repetitionPenalty = 1.2) {
  const n = logits.length;

  // ── 0. Frequency-scaled repetition penalty ──────────────
  //   Each token is penalised proportionally to how many times
  //   it has already been emitted:  penalty_factor = penalty^count.
  //   Positive logits are divided, negative logits are multiplied,
  //   so repeated tokens become exponentially less likely.
  const penalised = new Float32Array(logits);
  if (generatedTokens && generatedTokens.size > 0) {
    for (const [idx, count] of generatedTokens) {
      if (idx < n) {
        const factor = Math.pow(repetitionPenalty, count);
        if (penalised[idx] > 0) {
          penalised[idx] /= factor;
        } else {
          penalised[idx] *= factor;
        }
      }
    }
  }

  // ── 1. Temperature scaling ──────────────────────────────
  const scaled = new Float32Array(n);
  const invTemp = 1.0 / Math.max(temperature, 1e-8);
  for (let i = 0; i < n; i++) {
    scaled[i] = penalised[i] * invTemp;
  }

  // ── 2. Stable softmax ───────────────────────────────────
  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    if (scaled[i] > maxLogit) maxLogit = scaled[i];
  }

  let sumExp = 0;
  const probs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    probs[i] = Math.exp(scaled[i] - maxLogit);
    sumExp += probs[i];
  }
  for (let i = 0; i < n; i++) {
    probs[i] /= sumExp;
  }

  // ── 3. Build index array and sort descending by prob ────
  const indices = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  indices.sort((a, b) => probs[b] - probs[a]);

  // ── 4. Top-K: keep only the top K entries ───────────────
  const kCut = Math.min(topK, n);

  // ── 5. Top-P: keep tokens until cumulative prob >= P ────
  let cumProb = 0;
  let pCut = kCut;  // defaults to full top-K if topP >= 1.0
  for (let i = 0; i < kCut; i++) {
    cumProb += probs[indices[i]];
    if (cumProb >= topP) {
      pCut = i + 1;
      break;
    }
  }

  // ── 6. Renormalize surviving probabilities ──────────────
  let renormSum = 0;
  for (let i = 0; i < pCut; i++) {
    renormSum += probs[indices[i]];
  }

  // ── 7. Weighted random sample ───────────────────────────
  let r = Math.random() * renormSum;
  for (let i = 0; i < pCut; i++) {
    r -= probs[indices[i]];
    if (r <= 0) return indices[i];
  }

  // Fallback (floating-point edge case)
  return indices[0];
}

/**
 * CPU-side RMSNorm – squishes values into a safe range so the
 * LM Head dot products don't overflow FP32.
 *
 * In the real model an RMSNorm layer (with learned weights) sits
 * between the MLP output and the LM Head.  When a learned weight
 * vector (gamma) is provided, each normalised element is scaled
 * by the corresponding gamma value:
 *   rms = sqrt( mean(x²) + ε )
 *   out[i] = (x[i] / rms) * γ[i]
 *
 * If no weight vector is supplied, plain mathematical normalisation
 * is applied (backward-compatible fallback).
 */
function simpleRMSNorm(arr, weightVec) {
  const n = arr.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += arr[i] * arr[i];
  const rms = Math.sqrt(sumSq / n + 1e-5);
  const out = new Float32Array(n);
  if (weightVec) {
    for (let i = 0; i < n; i++) out[i] = (arr[i] / rms) * weightVec[i];
  } else {
    for (let i = 0; i < n; i++) out[i] = arr[i] / rms;
  }
  return out;
}

// ════════════════════════════════════════════════
// 8c. Unified Full MLP – Zero-Copy GPU Pipeline
// ════════════════════════════════════════════════
//
// Runs the entire SwiGLU MLP block in a SINGLE command encoder
// submission.  All intermediate tensors stay in VRAM — no CPU
// round-trips until the final result is read back.
//
//   Pass 1: gate_out  = gate_proj  × input   (M_mid × K_in → M_mid)
//   Pass 2: up_out    = up_proj    × input   (M_mid × K_in → M_mid)
//   Pass 3: relu2_out = ReLU²(gate_out) ⊙ up_out   (element-wise)
//   Pass 4: result    = down_proj  × relu2_out    (M_out × M_mid → M_out)
//

/**
 * Run the full SwiGLU MLP block on the GPU with unified command encoding.
 *
 * @param {GPUDevice}    device      – WebGPU device
 * @param {Uint32Array}  gateW       – packed gate_proj weights  (MLP_DIM × HIDDEN_DIM)
 * @param {Uint32Array}  upW         – packed up_proj weights    (MLP_DIM × HIDDEN_DIM)
 * @param {Uint32Array}  downW       – packed down_proj weights  (HIDDEN_DIM × MLP_DIM)
 * @param {Float32Array} inputVec    – embedding vector (length HIDDEN_DIM)
 * @param {number}       hiddenDim   – input/output dimension   (2560)
 * @param {number}       mlpDim      – intermediate dimension   (6912)
 * @param {number}       gateScale   – weight_scale for gate_proj  (default 1.0)
 * @param {number}       upScale     – weight_scale for up_proj    (default 1.0)
 * @param {number}       downScale   – weight_scale for down_proj  (default 1.0)
 * @param {Float32Array} ffnSubNorm  – sub-norm γ between ReLU²·mul & down_proj (dim mlpDim)
 * @returns {{ results: Float32Array, setupMs: number, computeMs: number }}
 */
async function runFullMLP(device, gateW, upW, downW, inputVec, hiddenDim, mlpDim,
                          gateScale = 1.0, upScale = 1.0, downScale = 1.0, ffnSubNorm = null) {
  const setupT0 = performance.now();

  const gateStride = Math.ceil(hiddenDim / 16);  // gate/up: K = hiddenDim
  const downStride = Math.ceil(mlpDim / 16);     // down:    K = mlpDim

  // ── Shared input buffer (uploaded once) ─────────────────
  const inputBuf = device.createBuffer({
    label: "mlp-input",
    size:  hiddenDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  // ── Weight buffers (uploaded once) ──────────────────────
  const gateWeightBuf = device.createBuffer({
    label: "mlp-gate-weights",
    size:  gateW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gateWeightBuf, 0, gateW);

  const upWeightBuf = device.createBuffer({
    label: "mlp-up-weights",
    size:  upW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(upWeightBuf, 0, upW);

  const downWeightBuf = device.createBuffer({
    label: "mlp-down-weights",
    size:  downW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(downWeightBuf, 0, downW);

  // ── GPU-only intermediate buffers (never leave VRAM) ───
  const gateOutBuf = device.createBuffer({
    label: "mlp-gate-out",
    size:  mlpDim * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const upOutBuf = device.createBuffer({
    label: "mlp-up-out",
    size:  mlpDim * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const siluOutBuf = device.createBuffer({
    label: "mlp-silu-out",
    size:  mlpDim * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // ── Final result + staging (only readback point) ───────
  const resultBuf = device.createBuffer({
    label: "mlp-result",
    size:  hiddenDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuf = device.createBuffer({
    label: "mlp-staging",
    size:  hiddenDim * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // ── Uniform buffers ────────────────────────────────────
  //  Helper to write [M(u32), K(u32), stride(u32), scale(f32)]
  function writeMatUniform(buf, Mu, Ku, stride, scale) {
    const d = new ArrayBuffer(16);
    new Uint32Array(d)[0] = Mu;
    new Uint32Array(d)[1] = Ku;
    new Uint32Array(d)[2] = stride;
    new Float32Array(d)[3] = scale;
    device.queue.writeBuffer(buf, 0, new Uint8Array(d));
  }

  //  gate params: M=mlpDim, K=hiddenDim, stride=gateStride, scale=gateScale
  const gateUniform = device.createBuffer({
    label: "mlp-gate-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeMatUniform(gateUniform, mlpDim, hiddenDim, gateStride, gateScale);

  //  up params: M=mlpDim, K=hiddenDim, stride=gateStride, scale=upScale
  const upUniform = device.createBuffer({
    label: "mlp-up-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeMatUniform(upUniform, mlpDim, hiddenDim, gateStride, upScale);

  //  ReLU²·mul params: n=mlpDim
  const relu2Uniform = device.createBuffer({
    label: "mlp-relu2-params",
    size:  4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(relu2Uniform, 0, new Uint32Array([mlpDim]));

  //  down params: M=hiddenDim, K=mlpDim, stride=downStride, scale=downScale
  const downUniform = device.createBuffer({
    label: "mlp-down-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeMatUniform(downUniform, hiddenDim, mlpDim, downStride, downScale);

  // ── Compile pipelines (FP32 tiled mat-vec) ──
  const matModule  = device.createShaderModule({ code: SHADER_2D_TILED });
  const relu2Module = device.createShaderModule({ code: SHADER_RELU2_MUL });
  const normModule = device.createShaderModule({ code: SHADER_RMSNORM });

  const matBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const matPipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [matBGL] }),
    compute: { module: matModule, entryPoint: "main" },
  });

  const siluBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const siluPipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [siluBGL] }),
    compute: { module: relu2Module, entryPoint: "main" },
  });

  const normBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const normPipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [normBGL] }),
    compute: { module: normModule, entryPoint: "main" },
  });

  // ── Bind groups ────────────────────────────────────────
  const gateBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: gateWeightBuf } },
      { binding: 2, resource: { buffer: gateOutBuf } },
      { binding: 3, resource: { buffer: gateUniform } },
    ],
  });

  const upBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: upWeightBuf } },
      { binding: 2, resource: { buffer: upOutBuf } },
      { binding: 3, resource: { buffer: upUniform } },
    ],
  });

  const siluBG = device.createBindGroup({
    layout: siluBGL,
    entries: [
      { binding: 0, resource: { buffer: gateOutBuf } },
      { binding: 1, resource: { buffer: upOutBuf } },
      { binding: 2, resource: { buffer: siluOutBuf } },
      { binding: 3, resource: { buffer: relu2Uniform } },
    ],
  });

  // ── ffn_sub_norm: RMSNorm between ReLU²·mul output and down_proj ──
  const ffnNormGammaBuf = device.createBuffer({
    label: "mlp-ffn-sub-norm-gamma",
    size:  mlpDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (ffnSubNorm) device.queue.writeBuffer(ffnNormGammaBuf, 0, ffnSubNorm);

  const ffnNormUniform = device.createBuffer({
    label: "mlp-ffn-sub-norm-params", size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ffnNormUniform, 0, new Uint32Array([mlpDim]));

  const normBG = device.createBindGroup({
    layout: normBGL,
    entries: [
      { binding: 0, resource: { buffer: siluOutBuf } },
      { binding: 1, resource: { buffer: ffnNormGammaBuf } },
      { binding: 2, resource: { buffer: ffnNormUniform } },
    ],
  });

  const downBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: siluOutBuf } },
      { binding: 1, resource: { buffer: downWeightBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: downUniform } },
    ],
  });

  const setupMs = performance.now() - setupT0;

  // ═══════════════════════════════════════════════════════
  //  SINGLE command encoder – all 4 passes, zero CPU trips
  // ═══════════════════════════════════════════════════════
  const computeT0 = performance.now();
  const encoder = device.createCommandEncoder();

  // Pass 1: gate_out = gate_proj × input
  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(matPipeline);
  pass1.setBindGroup(0, gateBG);
  pass1.dispatchWorkgroups(mlpDim);
  pass1.end();

  // Pass 2: up_out = up_proj × input
  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(matPipeline);
  pass2.setBindGroup(0, upBG);
  pass2.dispatchWorkgroups(mlpDim);
  pass2.end();

  // Pass 3: relu2_out = ReLU²(gate_out) ⊙ up_out
  const pass3 = encoder.beginComputePass();
  pass3.setPipeline(siluPipeline);
  pass3.setBindGroup(0, siluBG);
  pass3.dispatchWorkgroups(Math.ceil(mlpDim / WORKGROUP_SIZE));
  pass3.end();

  // Pass 3b: ffn_sub_norm – RMSNorm on ReLU²·mul output (in-place)
  const pass3b = encoder.beginComputePass();
  pass3b.setPipeline(normPipeline);
  pass3b.setBindGroup(0, normBG);
  pass3b.dispatchWorkgroups(1);  // single workgroup for reduction
  pass3b.end();

  // Pass 4: result = down_proj × relu2_out (after sub-norm)
  const pass4 = encoder.beginComputePass();
  pass4.setPipeline(matPipeline);
  pass4.setBindGroup(0, downBG);
  pass4.dispatchWorkgroups(hiddenDim);
  pass4.end();

  // Copy final result to staging (still inside the same encoder)
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, hiddenDim * 4);

  // Submit everything as ONE command buffer
  device.queue.submit([encoder.finish()]);

  // Only NOW do we touch the CPU – single async readback
  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  // Cleanup
  [
    inputBuf, gateWeightBuf, upWeightBuf, downWeightBuf,
    gateOutBuf, upOutBuf, siluOutBuf,
    resultBuf, stagingBuf,
    gateUniform, upUniform, relu2Uniform, downUniform,
    ffnNormGammaBuf, ffnNormUniform,
  ].forEach((b) => b.destroy());

  return { results, setupMs, computeMs };
}

// ════════════════════════════════════════════════
// 8d. Self-Attention – Unified 6-Pass GPU Pipeline
// ════════════════════════════════════════════════
//
//  Pass 1: q_vec  = q_proj  × input   (2560 × 2560 → 2560)
//  Pass 2: k_vec  = k_proj  × input   (640  × 2560 → 640)
//  Pass 3: v_vec  = v_proj  × input   (640  × 2560 → 640)
//  Pass 4: RoPE(q, k) + write k,v to KV cache
//  Pass 5: GQA Attention (20 Q heads, 5 KV heads)
//  Pass 6: result = o_proj  × attn_out (2560 × 2560 → 2560)
//
//  All intermediate buffers stay in VRAM.  Single command encoder.

/**
 * Run the full Self-Attention block on the GPU.
 *
 * @param {GPUDevice}    device      – WebGPU device
 * @param {Uint32Array}  qW          – packed q_proj weights
 * @param {Uint32Array}  kW          – packed k_proj weights
 * @param {Uint32Array}  vW          – packed v_proj weights
 * @param {Uint32Array}  oW          – packed o_proj weights
 * @param {Float32Array} inputVec    – normed hidden state (length HIDDEN_DIM)
 * @param {number}       seqPosition – current position in KV cache
 * @param {GPUBuffer}    kCache      – persistent K cache buffer for this layer
 * @param {GPUBuffer}    vCache      – persistent V cache buffer for this layer
 * @param {number}       qScale      – weight_scale for q_proj  (default 1.0)
 * @param {number}       kScale      – weight_scale for k_proj  (default 1.0)
 * @param {number}       vScale      – weight_scale for v_proj  (default 1.0)
 * @param {number}       oScale      – weight_scale for o_proj  (default 1.0)
 * @param {Float32Array} attnSubNorm – sub-norm γ between GQA out & o_proj (dim Q_DIM)
 * @returns {{ results: Float32Array, setupMs: number, computeMs: number }}
 */
async function runSelfAttention(device, qW, kW, vW, oW, inputVec, seqPosition, kCache, vCache,
                                qScale = 1.0, kScale = 1.0, vScale = 1.0, oScale = 1.0,
                                attnSubNorm = null) {
  const setupT0 = performance.now();

  const qStride = Math.ceil(HIDDEN_DIM / 16);  // Q: K = 2560, stride = 160
  const kStride = Math.ceil(HIDDEN_DIM / 16);  // K/V: K = 2560, stride = 160
  const oStride = Math.ceil(Q_DIM / 16);       // O: K = 2560, stride = 160

  // ── Shared input buffer (uploaded once) ────────────────
  const inputBuf = device.createBuffer({
    label: "attn-input",
    size:  HIDDEN_DIM * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  // ── Weight buffers ─────────────────────────────────────
  const qWeightBuf = device.createBuffer({
    label: "attn-q-weights",
    size:  qW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(qWeightBuf, 0, qW);

  const kWeightBuf = device.createBuffer({
    label: "attn-k-weights",
    size:  kW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(kWeightBuf, 0, kW);

  const vWeightBuf = device.createBuffer({
    label: "attn-v-weights",
    size:  vW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vWeightBuf, 0, vW);

  const oWeightBuf = device.createBuffer({
    label: "attn-o-weights",
    size:  oW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(oWeightBuf, 0, oW);

  // ── GPU-only intermediate buffers ──────────────────────
  const qVecBuf = device.createBuffer({
    label: "attn-q-vec",
    size:  Q_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const kVecBuf = device.createBuffer({
    label: "attn-k-vec",
    size:  KV_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const vVecBuf = device.createBuffer({
    label: "attn-v-vec",
    size:  KV_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const attnOutBuf = device.createBuffer({
    label: "attn-out",
    size:  Q_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // ── Final result + staging ─────────────────────────────
  const resultBuf = device.createBuffer({
    label: "attn-o-result",
    size:  HIDDEN_DIM * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuf = device.createBuffer({
    label: "attn-staging",
    size:  HIDDEN_DIM * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // ── Uniform buffers ────────────────────────────────────
  // Helper to write [M(u32), K(u32), stride(u32), weight_scale(f32)]
  function writeAttnMatUniform(buf, Mu, Ku, stride, scale) {
    const d = new ArrayBuffer(16);
    new Uint32Array(d)[0] = Mu;
    new Uint32Array(d)[1] = Ku;
    new Uint32Array(d)[2] = stride;
    new Float32Array(d)[3] = scale;
    device.queue.writeBuffer(buf, 0, new Uint8Array(d));
  }

  // Q proj: M=Q_DIM(2560), K=HIDDEN_DIM(2560), stride=160, scale=qScale
  const qUniform = device.createBuffer({
    label: "attn-q-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(qUniform, Q_DIM, HIDDEN_DIM, qStride, qScale);

  // K proj: M=KV_DIM(640), K=HIDDEN_DIM(2560), stride=160, scale=kScale
  const kProjUniform = device.createBuffer({
    label: "attn-k-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(kProjUniform, KV_DIM, HIDDEN_DIM, kStride, kScale);

  // V proj: M=KV_DIM(640), K=HIDDEN_DIM(2560), stride=160, scale=vScale
  const vProjUniform = device.createBuffer({
    label: "attn-v-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(vProjUniform, KV_DIM, HIDDEN_DIM, kStride, vScale);

  // RoPE: seq_pos, q_dim, kv_dim, head_dim
  const ropeUniform = device.createBuffer({
    label: "attn-rope-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ropeUniform, 0, new Uint32Array([seqPosition, Q_DIM, KV_DIM, HEAD_DIM]));

  // GQA Attention: seq_len(pos+1), kv_dim, head_dim, num_q_heads
  const gqaUniform = device.createBuffer({
    label: "attn-gqa-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gqaUniform, 0, new Uint32Array([seqPosition + 1, KV_DIM, HEAD_DIM, NUM_Q_HEADS]));

  // O proj: M=HIDDEN_DIM(2560), K=Q_DIM(2560), stride=160, scale=oScale
  const oUniform = device.createBuffer({
    label: "attn-o-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(oUniform, HIDDEN_DIM, Q_DIM, oStride, oScale);

  // ── attn_sub_norm: RMSNorm between GQA output and O projection ──
  const normGammaBuf = device.createBuffer({
    label: "attn-sub-norm-gamma",
    size:  Q_DIM * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (attnSubNorm) device.queue.writeBuffer(normGammaBuf, 0, attnSubNorm);

  const normUniform = device.createBuffer({
    label: "attn-sub-norm-params", size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(normUniform, 0, new Uint32Array([Q_DIM]));

  // ── Compile pipelines (FP32 tiled mat-vec projections) ──
  const matModule  = device.createShaderModule({ code: SHADER_2D_TILED });
  const ropeModule = device.createShaderModule({ code: SHADER_ROPE_CACHE });
  const gqaModule  = device.createShaderModule({ code: SHADER_GQA_ATTENTION });
  const normModule = device.createShaderModule({ code: SHADER_RMSNORM });

  const matBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const matPipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [matBGL] }),
    compute: { module: matModule, entryPoint: "main" },
  });

  const ropeBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const ropePipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [ropeBGL] }),
    compute: { module: ropeModule, entryPoint: "main" },
  });

  const gqaBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const gqaPipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [gqaBGL] }),
    compute: { module: gqaModule, entryPoint: "main" },
  });

  const normBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const normPipeline = device.createComputePipeline({
    layout:  device.createPipelineLayout({ bindGroupLayouts: [normBGL] }),
    compute: { module: normModule, entryPoint: "main" },
  });

  // ── Bind groups ────────────────────────────────────────
  // Pass 1: q_vec = q_proj × input
  const qBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: qWeightBuf } },
      { binding: 2, resource: { buffer: qVecBuf } },
      { binding: 3, resource: { buffer: qUniform } },
    ],
  });

  // Pass 2: k_vec = k_proj × input
  const kBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: kWeightBuf } },
      { binding: 2, resource: { buffer: kVecBuf } },
      { binding: 3, resource: { buffer: kProjUniform } },
    ],
  });

  // Pass 3: v_vec = v_proj × input
  const vBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: vWeightBuf } },
      { binding: 2, resource: { buffer: vVecBuf } },
      { binding: 3, resource: { buffer: vProjUniform } },
    ],
  });

  // Pass 4: RoPE + Cache Write
  const ropeBG = device.createBindGroup({
    layout: ropeBGL,
    entries: [
      { binding: 0, resource: { buffer: qVecBuf } },
      { binding: 1, resource: { buffer: kVecBuf } },
      { binding: 2, resource: { buffer: vVecBuf } },
      { binding: 3, resource: { buffer: kCache } },
      { binding: 4, resource: { buffer: vCache } },
      { binding: 5, resource: { buffer: ropeUniform } },
    ],
  });

  // Pass 5: GQA Attention
  const gqaBG = device.createBindGroup({
    layout: gqaBGL,
    entries: [
      { binding: 0, resource: { buffer: qVecBuf } },
      { binding: 1, resource: { buffer: kCache } },
      { binding: 2, resource: { buffer: vCache } },
      { binding: 3, resource: { buffer: attnOutBuf } },
      { binding: 4, resource: { buffer: gqaUniform } },
    ],
  });

  // Pass 5b: RMSNorm on attnOutBuf (in-place, before o_proj)
  const normBG = device.createBindGroup({
    layout: normBGL,
    entries: [
      { binding: 0, resource: { buffer: attnOutBuf } },
      { binding: 1, resource: { buffer: normGammaBuf } },
      { binding: 2, resource: { buffer: normUniform } },
    ],
  });

  // Pass 6: result = o_proj × attn_out (after sub-norm)
  const oBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: attnOutBuf } },
      { binding: 1, resource: { buffer: oWeightBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
      { binding: 3, resource: { buffer: oUniform } },
    ],
  });

  const setupMs = performance.now() - setupT0;

  // ═══════════════════════════════════════════════════════
  //  SINGLE command encoder – all 6 passes, zero CPU trips
  // ═══════════════════════════════════════════════════════
  const computeT0 = performance.now();
  const encoder = device.createCommandEncoder();

  // Pass 1: Q projection (2560 × 2560 → 2560)
  const p1 = encoder.beginComputePass();
  p1.setPipeline(matPipeline);
  p1.setBindGroup(0, qBG);
  p1.dispatchWorkgroups(Q_DIM);
  p1.end();

  // Pass 2: K projection (640 × 2560 → 640)
  const p2 = encoder.beginComputePass();
  p2.setPipeline(matPipeline);
  p2.setBindGroup(0, kBG);
  p2.dispatchWorkgroups(KV_DIM);
  p2.end();

  // Pass 3: V projection (640 × 2560 → 640)
  const p3 = encoder.beginComputePass();
  p3.setPipeline(matPipeline);
  p3.setBindGroup(0, vBG);
  p3.dispatchWorkgroups(KV_DIM);
  p3.end();

  // Pass 4: RoPE + KV Cache write
  const p4 = encoder.beginComputePass();
  p4.setPipeline(ropePipeline);
  p4.setBindGroup(0, ropeBG);
  p4.dispatchWorkgroups(Math.ceil(Q_DIM / 2 / WORKGROUP_SIZE));
  p4.end();

  // Pass 5: GQA Attention (20 heads)
  const p5 = encoder.beginComputePass();
  p5.setPipeline(gqaPipeline);
  p5.setBindGroup(0, gqaBG);
  p5.dispatchWorkgroups(NUM_Q_HEADS);
  p5.end();

  // Pass 5b: attn_sub_norm – RMSNorm on attention output (in-place)
  const p5b = encoder.beginComputePass();
  p5b.setPipeline(normPipeline);
  p5b.setBindGroup(0, normBG);
  p5b.dispatchWorkgroups(1);  // single workgroup for reduction
  p5b.end();

  // Pass 6: O projection (2560 × 2560 → 2560)
  const p6 = encoder.beginComputePass();
  p6.setPipeline(matPipeline);
  p6.setBindGroup(0, oBG);
  p6.dispatchWorkgroups(HIDDEN_DIM);
  p6.end();

  // Copy final result to staging
  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, HIDDEN_DIM * 4);
  device.queue.submit([encoder.finish()]);

  // Single async readback
  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  // Cleanup (do NOT destroy kCache/vCache – they persist!)
  [
    inputBuf, qWeightBuf, kWeightBuf, vWeightBuf, oWeightBuf,
    qVecBuf, kVecBuf, vVecBuf, attnOutBuf,
    resultBuf, stagingBuf,
    qUniform, kProjUniform, vProjUniform, ropeUniform, gqaUniform, oUniform,
    normGammaBuf, normUniform,
  ].forEach(b => b.destroy());

  return { results, setupMs, computeMs };
}

// ════════════════════════════════════════════════
// 9. Validation
// ════════════════════════════════════════════════

function compareResults(cpu, gpu, epsilon = 1e-3) {
  if (cpu.length !== gpu.length)
    return { pass: false, maxErr: Infinity, idx: -1 };
  let maxErr = 0,
    worstIdx = 0;
  for (let i = 0; i < cpu.length; i++) {
    const err = Math.abs(cpu[i] - gpu[i]);
    if (err > maxErr) {
      maxErr   = err;
      worstIdx = i;
    }
  }
  return { pass: maxErr < epsilon, maxErr, idx: worstIdx };
}

// ════════════════════════════════════════════════
// 10. Display Helper
// ════════════════════════════════════════════════

function log(text, className) {
  console.log(text);
  const el = document.getElementById("output");
  if (el) {
    const span = document.createElement("span");
    if (className) span.className = className;
    span.textContent = text + "\n";
    el.appendChild(span);
  }
}

// ════════════════════════════════════════════════
// 11. Main
// ════════════════════════════════════════════════

async function main() {
  const el = document.getElementById("output");
  if (el) el.textContent = "";

  log("╔════════════════════════════════════════════════════════╗");
  log("║  BitNet 1.58-bit WebGPU Kernel – Optimised PoC        ║");
  log("║  Bit-Packed · Branchless · Tiled                       ║");
  log("╚════════════════════════════════════════════════════════╝");
  log("");

  // ── Tokenizer + Embeddings + LM Head ──
  await initTokenizer();
  await loadEmbeddings();
  await loadLMHead();

  const device = await initWebGPU();
  log("✔ WebGPU device acquired", "info");

  // ── Allocate 26 persistent KV cache buffer pairs ──
  for (let i = 0; i < NUM_LAYERS; i++) {
    kCacheBufs.push(device.createBuffer({
      label: `kv-cache-k-layer-${i}`,
      size:  MAX_SEQ_LEN * KV_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    vCacheBufs.push(device.createBuffer({
      label: `kv-cache-v-layer-${i}`,
      size:  MAX_SEQ_LEN * KV_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
  }
  const kvTotalKB = NUM_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM * 4 / 1024;
  log(`  KV Cache allocated: ${NUM_LAYERS} layers × 2 × ${MAX_SEQ_LEN} × ${KV_DIM} × 4 = ${(kvTotalKB / 1024).toFixed(1)} MB`, 'info');
  log("");

  // ─────────────────────────────────────────────
  // TEST 1 – 1D Bit-Packed Branchless Kernel
  // ─────────────────────────────────────────────

  const N = 1024;
  log(`━━━ Test 1: 1D Bit-Packed Branchless Kernel (N = ${N}) ━━━`);
  log("");

  const { inputs, weights } = generateTestData(N);
  const packedPreview        = packWeights(weights);

  log(`  Inputs        : ${N} × f32 = ${(N * 4).toLocaleString()} bytes`);
  log(
    `  Packed weights: ${packedPreview.length} × u32 = ` +
    `${packedPreview.byteLength.toLocaleString()} bytes  ` +
    `(was ${(N * 4).toLocaleString()} bytes unpacked)`,
    "info",
  );
  log(
    `  Compression   : ${((1 - packedPreview.byteLength / (N * 4)) * 100).toFixed(1)}% smaller`,
    "info",
  );
  log("");

  // CPU reference
  const cpuT0       = performance.now();
  const cpuResult1D = cpuReference1D(inputs, weights);
  const cpuMs1D     = performance.now() - cpuT0;

  // GPU
  const {
    results: gpuResult1D,
    gpuMs:   gpuMs1D,
  } = await run1DKernel(device, inputs, weights);

  // Validate
  const v1 = compareResults(cpuResult1D, gpuResult1D);

  log(`  CPU time : ${cpuMs1D.toFixed(3)} ms`);
  log(`  GPU time : ${gpuMs1D.toFixed(3)} ms  (incl. pipeline + readback)`);
  log(`  Max |err|: ${v1.maxErr.toExponential(2)}  (at index ${v1.idx})`);
  log("");

  // Detail table (first 16 elements)
  log("  ┌───────┬──────────┬────────┬──────────┬──────────┐");
  log("  │  Idx  │   Input  │ Weight │    CPU   │    GPU   │");
  log("  ├───────┼──────────┼────────┼──────────┼──────────┤");
  for (let i = 0; i < Math.min(16, N); i++) {
    const inp = inputs[i].toFixed(3).padStart(8);
    const w   = String(weights[i]).padStart(6);
    const c   = cpuResult1D[i].toFixed(3).padStart(8);
    const g   = gpuResult1D[i].toFixed(3).padStart(8);
    log(`  │ ${String(i).padStart(5)} │ ${inp} │ ${w} │ ${c} │ ${g} │`);
  }
  if (N > 16) log("  │  ...  │    ...   │   ...  │    ...   │    ...   │");
  log("  └───────┴──────────┴────────┴──────────┴──────────┘");
  log("");

  if (v1.pass) {
    log("  ✅ PASS – 1D kernel: GPU output matches CPU reference!", "pass");
  } else {
    log("  ❌ FAIL – 1D kernel: mismatch detected!", "fail");
    log(
      `     index ${v1.idx}: CPU = ${cpuResult1D[v1.idx]}, GPU = ${gpuResult1D[v1.idx]}`,
      "fail",
    );
  }
  log("");

  // ─────────────────────────────────────────────
  // TEST 2 – 2D Tiled Matrix–Vector Multiply
  // ─────────────────────────────────────────────

  const M = 4096;
  const K = 4096;

  log(`━━━ Test 2: 2D Tiled Mat-Vec Kernel (${M}×${K}, ` +
      `tile = ${TILE_K}, wg = ${WORKGROUP_SIZE}) ━━━`);
  log(`  Uses var<workgroup> shared memory for input tile caching`, "info");
  log("");

  const { inputVec, weightMatrix } = generateMatrixTestData(M, K);
  const { packed: packedMat, packedStride } = packWeightMatrix(
    weightMatrix,
    M,
    K,
  );

  log(`  Weight matrix : ${M} × ${K} = ${(M * K).toLocaleString()} ternary values`);
  log(
    `  Packed buffer : ${packedMat.length} × u32 = ` +
    `${packedMat.byteLength.toLocaleString()} bytes  ` +
    `(was ${(M * K * 4).toLocaleString()} bytes)`,
    "info",
  );
  log(`  Packed stride : ${packedStride} u32s / row`);
  log("");

  // CPU reference
  const cpuT0_2     = performance.now();
  const cpuResult2D = cpuReferenceMatVec(weightMatrix, inputVec, M, K);
  const cpuMs2D     = performance.now() - cpuT0_2;

  // GPU
  const {
    results:   gpuResult2D,
    setupMs:   setupMs2D,
    computeMs: computeMs2D,
  } = await run2DKernel(device, weightMatrix, inputVec, M, K);

  // Validate
  const v2 = compareResults(cpuResult2D, gpuResult2D, 0.1);

  log(`  CPU time     : ${cpuMs2D.toFixed(3)} ms`);
  log(`  GPU setup    : ${setupMs2D.toFixed(3)} ms  (buffers + pipeline compile)`);
  log(`  GPU compute  : ${computeMs2D.toFixed(3)} ms  (submit + readback only)`);
  log(`  Max |err|    : ${v2.maxErr.toExponential(2)}  (at row ${v2.idx})`);
  log("");

  log(`  ⚡ Compute race: CPU ${cpuMs2D.toFixed(3)} ms  vs  GPU ${computeMs2D.toFixed(3)} ms`);
  const speedup = cpuMs2D / computeMs2D;
  log(`  → GPU is ${speedup.toFixed(1)}× ${speedup >= 1 ? "faster" : "slower"} than CPU (compute only)`, "info");
  log("");

  const PREVIEW_ROWS = Math.min(16, M);
  log("  ┌───────┬──────────────┬──────────────┬──────────────┐");
  log("  │  Row  │     CPU      │     GPU      │    |Δ|       │");
  log("  ├───────┼──────────────┼──────────────┼──────────────┤");
  for (let i = 0; i < PREVIEW_ROWS; i++) {
    const c   = cpuResult2D[i].toFixed(4).padStart(12);
    const g   = gpuResult2D[i].toFixed(4).padStart(12);
    const d   = Math.abs(cpuResult2D[i] - gpuResult2D[i]).toExponential(2).padStart(12);
    log(`  │ ${String(i).padStart(5)} │ ${c} │ ${g} │ ${d} │`);
  }
  if (M > PREVIEW_ROWS) log("  │  ...  │     ...      │     ...      │     ...      │");
  log("  └───────┴──────────────┴──────────────┴──────────────┘");
  log("");

  if (v2.pass) {
    log("  ✅ PASS – 2D tiled kernel: GPU output matches CPU reference!", "pass");
  } else {
    log("  ❌ FAIL – 2D tiled kernel: mismatch detected!", "fail");
    log(
      `     row ${v2.idx}: CPU = ${cpuResult2D[v2.idx]}, ` +
      `GPU = ${gpuResult2D[v2.idx]}`,
      "fail",
    );
  }
  log("");

  // ─────────────────────────────────────────────
  // TEST 3 – Real AI Weights Integration (down_proj only – legacy)
  // ─────────────────────────────────────────────

  log(`━━━ Test 3: Real AI Weights – microsoft/bitnet-b1.58-2B-4T ━━━`);
  log(`  Layer: model.layers.0.mlp.down_proj  (${REAL_M} × ${REAL_K})`);
  log("");

  try {
    // Fetch pre-packed binary weights (now from weights/ directory)
    log("  Fetching weights/bitnet_layer_0_down_proj.bin ...");
    const resp = await fetch("weights/bitnet_layer_0_down_proj.bin");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    const buf  = await resp.arrayBuffer();
    const packedReal = new Uint32Array(buf);

    const expectedStride = Math.ceil(REAL_K / 16);
    const expectedLen    = REAL_M * expectedStride;
    log(`  Loaded ${packedReal.length.toLocaleString()} u32 words ` +
        `(${packedReal.byteLength.toLocaleString()} bytes)`, "info");
    log(`  Expected: ${expectedLen.toLocaleString()} u32 words ` +
        `(${REAL_M} rows × ${expectedStride} stride)`, "info");
    if (packedReal.length !== expectedLen) {
      throw new Error(
        `Size mismatch: got ${packedReal.length}, expected ${expectedLen}`,
      );
    }
    log("");

    // Store weights for interactive use (legacy)
    realWeights = packedReal;

    // Create mock input vector (random floats in [-1, 1])
    const mockInput = new Float32Array(REAL_K);
    for (let i = 0; i < REAL_K; i++) {
      mockInput[i] = Math.random() * 2 - 1;
    }
    log(`  Mock input vector: ${REAL_K} × f32 (random [-1, 1])`);
    log("");

    // Run on GPU with pre-packed weights
    const {
      results:   gpuResult3,
      setupMs:   setupMs3,
      computeMs: computeMs3,
    } = await run2DKernelPacked(device, packedReal, mockInput, REAL_M, REAL_K);

    log(`  GPU setup   : ${setupMs3.toFixed(3)} ms  (buffers + pipeline)`);
    log(`  GPU compute : ${computeMs3.toFixed(3)} ms  (dispatch + readback)`);
    log("");

    // Print first 10 output values
    log("  ┌───────┬──────────────────┐");
    log("  │  Row  │    GPU Output     │");
    log("  ├───────┼──────────────────┤");
    for (let i = 0; i < 10; i++) {
      const val = gpuResult3[i].toFixed(6).padStart(16);
      log(`  │ ${String(i).padStart(5)} │ ${val} │`);
    }
    log("  │  ...  │       ...        │");
    log("  └───────┴──────────────────┘");
    log("");

    // Quick sanity: check output isn't all zeros
    let nonZero = 0;
    for (let i = 0; i < gpuResult3.length; i++) {
      if (gpuResult3[i] !== 0) nonZero++;
    }
    const pctNonZero = ((nonZero / gpuResult3.length) * 100).toFixed(1);
    log(`  Output stats: ${nonZero.toLocaleString()} / ${gpuResult3.length.toLocaleString()} ` +
        `non-zero values (${pctNonZero}%)`, "info");

    if (nonZero > 0) {
      log("  ✅ PASS – Real AI weights: GPU produced non-trivial output!", "pass");
    } else {
      log("  ⚠️  WARNING – All outputs are zero (unexpected)", "fail");
    }
  } catch (err) {
    log(`  ❌ FAIL – Could not run real weights test: ${err.message}`, "fail");
    log(`     Make sure weights/ directory is served alongside index.html`, "info");
  }
  log("");

  // ─────────────────────────────────────────────
  // Load All 26 Transformer Layers
  // ─────────────────────────────────────────────

  try {
    await loadAllLayers();
  } catch (err) {
    log(`  ❌ FAIL – Could not load transformer layers: ${err.message}`, "fail");
    log(`     Run: python extract_all_layers.py  to generate the weights/ directory`, "info");
  }

  // ─────────────────────────────────────────────
  // Load Final RMSNorm Weights
  // ─────────────────────────────────────────────
  try {
    log('Loading final RMSNorm weights …', 'info');
    const fnResp = await fetch('weights/bitnet_final_norm.bin');
    if (!fnResp.ok) throw new Error(`HTTP ${fnResp.status}`);
    const fnBuf = await fnResp.arrayBuffer();
    finalNormWeights = new Float32Array(fnBuf);
    log(`  ✔ Final norm loaded: ${finalNormWeights.length} dims ` +
        `(${(fnBuf.byteLength / 1024).toFixed(1)} KB)`, 'pass');
  } catch (err) {
    log(`  ❌ FAIL – Could not load final norm: ${err.message}`, 'fail');
    log(`     Run: python extract_rmsnorm.py  to generate the norm weights`, 'info');
  }
  log("");

  // ─────────────────────────────────────────────
  // Summary
  // ─────────────────────────────────────────────

  log("════════════════════════════════════════════════════════");
  const allPass = v1.pass && v2.pass;
  if (allPass) {
    log("  ✅ ALL TESTS PASSED", "pass bold");
  } else {
    log("  ❌ SOME TESTS FAILED", "fail bold");
  }
  log("════════════════════════════════════════════════════════");

  // ── Keep device + weights alive for interactive use ──
  gpuDevice = device;
  log("");

  if (layers.length === NUM_LAYERS && lmHeadWeights) {
    log(`✔ Interactive mode ready – ${NUM_LAYERS}-layer brain loaded! Type a prompt and click Generate!`, "pass");
    const inputEl = document.getElementById("user-text");
    const btnEl   = document.getElementById("compute-btn");
    if (inputEl) inputEl.disabled = false;
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.addEventListener("click", onComputeClick);
    }
    // Also allow Enter key in the input
    if (inputEl) {
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !btnEl.disabled) onComputeClick();
      });
    }
  } else {
    log("⚠  Interactive mode unavailable (weights failed to load)", "fail");
  }
}

// ════════════════════════════════════════════════
// 12. Auto-Regressive Generation
// ════════════════════════════════════════════════

// EOS token IDs for Llama 3
const EOS_TOKENS = new Set([128001, 128009]);

/**
 * Run one full transformer forward pass for a single token.
 *
 * Pipeline:  Embedding → [Layer 0..25: RMSNorm → Self-Attention →
 *            Residual → RMSNorm → SwiGLU MLP → Residual] →
 *            RMSNorm → LM Head
 *
 * Advances the global `seqPos` by 1 and updates ALL 26 KV caches.
 *
 * @param {number} tokenId – original Llama 3 token ID
 * @returns {Promise<{logits: Float32Array, totalMs: number}>}
 */
async function forward(tokenId) {
  const t0 = performance.now();

  // 1. Embedding lookup
  let hidden = getEmbeddingByTokenId(tokenId);

  // 2. Deep transformer loop: 30 layers
  for (let i = 0; i < NUM_LAYERS; i++) {
    const layer = layers[i];

    // ── Pre-attention RMSNorm (with learned γ weights) ──
    const normedForAttn = simpleRMSNorm(hidden, layer.attnNorm);

    // ── Self-Attention (Q/K/V proj → RoPE → KV cache → GQA → sub-norm → O proj) ──
    const attnResult = await runSelfAttention(
      gpuDevice,
      layer.qW, layer.kW, layer.vW, layer.oW,
      normedForAttn, seqPos,
      kCacheBufs[i], vCacheBufs[i],
      layer.qScale, layer.kScale, layer.vScale, layer.oScale,
      layer.attnSubNorm,
    );

    // ── Post-attention residual connection ──
    const postAttn = new Float32Array(HIDDEN_DIM);
    for (let d = 0; d < HIDDEN_DIM; d++) {
      postAttn[d] = hidden[d] + attnResult.results[d];
    }

    // ── Pre-MLP RMSNorm (with learned γ weights) ──
    const normedForMLP = simpleRMSNorm(postAttn, layer.mlpNorm);

    // ── ReLU²-gated MLP (gate → up → ReLU²·mul → sub-norm → down) ──
    const { results: mlpOut } = await runFullMLP(
      gpuDevice,
      layer.gateW, layer.upW, layer.downW,
      normedForMLP, HIDDEN_DIM, MLP_DIM,
      layer.gateScale, layer.upScale, layer.downScale,
      layer.ffnSubNorm,
    );

    // ── Post-MLP residual connection ──
    hidden = new Float32Array(HIDDEN_DIM);
    for (let d = 0; d < HIDDEN_DIM; d++) {
      hidden[d] = postAttn[d] + mlpOut[d];
    }
  }

  // Advance sequence position (once per token, shared across all layers)
  seqPos += 1;

  // 3. Final RMSNorm (with learned γ weights)
  const normedFinal = simpleRMSNorm(hidden, finalNormWeights);

  // 4. LM Head → logits
  const { results: logits } = await runLMHeadKernel(
    gpuDevice, lmHeadWeights, normedFinal, LM_HEAD_ROWS, HIDDEN_DIM,
  );

  const totalMs = performance.now() - t0;
  return { logits, totalMs };
}

/**
 * Decode a sparse logits index back to a printable string.
 *
 * @param {number} sparseIdx – index into the logits array (dense row)
 * @returns {string} decoded token text
 */
function decodeToken(sparseIdx) {
  if (!reverseVocabMap || reverseVocabMap[String(sparseIdx)] === undefined) {
    return `<unmapped:${sparseIdx}>`;
  }
  const origId = parseInt(reverseVocabMap[String(sparseIdx)], 10);
  if (!tokenizer) return `<no-tokenizer:${origId}>`;
  try {
    return tokenizer.decode([origId], true);
  } catch {
    return `<decode-error:${origId}>`;
  }
}

/**
 * Auto-regressive text generation with prefill + decode.
 *
 * Phase 1 – Prefill: feed every prompt token through forward() to
 *   warm the KV cache.  No text is emitted.
 *
 * Phase 2 – Decode:  argmax the last prefill logits → first new token.
 *   Then loop: emit token → check EOS → forward(newToken) → argmax.
 *
 * @param {string}   prompt    – user input text
 * @param {number}   maxTokens – max NEW tokens to generate (default 20)
 * @param {function} onToken   – callback(tokenStr, stats) for streaming UI
 * @returns {Promise<{text: string, tokens: number, totalMs: number}>}
 */
async function generateText(prompt, maxTokens = 20, onToken = null) {
  if (!tokenizer) throw new Error('Tokenizer not initialised.');

  // Reset KV cache for new sequence
  seqPos = 0;

  // Tokenize the prompt
  const encoded = tokenizer.encode(prompt);
  const promptIds = Array.from(encoded.ids);
  if (promptIds.length === 0) throw new Error('Empty prompt after tokenization.');

  const t0 = performance.now();
  let generatedText = '';
  let generatedCount = 0;
  let lastLogits = null;

  // ── Phase 1: Prefill ──
  // Feed each prompt token through the model to build the KV cache.
  // We only need the logits from the LAST prompt token.
  for (let i = 0; i < promptIds.length; i++) {
    const { logits } = await forward(promptIds[i]);
    lastLogits = logits;
  }

  // ── Phase 2: Decode ──
  // Sample with Repetition Penalty + Temperature + Top-K + Top-P.
  // Track generated sparse indices so the penalty can tax repeat tokens.
  const generatedHistory = new Map();   // sparseIdx → count

  while (generatedCount < maxTokens) {
    const sparseIdx = sampleToken(lastLogits, 1.0, 50, 0.9, generatedHistory, 1.5);
    generatedHistory.set(sparseIdx, (generatedHistory.get(sparseIdx) || 0) + 1);

    const origId = reverseVocabMap
      ? parseInt(reverseVocabMap[String(sparseIdx)] ?? '-1', 10)
      : -1;

    // Check EOS
    if (EOS_TOKENS.has(origId)) break;

    // Decode and accumulate
    const tokenStr = decodeToken(sparseIdx);
    generatedText += tokenStr;
    generatedCount++;

    // Stream callback
    if (onToken) {
      onToken(tokenStr, {
        tokenNum: generatedCount,
        tokenId: origId,
        sparseIdx,
        logit: lastLogits[sparseIdx],
      });
    }

    // Guard against exceeding KV cache
    if (seqPos >= MAX_SEQ_LEN - 1) break;

    // Forward pass for the newly generated token
    const result = await forward(origId);
    lastLogits = result.logits;
  }

  const totalMs = performance.now() - t0;
  return { text: generatedText, tokens: generatedCount, totalMs };
}

// ════════════════════════════════════════════════
// 13. Interactive Compute Handler
// ════════════════════════════════════════════════

/**
 * Called when the user clicks "Generate" (or presses Enter).
 *
 * Auto-regressive generation with streaming UI updates.
 *   1. Tokenize prompt → prefill KV cache
 *   2. Decode loop: argmax → emit token → forward → repeat
 *   3. Stop on EOS or maxTokens
 */
async function onComputeClick() {
  const inputEl  = document.getElementById("user-text");
  const btnEl    = document.getElementById("compute-btn");
  const outEl    = document.getElementById("interactive-output");
  if (!inputEl || !outEl) return;

  const text = inputEl.value.trim();
  if (!text) return;

  // Disable controls while generating
  btnEl.disabled   = true;
  inputEl.disabled = true;
  outEl.style.display = "block";

  // Clear and set up output with prompt echo
  outEl.textContent = "";

  const promptSpan = document.createElement("span");
  promptSpan.className = "info";
  promptSpan.textContent = text;
  outEl.appendChild(promptSpan);

  // Streaming text node for generated tokens
  const genSpan = document.createElement("span");
  genSpan.className = "pass bold";
  outEl.appendChild(genSpan);

  const statsEl = document.createElement("div");
  statsEl.style.marginTop = "0.75rem";
  outEl.appendChild(statsEl);

  const ilog = (msg, cls) => {
    const span = document.createElement("span");
    if (cls) span.className = cls;
    span.textContent = msg + "\n";
    statsEl.appendChild(span);
  };

  try {
    const maxTokens = 20;
    let tokenCount = 0;

    const result = await generateText(text, maxTokens, (tokenStr, stats) => {
      // Stream each token into the UI as it's generated
      genSpan.textContent += tokenStr;
      tokenCount = stats.tokenNum;
    });

    // Final stats
    ilog("");
    ilog(`═══════════════════════════════════════════`, "info");
    ilog(`  Generated ${result.tokens} token(s) in ${result.totalMs.toFixed(1)} ms`, "info");
    if (result.tokens > 0) {
      ilog(`  Avg: ${(result.totalMs / (result.tokens + tokenizer.encode(text).ids.length)).toFixed(1)} ms/token (incl. prefill)`, "info");
    }
    ilog(`═══════════════════════════════════════════`, "info");

  } catch (err) {
    ilog(`❌ Error: ${err.message}`, "fail");
    console.error(err);
  } finally {
    btnEl.disabled   = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

main().catch((err) => {
  const msg = `\n❌ Fatal error: ${err.message ?? err}`;
  console.error(msg, err);
  const el = document.getElementById("output");
  if (el) {
    el.textContent = "";
    const span     = document.createElement("span");
    span.className = "fail";
    span.textContent = msg;
    el.appendChild(span);
  }
});
