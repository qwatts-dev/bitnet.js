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
let gateWeights     = null;   // Uint32Array – packed gate_proj (6912×2560)
let upWeights       = null;   // Uint32Array – packed up_proj   (6912×2560)
let downWeights     = null;   // Uint32Array – packed down_proj (2560×6912)
let embeddingData   = null;   // Uint16Array – FP16 sparse embed_tokens
let vocabMap        = null;   // Object – original token ID → dense row index
const HIDDEN_DIM    = 2560;   // hidden_size of bitnet-b1.58-2B-4T
const MLP_DIM       = 6912;   // intermediate_size (SwiGLU)
const REAL_M        = 2560;   // rows  (down_proj output dim) — kept for Test 3
const REAL_K        = 6912;   // cols  (down_proj input  dim) — kept for Test 3
const EMBED_DIM     = 2560;   // hidden_size of bitnet-b1.58-2B-4T

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

  // Thread 0 writes the final dot-product to the output.
  if (tid == 0u) {
    output_vec[row] = shared_acc[0];
  }
}
`;

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
// 5c. WGSL – SiLU · Element-wise Multiply (SwiGLU fusion)
// ════════════════════════════════════════════════
//
//  SwiGLU(x) = SiLU(gate_proj(x)) ⊙ up_proj(x)
//  SiLU(g)   = g / (1 + exp(-g))    (a.k.a. swish with β=1)
//
//  This shader takes two f32 vectors of size N and writes:
//    result[i] = SiLU(gate[i]) * up[i]

const SHADER_SILU_MUL = /* wgsl */ `

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

  let gate = gate_vec[idx];
  let up   = up_vec[idx];

  // SiLU (swish β=1): gate * σ(gate) = gate / (1 + exp(-gate))
  let silu = gate / (1.0 + exp(-gate));

  result_vec[idx] = silu * up;
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
  const device = await adapter.requestDevice();
  device.lost.then((info) =>
    console.error(`WebGPU device lost: ${info.message}`),
  );
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
  device.queue.writeBuffer(
    uniformBuf,
    0,
    new Uint32Array([M, K, packedStride, 0]),
  );

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
  device.queue.writeBuffer(
    uniformBuf,
    0,
    new Uint32Array([M, K, packedStride, 0]),
  );

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
// 8b. SiLU · Multiply Kernel – Pipeline, Buffers, Dispatch
// ════════════════════════════════════════════════

/**
 * Run the SiLU·Mul element-wise kernel on the GPU.
 *   result[i] = SiLU(gateData[i]) * upData[i]
 *
 * @param {GPUDevice}    device   – WebGPU device
 * @param {Float32Array} gateData – gate_proj output (length n)
 * @param {Float32Array} upData   – up_proj output   (length n)
 * @param {number}       n        – vector length
 * @returns {{ results: Float32Array, computeMs: number }}
 */
async function runSiLUMulKernel(device, gateData, upData, n) {
  const setupT0 = performance.now();

  // ── Buffers ──
  const gateBuf = device.createBuffer({
    label: "silu-gate",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gateBuf, 0, new Float32Array(gateData));

  const upBuf = device.createBuffer({
    label: "silu-up",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(upBuf, 0, new Float32Array(upData));

  const resultBuf = device.createBuffer({
    label: "silu-result",
    size:  n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "silu-staging",
    size:  n * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "silu-params",
    size:  4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([n]));

  // ── Pipeline ──
  const module = device.createShaderModule({ code: SHADER_SILU_MUL });
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
// 8c. Unified Full MLP – Zero-Copy GPU Pipeline
// ════════════════════════════════════════════════
//
// Runs the entire SwiGLU MLP block in a SINGLE command encoder
// submission.  All intermediate tensors stay in VRAM — no CPU
// round-trips until the final result is read back.
//
//   Pass 1: gate_out  = gate_proj  × input   (M_mid × K_in → M_mid)
//   Pass 2: up_out    = up_proj    × input   (M_mid × K_in → M_mid)
//   Pass 3: silu_out  = SiLU(gate_out) ⊙ up_out   (element-wise)
//   Pass 4: result    = down_proj  × silu_out     (M_out × M_mid → M_out)
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
 * @returns {{ results: Float32Array, setupMs: number, computeMs: number }}
 */
async function runFullMLP(device, gateW, upW, downW, inputVec, hiddenDim, mlpDim) {
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
  //  gate/up params: M=mlpDim, K=hiddenDim, stride=gateStride
  const gateUpUniform = device.createBuffer({
    label: "mlp-gate-up-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    gateUpUniform, 0,
    new Uint32Array([mlpDim, hiddenDim, gateStride, 0]),
  );

  //  SiLU·mul params: n=mlpDim
  const siluUniform = device.createBuffer({
    label: "mlp-silu-params",
    size:  4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(siluUniform, 0, new Uint32Array([mlpDim]));

  //  down params: M=hiddenDim, K=mlpDim, stride=downStride
  const downUniform = device.createBuffer({
    label: "mlp-down-params",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    downUniform, 0,
    new Uint32Array([hiddenDim, mlpDim, downStride, 0]),
  );

  // ── Compile pipelines ──────────────────────────────────
  const matModule  = device.createShaderModule({ code: SHADER_2D_TILED });
  const siluModule = device.createShaderModule({ code: SHADER_SILU_MUL });

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
    compute: { module: siluModule, entryPoint: "main" },
  });

  // ── Bind groups ────────────────────────────────────────
  const gateBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: gateWeightBuf } },
      { binding: 2, resource: { buffer: gateOutBuf } },
      { binding: 3, resource: { buffer: gateUpUniform } },
    ],
  });

  const upBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: upWeightBuf } },
      { binding: 2, resource: { buffer: upOutBuf } },
      { binding: 3, resource: { buffer: gateUpUniform } },  // same dims
    ],
  });

  const siluBG = device.createBindGroup({
    layout: siluBGL,
    entries: [
      { binding: 0, resource: { buffer: gateOutBuf } },
      { binding: 1, resource: { buffer: upOutBuf } },
      { binding: 2, resource: { buffer: siluOutBuf } },
      { binding: 3, resource: { buffer: siluUniform } },
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

  // Pass 3: silu_out = SiLU(gate_out) ⊙ up_out
  const pass3 = encoder.beginComputePass();
  pass3.setPipeline(siluPipeline);
  pass3.setBindGroup(0, siluBG);
  pass3.dispatchWorkgroups(Math.ceil(mlpDim / WORKGROUP_SIZE));
  pass3.end();

  // Pass 4: result = down_proj × silu_out
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
    gateUpUniform, siluUniform, downUniform,
  ].forEach((b) => b.destroy());

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

  // ── Tokenizer + Embeddings ──
  await initTokenizer();
  await loadEmbeddings();

  const device = await initWebGPU();
  log("✔ WebGPU device acquired", "info");
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
    // Fetch pre-packed binary weights
    log("  Fetching bitnet_layer_0_down_proj.bin ...");
    const resp = await fetch("bitnet_layer_0_down_proj.bin");
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
    log(`     Make sure bitnet_layer_0_down_proj.bin is served alongside index.html`, "info");
  }
  log("");

  // ─────────────────────────────────────────────
  // Load Full MLP Weights (gate_proj + up_proj + down_proj)
  // ─────────────────────────────────────────────

  log(`━━━ Loading Full SwiGLU MLP Weights (Layer 0) ━━━`);
  log("");

  try {
    const mlpFiles = [
      { name: "gate_proj", file: "bitnet_layer_0_gate_proj.bin", M: MLP_DIM, K: HIDDEN_DIM },
      { name: "up_proj",   file: "bitnet_layer_0_up_proj.bin",   M: MLP_DIM, K: HIDDEN_DIM },
      { name: "down_proj", file: "bitnet_layer_0_down_proj.bin", M: HIDDEN_DIM, K: MLP_DIM },
    ];

    const [gateResp, upResp, downResp] = await Promise.all(
      mlpFiles.map(f => fetch(f.file)),
    );
    const responses = [gateResp, upResp, downResp];

    for (let i = 0; i < mlpFiles.length; i++) {
      if (!responses[i].ok) {
        throw new Error(`${mlpFiles[i].file}: HTTP ${responses[i].status}`);
      }
    }

    const [gateBuf, upBuf, downBuf] = await Promise.all(
      responses.map(r => r.arrayBuffer()),
    );

    gateWeights = new Uint32Array(gateBuf);
    upWeights   = new Uint32Array(upBuf);
    downWeights = new Uint32Array(downBuf);

    for (let i = 0; i < mlpFiles.length; i++) {
      const w = [gateWeights, upWeights, downWeights][i];
      const { name, M: mRows, K: kCols } = mlpFiles[i];
      const stride = Math.ceil(kCols / 16);
      const expected = mRows * stride;
      log(`  ${name}: ${w.length.toLocaleString()} u32 (${(w.byteLength / 1024).toFixed(1)} KB)` +
          ` — ${mRows}×${kCols}, stride ${stride}`, "info");
      if (w.length !== expected) {
        throw new Error(`${name} size mismatch: got ${w.length}, expected ${expected}`);
      }
    }

    log("");
    log("  ✅ All MLP weights loaded – SwiGLU pipeline ready", "pass");
  } catch (err) {
    log(`  ❌ FAIL – Could not load MLP weights: ${err.message}`, "fail");
    log(`     Run: python extract_full_mlp.py  to generate all .bin files`, "info");
    gateWeights = null;
    upWeights   = null;
    downWeights = null;
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

  if (gateWeights && upWeights && downWeights) {
    log("✔ Interactive mode ready (full SwiGLU MLP) – type text above and click Compute!", "pass");
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
// 12. Interactive Compute Handler
// ════════════════════════════════════════════════

/**
 * Called when the user clicks "Compute" (or presses Enter).
 *
 * Full SwiGLU MLP pipeline (Layer 0) – unified GPU orchestration:
 *   All 4 compute passes run in a SINGLE command encoder submission.
 *   Intermediate tensors never leave VRAM.
 */
async function onComputeClick() {
  const inputEl  = document.getElementById("user-text");
  const btnEl    = document.getElementById("compute-btn");
  const outEl    = document.getElementById("interactive-output");
  if (!inputEl || !outEl) return;

  const text = inputEl.value.trim();
  if (!text) return;

  // Disable controls while running
  btnEl.disabled    = true;
  inputEl.disabled  = true;
  outEl.style.display = "block";
  outEl.textContent   = "Computing…\n";

  const ilog = (msg, cls) => {
    const span = document.createElement("span");
    if (cls) span.className = cls;
    span.textContent = msg + "\n";
    outEl.appendChild(span);
  };

  try {
    // 1. Tokenize → real embedding (2560 dims, no padding)
    const emb = getRealEmbedding(text);

    // 2. Run full SwiGLU MLP in a single GPU submission
    const { results, setupMs, computeMs } = await runFullMLP(
      gpuDevice, gateWeights, upWeights, downWeights,
      emb, HIDDEN_DIM, MLP_DIM,
    );

    // 3. Display results
    outEl.textContent = "";
    ilog(`Input : "${text}"`);
    ilog("");
    ilog(`SwiGLU MLP Pipeline (Layer 0) – Unified GPU:`, "info");
    ilog(`  Setup (buffers + pipelines) : ${setupMs.toFixed(3)} ms`, "info");
    ilog(`  Compute (submit + readback) : ${computeMs.toFixed(3)} ms`, "info");
    ilog(`  ─────────────────────────────────────`);
    ilog(`  Total                       : ${(setupMs + computeMs).toFixed(3)} ms`, "info");
    ilog("");

    // Show last 10 values of final output (2560-dim vector)
    const startIdx = results.length - 10;
    ilog("┌───────┬──────────────────┐");
    ilog("│  Idx  │    GPU Output     │");
    ilog("├───────┼──────────────────┤");
    for (let i = startIdx; i < results.length; i++) {
      const val = results[i].toFixed(6).padStart(16);
      ilog(`│ ${String(i).padStart(5)} │ ${val} │`);
    }
    ilog("└───────┴──────────────────┘");
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
