/**
 * bitnet.js  –  v0.11.1
 *
 * BitNet b1.58-2B-4T WebGPU inference engine with:
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
 * API (v0.11.1 – Object-Oriented)
 * ────────────────────────────────
 *   const engine = new BitNetEngine();
 *   await engine.init();                           // load weights & WebGPU
 *   await engine.generate("Hello!", onToken, 20);  // auto-regressive gen
 *   engine.reset();                                // clear KV cache
 *
 * Usage
 * ─────
 *   Browser  : <script type="module" src="bitnet.js">
 *   Node ≥ 22: node --experimental-webgpu bitnet.js
 */

// ════════════════════════════════════════════════
// Tokenizer (@huggingface/tokenizers – standalone, ~8.3 kB)
// ════════════════════════════════════════════════

import { Tokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers';

// ════════════════════════════════════════════════
// Architecture Constants (module-level)
// ════════════════════════════════════════════════

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

// EOS token IDs for Llama 3
const EOS_TOKENS = new Set([128001, 128009]);

const WORKGROUP_SIZE   = 64;
const TILE_K           = 256;   // tile width for 2D kernel
const ELEMS_PER_THREAD = TILE_K / WORKGROUP_SIZE; // 4

// ════════════════════════════════════════════════
// WGSL Shader Constants (module-level)
// ════════════════════════════════════════════════

// 1. WGSL – Bit-Packed Branchless 1D Kernel
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

  if (idx >= params.n) { return; }

  let inp = inputs[idx];

  let pack_idx = idx / 16u;
  let bit_pos  = (idx % 16u) * 2u;
  let code     = (packed_weights[pack_idx] >> bit_pos) & 3u;

  let bit0     = code & 1u;
  let bit1     = (code >> 1u) & 1u;
  let mask_pos = 0u - bit0;
  let mask_neg = 0u - bit1;

  let pos_val = bitcast<f32>(bitcast<u32>(inp) & mask_pos);
  let neg_val = bitcast<f32>(bitcast<u32>(inp) & mask_neg);

  result[idx] = pos_val - neg_val;
}
`;

// 2. WGSL – 2D Tiled Matrix–Vector Multiply
const SHADER_2D_TILED = /* wgsl */ `

const TILE_K_C: u32         = ${TILE_K}u;
const WG_SIZE_C: u32        = ${WORKGROUP_SIZE}u;
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD}u;

struct MatParams {
  M: u32,
  K: u32,
  packed_stride: u32,
  weight_scale: f32,
}

@group(0) @binding(0) var<storage, read>       input_vec:      array<f32>;
@group(0) @binding(1) var<storage, read>       packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_vec:     array<f32>;
@group(0) @binding(3) var<uniform>             params:         MatParams;

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

    for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {
      let local_idx  = tid * ELEMS_PER_THREAD + e;
      let global_col = t * TILE_K_C + local_idx;
      tile[local_idx] = select(0.0, input_vec[global_col],
                               global_col < params.K);
    }

    workgroupBarrier();

    for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {
      let local_idx  = tid * ELEMS_PER_THREAD + e;
      let global_col = t * TILE_K_C + local_idx;

      let in_bounds  = u32(global_col < params.K);
      let bound_mask = 0u - in_bounds;

      let pack_idx = row * params.packed_stride + global_col / 16u;
      let bit_pos  = (global_col % 16u) * 2u;
      let code     = (packed_weights[pack_idx] >> bit_pos) & 3u;

      let bit0     = code & 1u;
      let bit1     = (code >> 1u) & 1u;
      let mask_pos = (0u - bit0) & bound_mask;
      let mask_neg = (0u - bit1) & bound_mask;

      let inp     = tile[local_idx];
      let pos_val = bitcast<f32>(bitcast<u32>(inp) & mask_pos);
      let neg_val = bitcast<f32>(bitcast<u32>(inp) & mask_neg);
      acc = acc + pos_val - neg_val;
    }

    workgroupBarrier();
  }

  shared_acc[tid] = acc;
  workgroupBarrier();

  for (var s: u32 = WG_SIZE_C / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_acc[tid] = shared_acc[tid] + shared_acc[tid + s];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output_vec[row] = shared_acc[0] * params.weight_scale;
  }
}
`;

// 5c. WGSL – ReLU² · Element-wise Multiply (SwiGLU fusion)
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

  let gate = max(0.0, gate_vec[idx]);
  result_vec[idx] = (gate * gate) * up_vec[idx];
}
`;

// 5c′. WGSL – In-Place RMSNorm (GPU)
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

  var partial: f32 = 0.0;
  for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
    let val = vec[i];
    partial = partial + val * val;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();

  for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
    }
    workgroupBarrier();
  }

  let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(n) + EPS);

  for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
    vec[i] = vec[i] * inv_rms * gamma[i];
  }
}
`;

// 5d. WGSL – RoPE + KV Cache Write
const SHADER_ROPE_CACHE = /* wgsl */ `

struct RoPEParams {
  seq_pos:  u32,
  q_dim:    u32,
  kv_dim:   u32,
  head_dim: u32,
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
  let half_head = params.head_dim / 2u;
  let q_pairs   = params.q_dim  / 2u;
  let kv_pairs  = params.kv_dim / 2u;

  if (pair_idx >= q_pairs) { return; }

  let pos = f32(params.seq_pos);

  let head = pair_idx / half_head;
  let d    = pair_idx % half_head;

  let freq  = 1.0 / pow(${ROPE_THETA}, f32(d * 2u) / f32(params.head_dim));
  let theta = pos * freq;
  let cos_t = cos(theta);
  let sin_t = sin(theta);

  // ── Apply RoPE to Q ──
  {
    let i0 = head * params.head_dim + d;
    let i1 = i0 + half_head;
    let q0 = q_vec[i0];
    let q1 = q_vec[i1];
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

    let cache_off = params.seq_pos * params.kv_dim;
    k_cache[cache_off + i0] = k0r;
    k_cache[cache_off + i1] = k1r;

    v_cache[cache_off + i0] = v_vec[i0];
    v_cache[cache_off + i1] = v_vec[i1];
  }
}
`;

// 5e. WGSL – Grouped-Query Attention (GQA)
const SHADER_GQA_ATTENTION = /* wgsl */ `

const WG_ATTN: u32   = ${HEAD_DIM}u;
const MAX_SEQ: u32   = ${MAX_SEQ_LEN}u;
const GQA_GROUP: u32 = ${GQA_GROUP_SIZE}u;

struct AttnParams {
  seq_len:     u32,
  kv_dim:      u32,
  head_dim:    u32,
  num_q_heads: u32,
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

  for (var p: u32 = 0u; p < params.seq_len; p = p + 1u) {
    let q_val = q_vec[q_offset + tid];
    let k_val = k_cache[p * params.kv_dim + kv_offset + tid];
    temp[tid] = q_val * k_val;
    workgroupBarrier();

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

  // Softmax: find max
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

  // exp(score - max) and sum
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

  // Normalize
  if (tid < params.seq_len) {
    scores[tid] = scores[tid] / sum_exp;
  }
  workgroupBarrier();

  // Weighted V sum
  var out_val: f32 = 0.0;
  for (var p: u32 = 0u; p < params.seq_len; p = p + 1u) {
    let v_val = v_cache[p * params.kv_dim + kv_offset + tid];
    out_val = out_val + scores[p] * v_val;
  }

  attn_out[q_offset + tid] = out_val;
}
`;

// 5c-LM. WGSL – Dense LM Head Mat-Vec (Float32, NOT ternary)
const SHADER_LM_HEAD = /* wgsl */ `

const WG_SIZE: u32 = ${WORKGROUP_SIZE}u;

struct Params {
  M: u32,
  K: u32,
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

  var acc: f32 = 0.0;
  let row_offset = row * params.K;
  for (var col: u32 = tid; col < params.K; col = col + WG_SIZE) {
    acc = acc + weight_mat[row_offset + col] * input_vec[col];
  }

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
// Pure Helper Functions (module-level)
// ════════════════════════════════════════════════

/**
 * Fetch a URL using the browser Cache API so repeated loads are instant.
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
 */
function fp16ToNumber(h) {
  const sign = (h >>> 15) & 1;
  const exp  = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;

  let val;
  if (exp === 0) {
    val = (mant / 1024) * Math.pow(2, -14);
  } else if (exp === 31) {
    val = mant === 0 ? Infinity : NaN;
  } else {
    val = (1 + mant / 1024) * Math.pow(2, exp - 15);
  }
  return sign ? -val : val;
}

/**
 * Pack a 1-D array of ternary weights {-1, 0, +1} into a Uint32Array.
 * 16 weights per u32, 2 bits each.
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
    inputs[i] = rng() * 20 - 10;
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

/**
 * CPU-side RMSNorm with optional learned gamma weights.
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

/** Return the index of the maximum value in a typed array. */
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
// GPU Dispatch Functions (module-level, pure)
// ════════════════════════════════════════════════

async function run1DKernel(device, inputData, weightData) {
  const n             = inputData.length;
  const packedWeights = packWeights(weightData);

  const inputBuf = device.createBuffer({
    label: "1d-input", size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputData));

  const weightBuf = device.createBuffer({
    label: "1d-packed-weights", size: packedWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, packedWeights);

  const resultBuf = device.createBuffer({
    label: "1d-result", size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "1d-staging", size: n * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "1d-params", size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([n]));

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

async function run2DKernel(device, weightMatrix, inputVec, M, K) {
  const setupT0 = performance.now();

  const { packed, packedStride } = packWeightMatrix(weightMatrix, M, K);

  const inputBuf = device.createBuffer({
    label: "2d-input", size: K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const weightBuf = device.createBuffer({
    label: "2d-packed-weights", size: packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, packed);

  const resultBuf = device.createBuffer({
    label: "2d-result", size: M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "2d-staging", size: M * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "2d-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformData = new ArrayBuffer(16);
  const uniformU32 = new Uint32Array(uniformData);
  const uniformF32 = new Float32Array(uniformData);
  uniformU32[0] = M;
  uniformU32[1] = K;
  uniformU32[2] = packedStride;
  uniformF32[3] = 1.0;
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array(uniformData));

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

async function run2DKernelPacked(device, packedWeights, inputVec, M, K) {
  const packedStride = Math.ceil(K / 16);
  const setupT0 = performance.now();

  const inputBuf = device.createBuffer({
    label: "2d-input", size: K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const weightBuf = device.createBuffer({
    label: "2d-packed-weights", size: packedWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, packedWeights);

  const resultBuf = device.createBuffer({
    label: "2d-result", size: M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "2d-staging", size: M * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "2d-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  {
    const uData = new ArrayBuffer(16);
    const uU32  = new Uint32Array(uData);
    const uF32  = new Float32Array(uData);
    uU32[0] = M;
    uU32[1] = K;
    uU32[2] = packedStride;
    uF32[3] = 1.0;
    device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uData));
  }

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

async function runReLU2MulKernel(device, gateData, upData, n) {
  const setupT0 = performance.now();

  const gateBuf = device.createBuffer({
    label: "relu2-gate", size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gateBuf, 0, new Float32Array(gateData));

  const upBuf = device.createBuffer({
    label: "relu2-up", size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(upBuf, 0, new Float32Array(upData));

  const resultBuf = device.createBuffer({
    label: "relu2-result", size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "relu2-staging", size: n * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "relu2-params", size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([n]));

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

async function runLMHeadKernel(device, lmWeights, inputVec, M, K) {
  const setupT0 = performance.now();

  const inputBuf = device.createBuffer({
    label: "lm-head-input", size: K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const weightBuf = device.createBuffer({
    label: "lm-head-weights", size: lmWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuf, 0, lmWeights);

  const resultBuf = device.createBuffer({
    label: "lm-head-result", size: M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuf = device.createBuffer({
    label: "lm-head-staging", size: M * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const uniformBuf = device.createBuffer({
    label: "lm-head-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([M, K, 0, 0]));

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

async function runFullMLP(device, gateW, upW, downW, inputVec, hiddenDim, mlpDim,
                          gateScale = 1.0, upScale = 1.0, downScale = 1.0, ffnSubNorm = null) {
  const setupT0 = performance.now();

  const gateStride = Math.ceil(hiddenDim / 16);
  const downStride = Math.ceil(mlpDim / 16);

  const inputBuf = device.createBuffer({
    label: "mlp-input", size: hiddenDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const gateWeightBuf = device.createBuffer({
    label: "mlp-gate-weights", size: gateW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gateWeightBuf, 0, gateW);

  const upWeightBuf = device.createBuffer({
    label: "mlp-up-weights", size: upW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(upWeightBuf, 0, upW);

  const downWeightBuf = device.createBuffer({
    label: "mlp-down-weights", size: downW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(downWeightBuf, 0, downW);

  const gateOutBuf = device.createBuffer({
    label: "mlp-gate-out", size: mlpDim * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const upOutBuf = device.createBuffer({
    label: "mlp-up-out", size: mlpDim * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const siluOutBuf = device.createBuffer({
    label: "mlp-silu-out", size: mlpDim * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const resultBuf = device.createBuffer({
    label: "mlp-result", size: hiddenDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuf = device.createBuffer({
    label: "mlp-staging", size: hiddenDim * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  function writeMatUniform(buf, Mu, Ku, stride, scale) {
    const d = new ArrayBuffer(16);
    new Uint32Array(d)[0] = Mu;
    new Uint32Array(d)[1] = Ku;
    new Uint32Array(d)[2] = stride;
    new Float32Array(d)[3] = scale;
    device.queue.writeBuffer(buf, 0, new Uint8Array(d));
  }

  const gateUniform = device.createBuffer({
    label: "mlp-gate-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeMatUniform(gateUniform, mlpDim, hiddenDim, gateStride, gateScale);

  const upUniform = device.createBuffer({
    label: "mlp-up-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeMatUniform(upUniform, mlpDim, hiddenDim, gateStride, upScale);

  const relu2Uniform = device.createBuffer({
    label: "mlp-relu2-params", size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(relu2Uniform, 0, new Uint32Array([mlpDim]));

  const downUniform = device.createBuffer({
    label: "mlp-down-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeMatUniform(downUniform, hiddenDim, mlpDim, downStride, downScale);

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

  const ffnNormGammaBuf = device.createBuffer({
    label: "mlp-ffn-sub-norm-gamma", size: mlpDim * 4,
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

  const computeT0 = performance.now();
  const encoder = device.createCommandEncoder();

  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(matPipeline);
  pass1.setBindGroup(0, gateBG);
  pass1.dispatchWorkgroups(mlpDim);
  pass1.end();

  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(matPipeline);
  pass2.setBindGroup(0, upBG);
  pass2.dispatchWorkgroups(mlpDim);
  pass2.end();

  const pass3 = encoder.beginComputePass();
  pass3.setPipeline(siluPipeline);
  pass3.setBindGroup(0, siluBG);
  pass3.dispatchWorkgroups(Math.ceil(mlpDim / WORKGROUP_SIZE));
  pass3.end();

  const pass3b = encoder.beginComputePass();
  pass3b.setPipeline(normPipeline);
  pass3b.setBindGroup(0, normBG);
  pass3b.dispatchWorkgroups(1);
  pass3b.end();

  const pass4 = encoder.beginComputePass();
  pass4.setPipeline(matPipeline);
  pass4.setBindGroup(0, downBG);
  pass4.dispatchWorkgroups(hiddenDim);
  pass4.end();

  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, hiddenDim * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

  [
    inputBuf, gateWeightBuf, upWeightBuf, downWeightBuf,
    gateOutBuf, upOutBuf, siluOutBuf,
    resultBuf, stagingBuf,
    gateUniform, upUniform, relu2Uniform, downUniform,
    ffnNormGammaBuf, ffnNormUniform,
  ].forEach((b) => b.destroy());

  return { results, setupMs, computeMs };
}

async function runSelfAttention(device, qW, kW, vW, oW, inputVec, seqPosition, kCache, vCache,
                                qScale = 1.0, kScale = 1.0, vScale = 1.0, oScale = 1.0,
                                attnSubNorm = null) {
  const setupT0 = performance.now();

  const qStride = Math.ceil(HIDDEN_DIM / 16);
  const kStride = Math.ceil(HIDDEN_DIM / 16);
  const oStride = Math.ceil(Q_DIM / 16);

  const inputBuf = device.createBuffer({
    label: "attn-input", size: HIDDEN_DIM * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputVec));

  const qWeightBuf = device.createBuffer({
    label: "attn-q-weights", size: qW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(qWeightBuf, 0, qW);

  const kWeightBuf = device.createBuffer({
    label: "attn-k-weights", size: kW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(kWeightBuf, 0, kW);

  const vWeightBuf = device.createBuffer({
    label: "attn-v-weights", size: vW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vWeightBuf, 0, vW);

  const oWeightBuf = device.createBuffer({
    label: "attn-o-weights", size: oW.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(oWeightBuf, 0, oW);

  const qVecBuf = device.createBuffer({
    label: "attn-q-vec", size: Q_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const kVecBuf = device.createBuffer({
    label: "attn-k-vec", size: KV_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const vVecBuf = device.createBuffer({
    label: "attn-v-vec", size: KV_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const attnOutBuf = device.createBuffer({
    label: "attn-out", size: Q_DIM * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const resultBuf = device.createBuffer({
    label: "attn-o-result", size: HIDDEN_DIM * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuf = device.createBuffer({
    label: "attn-staging", size: HIDDEN_DIM * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  function writeAttnMatUniform(buf, Mu, Ku, stride, scale) {
    const d = new ArrayBuffer(16);
    new Uint32Array(d)[0] = Mu;
    new Uint32Array(d)[1] = Ku;
    new Uint32Array(d)[2] = stride;
    new Float32Array(d)[3] = scale;
    device.queue.writeBuffer(buf, 0, new Uint8Array(d));
  }

  const qUniform = device.createBuffer({
    label: "attn-q-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(qUniform, Q_DIM, HIDDEN_DIM, qStride, qScale);

  const kProjUniform = device.createBuffer({
    label: "attn-k-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(kProjUniform, KV_DIM, HIDDEN_DIM, kStride, kScale);

  const vProjUniform = device.createBuffer({
    label: "attn-v-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(vProjUniform, KV_DIM, HIDDEN_DIM, kStride, vScale);

  const ropeUniform = device.createBuffer({
    label: "attn-rope-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ropeUniform, 0, new Uint32Array([seqPosition, Q_DIM, KV_DIM, HEAD_DIM]));

  const gqaUniform = device.createBuffer({
    label: "attn-gqa-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gqaUniform, 0, new Uint32Array([seqPosition + 1, KV_DIM, HEAD_DIM, NUM_Q_HEADS]));

  const oUniform = device.createBuffer({
    label: "attn-o-params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writeAttnMatUniform(oUniform, HIDDEN_DIM, Q_DIM, oStride, oScale);

  const normGammaBuf = device.createBuffer({
    label: "attn-sub-norm-gamma", size: Q_DIM * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (attnSubNorm) device.queue.writeBuffer(normGammaBuf, 0, attnSubNorm);

  const normUniform = device.createBuffer({
    label: "attn-sub-norm-params", size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(normUniform, 0, new Uint32Array([Q_DIM]));

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

  const qBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: qWeightBuf } },
      { binding: 2, resource: { buffer: qVecBuf } },
      { binding: 3, resource: { buffer: qUniform } },
    ],
  });

  const kBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: kWeightBuf } },
      { binding: 2, resource: { buffer: kVecBuf } },
      { binding: 3, resource: { buffer: kProjUniform } },
    ],
  });

  const vBG = device.createBindGroup({
    layout: matBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: vWeightBuf } },
      { binding: 2, resource: { buffer: vVecBuf } },
      { binding: 3, resource: { buffer: vProjUniform } },
    ],
  });

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

  const normBG = device.createBindGroup({
    layout: normBGL,
    entries: [
      { binding: 0, resource: { buffer: attnOutBuf } },
      { binding: 1, resource: { buffer: normGammaBuf } },
      { binding: 2, resource: { buffer: normUniform } },
    ],
  });

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

  const computeT0 = performance.now();
  const encoder = device.createCommandEncoder();

  const p1 = encoder.beginComputePass();
  p1.setPipeline(matPipeline);
  p1.setBindGroup(0, qBG);
  p1.dispatchWorkgroups(Q_DIM);
  p1.end();

  const p2 = encoder.beginComputePass();
  p2.setPipeline(matPipeline);
  p2.setBindGroup(0, kBG);
  p2.dispatchWorkgroups(KV_DIM);
  p2.end();

  const p3 = encoder.beginComputePass();
  p3.setPipeline(matPipeline);
  p3.setBindGroup(0, vBG);
  p3.dispatchWorkgroups(KV_DIM);
  p3.end();

  const p4 = encoder.beginComputePass();
  p4.setPipeline(ropePipeline);
  p4.setBindGroup(0, ropeBG);
  p4.dispatchWorkgroups(Math.ceil(Q_DIM / 2 / WORKGROUP_SIZE));
  p4.end();

  const p5 = encoder.beginComputePass();
  p5.setPipeline(gqaPipeline);
  p5.setBindGroup(0, gqaBG);
  p5.dispatchWorkgroups(NUM_Q_HEADS);
  p5.end();

  const p5b = encoder.beginComputePass();
  p5b.setPipeline(normPipeline);
  p5b.setBindGroup(0, normBG);
  p5b.dispatchWorkgroups(1);
  p5b.end();

  const p6 = encoder.beginComputePass();
  p6.setPipeline(matPipeline);
  p6.setBindGroup(0, oBG);
  p6.dispatchWorkgroups(HIDDEN_DIM);
  p6.end();

  encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, HIDDEN_DIM * 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();

  const computeMs = performance.now() - computeT0;

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
// BitNetEngine Class
// ════════════════════════════════════════════════

export class BitNetEngine {

  constructor() {
    // ── Instance state (encapsulated) ──
    this.device          = null;   // GPUDevice
    this.tokenizer       = null;   // Tokenizer instance
    this.layers          = [];     // layers[0..29]
    this.kCacheBufs      = [];     // GPUBuffer per layer – persistent KV cache (K)
    this.vCacheBufs      = [];     // GPUBuffer per layer – persistent KV cache (V)
    this.seqPos          = 0;      // current token position in sequence
    this.vocabMap        = null;   // Object – original token ID → dense row index
    this.embeddingData   = null;   // Uint16Array – FP16 sparse embed_tokens
    this.lmHeadWeights   = null;   // Float32Array – dense LM head (vocab-sliced)
    this.reverseVocabMap = null;   // Object – dense row index → original token ID
    this.finalNormWeights = null;  // Float32Array – learned RMSNorm weights for final norm
    this.layerScales     = null;   // Object – per-layer weight scales
    this.realWeights     = null;   // Uint32Array – kept for Test 3 compat
  }

  // ════════════════════════════════════════════════
  // Public API
  // ════════════════════════════════════════════════

  /**
   * Initialise the engine: request WebGPU adapter, load tokenizer,
   * embeddings, LM head, all transformer layers, final norm, and
   * allocate the KV cache buffers.
   *
   * @param {string} weightsPath – relative path to weights directory
   */
  async init(weightsPath = 'weights') {
    await this._initTokenizer();
    await this._loadEmbeddings(weightsPath);
    await this._loadLMHead(weightsPath);

    this.device = await this._initWebGPU();
    log("✔ WebGPU device acquired", "info");

    // Allocate persistent KV cache buffer pairs
    for (let i = 0; i < NUM_LAYERS; i++) {
      this.kCacheBufs.push(this.device.createBuffer({
        label: `kv-cache-k-layer-${i}`,
        size:  MAX_SEQ_LEN * KV_DIM * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
      this.vCacheBufs.push(this.device.createBuffer({
        label: `kv-cache-v-layer-${i}`,
        size:  MAX_SEQ_LEN * KV_DIM * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
    }
    const kvTotalKB = NUM_LAYERS * 2 * MAX_SEQ_LEN * KV_DIM * 4 / 1024;
    log(`  KV Cache allocated: ${NUM_LAYERS} layers × 2 × ${MAX_SEQ_LEN} × ${KV_DIM} × 4 = ${(kvTotalKB / 1024).toFixed(1)} MB`, 'info');
    log("");

    await this._loadAllLayers(weightsPath);

    // Load final RMSNorm weights
    try {
      log('Loading final RMSNorm weights …', 'info');
      const fnResp = await fetch(`${weightsPath}/bitnet_final_norm.bin`);
      if (!fnResp.ok) throw new Error(`HTTP ${fnResp.status}`);
      const fnBuf = await fnResp.arrayBuffer();
      this.finalNormWeights = new Float32Array(fnBuf);
      log(`  ✔ Final norm loaded: ${this.finalNormWeights.length} dims ` +
          `(${(fnBuf.byteLength / 1024).toFixed(1)} KB)`, 'pass');
    } catch (err) {
      log(`  ❌ FAIL – Could not load final norm: ${err.message}`, 'fail');
      log(`     Run: python extract_rmsnorm.py  to generate the norm weights`, 'info');
    }
    log("");
  }

  /**
   * Auto-regressive text generation with prefill + decode.
   *
   * @param {string}   prompt    – user input text
   * @param {function} onToken   – callback(tokenStr, stats) for streaming UI
   * @param {number}   maxTokens – max NEW tokens to generate (default 20)
   * @returns {Promise<{text: string, tokens: number, totalMs: number}>}
   */
  async generate(prompt, onToken = null, maxTokens = 20) {
    if (!this.tokenizer) throw new Error('Tokenizer not initialised.');

    // Reset KV cache for new sequence
    this.seqPos = 0;

    // Tokenize the prompt
    const encoded = this.tokenizer.encode(prompt);
    const promptIds = Array.from(encoded.ids);
    if (promptIds.length === 0) throw new Error('Empty prompt after tokenization.');

    const t0 = performance.now();
    let generatedText = '';
    let generatedCount = 0;
    let lastLogits = null;

    // ── Phase 1: Prefill ──
    for (let i = 0; i < promptIds.length; i++) {
      const { logits } = await this._forward(promptIds[i]);
      lastLogits = logits;
    }

    // ── Phase 2: Decode ──
    const generatedHistory = new Map();

    while (generatedCount < maxTokens) {
      const sparseIdx = this._sampleToken(lastLogits, 1.0, 50, 0.9, generatedHistory, 1.5);
      generatedHistory.set(sparseIdx, (generatedHistory.get(sparseIdx) || 0) + 1);

      const origId = this.reverseVocabMap
        ? parseInt(this.reverseVocabMap[String(sparseIdx)] ?? '-1', 10)
        : -1;

      // Check EOS
      if (EOS_TOKENS.has(origId)) break;

      // Decode and accumulate
      const tokenStr = this._decodeToken(sparseIdx);
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
      if (this.seqPos >= MAX_SEQ_LEN - 1) break;

      // Forward pass for the newly generated token
      const result = await this._forward(origId);
      lastLogits = result.logits;
    }

    const totalMs = performance.now() - t0;
    return { text: generatedText, tokens: generatedCount, totalMs };
  }

  /**
   * Reset the sequence position (clears the KV cache logically).
   * Useful for starting a new conversation.
   */
  reset() {
    this.seqPos = 0;
  }

  // ════════════════════════════════════════════════
  // Private Methods
  // ════════════════════════════════════════════════

  /**
   * Run one full transformer forward pass for a single token.
   */
  async _forward(tokenId) {
    const t0 = performance.now();

    // 1. Embedding lookup
    let hidden = this._getEmbeddingByTokenId(tokenId);

    // 2. Deep transformer loop: 30 layers
    for (let i = 0; i < NUM_LAYERS; i++) {
      const layer = this.layers[i];

      // Pre-attention RMSNorm
      const normedForAttn = simpleRMSNorm(hidden, layer.attnNorm);

      // Self-Attention
      const attnResult = await runSelfAttention(
        this.device,
        layer.qW, layer.kW, layer.vW, layer.oW,
        normedForAttn, this.seqPos,
        this.kCacheBufs[i], this.vCacheBufs[i],
        layer.qScale, layer.kScale, layer.vScale, layer.oScale,
        layer.attnSubNorm,
      );

      // Post-attention residual connection
      const postAttn = new Float32Array(HIDDEN_DIM);
      for (let d = 0; d < HIDDEN_DIM; d++) {
        postAttn[d] = hidden[d] + attnResult.results[d];
      }

      // Pre-MLP RMSNorm
      const normedForMLP = simpleRMSNorm(postAttn, layer.mlpNorm);

      // ReLU²-gated MLP
      const { results: mlpOut } = await runFullMLP(
        this.device,
        layer.gateW, layer.upW, layer.downW,
        normedForMLP, HIDDEN_DIM, MLP_DIM,
        layer.gateScale, layer.upScale, layer.downScale,
        layer.ffnSubNorm,
      );

      // Post-MLP residual connection
      hidden = new Float32Array(HIDDEN_DIM);
      for (let d = 0; d < HIDDEN_DIM; d++) {
        hidden[d] = postAttn[d] + mlpOut[d];
      }
    }

    // Advance sequence position
    this.seqPos += 1;

    // 3. Final RMSNorm
    const normedFinal = simpleRMSNorm(hidden, this.finalNormWeights);

    // 4. LM Head → logits
    const { results: logits } = await runLMHeadKernel(
      this.device, this.lmHeadWeights, normedFinal, LM_HEAD_ROWS, HIDDEN_DIM,
    );

    const totalMs = performance.now() - t0;
    return { logits, totalMs };
  }

  /**
   * Sample a token from logits using Repetition Penalty + Temperature +
   * Top-K + Top-P (nucleus) sampling.
   */
  _sampleToken(logits, temperature = 1.0, topK = 50, topP = 0.9,
               generatedTokens = null, repetitionPenalty = 1.2) {
    const n = logits.length;

    // 0. Frequency-scaled repetition penalty
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

    // 1. Temperature scaling
    const scaled = new Float32Array(n);
    const invTemp = 1.0 / Math.max(temperature, 1e-8);
    for (let i = 0; i < n; i++) {
      scaled[i] = penalised[i] * invTemp;
    }

    // 2. Stable softmax
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

    // 3. Build index array and sort descending by prob
    const indices = new Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
    indices.sort((a, b) => probs[b] - probs[a]);

    // 4. Top-K
    const kCut = Math.min(topK, n);

    // 5. Top-P
    let cumProb = 0;
    let pCut = kCut;
    for (let i = 0; i < kCut; i++) {
      cumProb += probs[indices[i]];
      if (cumProb >= topP) {
        pCut = i + 1;
        break;
      }
    }

    // 6. Renormalize
    let renormSum = 0;
    for (let i = 0; i < pCut; i++) {
      renormSum += probs[indices[i]];
    }

    // 7. Weighted random sample
    let r = Math.random() * renormSum;
    for (let i = 0; i < pCut; i++) {
      r -= probs[indices[i]];
      if (r <= 0) return indices[i];
    }

    return indices[0];
  }

  /**
   * Decode a sparse logits index back to a printable string.
   */
  _decodeToken(sparseIdx) {
    if (!this.reverseVocabMap || this.reverseVocabMap[String(sparseIdx)] === undefined) {
      return `<unmapped:${sparseIdx}>`;
    }
    const origId = parseInt(this.reverseVocabMap[String(sparseIdx)], 10);
    if (!this.tokenizer) return `<no-tokenizer:${origId}>`;
    try {
      return this.tokenizer.decode([origId], true);
    } catch {
      return `<decode-error:${origId}>`;
    }
  }

  /**
   * Initialise the Llama-3 tokenizer.
   */
  async _initTokenizer() {
    const BASE = 'https://huggingface.co/Xenova/llama3-tokenizer/resolve/main';
    log('Loading tokenizer (Xenova/llama3-tokenizer) …', 'info');

    const [tokenizerJson, tokenizerConfig] = await Promise.all([
      fetchWithCache(`${BASE}/tokenizer.json`),
      fetchWithCache(`${BASE}/tokenizer_config.json`),
    ]);

    this.tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    log('✔ Tokenizer ready', 'info');
    log('');
  }

  /**
   * Initialise the WebGPU device with extended limits.
   */
  async _initWebGPU() {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error(
        "WebGPU is not available.\n" +
        "• Browser: use a recent Chromium-based browser.\n" +
        "• Node ≥ 22: run with  node --experimental-webgpu"
      );
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter found.");

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

  /**
   * Load sparse FP16 embeddings + vocab map.
   */
  async _loadEmbeddings(weightsPath = 'weights') {
    log('Loading sparse embeddings (FP16) …', 'info');

    const [mapResp, binResp] = await Promise.all([
      fetch(`${weightsPath}/vocab_map.json`),
      fetch(`${weightsPath}/sparse_embeddings.bin`),
    ]);
    if (!mapResp.ok) throw new Error(`vocab_map.json fetch failed: HTTP ${mapResp.status}`);
    if (!binResp.ok) throw new Error(`sparse_embeddings.bin fetch failed: HTTP ${binResp.status}`);

    this.vocabMap = await mapResp.json();
    const buf = await binResp.arrayBuffer();
    this.embeddingData = new Uint16Array(buf);

    const totalRows = this.embeddingData.length / EMBED_DIM;
    const mapEntries = Object.keys(this.vocabMap).length;
    log(`✔ Embeddings loaded: ${totalRows.toLocaleString()} rows × ${EMBED_DIM} dims ` +
        `(${(buf.byteLength / 1024 / 1024).toFixed(1)} MB, FP16)`, 'info');
    log(`✔ Vocab map loaded: ${mapEntries.toLocaleString()} token ID entries ` +
        `(${(JSON.stringify(this.vocabMap).length / 1024).toFixed(0)} KB)`, 'info');
    log('');
  }

  /**
   * Load sparse FP16 LM head weights and decode to Float32.
   */
  async _loadLMHead(weightsPath = 'weights') {
    log('Loading sparse LM head (FP16) …', 'info');

    const resp = await fetch(`${weightsPath}/sparse_lm_head.bin`);
    if (!resp.ok) throw new Error(`sparse_lm_head.bin fetch failed: HTTP ${resp.status}`);
    const buf = await resp.arrayBuffer();
    const fp16 = new Uint16Array(buf);

    const expectedLen = LM_HEAD_ROWS * HIDDEN_DIM;
    if (fp16.length !== expectedLen) {
      throw new Error(`LM head size mismatch: got ${fp16.length}, expected ${expectedLen}`);
    }

    this.lmHeadWeights = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
      this.lmHeadWeights[i] = fp16ToNumber(fp16[i]);
    }

    log(`✔ LM head loaded: ${LM_HEAD_ROWS.toLocaleString()} rows × ${HIDDEN_DIM} dims ` +
        `(${(buf.byteLength / 1024 / 1024).toFixed(1)} MB FP16 → ` +
        `${(this.lmHeadWeights.byteLength / 1024 / 1024).toFixed(1)} MB F32)`, 'info');

    // Build reverse vocab map
    if (this.vocabMap) {
      this.reverseVocabMap = {};
      for (const [tokenId, rowIdx] of Object.entries(this.vocabMap)) {
        this.reverseVocabMap[String(rowIdx)] = tokenId;
      }
      log(`✔ Reverse vocab map built: ${Object.keys(this.reverseVocabMap).length.toLocaleString()} entries`, 'info');
    }
    log('');
  }

  /**
   * Load all transformer layers from disk.
   */
  async _loadAllLayers(weightsPath = 'weights') {
    log(`━━━ Loading All ${NUM_LAYERS} Transformer Layers (${NUM_LAYERS * 7} matrices) ━━━`);
    log('');

    const PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
    const t0 = performance.now();
    let totalBytes = 0;

    const scalesResp = await fetch(`${weightsPath}/bitnet_layer_scales.json`);
    if (scalesResp.ok) {
      this.layerScales = await scalesResp.json();
      log(`  ✔ Weight scales loaded (${Object.keys(this.layerScales).length} layers)`, 'pass');
    } else {
      log(`  ⚠ bitnet_layer_scales.json not found – scales will default to 1.0`, 'fail');
      this.layerScales = {};
    }

    for (let i = 0; i < NUM_LAYERS; i++) {
      const layerT0 = performance.now();

      const projUrls = PROJ_NAMES.map(p => `${weightsPath}/bitnet_layer_${i}_${p}.bin`);
      const normUrls = [
        `${weightsPath}/bitnet_layer_${i}_attn_norm.bin`,
        `${weightsPath}/bitnet_layer_${i}_mlp_norm.bin`,
        `${weightsPath}/bitnet_layer_${i}_attn_sub_norm.bin`,
        `${weightsPath}/bitnet_layer_${i}_ffn_sub_norm.bin`,
      ];
      const urls = [...projUrls, ...normUrls];
      const responses = await Promise.all(urls.map(u => fetch(u)));

      for (let j = 0; j < responses.length; j++) {
        if (!responses[j].ok) {
          throw new Error(`${urls[j]}: HTTP ${responses[j].status}`);
        }
      }

      const buffers = await Promise.all(responses.map(r => r.arrayBuffer()));

      const ls = (this.layerScales && this.layerScales[String(i)]) || {};

      const layerWeights = {
        qW:      new Uint32Array(buffers[0]),
        kW:      new Uint32Array(buffers[1]),
        vW:      new Uint32Array(buffers[2]),
        oW:      new Uint32Array(buffers[3]),
        gateW:   new Uint32Array(buffers[4]),
        upW:     new Uint32Array(buffers[5]),
        downW:   new Uint32Array(buffers[6]),
        attnNorm:    new Float32Array(buffers[7]),
        mlpNorm:     new Float32Array(buffers[8]),
        attnSubNorm: new Float32Array(buffers[9]),
        ffnSubNorm:  new Float32Array(buffers[10]),
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

      this.layers.push(layerWeights);

      const layerMs = performance.now() - layerT0;
      log(`  Layer ${String(i).padStart(2)}/29 loaded: ` +
          `${(layerBytes / 1024 / 1024).toFixed(2)} MB  ` +
          `(${layerMs.toFixed(0)} ms)`, 'info');
    }

    const elapsed = performance.now() - t0;
    log('');
    log(`  ✅ All ${NUM_LAYERS} layers loaded: ${this.layers.length * 7} matrices, ` +
        `${(totalBytes / 1024 / 1024).toFixed(2)} MB total ` +
        `(${(elapsed / 1000).toFixed(1)}s)`, 'pass');
    log('');
  }

  /**
   * Look up the real high-precision embedding for `text` via the
   * sparse FP16 dictionary.
   */
  _getRealEmbedding(text) {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialised – call init() first.');
    }
    if (!this.embeddingData || !this.vocabMap) {
      throw new Error('Embeddings not loaded – call init() first.');
    }

    const encoded = this.tokenizer.encode(text);
    const ids = encoded.ids;

    if (ids.length === 0) {
      throw new Error('Tokenizer produced an empty sequence.');
    }

    const tokenId = ids[0];

    let rowIndex = this.vocabMap[String(tokenId)];
    let oov = false;
    if (rowIndex === undefined) {
      rowIndex = 0;
      oov = true;
    }

    const fp16Start = rowIndex * EMBED_DIM;
    const embedding = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) {
      embedding[i] = fp16ToNumber(this.embeddingData[fp16Start + i]);
    }

    const oovTag = oov ? ' [OOV → fallback row 0]' : '';
    log(`  Tokenized "${text.length > 40 ? text.slice(0, 37) + '…' : text}"` +
        ` → ${ids.length} token(s), first ID = ${tokenId}${oovTag}`, 'info');
    log(`  Real embedding (FP16→F32): ${EMBED_DIM} dims`, 'info');

    return embedding;
  }

  /**
   * Look up the embedding for a raw token ID (no tokenization step).
   */
  _getEmbeddingByTokenId(tokenId) {
    if (!this.embeddingData || !this.vocabMap) {
      throw new Error('Embeddings not loaded – call init() first.');
    }

    let rowIndex = this.vocabMap[String(tokenId)];
    if (rowIndex === undefined) {
      rowIndex = 0;
    }

    const fp16Start = rowIndex * EMBED_DIM;
    const embedding = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) {
      embedding[i] = fp16ToNumber(this.embeddingData[fp16Start + i]);
    }
    return embedding;
  }
}

// ════════════════════════════════════════════════
// UI Bootstrap (main + interactive handler)
// ════════════════════════════════════════════════

async function main() {
  const el = document.getElementById("output");
  if (el) el.textContent = "";

  log("╔════════════════════════════════════════════════════════╗");
  log("║  bitnet.js – v0.11.1                                   ║");
  log("║  BitNet b1.58 · WebGPU · Bit-Packed · Branchless       ║");
  log("╚════════════════════════════════════════════════════════╝");
  log("");

  // ── Instantiate the engine ──
  const engine = new BitNetEngine();

  // ── Tokenizer + Embeddings + LM Head ──
  await engine._initTokenizer();
  await engine._loadEmbeddings('weights');
  await engine._loadLMHead('weights');

  const device = await engine._initWebGPU();
  engine.device = device;
  log("✔ WebGPU device acquired", "info");

  // ── Allocate persistent KV cache buffer pairs ──
  for (let i = 0; i < NUM_LAYERS; i++) {
    engine.kCacheBufs.push(device.createBuffer({
      label: `kv-cache-k-layer-${i}`,
      size:  MAX_SEQ_LEN * KV_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    engine.vCacheBufs.push(device.createBuffer({
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

  const cpuT0       = performance.now();
  const cpuResult1D = cpuReference1D(inputs, weights);
  const cpuMs1D     = performance.now() - cpuT0;

  const {
    results: gpuResult1D,
    gpuMs:   gpuMs1D,
  } = await run1DKernel(device, inputs, weights);

  const v1 = compareResults(cpuResult1D, gpuResult1D);

  log(`  CPU time : ${cpuMs1D.toFixed(3)} ms`);
  log(`  GPU time : ${gpuMs1D.toFixed(3)} ms  (incl. pipeline + readback)`);
  log(`  Max |err|: ${v1.maxErr.toExponential(2)}  (at index ${v1.idx})`);
  log("");

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

  const cpuT0_2     = performance.now();
  const cpuResult2D = cpuReferenceMatVec(weightMatrix, inputVec, M, K);
  const cpuMs2D     = performance.now() - cpuT0_2;

  const {
    results:   gpuResult2D,
    setupMs:   setupMs2D,
    computeMs: computeMs2D,
  } = await run2DKernel(device, weightMatrix, inputVec, M, K);

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

    engine.realWeights = packedReal;

    const mockInput = new Float32Array(REAL_K);
    for (let i = 0; i < REAL_K; i++) {
      mockInput[i] = Math.random() * 2 - 1;
    }
    log(`  Mock input vector: ${REAL_K} × f32 (random [-1, 1])`);
    log("");

    const {
      results:   gpuResult3,
      setupMs:   setupMs3,
      computeMs: computeMs3,
    } = await run2DKernelPacked(device, packedReal, mockInput, REAL_M, REAL_K);

    log(`  GPU setup   : ${setupMs3.toFixed(3)} ms  (buffers + pipeline)`);
    log(`  GPU compute : ${computeMs3.toFixed(3)} ms  (dispatch + readback)`);
    log("");

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
  // Load All Transformer Layers
  // ─────────────────────────────────────────────

  try {
    await engine._loadAllLayers();
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
    engine.finalNormWeights = new Float32Array(fnBuf);
    log(`  ✔ Final norm loaded: ${engine.finalNormWeights.length} dims ` +
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

  log("");

  if (engine.layers.length === NUM_LAYERS && engine.lmHeadWeights) {
    log(`✔ Interactive mode ready – ${NUM_LAYERS}-layer brain loaded! Type a prompt and click Generate!`, "pass");
    const inputEl = document.getElementById("user-text");
    const btnEl   = document.getElementById("compute-btn");
    if (inputEl) inputEl.disabled = false;
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.addEventListener("click", () => onComputeClick(engine));
    }
    if (inputEl) {
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !btnEl.disabled) onComputeClick(engine);
      });
    }
  } else {
    log("⚠  Interactive mode unavailable (weights failed to load)", "fail");
  }
}

/**
 * Called when the user clicks "Generate" (or presses Enter).
 * Uses the BitNetEngine instance for generation.
 */
async function onComputeClick(engine) {
  const inputEl  = document.getElementById("user-text");
  const btnEl    = document.getElementById("compute-btn");
  const outEl    = document.getElementById("interactive-output");
  if (!inputEl || !outEl) return;

  const text = inputEl.value.trim();
  if (!text) return;

  btnEl.disabled   = true;
  inputEl.disabled = true;
  outEl.style.display = "block";

  outEl.textContent = "";

  const promptSpan = document.createElement("span");
  promptSpan.className = "info";
  promptSpan.textContent = text;
  outEl.appendChild(promptSpan);

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

    const result = await engine.generate(text, (tokenStr, stats) => {
      genSpan.textContent += tokenStr;
      tokenCount = stats.tokenNum;
    }, maxTokens);

    ilog("");
    ilog(`═══════════════════════════════════════════`, "info");
    ilog(`  Generated ${result.tokens} token(s) in ${result.totalMs.toFixed(1)} ms`, "info");
    if (result.tokens > 0) {
      ilog(`  Avg: ${(result.totalMs / (result.tokens + engine.tokenizer.encode(text).ids.length)).toFixed(1)} ms/token (incl. prefill)`, "info");
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
