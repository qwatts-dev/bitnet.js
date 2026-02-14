/**
 * bitnet-kernel.js
 *
 * A vanilla-JS proof-of-concept that runs a 1.58-bit (ternary-weight)
 * matrix–vector kernel entirely on the GPU via the WebGPU compute API.
 *
 * Instead of multiplying input × weight the WGSL shader uses only
 * additions and subtractions driven by the ternary weight values
 * {-1, 0, +1}, which is the core insight behind BitNet b1.58.
 *
 * Usage
 * ─────
 *   Browser  : import or <script type="module" src="bitnet-kernel.js">
 *   Node ≥ 22: node --experimental-webgpu bitnet-kernel.js
 *
 * If the built-in navigator.gpu is unavailable the script will print
 * an explanatory error and exit.
 */

// ──────────────────────────────────────────────
// 1. WGSL Compute Shader
// ──────────────────────────────────────────────
//
// Bindings
//   @group(0) @binding(0)  inputs   – array<f32>   (read)
//   @group(0) @binding(1)  weights  – array<i32>   (read)  values in {-1, 0, 1}
//   @group(0) @binding(2)  result   – array<f32>   (write)
//   @group(0) @binding(3)  params   – uniform { n: u32 } (element count)
//
// Each invocation processes one element.  Rather than multiplying
//   result[i] = input[i] * f32(weight[i])
// the shader branches on the weight:
//   weight ==  1  →  result[i] =  input[i]   (copy / add)
//   weight == -1  →  result[i] = -input[i]   (negate / subtract)
//   weight ==  0  →  result[i] =  0.0        (skip)
//
// This completely eliminates multiplication, which is the whole point
// of ternary-weight quantisation.
// ──────────────────────────────────────────────

const SHADER_SOURCE = /* wgsl */ `

struct Params {
  n: u32,          // number of elements
}

@group(0) @binding(0) var<storage, read>       inputs:  array<f32>;
@group(0) @binding(1) var<storage, read>       weights: array<i32>;
@group(0) @binding(2) var<storage, read_write> result:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;

  // Guard: do nothing for out-of-range invocations.
  if (idx >= params.n) {
    return;
  }

  let inp = inputs[idx];
  let w   = weights[idx];

  // ── Ternary logic (no multiplication!) ──
  //
  //  select(falseVal, trueVal, condition)
  //
  //  Step 1 – start at 0.0
  //  Step 2 – if w ==  1 ➜ add  inp   (result becomes +inp)
  //  Step 3 – if w == -1 ➜ subtract inp (result becomes -inp)
  //
  //  Because w ∈ {-1, 0, 1} at most one branch fires.

  var out: f32 = 0.0;
  out = select(out, inp, w == 1);       // +input when weight is  1
  out = select(out, -inp, w == -1);     // -input when weight is -1
  // (w == 0 → out stays 0.0)

  result[idx] = out;
}
`;

// ──────────────────────────────────────────────
// 2. Initialise WebGPU
// ──────────────────────────────────────────────

async function initWebGPU() {
  // In Node.js ≥ 22 with --experimental-webgpu, `navigator.gpu` is
  // available globally just like in the browser.
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error(
      "WebGPU is not available in this environment.\n" +
        "• Browser: use a recent Chromium-based browser.\n" +
        "• Node.js ≥ 22: run with  node --experimental-webgpu bitnet-kernel.js\n"
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error(
      "Failed to obtain a GPU adapter. " +
        "This usually means no compatible GPU / driver was found."
    );
  }

  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message}`);
  });

  console.log("✔ WebGPU device acquired");
  return device;
}

// ──────────────────────────────────────────────
// 3. Create GPU Buffers
// ──────────────────────────────────────────────

function createBuffers(device, inputData, weightData) {
  const n = inputData.length;

  // — Input buffer (f32) —
  const inputBuffer = device.createBuffer({
    label: "input-buffer",
    size: n * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, new Float32Array(inputData));

  // — Weight buffer (i32, values in {-1, 0, 1}) —
  const weightBuffer = device.createBuffer({
    label: "weight-buffer",
    size: n * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuffer, 0, new Int32Array(weightData));

  // — Result buffer (f32, GPU-writable, mappable for read-back) —
  const resultBuffer = device.createBuffer({
    label: "result-buffer",
    size: n * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // — Staging buffer (for reading results back to the CPU) —
  const stagingBuffer = device.createBuffer({
    label: "staging-buffer",
    size: n * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // — Uniform buffer containing the element count —
  const uniformBuffer = device.createBuffer({
    label: "params-uniform",
    size: 4, // one u32
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([n]));

  return { inputBuffer, weightBuffer, resultBuffer, stagingBuffer, uniformBuffer, n };
}

// ──────────────────────────────────────────────
// 4. Build Pipeline & Bind Group
// ──────────────────────────────────────────────

function buildPipeline(device, buffers) {
  const shaderModule = device.createShaderModule({
    label: "bitnet-shader",
    code: SHADER_SOURCE,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: "bitnet-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "bitnet-pipeline-layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: "bitnet-pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  const bindGroup = device.createBindGroup({
    label: "bitnet-bind-group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: buffers.inputBuffer } },
      { binding: 1, resource: { buffer: buffers.weightBuffer } },
      { binding: 2, resource: { buffer: buffers.resultBuffer } },
      { binding: 3, resource: { buffer: buffers.uniformBuffer } },
    ],
  });

  return { pipeline, bindGroup };
}

// ──────────────────────────────────────────────
// 5. Dispatch & Readback
// ──────────────────────────────────────────────

async function runKernel(device, pipeline, bindGroup, buffers) {
  const workgroupSize = 64;
  const dispatchCount = Math.ceil(buffers.n / workgroupSize);

  // Encode commands
  const encoder = device.createCommandEncoder({ label: "bitnet-encoder" });

  const pass = encoder.beginComputePass({ label: "bitnet-pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchCount);
  pass.end();

  // Copy result → staging so we can map-read it on the CPU
  encoder.copyBufferToBuffer(
    buffers.resultBuffer,
    0,
    buffers.stagingBuffer,
    0,
    buffers.n * Float32Array.BYTES_PER_ELEMENT
  );

  // Submit
  device.queue.submit([encoder.finish()]);

  // Map the staging buffer and read back
  await buffers.stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArray = new Float32Array(buffers.stagingBuffer.getMappedRange());
  const results = Array.from(copyArray);
  buffers.stagingBuffer.unmap();

  return results;
}

// ──────────────────────────────────────────────
// 6. Main – wire everything together
// ──────────────────────────────────────────────

// ──────────────────────────────────────────────
// Helper – write a line to both the page and the console
// ──────────────────────────────────────────────

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

async function main() {
  // Clear the "Initialising…" placeholder
  const el = document.getElementById("output");
  if (el) el.textContent = "";

  log("╔══════════════════════════════════════════════╗");
  log("║  BitNet 1.58-bit WebGPU Kernel – PoC        ║");
  log("╚══════════════════════════════════════════════╝");
  log("");

  // ── Test data ──
  const inputs  = [5.5, 2.1, 3.0, 8.2];
  const weights = [1,   0,  -1,   1  ];  // ternary: {-1, 0, +1}

  log(`Input values : [${inputs.join(", ")}]`);
  log(`Weights      : [${weights.join(", ")}]`);
  log("");

  // Step 1 – Acquire the GPU device
  const device = await initWebGPU();

  // Step 2 – Create buffers and upload data
  const buffers = createBuffers(device, inputs, weights);
  log(`✔ Buffers created  (n = ${buffers.n})`, "info");

  // Step 3 – Build the compute pipeline
  const { pipeline, bindGroup } = buildPipeline(device, buffers);
  log("✔ Compute pipeline compiled", "info");

  // Step 4 – Dispatch on the GPU and read back
  const results = await runKernel(device, pipeline, bindGroup, buffers);
  log("✔ Kernel executed & results read back", "info");
  log("");

  // ── Pretty-print ──
  log("┌────────┬────────┬────────────────────────────────┬────────┐");
  log("│ Input  │ Weight │ Operation                      │ Output │");
  log("├────────┼────────┼────────────────────────────────┼────────┤");
  for (let i = 0; i < inputs.length; i++) {
    const w = weights[i];
    const op =
      w ===  1 ? "+input  (add to accumulator)   " :
      w === -1 ? "-input  (subtract from accum.)  " :
                 " skip   (weight is zero)         ";
    log(
      `│ ${inputs[i].toFixed(1).padStart(6)} │ ${String(w).padStart(6)} │ ${op}│ ${results[i].toFixed(1).padStart(6)} │`
    );
  }
  log("└────────┴────────┴────────────────────────────────┴────────┘");

  // Expected: [5.5, 0.0, -3.0, 8.2]
  const expected = inputs.map((v, i) =>
    weights[i] === 1 ? v : weights[i] === -1 ? -v : 0
  );
  log("");
  log(`Expected : [${expected.join(", ")}]`);
  log(`Got      : [${results.join(", ")}]`);

  const pass = results.every((v, i) => Math.abs(v - expected[i]) < 1e-6);
  log(
    pass ? "\n✅ PASS – ternary kernel output matches expected values!"
         : "\n❌ FAIL – mismatch detected.",
    pass ? "pass" : "fail"
  );

  // Clean up
  device.destroy();
  log("\n✔ GPU device destroyed – done.", "info");
}

main().catch((err) => {
  const msg = `\n❌ Fatal error: ${err.message ?? err}`;
  console.error(msg);
  const el = document.getElementById("output");
  if (el) {
    el.textContent = "";
    const span = document.createElement("span");
    span.className = "fail";
    span.textContent = msg;
    el.appendChild(span);
  }
});
