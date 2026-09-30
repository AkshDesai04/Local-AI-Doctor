"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  INSTALLER_SIZE_LIMIT,
  artifactName,
  checkBundle,
  checkInstallerSize,
  checkSelfCheck,
  listFiles,
  parseSelfCheck,
  resolveVariant,
} = require("../scripts/build-guard.cjs");

const CUDA_BUNDLE = [
  "local-ai-doctor-backend.exe",
  "_internal/torch/lib/torch_cuda.dll",
  "_internal/torch/lib/cudart64_12.dll",
  "_internal/torch/lib/cublas64_12.dll",
  "_internal/torch/lib/cublasLt64_12.dll",
  "_internal/torch/lib/cudnn64_9.dll",
  "_internal/torch/lib/nvrtc64_120_0.dll",
  "_internal/torchvision/cudart64_12.dll",
  "_internal/bitsandbytes/libbitsandbytes_cuda128.dll",
];

function report(overrides = {}) {
  return {
    torch_version: "2.8.0+cu128",
    torch_cuda_build: "12.8",
    cuda_available: true,
    device_name: "Fixture GPU",
    device_capability: [8, 9],
    bitsandbytes_version: null,
    variant_expected: "cuda",
    problems: [],
    ...overrides,
  };
}

test("the variant defaults to CUDA and rejects anything unknown", () => {
  assert.equal(resolveVariant(undefined), "cuda");
  assert.equal(resolveVariant(""), "cuda");
  assert.equal(resolveVariant(" CPU "), "cpu");
  assert.throws(() => resolveVariant("rocm"), /must be "cuda" or "cpu"/);
});

test("installer names carry the version and the PyTorch variant", () => {
  assert.equal(artifactName("0.1.5", "cuda"), "Local-AI-Doctor-0.1.5-cuda.exe");
  assert.equal(artifactName("0.1.5-beta.42", "cpu"), "Local-AI-Doctor-0.1.5-beta.42-cpu.exe");
});

test("the self-check report is the last JSON line of the output", () => {
  const parsed = parseSelfCheck(`warning from torch\r\n${JSON.stringify(report())}\r\n`);
  assert.equal(parsed.torch_version, "2.8.0+cu128");
  assert.throws(() => parseSelfCheck("Traceback (most recent call last):"), /did not print a JSON report/);
});

test("the self-check must match the requested variant", () => {
  assert.deepEqual(checkSelfCheck(report(), "cuda", { requireCuda: true }), []);
  assert.deepEqual(checkSelfCheck(report({ torch_version: "2.8.0+cpu", torch_cuda_build: null, cuda_available: false }), "cpu"), []);
  assert.match(checkSelfCheck(report({ torch_version: "2.8.0", torch_cuda_build: null }), "cuda")[0], /expected a cuda PyTorch build/);
  assert.match(checkSelfCheck(report({ torch_cuda_build: "12.6" }), "cuda")[0], /expected a cuda PyTorch build/);
  assert.match(checkSelfCheck(report(), "cpu")[0], /expected a cpu PyTorch build/);
});

test("a GPU build machine requires usable CUDA and passes on backend problems", () => {
  assert.deepEqual(checkSelfCheck(report({ cuda_available: false }), "cuda"), []);
  assert.match(checkSelfCheck(report({ cuda_available: false }), "cuda", { requireCuda: true })[0], /cannot use CUDA/);
  assert.deepEqual(checkSelfCheck(report({ problems: ["matmul mismatch"] }), "cuda"), ["matmul mismatch"]);
});

test("a CUDA bundle needs the core runtime DLLs from the pinned wheels", () => {
  assert.deepEqual(checkBundle(CUDA_BUNDLE, "cuda"), []);
  const missing = CUDA_BUNDLE.filter((file) => !file.endsWith("cudnn64_9.dll"));
  assert.deepEqual(checkBundle(missing, "cuda"), ["missing _internal/torch/lib/cudnn64_9.dll"]);
  assert.deepEqual(checkBundle(["_internal/torch/lib/torch_cpu.dll"], "cpu"), []);
  assert.deepEqual(checkBundle(CUDA_BUNDLE, "cpu"), ["the CPU bundle contains torch_cuda.dll"]);
});

test("CUDA DLLs outside the pinned wheel directories are rejected", () => {
  const leaked = [...CUDA_BUNDLE, "_internal/nvrtc64_120_0.dll", "_internal\\cuBLAS64_12.dll", "_internal/av.libs/avcodec-62.dll"];
  assert.deepEqual(checkBundle(leaked, "cuda"), [
    "CUDA DLL outside the pinned wheels: _internal/nvrtc64_120_0.dll",
    "CUDA DLL outside the pinned wheels: _internal/cuBLAS64_12.dll",
  ]);
});

test("installers must stay below 1.95 GiB", () => {
  assert.equal(INSTALLER_SIZE_LIMIT, 2_093_796_556);
  assert.deepEqual(checkInstallerSize(INSTALLER_SIZE_LIMIT - 1), []);
  assert.match(checkInstallerSize(INSTALLER_SIZE_LIMIT)[0], /1\.95 GiB/);
  assert.match(checkInstallerSize(2_115_333_309)[0], /2115333309 bytes/);
});

test("bundle files are listed relative to the bundle root", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lad-guard-"));
  try {
    fs.mkdirSync(path.join(root, "_internal", "torch", "lib"), { recursive: true });
    fs.writeFileSync(path.join(root, "_internal", "torch", "lib", "torch_cuda.dll"), "");
    fs.writeFileSync(path.join(root, "local-ai-doctor-backend.exe"), "");
    assert.deepEqual(
      listFiles(root).map((file) => file.replaceAll("\\", "/")).sort(),
      ["_internal/torch/lib/torch_cuda.dll", "local-ai-doctor-backend.exe"],
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
