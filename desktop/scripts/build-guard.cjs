"use strict";

// Release guards for the Windows desktop build. The exported functions are
// pure; the command-line entry point is used by the PowerShell build scripts.

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const manifest = require("../package.json");

const VARIANT_VARIABLE = "LAD_DESKTOP_TORCH_VARIANT";
const DEFAULT_VARIANT = "cuda";
const VARIANTS = {
  cuda: {
    torchSuffix: "+cu128",
    torchCuda: "12.8",
    requiredDlls: ["torch_cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"],
  },
  cpu: { torchSuffix: "+cpu", torchCuda: null, requiredDlls: [] },
};
// 1.95 GiB keeps headroom below GitHub's 2 GiB release-asset limit and the
// NSIS payload limit.
const INSTALLER_SIZE_LIMIT = 2_093_796_556;
const CUDA_RUNTIME_DLL = /^(nvrtc|cudart|cublas|cudnn|cufft|cusparse|cusolver|curand|nvjitlink).*\.dll$/i;
// Wheels that ship their own CUDA runtime DLLs next to their extension modules.
const CUDA_DLL_DIRECTORIES = ["_internal/torch/lib", "_internal/torchvision", "_internal/bitsandbytes"];

function resolveVariant(value) {
  const variant = String(value ?? "").trim().toLowerCase() || DEFAULT_VARIANT;
  if (!Object.hasOwn(VARIANTS, variant)) {
    throw new Error(`${VARIANT_VARIABLE} must be "cuda" or "cpu"; found "${value}".`);
  }
  return variant;
}

function artifactName(version, variant, pattern = manifest.build.win.artifactName) {
  return pattern.replace("${version}", version).replace(`\${env.${VARIANT_VARIABLE}}`, variant);
}

function parseSelfCheck(output) {
  const line = String(output).split(/\r?\n/).map((item) => item.trim()).reverse().find((item) => item.startsWith("{"));
  if (!line) throw new Error("The backend self-check did not print a JSON report.");
  return JSON.parse(line);
}

function checkSelfCheck(report, variant, { requireCuda = false } = {}) {
  const expected = VARIANTS[variant];
  const problems = [...(report.problems ?? [])];
  if (!String(report.torch_version ?? "").endsWith(expected.torchSuffix) || (report.torch_cuda_build ?? null) !== expected.torchCuda) {
    // Same wording as backend_launcher.py so a duplicate collapses into one line.
    problems.push(
      `expected a ${variant} PyTorch build (*${expected.torchSuffix}, CUDA ${expected.torchCuda ?? "none"}) `
      + `but found ${report.torch_version} (CUDA ${report.torch_cuda_build ?? "none"})`,
    );
  }
  if (requireCuda && report.cuda_available !== true) {
    problems.push("this machine has an NVIDIA GPU, but the bundled PyTorch cannot use CUDA");
  }
  return [...new Set(problems)];
}

// `files` are bundle-relative paths, as returned by listFiles().
function checkBundle(files, variant) {
  const normalized = files.map((file) => file.replaceAll("\\", "/"));
  const lower = new Set(normalized.map((file) => file.toLowerCase()));
  const problems = VARIANTS[variant].requiredDlls
    .filter((name) => !lower.has(`_internal/torch/lib/${name}`.toLowerCase()))
    .map((name) => `missing _internal/torch/lib/${name}`);
  for (const file of normalized) {
    const name = path.posix.basename(file);
    if (!CUDA_RUNTIME_DLL.test(name)) continue;
    const directory = path.posix.dirname(file).toLowerCase();
    if (!CUDA_DLL_DIRECTORIES.some((allowed) => directory === allowed.toLowerCase())) {
      problems.push(`CUDA DLL outside the pinned wheels: ${file}`);
    }
  }
  if (variant === "cpu" && lower.has("_internal/torch/lib/torch_cuda.dll")) {
    problems.push("the CPU bundle contains torch_cuda.dll");
  }
  return problems;
}

function checkInstallerSize(bytes, limit = INSTALLER_SIZE_LIMIT) {
  return bytes < limit
    ? []
    : [`the installer is ${bytes} bytes, at or above the ${limit}-byte (1.95 GiB) limit`];
}

function listFiles(root) {
  return fs.readdirSync(root, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => path.relative(root, path.join(entry.parentPath, entry.name)));
}

function nvidiaSmiAvailable() {
  // Presence, not success: a broken driver on a GPU machine must fail loudly.
  return !spawnSync("nvidia-smi", ["-L"], { stdio: "ignore", windowsHide: true }).error;
}

function fail(problems, context) {
  if (problems.length === 0) return;
  console.error(`${context}:\n  - ${problems.join("\n  - ")}`);
  process.exit(1);
}

function main([command, ...args]) {
  const variant = resolveVariant(process.env[VARIANT_VARIABLE]);
  switch (command) {
    case "variant":
      console.log(variant);
      return;
    case "artifact-name":
      console.log(artifactName(manifest.version, variant));
      return;
    case "bundle": {
      fail(checkBundle(listFiles(args[0]), variant), `The ${variant} backend bundle failed validation`);
      console.log(`The ${variant} backend bundle contains the expected PyTorch runtime DLLs.`);
      return;
    }
    case "self-check": {
      const [executable, ...executableArgs] = args;
      const requireCuda = variant === "cuda" && nvidiaSmiAvailable();
      const checkArgs = [...executableArgs, "--self-check", "--variant", variant, ...(requireCuda ? ["--require-cuda"] : [])];
      const result = spawnSync(executable, checkArgs, { encoding: "utf8", windowsHide: true, timeout: 600_000 });
      process.stderr.write(result.stderr ?? "");
      if (result.error) fail([result.error.message], "The backend self-check could not run");
      const report = parseSelfCheck(result.stdout);
      console.log(JSON.stringify(report));
      const problems = checkSelfCheck(report, variant, { requireCuda });
      if (result.status !== 0 && problems.length === 0) problems.push(`exit code ${result.status}`);
      fail(problems, "The backend self-check failed");
      console.log(`Backend self-check passed (${variant}${requireCuda ? ", CUDA kernels verified" : ""}).`);
      return;
    }
    default:
      throw new Error(`Unknown command "${command}". Use variant, artifact-name, bundle <dir>, or self-check <exe> [args].`);
  }
}

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

module.exports = {
  DEFAULT_VARIANT,
  INSTALLER_SIZE_LIMIT,
  VARIANT_VARIABLE,
  VARIANTS,
  artifactName,
  checkBundle,
  checkInstallerSize,
  checkSelfCheck,
  listFiles,
  parseSelfCheck,
  resolveVariant,
};
