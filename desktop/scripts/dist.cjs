"use strict";

// Builds the Windows installer for one PyTorch variant (LAD_DESKTOP_TORCH_VARIANT,
// default "cuda") and refuses to leave an oversized or misnamed artifact.

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const manifest = require("../package.json");
const { VARIANT_VARIABLE, artifactName, checkInstallerSize, resolveVariant } = require("./build-guard.cjs");

const desktopRoot = path.resolve(__dirname, "..");
const variant = resolveVariant(process.env[VARIANT_VARIABLE]);
const artifact = artifactName(manifest.version, variant);
const environment = { ...process.env, [VARIANT_VARIABLE]: variant };

// npm and electron-builder are .cmd shims on Windows, so they need a shell.
// Every argument here is a fixed token without spaces.
function run(command, args) {
  const commandLine = [command, ...args].join(" ");
  const result = spawnSync(commandLine, { cwd: desktopRoot, env: environment, stdio: "inherit", shell: true });
  if (result.status !== 0) {
    console.error(`${commandLine} failed with exit code ${result.status}.`);
    process.exit(result.status || 1);
  }
}

function powershell(script, ...args) {
  run("powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join("scripts", script), ...args]);
}

console.log(`Building the ${variant} desktop installer ${artifact}.`);
run("npm", ["run", "build:frontend"]);
powershell("build-backend.ps1");
run("electron-builder", ["--win", "nsis", "--x64", "--publish", "never"]);
powershell("verify-packaged-layout.ps1");
powershell("finalize-release.ps1", "-ArtifactName", artifact);

const size = fs.statSync(path.join(desktopRoot, "..", "release", artifact)).size;
const problems = checkInstallerSize(size);
if (problems.length > 0) {
  console.error(`Refusing to ship ${artifact}: ${problems.join("; ")}.`);
  process.exit(1);
}
console.log(`${artifact}: ${size} bytes (${(size / 1024 ** 3).toFixed(3)} GiB).`);
