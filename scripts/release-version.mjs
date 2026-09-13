import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const versionFile = resolve(repositoryRoot, "VERSION");
const semverPattern = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;

function readVersion() {
  const version = readFileSync(versionFile, "utf8").trim();
  if (!semverPattern.test(version)) {
    throw new Error(`VERSION must contain a stable X.Y.Z version; received ${JSON.stringify(version)}`);
  }
  return version;
}

function manifestVersions() {
  const pyprojectPath = resolve(repositoryRoot, "pyproject.toml");
  const pyproject = readFileSync(pyprojectPath, "utf8");
  const projectSection = pyproject.match(/\[project\][\s\S]*?(?=\n\[|$)/)?.[0];
  const pythonVersion = projectSection?.match(/^version\s*=\s*"([^"]+)"\s*$/m)?.[1];
  if (!pythonVersion) {
    throw new Error("Could not find [project].version in pyproject.toml");
  }

  const versions = new Map([["pyproject.toml", pythonVersion]]);
  for (const relativePath of [
    "frontend/package.json",
    "frontend/package-lock.json",
    "desktop/package.json",
    "desktop/package-lock.json",
  ]) {
    const path = resolve(repositoryRoot, relativePath);
    let manifest;
    try {
      manifest = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
      throw new Error(`Could not read ${relativePath}: ${error.message}`);
    }
    versions.set(relativePath, manifest.version);
    if (relativePath.endsWith("package-lock.json") && manifest.packages?.[""]?.version) {
      versions.set(`${relativePath}#packages[\"\"]`, manifest.packages[""].version);
    }
  }
  return versions;
}

function checkVersion() {
  const version = readVersion();
  const mismatches = [...manifestVersions()].filter(([, manifestVersion]) => manifestVersion !== version);
  if (mismatches.length) {
    const details = mismatches
      .map(([path, manifestVersion]) => `${path}=${JSON.stringify(manifestVersion)}`)
      .join(", ");
    throw new Error(`Release version ${version} is out of sync: ${details}`);
  }
  console.log(version);
  return version;
}

function updateJsonVersion(relativePath, version) {
  const path = resolve(repositoryRoot, relativePath);
  const manifest = JSON.parse(readFileSync(path, "utf8"));
  manifest.version = version;
  if (relativePath.endsWith("package-lock.json") && manifest.packages?.[""]) {
    manifest.packages[""].version = version;
  }
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
}

function bumpVersion(level) {
  if (!new Set(["patch", "minor", "major"]).has(level)) {
    throw new Error("Choose exactly one release level: patch, minor, or major");
  }

  const current = readVersion();
  let [major, minor, patch] = current.split(".").map(Number);
  if (level === "major") {
    major += 1;
    minor = 0;
    patch = 0;
  } else if (level === "minor") {
    minor += 1;
    patch = 0;
  } else {
    patch += 1;
  }
  const next = `${major}.${minor}.${patch}`;

  writeFileSync(versionFile, `${next}\n`);

  const pyprojectPath = resolve(repositoryRoot, "pyproject.toml");
  const pyproject = readFileSync(pyprojectPath, "utf8");
  const nextPyproject = pyproject.replace(
    /(\[project\][\s\S]*?^version\s*=\s*")[^"]+("\s*$)/m,
    (_match, prefix, suffix) => `${prefix}${next}${suffix}`,
  );
  if (nextPyproject === pyproject) {
    throw new Error("Could not update [project].version in pyproject.toml");
  }
  writeFileSync(pyprojectPath, nextPyproject);

  for (const relativePath of [
    "frontend/package.json",
    "frontend/package-lock.json",
    "desktop/package.json",
    "desktop/package-lock.json",
  ]) {
    updateJsonVersion(relativePath, next);
  }

  checkVersion();
  console.error(`Bumped the project version from ${current} to ${next} (${level}).`);
}

const [command, ...extraArguments] = process.argv.slice(2);
if (extraArguments.length) {
  throw new Error("Unexpected extra arguments");
}

if (command === "current") {
  console.log(readVersion());
} else if (command === "check") {
  checkVersion();
} else if (["patch", "minor", "major"].includes(command)) {
  bumpVersion(command);
} else {
  throw new Error(
    "Usage: node scripts/release-version.mjs <current|check|patch|minor|major>",
  );
}
