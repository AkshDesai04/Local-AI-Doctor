"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const manifest = require("../package.json");

test("the Windows artifact installs persistently instead of extracting on every launch", () => {
  assert.equal(manifest.build.compression, "normal");
  assert.deepEqual(manifest.build.win.target, [{ target: "nsis", arch: ["x64"] }]);
  assert.equal(manifest.build.win.artifactName, "Local-AI-Doctor-${version}.exe");
  assert.equal(manifest.build.nsis.guid, "df0eb923-5a87-57ad-bb13-24a35a6c435a");
  assert.equal(manifest.build.nsis.oneClick, false);
  assert.equal(manifest.build.nsis.perMachine, false);
  assert.equal(manifest.build.nsis.allowElevation, false);
  assert.equal(manifest.build.nsis.allowToChangeInstallationDirectory, true);
  assert.equal(manifest.build.nsis.runAfterFinish, true);
  assert.equal(manifest.build.nsis.createDesktopShortcut, true);
  assert.equal(manifest.build.nsis.createStartMenuShortcut, true);
  assert.equal(manifest.build.nsis.deleteAppDataOnUninstall, false);
  assert.equal(manifest.build.nsis.differentialPackage, false);
  assert.equal(manifest.build.nsis.useZip, false);
  assert.equal(manifest.build.nsis.include, "installer.nsh");
});

test("the immediate startup window is included in the packaged application", () => {
  assert.ok(manifest.build.files.includes("startup.html"));
  const startupPage = fs.readFileSync(path.join(__dirname, "..", "startup.html"), "utf8");
  assert.match(startupPage, /Starting Local AI Doctor/);
});
