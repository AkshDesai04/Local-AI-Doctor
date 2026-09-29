"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {
  isBackendReady,
  waitForBackend,
} = require("../lib/lifecycle.cjs");

test("backend readiness requires the API, database, and model worker", () => {
  assert.equal(isBackendReady({ status: "ok", database: "ready", worker: "ready" }), true);
  assert.equal(isBackendReady({ status: "ok", database: "ready", worker: "unavailable" }), false);
  assert.equal(isBackendReady({ status: "ok", worker: "ready" }), false);
});

test("the shell leaves runtime.device to the profile default and the user's config", () => {
  // An environment override would outrank a device chosen in local.yaml.
  const main = fs.readFileSync(path.join(__dirname, "..", "main.cjs"), "utf8");
  assert.doesNotMatch(main, /LAD_RUNTIME__DEVICE/);
  const defaults = fs.readFileSync(path.join(__dirname, "..", "..", "config", "default.yaml"), "utf8");
  assert.match(defaults, /^ {2}native-windows:\n {4}runtime:\n {6}device: auto$/m);
});

test("waitForBackend retries until the backend reports healthy", async () => {
  let attempts = 0;
  let sleeps = 0;
  const health = await waitForBackend({
    timeoutMs: 2_000,
    intervalMs: 25,
    healthCheck: async () => {
      attempts += 1;
      if (attempts < 3) throw new Error("not ready");
      return { status: "ok", database: "ready", worker: "ready" };
    },
    sleep: async (milliseconds) => {
      sleeps += milliseconds;
    },
  });
  assert.deepEqual(health, { status: "ok", database: "ready", worker: "ready" });
  assert.equal(attempts, 3);
  assert.equal(sleeps, 50);
});

test("waitForBackend fails immediately when the managed process exits", async () => {
  await assert.rejects(
    waitForBackend({ backendExited: () => true }),
    /exited before becoming healthy/,
  );
});
