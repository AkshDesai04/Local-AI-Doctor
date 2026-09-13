"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { FRONTEND_DELAY_MS, waitForBackend } = require("../lib/lifecycle.cjs");

test("the frontend startup delay remains exactly sixteen seconds", () => {
  assert.equal(FRONTEND_DELAY_MS, 16_000);
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
      return { status: "ok" };
    },
    sleep: async (milliseconds) => {
      sleeps += milliseconds;
    },
  });
  assert.deepEqual(health, { status: "ok" });
  assert.equal(attempts, 3);
  assert.equal(sleeps, 50);
});

test("waitForBackend fails immediately when the managed process exits", async () => {
  await assert.rejects(
    waitForBackend({ backendExited: () => true }),
    /exited before becoming healthy/,
  );
});
