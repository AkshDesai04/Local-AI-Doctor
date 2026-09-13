"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const { createFrontendServer, safeAssetPath, validFrontendHost } = require("../lib/frontend-server.cjs");

function availablePort() {
  return new Promise((resolve, reject) => {
    const server = http.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

function request(port, requestPath, host = `127.0.0.1:${port}`) {
  return new Promise((resolve, reject) => {
    http.get({ host: "127.0.0.1", port, path: requestPath, headers: { Host: host } }, (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => resolve({
        status: response.statusCode,
        headers: response.headers,
        body: Buffer.concat(chunks).toString("utf8"),
      }));
    }).once("error", reject);
  });
}

test("the frontend server exposes health and SPA fallback on loopback only", async (context) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lad-desktop-"));
  fs.writeFileSync(path.join(root, "index.html"), "desktop shell", "utf8");
  const port = await availablePort();
  const frontend = createFrontendServer({ frontendRoot: root, frontendPort: port });
  await frontend.start();
  context.after(async () => {
    await frontend.close();
    fs.rmSync(root, { recursive: true, force: true });
  });

  const health = await request(port, "/healthz");
  assert.equal(health.status, 200);
  assert.equal(JSON.parse(health.body).status, "ok");

  const fallback = await request(port, "/chats/example");
  assert.equal(fallback.status, 200);
  assert.equal(fallback.body, "desktop shell");
  assert.equal(fallback.headers["x-content-type-options"], "nosniff");

  const rejected = await request(port, "/", `evil.example:${port}`);
  assert.equal(rejected.status, 400);
});

test("asset path resolution cannot escape the frontend directory", () => {
  const root = path.resolve("frontend", "dist");
  assert.equal(safeAssetPath(root, "/../../secret.txt"), null);
  assert.equal(safeAssetPath(root, "/%2e%2e/%2e%2e/secret.txt"), null);
  assert.equal(validFrontendHost("127.0.0.1:6969", 6969), true);
  assert.equal(validFrontendHost("example.com:6969", 6969), false);
});
