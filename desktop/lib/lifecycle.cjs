"use strict";

const http = require("node:http");
const net = require("node:net");

const BACKEND_HOST = "127.0.0.1";
const BACKEND_PORT = 6767;
const FRONTEND_HOST = "127.0.0.1";
const FRONTEND_PORT = 6969;

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function isPortAvailable(port, host = BACKEND_HOST) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once("error", (error) => {
      if (error && error.code === "EADDRINUSE") {
        resolve(false);
        return;
      }
      reject(error);
    });
    server.listen({ host, port, exclusive: true }, () => {
      server.close((error) => error ? reject(error) : resolve(true));
    });
  });
}

function isBackendReady(payload) {
  return Boolean(
    payload
    && payload.status === "ok"
    && payload.database === "ready"
    && payload.worker === "ready"
  );
}

function readHealth({ host = BACKEND_HOST, port = BACKEND_PORT, timeoutMs = 2_000 } = {}) {
  return new Promise((resolve, reject) => {
    const request = http.get(
      {
        host,
        port,
        path: "/api/v1/health",
        headers: {
          Accept: "application/json",
          Host: `${host}:${port}`,
        },
        timeout: timeoutMs,
      },
      (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => {
          if (response.statusCode !== 200) {
            reject(new Error(`Backend health returned HTTP ${response.statusCode}.`));
            return;
          }
          try {
            const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
            if (!isBackendReady(payload)) {
              reject(new Error(
                `Backend is not ready (status=${String(payload.status ?? "unknown")}, `
                + `database=${String(payload.database ?? "unknown")}, `
                + `worker=${String(payload.worker ?? "unknown")}).`,
              ));
              return;
            }
            resolve(payload);
          } catch (error) {
            reject(new Error(`Backend returned invalid health JSON: ${error.message}`));
          }
        });
      },
    );
    request.once("timeout", () => request.destroy(new Error("Backend health request timed out.")));
    request.once("error", reject);
  });
}

async function waitForBackend({
  timeoutMs = 600_000,
  intervalMs = 500,
  healthCheck = readHealth,
  sleep = delay,
  backendExited = () => false,
} = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    if (backendExited()) {
      throw new Error("The backend exited before becoming healthy.");
    }
    try {
      return await healthCheck();
    } catch (error) {
      lastError = error;
    }
    await sleep(intervalMs);
  }
  throw new Error(`Backend did not become healthy within ${Math.ceil(timeoutMs / 1000)} seconds.${lastError ? ` ${lastError.message}` : ""}`);
}

module.exports = {
  BACKEND_HOST,
  BACKEND_PORT,
  FRONTEND_HOST,
  FRONTEND_PORT,
  delay,
  isBackendReady,
  isPortAvailable,
  readHealth,
  waitForBackend,
};
