"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const MIME_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".gif", "image/gif"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".jpeg", "image/jpeg"],
  [".jpg", "image/jpeg"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"],
]);

const SECURITY_HEADERS = {
  "Cache-Control": "no-store",
  "Content-Security-Policy": "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' ws://127.0.0.1:6969 ws://localhost:6969",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};

function validFrontendHost(value, frontendPort) {
  return value === `127.0.0.1:${frontendPort}` || value === `localhost:${frontendPort}`;
}

function applyHeaders(response) {
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) response.setHeader(name, value);
}

function proxyHttp(request, response, backendHost, backendPort) {
  const headers = { ...request.headers, host: `${backendHost}:${backendPort}` };
  const proxy = http.request(
    {
      host: backendHost,
      port: backendPort,
      method: request.method,
      path: request.url,
      headers,
    },
    (backendResponse) => {
      response.writeHead(backendResponse.statusCode ?? 502, backendResponse.statusMessage, backendResponse.headers);
      backendResponse.pipe(response);
    },
  );
  proxy.on("error", () => {
    if (!response.headersSent) {
      applyHeaders(response);
      response.writeHead(502, { "Content-Type": "application/json; charset=utf-8" });
    }
    response.end(JSON.stringify({ error: { code: "backend_unavailable", message: "The local backend is unavailable." } }));
  });
  request.pipe(proxy);
}

function serializeUpgradeResponse(response) {
  const status = `HTTP/${response.httpVersion} ${response.statusCode} ${response.statusMessage}\r\n`;
  const headers = response.rawHeaders.reduce((result, value, index) => {
    return result + value + (index % 2 === 0 ? ": " : "\r\n");
  }, "");
  return `${status}${headers}\r\n`;
}

function proxyWebSocket(request, clientSocket, clientHead, backendHost, backendPort) {
  let backendSocket;
  const destroyPair = () => {
    if (!clientSocket.destroyed) clientSocket.destroy();
    if (backendSocket && !backendSocket.destroyed) backendSocket.destroy();
  };
  const headers = { ...request.headers, host: `${backendHost}:${backendPort}` };
  const proxy = http.request({
    host: backendHost,
    port: backendPort,
    method: request.method,
    path: request.url,
    headers,
  });
  proxy.on("upgrade", (response, upgradedSocket, backendHead) => {
    backendSocket = upgradedSocket;
    backendSocket.once("error", destroyPair);
    backendSocket.once("close", () => {
      if (!clientSocket.destroyed) clientSocket.destroy();
    });
    clientSocket.write(serializeUpgradeResponse(response));
    if (backendHead.length) clientSocket.write(backendHead);
    if (clientHead.length) backendSocket.write(clientHead);
    backendSocket.pipe(clientSocket);
    clientSocket.pipe(backendSocket);
  });
  proxy.on("response", (response) => {
    clientSocket.write(serializeUpgradeResponse(response));
    response.pipe(clientSocket);
  });
  proxy.on("error", destroyPair);
  clientSocket.once("error", destroyPair);
  clientSocket.once("close", () => {
    proxy.destroy();
    if (backendSocket && !backendSocket.destroyed) backendSocket.destroy();
  });
  proxy.end();
}

function safeAssetPath(frontendRoot, requestPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(requestPath);
  } catch {
    return null;
  }
  const relative = decoded.replace(/^\/+/, "");
  const candidate = path.resolve(frontendRoot, relative);
  const root = path.resolve(frontendRoot);
  return candidate === root || candidate.startsWith(`${root}${path.sep}`) ? candidate : null;
}

function serveFrontend(request, response, frontendRoot) {
  const requestUrl = new URL(request.url, "http://127.0.0.1");
  const asset = safeAssetPath(frontendRoot, requestUrl.pathname);
  const index = path.join(frontendRoot, "index.html");
  let target = asset;
  if (!target || !fs.existsSync(target) || !fs.statSync(target).isFile()) target = index;
  if (!fs.existsSync(target) || !fs.statSync(target).isFile()) {
    applyHeaders(response);
    response.writeHead(503, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("The desktop frontend bundle is missing.");
    return;
  }
  applyHeaders(response);
  response.writeHead(200, { "Content-Type": MIME_TYPES.get(path.extname(target).toLowerCase()) ?? "application/octet-stream" });
  fs.createReadStream(target).pipe(response);
}

function createFrontendServer({
  frontendRoot,
  frontendHost = "127.0.0.1",
  frontendPort = 6969,
  backendHost = "127.0.0.1",
  backendPort = 6767,
}) {
  const sockets = new Set();
  const server = http.createServer((request, response) => {
    if (!validFrontendHost(request.headers.host, frontendPort)) {
      response.writeHead(400, { "Content-Type": "application/json; charset=utf-8" });
      response.end(JSON.stringify({ error: { code: "host_rejected", message: "Request host is not allowed." } }));
      return;
    }
    if (request.url === "/healthz") {
      applyHeaders(response);
      response.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      response.end('{"status":"ok"}');
      return;
    }
    if (request.url.startsWith("/api/")) {
      proxyHttp(request, response, backendHost, backendPort);
      return;
    }
    serveFrontend(request, response, frontendRoot);
  });
  server.on("upgrade", (request, socket, head) => {
    if (!validFrontendHost(request.headers.host, frontendPort) || !request.url.startsWith("/ws/")) {
      socket.destroy();
      return;
    }
    proxyWebSocket(request, socket, head, backendHost, backendPort);
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  });
  return {
    server,
    start: () => new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(frontendPort, frontendHost, () => {
        server.removeListener("error", reject);
        resolve();
      });
    }),
    close: () => new Promise((resolve) => {
      server.close(() => resolve());
      for (const socket of sockets) socket.destroy();
    }),
  };
}

module.exports = { createFrontendServer, safeAssetPath, validFrontendHost };
