"use strict";

const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn, spawnSync } = require("node:child_process");
const { randomUUID } = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const { createFrontendServer } = require("./lib/frontend-server.cjs");
const {
  BACKEND_HOST,
  BACKEND_PORT,
  FRONTEND_HOST,
  FRONTEND_PORT,
  isPortAvailable,
  waitForBackend,
} = require("./lib/lifecycle.cjs");

let backendProcess;
let backendExited = false;
let backendSpawnFailed = false;
let backendShutdownFile;
let frontendServer;
let mainWindow;
let shutdownComplete = false;
let shutdownStarted;

function writeStartupTrace(paths, launchId, event) {
  if (process.env.LAD_DESKTOP_SMOKE_TRACE !== "1") return;
  const runtimeDirectory = ensureDirectory(path.join(paths.userData, "runtime"));
  const tracePath = path.join(runtimeDirectory, "startup-trace.jsonl");
  fs.appendFileSync(
    tracePath,
    `${JSON.stringify({ launchId, event, monotonicMs: Number(process.hrtime.bigint() / 1_000_000n) })}\n`,
    "utf8",
  );
}

function clearBackendShutdownFile() {
  const shutdownFile = backendShutdownFile;
  backendShutdownFile = undefined;
  if (!shutdownFile) return;
  try {
    fs.rmSync(shutdownFile, { force: true });
  } catch {
    // A stale, randomly named marker is harmless because paths are never reused.
  }
}

function locations() {
  const repositoryRoot = path.resolve(__dirname, "..");
  const resourcesRoot = app.isPackaged ? process.resourcesPath : repositoryRoot;
  const userData = app.getPath("userData");
  return {
    repositoryRoot,
    resourcesRoot,
    userData,
    frontendRoot: app.isPackaged ? path.join(resourcesRoot, "frontend") : path.join(repositoryRoot, "frontend", "dist"),
    defaultConfig: app.isPackaged ? path.join(resourcesRoot, "config", "default.yaml") : path.join(repositoryRoot, "config", "default.yaml"),
    userConfig: path.join(userData, "config", "local.yaml"),
    backendExecutable: path.join(resourcesRoot, "backend", "local-ai-doctor-backend", "local-ai-doctor-backend.exe"),
  };
}

function ensureDirectory(target) {
  fs.mkdirSync(target, { recursive: true });
  return target;
}

function ensureUserConfig(target) {
  ensureDirectory(path.dirname(target));
  if (fs.existsSync(target)) return;
  try {
    fs.writeFileSync(target, "schema_version: 1\n", { encoding: "utf8", flag: "wx" });
  } catch (error) {
    if (!error || error.code !== "EEXIST") throw error;
  }
}

function backendEnvironment(paths, shutdownFile) {
  const data = ensureDirectory(path.join(paths.userData, "data"));
  const cache = ensureDirectory(path.join(paths.userData, "cache"));
  ensureUserConfig(paths.userConfig);
  const environment = { ...process.env };
  for (const name of Object.keys(environment)) {
    if (name.startsWith("LAD_DESKTOP_")) delete environment[name];
  }
  return {
    ...environment,
    LAD_CONFIG: paths.defaultConfig,
    LAD_USER_CONFIG: paths.userConfig,
    LAD_PROFILE: "native-windows",
    LAD_DESKTOP_SHUTDOWN_FILE: shutdownFile,
    // runtime.device is deliberately not set here: the native-windows profile
    // defaults to auto, and a device chosen in the user's local.yaml must win.
    LAD_SERVER__HOST: BACKEND_HOST,
    LAD_SERVER__PORT: String(BACKEND_PORT),
    LAD_SERVER__ALLOWED_ORIGINS: JSON.stringify([
      `http://${FRONTEND_HOST}:${FRONTEND_PORT}`,
      `http://localhost:${FRONTEND_PORT}`,
    ]),
    LAD_PATHS__DATABASE: path.join(data, "workbench.sqlite3"),
    LAD_PATHS__UPLOADS: ensureDirectory(path.join(data, "uploads")),
    LAD_PATHS__CACHE: cache,
    LAD_PATHS__EXPORTS: ensureDirectory(path.join(data, "exports")),
    LAD_PATHS__BACKUPS: ensureDirectory(path.join(data, "backups")),
    HF_HOME: ensureDirectory(path.join(cache, "huggingface")),
    TORCH_HOME: ensureDirectory(path.join(cache, "torch")),
    PYTHONUTF8: "1",
    PYTHONUNBUFFERED: "1",
  };
}

function developmentBackend(paths) {
  const configured = process.env.LAD_DESKTOP_PYTHON;
  const workspacePython = path.join(paths.repositoryRoot, ".venv", "Scripts", "python.exe");
  const executable = configured || (fs.existsSync(workspacePython) ? workspacePython : "python");
  return {
    executable,
    arguments: [path.join(__dirname, "backend_launcher.py")],
    cwd: paths.repositoryRoot,
  };
}

function packagedBackend(paths) {
  if (!fs.existsSync(paths.backendExecutable)) {
    throw new Error(`The packaged backend is missing at ${paths.backendExecutable}.`);
  }
  return { executable: paths.backendExecutable, arguments: [], cwd: paths.resourcesRoot };
}

function pipeBackendLogs(child, userData) {
  const logDirectory = ensureDirectory(path.join(userData, "logs"));
  const log = fs.createWriteStream(path.join(logDirectory, "backend.log"), { flags: "a" });
  child.stdout.pipe(log, { end: false });
  child.stderr.pipe(log, { end: false });
  child.once("close", () => log.end());
}

function startBackend(paths) {
  const command = app.isPackaged ? packagedBackend(paths) : developmentBackend(paths);
  const runtimeDirectory = ensureDirectory(path.join(paths.userData, "runtime"));
  backendShutdownFile = path.join(runtimeDirectory, `shutdown-${process.pid}-${Date.now()}-${randomUUID()}.signal`);
  fs.rmSync(backendShutdownFile, { force: true });
  backendExited = false;
  backendSpawnFailed = false;
  backendProcess = spawn(command.executable, command.arguments, {
    cwd: command.cwd,
    env: backendEnvironment(paths, backendShutdownFile),
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const child = backendProcess;
  pipeBackendLogs(child, paths.userData);
  child.once("exit", () => {
    backendExited = true;
    clearBackendShutdownFile();
  });
  child.once("error", () => {
    // A failed spawn has no PID and will not reliably emit `exit`. Other
    // ChildProcess errors do not prove that an already-running process ended.
    if (!child.pid) {
      backendSpawnFailed = true;
      backendExited = true;
      clearBackendShutdownFile();
    }
  });
  child.once("close", () => {
    backendExited = true;
    clearBackendShutdownFile();
  });
}

function waitForExit(child, timeoutMs, exited = () => false) {
  if (!child || child.exitCode !== null || child.signalCode !== null || exited()) {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      clearInterval(poll);
      child.removeListener("exit", completed);
      child.removeListener("close", completed);
      child.removeListener("error", spawnFailed);
      resolve(result);
    };
    const completed = () => finish(true);
    const spawnFailed = () => {
      if (!child.pid) finish(true);
    };
    const timeout = setTimeout(() => finish(false), timeoutMs);
    const poll = setInterval(() => {
      if (child.exitCode !== null || child.signalCode !== null || exited()) finish(true);
    }, 100);
    child.once("exit", completed);
    child.once("close", completed);
    child.once("error", spawnFailed);
  });
}

async function stopBackend() {
  const child = backendProcess;
  backendProcess = undefined;
  if (
    !child
    || child.exitCode !== null
    || child.signalCode !== null
    || backendSpawnFailed
  ) {
    clearBackendShutdownFile();
    return;
  }
  let shutdownSignaled = false;
  if (backendShutdownFile) {
    try {
      fs.writeFileSync(backendShutdownFile, "shutdown\n", { encoding: "utf8", flag: "wx" });
      shutdownSignaled = true;
    } catch (error) {
      shutdownSignaled = Boolean(error && error.code === "EEXIST");
    }
  }
  if (shutdownSignaled && await waitForExit(child, 20_000, () => backendExited)) {
    clearBackendShutdownFile();
    return;
  }
  if (process.platform === "win32" && child.pid) {
    spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { windowsHide: true });
  } else {
    child.kill("SIGTERM");
    if (!await waitForExit(child, 3_000, () => backendExited)) child.kill("SIGKILL");
  }
  await waitForExit(child, 5_000, () => backendExited);
  clearBackendShutdownFile();
}

async function shutdown() {
  if (shutdownStarted) return shutdownStarted;
  shutdownStarted = (async () => {
    try {
      if (frontendServer) {
        await frontendServer.close();
        frontendServer = undefined;
      }
    } finally {
      try {
        await stopBackend();
      } finally {
        shutdownComplete = true;
      }
    }
  })();
  return shutdownStarted;
}

async function createWindow() {
  const startupUrl = pathToFileURL(path.join(__dirname, "startup.html")).href;
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 980,
    minWidth: 1024,
    minHeight: 700,
    backgroundColor: "#0b0f14",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://") || url.startsWith("http://")) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (
      url !== startupUrl
      && !url.startsWith(`http://${FRONTEND_HOST}:${FRONTEND_PORT}/`)
    ) {
      event.preventDefault();
    }
  });
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = undefined;
  });
  await mainWindow.loadFile(path.join(__dirname, "startup.html"));
}

async function showApplication() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    throw new Error("The startup window closed before the application became ready.");
  }
  await mainWindow.loadURL(`http://${FRONTEND_HOST}:${FRONTEND_PORT}/`);
}

function setStartupStatus(message) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const source = `document.getElementById("startup-status").textContent = ${JSON.stringify(message)}`;
  void mainWindow.webContents.executeJavaScript(source, true).catch(() => {});
}

async function startApplication() {
  const launchId = randomUUID();
  await createWindow();
  const paths = locations();
  const availability = await Promise.all([
    isPortAvailable(BACKEND_PORT, BACKEND_HOST),
    isPortAvailable(FRONTEND_PORT, FRONTEND_HOST),
  ]);
  if (!availability[0] || !availability[1]) {
    const occupied = [!availability[0] ? BACKEND_PORT : null, !availability[1] ? FRONTEND_PORT : null].filter(Boolean).join(", ");
    throw new Error(`Required local port${occupied.includes(",") ? "s are" : " is"} already in use: ${occupied}. Stop the process or container using the port and try again.`);
  }
  if (!fs.existsSync(paths.frontendRoot)) {
    throw new Error(`The frontend bundle is missing at ${paths.frontendRoot}. Run the frontend build first.`);
  }
  startBackend(paths);
  await waitForBackend({ backendExited: () => backendExited });
  writeStartupTrace(paths, launchId, "backend_healthy");
  setStartupStatus("The local backend is fully ready. Starting the interface…");
  if (backendExited) throw new Error("The backend exited after reporting ready.");
  writeStartupTrace(paths, launchId, "frontend_start_requested");
  frontendServer = createFrontendServer({
    frontendRoot: paths.frontendRoot,
    frontendHost: FRONTEND_HOST,
    frontendPort: FRONTEND_PORT,
    backendHost: BACKEND_HOST,
    backendPort: BACKEND_PORT,
  });
  await frontendServer.start();
  writeStartupTrace(paths, launchId, "frontend_listening");
  await showApplication();
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
  app.whenReady().then(startApplication).catch(async (error) => {
    try {
      await shutdown();
    } catch {
      // The original startup failure is the useful error to present.
    }
    dialog.showErrorBox("Local AI Doctor could not start", error instanceof Error ? error.message : String(error));
    app.quit();
  });
}

app.on("window-all-closed", () => app.quit());
app.on("before-quit", (event) => {
  if (shutdownComplete) return;
  event.preventDefault();
  void shutdown().catch(() => {}).then(() => app.quit());
});
