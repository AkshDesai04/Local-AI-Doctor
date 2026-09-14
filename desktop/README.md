# Desktop packaging

The Electron shell owns the two loopback services used by the application:

- the packaged Python/FastAPI backend listens on `127.0.0.1:6767`;
- the Electron-owned static server and reverse proxy listens on `127.0.0.1:6969`.

Startup is deliberately ordered. Electron immediately shows a lightweight
native startup window, starts the backend, waits for `/api/v1/health` to return
`status: ok`, `database: ready`, and `worker: ready`, starts the frontend server,
and then navigates that window to the application. The health endpoint is only
served after database initialization, worker startup, model discovery, and run
recovery complete. Worker readiness requires an IPC handshake from the spawned
inference process, so no fixed delay is needed. Closing the app creates a private
per-launch shutdown marker so Uvicorn can run its normal lifespan cleanup
before a forced termination is considered.

## Development

Build the web client once, install the desktop packages, and launch Electron:

```powershell
npm --prefix frontend run build
npm --prefix desktop install
npm --prefix desktop start
```

The development shell uses `.venv\Scripts\python.exe` when present. Set
`LAD_DESKTOP_PYTHON` to choose another Python interpreter.

## Windows installer executable

The packaging environment needs Python 3.12 with the project, `ml`, and
PyInstaller dependencies installed. GitHub uses the pinned CPU-only PyTorch
wheels so its single asset remains below GitHub's 2 GiB limit. Local builds use
the PyTorch flavor installed in the selected environment. Set `PYTHON` to that
environment's interpreter when it is not the repository `.venv`; the build
refuses to package an installed project version that differs from `VERSION`.
Then run:

```powershell
$env:PYTHON = "C:\path\to\python-environment\Scripts\python.exe"
npm --prefix desktop ci
npm --prefix desktop run dist
```

The only distributable artifact is
`release\Local-AI-Doctor-<version>.exe`. It is an Electron NSIS installer. Run
it once, complete the per-user installation, and use the desktop or Start menu
shortcut afterward. The installed application reuses its on-disk Electron and
PyInstaller payload instead of extracting several gigabytes on every launch.
A small native startup window appears while the backend starts; the production
frontend still starts only after the complete backend readiness contract passes.
Models are intentionally not bundled; the model-directory setting
remains user configurable and persistent.

The executable is currently unsigned and can trigger Windows SmartScreen. The
desktop shell asks the packaged backend to select the best usable local device:
a package built from a CUDA-enabled PyTorch environment can therefore use CUDA,
while a CPU-only package selects CPU and reports CUDA as unavailable. The
GitHub workflow deliberately builds with CPU PyTorch because the complete
CUDA-enabled runtime is larger than GitHub's 2 GiB per-asset limit. To create a
local CUDA-capable EXE, point `PYTHON` at the pinned CUDA environment before
running `npm --prefix desktop run dist`.
