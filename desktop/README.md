# Desktop packaging

The Electron shell owns the two loopback services used by the application:

- the packaged Python/FastAPI backend listens on `127.0.0.1:6767`;
- the Electron-owned static server and reverse proxy listens on `127.0.0.1:6969`.

Startup is deliberately ordered. Electron starts the backend, waits for
`/api/v1/health` to return `status: ok`, waits another 16 seconds, starts the
frontend server, and only then creates the browser window. Closing the app
creates a private per-launch shutdown marker so Uvicorn can run its normal
lifespan cleanup before a forced termination is considered.

## Development

Build the web client once, install the desktop packages, and launch Electron:

```powershell
npm --prefix frontend run build
npm --prefix desktop install
npm --prefix desktop start
```

The development shell uses `.venv\Scripts\python.exe` when present. Set
`LAD_DESKTOP_PYTHON` to choose another Python interpreter.

## Portable Windows executable

The packaging environment needs Python 3.12 with the project, `ml`, and
PyInstaller dependencies installed. GitHub and the supported local release
build use the pinned CPU-only PyTorch wheels so the single asset remains below
GitHub's 2 GiB limit. Set `PYTHON` to that environment's interpreter when it is
not the repository `.venv`; the build refuses to package an installed project
version that differs from `VERSION`. Then run:

```powershell
$env:PYTHON = "C:\path\to\cpu-venv\Scripts\python.exe"
npm --prefix desktop ci
npm --prefix desktop run dist
```

The only distributable artifact is
`release\Local-AI-Doctor-<version>.exe`. It is an Electron `portable`
target: the executable extracts its packaged application payload, including the
PyInstaller backend directory, at launch. Models are intentionally not bundled;
the model-directory setting remains user configurable and persistent.

The executable is currently unsigned and can trigger Windows SmartScreen. The
complete CUDA-enabled PyTorch stack is larger than GitHub's 2 GiB per-asset
limit, so CUDA remains available through the native and Docker workflows rather
than the GitHub-built desktop executable. The packaged backend therefore fixes
its runtime device to CPU and skips CUDA initialization during application
startup.
