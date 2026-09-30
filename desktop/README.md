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

Packaging uses a dedicated Python 3.12 environment, `.venv-desktop` at the
repository root, never the development `.venv` or a `python` found on `PATH`.
Create or refresh it once (it downloads roughly 3 GB of CUDA wheels the first
time; later runs only reinstall what changed):

```powershell
.\desktop\scripts\prepare-build-env.ps1
```

It installs `requirements.lock`, `requirements-ml.lock`, and
`requirements-desktop.lock`, then the exact CUDA wheels from
`requirements-cuda.lock` (so anything added there, such as bitsandbytes, comes
along), then the current project, and fails unless PyTorch reports `+cu128`
with CUDA 12.8. Then build:

```powershell
npm --prefix desktop ci
npm --prefix desktop run dist
```

The only distributable artifact is
`release\Local-AI-Doctor-<version>-cuda.exe`, where `<version>` comes from the
root `VERSION` file. The build fails instead of producing it when:

- the packaging environment's PyTorch is not the requested variant, or the
  installed project version differs from `VERSION`;
- the bundle lacks `torch_cuda.dll`, `cudart64_12.dll`, `cublas64_12.dll`,
  `cublasLt64_12.dll`, or `cudnn64_9.dll`, or contains a CUDA runtime DLL
  outside the pinned wheels (`torch\lib`, `torchvision`, `bitsandbytes`).
  CUDA Toolkit directories are removed from `PATH` while packaging, so a local
  toolkit cannot leak into the bundle;
- the packaged backend's `--self-check` fails. On a machine with `nvidia-smi`
  it also runs a CUDA matmul, a bfloat16 attention call, and an NVRTC-compiled
  kernel and compares them with CPU results;
- the installer is 1.95 GiB (2,093,796,556 bytes) or larger, which keeps room
  under GitHub's 2 GiB release-asset limit and the NSIS payload limit.

The packaged backend can be checked directly without starting the server:

```powershell
.\desktopuildackend\local-ai-doctor-backend\local-ai-doctor-backend.exe --self-check --variant cuda --require-cuda
```

`$env:PYTHON` selects a different packaging interpreter; it must still contain
the requested variant.

### CPU-only variant

A CPU-only installer is only built on request:

```powershell
.\desktop\scripts\prepare-build-env.ps1 -Variant cpu
$env:LAD_DESKTOP_TORCH_VARIANT = "cpu"
npm --prefix desktop run dist
Remove-Item Env:LAD_DESKTOP_TORCH_VARIANT
```

It produces `release\Local-AI-Doctor-<version>-cpu.exe`. Run
`prepare-build-env.ps1` without `-Variant` again before the next CUDA build.

### Smoke tests

```powershell
.\desktop\scripts\smoke-backend.ps1
.\desktop\scripts\smoke-desktop.ps1
```

Both honor `LAD_DESKTOP_TORCH_VARIANT`. On a machine with `nvidia-smi`, the
CUDA variant must report a usable CUDA device from `/api/v1/hardware` and
select it. The desktop smoke needs ports 6767 and 6969 free, so stop the Docker
stack first.

### Installing and running

Run the installer once, complete the per-user installation, and use the
desktop or Start menu shortcut afterward. The installed application reuses its
on-disk Electron and PyInstaller payload instead of extracting several
gigabytes on every launch. A small native startup window appears while the
backend starts; the production frontend still starts only after the complete
backend readiness contract passes. Models are intentionally not bundled; the
model-directory setting remains user configurable and persistent.

The executable is currently unsigned and can trigger Windows SmartScreen. The
desktop shell does not override `runtime.device`: the `native-windows` profile
defaults to `auto`, which selects a usable CUDA device and otherwise CPU, and a
device set in the desktop `local.yaml` takes precedence. GitHub releases build
the same CUDA variant; the runner has no GPU, so there the self-check verifies
the `+cu128` build and DLL set without running kernels.
