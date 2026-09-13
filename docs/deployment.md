# WSL2 container deployment

Docker is intentionally invoked only inside WSL2. The checked-in PowerShell wrapper discovers a
WSL2 distribution when there is one unambiguous choice, resolves a Linux user, verifies Docker
Engine connectivity, and then invokes Linux `docker compose`. It never calls Windows `docker.exe`,
uses `sudo`, or accepts a password.

The CPU and NVIDIA backend images share the same application build, while a separate minimal image
serves the compiled frontend. The CPU target installs a PyTorch CPU wheel and verifies that no NVIDIA
Python packages entered the final image. The NVIDIA target installs the CUDA PyTorch and Torchvision
wheels pinned by `requirements-cuda.lock`; the host driver is provided by WSL GPU passthrough and
NVIDIA Container Toolkit. All final images run as UID/GID `10001`, drop all Linux
capabilities, use a read-only root filesystem, and receive `SIGTERM` with a bounded graceful-shutdown
period.

## Local configuration

From PowerShell in the repository root:

```powershell
Copy-Item .env.example .env
```

Edit the ignored `.env` and set at least:

```dotenv
MODEL_PATH=/wsl/path/to/models
APP_CONFIG_PATH=./config/local.example.yaml
WSL_DISTRIBUTION=your-wsl2-distribution
WSL_USER=your-linux-user
```

`MODEL_PATH` must be the Linux path visible inside the selected distribution, not a Windows path.
The wrapper parameters `-Distribution` and `-WslUser` take precedence over these values. If
`WSL_DISTRIBUTION` is empty, the wrapper selects the default WSL2 distribution, or the only WSL2
distribution if there is exactly one. If `WSL_USER` is empty, WSL's configured default user is used.

The local configuration file is bind-mounted read-only at `/app/config/local.yaml`. Portable
defaults are baked into the image at `/app/config/default.yaml`. Runtime precedence remains:
portable defaults, selected container profile, the mounted local file, then explicit `LAD_`
environment overrides.

Do not put passwords, tokens, or private keys in `.env`, Compose build arguments, or configuration
files. The local model directory is mounted read-only at `/models`; startup fails if that mount is
writable.

## Prerequisites and verification

Inspect WSL and validate Linux-side Docker access:

```powershell
wsl.exe --list --verbose
$Distro = "your-wsl2-distribution"
$WslUser = "your-linux-user"
wsl.exe --distribution $Distro --user $WslUser -- uname -r
wsl.exe --distribution $Distro --user $WslUser -- docker info
wsl.exe --distribution $Distro --user $WslUser -- docker compose version
```

The kernel string should identify WSL2 and `docker info` must show a reachable server. Docker Engine
inside the selected distribution is preferred. Docker Desktop's WSL backend also works, but Docker
Desktop must be running and integration must be enabled for that distribution. If the Docker socket
is not available to the selected user, stop here and have the WSL Docker installation or user access
provisioned interactively; do not weaken socket permissions or pass a password through a command.

The wrapper performs these checks without requiring Compose configuration:

```powershell
.\scripts\wsl-docker.ps1 -Action Check -Distribution $Distro -WslUser $WslUser
```

Validate interpolated Compose configuration before building:

```powershell
.\scripts\wsl-docker.ps1 -Action ConfigCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action ConfigNvidia -Distribution $Distro -WslUser $WslUser
```

## CPU build and startup

```powershell
.\scripts\wsl-docker.ps1 -Action BuildCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action UpCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action HealthCpu -Distribution $Distro -WslUser $WslUser
```

The `UpCpu` and `UpNvidia` wrapper actions also start one named, idle Linux process so WSL keeps the
distribution resident after PowerShell exits. This is necessary because WSL does not treat systemd
services alone as an active session; without it, the Docker daemon and published localhost ports can
disappear as soon as the launcher finishes. The process is reused on later starts and ends when the
distribution is shut down.

The equivalent raw commands, when PowerShell is already in the repository root, are:

```powershell
wsl.exe --distribution $Distro --user $WslUser -- docker compose --profile cpu build --pull app-cpu frontend-cpu
wsl.exe --distribution $Distro --user $WslUser -- docker compose --profile cpu up --detach --wait frontend-cpu
wsl.exe --distribution $Distro --user $WslUser -- docker compose --profile cpu exec -T app-cpu python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:6767/api/v1/health').read().decode())"
wsl.exe --distribution $Distro --user $WslUser -- docker compose --profile cpu exec -T frontend-cpu wget --quiet --output-document=- http://127.0.0.1:6969/healthz
```

Open the frontend at `http://127.0.0.1:6969`. The REST API and WebSocket backend are fixed at
`http://127.0.0.1:6767`; its health endpoint is `http://127.0.0.1:6767/api/v1/health`. Compose does not
start the frontend container until the backend is healthy. On every frontend container start, including
Docker daemon restarts, its entrypoint independently waits for a healthy backend and then holds for 16
seconds before starting Nginx on port 6969. The frontend proxy carries same-origin API and WebSocket
traffic to the private backend service, which remains directly available on port 6767 for local
diagnostics and API clients. External access requires a separately reviewed, authenticated configuration
and is not enabled by this deployment.

## NVIDIA build and startup

Before startup, confirm WSL can see the GPU and that Docker's NVIDIA runtime works:

```powershell
.\scripts\wsl-docker.ps1 -Action CheckNvidia -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action BuildNvidia -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action UpNvidia -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action NvidiaSmoke -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action HealthNvidia -Distribution $Distro -WslUser $WslUser
```

The NVIDIA profile is structurally valid without a GPU, but startup requires all of:

- a Windows NVIDIA driver supporting WSL GPU passthrough;
- `nvidia-smi` working inside the selected WSL2 distribution; and
- NVIDIA Container Toolkit configured for that distribution's Docker daemon.

`NvidiaSmoke` verifies `torch.cuda.is_available()` and prints the selected device. It does not load a
model. GPU selection can be narrowed with `NVIDIA_VISIBLE_DEVICES` in the ignored `.env`.

Do not run the CPU and NVIDIA service pairs simultaneously because both profiles reserve the permanent
host ports 6969 and 6767. Stop one profile before starting the other.

## Development loop

The container development loop deliberately uses the production build graph instead of a privileged
container or a second dependency set. After source changes, rebuild and replace the CPU service:

```powershell
.\scripts\wsl-docker.ps1 -Action ConfigCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action UpCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action LogsCpu -Distribution $Distro -WslUser $WslUser
```

`UpCpu` includes `--build`; BuildKit reuses unchanged dependency and frontend layers. Native Vite and
Python hot-reload remain the faster choice for UI-only or backend-only iteration.

The repository and model directory are on mounted Windows filesystems on many WSL installations.
Build context traversal and large numbers of small file reads are slower there than in WSL's native
filesystem. Large sequential model reads are usually less affected. The `.dockerignore` keeps runtime
state, model weights, tests, documentation, and dependency caches out of the build context. Do not
relocate the repository solely for performance without coordinating that change.

## Operations

Inspect status and bounded logs:

```powershell
.\scripts\wsl-docker.ps1 -Action Ps -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action LogsCpu -Distribution $Distro -WslUser $WslUser
# Use LogsNvidia when that profile is active.
```

Compose uses rotating `json-file` logs (`10 MiB`, three files). Normal application logging redacts
private paths and does not include prompts, uploaded content, generated tokens, or credentials.

Verify model mount flags from the running service:

```powershell
wsl.exe --distribution $Distro --user $WslUser -- docker compose --profile cpu exec -T app-cpu python -c "import os; print(bool(os.statvfs('/models').f_flag & os.ST_RDONLY))"
```

The result must be `True`. The application never writes to `/models`.

Gracefully stop one service, or remove both stacks while preserving all named volumes:

```powershell
.\scripts\wsl-docker.ps1 -Action StopCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action StopNvidia -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action Down -Distribution $Distro -WslUser $WslUser
```

`stop` and `down` send `SIGTERM`, wait for `STOP_GRACE_PERIOD`, and then stop the container. Do not add
`--volumes` unless permanent deletion of all application state is explicitly intended.

## Database backup and restore

Create an online, integrity-checked SQLite backup in the persistent backups volume while the app is
running:

```powershell
.\scripts\wsl-docker.ps1 -Action Backup -Distribution $Distro -WslUser $WslUser
```

The command prints a volume-internal filename such as
`/data/backups/workbench-YYYYMMDDTHHMMSSZ.sqlite3`. Copy or export that file separately if the Docker
volume itself is at risk.

Restore is intentionally offline and replaces the current database. Stop the app, pass only the
printed filename, then restart and check health:

```powershell
.\scripts\wsl-docker.ps1 -Action StopCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action Restore -BackupFile workbench-YYYYMMDDTHHMMSSZ.sqlite3 -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action UpCpu -Distribution $Distro -WslUser $WslUser
.\scripts\wsl-docker.ps1 -Action HealthCpu -Distribution $Distro -WslUser $WslUser
```

Use `StopNvidia`/`UpNvidia` instead when that profile owns the database. The restore utility rejects
paths outside `/data/backups`, checks the source database, restores through a temporary database, and
atomically replaces the stopped database.

## Updating without data loss

1. Create and retain a database backup.
2. Stop the active service gracefully.
3. Update the working tree through the project's normal Git workflow.
4. Re-run `BuildCpu` or `BuildNvidia` and then the matching `Up` action.
5. Verify health and inspect logs.

Named volumes for SQLite, uploads, caches, exports, and backups are external to the container lifecycle
and have stable configurable names. Rebuilding or replacing a container does not remove them. Changing
`COMPOSE_PROJECT_NAME` is safe because explicit volume names are used; changing a `*_VOLUME_NAME`
selects different state and should be treated as a migration.

## Resource tuning

`CONTAINER_CPUS`, `CONTAINER_MEMORY_LIMIT`, `CONTAINER_MEMORY_RESERVATION`,
`CONTAINER_PIDS_LIMIT`, `CPU_THREADS`, `MODEL_WORKERS`, and `TMPFS_SIZE` are deployment controls in the
ignored `.env`. Application-level RAM/VRAM budgets and inference behavior belong in the mounted typed
configuration. Keep one Uvicorn process unless the isolated worker design and SQLite write behavior
have been validated with a higher value; additional web workers do not make a single loaded model
faster.
