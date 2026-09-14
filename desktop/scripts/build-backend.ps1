[CmdletBinding()]
param(
    [string]$Python = $env:PYTHON
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot

if (-not $Python) {
    $WorkspacePython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $WorkspacePython) { $WorkspacePython } else { "python" }
}

& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Install the desktop build dependencies before packaging."
}

$TorchRuntime = (& $Python -c "import torch; print(str(torch.__version__)+'|'+str(torch.version.cuda or 'cpu')+'|'+str(torch.cuda.is_available()))" 2>$null)
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch cannot be imported in the selected packaging environment."
}
Write-Host "Packaging PyTorch runtime: $(([string]$TorchRuntime).Trim())"

$ExpectedVersion = (Get-Content -LiteralPath (Join-Path $RepositoryRoot "VERSION") -Raw).Trim()
$InstalledVersion = (& $Python -c "import importlib.metadata as metadata; print(metadata.version('local-ai-doctor'))" 2>$null)
if ($LASTEXITCODE -ne 0) {
    throw "Local AI Doctor is not installed in the selected Python environment. Install the current project before packaging."
}
$InstalledVersion = ([string]$InstalledVersion).Trim()
if ($InstalledVersion -ne $ExpectedVersion) {
    throw "The selected Python environment contains Local AI Doctor $InstalledVersion, but VERSION is $ExpectedVersion. Reinstall the current project before packaging."
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath (Join-Path $DesktopRoot "build\backend") `
    --workpath (Join-Path $DesktopRoot "build\pyinstaller") `
    (Join-Path $DesktopRoot "backend.spec")

if ($LASTEXITCODE -ne 0) {
    throw "The backend executable build failed with exit code $LASTEXITCODE."
}
