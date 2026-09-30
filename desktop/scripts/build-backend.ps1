[CmdletBinding()]
param(
    # Packaging interpreter. Defaults to the dedicated .venv-desktop created by
    # prepare-build-env.ps1; never falls back to the repository .venv or PATH.
    [string]$Python = $env:PYTHON
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$Guard = Join-Path $PSScriptRoot "build-guard.cjs"
$Bundle = Join-Path $DesktopRoot "build\backend\local-ai-doctor-backend"

# Windows PowerShell 5.1 turns native stderr into a terminating error when it
# is redirected under $ErrorActionPreference = "Stop". Never redirect it; use
# the exit code instead.
function Invoke-Native {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory)] [string]$Failure
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Failure (exit code $LASTEXITCODE)." }
}

$Variant = ([string](& node $Guard variant)).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not resolve the desktop PyTorch variant." }

if (-not $Python) {
    $Python = Join-Path $RepositoryRoot ".venv-desktop\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "The desktop build environment is missing. Run .\desktop\scripts\prepare-build-env.ps1 -Variant $Variant first."
    }
}

# A CUDA Toolkit on PATH must not leak its (differently versioned) DLLs into
# the bundle, and the packaged self-check must not borrow them either.
$CudaRoots = @(
    Get-ChildItem Env: | Where-Object { $_.Name -like "CUDA_PATH*" -and $_.Value } |
        ForEach-Object { $_.Value.TrimEnd("\") }
)
$env:PATH = (@($env:PATH -split ";" | Where-Object {
    $Entry = $_.TrimEnd("\")
    $Entry -and $Entry -notmatch "NVIDIA GPU Computing Toolkit\\CUDA" -and
        -not ($CudaRoots | Where-Object { $Entry.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })
}) -join ";")

Invoke-Native $Python @("-c", "import PyInstaller") `
    "PyInstaller is not installed in $Python. Run .\desktop\scripts\prepare-build-env.ps1"
Invoke-Native $Python @((Join-Path $DesktopRoot "backend_launcher.py"), "--self-check", "--variant", $Variant) `
    "$Python does not contain the $Variant PyTorch build. Run .\desktop\scripts\prepare-build-env.ps1 -Variant $Variant"

$ExpectedVersion = (Get-Content -LiteralPath (Join-Path $RepositoryRoot "VERSION") -Raw).Trim()
$InstalledVersion = ([string](& $Python -c "import importlib.metadata as metadata; print(metadata.version('local-ai-doctor'))")).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Local AI Doctor is not installed in the selected Python environment. Run .\desktop\scripts\prepare-build-env.ps1"
}
if ($InstalledVersion -ne $ExpectedVersion) {
    throw "The selected Python environment contains Local AI Doctor $InstalledVersion, but VERSION is $ExpectedVersion. Run .\desktop\scripts\prepare-build-env.ps1"
}

Invoke-Native $Python @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--distpath", (Join-Path $DesktopRoot "build\backend"),
    "--workpath", (Join-Path $DesktopRoot "build\pyinstaller"),
    (Join-Path $DesktopRoot "backend.spec")
) "The backend executable build failed"

Invoke-Native "node" @($Guard, "bundle", $Bundle) "The packaged backend has the wrong PyTorch runtime DLLs"
Invoke-Native "node" @($Guard, "self-check", (Join-Path $Bundle "local-ai-doctor-backend.exe")) `
    "The packaged backend failed its $Variant self-check"
$BundleBytes = (Get-ChildItem -LiteralPath $Bundle -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Packaged {0} backend: {1:N0} bytes ({2:N2} GiB)." -f $Variant, $BundleBytes, ($BundleBytes / 1GB))
