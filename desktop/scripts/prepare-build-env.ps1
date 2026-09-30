[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Variant = "cuda",

    # Interpreter used only to create .venv-desktop. Defaults to `py -3.12`.
    [string]$BasePython
)

# Creates or refreshes the dedicated desktop packaging environment. The
# repository .venv is deliberately not used: reinstalling requirements-ml.lock
# there silently swaps CUDA PyTorch for the CPU wheel, which then ships.

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$Environment = Join-Path $RepositoryRoot ".venv-desktop"
$Python = Join-Path $Environment "Scripts\python.exe"

# Never combine $ErrorActionPreference = "Stop" with stderr redirection of a
# native command in Windows PowerShell 5.1; check the exit code instead.
function Invoke-Native {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(Mandatory)] [string[]]$Arguments,
        [Parameter(Mandatory)] [string]$Failure
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Failure (exit code $LASTEXITCODE)." }
}

function Get-Requirement([string]$Name) { Join-Path $RepositoryRoot $Name }

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    if ($BasePython) {
        Invoke-Native $BasePython @("-m", "venv", $Environment) "Could not create $Environment"
    } else {
        Invoke-Native "py" @("-3.12", "-m", "venv", $Environment) "Could not create $Environment with Python 3.12 (py -3.12)"
    }
}
$PythonVersion = ([string](& $Python -c "import sys; print('%d.%d' % sys.version_info[:2])")).Trim()
if ($LASTEXITCODE -ne 0 -or $PythonVersion -ne "3.12") {
    throw "$Environment must use Python 3.12 but uses '$PythonVersion'. Delete it and run this script again."
}

$Pip = @("-m", "pip", "install", "--disable-pip-version-check")
Invoke-Native $Python ($Pip + @("pip==26.2.1")) "Could not install the pinned pip"
Invoke-Native $Python ($Pip + @(
    "-r", (Get-Requirement "requirements.lock"),
    "-r", (Get-Requirement "requirements-ml.lock"),
    "-r", (Get-Requirement "requirements-desktop.lock")
)) "Could not install the locked runtime, ML, and desktop packages"

# The PyTorch flavor goes last with exact local-version pins. pip skips pins
# that are already satisfied, so rerunning this script is cheap.
if ($Variant -eq "cuda") {
    Invoke-Native $Python ($Pip + @("--no-deps", "-r", (Get-Requirement "requirements-cuda.lock"))) `
        "Could not install the pinned CUDA wheels from requirements-cuda.lock"
} else {
    Invoke-Native $Python ($Pip + @(
        "--no-deps", "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.8.0+cpu", "torchvision==0.23.0+cpu"
    )) "Could not install the pinned CPU wheels"
}
Invoke-Native $Python ($Pip + @("--no-deps", "--editable", $RepositoryRoot)) "Could not install the current project"

Invoke-Native $Python @((Join-Path $DesktopRoot "backend_launcher.py"), "--self-check", "--variant", $Variant) `
    "$Environment does not contain the expected $Variant PyTorch build"
Write-Host "Desktop build environment ready ($Variant): $Environment"
