[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$UnpackedRoot = Join-Path $RepositoryRoot "release\win-unpacked"
$RequiredFiles = @(
    (Join-Path $UnpackedRoot "Local AI Doctor.exe"),
    (Join-Path $UnpackedRoot "resources\frontend\index.html"),
    (Join-Path $UnpackedRoot "resources\config\default.yaml"),
    (Join-Path $UnpackedRoot "resources\backend\local-ai-doctor-backend\local-ai-doctor-backend.exe")
)

$Missing = @($RequiredFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($Missing.Count -gt 0) {
    throw "The packaged application is missing required files: $($Missing -join ', ')"
}

Write-Host "Packaged Electron resource layout verified."
