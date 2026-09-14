[CmdletBinding()]
param(
    [string]$ApplicationRoot
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
if ([string]::IsNullOrWhiteSpace($ApplicationRoot)) {
    $ApplicationRoot = Join-Path $RepositoryRoot "release\win-unpacked"
}
$ApplicationRoot = [IO.Path]::GetFullPath($ApplicationRoot)
$RequiredFiles = @(
    (Join-Path $ApplicationRoot "Local AI Doctor.exe"),
    (Join-Path $ApplicationRoot "resources\frontend\index.html"),
    (Join-Path $ApplicationRoot "resources\config\default.yaml"),
    (Join-Path $ApplicationRoot "resources\backend\local-ai-doctor-backend\local-ai-doctor-backend.exe")
)

$Missing = @($RequiredFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($Missing.Count -gt 0) {
    throw "The packaged application is missing required files: $($Missing -join ', ')"
}

Write-Host "Packaged Electron resource layout verified."
