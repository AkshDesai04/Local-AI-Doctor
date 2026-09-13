[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$ReleaseRoot = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot "release"))
$Manifest = Get-Content -LiteralPath (Join-Path $DesktopRoot "package.json") -Raw | ConvertFrom-Json
$ExpectedName = "Local-AI-Doctor-$($Manifest.version).exe"
$ExpectedPath = [IO.Path]::GetFullPath((Join-Path $ReleaseRoot $ExpectedName))

if (-not (Test-Path -LiteralPath $ExpectedPath -PathType Leaf)) {
    throw "Expected desktop release artifact is missing: $ExpectedPath"
}

foreach ($Item in @(Get-ChildItem -LiteralPath $ReleaseRoot -Force)) {
    $ResolvedItem = [IO.Path]::GetFullPath($Item.FullName)
    if (-not $ResolvedItem.StartsWith("$ReleaseRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean an item outside the release directory: $ResolvedItem"
    }
    if ($ResolvedItem -eq $ExpectedPath) { continue }
    Remove-Item -LiteralPath $ResolvedItem -Recurse -Force
}

$Remaining = @(Get-ChildItem -LiteralPath $ReleaseRoot -Force)
if ($Remaining.Count -ne 1 -or [IO.Path]::GetFullPath($Remaining[0].FullName) -ne $ExpectedPath) {
    throw "The release directory must contain exactly $ExpectedName."
}

Write-Host "Release directory contains only $ExpectedName."
