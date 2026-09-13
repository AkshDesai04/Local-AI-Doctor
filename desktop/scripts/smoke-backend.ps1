[CmdletBinding()]
param(
    [int]$Port = 16767,
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$Executable = Join-Path $DesktopRoot "build\backend\local-ai-doctor-backend\local-ai-doctor-backend.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "The packaged backend is missing at $Executable."
}

$TemporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$TemporaryRoot = [IO.Path]::GetFullPath(
    (Join-Path $TemporaryBase "local-ai-doctor-desktop-smoke-$([guid]::NewGuid().ToString('N'))")
)
if (-not $TemporaryRoot.StartsWith($TemporaryBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to create smoke-test state outside the system temporary directory."
}

New-Item -ItemType Directory -Path $TemporaryRoot | Out-Null
$ConfigDirectory = New-Item -ItemType Directory -Path (Join-Path $TemporaryRoot "config")
$ModelDirectory = New-Item -ItemType Directory -Path (Join-Path $TemporaryRoot "models")
$UserConfig = Join-Path $ConfigDirectory.FullName "local.yaml"
$ShutdownFile = Join-Path $TemporaryRoot "shutdown.signal"
[IO.File]::WriteAllText(
    $UserConfig,
    "schema_version: 1`n",
    [Text.UTF8Encoding]::new($false)
)

$StartInfo = [Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $Executable
$StartInfo.WorkingDirectory = $RepositoryRoot
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true
$StartInfo.Environment["LAD_CONFIG"] = (Join-Path $RepositoryRoot "config\default.yaml")
$StartInfo.Environment["LAD_USER_CONFIG"] = $UserConfig
$StartInfo.Environment["LAD_PROFILE"] = "native-windows"
$StartInfo.Environment["LAD_SERVER__HOST"] = "127.0.0.1"
$StartInfo.Environment["LAD_SERVER__PORT"] = [string]$Port
$StartInfo.Environment["LAD_SERVER__ALLOWED_ORIGINS"] = ConvertTo-Json -InputObject @("http://127.0.0.1:16969") -Compress
$StartInfo.Environment["LAD_RUNTIME__DEVICE"] = "cpu"
$StartInfo.Environment["LAD_DESKTOP_SHUTDOWN_FILE"] = $ShutdownFile
$StartInfo.Environment["LAD_PATHS__MODEL_ROOTS"] = ConvertTo-Json -InputObject @($ModelDirectory.FullName) -Compress
$StartInfo.Environment["LAD_PATHS__DATABASE"] = (Join-Path $TemporaryRoot "workbench.sqlite3")
$StartInfo.Environment["LAD_PATHS__UPLOADS"] = (Join-Path $TemporaryRoot "uploads")
$StartInfo.Environment["LAD_PATHS__CACHE"] = (Join-Path $TemporaryRoot "cache")
$StartInfo.Environment["LAD_PATHS__EXPORTS"] = (Join-Path $TemporaryRoot "exports")
$StartInfo.Environment["LAD_PATHS__BACKUPS"] = (Join-Path $TemporaryRoot "backups")
$StartInfo.Environment["LOCAL_AI_DOCTOR_STARTUP_TRACE"] = "1"

$Process = [Diagnostics.Process]::new()
$Process.StartInfo = $StartInfo
$Started = $false
$ShutdownFailure = $null
try {
    if (-not $Process.Start()) {
        throw "The packaged backend process could not be started."
    }
    $Started = $true

    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $StartedAt = [DateTime]::UtcNow
    $NextProgress = $StartedAt.AddSeconds(30)
    $Health = $null
    $LastHealthError = $null
    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($Process.HasExited) {
            throw "The packaged backend exited before it became healthy (exit code $($Process.ExitCode))."
        }
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/health" -TimeoutSec 2
            if ($Health.status -eq "ok" -and $Health.worker -eq "ready") { break }
        } catch {
            $LastHealthError = $_.Exception.Message
        }
        if ([DateTime]::UtcNow -ge $NextProgress) {
            $Elapsed = [math]::Round(([DateTime]::UtcNow - $StartedAt).TotalSeconds)
            $CpuSeconds = if ($Process.HasExited) { 0 } else { [math]::Round($Process.TotalProcessorTime.TotalSeconds, 1) }
            $ChildSummary = @(
                Get-CimInstance Win32_Process -Filter "ParentProcessId = $($Process.Id)" -ErrorAction SilentlyContinue |
                    ForEach-Object {
                        $Child = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
                        if ($Child) { "$($Child.Id):$([math]::Round($Child.CPU, 1))s" }
                    }
            ) -join ", "
            Write-Host "Waiting for packaged backend: ${Elapsed}s elapsed, ${CpuSeconds}s parent CPU, children [$ChildSummary]; last HTTP error: $LastHealthError"
            $NextProgress = [DateTime]::UtcNow.AddSeconds(30)
        }
        Start-Sleep -Milliseconds 500
    }
    if ($null -eq $Health -or $Health.status -ne "ok" -or $Health.worker -ne "ready") {
        throw "The packaged backend did not become healthy within $TimeoutSeconds seconds. Last health error: $LastHealthError"
    }
    $ExpectedVersion = (Get-Content -LiteralPath (Join-Path $RepositoryRoot "VERSION") -Raw).Trim()
    $OpenApi = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/openapi.json" -TimeoutSec 10
    if ($OpenApi.info.version -ne $ExpectedVersion) {
        throw "Packaged backend reports version $($OpenApi.info.version); expected $ExpectedVersion."
    }
    Write-Host "Packaged backend smoke passed on 127.0.0.1:$Port."
} finally {
    if ($Started -and -not $Process.HasExited) {
        try {
            [IO.File]::WriteAllText(
                $ShutdownFile,
                "shutdown`n",
                [Text.UTF8Encoding]::new($false)
            )
            if (-not $Process.WaitForExit(30000)) {
                $ShutdownFailure = "The packaged backend ignored its graceful shutdown marker."
                & taskkill.exe /PID $Process.Id /T /F *> $null
                $Process.WaitForExit()
            }
        } catch {
            $ShutdownFailure = "The packaged backend could not complete graceful shutdown: $($_.Exception.Message)"
            if (-not $Process.HasExited) {
                & taskkill.exe /PID $Process.Id /T /F *> $null
                $Process.WaitForExit()
            }
        }
    }
    $Process.Dispose()
    if (
        (Test-Path -LiteralPath $TemporaryRoot) -and
        $TemporaryRoot.StartsWith($TemporaryBase, [StringComparison]::OrdinalIgnoreCase)
    ) {
        Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
    }
    if ($ShutdownFailure) { throw $ShutdownFailure }
}
