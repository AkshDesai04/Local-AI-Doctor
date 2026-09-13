[CmdletBinding()]
param(
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$Manifest = Get-Content -LiteralPath (Join-Path $DesktopRoot "package.json") -Raw | ConvertFrom-Json
$Executable = Join-Path $RepositoryRoot "release\Local-AI-Doctor-$($Manifest.version).exe"
$BackendUri = "http://127.0.0.1:6767/api/v1/health"
$FrontendUri = "http://127.0.0.1:6969/healthz"

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "The portable Electron executable is missing at $Executable."
}

$Occupied = @(
    6767, 6969 | ForEach-Object {
        Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
    }
)
if ($Occupied.Count -gt 0) {
    throw "Desktop smoke requires free loopback ports 6767 and 6969."
}

$TemporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$TemporaryRoot = [IO.Path]::GetFullPath(
    (Join-Path $TemporaryBase "local-ai-doctor-electron-smoke-$([guid]::NewGuid().ToString('N'))")
)
if (-not $TemporaryRoot.StartsWith($TemporaryBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to create Electron smoke state outside the system temporary directory."
}
New-Item -ItemType Directory -Path $TemporaryRoot | Out-Null

function Test-HttpEndpoint {
    param([Parameter(Mandatory)] [string]$Uri)

    try {
        $Response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $Response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Get-ProcessStartKey {
    param([Parameter(Mandatory)] [int]$TargetProcessId)

    $Candidate = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
    if (-not $Candidate) { return $null }
    try {
        return $Candidate.StartTime.ToUniversalTime().Ticks.ToString([Globalization.CultureInfo]::InvariantCulture)
    } catch {
        return $null
    }
}

function Register-ManagedProcess {
    param(
        [Parameter(Mandatory)] [hashtable]$ManagedProcesses,
        [Parameter(Mandatory)] [int]$TargetProcessId
    )

    if ($ManagedProcesses.ContainsKey($TargetProcessId)) { return $true }
    $StartKey = Get-ProcessStartKey -TargetProcessId $TargetProcessId
    if ($null -eq $StartKey) { return $false }
    $ManagedProcesses[$TargetProcessId] = $StartKey
    return $true
}

function Test-ManagedProcessAlive {
    param(
        [Parameter(Mandatory)] [hashtable]$ManagedProcesses,
        [Parameter(Mandatory)] [int]$TargetProcessId
    )

    if (-not $ManagedProcesses.ContainsKey($TargetProcessId)) { return $false }
    $StartKey = Get-ProcessStartKey -TargetProcessId $TargetProcessId
    return $null -ne $StartKey -and $StartKey -eq $ManagedProcesses[$TargetProcessId]
}

function Get-ManagedProcess {
    param(
        [Parameter(Mandatory)] [hashtable]$ManagedProcesses,
        [Parameter(Mandatory)] [int]$TargetProcessId
    )

    if (-not $ManagedProcesses.ContainsKey($TargetProcessId)) { return $null }
    $Candidate = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
    if (-not $Candidate) { return $null }
    try {
        $StartKey = $Candidate.StartTime.ToUniversalTime().Ticks.ToString(
            [Globalization.CultureInfo]::InvariantCulture
        )
    } catch {
        return $null
    }
    if ($StartKey -ne $ManagedProcesses[$TargetProcessId]) { return $null }
    return $Candidate
}

function Update-ManagedProcessTree {
    param([Parameter(Mandatory)] [hashtable]$ManagedProcesses)

    $Processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $KnownProcessIds = [Collections.Generic.HashSet[int]]::new()
    $LiveParentIds = [Collections.Generic.HashSet[int]]::new()
    foreach ($KnownProcessId in $ManagedProcesses.Keys) {
        $TypedProcessId = [int]$KnownProcessId
        $null = $KnownProcessIds.Add($TypedProcessId)
        if (Test-ManagedProcessAlive -ManagedProcesses $ManagedProcesses -TargetProcessId $TypedProcessId) {
            $null = $LiveParentIds.Add($TypedProcessId)
        }
    }

    $Changed = $true
    while ($Changed) {
        $Changed = $false
        foreach ($Candidate in $Processes) {
            $CandidateId = [int]$Candidate.ProcessId
            $ParentId = [int]$Candidate.ParentProcessId
            if (
                -not $KnownProcessIds.Contains($CandidateId) -and
                $LiveParentIds.Contains($ParentId) -and
                (Register-ManagedProcess -ManagedProcesses $ManagedProcesses -TargetProcessId $CandidateId)
            ) {
                $null = $KnownProcessIds.Add($CandidateId)
                $null = $LiveParentIds.Add($CandidateId)
                $Changed = $true
            }
        }
    }
}

function Get-LiveManagedProcessIds {
    param([Parameter(Mandatory)] [hashtable]$ManagedProcesses)

    Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
    return @(
        $ManagedProcesses.Keys |
            Where-Object { Test-ManagedProcessAlive -ManagedProcesses $ManagedProcesses -TargetProcessId ([int]$_) } |
            ForEach-Object { [int]$_ } |
            Sort-Object -Unique
    )
}

function Get-ManagedListenerProcessId {
    param(
        [Parameter(Mandatory)] [int]$Port,
        [Parameter(Mandatory)] [hashtable]$ManagedProcesses
    )

    Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
    $Owners = @(
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    foreach ($Owner in $Owners) {
        if (Test-ManagedProcessAlive -ManagedProcesses $ManagedProcesses -TargetProcessId ([int]$Owner)) {
            return [int]$Owner
        }
    }
    return $null
}

function Stop-ManagedProcessTree {
    param([Parameter(Mandatory)] [hashtable]$ManagedProcesses)

    # Re-scan between passes so a child created during cleanup can only be
    # registered through an already verified member of this process tree.
    for ($Attempt = 0; $Attempt -lt 3; $Attempt += 1) {
        $LiveProcessIds = @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses)
        if ($LiveProcessIds.Count -eq 0) { return }
        foreach ($TargetProcessId in $LiveProcessIds) {
            $ManagedProcess = Get-ManagedProcess `
                -ManagedProcesses $ManagedProcesses `
                -TargetProcessId $TargetProcessId
            if ($ManagedProcess) {
                try { $ManagedProcess.Kill() } catch { }
            }
        }
        Start-Sleep -Milliseconds 250
    }
}

$StartInfo = [Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $Executable
$StartInfo.Arguments = "--disable-gpu --user-data-dir=`"$TemporaryRoot`""
$StartInfo.WorkingDirectory = $RepositoryRoot
$StartInfo.UseShellExecute = $false
$StartInfo.Environment["LAD_LOGGING__LEVEL"] = "info"
$Process = [Diagnostics.Process]::new()
$Process.StartInfo = $StartInfo
$Started = $false
$ManagedProcesses = @{}
$ShutdownRequested = $false
$GracefulShutdownVerified = $false

try {
    if (-not $Process.Start()) {
        throw "The portable Electron executable could not be started."
    }
    $Started = $true
    if (-not (Register-ManagedProcess -ManagedProcesses $ManagedProcesses -TargetProcessId $Process.Id)) {
        throw "The portable Electron launcher could not be registered for safe process-tree cleanup."
    }
    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $StartedAt = [DateTime]::UtcNow
    $NextProgress = $StartedAt.AddSeconds(30)
    $Health = $null

    while ([DateTime]::UtcNow -lt $Deadline) {
        Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
        $LiveProcessIds = @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses)
        if ($Process.HasExited -and $LiveProcessIds.Count -eq 0) {
            throw "The portable Electron process tree exited before the backend became healthy (launcher exit code $($Process.ExitCode))."
        }
        if (Test-HttpEndpoint -Uri $FrontendUri) {
            throw "The frontend became available before the backend reported healthy."
        }
        try {
            $Health = Invoke-RestMethod -Uri $BackendUri -TimeoutSec 2
            if ($Health.status -eq "ok" -and $Health.worker -eq "ready") { break }
        } catch {
            $Health = $null
        }
        if ([DateTime]::UtcNow -ge $NextProgress) {
            $Elapsed = [math]::Round(([DateTime]::UtcNow - $StartedAt).TotalSeconds)
            Write-Host "Waiting for Electron-managed backend: ${Elapsed}s elapsed; managed PIDs [$($LiveProcessIds -join ', ')]."
            $NextProgress = [DateTime]::UtcNow.AddSeconds(30)
        }
        Start-Sleep -Milliseconds 250
    }
    if ($null -eq $Health -or $Health.status -ne "ok" -or $Health.worker -ne "ready") {
        throw "The Electron-managed backend did not become healthy within $TimeoutSeconds seconds."
    }

    $DelayTimer = [Diagnostics.Stopwatch]::StartNew()
    $FrontendReady = $false
    while ([DateTime]::UtcNow -lt $Deadline) {
        Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
        $LiveProcessIds = @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses)
        if ($LiveProcessIds.Count -eq 0) {
            throw "The Electron process tree exited before the frontend became healthy."
        }
        if (Test-HttpEndpoint -Uri $FrontendUri) {
            # Health polling observes readiness after the backend's own timer
            # starts, so allow up to two seconds of scheduling/polling skew.
            if ($DelayTimer.Elapsed.TotalSeconds -lt 14) {
                throw "The frontend started only $([math]::Round($DelayTimer.Elapsed.TotalSeconds, 2)) seconds after observed backend health."
            }
            $FrontendReady = $true
            break
        }
        Start-Sleep -Milliseconds 100
    }
    if (-not $FrontendReady) {
        throw "The Electron-managed frontend did not become healthy within $TimeoutSeconds seconds."
    }

    $Index = Invoke-WebRequest -Uri "http://127.0.0.1:6969/" -UseBasicParsing -TimeoutSec 10
    if ($Index.StatusCode -ne 200 -or $Index.Content -notmatch '<div id="root"></div>') {
        throw "The Electron frontend did not serve the production application shell."
    }

    $FrontendProcessId = Get-ManagedListenerProcessId -Port 6969 -ManagedProcesses $ManagedProcesses
    if ($null -eq $FrontendProcessId) {
        throw "The frontend listener is not owned by the launched Electron process tree."
    }
    $WindowDeadline = [DateTime]::UtcNow.AddSeconds(30)
    while ([DateTime]::UtcNow -lt $WindowDeadline -and -not $ShutdownRequested) {
        $FrontendProcess = Get-ManagedProcess `
            -ManagedProcesses $ManagedProcesses `
            -TargetProcessId $FrontendProcessId
        if ($FrontendProcess -and $FrontendProcess.MainWindowHandle -ne 0) {
            $ShutdownRequested = $FrontendProcess.CloseMainWindow()
        }
        if (-not $ShutdownRequested) { Start-Sleep -Milliseconds 250 }
    }
    if (-not $ShutdownRequested) {
        throw "The Electron window could not be closed through its normal application shutdown path."
    }

    $ShutdownDeadline = [DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $ShutdownDeadline) {
        $LiveProcessIds = @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses)
        $Listeners = @(
            6767, 6969 | ForEach-Object {
                Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
            }
        )
        if ($LiveProcessIds.Count -eq 0 -and $Listeners.Count -eq 0) {
            $GracefulShutdownVerified = $true
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $GracefulShutdownVerified) {
        throw "Electron did not gracefully stop its frontend, backend, and child process tree within 60 seconds."
    }
    $BackendLog = Join-Path $TemporaryRoot "logs\backend.log"
    if (-not (Test-Path -LiteralPath $BackendLog -PathType Leaf)) {
        throw "The Electron backend log is missing, so file-signaled shutdown could not be verified."
    }
    $BackendLogContent = Get-Content -LiteralPath $BackendLog -Raw
    if ($BackendLogContent -notmatch "Application shutdown complete\.") {
        throw "The backend exited without completing its ASGI shutdown; the file-marker shutdown path failed."
    }

    Write-Host "Electron smoke passed; frontend followed backend health after $([math]::Round($DelayTimer.Elapsed.TotalSeconds, 2)) seconds and file-signaled shutdown completed gracefully."
} finally {
    if ($Started -and -not $GracefulShutdownVerified) {
        Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
        if (-not $ShutdownRequested) {
            $FrontendProcessId = Get-ManagedListenerProcessId -Port 6969 -ManagedProcesses $ManagedProcesses
            if ($null -ne $FrontendProcessId) {
                $FrontendProcess = Get-ManagedProcess `
                    -ManagedProcesses $ManagedProcesses `
                    -TargetProcessId $FrontendProcessId
                if ($FrontendProcess -and $FrontendProcess.MainWindowHandle -ne 0) {
                    $ShutdownRequested = $FrontendProcess.CloseMainWindow()
                }
            }
        }
        if ($ShutdownRequested) {
            $CleanupDeadline = [DateTime]::UtcNow.AddSeconds(10)
            while (
                [DateTime]::UtcNow -lt $CleanupDeadline -and
                @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses).Count -gt 0
            ) {
                Start-Sleep -Milliseconds 250
            }
        }
        Stop-ManagedProcessTree -ManagedProcesses $ManagedProcesses
        if (-not $Process.HasExited) {
            # This is the exact Process instance started above, not a listener
            # discovered by port number, so terminating it cannot target a
            # process outside the smoke test.
            try { $Process.Kill() } catch { }
        }
    }
    $Process.Dispose()

    if (
        (Test-Path -LiteralPath $TemporaryRoot) -and
        $TemporaryRoot.StartsWith($TemporaryBase, [StringComparison]::OrdinalIgnoreCase)
    ) {
        Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
