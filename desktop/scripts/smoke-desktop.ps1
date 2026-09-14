[CmdletBinding()]
param(
    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 600,

    [ValidateRange(1, 3600)]
    [int]$InstallTimeoutSeconds = 1800,

    [ValidateRange(1, 1800)]
    [int]$UninstallTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $DesktopRoot
$Manifest = Get-Content -LiteralPath (Join-Path $DesktopRoot "package.json") -Raw | ConvertFrom-Json
$Installer = Join-Path $RepositoryRoot "release\Local-AI-Doctor-$($Manifest.version).exe"
$InstallerGuid = [string]$Manifest.build.nsis.guid
$ExpectedInstallerGuid = "df0eb923-5a87-57ad-bb13-24a35a6c435a"
$BackendUri = "http://127.0.0.1:6767/api/v1/health"
$FrontendUri = "http://127.0.0.1:6969/healthz"

if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "The Electron installer is missing at $Installer."
}
if ($InstallerGuid -ne $ExpectedInstallerGuid) {
    throw "The NSIS installer GUID must remain $ExpectedInstallerGuid; found $InstallerGuid."
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
$TemporaryPrefix = if (
    $TemporaryBase.EndsWith([string][IO.Path]::DirectorySeparatorChar) -or
    $TemporaryBase.EndsWith([string][IO.Path]::AltDirectorySeparatorChar)
) {
    $TemporaryBase
} else {
    "$TemporaryBase$([IO.Path]::DirectorySeparatorChar)"
}
$TemporaryRoot = [IO.Path]::GetFullPath(
    # Keep the isolated install path short. NSIS still uses legacy Win32 path
    # handling while uninstalling, so a verbose test-only prefix can push
    # otherwise valid packaged dependency paths past MAX_PATH and leave them
    # behind even though the uninstaller and registry cleanup succeeded.
    (Join-Path $TemporaryBase "lad-$([IO.Path]::GetRandomFileName())")
)
if (-not $TemporaryRoot.StartsWith($TemporaryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to create Electron smoke state outside the system temporary directory."
}
$InstallRoot = [IO.Path]::GetFullPath((Join-Path $TemporaryRoot "i"))
$UserDataRoot = [IO.Path]::GetFullPath((Join-Path $TemporaryRoot "u"))
foreach ($ScopedPath in @($InstallRoot, $UserDataRoot)) {
    if (-not $ScopedPath.StartsWith("$TemporaryRoot$([IO.Path]::DirectorySeparatorChar)", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use a smoke-test path outside the isolated temporary root: $ScopedPath"
    }
}

$LocalAppDataRoot = [IO.Path]::GetFullPath($env:LOCALAPPDATA)
$LocalAppDataPrefix = if (
    $LocalAppDataRoot.EndsWith([string][IO.Path]::DirectorySeparatorChar) -or
    $LocalAppDataRoot.EndsWith([string][IO.Path]::AltDirectorySeparatorChar)
) {
    $LocalAppDataRoot
} else {
    "$LocalAppDataRoot$([IO.Path]::DirectorySeparatorChar)"
}
$UpdaterCacheDirectory = [IO.Path]::GetFullPath(
    (Join-Path $LocalAppDataRoot "local-ai-doctor-desktop-updater")
)
$UpdaterCacheFile = [IO.Path]::GetFullPath((Join-Path $UpdaterCacheDirectory "installer.exe"))
if (-not $UpdaterCacheFile.StartsWith($LocalAppDataPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to inspect an updater cache outside LOCALAPPDATA."
}
$UpdaterCacheDirectoryExisted = Test-Path -LiteralPath $UpdaterCacheDirectory -PathType Container
if (Test-Path -LiteralPath $UpdaterCacheFile -PathType Leaf) {
    throw "Refusing to overwrite an existing Local AI Doctor installer cache at $UpdaterCacheFile."
}

$InstallIdentityRegistryPaths = @(
    "Registry::HKEY_CURRENT_USER\Software\$InstallerGuid",
    "Registry::HKEY_CURRENT_USER\Software\WOW6432Node\$InstallerGuid",
    "Registry::HKEY_LOCAL_MACHINE\Software\$InstallerGuid",
    "Registry::HKEY_LOCAL_MACHINE\Software\WOW6432Node\$InstallerGuid"
)
$UninstallRegistryPaths = @(
    "Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InstallerGuid",
    "Registry::HKEY_CURRENT_USER\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$InstallerGuid",
    "Registry::HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InstallerGuid",
    "Registry::HKEY_LOCAL_MACHINE\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$InstallerGuid"
)
$AllInstallerRegistryPaths = @($InstallIdentityRegistryPaths + $UninstallRegistryPaths)

function Test-HttpEndpoint {
    param([Parameter(Mandatory)] [string]$Uri)

    try {
        $Response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $Response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-SamePath {
    param(
        [Parameter(Mandatory)] [string]$Left,
        [Parameter(Mandatory)] [string]$Right
    )

    try {
        $LeftPath = [IO.Path]::GetFullPath($Left).TrimEnd(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        )
        $RightPath = [IO.Path]::GetFullPath($Right).TrimEnd(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        )
        return $LeftPath.Equals($RightPath, [StringComparison]::OrdinalIgnoreCase)
    } catch {
        return $false
    }
}

function Get-NamedProductRegistrations {
    $RegistryRoots = @(
        "Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "Registry::HKEY_CURRENT_USER\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        "Registry::HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "Registry::HKEY_LOCAL_MACHINE\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    )
    $Matches = [Collections.Generic.List[string]]::new()
    foreach ($RegistryRoot in $RegistryRoots) {
        if (-not (Test-Path -LiteralPath $RegistryRoot)) { continue }
        foreach ($RegistryKey in @(Get-ChildItem -LiteralPath $RegistryRoot -ErrorAction SilentlyContinue)) {
            $Properties = Get-ItemProperty -LiteralPath $RegistryKey.PSPath -ErrorAction SilentlyContinue
            if ($Properties -and [string]$Properties.DisplayName -like "Local AI Doctor*") {
                $Matches.Add($RegistryKey.PSPath)
            }
        }
    }
    return @($Matches)
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

function Invoke-BoundedProcess {
    param(
        [Parameter(Mandatory)] [string]$FileName,
        [Parameter(Mandatory)] [string]$Arguments,
        [Parameter(Mandatory)] [string]$WorkingDirectory,
        [Parameter(Mandatory)] [int]$ProcessTimeoutSeconds,
        [Parameter(Mandatory)] [string]$Operation
    )

    $StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $FileName
    $StartInfo.Arguments = $Arguments
    $StartInfo.WorkingDirectory = $WorkingDirectory
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $Child = [Diagnostics.Process]::new()
    $Child.StartInfo = $StartInfo
    $Timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $Child.Start()) {
            throw "$Operation could not be started."
        }
        if (-not $Child.WaitForExit($ProcessTimeoutSeconds * 1000)) {
            if (-not $Child.HasExited) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $Child.Id /T /F *> $null
                $Child.WaitForExit()
            }
            throw "$Operation did not finish within $ProcessTimeoutSeconds seconds."
        }
        if ($Child.ExitCode -ne 0) {
            throw "$Operation exited with code $($Child.ExitCode)."
        }
        return [math]::Round($Timer.Elapsed.TotalSeconds, 2)
    } finally {
        $Timer.Stop()
        $Child.Dispose()
    }
}

function Get-StartupTraceMeasurement {
    param([Parameter(Mandatory)] [string]$TracePath)

    if (-not (Test-Path -LiteralPath $TracePath -PathType Leaf)) { return $null }
    try {
        $Events = @(
            Get-Content -LiteralPath $TracePath |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                ForEach-Object { $_ | ConvertFrom-Json }
        )
    } catch {
        return $null
    }

    $CompleteLaunches = [Collections.Generic.List[object]]::new()
    foreach ($Launch in @($Events | Group-Object -Property launchId)) {
        $BackendEvents = @($Launch.Group | Where-Object { $_.event -eq "backend_healthy" })
        $RequestEvents = @($Launch.Group | Where-Object { $_.event -eq "frontend_start_requested" })
        $ListeningEvents = @($Launch.Group | Where-Object { $_.event -eq "frontend_listening" })
        if ($BackendEvents.Count -eq 1 -and $RequestEvents.Count -eq 1 -and $ListeningEvents.Count -eq 1) {
            $CompleteLaunches.Add([pscustomobject]@{
                LaunchId = [string]$Launch.Name
                BackendHealthyMs = [long]$BackendEvents[0].monotonicMs
                FrontendRequestedMs = [long]$RequestEvents[0].monotonicMs
                FrontendListeningMs = [long]$ListeningEvents[0].monotonicMs
            })
        }
    }
    if ($CompleteLaunches.Count -eq 0) { return $null }
    return $CompleteLaunches |
        Sort-Object -Property FrontendListeningMs |
        Select-Object -Last 1
}

function Wait-ForUninstallCleanup {
    param(
        [Parameter(Mandatory)] [string]$TargetInstallRoot,
        [Parameter(Mandatory)] [string[]]$RegistryPaths,
        [Parameter(Mandatory)] [int]$CleanupTimeoutSeconds
    )

    $Deadline = [DateTime]::UtcNow.AddSeconds($CleanupTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $Deadline) {
        $RegistryRemaining = @($RegistryPaths | Where-Object { Test-Path -LiteralPath $_ })
        if (-not (Test-Path -LiteralPath $TargetInstallRoot) -and $RegistryRemaining.Count -eq 0) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

$ExistingIdentityKeys = @($AllInstallerRegistryPaths | Where-Object { Test-Path -LiteralPath $_ })
$ExistingNamedRegistrations = @(Get-NamedProductRegistrations)
if ($ExistingIdentityKeys.Count -gt 0 -or $ExistingNamedRegistrations.Count -gt 0) {
    $Existing = @($ExistingIdentityKeys + $ExistingNamedRegistrations | Sort-Object -Unique)
    throw "Desktop installer smoke refuses to replace an existing Local AI Doctor installation: $($Existing -join ', ')"
}

New-Item -ItemType Directory -Path $TemporaryRoot | Out-Null

$Process = $null
$Started = $false
$ManagedProcesses = @{}
$ShutdownRequested = $false
$GracefulShutdownVerified = $false
$InstallationAttempted = $false
$UninstallVerified = $false
$Uninstaller = $null
$InstallElapsedSeconds = $null
$BackendElapsedSeconds = $null
$FrontendElapsedSeconds = $null
$ReadinessToFrontendMilliseconds = $null
$PrimaryFailure = $null
$CleanupFailures = [Collections.Generic.List[string]]::new()

try {
    # /D must be the final, unquoted argument. Electron Builder's NSIS parser
    # treats the complete remainder, including spaces, as the destination.
    $InstallationAttempted = $true
    $InstallElapsedSeconds = Invoke-BoundedProcess `
        -FileName $Installer `
        -Arguments "/S /currentuser --no-desktop-shortcut /D=$InstallRoot" `
        -WorkingDirectory $TemporaryRoot `
        -ProcessTimeoutSeconds $InstallTimeoutSeconds `
        -Operation "The Local AI Doctor installer"

    & (Join-Path $PSScriptRoot "verify-packaged-layout.ps1") -ApplicationRoot $InstallRoot
    $InstalledExecutable = Join-Path $InstallRoot "Local AI Doctor.exe"
    $Uninstallers = @(Get-ChildItem -LiteralPath $InstallRoot -File -Filter "Uninstall*.exe")
    if ($Uninstallers.Count -ne 1) {
        throw "The isolated installation must contain exactly one uninstaller; found $($Uninstallers.Count)."
    }
    $Uninstaller = $Uninstallers[0].FullName

    $CurrentUserInstallKey = "Registry::HKEY_CURRENT_USER\Software\$InstallerGuid"
    $CurrentUserUninstallKey = "Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Uninstall\$InstallerGuid"
    if (
        -not (Test-Path -LiteralPath $CurrentUserInstallKey) -or
        -not (Test-Path -LiteralPath $CurrentUserUninstallKey)
    ) {
        throw "The silent NSIS install did not register the expected current-user installation."
    }
    $RegisteredLocation = [string](
        Get-ItemProperty -LiteralPath $CurrentUserInstallKey -Name InstallLocation
    ).InstallLocation
    if (-not (Test-SamePath -Left $RegisteredLocation -Right $InstallRoot)) {
        throw "NSIS registered '$RegisteredLocation' instead of the isolated path '$InstallRoot'."
    }
    if (Test-Path -LiteralPath $UpdaterCacheFile -PathType Leaf) {
        throw "The NSIS install left a redundant full installer copy at $UpdaterCacheFile."
    }

    $StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $InstalledExecutable
    $StartInfo.Arguments = "--disable-gpu --user-data-dir=`"$UserDataRoot`""
    $StartInfo.WorkingDirectory = $InstallRoot
    $StartInfo.UseShellExecute = $false
    $StartInfo.Environment["LAD_LOGGING__LEVEL"] = "info"
    $StartInfo.Environment["LAD_DESKTOP_SMOKE_TRACE"] = "1"
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    if (-not $Process.Start()) {
        throw "The installed Electron application could not be started."
    }
    $Started = $true
    if (-not (Register-ManagedProcess -ManagedProcesses $ManagedProcesses -TargetProcessId $Process.Id)) {
        throw "The installed Electron process could not be registered for safe process-tree cleanup."
    }

    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $StartedAt = [DateTime]::UtcNow
    $NextProgress = $StartedAt.AddSeconds(30)
    $Health = $null
    while ([DateTime]::UtcNow -lt $Deadline) {
        Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
        $LiveProcessIds = @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses)
        if ($Process.HasExited -and $LiveProcessIds.Count -eq 0) {
            throw "The installed Electron process tree exited before the backend became healthy (launcher exit code $($Process.ExitCode))."
        }
        if (Test-HttpEndpoint -Uri $FrontendUri) {
            throw "The frontend became available before the backend reported healthy."
        }
        try {
            $Health = Invoke-RestMethod -Uri $BackendUri -TimeoutSec 2
            if (
                $Health.status -eq "ok" -and
                $Health.database -eq "ready" -and
                $Health.worker -eq "ready"
            ) { break }
        } catch {
            $Health = $null
        }
        if ([DateTime]::UtcNow -ge $NextProgress) {
            $Elapsed = [math]::Round(([DateTime]::UtcNow - $StartedAt).TotalSeconds)
            Write-Host "Waiting for installed Electron backend: ${Elapsed}s elapsed; managed PIDs [$($LiveProcessIds -join ', ')]."
            $NextProgress = [DateTime]::UtcNow.AddSeconds(30)
        }
        Start-Sleep -Milliseconds 250
    }
    if (
        $null -eq $Health -or
        $Health.status -ne "ok" -or
        $Health.database -ne "ready" -or
        $Health.worker -ne "ready"
    ) {
        throw "The installed Electron backend did not become healthy within $TimeoutSeconds seconds."
    }
    $BackendElapsedSeconds = [math]::Round(([DateTime]::UtcNow - $StartedAt).TotalSeconds, 2)

    $FrontendReady = $false
    while ([DateTime]::UtcNow -lt $Deadline) {
        Update-ManagedProcessTree -ManagedProcesses $ManagedProcesses
        $LiveProcessIds = @(Get-LiveManagedProcessIds -ManagedProcesses $ManagedProcesses)
        if ($LiveProcessIds.Count -eq 0) {
            throw "The installed Electron process tree exited before the frontend became healthy."
        }
        if (Test-HttpEndpoint -Uri $FrontendUri) {
            $FrontendReady = $true
            break
        }
        Start-Sleep -Milliseconds 100
    }
    if (-not $FrontendReady) {
        throw "The installed Electron frontend did not become healthy within $TimeoutSeconds seconds."
    }
    $FrontendElapsedSeconds = [math]::Round(([DateTime]::UtcNow - $StartedAt).TotalSeconds, 2)

    $TracePath = Join-Path $UserDataRoot "runtime\startup-trace.jsonl"
    $TraceDeadline = [DateTime]::UtcNow.AddSeconds(10)
    $TraceMeasurement = $null
    while ([DateTime]::UtcNow -lt $TraceDeadline -and $null -eq $TraceMeasurement) {
        $TraceMeasurement = Get-StartupTraceMeasurement -TracePath $TracePath
        if ($null -eq $TraceMeasurement) { Start-Sleep -Milliseconds 50 }
    }
    if ($null -eq $TraceMeasurement) {
        throw "The installed Electron application did not write a complete monotonic startup trace."
    }
    $ReadinessToFrontendMilliseconds = $TraceMeasurement.FrontendRequestedMs - $TraceMeasurement.BackendHealthyMs
    if ($ReadinessToFrontendMilliseconds -lt 0) {
        throw "The frontend start was requested before complete backend readiness was recorded."
    }
    if ($TraceMeasurement.FrontendListeningMs -lt $TraceMeasurement.FrontendRequestedMs) {
        throw "The startup trace reports the frontend listening before its start was requested."
    }

    $Index = Invoke-WebRequest -Uri "http://127.0.0.1:6969/" -UseBasicParsing -TimeoutSec 10
    if ($Index.StatusCode -ne 200 -or $Index.Content -notmatch '<div id="root"></div>') {
        throw "The installed Electron frontend did not serve the production application shell."
    }

    $FrontendProcessId = Get-ManagedListenerProcessId -Port 6969 -ManagedProcesses $ManagedProcesses
    if ($null -eq $FrontendProcessId) {
        throw "The frontend listener is not owned by the installed Electron process tree."
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
        throw "The installed Electron window could not be closed through its normal application shutdown path."
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
    $BackendLog = Join-Path $UserDataRoot "logs\backend.log"
    if (-not (Test-Path -LiteralPath $BackendLog -PathType Leaf)) {
        throw "The Electron backend log is missing, so file-signaled shutdown could not be verified."
    }
    $BackendLogContent = Get-Content -LiteralPath $BackendLog -Raw
    if ($BackendLogContent -notmatch "Application shutdown complete\.") {
        throw "The backend exited without completing its ASGI shutdown; the file-marker shutdown path failed."
    }

    $UninstallElapsedSeconds = Invoke-BoundedProcess `
        -FileName $Uninstaller `
        -Arguments "/S /currentuser" `
        -WorkingDirectory $TemporaryRoot `
        -ProcessTimeoutSeconds $UninstallTimeoutSeconds `
        -Operation "The isolated Local AI Doctor uninstaller"
    if (-not (Wait-ForUninstallCleanup `
        -TargetInstallRoot $InstallRoot `
        -RegistryPaths $AllInstallerRegistryPaths `
        -CleanupTimeoutSeconds $UninstallTimeoutSeconds
    )) {
        $RemainingKeys = @($AllInstallerRegistryPaths | Where-Object { Test-Path -LiteralPath $_ })
        throw "The isolated uninstall did not clean its install root and registry keys. Remaining keys: $($RemainingKeys -join ', ')"
    }
    $UninstallVerified = $true

    $ListenersAfterUninstall = @(
        6767, 6969 | ForEach-Object {
            Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
        }
    )
    if ($ListenersAfterUninstall.Count -gt 0) {
        throw "The desktop smoke left a listener on port 6767 or 6969 after uninstall."
    }
    if (Test-Path -LiteralPath $UpdaterCacheFile -PathType Leaf) {
        throw "The desktop smoke left the redundant NSIS installer cache at $UpdaterCacheFile."
    }
} catch {
    $PrimaryFailure = $_
} finally {
    if ($Started -and -not $GracefulShutdownVerified -and $null -ne $Process) {
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
            try { $Process.Kill() } catch { }
        }
    }
    if ($null -ne $Process) { $Process.Dispose() }

    if ($InstallationAttempted -and -not $UninstallVerified -and -not $Uninstaller -and (Test-Path -LiteralPath $InstallRoot -PathType Container)) {
        $FallbackUninstallers = @(Get-ChildItem -LiteralPath $InstallRoot -File -Filter "Uninstall*.exe" -ErrorAction SilentlyContinue)
        if ($FallbackUninstallers.Count -eq 1) {
            $Uninstaller = $FallbackUninstallers[0].FullName
        }
    }
    if ($InstallationAttempted -and -not $UninstallVerified -and $Uninstaller -and (Test-Path -LiteralPath $Uninstaller -PathType Leaf)) {
        try {
            $null = Invoke-BoundedProcess `
                -FileName $Uninstaller `
                -Arguments "/S /currentuser" `
                -WorkingDirectory $TemporaryRoot `
                -ProcessTimeoutSeconds $UninstallTimeoutSeconds `
                -Operation "Cleanup of the isolated Local AI Doctor installation"
            if (-not (Wait-ForUninstallCleanup `
                -TargetInstallRoot $InstallRoot `
                -RegistryPaths $AllInstallerRegistryPaths `
                -CleanupTimeoutSeconds $UninstallTimeoutSeconds
            )) {
                $CleanupFailures.Add("The fallback uninstaller left its install root or registry keys behind.")
            }
        } catch {
            $CleanupFailures.Add($_.Exception.Message)
        }
    }

    if ($InstallationAttempted -and -not $UninstallVerified) {
        foreach ($RegistryPath in @($AllInstallerRegistryPaths | Where-Object { Test-Path -LiteralPath $_ })) {
            try {
                # Preflight proved that none of these exact GUID-scoped keys
                # existed before this installer invocation.
                Remove-Item -LiteralPath $RegistryPath -Recurse -Force
            } catch {
                $CleanupFailures.Add("Could not remove smoke-created registry key ${RegistryPath}: $($_.Exception.Message)")
            }
        }
        $RegistryKeysStillPresent = @($AllInstallerRegistryPaths | Where-Object { Test-Path -LiteralPath $_ })
        if ($RegistryKeysStillPresent.Count -gt 0) {
            $CleanupFailures.Add("Smoke-created registry keys remain: $($RegistryKeysStillPresent -join ', ')")
        }
    }

    if (Test-Path -LiteralPath $UpdaterCacheFile -PathType Leaf) {
        try {
            Remove-Item -LiteralPath $UpdaterCacheFile -Force
        } catch {
            $CleanupFailures.Add("Could not remove the smoke-created installer cache: $($_.Exception.Message)")
        }
    }
    if (-not $UpdaterCacheDirectoryExisted -and (Test-Path -LiteralPath $UpdaterCacheDirectory -PathType Container)) {
        try {
            Remove-Item -LiteralPath $UpdaterCacheDirectory -Recurse -Force
        } catch {
            $CleanupFailures.Add("Could not remove the smoke-created updater cache directory: $($_.Exception.Message)")
        }
    }
    if (Test-Path -LiteralPath $UpdaterCacheFile -PathType Leaf) {
        $CleanupFailures.Add("The smoke-created installer cache still exists at $UpdaterCacheFile.")
    }

    if (Test-Path -LiteralPath $TemporaryRoot) {
        try {
            Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
        } catch {
            $CleanupFailures.Add("Could not remove isolated Electron smoke state: $($_.Exception.Message)")
        }
    }
    if (Test-Path -LiteralPath $TemporaryRoot) {
        $CleanupFailures.Add("Isolated Electron smoke state still exists at $TemporaryRoot.")
    }

    $RemainingListeners = @(
        6767, 6969 | ForEach-Object {
            Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
        }
    )
    if ($RemainingListeners.Count -gt 0) {
        $ListenerOwners = @($RemainingListeners | Select-Object -ExpandProperty OwningProcess -Unique)
        $CleanupFailures.Add("Ports 6767 or 6969 remain occupied after cleanup by PID(s): $($ListenerOwners -join ', ')")
    }
}

if ($null -ne $PrimaryFailure) {
    foreach ($CleanupFailure in $CleanupFailures) { Write-Warning $CleanupFailure }
    throw $PrimaryFailure
}
if ($CleanupFailures.Count -gt 0) {
    throw "Desktop smoke completed but cleanup failed: $($CleanupFailures -join '; ')"
}

Write-Host (
    "Installed Electron smoke passed: install ${InstallElapsedSeconds}s; backend ${BackendElapsedSeconds}s; " +
    "frontend ${FrontendElapsedSeconds}s; readiness-to-frontend request ${ReadinessToFrontendMilliseconds}ms; " +
    "uninstall ${UninstallElapsedSeconds}s; graceful shutdown and cleanup verified."
)
