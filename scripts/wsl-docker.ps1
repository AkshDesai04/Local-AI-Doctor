[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet(
        "Check", "CheckNvidia", "ConfigCpu", "ConfigNvidia", "BuildCpu", "BuildNvidia",
        "UpCpu", "UpNvidia", "HealthCpu", "HealthNvidia", "NvidiaSmoke", "LogsCpu",
        "LogsNvidia", "StopCpu", "StopNvidia", "Down", "Ps", "Backup", "Restore"
    )]
    [string]$Action = "Check",

    [Parameter()]
    [string]$Distribution = $env:LAD_WSL_DISTRIBUTION,

    [Parameter()]
    [string]$WslUser = $env:LAD_WSL_USER,

    [Parameter()]
    [string]$BackupFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentFile = Join-Path $RepositoryRoot ".env"

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*?)\s*$") {
            $value = $Matches[1]
            if ($value.Length -ge 2 -and (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            )) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }
    return $null
}

function Get-Wsl2Distribution {
    param([string]$Requested)

    $rawListing = (& wsl.exe --list --verbose 2>&1 | Out-String) -replace "`0", ""
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list WSL distributions. Enable WSL and install a WSL2 distribution."
    }

    $distributions = @()
    foreach ($line in ($rawListing -split "`r?`n")) {
        if ($line -match '^\s*(?<default>\*)?\s*(?<name>.+?)\s{2,}(?<state>Running|Stopped)\s{2,}(?<version>[12])\s*$') {
            $distributions += [pscustomobject]@{
                Name = $Matches.name.Trim()
                Default = $Matches.default -eq "*"
                Version = [int]$Matches.version
            }
        }
    }
    if ($distributions.Count -eq 0) {
        throw "No WSL distributions were detected."
    }

    if ($Requested) {
        $selected = @($distributions | Where-Object Name -eq $Requested)
        if ($selected.Count -ne 1) {
            $available = ($distributions.Name -join ", ")
            throw "WSL distribution '$Requested' was not found. Available distributions: $available"
        }
        $distribution = $selected[0]
    }
    else {
        $defaults = @($distributions | Where-Object { $_.Default -and $_.Version -eq 2 })
        $wsl2 = @($distributions | Where-Object Version -eq 2)
        if ($defaults.Count -eq 1) {
            $distribution = $defaults[0]
        }
        elseif ($wsl2.Count -eq 1) {
            $distribution = $wsl2[0]
        }
        else {
            throw "Select a WSL2 distribution with -Distribution or LAD_WSL_DISTRIBUTION."
        }
    }

    if ($distribution.Version -ne 2) {
        throw "WSL distribution '$($distribution.Name)' uses WSL$($distribution.Version); Docker requires WSL2."
    }
    return $distribution.Name
}

function Invoke-Wsl {
    param([Parameter(Mandatory)] [string[]]$Command)

    $arguments = @("--distribution", $script:Distribution, "--user", $script:WslUser, "--") + $Command
    & wsl.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code ${LASTEXITCODE}: $($Command[0])"
    }
}

function Invoke-WslCapture {
    param([Parameter(Mandatory)] [string[]]$Command)

    $arguments = @("--distribution", $script:Distribution, "--user", $script:WslUser, "--") + $Command
    $rawResult = (& wsl.exe @arguments 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code ${LASTEXITCODE}: $($Command[0])"
    }
    $resultLines = @($rawResult -split "`r?`n" | Where-Object { $_.Trim() })
    return $resultLines[-1].Trim()
}

function Invoke-Compose {
    param([Parameter(Mandatory)] [string[]]$Arguments)
    Invoke-Wsl -Command ($script:ComposeCommand + $Arguments)
}

if (-not $Distribution) {
    $Distribution = Get-DotEnvValue -Path $EnvironmentFile -Name "WSL_DISTRIBUTION"
}
$Distribution = Get-Wsl2Distribution -Requested $Distribution
$script:Distribution = $Distribution

if (-not $WslUser) {
    $WslUser = Get-DotEnvValue -Path $EnvironmentFile -Name "WSL_USER"
}
if (-not $WslUser) {
    $defaultUserArguments = @("--distribution", $Distribution, "--", "id", "-un")
    $rawDefaultUser = (& wsl.exe @defaultUserArguments 2>&1 | Out-String).Trim()
    $defaultUserLines = @($rawDefaultUser -split "`r?`n" | Where-Object { $_.Trim() })
    $WslUser = if ($defaultUserLines.Count) { $defaultUserLines[-1].Trim() } else { "" }
    if ($LASTEXITCODE -ne 0 -or -not $WslUser) {
        throw "Could not resolve the WSL user. Supply -WslUser or LAD_WSL_USER."
    }
}
$script:WslUser = $WslUser

$resolvedUser = Invoke-WslCapture -Command @("id", "-un")
if ($resolvedUser -ne $WslUser) {
    throw "WSL selected user '$resolvedUser' instead of requested user '$WslUser'."
}

Write-Host "Using WSL2 distribution '$Distribution' as Linux user '$WslUser'."
Invoke-Wsl -Command @("docker", "info", "--format", "Docker Engine {{.ServerVersion}} is reachable")
Invoke-Wsl -Command @("docker", "compose", "version")

if ($Action -eq "Check") {
    return
}
if ($Action -eq "CheckNvidia") {
    Invoke-Wsl -Command @("nvidia-smi")
    return
}

if (-not (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf)) {
    throw "Create the ignored .env file from .env.example before using Compose actions."
}

$RepositoryWindowsForWsl = $RepositoryRoot.Replace("\", "/")
$EnvironmentWindowsForWsl = $EnvironmentFile.Replace("\", "/")
$RepositoryWsl = Invoke-WslCapture -Command @("wslpath", "-a", "-u", $RepositoryWindowsForWsl)
$EnvironmentWsl = Invoke-WslCapture -Command @("wslpath", "-a", "-u", $EnvironmentWindowsForWsl)
$ComposeWsl = "$RepositoryWsl/compose.yaml"
$script:ComposeCommand = @(
    "docker", "compose", "--ansi", "never", "--project-directory", $RepositoryWsl,
    "--env-file", $EnvironmentWsl, "--file", $ComposeWsl
)

$healthProbe = "import os,urllib.request;u='http://127.0.0.1:'+os.environ.get('CONTAINER_LISTEN_PORT','8000')+'/api/v1/health';r=urllib.request.urlopen(u,timeout=3);print(r.read().decode())"
$nvidiaProbe = "import torch;print({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None});raise SystemExit(0 if torch.cuda.is_available() else 1)"

switch ($Action) {
    "ConfigCpu" { Invoke-Compose -Arguments @("--profile", "cpu", "config") }
    "ConfigNvidia" { Invoke-Compose -Arguments @("--profile", "nvidia", "config") }
    "BuildCpu" { Invoke-Compose -Arguments @("--profile", "cpu", "build", "--pull", "app-cpu") }
    "BuildNvidia" { Invoke-Compose -Arguments @("--profile", "nvidia", "build", "--pull", "app-nvidia") }
    "UpCpu" { Invoke-Compose -Arguments @("--profile", "cpu", "up", "--build", "--detach", "--wait", "app-cpu") }
    "UpNvidia" {
        Invoke-Wsl -Command @("nvidia-smi")
        Invoke-Compose -Arguments @("--profile", "nvidia", "up", "--build", "--detach", "--wait", "app-nvidia")
    }
    "HealthCpu" { Invoke-Compose -Arguments @("--profile", "cpu", "exec", "-T", "app-cpu", "python", "-c", $healthProbe) }
    "HealthNvidia" { Invoke-Compose -Arguments @("--profile", "nvidia", "exec", "-T", "app-nvidia", "python", "-c", $healthProbe) }
    "NvidiaSmoke" { Invoke-Compose -Arguments @("--profile", "nvidia", "exec", "-T", "app-nvidia", "python", "-c", $nvidiaProbe) }
    "LogsCpu" { Invoke-Compose -Arguments @("--profile", "cpu", "logs", "--follow", "--tail", "200", "app-cpu") }
    "LogsNvidia" { Invoke-Compose -Arguments @("--profile", "nvidia", "logs", "--follow", "--tail", "200", "app-nvidia") }
    "StopCpu" { Invoke-Compose -Arguments @("--profile", "cpu", "stop", "app-cpu") }
    "StopNvidia" { Invoke-Compose -Arguments @("--profile", "nvidia", "stop", "app-nvidia") }
    "Down" { Invoke-Compose -Arguments @("--profile", "cpu", "--profile", "nvidia", "down", "--remove-orphans") }
    "Ps" { Invoke-Compose -Arguments @("--profile", "cpu", "--profile", "nvidia", "ps", "--all") }
    "Backup" { Invoke-Compose -Arguments @("--profile", "maintenance", "run", "--rm", "database-maintenance", "backup") }
    "Restore" {
        if (-not $BackupFile) {
            throw "Restore requires -BackupFile with a filename from the backups volume."
        }
        if ([IO.Path]::GetFileName($BackupFile) -ne $BackupFile) {
            throw "BackupFile must be a filename, not a path."
        }
        Invoke-Compose -Arguments @(
            "--profile", "maintenance", "run", "--rm", "database-maintenance",
            "restore", "/data/backups/$BackupFile"
        )
    }
}
