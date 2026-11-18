# Builds ctx and installs it to %LOCALAPPDATA%\agents-context\bin, then restarts the daemon.
#
# - Stops only the `embed-server` process: `ctx mcp` servers of open Claude/OpenCode sessions
#   keep running (they pick up the new binary on their next start).
# - Those MCP servers keep ctx.exe locked, so the old file is renamed aside, not overwritten.
# - The daemon is launched through WMI via the Startup launcher, so it is not a child of the
#   shell running this script and survives it.
# - Fails loudly if the daemon does not come back: a silent failure leaves every hook and search
#   without a daemon.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $env:LOCALAPPDATA 'agents-context\bin'
$launcher = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup\ctx-embed-server.vbs'

# wscript returns 0 even for a missing script, which would only surface as a 3-minute wait.
if (-not (Test-Path $launcher)) { throw "daemon launcher missing: $launcher" }
$health = "$(if ($env:CTX_EMBED_URL) { $env:CTX_EMBED_URL.TrimEnd('/') } else { 'http://127.0.0.1:7997' })/health"

cargo build --manifest-path (Join-Path $repo 'Cargo.toml') -p ctx --release --features directml
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }

Get-CimInstance Win32_Process -Filter "Name='ctx.exe'" |
    Where-Object { $_.CommandLine -match 'embed-server' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Milliseconds 800

New-Item -ItemType Directory -Force $bin | Out-Null
Get-ChildItem (Join-Path $bin 'ctx.exe.*.old') -ErrorAction SilentlyContinue |
    ForEach-Object { try { Remove-Item $_ -ErrorAction Stop } catch { } }  # still locked: next time
$exe = Join-Path $bin 'ctx.exe'
if (Test-Path $exe) { Move-Item $exe "$exe.$(Get-Date -Format yyyyMMddHHmmss).old" }
Copy-Item (Join-Path $repo 'target\release\ctx.exe') $exe

# Keep the OpenCode plugins in sync (indexing + auto-inject), if OpenCode is installed.
$plugins = Join-Path $env:USERPROFILE '.config\opencode\plugins'
if (Test-Path (Split-Path -Parent $plugins)) {
    New-Item -ItemType Directory -Force $plugins | Out-Null
    foreach ($js in 'ctx-index.js', 'ctx-inject.js') {
        Copy-Item (Join-Path $repo "integrations\opencode\$js") (Join-Path $plugins $js) -Force
    }
    # The read-only agent `ctx replay` runs with; without it the replay refuses to start.
    $agents = Join-Path $env:USERPROFILE '.config\opencode\agents'
    New-Item -ItemType Directory -Force $agents | Out-Null
    Copy-Item (Join-Path $repo 'integrations\opencode\agents\ctx-replay.md') (Join-Path $agents 'ctx-replay.md') -Force
}

function Test-Daemon([int]$seconds) {
    for ($i = 0; $i -lt $seconds; $i += 2) {
        try { Invoke-RestMethod $health -TimeoutSec 2 | Out-Null; return $true } catch { Start-Sleep 2 }
    }
    return $false
}

foreach ($attempt in 1..2) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "wscript.exe `"$launcher`"" }
    if ($r.ReturnValue -eq 0 -and (Test-Daemon 90)) { Write-Host "ctx installed; daemon up (attempt $attempt)"; exit 0 }
    Write-Warning "daemon did not come up (attempt $attempt)"
}
throw "daemon failed to start; see $(Join-Path $env:LOCALAPPDATA 'agents-context\embed-server.log')"
