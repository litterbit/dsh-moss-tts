# Download the model weights declared in scripts/model_lock.json.
#
#   pwsh scripts/fetch_weights.ps1 -Base "https://huggingface.co/<owner>/moss-tts-nano-moss/resolve/<commit>"
#   pwsh scripts/fetch_weights.ps1            # uses $env:MOSS_WEIGHTS_BASE
#
# Files already present with the right size+sha256 are skipped, so this is
# safe to run repeatedly (and it is what the release workflow uses with a
# layer-cache around it). Nothing is trusted: every download is verified
# against the committed lock file afterwards by scripts/pack_release.ps1.
#
# ASCII only (Windows PowerShell 5.1 reads .ps1 as ANSI without BOM).
param(
    [string]$Base = $env:MOSS_WEIGHTS_BASE,
    [string]$Weights = "weights"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$weightsDir = if ([System.IO.Path]::IsPathRooted($Weights)) { $Weights } else { Join-Path $root $Weights }
$lockPath = Join-Path $root "scripts/model_lock.json"

if (-not $Base) { throw "no base URL: pass -Base or set MOSS_WEIGHTS_BASE" }
if (-not (Test-Path $lockPath)) { throw "missing $lockPath" }
$Base = $Base.TrimEnd("/")
$lock = Get-Content $lockPath -Raw | ConvertFrom-Json

foreach ($file in $lock.files) {
    $relative = $file.path -replace "/", "\"
    $target = Join-Path $weightsDir $relative
    $parent = Split-Path $target -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }

    if (Test-Path $target) {
        $item = Get-Item $target
        if ($item.Length -eq $file.bytes) {
            $hash = (Get-FileHash $target -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($hash -eq $file.sha256) { Write-Host "[skip] $relative"; continue }
        }
        Remove-Item $target -Force
    }
    $url = "$Base/$($file.path)"
    Write-Host "[get ] $url"
    & curl.exe -fL --retry 3 --retry-delay 2 -o $target $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    $item = Get-Item $target
    if ($item.Length -ne $file.bytes) { throw "size mismatch for $relative" }
    $hash = (Get-FileHash $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $file.sha256) { throw "sha256 mismatch for $relative" }
}

Write-Host "[ok] weights ready in $weightsDir"
