# Generate scripts/model_lock.json from the weights directory.
#
# The lock file is the *trusted* source of truth for CI: the release workflow
# downloads weights from HuggingFace, then verifies them against this file
# before packaging. Never regenerate it inside the release job, or the check
# becomes circular.
#
#   pwsh scripts/make_model_lock.ps1 -Weights weights [-ModelRevision q8x4-f16-r1]
#
# ASCII only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI without BOM).
param(
    [string]$Weights = "weights",
    [string]$ModelId = "moss-tts-nano-100m-q8x4",
    [string]$ModelRevision = "q8x4-f16-r1",
    [string]$EngineVersion = "0.1.0",
    [string]$HfCommit = "",
    [string]$Out = "scripts/model_lock.json"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$weightsDir = if ([System.IO.Path]::IsPathRooted($Weights)) { $Weights } else { Join-Path $root $Weights }
$outPath = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $root $Out }

$wanted = @(
    "llm_q8x4.moss",
    "decoder_f16.moss",
    "encoder_f16.moss",
    "tokenizer.mossvocab"
)

$files = New-Object System.Collections.ArrayList
foreach ($name in $wanted) {
    $path = Join-Path $weightsDir $name
    if (-not (Test-Path $path)) { throw "missing weight file: $path" }
    $item = Get-Item $path
    [void]$files.Add([ordered]@{
        path   = $name
        bytes  = $item.Length
        sha256 = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}

$voiceDir = Join-Path $weightsDir "voices"
if (Test-Path $voiceDir) {
    foreach ($voice in (Get-ChildItem $voiceDir -Filter *.mossclone | Sort-Object Name)) {
        [void]$files.Add([ordered]@{
            path   = "voices/" + $voice.Name
            bytes  = $voice.Length
            sha256 = (Get-FileHash $voice.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
}

# hf_commit records which HuggingFace commit holds these bytes; release.yml pins the
# revision to it. Keep the old value when regenerating (use -HfCommit to change it),
# otherwise every hash refresh silently drops the pin.
if (-not $HfCommit -and (Test-Path $outPath)) {
    try { $HfCommit = (Get-Content $outPath -Raw | ConvertFrom-Json).hf_commit } catch { $HfCommit = "" }
}

$lock = [ordered]@{
    schema_version     = 1
    model_id           = $ModelId
    model_revision     = $ModelRevision
    model_format       = "moss-v1"
    min_engine_version = $EngineVersion
}
if ($HfCommit) { $lock["hf_commit"] = $HfCommit }
$lock["files"] = $files

$json = $lock | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText($outPath, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
$totalMb = [math]::Round((($files | ForEach-Object { $_.bytes }) | Measure-Object -Sum).Sum / 1MB, 2)
Write-Host "[ok] $outPath  ($($files.Count) files, $totalMb MB)"
