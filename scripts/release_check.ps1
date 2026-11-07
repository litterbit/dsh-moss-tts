# Local replica of the release gate. One command, same order as .github/workflows/release.yml.
#
#   pwsh scripts/release_check.ps1 [-SkipBuild] [-Tag v0.1.0]
#
# Steps: build -> kernel tests (avx2+scalar) -> tokenizer golden -> real-model selftest
#        -> HTTP smoke (stream/cancel/security/pace) -> stdio smoke
#        -> playback simulation (no dropped audio) -> pack full/lite/sdk -> verify zip contents.
# The DSH plugin lives in its own repository (litterbit/dsh-plugin-moss-tts) and has its own CI.
# Exits non-zero on the first failure.
#
# ASCII only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI without BOM);
# the Chinese text below is inside a UTF-8-BOM file, which is fine.
param(
    [switch]$SkipBuild,
    [string]$Tag = "v0.1.0",
    [string]$Python = "C:\Users\PC\miniforge3\envs\audio\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
function Resolve([string]$p) { if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $root $p } }

$buildDir = Join-Path $root "csrc/build"
$weightsDir = Resolve "weights"
$step = 0
function Step([string]$title) {
    $script:step++
    Write-Host ""
    Write-Host ("[" + $script:step + "] " + $title) -ForegroundColor Cyan
}
function Fail([string]$message) { Write-Host "[x] $message" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $Python)) {
    $fallback = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($fallback) { $Python = $fallback } else { Fail "python not found (pass -Python <path>)" }
}

if (-not $SkipBuild) {
    Step "build"
    & cmd /c (Join-Path $root "csrc\build.bat")
    if ($LASTEXITCODE -ne 0) { Fail "build failed" }
}

Step "kernel unit tests (avx2)"
& (Join-Path $buildDir "test_kernels.exe")
if ($LASTEXITCODE -ne 0) { Fail "test_kernels (avx2) failed" }

Step "kernel unit tests (scalar)"
$env:MOSS_FORCE_SCALAR = "1"
& (Join-Path $buildDir "test_kernels.exe")
if ($LASTEXITCODE -ne 0) { $env:MOSS_FORCE_SCALAR = $null; Fail "test_kernels (scalar) failed" }
$env:MOSS_FORCE_SCALAR = $null

Step "tokenizer golden (61 sentences)"
& (Join-Path $buildDir "moss_spmtest.exe") --vocab (Join-Path $root "csrc/tests/data/tokenizer.mossvocab") `
    --corpus (Join-Path $root "csrc/tests/data/spm_corpus.txt") `
    --golden (Join-Path $root "csrc/tests/data/spm_golden.txt")
if ($LASTEXITCODE -ne 0) { Fail "tokenizer golden failed" }

Step "real-model selftest"
& (Join-Path $buildDir "moss_tts.exe") --models $weightsDir --selftest
if ($LASTEXITCODE -ne 0) { Fail "selftest failed" }

Step "HTTP smoke (stream / cancel / security)"
$server = Start-Process -FilePath (Join-Path $buildDir "moss_tts.exe") `
    -ArgumentList "--models", $weightsDir, "--serve", "--no-open", "--port", "8790", `
                  "--web", (Join-Path $root "web/app"), "--cache", (Join-Path $root "runs/cache"), `
                  "--data", (Join-Path $root "runs/userdata") -PassThru -NoNewWindow
try {
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8790/api/v1/health" -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { Fail "server did not become ready" }
    & $Python (Join-Path $root "tools/http_smoke.py") --base "http://127.0.0.1:8790" --cross-origin "https://litterbit.github.io"
    if ($LASTEXITCODE -ne 0) { Fail "http smoke failed" }
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}

Step "stdio smoke (plugin channel)"
& $Python (Join-Path $root "tools/stdio_smoke.py") --exe (Join-Path $buildDir "moss_tts.exe") --models $weightsDir
if ($LASTEXITCODE -ne 0) { Fail "stdio smoke failed" }

Step "playback simulation (worklet + player.js must not drop audio)"
node (Join-Path $root "tools/playback_sim.mjs") --exe (Join-Path $buildDir "moss_tts.exe") --models $weightsDir --port 8791
if ($LASTEXITCODE -ne 0) { Fail "playback simulation failed" }

Step "script encoding guard"
# Windows PowerShell 5.1 reads a .ps1 with no BOM as ANSI: a stray Chinese byte can eat the
# next quote and break the parser in ways that only show up on a clean machine. Keep every
# script either pure ASCII or explicitly BOM'd.
foreach ($file in (Get-ChildItem (Join-Path $root "scripts") -Filter *.ps1)) {
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    $nonAscii = ($bytes | Where-Object { $_ -gt 127 }).Count
    if (-not $hasBom -and $nonAscii -gt 0) { Fail ("$($file.Name): non-ASCII without UTF-8 BOM") }
}
Write-Host "    all scripts ASCII or BOM'd"

Step "pack full / lite / sdk"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts/pack_release.ps1") -Mode all -Tag $Tag
if ($LASTEXITCODE -ne 0) { Fail "pack failed" }

Step "verify packages"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $root "dist/MossTTS-win-x64-full.zip"))
try {
    # Compress-Archive stores entry names with backslashes on Windows PowerShell 5.1: normalize first
    $names = $zip.Entries | ForEach-Object { $_.FullName.Replace('\', '/') }
    $required = @("moss_tts.exe", "moss_clone.exe", "web/index.html", "web/app.js", "web/player.js",
                  "web/pcm-worklet.js",
                  "models/llm_q8x4.moss", "models/decoder_f16.moss", "models/encoder_f16.moss",
                  "models/tokenizer.mossvocab", "models/manifest.json", "LICENSE", "NOTICE", "README.txt")
    $missing = $required | Where-Object { $names -notcontains $_ }
    if ($missing) { Fail ("missing in full zip: " + ($missing -join ", ")) }
    if (-not ($names | Where-Object { $_ -like "models/voices/*.mossclone" })) { Fail "no voices in full zip" }
    Write-Host ("    full zip: " + $names.Count + " entries")
} finally { $zip.Dispose() }
$mb = [math]::Round((Get-Item (Join-Path $root "dist/MossTTS-win-x64-full.zip")).Length / 1MB, 1)
if ($mb -lt 140) { Fail "full zip too small: $mb MB" }

Write-Host ""
Write-Host ("[ok] release gate passed (" + $mb + " MB full zip, " + $script:step + " steps)") -ForegroundColor Green
