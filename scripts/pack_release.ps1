# Assemble release packages into dist/.
#
#   pwsh scripts/pack_release.ps1 -Mode full|lite|sdk|all [-Tag v0.1.0] [-SkipVerify]
#
# Outputs (asset names are intentionally version-less, so the
# "releases/latest/download/<name>" URLs used by the landing page stay valid
# forever):
#   dist/MossTTS-win-x64-full.zip   exe + web + models + docs  (~160 MB)
#   dist/MossTTS-win-x64-lite.zip   exe + web + docs, no weights (~3 MB)
#   dist/MossTTS-sdk.zip            header + static lib + api doc
#   dist/sha256sums.txt
#
# Verifies every weight against scripts/model_lock.json before packing: the
# lock file is committed and reviewed, so this catches a wrong/updated weight
# instead of silently shipping it.
#
# ASCII only (Windows PowerShell 5.1 reads .ps1 as ANSI without BOM).
param(
    [ValidateSet("full", "lite", "sdk", "all")]
    [string]$Mode = "full",
    [string]$Tag = "v0.1.0",
    [string]$Weights = "weights",
    [string]$OutDir = "dist",
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
function Resolve([string]$p) { if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $root $p } }

$weightsDir = Resolve $Weights
$distDir = Resolve $OutDir
$stageRoot = Join-Path $distDir "stage"
$lockPath = Join-Path $root "scripts/model_lock.json"
$buildDir = Join-Path $root "csrc/build"

if (-not (Test-Path $lockPath)) { throw "missing $lockPath (run scripts/make_model_lock.ps1)" }
$lock = Get-Content $lockPath -Raw | ConvertFrom-Json

# ---- weight verification -----------------------------------------------------
function Verify-Weights {
    $bad = 0
    foreach ($file in $lock.files) {
        $path = Join-Path $weightsDir ($file.path -replace '/', '\')
        if (-not (Test-Path $path)) { Write-Host "[x] missing $($file.path)"; $bad++; continue }
        $item = Get-Item $path
        if ($item.Length -ne $file.bytes) {
            Write-Host "[x] size mismatch $($file.path): $($item.Length) != $($file.bytes)"; $bad++; continue
        }
        $hash = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $file.sha256) {
            Write-Host "[x] sha256 mismatch $($file.path)"; $bad++
        }
    }
    if ($bad -gt 0) { throw "$bad weight file(s) do not match scripts/model_lock.json" }
    Write-Host "[ok] weights match model_lock.json ($($lock.files.Count) files)"
}

# ---- package contents --------------------------------------------------------
function New-Stage([string]$name) {
    $dir = Join-Path $stageRoot $name
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    return $dir
}

function Copy-Common([string]$dir) {
    Copy-Item (Join-Path $buildDir "moss_tts.exe") $dir
    Copy-Item (Join-Path $buildDir "moss_clone.exe") $dir
    Copy-Item (Join-Path $root "LICENSE") $dir
    Copy-Item (Join-Path $root "NOTICE") $dir
    Copy-Item (Join-Path $root "web/app") (Join-Path $dir "web") -Recurse
    # Ship the DSH plugin too: after install, point the profile at <install dir>\dsh-plugin\index.mjs
    Copy-Item (Join-Path $root "dsh-plugin") (Join-Path $dir "dsh-plugin") -Recurse
}

function Write-Readme([string]$dir, [string]$kind) {
    # README body lives in doc/release_readme.txt (UTF-8). Keeping the Chinese out of
    # this script means the script can stay pure ASCII, so Windows PowerShell 5.1
    # cannot mis-decode it as ANSI even when the BOM gets lost by an editor.
    $templatePath = Join-Path $root "doc/release_readme.txt"
    if (-not (Test-Path $templatePath)) { throw "missing $templatePath" }
    $template = [System.IO.File]::ReadAllText($templatePath, [System.Text.Encoding]::UTF8)
    $kept = foreach ($line in ($template -split "`r?`n")) {
        if ($kind -ne "full" -and $line.StartsWith("<!--FULL-->")) { continue }
        if ($kind -ne "lite" -and $line.StartsWith("<!--LITE-->")) { continue }
        ($line -replace '^<!--(FULL|LITE)-->', '')
    }
    $text = ($kept -join "`r`n").Replace("{{TAG}}", $Tag)
    [System.IO.File]::WriteAllText((Join-Path $dir "README.txt"), $text,
        (New-Object System.Text.UTF8Encoding($true)))
}

function New-Zip([string]$dir, [string]$zipName) {
    $zipPath = Join-Path $distDir $zipName
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $dir "*") -DestinationPath $zipPath -CompressionLevel Optimal
    $mb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host "[ok] $zipPath ($mb MB)"
}

# ---- build modes -------------------------------------------------------------
$modes = if ($Mode -eq "all") { @("full", "lite", "sdk") } else { @($Mode) }
if (-not $SkipVerify -and ($modes -contains "full" -or $modes -contains "lite")) { Verify-Weights }
New-Item -ItemType Directory -Force -Path $distDir, $stageRoot | Out-Null

foreach ($m in $modes) {
    if (-not (Test-Path (Join-Path $buildDir "moss_tts.exe"))) { throw "missing csrc/build/moss_tts.exe (run csrc/build.bat)" }
    switch ($m) {
        "full" {
            $dir = New-Stage "MossTTS-win-x64-full"
            Copy-Common $dir
            $modelsDir = Join-Path $dir "models"
            New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
            foreach ($file in $lock.files) {
                $src = Join-Path $weightsDir ($file.path -replace '/', '\')
                $dst = Join-Path $modelsDir ($file.path -replace '/', '\')
                $dstParent = Split-Path $dst -Parent
                if (-not (Test-Path $dstParent)) { New-Item -ItemType Directory -Force -Path $dstParent | Out-Null }
                Copy-Item $src $dst
            }
            Copy-Item $lockPath (Join-Path $modelsDir "manifest.json")
            Write-Readme $dir "full"
            New-Zip $dir "MossTTS-win-x64-full.zip"
        }
        "lite" {
            $dir = New-Stage "MossTTS-win-x64-lite"
            Copy-Common $dir
            $modelsDir = Join-Path $dir "models"
            New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
            Copy-Item $lockPath (Join-Path $modelsDir "manifest.json")
            Write-Readme $dir "lite"
            New-Zip $dir "MossTTS-win-x64-lite.zip"
        }
        "sdk" {
            $dir = New-Stage "MossTTS-sdk"
            New-Item -ItemType Directory -Force -Path (Join-Path $dir "include/moss"), (Join-Path $dir "lib"), (Join-Path $dir "doc") | Out-Null
            Copy-Item (Join-Path $root "csrc/include/moss/tts.h") (Join-Path $dir "include/moss")
            Copy-Item (Join-Path $buildDir "moss_tts.lib") (Join-Path $dir "lib")
            if (Test-Path (Join-Path $root "doc/api.md")) { Copy-Item (Join-Path $root "doc/api.md") (Join-Path $dir "doc") }
            Copy-Item (Join-Path $root "LICENSE") $dir
            Copy-Item (Join-Path $root "NOTICE") $dir
            New-Zip $dir "MossTTS-sdk.zip"
        }
    }
}

# ---- checksums ---------------------------------------------------------------
$zips = Get-ChildItem $distDir -Filter "*.zip" | Sort-Object Name
$lines = foreach ($zip in $zips) {
    $hash = (Get-FileHash $zip.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($zip.Name)"
}
[System.IO.File]::WriteAllText((Join-Path $distDir "sha256sums.txt"), (($lines -join "`n") + "`n"),
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host "[ok] $(Join-Path $distDir 'sha256sums.txt')"
