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
}

function Write-Readme([string]$dir, [string]$kind) {
    $models = if ($kind -eq "full") {
        "本包已自带权重（models/，约 160 MB）。`r`n"
    } else {
        "本包**不含**权重。把 weights/ 里的 llm_q8x4.moss / decoder_f16.moss / encoder_f16.moss /`r`n" +
        "tokenizer.mossvocab 与 voices/ 放到 exe 同级的 models/ 目录下即可（或参考 doc/api.md）。`r`n"
    }
    $text = @"
MOSS-TTS CPU 推理引擎 $Tag  （Windows x64）

一、三步开始
  1. 解压到任意目录，例如 D:\MossTTS
  2. 双击 moss_tts.exe（不带参数 = 启动本地服务并自动打开浏览器）
  3. 浏览器里输入文字，Ctrl+Enter 合成

  默认地址 http://127.0.0.1:8759/ ，只监听本机，不弹防火墙。
  首次运行 Windows 可能提示"已保护你的电脑"：更多信息 -> 仍要运行（未购买代码签名证书）。

$models
二、命令行
  moss_tts.exe --text "你好，世界。" --voice Junhao --out hello.wav
  moss_tts.exe --text "..." --pcm-raw - > out.pcm      （流式 s16le / 48 kHz / 立体声）
  moss_tts.exe --list-voices
  moss_clone.exe --encoder models/encoder_f16.moss --in ref.wav --out models/voices/me.mossclone --name me
  moss_tts.exe --version

三、程序自己的数据
  便携模式：models/ 在 exe 旁边时，缓存与克隆音色写到 exe 同级的 userdata/。
  共享模式：找不到 models/ 时，回退到 %LOCALAPPDATA%\MossTTS\。
  删掉 userdata/ 或 models/cache/ 都不会丢东西，只是下次首音慢一点（约 0.2 s）。

四、接入自己的程序
  本地 HTTP（只监听 127.0.0.1）：
    GET  /api/v1/health | /info | /voices
    POST /api/v1/tts/stream    JSON -> chunked s16le
    POST /api/v1/tts/wav       JSON -> 完整 WAV
    POST /api/v1/voices/clone?name=xxx
    GET  /api/v1/jobs/<id>     统计
    POST /api/v1/jobs/<id>/cancel
  C 接口：include/moss/tts.h + lib/moss_tts.lib
  详见仓库 doc/api.md。

五、许可
  Apache-2.0，见 LICENSE；模型派生与改动说明见 NOTICE。
  请只使用你有权使用的声音做克隆。
"@
    # BOM 让老版记事本也能正确识别 UTF-8
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
