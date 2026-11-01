@echo off
rem Build with cl directly (cmake's compiler probe hangs inside the sandbox).
rem ASCII only: cmd.exe reads .bat as ANSI/936, UTF-8 Chinese comments break parsing.
rem Static CRT (/MT) is explicit: the release must not need VC++ redistributables.
setlocal enabledelayedexpansion

set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (echo [x] vcvars64 failed & exit /b 1)

set "OUT=build"
if not exist "%OUT%" mkdir "%OUT%"

set "CFLAGS=/nologo /std:c++20 /utf-8 /O2 /Oi /EHsc /fp:fast /MT /D_CRT_SECURE_NO_WARNINGS /D_WIN32_WINNT=0x0A00 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /Iinclude /c /Fo%OUT%\"

rem ---- baseline objects (no AVX2 required) ----
cl %CFLAGS% /arch:SSE2 src\fp16.cpp src\cpu_features.cpp src\quant.cpp src\gemv.cpp src\weights.cpp src\sampler.cpp src\parallel.cpp src\audio_layer.cpp src\attn_scalar.cpp src\decoder.cpp src\encoder.cpp src\wav.cpp src\resample.cpp src\profile.cpp src\prompt.cpp src\spm.cpp src\engine.cpp src\capi.cpp src\llm.cpp src\main.cpp src\decoder_main.cpp src\tts_main.cpp src\clone_main.cpp
if errorlevel 1 (echo [x] base compile failed & exit /b 1)

rem ---- AVX2 kernel ----
cl %CFLAGS% /arch:AVX2 src\gemv_avx2.cpp src\attn_avx2.cpp
if errorlevel 1 (echo [x] avx2 compile failed & exit /b 1)

rem ---- test ----
cl %CFLAGS% /arch:SSE2 tests\test_kernels.cpp tests\test_spm.cpp
if errorlevel 1 (echo [x] test compile failed & exit /b 1)

rem ---- link ----
set "COMMON=%OUT%\fp16.obj %OUT%\cpu_features.obj %OUT%\quant.obj %OUT%\gemv.obj %OUT%\gemv_avx2.obj %OUT%\weights.obj %OUT%\sampler.obj %OUT%\parallel.obj"
set "AUDIO=%OUT%\audio_layer.obj %OUT%\attn_scalar.obj %OUT%\attn_avx2.obj"
set "LD=/nologo /O2 /EHsc /MT"

cl %LD% %COMMON% %OUT%\llm.obj %OUT%\main.obj /Fe:%OUT%\moss_llm.exe
if errorlevel 1 (echo [x] link moss_llm failed & exit /b 1)
cl %LD% %COMMON% %AUDIO% %OUT%\decoder.obj %OUT%\decoder_main.obj /Fe:%OUT%\moss_decoder.exe
if errorlevel 1 (echo [x] link moss_decoder failed & exit /b 1)
cl %LD% %COMMON% %AUDIO% %OUT%\llm.obj %OUT%\decoder.obj %OUT%\encoder.obj %OUT%\engine.obj %OUT%\spm.obj %OUT%\prompt.obj %OUT%\profile.obj %OUT%\wav.obj %OUT%\resample.obj %OUT%\tts_main.obj /Fe:%OUT%\moss_tts.exe
if errorlevel 1 (echo [x] link moss_tts failed & exit /b 1)
cl %LD% %COMMON% %AUDIO% %OUT%\encoder.obj %OUT%\wav.obj %OUT%\resample.obj %OUT%\profile.obj %OUT%\clone_main.obj /Fe:%OUT%\moss_clone.exe
if errorlevel 1 (echo [x] link moss_clone failed & exit /b 1)
cl %LD% %COMMON% %OUT%\test_kernels.obj /Fe:%OUT%\test_kernels.exe
cl %LD% %COMMON% %OUT%\spm.obj %OUT%\test_spm.obj /Fe:%OUT%\moss_spmtest.exe
if errorlevel 1 (echo [x] link test_kernels failed & exit /b 1)

lib /nologo /OUT:%OUT%\moss_tts.lib %COMMON% %AUDIO% %OUT%\llm.obj %OUT%\decoder.obj %OUT%\encoder.obj %OUT%\engine.obj %OUT%\spm.obj %OUT%\prompt.obj %OUT%\profile.obj %OUT%\wav.obj %OUT%\resample.obj %OUT%\capi.obj
if errorlevel 1 (echo [x] static lib failed & exit /b 1)
echo [ok] moss_spmtest.exe build\moss_llm.exe build\moss_decoder.exe build\moss_tts.exe build\moss_clone.exe build\test_kernels.exe
