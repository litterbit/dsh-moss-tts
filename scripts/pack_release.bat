@echo off
rem Assemble the release folder: dist\moss-tts-cpu-win64
rem Usage: pack_release.bat [zip]
rem ASCII only: cmd.exe reads .bat as ANSI/936 (same rule as build.bat).
rem Also: cmd treats \" at the end of a quoted path as an escaped quote, so no trailing backslash.
setlocal

set "ROOT=%~dp0.."
set "DIST=%ROOT%\dist\moss-tts-cpu-win64"
set "BUILD=%ROOT%\csrc\build"
set "WEIGHTS=%ROOT%\weights"

if exist "%DIST%" rd /s /q "%DIST%"
mkdir "%DIST%\models\voices"
mkdir "%DIST%\include\moss"
mkdir "%DIST%\lib"

copy /y "%BUILD%\moss_tts.exe" "%DIST%" >nul
copy /y "%BUILD%\moss_clone.exe" "%DIST%" >nul
copy /y "%ROOT%\csrc\include\moss\tts.h" "%DIST%\include\moss" >nul
copy /y "%BUILD%\moss_tts.lib" "%DIST%\lib" >nul
copy /y "%ROOT%\doc\release_readme.md" "%DIST%\README.md" >nul

copy /y "%WEIGHTS%\llm_q8x4.moss" "%DIST%\models" >nul
copy /y "%WEIGHTS%\decoder_f16.moss" "%DIST%\models" >nul
copy /y "%WEIGHTS%\encoder_f16.moss" "%DIST%\models" >nul
copy /y "%WEIGHTS%\tokenizer.mossvocab" "%DIST%\models" >nul
copy /y "%WEIGHTS%\voices\*.mossclone" "%DIST%\models\voices" >nul

echo [ok] %DIST%

if /i "%~1"=="zip" (
  if exist "%ROOT%\dist\moss-tts-cpu-win64.zip" del /q "%ROOT%\dist\moss-tts-cpu-win64.zip"
  powershell -NoProfile -Command "Compress-Archive -Path '%DIST%' -DestinationPath '%ROOT%\dist\moss-tts-cpu-win64.zip' -CompressionLevel Optimal"
  echo [ok] %ROOT%\dist\moss-tts-cpu-win64.zip
)