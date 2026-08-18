@echo off
set "KMP_DUPLICATE_LIB_OK=TRUE"
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0run_passive_seeds.ps1' -Seeds @(202,303)" > "%~dp0seed-logs\passive_full_seeds.log" 2>&1
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" (
  echo %TRAIN_EXIT%> "%~dp0seed-logs\passive_full_seeds.exitcode"
  exit /b %TRAIN_EXIT%
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0normalize_passive_seed_outputs.ps1' -Seeds @(202,303)" >> "%~dp0seed-logs\passive_full_seeds.log" 2>&1
echo %ERRORLEVEL%> "%~dp0seed-logs\passive_full_seeds.exitcode"
