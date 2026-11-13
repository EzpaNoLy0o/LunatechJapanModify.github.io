@echo off
rem Run the automated tests (opens and closes real application windows).
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup_windows.bat dev first.
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -q -r requirements-dev.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pytest %*
