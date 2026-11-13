@echo off
rem Start LJM CFD (Claude edition) without a console window.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo The virtual environment was not found. Run scripts\setup_windows.bat first.
    pause
    exit /b 1
)
set "PYTHONPATH=%~dp0src"
start "" ".venv\Scripts\pythonw.exe" -m ljmcfd %*
