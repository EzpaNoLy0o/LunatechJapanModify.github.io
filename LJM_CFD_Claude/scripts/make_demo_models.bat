@echo off
rem Generate demo CAD models (Ahmed body, demo car assembly, open panel) into .\demo
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup_windows.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" scripts\make_demo_models.py demo
