@echo off
rem Start LJM CFD with a console so that startup errors and logs are visible.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment was not found. Run scripts\setup_windows.bat first.
    pause
    exit /b 1
)
set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -X faulthandler -m ljmcfd %*
echo.
echo LJM CFD exited with code %errorlevel%. Logs: workspace\logs
pause
