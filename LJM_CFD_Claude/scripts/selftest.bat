@echo off
rem Self-test: import a small box, mesh it, run a short CFD analysis (if OpenFOAM is
rem available in WSL) and open the 3D view once. Prints OK / failed / skipped per step.
rem The same text is saved to workspace\logs\selftest.txt.
rem   scripts\selftest.bat              full check
rem   scripts\selftest.bat --no-solver  skip the CFD analysis
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment was not found. Run scripts\setup_windows.bat first.
    exit /b 1
)
set "PYTHONPATH=%~dp0..\src"
".venv\Scripts\python.exe" -m ljmcfd --self-test %*
