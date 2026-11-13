@echo off
rem LJM CFD (Claude edition) - Windows setup. Creates .venv and installs dependencies.
rem Messages are ASCII-only on purpose (cmd.exe code page independent).
setlocal
cd /d "%~dp0.."

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py.exe" was not found.
    echo         Install Python 3.12 64-bit from https://www.python.org/downloads/windows/
    exit /b 1
)
py -3.12 -c "import struct,sys; sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.12 64-bit is required but was not found by "py -3.12".
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment .venv ...
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
) else (
    echo [1/3] Virtual environment .venv already exists.
)

echo [2/3] Installing dependencies ^(this downloads about 600 MB^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
if /i "%~1"=="dev" (
    ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 exit /b 1

echo [3/3] Verifying imports ...
".venv\Scripts\python.exe" -c "import PySide6, vtk, pyvista, pyvistaqt, OCP, gmsh, scipy; print('Python packages OK - VTK', vtk.vtkVersion.GetVTKVersion())"
if errorlevel 1 (
    echo [ERROR] Import check failed. See the message above.
    exit /b 1
)
echo.
echo Setup finished. Start the application with run_ljm_cfd.bat
endlocal
