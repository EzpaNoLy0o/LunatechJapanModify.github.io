@echo off
rem Build a distributable folder with PyInstaller:
rem   dist\LJM_CFD_Claude\LJM_CFD_Claude.exe   (GUI)
rem   dist\LJM_CFD_Claude\ljmcfd-worker.exe    (background jobs, started by the GUI)
rem The folder runs without Python. OpenFOAM is NOT included; it still lives in WSL2.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment was not found. Run scripts\setup_windows.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -q "pyinstaller>=6.10,<7"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m PyInstaller packaging\ljmcfd.spec --noconfirm --distpath dist --workpath build
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)
echo.
echo Checking the built application (import, mesh, 3D view; CFD is skipped here)...
set "LJMCFD_HOME=%~dp0..\build\selftest_home"
"dist\LJM_CFD_Claude\ljmcfd-worker.exe" --self-test --no-solver
if errorlevel 1 (
    echo The built application did not pass the self-test. See the messages above.
    exit /b 1
)
echo.
echo Done: dist\LJM_CFD_Claude\LJM_CFD_Claude.exe
echo Copy the whole dist\LJM_CFD_Claude folder to a path without non-ASCII characters.
