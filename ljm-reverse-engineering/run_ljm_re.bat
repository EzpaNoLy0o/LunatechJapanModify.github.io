@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo 先に setup_windows.bat を実行してください。
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
