@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo 先に setup_windows.bat --dev を実行してください。 & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pytest -q %*
pause
