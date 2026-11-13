@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem コンソール付きで起動（エラー調査用）。ログは %%LOCALAPPDATA%%\LJM_ReverseEngineering\logs にも保存されます。
if not exist ".venv\Scripts\python.exe" (
  echo 先に setup_windows.bat を実行してください。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "%~dp0main.py" --debug %*
pause
