@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo 先に setup_windows.bat を実行してください。 & pause & exit /b 1 )
".venv\Scripts\python.exe" -c "import pytest, pytestqt" 2>nul
if errorlevel 1 (
  echo テスト用ライブラリ ^(pytest^) をインストールしています...
  ".venv\Scripts\python.exe" -m pip install pytest==9.2.0 pytest-qt==4.5.0
  if errorlevel 1 ( echo [エラー] pytest のインストールに失敗しました。 & pause & exit /b 1 )
)
".venv\Scripts\python.exe" -m pytest -q %*
pause
