@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pyinstaller.exe" (
  echo PyInstaller がありません。setup_windows.bat --dev を実行してください。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean ljm_re.spec
if errorlevel 1 ( echo [エラー] EXE の作成に失敗しました。 & pause & exit /b 1 )
echo 自己診断を実行しています...
start "" /wait "dist\LJM_ReverseEngineering\LJM_ReverseEngineering.exe" --self-test "%TEMP%\ljm_selftest"
if errorlevel 1 ( echo [警告] 自己診断に失敗しました。ログ: %LOCALAPPDATA%\LJM_ReverseEngineering\logs ) else ( echo 自己診断 OK: %TEMP%\ljm_selftest )
echo.
echo 作成しました: dist\LJM_ReverseEngineering\LJM_ReverseEngineering.exe
echo フォルダごと配布してください（onedir 形式。Qt 等の DLL は差し替え可能な状態で同梱されます）。
pause
