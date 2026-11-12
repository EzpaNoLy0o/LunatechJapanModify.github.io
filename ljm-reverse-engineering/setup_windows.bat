@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
echo ============================================================
echo  LJM Reverse Engineering - 環境構築 (Windows 11 / Python 3.12 64bit)
echo ============================================================
where py >nul 2>nul
if errorlevel 1 (
  echo [エラー] Python ランチャー ^(py^) が見つかりません。
  echo https://www.python.org/downloads/windows/ から Python 3.12 ^(64-bit^) をインストールしてください。
  pause
  exit /b 1
)
py -3.12 -c "import sys; assert sys.maxsize > 2**32, '64bit required'"
if errorlevel 1 (
  echo [エラー] Python 3.12 ^(64-bit^) が見つかりません。インストールしてから再実行してください。
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 仮想環境 .venv を作成しています...
  py -3.12 -m venv .venv
  if errorlevel 1 ( echo [エラー] 仮想環境の作成に失敗しました。 & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [エラー] ライブラリのインストールに失敗しました。ネットワーク接続を確認してください。 & pause & exit /b 1 )
if /i "%~1"=="--with-open3d" python -m pip install -r requirements-optional.txt
if /i "%~1"=="--dev" python -m pip install -r requirements-dev.txt
if /i "%~2"=="--with-open3d" python -m pip install -r requirements-optional.txt
if /i "%~2"=="--dev" python -m pip install -r requirements-dev.txt
echo.
echo 完了しました。run_ljm_re.bat で起動できます。
echo （テスト: setup_windows.bat --dev のあと run_tests.bat / EXE作成: build_exe.bat）
pause
