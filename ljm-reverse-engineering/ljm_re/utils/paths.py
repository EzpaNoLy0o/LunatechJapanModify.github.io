"""アプリのデータ/設定/ログの保存場所（Windows: %APPDATA% / %LOCALAPPDATA%）."""
from __future__ import annotations

import os
import sys

APP_DIR_NAME = "LJM_ReverseEngineering"


def config_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, APP_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def log_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, APP_DIR_NAME, "logs")
    else:
        d = os.path.join(os.path.expanduser("~"), ".local", "state", APP_DIR_NAME, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def resource_path(*parts) -> str:
    """PyInstaller でバンドルされた場合も含めたリソースパス."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base, *parts)
