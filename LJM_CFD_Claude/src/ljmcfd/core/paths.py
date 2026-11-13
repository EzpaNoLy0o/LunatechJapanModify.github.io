"""Windows ⇔ WSL2 のパス変換とアプリ用ディレクトリ。

GUI は Windows 側、OpenFOAM は WSL2 側で動くため、両者の間でやり取りする
パスは必ずこのモジュールを経由して変換する (文字列連結で作らない)。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$", re.S)
_WSL_UNC_RE = re.compile(r"^[\\/]{2}(?:wsl\$|wsl\.localhost)[\\/]+([^\\/]+)[\\/]*(.*)$", re.I | re.S)
_MNT_RE = re.compile(r"^/mnt/([a-z])(?:/(.*))?$", re.S)

#: OpenFOAM のケースパスとして安全な文字 (空白・全角・記号を含むとスクリプトが壊れやすい)
_SAFE_CASE_RE = re.compile(r"^[A-Za-z0-9_./:\\\-]+$")


def windows_to_wsl(path: str | os.PathLike) -> str:
    """Windows パスを WSL 内から見えるパスへ変換する。

    * ``C:\\LJM_CFD_Claude\\run`` → ``/mnt/c/LJM_CFD_Claude/run``
    * ``\\\\wsl.localhost\\Ubuntu-24.04\\home\\u\\case`` → ``/home/u/case``
    """
    text = str(path)
    m = _WSL_UNC_RE.match(text)
    if m:
        rest = m.group(2).replace("\\", "/").strip("/")
        return "/" + rest
    m = _DRIVE_RE.match(text)
    if m:
        rest = m.group(2).replace("\\", "/").strip("/")
        drive = m.group(1).lower()
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    if text.startswith("/"):
        return text  # 既に POSIX パス
    raise ValueError(f"WSL パスへ変換できません (絶対パスではありません): {text!r}")


def wsl_to_windows(path: str, distro: str = "Ubuntu-24.04") -> str:
    """WSL 内のパスを Windows から開けるパスへ変換する。

    * ``/mnt/c/LJM_CFD_Claude`` → ``C:\\LJM_CFD_Claude``
    * ``/home/u/case`` → ``\\\\wsl.localhost\\<distro>\\home\\u\\case``
    """
    if not path.startswith("/"):
        raise ValueError(f"WSL の絶対パスではありません: {path!r}")
    m = _MNT_RE.match(path)
    if m:
        rest = (m.group(2) or "").strip("/")
        return str(PureWindowsPath(f"{m.group(1).upper()}:\\", *PurePosixPath(rest).parts))
    parts = PurePosixPath(path).parts[1:]
    return "\\\\wsl.localhost\\" + "\\".join((distro, *parts))


def is_safe_case_path(path: str | os.PathLike) -> bool:
    """OpenFOAM ケースの置き場所として安全か (ASCII・空白なし)。"""
    return bool(_SAFE_CASE_RE.match(str(path)))


def is_ascii_path(path: str | os.PathLike) -> bool:
    return str(path).isascii()


def install_root() -> Path:
    """アプリの設置ディレクトリ (ソース実行時はリポジトリルート)。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # src/ljmcfd/core/paths.py → 3 つ上が src、その上がルート
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    """キャッシュ・ログ・一時ファイルを置く作業ディレクトリ。

    既定は設置ディレクトリ直下の ``workspace`` (標準配置では
    ``C:\\LJM_CFD_Claude\\workspace``)。日本語ユーザー名を含みがちな
    ``%LOCALAPPDATA%`` を避け、OpenCASCADE / OpenFOAM へ ASCII パスを渡すため。
    環境変数 ``LJMCFD_HOME`` で変更できる。
    """
    override = os.environ.get("LJMCFD_HOME")
    root = Path(override) if override else install_root() / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cad_cache_root() -> Path:
    path = workspace_root() / "cache" / "cad"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    path = workspace_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_dir() -> Path:
    path = workspace_root() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path
