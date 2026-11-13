"""ログ設定。ファイル (ローテーション) と標準エラーへ出力する。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .paths import log_dir

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("ljmcfd")
    if root.handlers:
        return root
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    try:
        file_handler = RotatingFileHandler(
            log_dir() / "ljmcfd.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        if sys.stderr is not None:
            sys.stderr.write(f"ログファイルを開けません: {exc}\n")
    # pythonw.exe (コンソールなし) で起動した場合 sys.stderr は None になる
    if sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    return root


def install_excepthook() -> None:
    """未捕捉例外を握りつぶさず、必ずログへ記録する。"""
    logger = logging.getLogger("ljmcfd")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("未捕捉の例外", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook
