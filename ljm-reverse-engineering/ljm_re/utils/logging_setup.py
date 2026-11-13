"""ログ設定（ファイル + GUIログパネル）."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from .paths import log_dir

_configured = False


def setup_logging(level=logging.INFO) -> str:
    global _configured
    path = os.path.join(log_dir(), "ljm_re.log")
    if _configured:
        return path
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5,
                                                  encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)
    logging.captureWarnings(True)
    _configured = True
    return path


class CallbackHandler(logging.Handler):
    """ログレコードをコールバック（GUIのログパネル）へ送る."""

    def __init__(self, callback, level=logging.INFO):
        super().__init__(level)
        self.callback = callback
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.callback(record.levelno, self.format(record))
        except Exception:
            pass
