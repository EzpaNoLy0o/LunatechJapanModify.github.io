"""GUI アプリケーションの起動。"""

from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path


def run_gui(argv: list[str] | None = None) -> int:
    """GUI を起動する。引数に CAD ファイルやプロジェクトを渡すと起動時に開く。"""
    argv = list(argv or [])
    os.environ.setdefault("QT_API", "pyside6")

    from .core import paths
    from .core.logging_setup import install_excepthook, setup_logging

    logger = setup_logging()
    install_excepthook()
    try:
        # VTK / OpenGL などネイティブ層のクラッシュもログへ残す
        crash_log = open(paths.log_dir() / "crash.log", "a", encoding="utf-8")  # noqa: SIM115 - プロセス終了まで保持
        faulthandler.enable(crash_log)
    except OSError as exc:
        logger.warning("クラッシュログを開けません: %s", exc)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from . import APP_EDITION, APP_NAME
    from .core.project import PROJECT_FILE
    from .gui import theme
    from .gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(f"{APP_NAME} {APP_EDITION}")
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)

    window = MainWindow()
    window.show()

    files = [a for a in argv if not a.startswith("-")]
    projects = [f for f in files if Path(f).name == PROJECT_FILE or Path(f).is_dir()]
    cad_files = [f for f in files if f not in projects]

    def open_startup_files() -> None:
        if projects:
            window.open_project(projects[0])
        if cad_files:
            window.import_files(cad_files)

    if files:
        QTimer.singleShot(0, open_startup_files)
    return app.exec()
