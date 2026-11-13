"""長時間処理をワーカースレッドで実行（UIをフリーズさせない）.

計算関数は fn(progress, *args) の形で、Progress.report/check を呼ぶとキャンセル可能になる。
結果の反映（ドキュメント変更・描画）は必ずメインスレッドの on_done で行う。
"""
from __future__ import annotations

import logging
import threading
import time
import traceback

from PySide6 import QtCore

from ..core.progress import CancelledError, Progress

log = logging.getLogger(__name__)


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object, float)
    failed = QtCore.Signal(str, str)
    cancelled = QtCore.Signal()


class _Runnable(QtCore.QRunnable):
    def __init__(self, fn, args, kwargs, signals, cancel_event):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = signals
        self.cancel_event = cancel_event
        self.setAutoDelete(True)

    def run(self):
        prog = Progress(lambda f, m: self.signals.progress.emit(f, m), self.cancel_event)
        t0 = time.perf_counter()
        try:
            res = self.fn(prog, *self.args, **self.kwargs)
            self.signals.finished.emit(res, time.perf_counter() - t0)
        except CancelledError:
            self.signals.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("処理中にエラー: %s\n%s", e, tb)
            self.signals.failed.emit(str(e), tb)


class TaskRunner(QtCore.QObject):
    """1度に1つの重い処理を実行する."""

    started = QtCore.Signal(str)
    progress = QtCore.Signal(float, str)
    ended = QtCore.Signal(str, bool, float)   # title, success, seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(1)
        self._busy = False
        self._cancel = None
        self._title = ""
        self._signals = None

    @property
    def busy(self):
        return self._busy

    def cancel(self):
        if self._cancel is not None:
            self._cancel.set()

    def run(self, title, fn, *args, on_done=None, on_error=None, on_cancel=None, **kwargs) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._title = title
        self._cancel = threading.Event()
        sig = _Signals()
        self._signals = sig  # 参照を保持

        def done(res, secs):
            self._finish(True, secs)
            if on_done:
                try:
                    on_done(res)
                except Exception as e:  # noqa: BLE001
                    log.exception("結果の反映中にエラー")
                    if on_error:
                        on_error(str(e), traceback.format_exc())

        def fail(msg, tb):
            self._finish(False, 0.0)
            if on_error:
                on_error(msg, tb)

        def canc():
            self._finish(False, 0.0)
            if on_cancel:
                on_cancel()

        # ワーカースレッドからの通知はメインスレッドで処理する（QueuedConnection）
        Q = QtCore.Qt.ConnectionType.QueuedConnection
        sig.progress.connect(self.progress, Q)
        sig.finished.connect(done, Q)
        sig.failed.connect(fail, Q)
        sig.cancelled.connect(canc, Q)
        self.started.emit(title)
        self.pool.start(_Runnable(fn, args, kwargs, sig, self._cancel))
        return True

    def _finish(self, ok, secs):
        self._busy = False
        self.ended.emit(self._title, ok, secs)

    def wait(self, msecs: int = 30000) -> bool:
        """テスト用: 処理完了まで待つ（イベントループを回す）."""
        t0 = time.monotonic()
        app = QtCore.QCoreApplication.instance()
        while self._busy and (time.monotonic() - t0) * 1000 < msecs:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.005)
        app.processEvents()
        return not self._busy
