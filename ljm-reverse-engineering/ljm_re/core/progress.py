"""長時間処理の進捗通知とキャンセル."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class CancelledError(Exception):
    """ユーザーによって処理がキャンセルされた."""


class Progress:
    """計算コアからGUIへ進捗を伝える軽量オブジェクト.

    計算コードは ``progress.report(0.5, "処理中")`` と ``progress.check()`` を呼ぶだけで、
    GUI の有無を意識しない。
    """

    def __init__(self, callback: Optional[Callable[[float, str], None]] = None,
                 cancel_event: Optional[threading.Event] = None,
                 min_interval: float = 0.05):
        self._callback = callback
        self._cancel = cancel_event or threading.Event()
        self._last = 0.0
        self._min_interval = min_interval
        self._base = 0.0
        self._span = 1.0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def check(self):
        if self._cancel.is_set():
            raise CancelledError("処理がキャンセルされました")

    def report(self, fraction: float, message: str = "", force: bool = False):
        self.check()
        if self._callback is None:
            return
        now = time.monotonic()
        if not force and now - self._last < self._min_interval and fraction < 1.0:
            return
        self._last = now
        f = self._base + self._span * max(0.0, min(1.0, float(fraction)))
        self._callback(f, message)

    def sub(self, start: float, end: float) -> "Progress":
        """全体の [start, end] 区間を担当するサブ進捗を作る."""
        p = Progress(self._callback, self._cancel, self._min_interval)
        p._base = self._base + self._span * start
        p._span = self._span * (end - start)
        return p


NULL_PROGRESS = Progress()


def ensure_progress(p: Optional[Progress]) -> Progress:
    return p if p is not None else Progress()
