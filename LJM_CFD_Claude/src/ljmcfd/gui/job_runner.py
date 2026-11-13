"""ジョブ (子プロセス) を QProcess で実行し、進捗・結果を Qt シグナルへ変換する。

GUI スレッドはブロックされない。ジョブは 1 本ずつ順番に実行する (CAD 変換や
メッシュ生成は CPU を使い切るため、同時実行しても速くならない)。
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ..core import paths
from ..core.jobs import job_command, job_environment, parse_protocol_line

logger = logging.getLogger("ljmcfd.jobs")


class Job(QObject):
    """1 回のジョブ実行。"""

    progress = Signal(float, str)
    log = Signal(str)
    succeeded = Signal(dict)
    failed = Signal(str, str)      # メッセージ, 詳細 (トレースバック等)
    finished = Signal()            # 成否にかかわらず最後に 1 回

    def __init__(self, name: str, args: dict, label: str = "", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.args = args
        self.label = label or name
        self._process: QProcess | None = None
        self._buffer = b""
        self._stderr = b""
        self._result: dict | None = None
        self._error: tuple[str, str] | None = None
        self._cancelled = False
        self._args_path = paths.temp_dir() / f"job_{uuid.uuid4().hex}.json"

    def start(self) -> None:
        self._args_path.write_text(json.dumps(self.args, ensure_ascii=False), encoding="utf-8")
        command = job_command(self.name, str(self._args_path))
        process = QProcess(self)
        env = QProcessEnvironment()
        for key, value in job_environment().items():
            env.insert(key, value)
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        logger.info("ジョブ開始: %s", self.label)
        process.start(command[0], command[1:])

    def cancel(self) -> None:
        self._cancelled = True
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    # ------------------------------------------------------------------ 内部
    def _read_stdout(self) -> None:
        self._buffer += bytes(self._process.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            self._handle_line(raw.decode("utf-8", "replace").rstrip("\r"))

    def _read_stderr(self) -> None:
        self._stderr += bytes(self._process.readAllStandardError())

    def _handle_line(self, line: str) -> None:
        if not line.strip():
            return
        event = parse_protocol_line(line)
        if event is None:
            self.log.emit(line)
            return
        kind = event["type"]
        if kind == "progress":
            self.progress.emit(float(event.get("value", 0.0)), str(event.get("message", "")))
        elif kind == "log":
            self.log.emit(str(event.get("message", "")))
        elif kind == "result":
            self._result = event.get("data") or {}
        elif kind == "error":
            self._error = (str(event.get("message", "")), str(event.get("detail", "")))

    def _on_error(self, error) -> None:
        if error == QProcess.FailedToStart:
            self._error = ("ジョブ用プロセスを起動できませんでした", self._process.errorString())
            self._finish()

    def _on_finished(self, exit_code: int, _status) -> None:
        self._read_stdout()
        if self._buffer:
            self._handle_line(self._buffer.decode("utf-8", "replace"))
            self._buffer = b""
        self._read_stderr()
        if self._cancelled:
            self._error = ("キャンセルされました", "")
        elif self._result is None and self._error is None:
            detail = self._stderr.decode("utf-8", "replace")
            self._error = (f"ジョブが結果を返さずに終了しました (終了コード {exit_code})", detail)
        self._finish()

    def _finish(self) -> None:
        if self._process is None:
            return
        self._process = None
        try:
            self._args_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("一時ファイルを削除できません: %s", exc)
        if self._error is not None:
            message, detail = self._error
            logger.error("ジョブ失敗: %s — %s%s", self.label, message, f"\n{detail}" if detail else "")
            self.failed.emit(message, detail)
        else:
            logger.info("ジョブ完了: %s", self.label)
            self.succeeded.emit(self._result or {})
        self.finished.emit()


class JobManager(QObject):
    """ジョブを順番に実行するキュー。"""

    busyChanged = Signal(bool)
    progress = Signal(str, float, str)   # ラベル, 進捗 0-1, メッセージ
    logLine = Signal(str, str)           # レベル ("info" | "error"), テキスト

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: deque[Job] = deque()
        self._current: Job | None = None

    def submit(
        self,
        name: str,
        args: dict,
        label: str = "",
        on_success: Callable[[dict], None] | None = None,
        on_failure: Callable[[str, str], None] | None = None,
    ) -> Job:
        job = Job(name, args, label, self)
        job.progress.connect(lambda value, msg, j=job: self.progress.emit(j.label, value, msg))
        job.log.connect(lambda text: self.logLine.emit("info", text))
        job.failed.connect(lambda msg, detail, j=job: self.logLine.emit("error", f"{j.label}: {msg}"))
        if on_success is not None:
            job.succeeded.connect(on_success)
        if on_failure is not None:
            job.failed.connect(on_failure)
        job.finished.connect(lambda j=job: self._on_job_finished(j))
        self._queue.append(job)
        self._pump()
        return job

    def is_busy(self) -> bool:
        return self._current is not None or bool(self._queue)

    def cancel_all(self) -> None:
        self._queue.clear()
        if self._current is not None:
            self._current.cancel()

    def _pump(self) -> None:
        if self._current is not None or not self._queue:
            return
        self._current = self._queue.popleft()
        self.busyChanged.emit(True)
        self._current.start()

    def _on_job_finished(self, job: Job) -> None:
        if self._current is job:
            self._current = None
        job.deleteLater()
        if self._queue:
            self._pump()
        else:
            self.busyChanged.emit(False)
