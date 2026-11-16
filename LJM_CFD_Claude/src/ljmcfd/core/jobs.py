"""重い処理を別プロセスで実行するためのジョブ機構 (Qt 非依存)。

OpenCASCADE / Gmsh / OpenFOAM の呼び出しは GIL を解放しない、あるいは長時間
ブロックするため、GUI スレッドや QThread ではなく **子プロセス** で実行する。

子プロセスは ``python -m ljmcfd --job <name> <args.json>`` で起動され、標準出力へ
``@@LJMJOB@@ {json}`` 形式の行を出して進捗・結果を親へ伝える。それ以外の行
(OpenCASCADE のメッセージ等) は通常ログとして扱われる。

ジョブ関数のシグネチャ: ``func(args: dict, ctx: JobContext) -> dict``
同じ関数をテストからプロセス内で直接呼ぶこともできる。
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

PROTOCOL_PREFIX = "@@LJMJOB@@ "

#: ジョブ名 → "モジュール:関数"。ステージが進むごとに追加する。
JOB_REGISTRY: dict[str, str] = {
    "cad-import": "ljmcfd.cad.import_job:import_cad_job",
    "cad-inspect": "ljmcfd.cad.inspect:inspect_part_job",
    "frontal-area": "ljmcfd.tunnel.frontal_area:frontal_area_job",
    "mesh-generate": "ljmcfd.mesh.gmsh_job:mesh_generate_job",
    "solver-check-env": "ljmcfd.solver.environment:check_environment_job",
    "solver-run": "ljmcfd.solver.run_job:solver_run_job",
}


class JobError(RuntimeError):
    """ユーザーへそのまま見せられる、ジョブの想定内エラー。"""


class JobContext:
    """ジョブ関数から進捗・ログを報告するためのインターフェース。"""

    def progress(self, fraction: float, message: str = "") -> None:  # pragma: no cover - 既定は無処理
        pass

    def log(self, message: str) -> None:  # pragma: no cover
        pass

    def event(self, name: str, data: dict) -> None:  # pragma: no cover
        """進捗以外の逐次データ (残差・荷重の履歴など) を送る。"""


class RecordingContext(JobContext):
    """テスト用: 報告内容をリストに記録する。"""

    def __init__(self) -> None:
        self.progress_events: list[tuple[float, str]] = []
        self.logs: list[str] = []
        self.events: list[tuple[str, dict]] = []

    def progress(self, fraction: float, message: str = "") -> None:
        self.progress_events.append((fraction, message))

    def log(self, message: str) -> None:
        self.logs.append(message)

    def event(self, name: str, data: dict) -> None:
        self.events.append((name, data))


class StdoutContext(JobContext):
    """子プロセス用: プロトコル行として標準出力へ書き出す。"""

    @staticmethod
    def _emit(payload: dict) -> None:
        sys.stdout.write(PROTOCOL_PREFIX + json.dumps(payload, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    def progress(self, fraction: float, message: str = "") -> None:
        self._emit({"type": "progress", "value": max(0.0, min(1.0, float(fraction))), "message": message})

    def log(self, message: str) -> None:
        self._emit({"type": "log", "message": message})

    def event(self, name: str, data: dict) -> None:
        self._emit({"type": "event", "name": name, "data": data})

    def result(self, data: dict) -> None:
        self._emit({"type": "result", "data": data})

    def error(self, message: str, detail: str = "") -> None:
        self._emit({"type": "error", "message": message, "detail": detail})


def job_command(name: str, args_path: str) -> list[str]:
    """ジョブを子プロセスとして起動するコマンドライン。"""
    if getattr(sys, "frozen", False):  # PyInstaller 等で固めた実行ファイル
        return [frozen_worker(), "--job", name, str(args_path)]
    return [worker_python(), "-m", "ljmcfd", "--job", name, str(args_path)]


def frozen_worker(executable: str | None = None) -> str:
    """配布版でジョブを実行する実行ファイル。

    GUI 用の実行ファイルは Windows ではコンソールなしで作るので、標準出力で親と通信するジョブには
    同じフォルダのコンソール版 ``ljmcfd-worker`` を使う。無ければ GUI と同じ実行ファイルで代用する。
    """
    exe = Path(executable or sys.executable)
    worker = exe.with_name("ljmcfd-worker" + exe.suffix)
    return str(worker if worker.is_file() else exe)


def worker_python(executable: str | None = None) -> str:
    """ジョブ用の Python 実行ファイル。

    GUI を ``pythonw.exe`` (コンソールなし) で起動していても、ジョブは標準出力で親と
    通信するため、同じフォルダの ``python.exe`` を使う。
    """
    exe = Path(executable or sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(exe)


def job_environment(base: dict | None = None) -> dict:
    """子プロセス用の環境変数。パッケージを確実に import でき、出力が UTF-8 になるようにする。"""
    env = dict(os.environ if base is None else base)
    package_parent = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH", "")
    if package_parent not in existing.split(os.pathsep):
        env["PYTHONPATH"] = package_parent + (os.pathsep + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def kill_process_tree(pid: int) -> None:
    """プロセスと、その子孫をすべて強制終了する。"""
    if pid <= 0:
        return
    if sys.platform == "win32":
        import subprocess

        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        return
    import signal

    try:
        os.killpg(pid, signal.SIGKILL)       # ジョブは setsid 済みなので pid = プロセスグループ id
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def resolve_job(name: str) -> Callable[[dict, JobContext], dict]:
    try:
        target = JOB_REGISTRY[name]
    except KeyError as exc:
        raise JobError(f"未登録のジョブです: {name}") from exc
    module_name, func_name = target.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def parse_protocol_line(line: str) -> dict | None:
    """プロトコル行なら dict、通常のログ行なら ``None``。"""
    if not line.startswith(PROTOCOL_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROTOCOL_PREFIX):])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and "type" in payload else None


def run_job_cli(name: str, args_path: str) -> int:
    """子プロセスのエントリポイント。終了コードを返す。"""
    if hasattr(os, "setsid"):
        # 自分を先頭とするプロセスグループを作る。GUI がジョブを強制終了するとき、
        # ジョブが起動した子 (メッシャーやソルバー) もまとめて止められるようにするため。
        try:
            os.setsid()
        except OSError:
            pass
    ctx = StdoutContext()
    try:
        with open(args_path, "r", encoding="utf-8") as fh:
            args = json.load(fh)
        result = resolve_job(name)(args, ctx)
        ctx.result(result if result is not None else {})
        return 0
    except JobError as exc:
        ctx.error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - 想定外の例外も必ず親へ報告する
        detail = traceback.format_exc()
        sys.stderr.write(detail)
        sys.stderr.flush()
        ctx.error(f"{type(exc).__name__}: {exc}", detail)
        return 1
