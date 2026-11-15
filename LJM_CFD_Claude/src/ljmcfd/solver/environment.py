"""OpenFOAM の実行環境。

* Windows: WSL2 のディストリビューション内の OpenFOAM を ``wsl.exe`` 経由で実行する。
* Linux:   その場の OpenFOAM を直接実行する (開発・テスト用)。

コマンドは必ず「ケースフォルダに書いたシェルスクリプトを bash で実行する」形にする。
Windows → wsl.exe → bash と引数を渡す途中で引用符や ``$`` が壊れるのを避けるため。
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.jobs import JobContext, JobError
from ..core.paths import is_safe_case_path, windows_to_wsl

KIND_NATIVE = "native"
KIND_WSL = "wsl"
DEFAULT_BASHRC = "/opt/openfoam14/etc/bashrc"
DEFAULT_DISTRO = "Ubuntu-24.04"
REQUIRED_VERSION = "14"
STEP_SCRIPT = ".ljmcfd_step.sh"


@dataclass
class FoamEnvironment:
    kind: str = KIND_NATIVE
    bashrc: str = DEFAULT_BASHRC
    distro: str = DEFAULT_DISTRO

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "FoamEnvironment":
        data = data or {}
        env = default_environment()
        return cls(kind=str(data.get("kind", env.kind)), bashrc=str(data.get("bashrc", env.bashrc)),
                   distro=str(data.get("distro", env.distro)))

    # ------------------------------------------------------------------ パス
    def path(self, path: str | os.PathLike) -> str:
        """ホスト側のパスを、OpenFOAM が動く側 (WSL / Linux) から見えるパスへ変換する。"""
        if self.kind == KIND_WSL:
            return windows_to_wsl(path)
        return str(path)

    def check_case_path(self, path: str | os.PathLike) -> None:
        """OpenFOAM のケースを置けるパスか。空白や日本語を含むとスクリプトやツールが壊れる。"""
        if not is_safe_case_path(self.path(path)):
            raise JobError(
                f"解析フォルダのパスに空白・日本語などが含まれています: {path}\n"
                "プロジェクトを C:\\LJM_CFD_Claude\\projects のような英数字だけのフォルダへ保存してください。"
            )

    # ------------------------------------------------------------------ 実行
    def script_argv(self, script_path: str | os.PathLike) -> list[str]:
        """シェルスクリプトを OpenFOAM 環境で実行するためのコマンドライン。"""
        if self.kind == KIND_WSL:
            return ["wsl.exe", "-d", self.distro, "--", "bash", self.path(script_path)]
        return ["bash", str(script_path)]

    def write_script(self, directory: str | os.PathLike, body: str, name: str = STEP_SCRIPT) -> Path:
        """OpenFOAM 環境を読み込んでから ``body`` を実行するスクリプトを書く (改行は必ず LF)。"""
        directory = Path(directory)
        text = (
            "#!/bin/bash\n"
            f"if [ ! -f {shlex.quote(self.bashrc)} ]; then echo '@@LJM_NO_OPENFOAM'; exit 97; fi\n"
            "set +e\n"
            f"source {shlex.quote(self.bashrc)} >/dev/null 2>&1\n"
            f"cd {shlex.quote(self.path(directory))} || exit 96\n"
            f"{body}\n"
        )
        path = directory / name
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return path

    def popen(self, directory: str | os.PathLike, body: str, name: str = STEP_SCRIPT, **kwargs) -> subprocess.Popen:
        """``body`` をケースフォルダで実行するプロセスを起動する。"""
        script = self.write_script(directory, body, name)
        flags = {}
        if sys.platform == "win32":
            flags["creationflags"] = subprocess.CREATE_NO_WINDOW     # コンソール窓を出さない
        try:
            return subprocess.Popen(
                self.script_argv(script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, **flags, **kwargs,
            )
        except FileNotFoundError as exc:
            if self.kind == KIND_WSL:
                raise JobError(
                    "wsl.exe が見つかりません。WSL2 がインストールされていません。"
                    "README の「OpenFOAM のセットアップ」を参照してください。"
                ) from exc
            raise JobError("bash が見つかりません。") from exc

    def run(self, directory: str | os.PathLike, body: str, timeout: float | None = None) -> tuple[int, str]:
        """``body`` を実行し、(終了コード, 出力) を返す。"""
        process = self.popen(directory, body)
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            return 124, decode_output(output) + "\n(タイムアウト)"
        return process.returncode, decode_output(output)

    def probe(self, directory: str | os.PathLike) -> dict:
        """OpenFOAM のバージョン・コア数などを調べる。使えない場合は理由つきで ``JobError``。"""
        body = (
            'echo "version=$WM_PROJECT_VERSION"\n'
            'echo "foamRun=$(command -v foamRun)"\n'
            'echo "mpirun=$(command -v mpirun)"\n'
            'echo "logical=$(nproc)"\n'
            'echo "physical=$(lscpu -p=core,socket 2>/dev/null | grep -v "^#" | sort -u | wc -l)"\n'
            'echo "uid=$(id -u)"\n'
        )
        code, output = self.run(directory, body, timeout=120)
        where = f"WSL ディストリビューション「{self.distro}」" if self.kind == KIND_WSL else "この環境"
        if "@@LJM_NO_OPENFOAM" in output or code == 97:
            raise JobError(
                f"{where}に OpenFOAM が見つかりません ({self.bashrc})。"
                "OpenFOAM Foundation 14 をインストールしてください (README の「OpenFOAM のセットアップ」)。"
            )
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if code != 0 or "version" not in values:
            detail = output.strip()[-600:]
            if self.kind == KIND_WSL:
                raise JobError(
                    f"{where}を起動できません。ディストリビューション名と WSL2 の状態を確認してください。\n{detail}"
                )
            raise JobError(f"OpenFOAM 環境を起動できません。\n{detail}")
        if not values.get("foamRun"):
            raise JobError(f"{where}に foamRun が見つかりません。OpenFOAM Foundation 14 が正しく入っているか確認してください。")

        def number(key: str, default: int) -> int:
            try:
                return max(int(values.get(key, default)), 1)
            except ValueError:
                return default

        logical = number("logical", 1)
        return {
            "kind": self.kind, "distro": self.distro if self.kind == KIND_WSL else None, "bashrc": self.bashrc,
            "version": values["version"].strip(),
            "version_ok": values["version"].strip() == REQUIRED_VERSION,
            "has_mpirun": bool(values.get("mpirun")),
            "logical_cores": logical,
            "physical_cores": min(number("physical", logical), logical),
            "is_root": values.get("uid", "").strip() == "0",
        }


def decode_output(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", "replace").replace("\r\n", "\n")


def default_environment() -> FoamEnvironment:
    """既定の実行環境。Windows では WSL、それ以外ではその場の OpenFOAM。"""
    if sys.platform == "win32":
        return FoamEnvironment(kind=KIND_WSL)
    return FoamEnvironment(kind=KIND_NATIVE)


def native_openfoam_available(bashrc: str = DEFAULT_BASHRC) -> bool:
    """この (Linux) 環境に OpenFOAM があるか。テストのスキップ判定に使う。"""
    return sys.platform != "win32" and Path(bashrc).is_file() and shutil.which("bash") is not None


def check_environment_job(args: dict, ctx: JobContext | None = None) -> dict:
    """OpenFOAM 環境を確認するジョブ。args: {"environment": {...}, "work_dir": 作業フォルダ}"""
    ctx = ctx or JobContext()
    env = FoamEnvironment.from_dict(args.get("environment"))
    work_dir = Path(args["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx.progress(0.3, "OpenFOAM 環境を確認中…")
    try:
        info = env.probe(work_dir)
    finally:
        (work_dir / STEP_SCRIPT).unlink(missing_ok=True)
    ctx.progress(1.0, "確認完了")
    return info
