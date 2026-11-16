"""OpenFOAM の出力ファイル (荷重・残差・流量・y+・ログ) の解析。

書式は OpenFOAM Foundation 14 の実物の出力で確認したもの:

* ``postProcessing/<forces 名>/<開始時刻>/forces.dat``
  ``時刻  ((圧力力 xyz) (粘性力 xyz)) ((圧力モーメント xyz) (粘性モーメント xyz))``
* ``postProcessing/residuals/<開始時刻>/residuals.dat``  1 行目のヘッダに列名、未計算は ``N/A``
* ``postProcessing/<流量名>/<開始時刻>/surfaceFieldValue.dat``  ``時刻  sum(phi)``
* ``postProcessing/yPlus/<開始時刻>/yPlus.dat``  ``時刻  パッチ  最小  最大  平均``

続きから計算すると開始時刻ごとにフォルダが増えるので、すべて読んで時刻順につなぐ。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf", re.I)
RESIDUAL_FIELDS = ("p", "Ux", "Uy", "Uz", "k", "omega")

# 致命的エラーを示す文字列。起動時のバナーには "sigFpe : Enabling floating point exception trapping" という
# 通常の行があるので、"sigFpe" や "floating point exception" (小文字) を単独で探してはいけない。
_FATAL_PATTERNS = (
    "FOAM FATAL ERROR", "FOAM FATAL IO ERROR", "Floating point exception", "sigFpe::sigHandler",
    "Segmentation fault", "sigSegv::sigHandler", "MPI_ABORT",
)


def _time_dirs(folder: Path) -> list[Path]:
    """開始時刻フォルダを時刻の昇順で返す。"""
    if not folder.is_dir():
        return []
    dirs = []
    for child in folder.iterdir():
        try:
            dirs.append((float(child.name), child))
        except ValueError:
            continue
    return [path for _, path in sorted(dirs)]


def parse_forces_line(line: str) -> list[float] | None:
    """forces.dat の 1 行を [時刻, 圧力力 xyz, 粘性力 xyz, 圧力モーメント xyz, 粘性モーメント xyz] にする。"""
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    values = _NUMBER.findall(line)
    if len(values) < 13:
        return None
    return [float(v) for v in values[:13]]


def _merge_by_time(blocks: list[np.ndarray], width: int) -> np.ndarray:
    """複数回の実行分をつなぐ。同じ時刻が重なる場合は後の実行の値を採用する。"""
    blocks = [b for b in blocks if len(b)]
    if not blocks:
        return np.empty((0, width))
    data = np.vstack(blocks)
    _, last = np.unique(data[::-1, 0], return_index=True)     # 逆順で最初 = 元の順で最後
    return data[len(data) - 1 - last]


def read_forces(case_dir: str | Path, name: str = "forces") -> np.ndarray:
    """荷重履歴 (N, 13)。列は ``parse_forces_line`` と同じ。時刻 0 (初期場) の行は除く。"""
    blocks = []
    for folder in _time_dirs(Path(case_dir) / "postProcessing" / name):
        file = folder / "forces.dat"
        if file.is_file():
            rows = [r for r in map(parse_forces_line, file.read_text(errors="replace").splitlines()) if r]
            blocks.append(np.array(rows, dtype=float).reshape(-1, 13))
    data = _merge_by_time(blocks, 13)
    return data[data[:, 0] > 0]


def parse_residuals_text(text: str) -> tuple[list[str], np.ndarray]:
    """residuals.dat の内容を (列名, 行列) にする。行列の 1 列目は時刻、未計算は NaN。"""
    names: list[str] = []
    rows = []
    for line in text.splitlines():
        if line.startswith("#"):
            fields = line[1:].split()
            if fields and fields[0] == "Time":
                names = fields[1:]
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append([float(parts[0])] + [float("nan") if p == "N/A" else float(p) for p in parts[1:]])
        except ValueError:
            continue
    width = len(names) + 1
    rows = [r for r in rows if len(r) == width]
    return names, np.array(rows, dtype=float).reshape(-1, width)


def read_residuals(case_dir: str | Path, name: str = "residuals") -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    blocks = []
    for folder in _time_dirs(Path(case_dir) / "postProcessing" / name):
        file = folder / "residuals.dat"
        if file.is_file():
            found, block = parse_residuals_text(file.read_text(errors="replace"))
            if found:
                names = found
                blocks.append(block)
    data = _merge_by_time(blocks, len(names) + 1)
    return names, data[data[:, 0] > 0] if len(data) else data


def read_flow_rate(case_dir: str | Path, name: str) -> np.ndarray:
    """パッチを通る体積流量の履歴 (N, 2) = [時刻, 流量 m³/s]。流出が正。"""
    blocks = []
    for folder in _time_dirs(Path(case_dir) / "postProcessing" / name):
        file = folder / "surfaceFieldValue.dat"
        if not file.is_file():
            continue
        rows = []
        for line in file.read_text(errors="replace").splitlines():
            parts = line.split()
            if line.startswith("#") or len(parts) < 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        blocks.append(np.array(rows, dtype=float).reshape(-1, 2))
    return _merge_by_time(blocks, 2)


def read_yplus(case_dir: str | Path, name: str = "yPlus") -> dict[str, dict[str, float]]:
    """最新時刻の y+ 統計。パッチ名 → {"min", "max", "average"}。"""
    latest: dict[str, tuple[float, dict]] = {}
    for folder in _time_dirs(Path(case_dir) / "postProcessing" / name):
        file = folder / "yPlus.dat"
        if not file.is_file():
            continue
        for line in file.read_text(errors="replace").splitlines():
            parts = line.split()
            if line.startswith("#") or len(parts) < 5:
                continue
            try:
                time, lo, hi, avg = float(parts[0]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            if parts[1] not in latest or time >= latest[parts[1]][0]:
                latest[parts[1]] = (time, {"min": lo, "max": hi, "average": avg})
    return {patch: stats for patch, (_, stats) in latest.items()}


def find_fatal_error(log_text: str) -> str | None:
    """ログから致命的エラーの記述を抜き出す。無ければ ``None``。"""
    for pattern in _FATAL_PATTERNS:
        index = log_text.find(pattern)
        if index >= 0:
            start = log_text.rfind("\n", 0, index) + 1
            return log_text[start:start + 1500].strip()
    return None


def reached_residual_control(log_text: str) -> bool:
    """残差が打ち切り基準を満たして SIMPLE が自ら終了したか。"""
    return "reached convergence criteria" in log_text or "solution converged" in log_text


def stopped_by_file(log_text: str) -> bool:
    """停止要求 (stop ファイル) を検知して終了したか。ログの ``stopAtFile(...): stopping now`` で判定する。"""
    return re.search(r"stopAtFile\([^)]*\):\s*stopping", log_text) is not None


def last_continuity_error(log_text: str) -> dict | None:
    """ログ最後の ``time step continuity errors`` 行。"""
    index = log_text.rfind("time step continuity errors")
    if index < 0:
        return None
    line = log_text[index:log_text.find("\n", index)]
    values = _NUMBER.findall(line)
    if len(values) < 3:
        return None
    return {"sum_local": float(values[0]), "global": float(values[1]), "cumulative": float(values[2])}


class FileTail:
    """追記されていくテキストファイルから、新しく完成した行だけを順に取り出す。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._pending = ""

    def read_new_lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return []
        if not chunk:
            return []
        text = self._pending + chunk.decode("utf-8", "replace")
        lines = text.split("\n")
        self._pending = lines.pop()       # 書きかけの最終行は次回へ持ち越す
        # Windows でテキストとして書かれたファイルは改行が CRLF になるので、行末の \r を取り除く
        return [line[:-1] if line.endswith("\r") else line for line in lines]
