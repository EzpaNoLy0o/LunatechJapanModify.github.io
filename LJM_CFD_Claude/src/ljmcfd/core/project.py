"""プロジェクトファイル (JSON) の読み書き。

プロジェクトは 1 つのフォルダで、その中の ``project.ljmcfd.json`` が設定を持つ。
メッシュ・ケース・VTK 結果などの大きなデータは JSON に埋め込まず、同じフォルダ内の
外部ファイルとして保存し、JSON からは相対パスで参照する (Stage 2 以降で追加)。

将来のステージで増える設定は ``sections`` (tunnel / mesh / solver / post) に入る。
未知のキーは読み込み時に保持し、保存時にそのまま書き戻す (前方互換)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .pose import Pose

PROJECT_FILE = "project.ljmcfd.json"
FORMAT_VERSION = 1
SECTION_NAMES = ("tunnel", "mesh", "solver", "post")


class ProjectError(RuntimeError):
    pass


@dataclass
class PartRecord:
    id: str
    name: str
    source_path: str                 # 保存時点の絶対パス
    source_rel: str | None           # プロジェクトフォルダからの相対パス (移動に強くするため)
    source_index: int                # 元ファイル内でのパーツ番号
    cache_key: str
    unit: str
    unit_source: str
    pose: Pose = field(default_factory=Pose)
    visible: bool = True
    color: tuple[float, float, float] = (0.74, 0.77, 0.82)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "source_path": self.source_path, "source_rel": self.source_rel,
            "source_index": self.source_index, "cache_key": self.cache_key,
            "unit": self.unit, "unit_source": self.unit_source,
            "pose": self.pose.to_dict(), "visible": self.visible, "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PartRecord":
        return cls(
            id=str(data["id"]), name=str(data.get("name", "")),
            source_path=str(data["source_path"]), source_rel=data.get("source_rel"),
            source_index=int(data.get("source_index", 0)), cache_key=str(data.get("cache_key", "")),
            unit=str(data.get("unit", "mm")), unit_source=str(data.get("unit_source", "user")),
            pose=Pose.from_dict(data.get("pose")), visible=bool(data.get("visible", True)),
            color=tuple(float(c) for c in data.get("color", (0.74, 0.77, 0.82)))[:3],
        )

    def resolve_source(self, project_dir: Path | None) -> Path | None:
        """元 CAD ファイルの現在位置。絶対パス → 相対パスの順に探し、無ければ ``None``。"""
        absolute = Path(self.source_path)
        if absolute.is_file():
            return absolute
        if project_dir is not None and self.source_rel:
            candidate = (project_dir / self.source_rel).resolve()
            if candidate.is_file():
                return candidate
        return None


@dataclass
class Project:
    name: str = "Untitled"
    display_unit: str = "mm"
    parts: list[PartRecord] = field(default_factory=list)
    vehicle_pose: Pose = field(default_factory=Pose)
    view: dict = field(default_factory=dict)          # 背景・透明度・カメラなどの表示設定
    sections: dict = field(default_factory=lambda: {name: {} for name in SECTION_NAMES})
    extra: dict = field(default_factory=dict)         # 未知のトップレベルキー (前方互換)

    def to_dict(self) -> dict:
        data = dict(self.extra)
        data.update({
            "format": "ljmcfd-project",
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "display_unit": self.display_unit,
            "vehicle_pose": self.vehicle_pose.to_dict(),
            "parts": [p.to_dict() for p in self.parts],
            "view": self.view,
        })
        data.update(self.sections)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        if data.get("format") != "ljmcfd-project":
            raise ProjectError("LJM CFD のプロジェクトファイルではありません")
        version = int(data.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise ProjectError(
                f"このプロジェクトは新しい形式 (v{version}) で保存されています。アプリを更新してください。"
            )
        known = {"format", "format_version", "name", "display_unit", "vehicle_pose", "parts", "view", *SECTION_NAMES}
        return cls(
            name=str(data.get("name", "Untitled")),
            display_unit=str(data.get("display_unit", "mm")),
            parts=[PartRecord.from_dict(p) for p in data.get("parts", [])],
            vehicle_pose=Pose.from_dict(data.get("vehicle_pose")),
            view=dict(data.get("view", {})),
            sections={name: dict(data.get(name, {})) for name in SECTION_NAMES},
            extra={k: v for k, v in data.items() if k not in known},
        )


def project_file(project_dir: str | Path) -> Path:
    return Path(project_dir) / PROJECT_FILE


def relative_or_none(target: str | Path, base: str | Path) -> str | None:
    """``base`` からの相対パス (POSIX 表記)。別ドライブ等で作れなければ ``None``。"""
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:
        return None


def save_project(project: Project, project_dir: str | Path) -> Path:
    """プロジェクトを保存する。書き込み途中で壊れないよう一時ファイル経由で置換する。"""
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    for record in project.parts:
        record.source_rel = relative_or_none(record.source_path, project_dir)
    target = project_file(project_dir)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_project(path: str | Path) -> tuple[Project, Path]:
    """プロジェクトを読み込む。``path`` はフォルダでも JSON ファイルでもよい。"""
    path = Path(path)
    file = project_file(path) if path.is_dir() else path
    if not file.is_file():
        raise ProjectError(f"プロジェクトファイルが見つかりません: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProjectError(f"プロジェクトファイルが壊れています: {exc}") from exc
    return Project.from_dict(data), file.parent
