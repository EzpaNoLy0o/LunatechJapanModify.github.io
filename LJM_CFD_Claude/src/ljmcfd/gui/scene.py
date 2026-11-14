"""シーン: アセンブリ + 選択状態 + 表示単位。GUI 各部はこのシグナルで同期する。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..cad.assembly import Assembly, Part
from ..cad.surface import FACE_ID
from ..core.pose import Pose
from ..core.project import PartRecord
from ..mesh.request import analysis_parts, build_mesh_request
from ..mesh.settings import MeshSettings
from ..tunnel import spec as tunnel_spec
from ..tunnel.spec import TunnelSpec


class Scene(QObject):
    partsAdded = Signal(list)           # list[Part]
    partRemoved = Signal(str)           # part id
    partChanged = Signal(str)           # 表示属性 (可視・色・名前・検査結果) の変更
    transformsChanged = Signal()        # 配置・姿勢・単位の変更 (行列が変わる)
    selectionChanged = Signal()
    faceSelectionChanged = Signal()
    displayUnitChanged = Signal(str)
    cleared = Signal()
    modified = Signal()                 # プロジェクトへ保存すべき変更があった
    tunnelChanged = Signal()            # 風洞・流れの条件の変更
    meshSettingsChanged = Signal()
    meshChanged = Signal()              # メッシュ (レポート) の生成・破棄

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.assembly = Assembly()
        self.selected_ids: list[str] = []
        self.selected_face: tuple[str, int] | None = None   # (part id, face id)
        self.display_unit = "mm"
        self.tunnel = TunnelSpec()
        self.mesh_settings = MeshSettings()
        self.mesh_report: dict | None = None    # 生成済みメッシュのレポート (無ければ None)

    # ------------------------------------------------------------------ 参照
    @property
    def parts(self) -> list[Part]:
        return self.assembly.parts

    def selected_parts(self) -> list[Part]:
        return [p for p in self.assembly.parts if p.id in self.selected_ids]

    # ------------------------------------------------------------------ 構成
    def add_import(self, meta: dict, records: list[PartRecord] | None = None) -> list[Part]:
        """インポート結果を追加する。``records`` があればプロジェクト保存時の状態を復元する。"""
        added = self.assembly.add_from_meta(meta)
        if records is not None:
            by_index = {r.source_index: r for r in records}
            kept = []
            for part in added:
                record = by_index.get(part.source_index)
                if record is None:      # 保存時に削除済みだったパーツ
                    self.assembly.parts.remove(part)
                    continue
                part.id = record.id
                part.name = record.name or part.name
                part.unit, part.unit_source = record.unit, record.unit_source
                part.pose = record.pose
                part.visible = record.visible
                part.color = record.color
                kept.append(part)
            added = kept
        if added:
            self.partsAdded.emit(added)
            self.transformsChanged.emit()
            self.modified.emit()
        return added

    def remove_parts(self, part_ids: list[str]) -> None:
        for part_id in list(part_ids):
            self.assembly.remove(part_id)
            if part_id in self.selected_ids:
                self.selected_ids.remove(part_id)
            if self.selected_face and self.selected_face[0] == part_id:
                self.selected_face = None
            self.partRemoved.emit(part_id)
        self.selectionChanged.emit()
        self.faceSelectionChanged.emit()
        self.transformsChanged.emit()
        self.modified.emit()

    def clear(self) -> None:
        self.assembly.clear()
        self.selected_ids = []
        self.selected_face = None
        self.tunnel = TunnelSpec()
        self.mesh_settings = MeshSettings()
        self.mesh_report = None
        self.cleared.emit()
        self.tunnelChanged.emit()
        self.meshSettingsChanged.emit()
        self.meshChanged.emit()
        self.selectionChanged.emit()
        self.faceSelectionChanged.emit()
        self.transformsChanged.emit()

    # ------------------------------------------------------------------ 選択
    def set_selection(self, part_ids: list[str]) -> None:
        valid = {p.id for p in self.assembly.parts}
        new = [pid for pid in dict.fromkeys(part_ids) if pid in valid]
        if new == self.selected_ids:
            return
        self.selected_ids = new
        self.selectionChanged.emit()

    def toggle_selection(self, part_id: str) -> None:
        ids = list(self.selected_ids)
        if part_id in ids:
            ids.remove(part_id)
        else:
            ids.append(part_id)
        self.set_selection(ids)

    def select_face(self, part_id: str | None, face_id: int = -1) -> None:
        new = (part_id, int(face_id)) if part_id is not None and face_id >= 0 else None
        if new == self.selected_face:
            return
        self.selected_face = new
        self.faceSelectionChanged.emit()

    def face_info(self) -> dict | None:
        """選択中の面の情報 (面積 [m²]、平均法線、曲面種別)。"""
        if self.selected_face is None:
            return None
        part_id, face_id = self.selected_face
        part = self.assembly.get(part_id)
        surface = part.surface
        mask = np.asarray(surface.cell_data[FACE_ID]) == face_id
        tris = np.asarray(surface.faces).reshape(-1, 4)[mask, 1:]
        if len(tris) == 0:
            return None
        matrix = self.assembly.world_matrix(part)
        pts = np.asarray(surface.points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
        cross = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
        total = cross.sum(axis=0)
        norm = np.linalg.norm(total)
        face_types = part.entry.get("face_types")
        return {
            "part": part,
            "face_id": face_id,
            "n_triangles": int(len(tris)),
            "area": float(0.5 * np.linalg.norm(cross, axis=1).sum()),
            "normal": (total / norm) if norm > 0 else np.zeros(3),
            "center": pts[np.unique(tris)].mean(axis=0),
            "surface_type": face_types[face_id] if face_types and face_id < len(face_types) else None,
        }

    # ------------------------------------------------------------------ 属性変更
    def set_visible(self, part_ids: list[str], visible: bool) -> None:
        for part_id in part_ids:
            part = self.assembly.get(part_id)
            if part.visible != visible:
                part.visible = visible
                self.partChanged.emit(part_id)
        self.transformsChanged.emit()   # 境界ボックス (表示パーツ基準) の更新
        self.modified.emit()

    def set_part_pose(self, part_ids: list[str], **fields: float) -> None:
        for part_id in part_ids:
            pose = self.assembly.get(part_id).pose
            for key, value in fields.items():
                setattr(pose, key, float(value))
        self.transformsChanged.emit()
        self.modified.emit()

    def reset_part_pose(self, part_ids: list[str]) -> None:
        for part_id in part_ids:
            self.assembly.get(part_id).pose = Pose()
        self.transformsChanged.emit()
        self.modified.emit()

    def set_vehicle_pose(self, **fields: float) -> None:
        for key, value in fields.items():
            setattr(self.assembly.pose, key, float(value))
        self.transformsChanged.emit()
        self.modified.emit()

    def reset_vehicle_pose(self) -> None:
        self.assembly.pose = Pose()
        self.transformsChanged.emit()
        self.modified.emit()

    def drop_to_ground(self) -> float:
        dz = self.assembly.drop_to_ground(0.0)
        self.transformsChanged.emit()
        self.modified.emit()
        return dz

    def center_xy(self) -> tuple[float, float]:
        shift = self.assembly.center_xy()
        self.transformsChanged.emit()
        self.modified.emit()
        return shift

    def set_part_unit(self, part_ids: list[str], unit: str) -> None:
        for part_id in part_ids:
            part = self.assembly.get(part_id)
            if part.unit != unit:
                part.unit = unit
                part.unit_source = "user"
                self.partChanged.emit(part_id)
        self.transformsChanged.emit()
        self.modified.emit()

    def set_inspection(self, part_id: str, result: dict) -> None:
        self.assembly.get(part_id).inspection = result
        self.partChanged.emit(part_id)

    def set_display_unit(self, unit: str) -> None:
        if unit != self.display_unit:
            self.display_unit = unit
            self.displayUnitChanged.emit(unit)
            self.modified.emit()

    # ------------------------------------------------------------------ 保存用
    def to_records(self) -> list[PartRecord]:
        return [
            PartRecord(
                id=p.id, name=p.name, source_path=p.source_path, source_rel=None,
                source_index=p.source_index, cache_key=p.cache_key, unit=p.unit, unit_source=p.unit_source,
                pose=Pose(**p.pose.to_dict()), visible=p.visible, color=tuple(p.color),
            )
            for p in self.assembly.parts
        ]

    # ------------------------------------------------------------------ 風洞・メッシュ
    def analysis_bounds(self) -> tuple[float, ...] | None:
        """解析対象 (表示中のパーツ) のワールド境界 [m]。"""
        return self.assembly.world_bounds(analysis_parts(self.assembly))

    def auto_size_tunnel(self) -> bool:
        """モデル寸法から風洞の余白を既定値にし、メッシュのプリセットも寸法に合わせる。"""
        bounds = self.analysis_bounds()
        if bounds is None:
            return False
        tunnel_spec.apply_mode_defaults(self.tunnel, bounds)
        self.mesh_settings.apply_preset(self.mesh_settings.preset, bounds)
        self.tunnelChanged.emit()
        self.meshSettingsChanged.emit()
        self.modified.emit()
        return True

    def notify_tunnel_changed(self) -> None:
        """``self.tunnel`` を直接書き換えた後に呼ぶ。"""
        self.tunnelChanged.emit()
        self.modified.emit()

    def notify_mesh_settings_changed(self) -> None:
        self.meshSettingsChanged.emit()
        self.modified.emit()

    def set_mesh_report(self, report: dict | None) -> None:
        self.mesh_report = report
        self.meshChanged.emit()

    def tunnel_issues(self) -> list[tuple[str, str]]:
        bounds = self.analysis_bounds()
        if bounds is None:
            return [("error", "解析対象のパーツがありません。CAD を読み込んでください。")]
        return tunnel_spec.validate(self.tunnel, bounds)

    def mesh_request(self, out_dir) -> dict:
        """現在の状態からメッシュ生成ジョブの引数を作る。"""
        return build_mesh_request(self.assembly, self.tunnel, self.mesh_settings, out_dir)

    def mesh_state(self) -> str:
        """メッシュの状態: "none" (未生成) / "current" (最新) / "stale" (設定変更後に未再生成)。"""
        if self.mesh_report is None:
            return "none"
        if self.analysis_bounds() is None:
            return "stale"
        current = self.mesh_request("")["input_hash"]
        return "current" if current == self.mesh_report.get("input_hash") else "stale"
