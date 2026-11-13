"""パーツとアセンブリ (車両全体) のデータモデル。Qt 非依存。

変換は 2 段階:

``x_world = A · P_i · (s_i · x_local)``

* ``s_i``  : パーツのファイル単位 → m のスケール
* ``P_i``  : パーツ個別の配置 (回転中心はパーツ自身の境界ボックス中心)
* ``A``    : 車両全体の姿勢 (回転中心は配置後の全体境界ボックスの XY 中心・最下点)

風向は常にグローバル +X。相対風向 (ヨー角など) は ``A`` で車両側を回して与える。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from ..core.pose import Pose, bounds_of_points, merge_bounds, pose_matrix, transform_points
from ..core.units import unit_scale

#: CAD に色が無いパーツへ順番に割り当てる色
DEFAULT_COLORS = [
    (0.74, 0.77, 0.82), (0.36, 0.62, 0.86), (0.90, 0.58, 0.26), (0.47, 0.78, 0.52),
    (0.82, 0.46, 0.52), (0.62, 0.55, 0.84), (0.80, 0.76, 0.42), (0.42, 0.76, 0.78),
]


@dataclass
class Part:
    id: str
    name: str
    source_path: str
    source_index: int
    cache_dir: str
    cache_key: str
    fmt: str
    entry: dict                      # インポートメタ情報のパーツ項目
    file_unit: str | None            # 元ファイルに記録されていた単位 (情報表示用)
    unit: str                        # キャッシュ座標の単位
    unit_source: str                 # "file" | "occt" | "heuristic" | "user"
    color: tuple[float, float, float]
    pose: Pose = field(default_factory=Pose)
    visible: bool = True
    inspection: dict | None = None
    _surface: pv.PolyData | None = field(default=None, repr=False)
    _edges: pv.PolyData | None = field(default=None, repr=False)
    _hull: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ 幾何
    @property
    def scale(self) -> float:
        return unit_scale(self.unit)

    @property
    def surface(self) -> pv.PolyData:
        if self._surface is None:
            self._surface = pv.read(Path(self.cache_dir) / self.entry["surface"])
        return self._surface

    @property
    def edges(self) -> pv.PolyData:
        if self._edges is None:
            self._edges = pv.read(Path(self.cache_dir) / self.entry["edges"])
        return self._edges

    @property
    def local_bounds(self) -> tuple[float, ...]:
        return tuple(self.entry["bounds"])

    @property
    def pivot(self) -> np.ndarray:
        """回転中心 [m]: スケール適用後のローカル境界ボックス中心。"""
        b = self.local_bounds
        return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]) * self.scale

    @property
    def hull_points(self) -> np.ndarray:
        """境界計算用の凸包頂点 (ローカル座標)。全点を毎回変換しないための縮約。"""
        if self._hull is None:
            pts = np.asarray(self.surface.points, dtype=float)
            hull = pts
            if len(pts) > 64:
                try:
                    from scipy.spatial import ConvexHull

                    hull = pts[ConvexHull(pts).vertices]
                except Exception:  # noqa: BLE001 - 平面形状などで凸包が作れない場合は全点を使う
                    hull = pts
            self._hull = hull
        return self._hull

    def base_matrix(self) -> np.ndarray:
        """パーツ配置 (スケール込み) の 4x4 行列。"""
        return pose_matrix(self.pose, self.pivot, self.scale)

    def release_meshes(self) -> None:
        self._surface = None
        self._edges = None
        self._hull = None


class Assembly:
    """読み込まれた全パーツと車両全体の姿勢。"""

    def __init__(self) -> None:
        self.parts: list[Part] = []
        self.pose = Pose()

    # ------------------------------------------------------------------ 構成
    def get(self, part_id: str) -> Part:
        for part in self.parts:
            if part.id == part_id:
                return part
        raise KeyError(part_id)

    def add_from_meta(self, meta: dict) -> list[Part]:
        """インポート結果 (``import_cad_job`` の戻り値) からパーツを追加する。"""
        added = []
        for entry in meta["parts"]:
            color = tuple(entry["color"]) if entry.get("color") else DEFAULT_COLORS[len(self.parts) % len(DEFAULT_COLORS)]
            part = Part(
                id=uuid.uuid4().hex[:12],
                name=entry["name"],
                source_path=meta["source"]["path"],
                source_index=int(entry["index"]),
                cache_dir=meta["cache_dir"],
                cache_key=meta["key"],
                fmt=meta["format"],
                entry=entry,
                file_unit=meta.get("file_unit"),
                unit=meta["geometry_unit"],
                unit_source=meta["unit_source"],
                color=color,
            )
            self.parts.append(part)
            added.append(part)
        return added

    def remove(self, part_id: str) -> Part:
        part = self.get(part_id)
        self.parts.remove(part)
        return part

    def clear(self) -> None:
        self.parts.clear()
        self.pose = Pose()

    # ------------------------------------------------------------------ 変換
    def _bounds(self, parts, matrix_of) -> tuple[float, ...] | None:
        return merge_bounds(
            bounds_of_points(transform_points(matrix_of(p), p.hull_points)) for p in parts
        )

    def base_bounds(self, parts=None) -> tuple[float, ...] | None:
        """車両姿勢を適用する前 (パーツ配置のみ) の境界 [m]。"""
        return self._bounds(self.parts if parts is None else parts, lambda p: p.base_matrix())

    def attitude_pivot(self) -> np.ndarray:
        """車両姿勢の回転中心 [m]: 姿勢適用前の全体境界の XY 中心・最下点。"""
        b = self.base_bounds()
        if b is None:
            return np.zeros(3)
        return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, b[4]])

    def attitude_matrix(self) -> np.ndarray:
        if self.pose.is_identity():
            return np.eye(4)
        return pose_matrix(self.pose, self.attitude_pivot(), 1.0)

    def world_matrix(self, part: Part, attitude: np.ndarray | None = None) -> np.ndarray:
        attitude = self.attitude_matrix() if attitude is None else attitude
        return attitude @ part.base_matrix()

    def world_bounds(self, parts=None, visible_only: bool = False) -> tuple[float, ...] | None:
        """最終的なワールド座標での境界 [m]。"""
        parts = self.parts if parts is None else parts
        if visible_only:
            parts = [p for p in parts if p.visible]
        attitude = self.attitude_matrix()
        return self._bounds(parts, lambda p: self.world_matrix(p, attitude))

    # ------------------------------------------------------------------ 配置補助
    def drop_to_ground(self, clearance: float = 0.0) -> float:
        """車両全体を上下に動かし、最下点を Z = ``clearance`` に合わせる。移動量 [m] を返す。"""
        b = self.world_bounds()
        if b is None:
            return 0.0
        dz = clearance - b[4]
        self.pose.tz += dz
        return dz

    def center_xy(self) -> tuple[float, float]:
        """車両全体の境界の XY 中心を原点へ移動する。移動量 [m] を返す。"""
        b = self.world_bounds()
        if b is None:
            return (0.0, 0.0)
        dx, dy = -(b[0] + b[1]) / 2, -(b[2] + b[3]) / 2
        self.pose.tx += dx
        self.pose.ty += dy
        return (dx, dy)
