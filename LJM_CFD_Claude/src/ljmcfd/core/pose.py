"""姿勢 (位置 + Yaw/Pitch/Roll) と同次変換行列。

座標系 (アプリ全体で共通):

* +X : 主流 (風) の進行方向。車両は通常ノーズを -X 側へ向けて置く。
* +Y : 車両の横方向。
* +Z : 鉛直上向き。地面は Z = 0。

回転は右手系・度単位で、合成順は ``R = Rz(yaw) · Ry(pitch) · Rx(roll)``
(固定軸で Roll → Pitch → Yaw の順に適用)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Pose:
    """平行移動 [m] と回転 [deg]。"""

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def is_identity(self) -> bool:
        return not any((self.tx, self.ty, self.tz, self.yaw, self.pitch, self.roll))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Pose":
        data = data or {}
        return cls(**{k: float(data.get(k, 0.0)) for k in cls.__dataclass_fields__})

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.tz], dtype=float)


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """``Rz(yaw) · Ry(pitch) · Rx(roll)`` の 3x3 回転行列。"""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def pose_matrix(pose: Pose, pivot=(0.0, 0.0, 0.0), scale: float = 1.0) -> np.ndarray:
    """姿勢の 4x4 同次変換行列を返す。

    ``x_world = t + pivot + R · (scale · x_local − pivot)``

    ``pivot`` は回転中心 (スケール適用後、メートル)。``scale`` はファイル単位 → m。
    """
    rot = rotation_matrix(pose.yaw, pose.pitch, pose.roll)
    pivot = np.asarray(pivot, dtype=float)
    m = np.eye(4)
    m[:3, :3] = rot * scale
    m[:3, 3] = pose.translation + pivot - rot @ pivot
    return m


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, 3) の点群へ 4x4 行列を適用する。"""
    pts = np.asarray(points, dtype=float)
    return pts @ matrix[:3, :3].T + matrix[:3, 3]


def bounds_of_points(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """VTK 形式 (xmin, xmax, ymin, ymax, zmin, zmax) の境界を返す。"""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        raise ValueError("点群が空のため境界を計算できません")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def merge_bounds(bounds_list) -> tuple[float, float, float, float, float, float] | None:
    """複数の境界を包含する境界。空なら ``None``。"""
    items = [b for b in bounds_list if b is not None]
    if not items:
        return None
    arr = np.asarray(items, dtype=float)
    return (
        arr[:, 0].min(), arr[:, 1].max(),
        arr[:, 2].min(), arr[:, 3].max(),
        arr[:, 4].min(), arr[:, 5].max(),
    )
