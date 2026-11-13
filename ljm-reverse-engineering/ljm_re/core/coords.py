"""座標系（フレーム）.

Frame は「ローカル→ワールド」の剛体変換を表す。
R の列ベクトルがローカル X/Y/Z 軸のワールド方向。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (as_vec, euler_xyz_to_matrix, matrix_to_euler_xyz, normalize,
                       line_plane_intersection, three_plane_intersection, orthonormal_basis)


@dataclass
class Frame:
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.origin = as_vec(self.origin)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)

    # --- 軸 -------------------------------------------------------------
    @property
    def x_axis(self):
        return self.R[:, 0]

    @property
    def y_axis(self):
        return self.R[:, 1]

    @property
    def z_axis(self):
        return self.R[:, 2]

    # --- 変換 -----------------------------------------------------------
    def to_local(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (p - self.origin) @ self.R

    def to_world(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.origin + q @ self.R.T

    def dir_to_local(self, d) -> np.ndarray:
        return np.asarray(d, dtype=float) @ self.R

    def dir_to_world(self, d) -> np.ndarray:
        return np.asarray(d, dtype=float) @ self.R.T

    def matrix(self) -> np.ndarray:
        """ローカル→ワールドの4x4行列."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.origin
        return T

    def inverse_matrix(self) -> np.ndarray:
        """ワールド→ローカルの4x4行列."""
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, 3] = -self.R.T @ self.origin
        return T

    @staticmethod
    def from_matrix(T) -> "Frame":
        T = np.asarray(T, dtype=float)
        return Frame(T[:3, 3].copy(), T[:3, :3].copy())

    def euler_xyz(self):
        return matrix_to_euler_xyz(self.R)

    @staticmethod
    def from_euler(origin, rx, ry, rz) -> "Frame":
        return Frame(as_vec(origin), euler_xyz_to_matrix(rx, ry, rz))

    def is_valid(self, tol=1e-6) -> bool:
        return bool(np.allclose(self.R.T @ self.R, np.eye(3), atol=tol)
                    and abs(np.linalg.det(self.R) - 1.0) < tol)

    def to_dict(self):
        return {"origin": self.origin.tolist(), "R": self.R.tolist()}

    @staticmethod
    def from_dict(d) -> "Frame":
        return Frame(np.array(d["origin"], float), np.array(d["R"], float))

    def copy(self) -> "Frame":
        return Frame(self.origin.copy(), self.R.copy())


WORLD = Frame()


def _frame_from_zx(origin, z, x_hint) -> Frame:
    z = normalize(as_vec(z))
    x = as_vec(x_hint) - np.dot(as_vec(x_hint), z) * z
    if np.linalg.norm(x) < 1e-9:
        x, _ = orthonormal_basis(z)
    x = normalize(x)
    y = np.cross(z, x)
    return Frame(as_vec(origin), np.column_stack([x, y, z]))


def frame_from_three_points(origin, x_point, xy_point) -> Frame:
    """3点から座標系: 原点 / X軸方向の点 / XY平面上（Y正側）の点."""
    o = as_vec(origin)
    x = as_vec(x_point) - o
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("原点とX軸の点が一致しています")
    x = normalize(x)
    w = as_vec(xy_point) - o
    z = np.cross(x, w)
    if np.linalg.norm(z) < 1e-9:
        raise ValueError("3点が一直線上にあります")
    z = normalize(z)
    y = np.cross(z, x)
    return Frame(o, np.column_stack([x, y, z]))


def frame_from_plane_axis(plane_origin, plane_normal, axis_point, axis_dir) -> Frame:
    """平面（Z軸=法線）+ 軸（X軸=軸を平面へ投影した方向, 原点=軸と平面の交点）.

    軸が平面に平行な場合は軸上点を平面に投影した点を原点とする。
    """
    n = normalize(as_vec(plane_normal))
    o = line_plane_intersection(axis_point, axis_dir, plane_origin, n)
    d = normalize(as_vec(axis_dir))
    if o is None or abs(np.dot(d, n)) > 0.999:
        # 軸が平面に垂直 → 原点は交点、X は任意
        if o is None:
            ap = as_vec(axis_point)
            o = ap - np.dot(ap - as_vec(plane_origin), n) * n
        if abs(np.dot(d, n)) > 0.999:
            x_hint, _ = orthonormal_basis(n)
            return _frame_from_zx(o, n, x_hint)
    return _frame_from_zx(o, n, d)


def frame_from_321(plane1, plane2, plane3) -> Frame:
    """3-2-1 位置決め: 第1平面=Z, 第2平面法線→X, 3平面の交点=原点.

    plane_i = (origin, normal)
    """
    o = three_plane_intersection([plane1, plane2, plane3])
    if o is None:
        raise ValueError("3平面の交点が定まりません（平行な平面が含まれています）")
    return _frame_from_zx(o, plane1[1], plane2[1])


def frame_from_axis_point(axis_point, axis_dir, x_point, origin_on_axis=True) -> Frame:
    """軸（Z軸）+ 点（X方向）。原点は点を軸に投影した位置."""
    d = normalize(as_vec(axis_dir))
    ap = as_vec(axis_point)
    xp = as_vec(x_point)
    foot = ap + np.dot(xp - ap, d) * d
    o = foot if origin_on_axis else ap
    return _frame_from_zx(o, d, xp - foot)


def frame_from_origin_z_x(origin, z, x) -> Frame:
    return _frame_from_zx(origin, z, x)
