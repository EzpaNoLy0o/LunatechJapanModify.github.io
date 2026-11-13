"""基本的な3D幾何計算ユーティリティ.

すべての長さは mm、角度は特に断りがなければ度(deg)で扱う。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def as_vec(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(3)


def as_points(p) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < EPS):
        raise ValueError("長さゼロのベクトルは正規化できません")
    return v / n


def safe_normalize(v, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        return np.asarray(fallback, dtype=float)
    return v / n


def orthonormal_basis(n) -> tuple[np.ndarray, np.ndarray]:
    """法線 n に直交する正規直交ベクトル (u, v) を返す。u×v = n。"""
    n = normalize(as_vec(n))
    # n と最も平行でない座標軸を基準にする
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(ref, n))
    v = np.cross(n, u)
    return u, v


def angle_between(a, b, undirected: bool = False) -> float:
    """2ベクトル間の角度 [deg]。undirected=True なら 0〜90° (向きを無視)。"""
    a = normalize(as_vec(a))
    b = normalize(as_vec(b))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if undirected:
        c = abs(c)
    return float(np.degrees(np.arccos(c)))


def point_plane_signed_distance(points, origin, normal) -> np.ndarray:
    n = normalize(as_vec(normal))
    return (as_points(points) - as_vec(origin)) @ n


def project_points_to_plane(points, origin, normal) -> np.ndarray:
    n = normalize(as_vec(normal))
    p = as_points(points)
    d = (p - as_vec(origin)) @ n
    return p - np.outer(d, n)


def point_line_distance(points, line_point, line_dir) -> np.ndarray:
    d = normalize(as_vec(line_dir))
    w = as_points(points) - as_vec(line_point)
    t = w @ d
    perp = w - np.outer(t, d)
    return np.linalg.norm(perp, axis=1)


def project_point_on_line(p, line_point, line_dir) -> np.ndarray:
    d = normalize(as_vec(line_dir))
    lp = as_vec(line_point)
    return lp + np.dot(as_vec(p) - lp, d) * d


@dataclass
class LineLineResult:
    distance: float
    point_on_1: np.ndarray
    point_on_2: np.ndarray
    angle_deg: float
    parallel: bool


def line_line_closest(p1, d1, p2, d2, parallel_tol_deg: float = 1e-6) -> LineLineResult:
    """2直線（無限長）の最短距離と最近接点."""
    p1, p2 = as_vec(p1), as_vec(p2)
    d1, d2 = normalize(as_vec(d1)), normalize(as_vec(d2))
    ang = angle_between(d1, d2, undirected=True)
    w0 = p1 - p2
    b = float(np.dot(d1, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = 1.0 - b * b
    if ang < parallel_tol_deg or denom < 1e-14:
        # 平行: p1 から直線2への距離
        q2 = project_point_on_line(p1, p2, d2)
        return LineLineResult(float(np.linalg.norm(p1 - q2)), p1.copy(), q2, ang, True)
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    q1 = p1 + s * d1
    q2 = p2 + t * d2
    return LineLineResult(float(np.linalg.norm(q1 - q2)), q1, q2, ang, False)


def line_plane_intersection(line_point, line_dir, plane_origin, plane_normal):
    """直線と平面の交点。平行なら None。"""
    d = normalize(as_vec(line_dir))
    n = normalize(as_vec(plane_normal))
    denom = float(np.dot(d, n))
    if abs(denom) < 1e-12:
        return None
    t = float(np.dot(as_vec(plane_origin) - as_vec(line_point), n)) / denom
    return as_vec(line_point) + t * d


def plane_plane_intersection(o1, n1, o2, n2):
    """2平面の交線 (point, direction)。平行なら None。"""
    n1, n2 = normalize(as_vec(n1)), normalize(as_vec(n2))
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < 1e-10:
        return None
    d = normalize(d)
    # 交線上の点: n1·x = n1·o1, n2·x = n2·o2, d·x = 0 を解く
    A = np.vstack([n1, n2, d])
    b = np.array([np.dot(n1, as_vec(o1)), np.dot(n2, as_vec(o2)), 0.0])
    p = np.linalg.solve(A, b)
    return p, d


def three_plane_intersection(planes):
    """3平面 [(origin, normal), ...] の交点。特異なら None。"""
    A = np.vstack([normalize(as_vec(n)) for _, n in planes])
    b = np.array([np.dot(normalize(as_vec(n)), as_vec(o)) for o, n in planes])
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    return np.linalg.solve(A, b)


# --------------------------------------------------------------------------
# 変換行列
# --------------------------------------------------------------------------

def make_transform(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = as_vec(t)
    return T


def invert_transform(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_transform(points, T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    p = as_points(points)
    return p @ T[:3, :3].T + T[:3, 3]


def apply_rotation(vectors, T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return as_points(vectors) @ T[:3, :3].T


def is_rigid(T, tol: float = 1e-6) -> bool:
    R = np.asarray(T, dtype=float)[:3, :3]
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol)


def rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_xyz_to_matrix(rx, ry, rz) -> np.ndarray:
    """固定軸 X→Y→Z の順に回転 (R = Rz @ Ry @ Rx)。単位は度。"""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def matrix_to_euler_xyz(R) -> tuple[float, float, float]:
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = np.arcsin(sy)
    if abs(np.cos(ry)) > 1e-8:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # ジンバルロック
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))


def rotation_between(a, b) -> np.ndarray:
    """ベクトル a を b に重ねる最小回転行列."""
    a = normalize(as_vec(a))
    b = normalize(as_vec(b))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        u, _ = orthonormal_basis(a)
        return 2.0 * np.outer(u, u) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rotation_about_axis(axis, deg) -> np.ndarray:
    k = normalize(as_vec(axis))
    a = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def angular_range(angles_rad):
    """円周上の角度サンプルの範囲 (開始角 [rad], 範囲 [rad])。最大ギャップの直後を開始とする."""
    a = np.sort(np.mod(np.asarray(angles_rad, dtype=float), 2 * np.pi))
    if a.size < 2:
        return (float(a[0]) if a.size else 0.0), 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    k = int(np.argmax(gaps))
    start = a[(k + 1) % a.size]
    return float(start), float(2 * np.pi - gaps[k])


def angular_coverage_deg(angles_rad) -> float:
    """円周上の角度サンプルがカバーする範囲 [deg] (360 - 最大ギャップ)."""
    a = np.sort(np.mod(np.asarray(angles_rad, dtype=float), 2 * np.pi))
    if a.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return float(np.degrees(2 * np.pi - gaps.max()))


def min_area_rect_axes(uv):
    """2D点群を囲む最小面積矩形の軸 (e1, e2)。凸包の各辺方向を試す（回転キャリパー法）."""
    uv = np.asarray(uv, dtype=float)
    ex = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if len(uv) < 3:
        return ex
    from scipy.spatial import ConvexHull
    try:
        hull = uv[ConvexHull(uv).vertices]
    except Exception:
        return ex
    e = np.roll(hull, -1, axis=0) - hull
    ln = np.linalg.norm(e, axis=1)
    ok = ln > 1e-12
    if not ok.any():
        return ex
    e1s = e[ok] / ln[ok, None]
    e2s = np.column_stack([-e1s[:, 1], e1s[:, 0]])
    p1 = hull @ e1s.T
    p2 = hull @ e2s.T
    area = np.ptp(p1, axis=0) * np.ptp(p2, axis=0)
    k = int(np.argmin(area))
    a = e1s[k]
    # 4方向の候補のうち元の u 軸 (1,0) に最も近いものを e1 とし、右手系の e2 を作る
    cands = [a, np.array([-a[1], a[0]]), -a, np.array([a[1], -a[0]])]
    e1 = max(cands, key=lambda c: c[0])
    e2 = np.array([-e1[1], e1[0]])
    return e1, e2
