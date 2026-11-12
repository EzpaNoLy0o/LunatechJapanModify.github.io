"""3D測定（GUIに依存しない純粋な計算）.

結果は dict で返し、GUIやCSV出力でそのまま使えるようにする。
長さは mm、角度は deg。座標成分は与えられた Frame（既定=ワールド）で表す。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .coords import Frame, WORLD
from .geometry import (angle_between, as_vec, line_line_closest, line_plane_intersection,
                       normalize, point_line_distance)

PARALLEL_TOL_DEG = 0.5      # 平行判定の既定許容角
PERPENDICULAR_TOL_DEG = 0.5


def distance_point_point(p1, p2, frame: Frame = WORLD) -> dict:
    p1, p2 = as_vec(p1), as_vec(p2)
    d = p2 - p1
    dl = frame.dir_to_local(d)
    return {
        "type": "point_point",
        "distance": float(np.linalg.norm(d)),
        "dx": float(dl[0]), "dy": float(dl[1]), "dz": float(dl[2]),
        "p1": p1.tolist(), "p2": p2.tolist(),
    }


def angle_three_points(a, vertex, b) -> dict:
    a, v, b = as_vec(a), as_vec(vertex), as_vec(b)
    if np.linalg.norm(a - v) < 1e-12 or np.linalg.norm(b - v) < 1e-12:
        raise ValueError("頂点と同じ位置の点があります")
    ang = angle_between(a - v, b - v)
    return {"type": "angle_3pt", "angle": ang, "p1": a.tolist(), "vertex": v.tolist(),
            "p2": b.tolist()}


def point_plane(p, plane_origin, plane_normal) -> dict:
    p = as_vec(p)
    n = normalize(as_vec(plane_normal))
    sd = float(np.dot(p - as_vec(plane_origin), n))
    foot = p - sd * n
    return {"type": "point_plane", "distance": abs(sd), "signed_distance": sd,
            "foot": foot.tolist(), "p": p.tolist()}


def plane_plane(o1, n1, o2, n2, parallel_tol_deg=PARALLEL_TOL_DEG,
                perp_tol_deg=PERPENDICULAR_TOL_DEG) -> dict:
    """平面間の角度・距離・平行/直角判定.

    距離は「平面2の中心から平面1まで」と「平面1の中心から平面2まで」の平均。
    非平行の場合、距離は参考値（中心間の法線方向距離）。
    """
    n1, n2 = normalize(as_vec(n1)), normalize(as_vec(n2))
    o1, o2 = as_vec(o1), as_vec(o2)
    ang_dir = angle_between(n1, n2)            # 0〜180
    ang = angle_between(n1, n2, undirected=True)  # 0〜90
    d12 = float(np.dot(o2 - o1, n1))
    d21 = float(np.dot(o1 - o2, n2))
    parallel = ang <= parallel_tol_deg
    perpendicular = abs(90.0 - ang) <= perp_tol_deg
    dist = 0.5 * (abs(d12) + abs(d21))
    return {"type": "plane_plane", "angle": ang, "angle_directed": ang_dir,
            "distance": dist, "distance_1to2": abs(d12), "distance_2to1": abs(d21),
            "parallel": bool(parallel), "perpendicular": bool(perpendicular),
            "parallel_tol_deg": parallel_tol_deg}


def axis_axis(p1, d1, p2, d2, extent2: Optional[tuple] = None,
              parallel_tol_deg=PARALLEL_TOL_DEG) -> dict:
    """2軸間の角度・最短距離・同軸度.

    extent2: 軸2の評価範囲 (tmin, tmax)（p2 基準）。同軸度は軸1を基準(データム)として、
    軸2の評価範囲端点の軸1からの最大距離×2（ISO 1101 同軸度の考え方）。
    """
    r = line_line_closest(p1, d1, p2, d2)
    d2n = normalize(as_vec(d2))
    out = {"type": "axis_axis", "angle": r.angle_deg, "distance": r.distance,
           "parallel": bool(r.angle_deg <= parallel_tol_deg),
           "closest_1": r.point_on_1.tolist(), "closest_2": r.point_on_2.tolist()}
    if extent2 is not None:
        ends = np.array([as_vec(p2) + extent2[0] * d2n, as_vec(p2) + extent2[1] * d2n])
        dd = point_line_distance(ends, p1, d1)
        out["coaxiality"] = float(2.0 * dd.max())
        out["offset_at_ends"] = dd.tolist()
    return out


def point_axis(p, axis_point, axis_dir) -> dict:
    d = float(point_line_distance(as_vec(p)[None], axis_point, axis_dir)[0])
    return {"type": "point_axis", "distance": d}


def axis_plane(axis_point, axis_dir, plane_origin, plane_normal) -> dict:
    """軸と平面: 交点と角度（軸と平面法線のなす角, 軸と平面のなす角）."""
    x = line_plane_intersection(axis_point, axis_dir, plane_origin, plane_normal)
    ang_n = angle_between(axis_dir, plane_normal, undirected=True)
    return {"type": "axis_plane", "intersection": None if x is None else x.tolist(),
            "angle_to_normal": ang_n, "angle_to_plane": 90.0 - ang_n}


def projected_distance(p1, p2, direction) -> dict:
    d = normalize(as_vec(direction))
    v = as_vec(p2) - as_vec(p1)
    t = float(np.dot(v, d))
    perp = float(np.linalg.norm(v - t * d))
    return {"type": "projected", "distance": abs(t), "signed": t, "perpendicular": perp,
            "direction": d.tolist()}


def bounding_box(points, frame: Frame = WORLD) -> dict:
    pts = frame.to_local(np.asarray(points, dtype=float))
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    size = hi - lo
    return {"type": "bbox", "min": lo.tolist(), "max": hi.tolist(), "size": size.tolist(),
            "center": ((lo + hi) / 2).tolist(), "diagonal": float(np.linalg.norm(size))}


def oriented_bounding_box(points) -> dict:
    """最小体積に近い有向バウンディングボックス（trimesh実装を利用）."""
    import trimesh
    pts = np.asarray(points, dtype=float)
    T, extents = trimesh.bounds.oriented_bounds(pts)
    # T は点を OBB ローカルへ移す変換
    Ti = np.linalg.inv(T)
    return {"type": "obb", "size": np.asarray(extents).tolist(), "transform": Ti.tolist(),
            "center": Ti[:3, 3].tolist()}


def coordinate(p, frame: Frame = WORLD) -> dict:
    q = frame.to_local(as_vec(p))
    return {"type": "coordinate", "x": float(q[0]), "y": float(q[1]), "z": float(q[2]),
            "world": as_vec(p).tolist()}


def scale_factor(measured: float, reference: float) -> dict:
    """既知寸法によるスケール確認."""
    if measured <= 0 or reference <= 0:
        raise ValueError("寸法は正の値で入力してください")
    s = reference / measured
    return {"type": "scale", "measured": measured, "reference": reference, "factor": s,
            "error_mm": measured - reference, "error_percent": (measured - reference) / reference * 100}


def distance_matrix(points) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
