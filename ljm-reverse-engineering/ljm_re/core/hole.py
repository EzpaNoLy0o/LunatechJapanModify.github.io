"""穴中心の認識.

方式:
  A. 内壁（円筒面）を選択 → 円筒フィット → 軸と基準平面の交点を穴中心とする
  B. 穴の縁（円形エッジ）の点を選択 → 3D円フィット
  C. 平面断面 → 断面ループの円フィット（内周が不完全な場合にも有効）
  D. 平面上の穴の自動検出（平面から少し内側の断面で円形ループを探す）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fitting.circle import fit_circle_2d, fit_circle_3d
from .fitting.cylinder import CylinderFit
from .geometry import line_plane_intersection, normalize
from .section import section_mesh


@dataclass
class HoleResult:
    center: np.ndarray
    axis: np.ndarray
    diameter: float
    method: str
    rms: float
    max_abs: float
    n_points: int
    coverage_deg: float
    sigma_diameter: float
    sigma_center: float
    depth: Optional[float] = None
    warnings: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def hole_from_cylinder(cyl: CylinderFit, ref_plane=None, which: str = "near") -> HoleResult:
    """円筒フィット結果から穴中心.

    ref_plane=(origin, normal) があれば軸と平面の交点。なければ軸方向範囲の端
    (which='near': 平面法線側/+軸側, 'far': 反対側, 'mid': 中央)。
    """
    a = cyl.axis_dir
    warnings = list(cyl.warnings)
    if ref_plane is not None:
        o, n = ref_plane
        c = line_plane_intersection(cyl.axis_point, a, o, n)
        if c is None:
            raise ValueError("穴の軸が基準平面と平行なため交点を計算できません")
        ang = np.degrees(np.arccos(min(1.0, abs(np.dot(normalize(a), normalize(n))))))
        if ang > 2.0:
            warnings.append(f"穴の軸が基準平面の法線に対して {ang:.2f}° 傾いています。")
        if np.dot(a, n) < 0:
            a = -a
        method = "cylinder∩plane"
    else:
        p0, p1 = cyl.end_points()
        c = {"near": p1, "far": p0, "mid": 0.5 * (p0 + p1)}[which]
        method = f"cylinder({which})"
    return HoleResult(center=np.asarray(c, float), axis=normalize(a), diameter=cyl.diameter,
                      method=method, rms=cyl.stats.rms, max_abs=cyl.stats.max_abs,
                      n_points=cyl.stats.n_used, coverage_deg=cyl.coverage_deg,
                      sigma_diameter=2 * cyl.sigma_radius, sigma_center=cyl.sigma_axis_pos,
                      depth=cyl.length, warnings=warnings)


def hole_from_edge_points(points, normal_hint=None, ransac=True) -> HoleResult:
    cf = fit_circle_3d(points, ransac=ransac, normal_hint=normal_hint)
    return HoleResult(center=cf.center, axis=cf.normal, diameter=cf.diameter, method="circle3d",
                      rms=cf.stats.rms, max_abs=cf.stats.max_abs, n_points=cf.stats.n_used,
                      coverage_deg=cf.coverage_deg, sigma_diameter=2 * cf.sigma_radius,
                      sigma_center=cf.sigma_center, warnings=list(cf.warnings))


def _loop_circle(sec, pl, ransac=True):
    uv = sec.to_2d(pl.points)
    cf = fit_circle_2d(uv, ransac=ransac)
    return cf, uv


def hole_from_section(V, F, plane_origin, plane_normal, pick_point=None, depth: float = 0.5,
                      ransac: bool = True) -> HoleResult:
    """平面から depth だけ材料側（-法線方向）にずらした断面で、穴の円を求める.

    pick_point に最も近い（または内包する）ループを使う。
    """
    n = normalize(np.asarray(plane_normal, float))
    o = np.asarray(plane_origin, float) - depth * n
    sec = section_mesh(V, F, o, n)
    if not sec.polylines:
        raise ValueError("断面が得られませんでした。断面深さを確認してください。")
    best = None
    pp = None if pick_point is None else sec.to_2d(np.asarray(pick_point, float)[None])[0]
    for pl in sec.polylines:
        if len(pl.points) < 6:
            continue
        try:
            cf, uv = _loop_circle(sec, pl, ransac)
        except ValueError:
            continue
        if pp is not None:
            dc = float(np.linalg.norm(pp - cf.center))
            # クリック点を内包する（中心に近い）ループを優先、なければ最も近いループ
            score = dc if dc < cf.radius * 1.5 else 1e6 + float(np.min(np.linalg.norm(uv - pp, axis=1)))
        else:
            score = cf.stats.rms / max(cf.radius, 1e-9)
        if best is None or score < best[0]:
            best = (score, cf, pl)
    if best is None:
        raise ValueError("円形の断面ループが見つかりませんでした")
    _, cf, pl = best
    c3 = sec.to_3d(cf.center[None])[0] + depth * n  # 平面上に戻す
    return HoleResult(center=c3, axis=n, diameter=2 * cf.radius, method="section-circle",
                      rms=cf.stats.rms, max_abs=cf.stats.max_abs, n_points=cf.stats.n_used,
                      coverage_deg=cf.coverage_deg, sigma_diameter=2 * cf.sigma_radius,
                      sigma_center=cf.sigma_center, warnings=list(cf.warnings),
                      extra={"section_depth": depth, "loop_closed": pl.closed})


def detect_holes_on_plane(V, F, plane_origin, plane_normal, depth: float = 0.5,
                          min_diameter: float = 2.0, max_diameter: float = 200.0,
                          rel_tol: float = 0.01, abs_tol: float = 0.1,
                          region_radius: Optional[float] = None) -> list:
    """平面上の穴を自動検出する。円形の閉ループ（外周輪郭を除く）を穴とみなす.

    rel_tol/abs_tol: 円とみなす最大残差 = max(abs_tol, rel_tol*半径)
    region_radius  : 指定すると平面中心からこの距離以内のループのみ
    """
    n = normalize(np.asarray(plane_normal, float))
    po = np.asarray(plane_origin, float)
    o = po - depth * n
    sec = section_mesh(V, F, o, n)
    holes = []
    for pl in sec.polylines:
        if len(pl.points) < 8:
            continue
        try:
            cf, uv = _loop_circle(sec, pl, ransac=False)
        except ValueError:
            continue
        d = 2 * cf.radius
        if not (min_diameter <= d <= max_diameter):
            continue
        if cf.stats.max_abs > max(abs_tol, rel_tol * cf.radius):
            continue
        if cf.coverage_deg < 270:
            continue
        c3 = sec.to_3d(cf.center[None])[0] + depth * n
        if region_radius is not None and np.linalg.norm(c3 - po) > region_radius:
            continue
        holes.append(HoleResult(center=c3, axis=n, diameter=d, method="auto-section",
                                rms=cf.stats.rms, max_abs=cf.stats.max_abs,
                                n_points=cf.stats.n_used, coverage_deg=cf.coverage_deg,
                                sigma_diameter=2 * cf.sigma_radius, sigma_center=cf.sigma_center,
                                warnings=list(cf.warnings),
                                extra={"section_depth": depth, "loop_closed": pl.closed}))
    # 同心の重複（面取り等）を除去: 中心がほぼ同じなら小さい方を残す
    holes.sort(key=lambda h: h.diameter)
    uniq = []
    for h in holes:
        if all(np.linalg.norm(h.center - u.center) > 0.2 * h.diameter for u in uniq):
            uniq.append(h)
    return uniq
