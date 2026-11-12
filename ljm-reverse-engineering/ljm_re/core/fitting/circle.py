"""円フィッティング（2D / 3D）.

- 代数的初期値: Taubin法（円弧の一部でもバイアスが小さい）
- 精密化: 幾何距離の非線形最小二乗 (Levenberg-Marquardt)
- 外れ値対策: 3点RANSAC（任意）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..geometry import angular_coverage_deg, normalize, orthonormal_basis
from .common import (FitStats, auto_threshold, covariance_from_jacobian, ransac_iterations,
                     residual_stats)
from .plane import fit_plane


@dataclass
class Circle2DFit:
    center: np.ndarray   # (2,)
    radius: float
    stats: FitStats
    inlier_mask: np.ndarray
    coverage_deg: float
    sigma_radius: float
    sigma_center: float
    warnings: list = field(default_factory=list)


@dataclass
class Circle3DFit:
    center: np.ndarray   # (3,)
    normal: np.ndarray   # 円の平面の法線
    radius: float
    stats: FitStats
    inlier_mask: np.ndarray
    coverage_deg: float
    sigma_radius: float
    sigma_center: float
    u_axis: np.ndarray
    v_axis: np.ndarray
    plane_rms: float     # 点群の平面性（円の平面に対する残差RMS）
    warnings: list = field(default_factory=list)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


def circle_taubin(xy: np.ndarray):
    """Taubin法 (SVD版, Chernov)。戻り値 (center(2,), r)."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        raise ValueError("円フィットには3点以上が必要です")
    centroid = xy.mean(axis=0)
    X = xy - centroid
    Z = (X ** 2).sum(axis=1)
    Zmean = Z.mean()
    if Zmean <= 0:
        raise ValueError("点が縮退しています")
    Z0 = (Z - Zmean) / (2.0 * np.sqrt(Zmean))
    M = np.column_stack([Z0, X])
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    A = vt[2].copy()
    A[0] = A[0] / (2.0 * np.sqrt(Zmean))
    A = np.append(A, -Zmean * A[0])
    if abs(A[0]) < 1e-15:
        raise ValueError("点が一直線上にあり円を決定できません")
    c = -A[1:3] / A[0] / 2.0 + centroid
    r = np.sqrt(A[1] ** 2 + A[2] ** 2 - 4 * A[0] * A[3]) / abs(A[0]) / 2.0
    return c, float(r)


def _circle_from_3(p):
    a, b, c = p
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = a @ a, b @ b, c @ c
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.linalg.norm(a - ctr))


def _refine_circle(xy, c0, r0):
    def fun(x):
        return np.hypot(xy[:, 0] - x[0], xy[:, 1] - x[1]) - x[2]

    def jac(x):
        dx = xy[:, 0] - x[0]
        dy = xy[:, 1] - x[1]
        rr = np.maximum(np.hypot(dx, dy), 1e-15)
        return np.column_stack([-dx / rr, -dy / rr, -np.ones(len(xy))])

    res = least_squares(fun, np.array([c0[0], c0[1], r0]), jac=jac, method="lm")
    x = res.x
    r = fun(x)
    cov = covariance_from_jacobian(jac(x), r)
    return x[:2], abs(float(x[2])), r, cov


def fit_circle_2d(xy, *, ransac: bool = False, threshold: Optional[float] = None,
                  max_iter: int = 1000, rng=None) -> Circle2DFit:
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if n < 3:
        raise ValueError("円フィットには3点以上が必要です")
    scale = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    mask = np.ones(n, bool)
    method = "taubin+lm"
    thr_auto = False
    iterations = None
    if ransac and n >= 6:
        rng = rng or np.random.default_rng(0)
        if threshold is None:
            c0, r0 = circle_taubin(xy)
            threshold = auto_threshold(np.hypot(*(xy - c0).T) - r0, scale, 2.5)
            thr_auto = True
        best = -1
        best_m = None
        needed = max_iter
        it = 0
        while it < min(needed, max_iter):
            it += 1
            m = _circle_from_3(xy[rng.choice(n, 3, replace=False)])
            if m is None:
                continue
            d = np.abs(np.hypot(*(xy - m[0]).T) - m[1])
            cnt = int(np.count_nonzero(d < threshold))
            if cnt > best:
                best, best_m = cnt, m
                needed = ransac_iterations(cnt / n, 3, 0.999, max_iter)
        iterations = it
        if best_m is not None:
            mask = np.abs(np.hypot(*(xy - best_m[0]).T) - best_m[1]) < threshold
            if mask.sum() < 3:
                mask[:] = True
        method = "ransac+lm"
    pts = xy[mask]
    c0, r0 = circle_taubin(pts)
    c, r, res, cov = _refine_circle(pts, c0, r0)
    if ransac and threshold is not None:
        # 精密化後にインライアを再判定して再フィット
        mask2 = np.abs(np.hypot(*(xy - c).T) - r) < threshold
        if mask2.sum() >= 3 and not np.array_equal(mask2, mask):
            mask = mask2
            c, r, res, cov = _refine_circle(xy[mask], c, r)
    stats = residual_stats(res, n, method, threshold, iterations, thr_auto)
    ang = np.arctan2(xy[mask][:, 1] - c[1], xy[mask][:, 0] - c[0])
    cov_deg = angular_coverage_deg(ang)
    sig_r = float(np.sqrt(cov[2, 2])) if cov is not None else float("nan")
    sig_c = float(np.sqrt(max(cov[0, 0], cov[1, 1]))) if cov is not None else float("nan")
    warnings = []
    if cov_deg < 90:
        warnings.append(f"円弧の範囲が {cov_deg:.0f}° と狭いため、半径・中心の不確かさが大きくなります。")
    return Circle2DFit(c, r, stats, mask, cov_deg, sig_r, sig_c, warnings)


def fit_circle_3d(points, *, ransac: bool = False, threshold: Optional[float] = None,
                  normal_hint=None, rng=None) -> Circle3DFit:
    """3D空間上の円（穴の縁など）。平面フィット→平面に投影→2D円フィット."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("円フィットには3点以上が必要です")
    pl = fit_plane(pts, method="lsq", orient_hint=normal_hint)
    nrm = pl.normal
    u, v = orthonormal_basis(nrm)
    rel = pts - pl.origin
    xy = np.column_stack([rel @ u, rel @ v])
    c2 = fit_circle_2d(xy, ransac=ransac, threshold=threshold, rng=rng)
    center = pl.origin + c2.center[0] * u + c2.center[1] * v
    plane_rms = float(np.sqrt(np.mean((rel @ nrm) ** 2)))
    # 3D残差: 円周からの距離（面内 + 面外）
    inl = pts[c2.inlier_mask]
    rr = inl - center
    h = rr @ nrm
    radial = np.linalg.norm(rr - np.outer(h, nrm), axis=1)
    res3 = np.hypot(radial - c2.radius, h)
    stats = residual_stats(res3, len(pts), c2.stats.method, c2.stats.threshold,
                           c2.stats.iterations, c2.stats.threshold_auto)
    warnings = list(c2.warnings)
    if plane_rms > max(0.05 * c2.radius, 1e-9) and plane_rms > 3 * c2.stats.rms:
        warnings.append("点群が一平面上にありません。円筒面を選択した場合は円筒フィットを使用してください。")
    return Circle3DFit(center, nrm, c2.radius, stats, c2.inlier_mask, c2.coverage_deg,
                       c2.sigma_radius, c2.sigma_center, u, v, plane_rms, warnings)


def circle_points_3d(center, normal, radius, n: int = 72) -> np.ndarray:
    u, v = orthonormal_basis(normalize(np.asarray(normal, float)))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center, float) + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
