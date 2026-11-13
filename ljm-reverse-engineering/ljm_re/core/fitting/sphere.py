"""球フィッティング（代数初期値 + 幾何最小二乗）."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .common import FitStats, auto_threshold, bbox_diagonal, covariance_from_jacobian, residual_stats


@dataclass
class SphereFit:
    center: np.ndarray
    radius: float
    stats: FitStats
    inlier_mask: np.ndarray
    sigma_radius: float
    sigma_center: float
    warnings: list = field(default_factory=list)

    @property
    def diameter(self):
        return 2 * self.radius


def _algebraic(pts):
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    r = np.sqrt(max(x[3] + c @ c, 1e-30))
    return c, float(r)


def _refine(pts, c, r):
    def fun(x):
        return np.linalg.norm(pts - x[:3], axis=1) - x[3]

    def jac(x):
        d = pts - x[:3]
        rr = np.maximum(np.linalg.norm(d, axis=1), 1e-15)
        return np.column_stack([-d / rr[:, None], -np.ones(len(pts))])

    res = least_squares(fun, np.r_[c, r], jac=jac, method="lm")
    x = res.x
    rr = fun(x)
    return x[:3], abs(float(x[3])), rr, covariance_from_jacobian(jac(x), rr)


def fit_sphere(points, *, threshold: Optional[float] = None, robust: bool = True) -> SphereFit:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 4:
        raise ValueError("球フィットには4点以上が必要です")
    c, r = _algebraic(pts)
    c, r, res, cov = _refine(pts, c, r)
    mask = np.ones(n, bool)
    thr_auto = False
    if robust:
        if threshold is None:
            threshold = auto_threshold(res, bbox_diagonal(pts), 3.0)
            thr_auto = True
        for _ in range(4):
            new = np.abs(np.linalg.norm(pts - c, axis=1) - r) < threshold
            if new.sum() < 4 or np.array_equal(new, mask):
                break
            mask = new
            c, r, res, cov = _refine(pts[mask], c, r)
    stats = residual_stats(res, n, "lm" + ("+trim" if robust else ""), threshold, None, thr_auto)
    sr = float(np.sqrt(cov[3, 3])) if cov is not None else float("nan")
    sc = float(np.sqrt(max(np.diag(cov)[:3]))) if cov is not None else float("nan")
    return SphereFit(c, r, stats, mask, sr, sc, [])
