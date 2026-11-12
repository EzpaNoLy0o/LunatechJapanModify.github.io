"""平面フィッティング（最小二乗 / RANSAC + 最小二乗再フィット）."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry import min_area_rect_axes, normalize, orthonormal_basis
from ..progress import Progress, ensure_progress
from .common import (FitStats, auto_threshold, bbox_diagonal, ransac_iterations,
                     residual_stats, subsample_indices)


@dataclass
class PlaneFit:
    origin: np.ndarray          # インライアの重心（平面上）
    normal: np.ndarray          # 単位法線
    stats: FitStats
    inlier_mask: np.ndarray     # 入力点に対するインライアのマスク
    u_axis: np.ndarray          # 平面内の主軸1
    v_axis: np.ndarray          # 平面内の主軸2
    extent_uv: np.ndarray       # [[umin, vmin], [umax, vmax]] インライアの範囲
    flatness: float             # インライアの (最大残差 - 最小残差) = 平面度相当
    sigma_normal_deg: float     # 法線方向の推定標準不確かさ [deg]
    warnings: list = field(default_factory=list)

    @property
    def d(self) -> float:
        """平面方程式 n·x = d の d."""
        return float(np.dot(self.normal, self.origin))


def _lsq_plane(points: np.ndarray):
    c = points.mean(axis=0)
    X = points - c
    # SVD: 最小特異値の右特異ベクトルが法線
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2]
    u = vt[0]
    v = np.cross(n, u)
    return c, normalize(n), normalize(u), normalize(v), s


def _plane_from_3(p: np.ndarray):
    n = np.cross(p[1] - p[0], p[2] - p[0])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    return p[0], n / nn


def fit_plane_lsq(points, orient_hint=None) -> PlaneFit:
    """全点を使う最小二乗平面."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("平面フィットには3点以上が必要です")
    return _finalize(pts, np.ones(len(pts), bool), "lsq", None, None, False, orient_hint)


def fit_plane(points, *, method: str = "ransac", threshold: Optional[float] = None,
              max_iter: int = 2000, prob: float = 0.999, orient_hint=None,
              rng: Optional[np.random.Generator] = None,
              progress: Optional[Progress] = None, max_eval_points: int = 60000) -> PlaneFit:
    """平面フィット.

    method='lsq'   : 最小二乗のみ
    method='ransac': RANSACで外れ値を除外後、インライアで最小二乗再フィット
    threshold      : インライア判定距離 [mm]。None なら残差のMADから自動決定
    orient_hint    : 法線の向きを合わせる参照ベクトル（例: メッシュ法線の平均）
    """
    progress = ensure_progress(progress)
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("平面フィットには3点以上が必要です")
    if method == "lsq" or n < 10:
        return _finalize(pts, np.ones(n, bool), "lsq", None, None, False, orient_hint)

    rng = rng or np.random.default_rng(0)
    scale = bbox_diagonal(pts)
    thr_auto = threshold is None
    if thr_auto:
        c0, n0, *_ = _lsq_plane(pts)
        res0 = (pts - c0) @ n0
        threshold = auto_threshold(res0, scale, k=2.5)

    eval_idx = subsample_indices(n, max_eval_points, rng)
    P = pts[eval_idx]
    best_count = -1
    best_model = None
    it = 0
    needed = max_iter
    while it < min(needed, max_iter):
        it += 1
        if it % 50 == 0:
            progress.report(it / max(1, min(needed, max_iter)), "RANSAC 平面探索中")
        sample = P[rng.choice(len(P), 3, replace=False)]
        model = _plane_from_3(sample)
        if model is None:
            continue
        o, nn = model
        d = np.abs((P - o) @ nn)
        cnt = int(np.count_nonzero(d < threshold))
        if cnt > best_count:
            best_count = cnt
            best_model = model
            needed = ransac_iterations(cnt / len(P), 3, prob, max_iter)

    if best_model is None:
        return _finalize(pts, np.ones(n, bool), "lsq", None, None, False, orient_hint)

    o, nn = best_model
    mask = np.abs((pts - o) @ nn) < threshold
    # インライアで最小二乗再フィットを繰り返す
    for _ in range(5):
        if mask.sum() < 3:
            break
        c, nrm, *_ = _lsq_plane(pts[mask])
        new_mask = np.abs((pts - c) @ nrm) < threshold
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    if mask.sum() < 3:
        mask = np.ones(n, bool)
    progress.report(1.0, "平面フィット完了", force=True)
    return _finalize(pts, mask, "ransac+lsq", threshold, it, thr_auto, orient_hint)


def _finalize(pts, mask, method, threshold, iterations, thr_auto, orient_hint) -> PlaneFit:
    inl = pts[mask]
    c, nrm, u, v, s = _lsq_plane(inl)
    if orient_hint is not None:
        h = np.asarray(orient_hint, dtype=float)
        if np.linalg.norm(h) > 0 and np.dot(nrm, h) < 0:
            nrm = -nrm
            v = -v
    res = (inl - c) @ nrm
    stats = residual_stats(res, len(pts), method, threshold, iterations, thr_auto)
    uv = np.column_stack([(inl - c) @ u, (inl - c) @ v])
    # 面内の軸は「点群を囲む最小面積矩形」の向きに揃える（矩形面の表示範囲・CAD面が正確になる）
    e1, e2 = min_area_rect_axes(uv)
    u2 = e1[0] * u + e1[1] * v
    u = normalize(u2)
    v = np.cross(nrm, u)
    uv = np.column_stack([(inl - c) @ u, (inl - c) @ v])
    extent = np.array([uv.min(axis=0), uv.max(axis=0)])
    # 法線の不確かさ: 傾き方向ごとの σ_d / sqrt(Σu²)
    m = len(inl)
    sigma_d = float(np.sqrt(np.sum(res ** 2) / max(1, m - 3)))
    su = np.sqrt(max(np.sum(uv[:, 0] ** 2), 1e-30))
    sv = np.sqrt(max(np.sum(uv[:, 1] ** 2), 1e-30))
    sigma_normal = float(np.degrees(max(sigma_d / su, sigma_d / sv)))
    warnings = []
    if stats.inlier_ratio < 0.7:
        warnings.append(f"インライア率が {stats.inlier_ratio*100:.0f}% と低いです。選択範囲に平面以外が含まれている可能性があります。")
    size = extent[1] - extent[0]
    if min(size) < 1e-9 or max(size) / max(min(size), 1e-9) > 50:
        warnings.append("選択領域が細長いため、法線の推定精度が低下している可能性があります。")
    return PlaneFit(origin=c, normal=nrm, stats=stats, inlier_mask=mask, u_axis=u, v_axis=v,
                    extent_uv=extent, flatness=float(res.max() - res.min()) if m else 0.0,
                    sigma_normal_deg=sigma_normal, warnings=warnings)


def plane_from_point_normal(origin, normal):
    n = normalize(np.asarray(normal, dtype=float))
    u, v = orthonormal_basis(n)
    return np.asarray(origin, float), n, u, v


@dataclass
class LineFit:
    point: np.ndarray
    direction: np.ndarray
    stats: FitStats
    t_range: tuple


def fit_line_3d(points) -> LineFit:
    """3D直線フィット（主成分分析）."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("直線フィットには2点以上が必要です")
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = normalize(vt[0])
    t = (pts - c) @ d
    perp = (pts - c) - np.outer(t, d)
    res = np.linalg.norm(perp, axis=1)
    stats = residual_stats(res, len(pts), "lsq")
    return LineFit(c, d, stats, (float(t.min()), float(t.max())))
