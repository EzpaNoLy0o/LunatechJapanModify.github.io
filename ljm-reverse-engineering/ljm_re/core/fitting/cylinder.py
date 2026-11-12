"""円筒フィッティング.

手順:
1. 初期軸候補の生成
   - 法線がある場合: 法線は軸に直交するため、法線の共分散の最小固有ベクトル
   - 法線付きRANSAC: 2点の法線の外積 = 軸方向, 法線直線の交点 = 中心
   - 点群の主成分 (3方向)
2. 各候補について、軸直交平面に投影して2D円フィット(Taubin)で中心・半径の初期値
3. 5パラメータ（軸傾き2 + 軸位置2 + 半径1）の非線形最小二乗で精密化
4. 自動/指定閾値でインライアを再判定し再フィット
5. ヤコビアンから不確かさ（半径・軸角度・軸位置）を推定し、円弧カバー範囲から警告

部分的にしかスキャンできていない円筒でも動作するが、
円弧範囲が狭い場合は不確かさが大きくなることを結果に明示する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..geometry import angular_coverage_deg, angular_range, normalize, orthonormal_basis
from ..progress import Progress, ensure_progress
from .circle import circle_taubin
from .common import (FitStats, auto_threshold, bbox_diagonal, covariance_from_jacobian,
                     numeric_jacobian, ransac_iterations, residual_stats, subsample_indices)


@dataclass
class CylinderFit:
    axis_point: np.ndarray     # 軸上の点（インライア重心の軸への射影）
    axis_dir: np.ndarray       # 軸方向単位ベクトル
    radius: float
    stats: FitStats
    inlier_mask: np.ndarray
    t_range: tuple             # axis_point 基準の軸方向範囲 (tmin, tmax)
    coverage_deg: float        # 円周方向のカバー範囲
    sigma_radius: float        # 半径の標準不確かさ (1σ)
    sigma_axis_deg: float      # 軸方向の標準不確かさ (1σ, deg)
    sigma_axis_pos: float      # 軸位置（軸直交方向）の標準不確かさ (1σ)
    warnings: list = field(default_factory=list)
    arc_ref: Optional[np.ndarray] = None   # 角度の基準方向（軸に直交）
    arc_start_deg: float = 0.0             # スキャンされた円弧の開始角（arc_ref 基準, 軸右ねじ方向）

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def length(self) -> float:
        return float(self.t_range[1] - self.t_range[0])

    def end_points(self):
        a = self.axis_point + self.t_range[0] * self.axis_dir
        b = self.axis_point + self.t_range[1] * self.axis_dir
        return a, b


def axis_from_normals(normals: np.ndarray) -> np.ndarray:
    """法線群に最も直交する方向（= 円筒軸）."""
    N = np.asarray(normals, dtype=float)
    C = N.T @ N
    w, V = np.linalg.eigh(C)
    return normalize(V[:, 0])


def _initial_circle(points, axis):
    u, v = orthonormal_basis(axis)
    c = points.mean(axis=0)
    rel = points - c
    xy = np.column_stack([rel @ u, rel @ v])
    try:
        c2, r = circle_taubin(xy)
    except ValueError:
        return None
    p0 = c + c2[0] * u + c2[1] * v
    return p0, r


class _CylModel:
    """軸 a0 と軸上点 p0 を基準とする局所パラメータ化 (α, β, s, t, r)."""

    def __init__(self, a0, p0):
        self.a0 = normalize(a0)
        self.u0, self.v0 = orthonormal_basis(self.a0)
        self.p0 = np.asarray(p0, float)

    def decode(self, x):
        a = normalize(self.a0 + x[0] * self.u0 + x[1] * self.v0)
        p = self.p0 + x[2] * self.u0 + x[3] * self.v0
        return a, p, x[4]

    def residuals(self, x, pts):
        a, p, r = self.decode(x)
        w = pts - p
        t = w @ a
        d = np.linalg.norm(w - np.outer(t, a), axis=1)
        return d - r


def _refine(points, axis, p0, r0, loss="linear"):
    model = _CylModel(axis, p0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0, r0])
    method = "lm" if loss == "linear" and len(points) >= 5 else "trf"
    try:
        res = least_squares(model.residuals, x0, args=(points,), method=method, loss=loss,
                            x_scale="jac", max_nfev=200)
        x = res.x
    except Exception:
        x = x0
    a, p, r = model.decode(x)
    return a, p, abs(float(r))


def _foot(points, a, p):
    c = points.mean(axis=0)
    return p + np.dot(c - p, a) * a


def _ransac_with_normals(points, normals, threshold, angle_thr_deg, max_iter, rng, progress):
    n = len(points)
    cos_thr = np.cos(np.radians(angle_thr_deg))
    best = -1
    best_model = None
    needed = max_iter
    it = 0
    while it < min(needed, max_iter):
        it += 1
        if it % 50 == 0:
            progress.report(it / max(1, min(needed, max_iter)), "RANSAC 円筒探索中")
        i, j = rng.choice(n, 2, replace=False)
        n1, n2 = normals[i], normals[j]
        a = np.cross(n1, n2)
        na = np.linalg.norm(a)
        if na < 0.05:  # 法線がほぼ平行 → 軸を決定できない
            continue
        a = a / na
        u, v = orthonormal_basis(a)
        P1 = np.array([points[i] @ u, points[i] @ v])
        P2 = np.array([points[j] @ u, points[j] @ v])
        D1 = np.array([n1 @ u, n1 @ v])
        D2 = np.array([n2 @ u, n2 @ v])
        A = np.column_stack([D1, -D2])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        ts = np.linalg.solve(A, P2 - P1)
        c2 = P1 + ts[0] * D1
        r = 0.5 * (np.linalg.norm(P1 - c2) + np.linalg.norm(P2 - c2))
        if not np.isfinite(r) or r <= 0:
            continue
        p = c2[0] * u + c2[1] * v
        w = points - p
        t = w @ a
        radial = w - np.outer(t, a)
        dist = np.linalg.norm(radial, axis=1)
        ok = np.abs(dist - r) < threshold
        if normals is not None:
            rn = radial / np.maximum(dist[:, None], 1e-12)
            ok &= np.abs(np.sum(rn * normals, axis=1)) > cos_thr
        cnt = int(np.count_nonzero(ok))
        if cnt > best:
            best = cnt
            best_model = (a, p, r)
            needed = ransac_iterations(cnt / n, 2, 0.999, max_iter)
    return best_model, it


def fit_cylinder(points, normals=None, *, threshold: Optional[float] = None,
                 ransac: bool = True, angle_threshold_deg: float = 25.0, max_iter: int = 1000,
                 init_axis=None, rng: Optional[np.random.Generator] = None,
                 progress: Optional[Progress] = None, max_eval_points: int = 40000) -> CylinderFit:
    progress = ensure_progress(progress)
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 6:
        raise ValueError("円筒フィットには6点以上が必要です")
    nrm = None
    if normals is not None:
        nrm = np.asarray(normals, dtype=float)
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1e-12
        nrm = np.where(ok[:, None], nrm / np.maximum(ln[:, None], 1e-12), 0.0)
    rng = rng or np.random.default_rng(0)
    scale = bbox_diagonal(pts)
    idx = subsample_indices(n, max_eval_points, rng)
    P = pts[idx]
    Nn = nrm[idx] if nrm is not None else None

    # ---- 1. 初期軸候補 --------------------------------------------------
    candidates = []
    if init_axis is not None:
        candidates.append(normalize(np.asarray(init_axis, float)))
    if Nn is not None:
        candidates.append(axis_from_normals(Nn))
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    candidates.extend([normalize(vt[k]) for k in range(3)])

    best = None
    best_rms = np.inf
    progress.report(0.05, "円筒の初期値を計算中")
    for a in candidates:
        init = _initial_circle(P, a)
        if init is None:
            continue
        p0, r0 = init
        if not np.isfinite(r0) or r0 <= 0 or r0 > 50 * scale:
            continue
        a1, p1, r1 = _refine(P, a, p0, r0)
        if r1 > 50 * scale:
            continue
        rms = float(np.sqrt(np.mean(_CylModel(a1, p1).residuals(np.array([0, 0, 0, 0, r1]), P) ** 2)))
        if rms < best_rms:
            best_rms, best = rms, (a1, p1, r1)

    ransac_iters = None
    thr_auto = threshold is None
    if ransac and Nn is not None and len(P) >= 10:
        thr_r = threshold
        if thr_r is None:
            if best is not None:
                a1, p1, r1 = best
                res0 = _CylModel(a1, p1).residuals(np.array([0, 0, 0, 0, r1]), P)
                thr_r = auto_threshold(res0, scale, 2.5)
            else:
                thr_r = 0.01 * scale
        model, ransac_iters = _ransac_with_normals(P, Nn, thr_r, angle_threshold_deg,
                                                   max_iter, rng, progress.sub(0.1, 0.6))
        if model is not None:
            a2, p2, r2 = model
            res2 = _CylModel(a2, p2).residuals(np.array([0, 0, 0, 0, r2]), P)
            mask2 = np.abs(res2) < thr_r
            if mask2.sum() >= 6:
                a3, p3, r3 = _refine(P[mask2], a2, p2, r2)
                rms3_all = float(np.sqrt(np.mean(np.minimum(
                    _CylModel(a3, p3).residuals(np.array([0, 0, 0, 0, r3]), P) ** 2, thr_r ** 2))))
                rms_best_all = np.inf
                if best is not None:
                    a1, p1, r1 = best
                    rms_best_all = float(np.sqrt(np.mean(np.minimum(
                        _CylModel(a1, p1).residuals(np.array([0, 0, 0, 0, r1]), P) ** 2, thr_r ** 2))))
                if rms3_all < rms_best_all:
                    best = (a3, p3, r3)

    if best is None:
        raise ValueError("円筒を推定できませんでした。選択範囲を確認してください。")

    # ---- 2. 全点でインライア判定 → 再フィット ------------------------------
    progress.report(0.7, "円筒を精密フィット中")
    a, p, r = best
    res_all = _CylModel(a, p).residuals(np.array([0, 0, 0, 0, r]), pts)
    if threshold is None:
        threshold = auto_threshold(res_all, scale, 3.0)
    mask = np.abs(res_all) < threshold
    if mask.sum() < 6:
        mask[:] = True
    for _ in range(4):
        a, p, r = _refine(pts[mask], a, p, r)
        res_all = _CylModel(a, p).residuals(np.array([0, 0, 0, 0, r]), pts)
        new_mask = np.abs(res_all) < threshold
        if new_mask.sum() < 6 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    inl = pts[mask]
    p = _foot(inl, a, p)
    model = _CylModel(a, p)
    x_hat = np.array([0, 0, 0, 0, r])
    res = model.residuals(x_hat, inl)
    method = "ransac(normals)+lm" if ransac_iters else "lm"
    stats = residual_stats(res, n, method, threshold, ransac_iters, thr_auto)

    # ---- 3. 不確かさ -------------------------------------------------------
    J_idx = subsample_indices(len(inl), 200000, rng)
    J_pts = inl[J_idx]
    J = numeric_jacobian(lambda x: model.residuals(x, J_pts), x_hat,
                         eps=np.array([1e-7, 1e-7, 1e-6 * max(r, 1), 1e-6 * max(r, 1), 1e-6 * max(r, 1)]))
    cov = covariance_from_jacobian(J, model.residuals(x_hat, J_pts))
    if cov is not None and len(J_idx) < len(inl):
        cov = cov * (len(J_idx) / len(inl))
    if cov is not None:
        sig_r = float(np.sqrt(max(cov[4, 4], 0)))
        sig_ang = float(np.degrees(np.sqrt(max(cov[0, 0], cov[1, 1], 0))))
        sig_pos = float(np.sqrt(max(cov[2, 2], cov[3, 3], 0)))
    else:
        sig_r = sig_ang = sig_pos = float("nan")

    w = inl - p
    t = w @ a
    u, v = orthonormal_basis(a)
    ang = np.arctan2(w @ v, w @ u)
    coverage = angular_coverage_deg(ang)
    arc_start, _ = angular_range(ang)
    t_range = (float(t.min()), float(t.max()))

    warnings = []
    if coverage < 120:
        warnings.append(f"円周方向のスキャン範囲が {coverage:.0f}° しかありません。直径・軸位置の推定不確かさが大きくなります（半径 1σ = {sig_r:.3g} mm）。")
    if stats.inlier_ratio < 0.7:
        warnings.append(f"インライア率が {stats.inlier_ratio*100:.0f}% と低いです。選択範囲に円筒以外の面が含まれている可能性があります。")
    if (t_range[1] - t_range[0]) < 0.3 * r:
        warnings.append("軸方向の長さが半径に比べて短いため、軸方向ベクトルの推定精度が低い可能性があります。円フィット（穴中心）の利用も検討してください。")
    progress.report(1.0, "円筒フィット完了", force=True)
    return CylinderFit(axis_point=p, axis_dir=a, radius=float(r), stats=stats, inlier_mask=mask,
                       t_range=t_range, coverage_deg=coverage, sigma_radius=sig_r,
                       sigma_axis_deg=sig_ang, sigma_axis_pos=sig_pos, warnings=warnings,
                       arc_ref=u, arc_start_deg=float(np.degrees(arc_start)))


def cylinder_residuals(points, axis_point, axis_dir, radius) -> np.ndarray:
    a = normalize(np.asarray(axis_dir, float))
    w = np.asarray(points, float) - np.asarray(axis_point, float)
    t = w @ a
    return np.linalg.norm(w - np.outer(t, a), axis=1) - radius
