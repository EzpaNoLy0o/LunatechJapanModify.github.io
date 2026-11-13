"""位置合わせ: 対応点による初期位置合わせ (Kabsch/Umeyama) と ICP 精密位置合わせ.

注意: 位置合わせのRMSが小さくても、測定精度が保証されるわけではない。
基準寸法による検証（スケール確認）を併用すること。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry import apply_transform
from .progress import Progress, ensure_progress


def kabsch(src, dst, with_scale: bool = False):
    """対応点 src→dst を最小二乗で重ねる剛体（+スケール）変換。戻り値 (T 4x4, rms)."""
    A = np.asarray(src, float)
    B = np.asarray(dst, float)
    if A.shape != B.shape or len(A) < 3:
        raise ValueError("対応点は3組以上必要です")
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    X, Y = A - ca, B - cb
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = 1.0
    if with_scale:
        s = float(np.trace(np.diag(S) @ D) / max((X ** 2).sum(), 1e-30))
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = cb - s * R @ ca
    rms = float(np.sqrt(np.mean(np.sum((apply_transform(A, T) - B) ** 2, axis=1))))
    if np.linalg.matrix_rank(X, tol=1e-9) < 2:
        raise ValueError("対応点が一直線上にあり、回転を決定できません")
    return T, rms


@dataclass
class ICPResult:
    transform: np.ndarray          # source → target (初期変換込み)
    rms: float                     # 残差RMS（法線があれば点-面距離, なければ点-点距離, 閾値内）
    fitness: float                 # 閾値内に対応がある source 点の割合（重複率）
    n_correspondences: int
    iterations: int
    converged: bool
    max_distance: float
    history: list = field(default_factory=list)


def _best_fit_point_to_plane(src, dst, nrm):
    """線形化した point-to-plane 最小化 (6自由度)."""
    c = np.cross(src, nrm)
    A = np.hstack([c, nrm])
    b = np.sum((dst - src) * nrm, axis=1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    rx, ry, rz, tx, ty, tz = x
    # 小角度近似の回転を正規直交化
    R = np.array([[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def icp(source, target, target_normals=None, init: Optional[np.ndarray] = None,
        max_iterations: int = 60, max_distance: Optional[float] = None,
        method: str = "point_to_plane", sample: int = 30000, tolerance: float = 1e-7,
        rng=None, progress: Optional[Progress] = None) -> ICPResult:
    """ICP 精密位置合わせ（source を target に合わせる）."""
    progress = ensure_progress(progress)
    rng = rng or np.random.default_rng(0)
    S = np.asarray(source, float)
    Tg = np.asarray(target, float)
    if len(S) > sample:
        S = S[rng.choice(len(S), sample, replace=False)]
    tree = cKDTree(Tg)
    T = np.eye(4) if init is None else np.asarray(init, float).copy()
    if method == "point_to_plane" and target_normals is None:
        method = "point_to_point"
    N = None if target_normals is None else np.asarray(target_normals, float)
    cur = apply_transform(S, T)
    d0, _ = tree.query(cur, workers=-1)
    if max_distance is None:
        max_distance = float(max(np.median(d0) * 3.0, 1e-6))
    hist = []
    prev = np.inf
    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        progress.report(it / max_iterations, f"ICP 反復 {it}")
        cur = apply_transform(S, T)
        d, idx = tree.query(cur, workers=-1, distance_upper_bound=max_distance)
        ok = np.isfinite(d)
        if ok.sum() < 6:
            break
        src = cur[ok]
        dst = Tg[idx[ok]]
        if method == "point_to_plane":
            dT = _best_fit_point_to_plane(src, dst, N[idx[ok]])
        else:
            dT, _ = kabsch(src, dst)
        T = dT @ T
        rms = float(np.sqrt(np.mean(d[ok] ** 2)))
        hist.append(rms)
        if abs(prev - rms) < tolerance * max(1.0, rms):
            converged = True
            break
        prev = rms
    cur = apply_transform(S, T)
    d, idx = tree.query(cur, workers=-1)
    ok = d <= max_distance
    if N is not None and ok.any():
        dp = np.abs(np.sum((cur[ok] - Tg[idx[ok]]) * N[idx[ok]], axis=1))
        rms = float(np.sqrt(np.mean(dp ** 2)))
    else:
        rms = float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("nan")
    return ICPResult(T, rms, float(ok.mean()), int(ok.sum()), it, converged, max_distance, hist)


def evaluate_alignment(source_world, target, threshold: float) -> dict:
    """位置合わせ評価: 重複領域の割合と残差統計."""
    tree = cKDTree(np.asarray(target, float))
    d, _ = tree.query(np.asarray(source_world, float), workers=-1)
    ok = d <= threshold
    out = {"threshold": threshold, "overlap_ratio": float(ok.mean()),
           "rms": float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("nan"),
           "mean": float(d[ok].mean()) if ok.any() else float("nan"),
           "p95": float(np.percentile(d[ok], 95)) if ok.any() else float("nan"),
           "n": int(len(d))}
    return out
