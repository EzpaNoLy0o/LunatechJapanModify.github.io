"""フィッティング共通: 統計量・ロバスト推定・共分散."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

MAD_TO_SIGMA = 1.4826


@dataclass
class FitStats:
    """フィッティング品質の統計量.

    - n_points: 入力点数
    - n_used: 最終フィットに使用した点数（インライア）
    - rms / max_abs / mean_abs: 使用点の残差統計 [mm]
    - inlier_ratio: n_used / n_points
    - method: 'lsq' / 'ransac+lsq' など
    - threshold: インライア判定閾値 [mm]（自動決定の場合も値を記録）
    """
    n_points: int
    n_used: int
    rms: float
    max_abs: float
    mean_abs: float
    inlier_ratio: float
    method: str
    threshold: Optional[float] = None
    iterations: Optional[int] = None
    threshold_auto: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FitStats":
        return FitStats(**d)


def residual_stats(residuals: np.ndarray, n_points: int, method: str,
                   threshold=None, iterations=None, threshold_auto=False) -> FitStats:
    r = np.abs(np.asarray(residuals, dtype=float))
    n_used = int(r.size)
    if n_used == 0:
        return FitStats(n_points, 0, float("nan"), float("nan"), float("nan"), 0.0,
                        method, threshold, iterations, threshold_auto)
    return FitStats(
        n_points=int(n_points),
        n_used=n_used,
        rms=float(np.sqrt(np.mean(r ** 2))),
        max_abs=float(r.max()),
        mean_abs=float(r.mean()),
        inlier_ratio=float(n_used / max(1, n_points)),
        method=method,
        threshold=None if threshold is None else float(threshold),
        iterations=iterations,
        threshold_auto=threshold_auto,
    )


def robust_sigma(residuals: np.ndarray) -> float:
    r = np.asarray(residuals, dtype=float)
    if r.size == 0:
        return 0.0
    med = np.median(r)
    return float(MAD_TO_SIGMA * np.median(np.abs(r - med)))


def auto_threshold(residuals: np.ndarray, scale: float, k: float = 3.0) -> float:
    """残差のMADから自動閾値を決める。ノイズゼロ時の下限はスケールの1e-6."""
    s = robust_sigma(residuals)
    return float(max(k * s, 1e-6 * max(scale, 1e-9)))


def covariance_from_jacobian(J: np.ndarray, residuals: np.ndarray) -> Optional[np.ndarray]:
    """ガウス-ニュートン近似の共分散 (J^T J)^-1 * s^2."""
    m, p = J.shape
    if m <= p:
        return None
    s2 = float(np.sum(residuals ** 2) / (m - p))
    JTJ = J.T @ J
    try:
        cov = np.linalg.inv(JTJ) * s2
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JTJ) * s2
    return cov


def numeric_jacobian(fun, x0: np.ndarray, eps: Optional[np.ndarray] = None) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    if eps is None:
        eps = 1e-6 * np.maximum(1.0, np.abs(x0))
    f0 = fun(x0)
    J = np.empty((f0.size, x0.size))
    for i in range(x0.size):
        dx = np.zeros_like(x0)
        dx[i] = eps[i]
        J[:, i] = (fun(x0 + dx) - fun(x0 - dx)) / (2 * eps[i])
    return J


def subsample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def bbox_diagonal(points: np.ndarray) -> float:
    p = np.asarray(points)
    if p.size == 0:
        return 0.0
    return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))


def ransac_iterations(inlier_ratio: float, sample_size: int, prob: float = 0.999,
                      max_iter: int = 10000) -> int:
    w = min(max(inlier_ratio, 1e-6), 1.0 - 1e-12)
    denom = np.log(1.0 - w ** sample_size)
    if denom >= 0:
        return 1
    return int(min(max_iter, np.ceil(np.log(1.0 - prob) / denom)))


@dataclass
class FitWarnings:
    items: list = field(default_factory=list)

    def add(self, msg: str):
        self.items.append(msg)
