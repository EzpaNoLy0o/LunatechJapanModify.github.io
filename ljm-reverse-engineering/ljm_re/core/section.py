"""メッシュ断面（平面との交線）の抽出.

交点はメッシュのエッジ単位で一意に識別し、位相に基づいて連結するため、
許容誤差に依存しない堅牢なポリライン化ができる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import normalize, orthonormal_basis
from .progress import Progress, ensure_progress


@dataclass
class Polyline:
    points: np.ndarray      # (k, 3) ワールド座標
    closed: bool

    def length(self) -> float:
        p = self.points
        if len(p) < 2:
            return 0.0
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if self.closed:
            L += float(np.linalg.norm(p[0] - p[-1]))
        return L


@dataclass
class SectionResult:
    origin: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    polylines: list = field(default_factory=list)

    def to_2d(self, pts) -> np.ndarray:
        rel = np.asarray(pts, float) - self.origin
        return np.column_stack([rel @ self.u_axis, rel @ self.v_axis])

    def to_3d(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float)
        return self.origin + np.outer(uv[:, 0], self.u_axis) + np.outer(uv[:, 1], self.v_axis)


def plane_basis(normal) -> tuple[np.ndarray, np.ndarray]:
    """断面平面の面内座標軸。座標軸に平行な断面では分かりやすい軸を採用する.

    Z断面 → (X, Y), X断面 → (Y, Z), Y断面 → (X, Z)
    """
    n = normalize(np.asarray(normal, float))
    ax = np.argmax(np.abs(n))
    if abs(n[ax]) > 0.999999:
        if ax == 2:
            u = np.array([1.0, 0, 0])
        elif ax == 0:
            u = np.array([0, 1.0, 0])
        else:
            u = np.array([1.0, 0, 0])
        v = np.cross(n, u)
        if ax == 1:  # Y断面は v=+Z になるよう符号調整
            if v[2] < 0:
                u, v = -u, -v
                # n×u = v を保つため u を反転すると v も反転する
        return u, normalize(v)
    return orthonormal_basis(n)


def section_mesh(V, F, origin, normal, u_axis=None, v_axis=None) -> SectionResult:
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    o = np.asarray(origin, float)
    n = normalize(np.asarray(normal, float))
    if u_axis is None or v_axis is None:
        u_axis, v_axis = plane_basis(n)
    res = SectionResult(o, n, np.asarray(u_axis, float), np.asarray(v_axis, float), [])
    if len(F) == 0:
        return res
    d = (V - o) @ n
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    # 平面上にちょうど乗る頂点はわずかに正側へずらす（縮退回避）
    d = np.where(np.abs(d) < 1e-12 * scale, 1e-12 * scale, d)
    s = d[F] > 0
    cnt = s.sum(axis=1)
    cut = (cnt == 1) | (cnt == 2)
    if not cut.any():
        return res
    Fc = F[cut]
    sc = s[cut]
    # 各面について符号が変わるエッジ2本を求める
    edges = np.stack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], axis=1)  # (k,3,2)
    se = np.stack([sc[:, 0] != sc[:, 1], sc[:, 1] != sc[:, 2], sc[:, 2] != sc[:, 0]], axis=1)
    # 各面ちょうど2本
    ei = np.argsort(~se, axis=1, kind="stable")[:, :2]
    k = len(Fc)
    e2 = edges[np.arange(k)[:, None], ei]  # (k,2,2)
    e2s = np.sort(e2, axis=2)
    nv = len(V) + 1
    keys = e2s[:, :, 0] * nv + e2s[:, :, 1]  # (k,2)
    # 交点座標（エッジごと）
    uk, inv = np.unique(keys.ravel(), return_inverse=True)
    a = uk // nv
    b = uk % nv
    t = d[a] / (d[a] - d[b])
    P = V[a] + t[:, None] * (V[b] - V[a])
    seg = inv.reshape(-1, 2)
    # グラフ連結
    nnode = len(uk)
    nbrs = [[] for _ in range(nnode)]
    for i, j in seg.tolist():
        if i == j:
            continue
        nbrs[i].append(j)
        nbrs[j].append(i)
    visited = np.zeros(nnode, bool)
    deg = np.array([len(x) for x in nbrs])
    # 開いた鎖は次数1の端点から開始
    order = list(np.nonzero(deg == 1)[0]) + list(np.nonzero(deg >= 2)[0])
    for start in order:
        if visited[start] or deg[start] == 0:
            continue
        chain = [start]
        visited[start] = True
        prev, cur = -1, start
        closed = False
        while True:
            nxt = None
            for c in nbrs[cur]:
                if c != prev and not visited[c]:
                    nxt = c
                    break
            if nxt is None:
                if len(chain) > 2 and start in nbrs[cur] and prev != start:
                    closed = True
                break
            visited[nxt] = True
            chain.append(nxt)
            prev, cur = cur, nxt
        if len(chain) >= 2:
            res.polylines.append(Polyline(P[np.array(chain)], closed))
    res.polylines.sort(key=lambda pl: -pl.length())
    return res


def section_series(V, F, origin, normal, offsets, progress: Optional[Progress] = None):
    """法線方向に offsets だけずらした連続断面."""
    progress = ensure_progress(progress)
    n = normalize(np.asarray(normal, float))
    u, v = plane_basis(n)
    out = []
    for i, off in enumerate(offsets):
        progress.report(i / max(1, len(offsets)), f"断面 {i + 1}/{len(offsets)}")
        out.append(section_mesh(V, F, np.asarray(origin, float) + off * n, n, u, v))
    return out


def resample_polyline(points, spacing: float, closed: bool = False) -> np.ndarray:
    P = np.asarray(points, float)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if L <= 0:
        return P[:1]
    m = max(2, int(np.ceil(L / spacing)) + 1)
    t = np.linspace(0, L, m, endpoint=not closed)
    out = np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])
    return out
