"""断面ポリライン（2D）から直線・円弧・円・スプラインのスケッチ要素を作成する.

自動分割アルゴリズム（split-and-fit）:
  1. 閉じた輪郭全体が円で近似できれば「円」
  2. 区間の両端点を通る直線（弦）からの最大偏差が許容値以内なら「直線」
  3. 両端点と中間点を通る円弧が許容値以内なら「円弧」
  4. どちらも不可なら最大偏差点で分割して再帰
  5. 隣接する同一直線 / 同一円の要素を統合

端点は元のポリラインの点を共有するため、要素同士は必ず連続（閉じた輪郭は閉じたまま）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fitting.circle import fit_circle_2d


@dataclass
class SketchEntity:
    kind: str                 # 'line' | 'arc' | 'circle' | 'spline'
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None
    mid: Optional[np.ndarray] = None       # 円弧上の中間点
    center: Optional[np.ndarray] = None
    radius: Optional[float] = None
    ctrl: Optional[np.ndarray] = None      # spline 制御点
    knots: Optional[np.ndarray] = None
    degree: int = 3
    fit_points: Optional[np.ndarray] = None
    closed: bool = False
    max_error: float = 0.0
    n_points: int = 0

    def to_dict(self):
        d = {"kind": self.kind, "closed": self.closed, "max_error": self.max_error,
             "n_points": self.n_points, "degree": self.degree}
        for k in ("start", "end", "mid", "center", "ctrl", "knots", "fit_points"):
            v = getattr(self, k)
            d[k] = None if v is None else np.asarray(v).tolist()
        d["radius"] = self.radius
        return d

    @staticmethod
    def from_dict(d):
        e = SketchEntity(kind=d["kind"])
        for k in ("start", "end", "mid", "center", "ctrl", "knots", "fit_points"):
            v = d.get(k)
            setattr(e, k, None if v is None else np.asarray(v, float))
        e.radius = d.get("radius")
        e.closed = d.get("closed", False)
        e.max_error = d.get("max_error", 0.0)
        e.n_points = d.get("n_points", 0)
        e.degree = d.get("degree", 3)
        return e

    def sample(self, n: int = 64) -> np.ndarray:
        """表示用に要素を折れ線化."""
        if self.kind == "line":
            return np.vstack([self.start, self.end])
        if self.kind == "circle":
            t = np.linspace(0, 2 * np.pi, n + 1)
            return self.center + self.radius * np.column_stack([np.cos(t), np.sin(t)])
        if self.kind == "arc":
            a0, a1 = arc_angles(self.center, self.start, self.mid, self.end)
            t = np.linspace(a0, a1, n)
            return self.center + self.radius * np.column_stack([np.cos(t), np.sin(t)])
        if self.kind == "spline":
            from scipy.interpolate import splev
            u = np.linspace(0, 1, max(n, 4 * len(self.ctrl)))
            tck = (self.knots, [self.ctrl[:, 0], self.ctrl[:, 1]], self.degree)
            x, y = splev(u, tck)
            return np.column_stack([x, y])
        raise ValueError(self.kind)


def circle_from_3pts(a, b, c):
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = a @ a, b @ b, c @ c
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.linalg.norm(a - ctr))


def arc_angles(center, start, mid, end):
    """start→mid→end を通る円弧の開始・終了角（終了角は回転方向を考慮）."""
    a0 = np.arctan2(start[1] - center[1], start[0] - center[0])
    am = np.arctan2(mid[1] - center[1], mid[0] - center[0])
    a1 = np.arctan2(end[1] - center[1], end[0] - center[0])
    two = 2 * np.pi
    ccw_m = (am - a0) % two
    ccw_e = (a1 - a0) % two
    if ccw_m <= ccw_e:  # 反時計回り
        return a0, a0 + ccw_e
    return a0, a0 - ((a0 - a1) % two)


def arc_is_ccw(center, start, mid, end) -> bool:
    a0, a1 = arc_angles(center, start, mid, end)
    return a1 >= a0


def _chord_dev(P):
    a, b = P[0], P[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-12:
        return np.linalg.norm(P - a, axis=1)
    nrm = np.array([-ab[1], ab[0]]) / L
    return np.abs((P - a) @ nrm)


def _arc_dev(P, max_radius):
    if len(P) < 4:
        return None
    m = len(P) // 2
    # 中間点は最小二乗円に投影して滑らかにする
    try:
        cf = fit_circle_2d(P)
    except ValueError:
        return None
    if not np.isfinite(cf.radius) or cf.radius > max_radius:
        return None
    dirm = P[m] - cf.center
    nm = np.linalg.norm(dirm)
    mid = cf.center + cf.radius * dirm / nm if nm > 1e-12 else P[m]
    c3 = circle_from_3pts(P[0], mid, P[-1])
    if c3 is None:
        return None
    ctr, r = c3
    if r > max_radius:
        return None
    dev = np.abs(np.linalg.norm(P - ctr, axis=1) - r)
    # 円弧の範囲外の点（反対側）がないか確認: 角度順序チェック
    a0, a1 = arc_angles(ctr, P[0], mid, P[-1])
    ang = np.arctan2(P[:, 1] - ctr[1], P[:, 0] - ctr[0])
    if a1 >= a0:
        rel = (ang - a0) % (2 * np.pi)
        span = a1 - a0
    else:
        rel = (a0 - ang) % (2 * np.pi)
        span = a0 - a1
    if np.any(rel > span + 1e-6):
        return None
    return dev, ctr, r, mid, span


def segment_polyline(points2d, closed: bool, tol: float = 0.1, max_radius: float = 1e4,
                     min_arc_points: int = 5, min_arc_deg: float = 5.0) -> list:
    """2Dポリラインを直線・円弧・円に自動分割する."""
    P = np.asarray(points2d, float)
    # 重複点の除去
    keep = np.ones(len(P), bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9
    P = P[keep]
    if closed and len(P) > 2 and np.linalg.norm(P[0] - P[-1]) < 1e-9:
        P = P[:-1]
    if len(P) < 2:
        return []
    if closed and len(P) >= 8:
        try:
            cf = fit_circle_2d(P)
            dev = np.abs(np.linalg.norm(P - cf.center, axis=1) - cf.radius)
        except ValueError:
            cf = None
        if cf is not None and dev.max() <= tol and cf.radius < max_radius and cf.coverage_deg > 300:
            return [SketchEntity("circle", center=cf.center, radius=cf.radius, closed=True,
                                 max_error=float(dev.max()), n_points=len(P))]

    ents: list = []

    def rec(i, j, depth=0):
        Q = P[i:j + 1]
        dev = _chord_dev(Q)
        if dev.max() <= tol or len(Q) <= 2:
            e = SketchEntity("line", start=P[i].copy(), end=P[j].copy(),
                             max_error=float(dev.max()), n_points=len(Q))
            e.i0, e.i1 = i, j
            ents.append(e)
            return
        if len(Q) >= min_arc_points:
            a = _arc_dev(Q, max_radius)
            if a is not None:
                adev, ctr, r, mid, span = a
                if adev.max() <= tol and np.degrees(span) >= min_arc_deg:
                    e = SketchEntity("arc", start=P[i].copy(), end=P[j].copy(), mid=mid,
                                     center=ctr, radius=r, max_error=float(adev.max()), n_points=len(Q))
                    e.i0, e.i1 = i, j
                    ents.append(e)
                    return
        k = i + int(np.argmax(dev))
        if k <= i or k >= j:
            k = (i + j) // 2
        rec(i, k, depth + 1)
        rec(k, j, depth + 1)

    if closed:
        P = np.vstack([P, P[:1]])
        # 開始点から最も遠い点で2分割
        far = int(np.argmax(np.linalg.norm(P[:-1] - P[0], axis=1)))
        far = max(1, far)
        rec(0, far)
        rec(far, len(P) - 1)
    else:
        rec(0, len(P) - 1)
    ents = _merge(ents, P, tol, max_radius)
    for e in ents:
        e.closed = False
    return ents


def _merge(ents, P, tol, max_radius):
    """隣接する同一直線・同一円の要素を統合（統合後の要素を元の点列で再検証する）."""
    changed = True
    while changed and len(ents) > 1:
        changed = False
        out = [ents[0]]
        for e in ents[1:]:
            prev = out[-1]
            i0, i1 = getattr(prev, "i0", None), getattr(e, "i1", None)
            if i0 is None or i1 is None or prev.kind != e.kind or prev.kind not in ("line", "arc"):
                out.append(e)
                continue
            Q = P[i0:i1 + 1]
            if prev.kind == "line":
                d1 = np.asarray(prev.end - prev.start)
                d2 = np.asarray(e.end - e.start)
                cosang = np.dot(d1, d2) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-15)
                dev = _chord_dev(Q)
                if cosang > np.cos(np.radians(5.0)) and dev.max() <= tol:
                    m = SketchEntity("line", start=prev.start, end=e.end, max_error=float(dev.max()),
                                     n_points=len(Q))
                    m.i0, m.i1 = i0, i1
                    out[-1] = m
                    changed = True
                    continue
            else:
                a = _arc_dev(Q, max_radius)
                if a is not None and a[0].max() <= tol and a[4] < 2 * np.pi - 1e-3:
                    adev, ctr, r, mid, span = a
                    m = SketchEntity("arc", start=prev.start, end=e.end, mid=mid, center=ctr, radius=r,
                                     max_error=float(adev.max()), n_points=len(Q))
                    m.i0, m.i1 = i0, i1
                    out[-1] = m
                    changed = True
                    continue
            out.append(e)
        ents = out
    return ents


def fit_spline_2d(points2d, closed: bool = False, tol: float = 0.1, degree: int = 3,
                  max_ctrl: int = 60) -> SketchEntity:
    """平滑化Bスプライン（scipy splprep）。許容誤差 tol から平滑化係数を決める."""
    from scipy.interpolate import splev, splprep
    P = np.asarray(points2d, float)
    keep = np.ones(len(P), bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9
    P = P[keep]
    if closed and np.linalg.norm(P[0] - P[-1]) > 1e-9:
        P = np.vstack([P, P[:1]])
    if len(P) <= degree:
        raise ValueError("スプラインには点が不足しています")
    s = len(P) * (tol / 2.0) ** 2
    for _ in range(12):
        tck, u = splprep([P[:, 0], P[:, 1]], s=s, k=degree, per=1 if closed else 0, quiet=2)
        x, y = splev(u, tck)
        err = np.hypot(x - P[:, 0], y - P[:, 1]).max()
        nctrl = len(tck[1][0])
        if err <= tol and nctrl <= max_ctrl:
            break
        if err > tol:
            s *= 0.4
        else:
            s *= 2.0
    n_ctrl = len(tck[0]) - int(tck[2]) - 1
    ctrl = np.column_stack(tck[1])[:n_ctrl]
    return SketchEntity("spline", ctrl=ctrl, knots=np.asarray(tck[0]), degree=int(tck[2]),
                        fit_points=P, closed=closed, max_error=float(err), n_points=len(P),
                        start=P[0].copy(), end=P[-1].copy())


def entities_bbox(ents) -> np.ndarray:
    pts = np.vstack([e.sample(32) for e in ents]) if ents else np.zeros((1, 2))
    return np.array([pts.min(0), pts.max(0)])
