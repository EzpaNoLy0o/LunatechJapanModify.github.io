"""認識した幾何要素・断面スケッチから CAD 形状 (B-Rep) を再構築する.

CAD 形状はスキャンメッシュとは別オブジェクトとして管理し、
元メッシュに対する偏差を計算できるようにする。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core.geometry import normalize, orthonormal_basis
from ..core.sketch_fit import SketchEntity, segment_polyline, fit_spline_2d
from . import occ_backend as occ

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 幾何要素 → B-Rep
# ---------------------------------------------------------------------------

def plane_to_face(plane, margin: float = 0.0):
    ext = np.asarray(plane.extent, float).copy()
    ext[0] -= margin
    ext[1] += margin
    return occ.make_plane_face(plane.origin, plane.normal, plane.u_axis, ext), {
        "type": "plane_face", "source": plane.id, "margin": margin}


def cylinder_to_shape(cyl, mode: str = "face", extend: float = 0.0):
    """mode='face': 円筒面（360°）/ 'partial': スキャン範囲の部分円筒面 / 'solid': 円柱ソリッド."""
    t0, t1 = cyl.t_range[0] - extend, cyl.t_range[1] + extend
    if mode == "solid":
        s = occ.make_cylinder_solid(cyl.axis_point, cyl.axis_dir, cyl.radius, t0, t1)
    elif mode == "partial" and cyl.coverage_deg < 359:
        # スキャンされた円弧の範囲（フィット時に記録した基準方向と開始角）で部分円筒面を作る
        ref = cyl.arc_ref
        if ref is not None:
            ref = np.asarray(ref, float) - np.dot(ref, cyl.axis_dir) * cyl.axis_dir
        a0 = np.radians(cyl.arc_start_deg)
        s = occ.make_cylinder_face(cyl.axis_point, cyl.axis_dir, cyl.radius, t0, t1, a0,
                                   a0 + np.radians(cyl.coverage_deg), x_dir=ref)
    else:
        s = occ.make_cylinder_face(cyl.axis_point, cyl.axis_dir, cyl.radius, t0, t1)
    return s, {"type": f"cylinder_{mode}", "source": cyl.id, "extend": extend}


def hole_to_shape(hole, mode: str = "edge", depth: Optional[float] = None):
    """mode='edge': 円形エッジ / 'cylinder': 穴の円柱（カット用ソリッド）."""
    if mode == "cylinder":
        d = depth or hole.depth or hole.diameter
        s = occ.make_cylinder_solid(hole.center, hole.axis, hole.diameter / 2, -float(d), 0.0)
    else:
        s = occ.make_circle_edge(hole.center, hole.axis, hole.diameter / 2)
    return s, {"type": f"hole_{mode}", "source": hole.id}


def sphere_to_solid(sph):
    return occ.make_sphere_solid(sph.center, sph.radius), {"type": "sphere", "source": sph.id}


def axis_to_edge(ax):
    a, b = ax.end_points()
    return occ.make_line_edge(a, b), {"type": "axis_edge", "source": ax.id}


# ---------------------------------------------------------------------------
# 断面 → スケッチ → 押し出し / 回転体
# ---------------------------------------------------------------------------

def sketch_section(section, tol: float = 0.1, mode: str = "lines_arcs", max_radius: float = 5000.0,
                   loops: Optional[list] = None):
    """断面の各ポリラインをスケッチ要素化して section.sketch に格納（呼び出し側でUndo管理）."""
    idx = range(len(section.polylines)) if loops is None else loops
    new = [list(s) for s in section.sketch] if section.sketch else [[] for _ in section.polylines]
    while len(new) < len(section.polylines):
        new.append([])
    for i in idx:
        uv = section.to_2d(section.polylines[i])
        closed = section.closed[i]
        if mode == "spline":
            ents = [fit_spline_2d(uv, closed, tol)]
        else:
            ents = segment_polyline(uv, closed, tol, max_radius)
        new[i] = [e.to_dict() for e in ents]
    return new


def _loop_entities(section, i, tol):
    ents = section.sketch_entities(i)
    if not ents:
        uv = section.to_2d(section.polylines[i])
        ents = segment_polyline(uv, section.closed[i], tol)
    return ents


def section_profile_face(section, loops=None, tol: float = 0.1):
    """断面の閉ループから平面プロファイル面（内側ループは穴）を作成."""
    frame = (section.origin, section.u_axis, section.v_axis)
    idx = [i for i in (range(len(section.polylines)) if loops is None else loops) if section.closed[i]]
    if not idx:
        raise ValueError("閉じた断面曲線がありません。押し出しには閉じた輪郭が必要です。")
    loop_ents = [_loop_entities(section, i, tol) for i in idx]
    return occ.profile_face_from_loops(loop_ents, frame)


def extrude_section(section, distance: float, loops=None, symmetric: bool = False,
                    tol: float = 0.1, direction=None):
    face = section_profile_face(section, loops, tol)
    d = section.normal if direction is None else np.asarray(direction, float)
    solid = occ.extrude(face, d, distance, symmetric)
    return solid, {"type": "extrude", "source": section.id, "distance": distance,
                   "symmetric": symmetric, "loops": loops, "tol": tol}


def revolve_section(section, axis_point, axis_dir, loop: int = 0, angle_deg: float = 360.0,
                    tol: float = 0.1, half: bool = True):
    """断面ループを軸周りに回転して回転体を作成.

    half=True: 断面平面上で軸の片側（ループの重心がある側）だけを切り出してプロファイルにする
              （軸を含む断面から、ブッシュ・プーリー等の単純な回転体を作る用途）。
    """
    from shapely.geometry import Polygon, box
    a = normalize(np.asarray(axis_dir, float))
    n = section.normal
    if abs(np.dot(a, n)) > 1e-3:
        raise ValueError("回転軸が断面平面上にありません（軸を含む断面を作成してください）")
    uv = section.to_2d(section.polylines[loop])
    if half:
        # 平面座標での軸線
        p2 = section.to_2d(np.asarray(axis_point, float)[None])[0]
        d2 = np.array([a @ section.u_axis, a @ section.v_axis])
        d2 /= np.linalg.norm(d2)
        nrm2 = np.array([-d2[1], d2[0]])
        side = np.sign(np.mean((uv - p2) @ nrm2)) or 1.0
        big = 10 * (np.ptp(uv, axis=0).max() + np.linalg.norm(uv.mean(0) - p2) + 1)
        # 軸に沿った大きな半平面ボックス（回転座標系で作成）
        Rm = np.column_stack([d2, nrm2 * side])
        local = (uv - p2) @ Rm
        poly = Polygon(local).buffer(0)
        clip = poly.intersection(box(-big, 0.0, big, big))
        if clip.is_empty:
            raise ValueError("軸の片側に断面がありません")
        if clip.geom_type == "MultiPolygon":
            clip = max(clip.geoms, key=lambda g: g.area)
        loc = np.asarray(clip.exterior.coords)[:-1]
        uv = loc @ Rm.T + p2
    ents = segment_polyline(uv, True, tol)
    frame = (section.origin, section.u_axis, section.v_axis)
    face = occ.make_planar_face(occ.make_wire(ents, frame))
    solid = occ.revolve(face, axis_point, a, angle_deg)
    return solid, {"type": "revolve", "source": section.id, "angle": angle_deg, "loop": loop,
                   "half": half, "tol": tol}


# ---------------------------------------------------------------------------
# 自由曲面（Stage 3 の基礎機能）
# ---------------------------------------------------------------------------

def _resample_loop(P, n, closed):
    from ..core.section import resample_polyline
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1)
    L = seg.sum()
    return resample_polyline(P, L / n, closed)[:n] if closed else resample_polyline(P, L / (n - 1), False)


def loft_curves(curves, closed: bool, solid: bool = False, ruled: bool = False, n_samples: int = 80):
    """複数断面曲線からロフト曲面（BRepOffsetAPI_ThruSections）."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    if len(curves) < 2:
        raise ValueError("ロフトには2本以上の断面曲線が必要です")
    ts = BRepOffsetAPI_ThruSections(bool(solid and closed), bool(ruled), 1e-6)
    ref_start = None
    prev = None
    for C in curves:
        P = _resample_loop(C, n_samples, closed)
        if closed:
            # 始点と周回方向を前の断面に揃える（ねじれ防止）
            if prev is not None:
                c0, c1 = prev.mean(0), P.mean(0)
                k = int(np.argmin(np.linalg.norm((P - c1) - (prev[0] - c0), axis=1)))
                P = np.roll(P, -k, axis=0)
                if np.dot(np.cross(prev[1] - prev[0], prev[2] - prev[0]),
                          np.cross(P[1] - P[0], P[2] - P[0])) < 0:
                    P = np.vstack([P[:1], P[1:][::-1]])
        else:
            if prev is not None and np.linalg.norm(P[0] - prev[0]) > np.linalg.norm(P[-1] - prev[0]):
                P = P[::-1]
        prev = P
        edge = occ.make_bspline_edge(P, closed=closed, tol=1e-3)
        mk = BRepBuilderAPI_MakeWire(edge)
        ts.AddWire(mk.Wire())
    ts.Build()
    if not ts.IsDone():
        raise RuntimeError("ロフト曲面の生成に失敗しました")
    return ts.Shape(), {"type": "loft", "n_sections": len(curves), "closed": closed, "solid": solid}


def _min_area_rect_axes(uv):
    from ..core.geometry import min_area_rect_axes
    return min_area_rect_axes(uv)


def _clamped_knots(a, b, n_ctrl, k):
    inner = np.linspace(a, b, n_ctrl - k + 1)[1:-1]
    return np.r_[[a] * (k + 1), inner, [b] * (k + 1)]


def _design(x, t, k):
    from scipy.interpolate import BSpline
    x = np.clip(x, t[k], t[-k - 1])
    D = BSpline.design_matrix(x, t, k).tocsr()
    D.sort_indices()
    return D


def fit_bspline_surface(points, n_ctrl_u: int = 10, n_ctrl_v: int = 10, degree: int = 3,
                        smoothing: float = 1e-3, normal_hint=None):
    """点群（選択領域）に Bスプライン曲面を最小二乗フィットする（制御点数を指定）.

    手法: 最小二乗平面上の (u, v)（データを囲む最小面積矩形の向き）で高さ関数 h(u,v) を
    テンソル積 Bスプライン + 2階差分ペナルティ（P-spline）で最小二乗近似し、
    Greville 点を用いて 3D の Bスプライン曲面（OCC Geom_BSplineSurface）へ厳密に変換する。
    smoothing: 平滑化の強さ（0 で純粋な最小二乗。データのない領域を滑らかに補うため小さな値を推奨）
    制約: 最小二乗平面から見て一価（オーバーハングがない）な領域に限る。
          曲面は矩形範囲全体に作られる（輪郭トリムは未実装）。
    戻り値 (face, info)。info に残差統計（入力点の高さ方向残差）を含む。
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from ..core.fitting.plane import fit_plane_lsq

    P = np.asarray(points, float)
    k = int(degree)
    if not 1 <= k <= 5:
        raise ValueError("次数は1〜5で指定してください")
    nu, nv = int(n_ctrl_u), int(n_ctrl_v)
    if nu < k + 1 or nv < k + 1:
        raise ValueError(f"制御点数は各方向 {k + 1} 以上が必要です")
    if len(P) < nu * nv:
        raise ValueError(f"点数が不足しています（制御点 {nu}×{nv} に対して {len(P)} 点）")
    pl = fit_plane_lsq(P, orient_hint=normal_hint)
    o, n = pl.origin, pl.normal
    rel = P - o
    uv0 = np.column_stack([rel @ pl.u_axis, rel @ pl.v_axis])
    e1, e2 = _min_area_rect_axes(uv0)
    u = e1[0] * pl.u_axis + e1[1] * pl.v_axis
    v = np.cross(n, u)
    U, Vv, H = rel @ u, rel @ v, rel @ n
    u0, u1, v0, v1 = U.min(), U.max(), Vv.min(), Vv.max()
    fu = _clamped_knots(u0, u1, nu, k)
    fv = _clamped_knots(v0, v1, nv, k)
    Bu, Bv = _design(U, fu, k), _design(Vv, fv, k)
    m = len(P)
    iu = Bu.indices.reshape(m, k + 1)
    iv = Bv.indices.reshape(m, k + 1)
    du = Bu.data.reshape(m, k + 1)
    dv = Bv.data.reshape(m, k + 1)
    cols = (iu[:, :, None] * nv + iv[:, None, :]).reshape(m, -1)
    vals = (du[:, :, None] * dv[:, None, :]).reshape(m, -1)
    rows = np.repeat(np.arange(m), (k + 1) ** 2)
    A = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(m, nu * nv))
    AtA = (A.T @ A).tocsr()
    Du = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nu - 2, nu)) if nu > 2 else sp.csr_matrix((0, nu))
    Dv = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nv - 2, nv)) if nv > 2 else sp.csr_matrix((0, nv))
    Pen = sp.kron(Du.T @ Du, sp.eye(nv)) + sp.kron(sp.eye(nu), Dv.T @ Dv)
    lam = float(smoothing) * AtA.diagonal().mean() / max(Pen.diagonal().mean(), 1e-30)
    M = (AtA + lam * Pen + 1e-12 * sp.eye(nu * nv)).tocsc()
    c = spsolve(M, A.T @ H)
    res = A @ c - H
    C = c.reshape(nu, nv)
    gu = np.array([fu[i + 1:i + k + 1].mean() for i in range(nu)])   # Greville 点
    gv = np.array([fv[j + 1:j + k + 1].mean() for j in range(nv)])
    poles = (o[None, None, :] + gu[:, None, None] * u[None, None, :] + gv[None, :, None] * v[None, None, :]
             + C[:, :, None] * n[None, None, :])
    face = occ.make_bspline_surface_face(poles, fu, fv, k, k)
    a = np.abs(res)
    info = {"type": "bspline_surface", "n_ctrl_u": nu, "n_ctrl_v": nv, "degree": k,
            "smoothing": smoothing, "n_points": int(m), "rms": float(np.sqrt(np.mean(res ** 2))),
            "max_abs": float(a.max()), "p95": float(np.percentile(a, 95)),
            "u_range": [float(u0), float(u1)], "v_range": [float(v0), float(v1)]}
    return face, info


# ---------------------------------------------------------------------------
# 偏差解析
# ---------------------------------------------------------------------------

def deviation_points_to_shape(points, shape, deflection: float = 0.02, max_points: int = 200000,
                              rng=None):
    """点群（メッシュ頂点）と CAD 形状の距離（符号付き）.

    CAD 形状を細かくテッセレーションし、VTK の vtkImplicitPolyDataDistance で計算する。
    戻り値 (indices, distances)。符号は CAD 面の法線方向が正。
    """
    import pyvista as pv
    rng = rng or np.random.default_rng(0)
    P = np.asarray(points, float)
    idx = np.arange(len(P))
    if len(P) > max_points:
        idx = np.sort(rng.choice(len(P), max_points, replace=False))
    bb = occ.bbox(shape)
    diag = float(np.linalg.norm(np.subtract(bb[1], bb[0]))) if bb else 1.0
    deflection = max(float(deflection), diag * 1e-4)
    V, F = occ.tessellate(shape, deflection, 0.3)
    if len(F) > 5_000_000:
        raise RuntimeError("偏差計算用のテッセレーションが大きすぎます。許容値を大きくしてください。")
    if len(F) == 0:
        raise ValueError("CAD形状に面がありません（エッジのみの形状は偏差計算できません）")
    surf = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    surf = surf.compute_normals(cell_normals=True, point_normals=True, auto_orient_normals=False,
                                consistent_normals=False)
    pc = pv.PolyData(P[idx])
    out = pc.compute_implicit_distance(surf)
    d = np.asarray(out["implicit_distance"], float)
    return idx, d


def deviation_stats(d, tol: float) -> dict:
    d = np.asarray(d, float)
    a = np.abs(d)
    return {"n": int(len(d)), "mean": float(d.mean()), "rms": float(np.sqrt(np.mean(d ** 2))),
            "max_abs": float(a.max()), "p95": float(np.percentile(a, 95)), "std": float(d.std()),
            "tol": tol, "within_tol": float(np.mean(a <= tol))}


def analytic_deviation(feature, points) -> np.ndarray:
    """幾何要素（平面/円筒/球/穴は円筒扱い不可）に対する解析的な符号付き偏差."""
    P = np.asarray(points, float)
    k = feature.kind
    if k == "plane":
        return (P - feature.origin) @ feature.normal
    if k == "cylinder":
        a = feature.axis_dir
        w = P - feature.axis_point
        t = w @ a
        return np.linalg.norm(w - np.outer(t, a), axis=1) - feature.radius
    if k == "sphere":
        return np.linalg.norm(P - feature.center, axis=1) - feature.radius
    raise ValueError(f"解析的偏差に未対応の要素: {k}")
