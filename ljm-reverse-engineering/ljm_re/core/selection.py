"""メッシュ上の領域選択アルゴリズム.

- 領域拡張（スマート選択）: 隣接面の法線角度が閾値以内の面を辿る
- ブラシ（球）選択: クリック点から半径以内で連結した面
- 矩形選択: 画面座標へ投影して判定（深度バッファで前面のみも可）
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .mesh_ops import face_normals


class MeshTopology:
    """選択処理用のキャッシュ（面隣接・KD木など）."""

    def __init__(self, V, F):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, np.int64)
        self.face_normals = face_normals(self.V, self.F)
        self.centroids = self.V[self.F].mean(axis=1)
        self._adj = None
        self._ctree = None
        self._vtree = None
        self._vf = None

    @property
    def adjacency(self):
        """面隣接（エッジ共有）のCSR行列."""
        if self._adj is None:
            F = self.F
            m = len(F)
            e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            e.sort(axis=1)
            fid = np.tile(np.arange(m), 3)
            key = e[:, 0] * (len(self.V) + 1) + e[:, 1]
            order = np.argsort(key, kind="stable")
            ks = key[order]
            fs = fid[order]
            same = ks[1:] == ks[:-1]
            a = fs[:-1][same]
            b = fs[1:][same]
            rows = np.concatenate([a, b])
            cols = np.concatenate([b, a])
            self._adj = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(m, m)).tocsr()
        return self._adj

    @property
    def centroid_tree(self):
        if self._ctree is None:
            self._ctree = cKDTree(self.centroids)
        return self._ctree

    @property
    def vertex_tree(self):
        if self._vtree is None:
            self._vtree = cKDTree(self.V)
        return self._vtree

    def nearest_face(self, point) -> int:
        _, i = self.centroid_tree.query(np.asarray(point, float))
        return int(i)

    def nearest_vertex(self, point) -> int:
        _, i = self.vertex_tree.query(np.asarray(point, float))
        return int(i)


def region_grow(topo: MeshTopology, seed_face: int, angle_tol_deg: float = 8.0,
                seed_dev_deg: Optional[float] = None, max_faces: Optional[int] = None,
                allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """スマート領域選択.

    angle_tol_deg: 隣接する面同士の法線角度の許容値（曲面も辿れる）
    seed_dev_deg : 開始面の法線からの最大偏差（平面を選ぶ場合に指定）
    allowed      : 選択可能な面のマスク（例: 人工面を除外）
    """
    adj = topo.adjacency
    N = topo.face_normals
    cos_tol = np.cos(np.radians(angle_tol_deg))
    cos_seed = np.cos(np.radians(seed_dev_deg)) if seed_dev_deg is not None else -2.0
    m = len(topo.F)
    sel = np.zeros(m, bool)
    if allowed is not None and not allowed[seed_face]:
        return sel
    n0 = N[seed_face]
    sel[seed_face] = True
    q = deque([seed_face])
    indptr, indices = adj.indptr, adj.indices
    count = 1
    while q:
        f = q.popleft()
        nb = indices[indptr[f]:indptr[f + 1]]
        if nb.size == 0:
            continue
        nb = nb[~sel[nb]]
        if nb.size == 0:
            continue
        ok = (N[nb] @ N[f]) >= cos_tol
        if cos_seed > -2.0:
            ok &= (N[nb] @ n0) >= cos_seed
        if allowed is not None:
            ok &= allowed[nb]
        nb = nb[ok]
        sel[nb] = True
        q.extend(nb.tolist())
        count += nb.size
        if max_faces is not None and count >= max_faces:
            break
    return sel


def brush_select(topo: MeshTopology, center, radius: float, connected: bool = True,
                 seed_face: Optional[int] = None) -> np.ndarray:
    """球ブラシ選択。connected=True ならクリック面から連結する面のみ."""
    c = np.asarray(center, float)
    mask = np.zeros(len(topo.F), bool)
    idx = topo.centroid_tree.query_ball_point(c, radius)
    if idx:
        mask[np.asarray(idx)] = True
    # 粗いメッシュ（大きな三角形）でも選べるよう、半径内に頂点を持つ面も含める
    vidx = topo.vertex_tree.query_ball_point(c, radius)
    if vidx:
        vm = np.zeros(len(topo.V), bool)
        vm[np.asarray(vidx)] = True
        mask |= vm[topo.F].any(axis=1)
    seed = topo.nearest_face(c) if seed_face is None else seed_face
    mask[seed] = True
    if not connected:
        return mask
    # mask 内で連結な成分のみ
    adj = topo.adjacency
    out = np.zeros_like(mask)
    out[seed] = True
    q = deque([seed])
    while q:
        f = q.popleft()
        nb = adj.indices[adj.indptr[f]:adj.indptr[f + 1]]
        nb = nb[mask[nb] & ~out[nb]]
        out[nb] = True
        q.extend(nb.tolist())
    return out


def project_to_display(points, mvp: np.ndarray, viewport_size) -> np.ndarray:
    """ワールド点を表示座標 (x, y, depth[0..1]) に投影.

    mvp: 4x4 の (投影 × ビュー) 行列（VTK の GetCompositeProjectionTransformMatrix 相当）
    viewport_size: (width, height) [pixel]
    """
    P = np.asarray(points, float)
    H = np.c_[P, np.ones(len(P))] @ np.asarray(mvp, float).T
    w = H[:, 3:4]
    w = np.where(np.abs(w) < 1e-15, 1e-15, w)
    ndc = H[:, :3] / w
    W, Hh = viewport_size
    x = (ndc[:, 0] + 1) * 0.5 * W
    y = (ndc[:, 1] + 1) * 0.5 * Hh
    z = (ndc[:, 2] + 1) * 0.5
    return np.column_stack([x, y, z])


def rect_select_points(points, mvp, viewport_size, rect, zbuffer: Optional[np.ndarray] = None,
                       depth_tol: float = 2e-3, behind_camera_mask=None) -> np.ndarray:
    """矩形内の点のマスク。rect=(x0, y0, x1, y1) 表示座標(左下原点)。

    zbuffer (H, W) を与えると、前面に見えている点のみ選択する。
    """
    d = project_to_display(points, mvp, viewport_size)
    x0, x1 = sorted((rect[0], rect[2]))
    y0, y1 = sorted((rect[1], rect[3]))
    m = (d[:, 0] >= x0) & (d[:, 0] <= x1) & (d[:, 1] >= y0) & (d[:, 1] <= y1)
    m &= (d[:, 2] >= 0) & (d[:, 2] <= 1)
    if zbuffer is not None and m.any():
        Hh, W = zbuffer.shape
        idx = np.nonzero(m)[0]
        xi = np.clip(d[idx, 0].astype(int), 0, W - 1)
        yi = np.clip(d[idx, 1].astype(int), 0, Hh - 1)
        # 近傍3x3の最大深度と比較（ピクセル境界の取りこぼし対策）
        zmax = zbuffer[yi, xi]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            zmax = np.maximum(zmax, zbuffer[np.clip(yi + dy, 0, Hh - 1), np.clip(xi + dx, 0, W - 1)])
        vis = d[idx, 2] <= zmax + depth_tol
        m[idx[~vis]] = False
    return m


def polygon_select_points(points, mvp, viewport_size, polygon_xy) -> np.ndarray:
    """多角形（投げ縄）内の点のマスク."""
    from matplotlib.path import Path
    d = project_to_display(points, mvp, viewport_size)
    path = Path(np.asarray(polygon_xy, float))
    return path.contains_points(d[:, :2]) & (d[:, 2] >= 0) & (d[:, 2] <= 1)


def faces_from_vertex_mask(F, vmask, mode: str = "all") -> np.ndarray:
    vm = np.asarray(vmask, bool)[np.asarray(F)]
    return vm.all(axis=1) if mode == "all" else vm.any(axis=1)


def vertices_from_face_mask(F, fmask, n_vertices: int) -> np.ndarray:
    out = np.zeros(n_vertices, bool)
    out[np.asarray(F)[np.asarray(fmask, bool)].ravel()] = True
    return out


def grow_selection(topo: MeshTopology, fmask, rings: int = 1) -> np.ndarray:
    m = np.asarray(fmask, bool).copy()
    adj = topo.adjacency
    for _ in range(rings):
        m = m | (adj @ m.astype(np.int8) > 0)
    return m


def shrink_selection(topo: MeshTopology, fmask, rings: int = 1) -> np.ndarray:
    m = np.asarray(fmask, bool).copy()
    adj = topo.adjacency
    for _ in range(rings):
        inv = ~m
        m = m & ~(adj @ inv.astype(np.int8) > 0)
    return m
