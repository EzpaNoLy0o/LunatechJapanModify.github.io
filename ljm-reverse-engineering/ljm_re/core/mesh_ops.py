"""メッシュ・点群の編集処理（非破壊: 常に新しい配列を返す）.

face_flags は面ごとのビットフラグ:
    FLAG_ARTIFICIAL (1) : 穴埋めなどで人工的に生成された面（実測面ではない）
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .progress import Progress, ensure_progress

log = logging.getLogger(__name__)

FLAG_ARTIFICIAL = 1


def compact(V, F, flags=None, colors=None):
    """未参照頂点を除去して添字を詰める."""
    V = np.asarray(V)
    F = np.asarray(F)
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(int(used.sum()))
    V2 = V[used]
    F2 = remap[F]
    c2 = colors[used] if colors is not None else None
    return V2, F2, (flags.copy() if flags is not None else None), c2, used


def keep_faces(V, F, face_mask, flags=None, colors=None):
    face_mask = np.asarray(face_mask, bool)
    F2 = np.asarray(F)[face_mask]
    fl = flags[face_mask] if flags is not None else None
    V2, F2, fl, c2, _ = compact(V, F2, fl, colors)
    return V2, F2, fl, c2


def remove_faces(V, F, face_mask, flags=None, colors=None):
    return keep_faces(V, F, ~np.asarray(face_mask, bool), flags, colors)


def face_components(F, n_vertices: int) -> np.ndarray:
    """面の連結成分ラベル（頂点共有で連結）."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = np.asarray(F)
    m = len(F)
    rows = np.repeat(np.arange(m), 3)
    cols = F.ravel()
    A = coo_matrix((np.ones(rows.size, bool), (rows, cols)), shape=(m, n_vertices)).tocsr()
    # 面-頂点の二部グラフで連結成分
    from scipy.sparse import bmat
    G = bmat([[None, A], [A.T, None]]).tocsr()
    _, labels = connected_components(G, directed=False)
    return labels[:m]


def remove_small_components(V, F, min_faces: int = 0, min_ratio: float = 0.01,
                            flags=None, colors=None):
    """ノイズ除去: 小さな浮遊片を削除。min_faces と最大成分比 min_ratio の両方で判定."""
    if len(F) == 0:
        return np.asarray(V), np.asarray(F), flags, colors, 0, 0
    labels = face_components(F, len(V))
    uniq, counts = np.unique(labels, return_counts=True)
    thr = max(min_faces, int(np.ceil(counts.max() * min_ratio)))
    keep_labels = uniq[counts >= thr]
    mask = np.isin(labels, keep_labels)
    removed = int((~mask).sum())
    V2, F2, fl, c2 = keep_faces(V, F, mask, flags, colors)
    return V2, F2, fl, c2, removed, int(len(uniq) - len(keep_labels))


def split_components(V, F, flags=None, colors=None, min_faces: int = 1):
    labels = face_components(F, len(V))
    out = []
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() < min_faces:
            continue
        out.append(keep_faces(V, F, mask, flags, colors))
    out.sort(key=lambda t: -len(t[1]))
    return out


def merge_meshes(items):
    """[(V, F, flags, colors), ...] を1つに結合."""
    Vs, Fs, fls, cs = [], [], [], []
    off = 0
    any_color = any(it[3] is not None for it in items)
    for V, F, fl, c in items:
        Vs.append(np.asarray(V, float))
        Fs.append(np.asarray(F) + off)
        fls.append(fl if fl is not None else np.zeros(len(F), np.uint8))
        if any_color:
            cs.append(c if c is not None else np.full((len(V), 3), 200, np.uint8))
        off += len(V)
    return (np.vstack(Vs), np.vstack(Fs), np.concatenate(fls),
            np.vstack(cs) if any_color else None)


def smooth_taubin(V, F, iterations: int = 10, lamb: float = 0.5, mu: float = -0.53,
                  vertex_mask: Optional[np.ndarray] = None):
    """Taubin スムージング（体積収縮が少ない）。vertex_mask があればその頂点のみ移動."""
    import trimesh
    if len(F) == 0:
        return np.asarray(V, float).copy()
    m = trimesh.Trimesh(np.asarray(V, float).copy(), np.asarray(F), process=False)
    trimesh.smoothing.filter_taubin(m, lamb=lamb, nu=-mu if mu < 0 else mu, iterations=iterations)
    V2 = np.asarray(m.vertices)
    if vertex_mask is not None:
        out = np.asarray(V, float).copy()
        out[vertex_mask] = V2[vertex_mask]
        return out
    return V2


def decimate(V, F, target_reduction: float = 0.5, flags=None):
    """二次誤差メトリクスによる簡略化 (VTK vtkQuadricDecimation)。フラグは破棄される."""
    import pyvista as pv
    if not 0.0 < target_reduction < 1.0:
        raise ValueError("削減率は 0〜1 の間で指定してください")
    if len(F) < 8:
        raise ValueError("面数が少なすぎるため簡略化できません")
    pd = pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())
    out = pd.decimate(target_reduction)
    out = out.triangulate()
    V2 = np.asarray(out.points, float)
    F2 = np.asarray(out.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    return V2, F2, np.zeros(len(F2), np.uint8)


def fix_normals(V, F):
    """面の向きを統一し、外向きに揃える (trimesh)."""
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=False)
    trimesh.repair.fix_normals(m, multibody=True)
    return np.asarray(m.faces, np.int64)


def flip_faces(F, face_mask=None):
    F = np.asarray(F).copy()
    if face_mask is None:
        return F[:, ::-1].copy()
    F[face_mask] = F[face_mask][:, ::-1]
    return F


def boundary_loops(F):
    """境界エッジ（1面のみに属するエッジ）の数を返す簡易版."""
    F = np.asarray(F)
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int(np.count_nonzero(cnt == 1))


def fill_holes(V, F, hole_size: float, flags=None, colors=None):
    """穴埋め (VTK vtkFillHolesFilter)。生成面には FLAG_ARTIFICIAL を立てる.

    hole_size: 埋める穴の最大サイズ（穴を囲む外接球の半径, mm）
    """
    import pyvista as pv
    V = np.asarray(V, float)
    F = np.asarray(F)
    n0 = len(F)
    pd = pv.PolyData(V, np.hstack([np.full((n0, 1), 3), F]).ravel())
    out = pd.fill_holes(hole_size).triangulate()
    V2 = np.asarray(out.points, float)
    F2 = np.asarray(out.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    if len(V2) != len(V) or not np.array_equal(F2[:n0], F):
        # 想定外の並び替え: 元の面集合との照合で人工面を判定
        key = lambda A: set(map(tuple, np.sort(A, axis=1).tolist()))
        orig = key(F)
        art = np.array([tuple(sorted(f)) not in orig for f in F2.tolist()])
    else:
        art = np.zeros(len(F2), bool)
        art[n0:] = True
    fl = np.zeros(len(F2), np.uint8)
    if flags is not None and len(V2) == len(V):
        fl[:min(n0, len(F2))] = flags[:min(n0, len(F2))]
    fl[art] |= FLAG_ARTIFICIAL
    c2 = colors if (colors is not None and len(V2) == len(V)) else None
    return V2, F2, fl, c2, int(art.sum())


def vertex_normals(V, F) -> np.ndarray:
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=False)
    return np.asarray(m.vertex_normals, float)


def face_normals(V, F) -> np.ndarray:
    V = np.asarray(V, float)
    F = np.asarray(F)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-30)


def face_areas(V, F) -> np.ndarray:
    V = np.asarray(V, float)
    F = np.asarray(F)
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


# ---------------------------------------------------------------------------
# 点群
# ---------------------------------------------------------------------------

def statistical_outlier_removal(P, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """統計的外れ値除去。戻り値は残す点のマスク."""
    P = np.asarray(P, float)
    if len(P) <= k:
        return np.ones(len(P), bool)
    tree = cKDTree(P)
    d, _ = tree.query(P, k=k + 1, workers=-1)
    md = d[:, 1:].mean(axis=1)
    thr = md.mean() + std_ratio * md.std()
    return md <= thr


def estimate_normals(P, k: int = 20, viewpoint=None, progress: Optional[Progress] = None):
    """点群の法線推定（近傍PCA）。向きは viewpoint 方向（未指定なら重心から外向き）."""
    progress = ensure_progress(progress)
    P = np.asarray(P, float)
    try:
        import open3d as o3d  # 高速版（任意）
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(k))
        N = np.asarray(pc.normals)
    except Exception:
        tree = cKDTree(P)
        _, idx = tree.query(P, k=min(k, len(P)), workers=-1)
        N = np.empty_like(P)
        chunk = 50000
        for s in range(0, len(P), chunk):
            progress.report(s / len(P), "法線推定中")
            nb = P[idx[s:s + chunk]]
            c = nb - nb.mean(axis=1, keepdims=True)
            cov = np.einsum("nki,nkj->nij", c, c)
            w, vec = np.linalg.eigh(cov)
            N[s:s + chunk] = vec[:, :, 0]
    ref = (P - P.mean(axis=0)) if viewpoint is None else (np.asarray(viewpoint, float) - P)
    flip = np.sum(N * ref, axis=1) < 0
    N[flip] *= -1
    return N


def poisson_reconstruction(P, N, depth: int = 9):
    """ポアソン表面再構成（Open3D がインストールされている場合のみ）."""
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("ポアソン再構成には Open3D が必要です (pip install open3d)") from e
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    pc.normals = o3d.utility.Vector3dVector(np.asarray(N, float))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    dens = np.asarray(dens)
    keep = dens > np.quantile(dens, 0.02)
    mesh.remove_vertices_by_mask(~keep)
    return np.asarray(mesh.vertices, float), np.asarray(mesh.triangles, np.int64)


def open3d_available() -> bool:
    try:
        import open3d  # noqa: F401
        return True
    except Exception:
        return False


def lod_decimate_for_display(V, F, max_faces: int):
    """表示用LOD: 面数が多すぎる場合、VTK の高速クラスタリングで間引く."""
    if len(F) <= max_faces:
        return None
    import pyvista as pv
    import vtk
    pd = pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())
    qc = vtk.vtkQuadricClustering()
    ratio = max_faces / len(F)
    div = int(max(32, min(2048, np.sqrt(max_faces / 2.0) * 1.2)))
    qc.SetNumberOfXDivisions(div)
    qc.SetNumberOfYDivisions(div)
    qc.SetNumberOfZDivisions(div)
    qc.SetInputData(pd)
    qc.Update()
    out = pv.wrap(qc.GetOutput())
    log.info("LOD生成: %d → %d 面 (比率 %.2f)", len(F), out.n_cells, ratio)
    return out
