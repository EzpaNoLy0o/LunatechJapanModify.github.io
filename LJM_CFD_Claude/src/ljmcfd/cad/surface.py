"""表示用サーフェスメッシュ (PyVista PolyData) の生成と STL 読み込み。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

from ..core.jobs import JobError

FACE_ID = "face_id"
#: STL を「面」に分割するときの折れ角 [deg]
STL_FEATURE_ANGLE = 30.0


def build_surface(points: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray) -> pv.PolyData:
    """三角形メッシュから法線付きの PolyData を作る (面番号をセルデータに保持)。"""
    n_tris = len(triangles)
    cells = np.empty((n_tris, 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = triangles
    mesh = pv.PolyData(np.asarray(points, dtype=np.float32), cells.ravel())
    mesh.cell_data[FACE_ID] = np.asarray(face_ids, dtype=np.int32)
    # 面ごとに節点が独立しているので、面内だけが滑らかになる
    mesh = mesh.compute_normals(
        cell_normals=False, point_normals=True, split_vertices=False,
        consistent_normals=False, auto_orient_normals=False,
    )
    return mesh


def face_boundary_edges(surface: pv.PolyData) -> pv.PolyData:
    """面の輪郭線 (CAD 風のエッジ表示用)。面間で節点を共有しないメッシュが前提。"""
    edges = surface.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False,
    )
    edges.clear_data()
    return edges


def _stl_solid_names(path: Path) -> list[str]:
    """ASCII STL の ``solid <name>`` を列挙する。バイナリ STL では空リスト。"""
    names: list[str] = []
    try:
        with open(path, "rb") as fh:
            head = fh.read(512)
            if not head.lstrip().lower().startswith(b"solid") or b"\0" in head:
                return []
            fh.seek(0)
            for raw in fh:
                m = re.match(rb"\s*solid\b[ \t]*(.*)$", raw.rstrip(b"\r\n"), re.I)
                if m:
                    names.append(m.group(1).decode("utf-8", "replace").strip())
    except OSError:
        return []
    return names


def segment_by_feature_angle(mesh: pv.PolyData, feature_angle: float = STL_FEATURE_ANGLE) -> pv.PolyData:
    """折れ角で節点を分割し、滑らかにつながる領域ごとに面番号を振る。"""
    split = mesh.compute_normals(
        cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feature_angle,
        consistent_normals=False, auto_orient_normals=False,
    )
    labelled = split.connectivity()
    region = np.asarray(labelled.cell_data["RegionId"], dtype=np.int32)
    normals = labelled.point_data.get("Normals")
    labelled.clear_data()
    labelled.cell_data[FACE_ID] = region
    if normals is not None:
        labelled.point_data["Normals"] = normals
        labelled.point_data.active_normals_name = "Normals"
    return labelled


def read_stl(path: str | Path, default_name: str | None = None) -> list[tuple[str, pv.PolyData]]:
    """STL を読み、(名前, サーフェス) のリストを返す。

    ASCII STL に複数の ``solid`` があればソリッドごとに分割する。座標はファイルの
    数値そのまま (STL は単位を持たない)。
    """
    path = Path(path)
    base = default_name or path.stem
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.ScalarTagsOn()
    reader.MergingOn()
    reader.Update()
    raw = pv.wrap(reader.GetOutput())
    if raw is None or raw.n_cells == 0:
        raise JobError(f"STL ファイルに三角形がありません: {path.name}")
    raw = raw.triangulate()

    labels = None
    for key in raw.cell_data.keys():
        if "solid" in key.lower():
            labels = np.asarray(raw.cell_data[key]).astype(np.int64)
            break
    names = _stl_solid_names(path)
    groups: list[tuple[str, pv.PolyData]] = []
    if labels is not None and len(np.unique(labels)) > 1:
        for order, label in enumerate(np.unique(labels)):
            sub = raw.extract_cells(np.flatnonzero(labels == label)).extract_surface(algorithm=None)
            name = names[order] if order < len(names) and names[order] else f"{base}_{order + 1}"
            groups.append((name, sub))
    else:
        groups.append((base, raw))

    result = []
    for name, mesh in groups:
        mesh.clear_data()
        result.append((name, segment_by_feature_angle(mesh)))
    return result
