"""メッシュ/点群ファイルの読み書き（STL / OBJ / PLY）.

- Windows の日本語パスに対応するため、ファイルの読み書きは Python の open() を経由する
  （Open3D の I/O は非ASCIIパスで失敗する既知の問題があるため使用しない）。
- STL には単位情報がないため、寸法から単位を推定してユーザーに確認を求める。
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MESH_EXTENSIONS = (".stl", ".obj", ".ply")
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4}
UNIT_LABELS = {"mm": "ミリメートル (mm)", "cm": "センチメートル (cm)", "m": "メートル (m)",
               "inch": "インチ (inch)"}


@dataclass
class LoadedData:
    kind: str                       # 'mesh' または 'points'
    vertices: np.ndarray
    faces: Optional[np.ndarray]
    vertex_normals: Optional[np.ndarray] = None
    vertex_colors: Optional[np.ndarray] = None   # uint8 (N,3)
    source_path: str = ""
    file_format: str = ""
    sha256: str = ""
    unit_guess: str = "mm"
    unit_message: str = ""
    warnings: list = field(default_factory=list)

    @property
    def n_vertices(self):
        return int(len(self.vertices))

    @property
    def n_faces(self):
        return 0 if self.faces is None else int(len(self.faces))

    def bbox_size(self):
        if len(self.vertices) == 0:
            return np.zeros(3)
        return self.vertices.max(axis=0) - self.vertices.min(axis=0)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def guess_unit(bbox_size) -> tuple[str, str]:
    """外形寸法から単位を推定する。戻り値 (単位, 説明メッセージ)."""
    m = float(np.max(bbox_size)) if np.size(bbox_size) else 0.0
    if m <= 0:
        return "mm", "寸法がゼロのため単位を推定できません。"
    if m < 3.0:
        return "m", (f"最大寸法が {m:.4g} と非常に小さいため、メートル単位で出力された可能性があります。"
                     "自動車部品として妥当な単位を選択してください。")
    if m > 5000.0:
        return "mm", (f"最大寸法が {m:.4g} と大きいため、単位が mm 以外（例: 0.1mm, μm）の可能性があります。"
                      "寸法を確認してください。")
    return "mm", f"最大寸法 {m:.4g}。Revo Scan の標準出力に合わせて mm と推定しました。"


def load_mesh_file(path: str, compute_hash: bool = True) -> LoadedData:
    import trimesh

    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in MESH_EXTENSIONS:
        raise ValueError(f"未対応のファイル形式です: {ext}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    warnings = []
    if ext == ".obj":
        # OBJ は mtl/テクスチャ参照があるためパス指定で読む（trimesh は Python I/O を使う）
        geom = trimesh.load(path, force=None, process=True, maintain_order=False)
    else:
        with open(path, "rb") as f:
            data = f.read()
        geom = trimesh.load(io.BytesIO(data), file_type=ext[1:], force=None, process=True)

    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.dump() if isinstance(g, trimesh.Trimesh)]
        clouds = [g for g in geom.dump() if isinstance(g, trimesh.PointCloud)]
        if meshes:
            geom = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            if len(meshes) > 1:
                warnings.append(f"{len(meshes)} 個のメッシュを結合して読み込みました。")
        elif clouds:
            geom = clouds[0]
        else:
            raise ValueError("ファイルにメッシュまたは点群が含まれていません")

    colors = None
    if isinstance(geom, trimesh.PointCloud):
        V = np.asarray(geom.vertices, dtype=float)
        try:
            c = np.asarray(geom.colors)
            if c.ndim == 2 and len(c) == len(V):
                colors = c[:, :3].astype(np.uint8)
        except Exception:
            colors = None
        normals = None
        md = getattr(geom, "metadata", {}) or {}
        try:
            ply_raw = md.get("_ply_raw", {})
            vx = ply_raw.get("vertex", {}).get("data", None)
            if vx is not None and all(k in vx.dtype.names for k in ("nx", "ny", "nz")):
                normals = np.column_stack([vx["nx"], vx["ny"], vx["nz"]]).astype(float)
        except Exception:
            normals = None
        kind, F = "points", None
    elif isinstance(geom, trimesh.Trimesh):
        if len(geom.faces) == 0:
            V = np.asarray(geom.vertices, dtype=float)
            kind, F, normals = "points", None, None
        else:
            V = np.asarray(geom.vertices, dtype=float)
            F = np.asarray(geom.faces, dtype=np.int64)
            kind = "mesh"
            normals = np.asarray(geom.vertex_normals, dtype=float)
            try:
                if geom.visual.kind == "vertex":
                    vc = np.asarray(geom.visual.vertex_colors)
                    if len(vc) == len(V):
                        colors = vc[:, :3].astype(np.uint8)
                elif geom.visual.kind == "texture":
                    vc = np.asarray(geom.visual.to_color().vertex_colors)
                    if len(vc) == len(V):
                        colors = vc[:, :3].astype(np.uint8)
            except Exception as e:  # テクスチャ変換失敗は致命的ではない
                log.debug("頂点カラー取得失敗: %s", e)
                colors = None
            # 退化三角形の除去
            nondeg = geom.nondegenerate_faces() if hasattr(geom, "nondegenerate_faces") else None
            if nondeg is not None and not np.all(nondeg):
                k = int((~nondeg).sum())
                F = F[nondeg]
                warnings.append(f"面積ゼロの三角形を {k} 個除外しました。")
    else:
        raise ValueError(f"未対応のデータ型です: {type(geom)}")

    if not np.all(np.isfinite(V)):
        bad = ~np.all(np.isfinite(V), axis=1)
        warnings.append(f"不正な座標値(NaN/Inf)を含む頂点が {int(bad.sum())} 個あります。")
        if F is not None:
            F = F[~np.any(bad[F], axis=1)]
        else:
            V = V[~bad]

    data = LoadedData(kind=kind, vertices=V, faces=F, vertex_normals=normals,
                      vertex_colors=colors, source_path=os.path.abspath(path),
                      file_format=ext[1:].upper(), warnings=warnings)
    data.unit_guess, data.unit_message = guess_unit(data.bbox_size())
    if compute_hash:
        try:
            data.sha256 = file_sha256(path)
        except OSError:
            data.sha256 = ""
    return data


def export_mesh(path: str, vertices, faces=None, vertex_colors=None):
    """STL / OBJ / PLY で書き出す。faces=None なら点群（PLYのみ）."""
    import trimesh

    ext = os.path.splitext(path)[1].lower()
    V = np.asarray(vertices, dtype=float)
    if faces is None:
        if ext != ".ply":
            raise ValueError("点群は PLY 形式でのみ出力できます")
        geom = trimesh.PointCloud(V, colors=vertex_colors)
    else:
        geom = trimesh.Trimesh(V, np.asarray(faces), vertex_colors=vertex_colors, process=False)
    if ext == ".stl":
        blob = trimesh.exchange.stl.export_stl(geom)
    elif ext == ".ply":
        blob = trimesh.exchange.ply.export_ply(geom, encoding="binary")
    elif ext == ".obj":
        blob = trimesh.exchange.obj.export_obj(geom, include_texture=False)
        if isinstance(blob, str):
            blob = blob.encode("utf-8")
    else:
        raise ValueError(f"未対応の出力形式です: {ext}")
    with open(path, "wb") as f:
        f.write(blob)
