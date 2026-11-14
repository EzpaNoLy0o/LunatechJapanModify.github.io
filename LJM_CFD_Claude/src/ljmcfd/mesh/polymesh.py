"""セル (四面体・プリズム・ピラミッド・六面体) から OpenFOAM の polyMesh を組み立てて書き出す。

OpenFOAM の面ベース表現の規約:

* 内部面を先に並べ、所有セル (owner) の昇順、同じ owner 内では隣接セル (neighbour) の昇順。
* 内部面の法線は owner → neighbour の向き (owner < neighbour)。
* 境界面はパッチごとにまとめ、法線は領域の外向き。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.jobs import JobError

TET, PRISM, PYRAMID, HEX = "tet", "prism", "pyramid", "hex"
CELL_ORDER = (TET, PRISM, PYRAMID, HEX)
CELL_NODE_COUNT = {TET: 4, PRISM: 6, PYRAMID: 5, HEX: 8}

#: セルの面 (外向き法線になる節点順)。正の向きのセル (下の ``orient_cells`` 参照) が前提。
CELL_FACES: dict[str, dict[str, list[tuple[int, ...]]]] = {
    TET: {"tri": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)], "quad": []},
    PRISM: {"tri": [(0, 2, 1), (3, 4, 5)], "quad": [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]},
    PYRAMID: {"tri": [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], "quad": [(0, 3, 2, 1)]},
    HEX: {"tri": [], "quad": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]},
}
#: 向きが負のセルを正に直す節点の並べ替え
_FLIP = {
    TET: (0, 2, 1, 3),
    PRISM: (0, 2, 1, 3, 5, 4),
    PYRAMID: (0, 3, 2, 1, 4),
    HEX: (0, 3, 2, 1, 4, 7, 6, 5),
}
#: 向き判定に使う (底面の 2 辺の相手, 高さ方向の相手) の節点番号
_ORIENT = {TET: (1, 2, 3), PRISM: (1, 2, 3), PYRAMID: (1, 3, 4), HEX: (1, 3, 4)}

PATCH_DEFAULT = "defaultFaces"


@dataclass
class BoundaryGroup:
    """境界パッチと、そこに属する面 (節点番号)。"""

    name: str
    type: str = "patch"            # OpenFOAM の幾何パッチ種別 ("patch" / "wall")
    tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    quads: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.int64))
    info: dict = field(default_factory=dict)   # 付加情報 (パーツ id など)。レポートへそのまま残す


@dataclass
class PolyMesh:
    points: np.ndarray             # (N, 3)
    faces: np.ndarray              # (F, 4)  三角形は 4 列目が -1
    owner: np.ndarray              # (F,)
    neighbour: np.ndarray          # (内部面数,)
    patches: list[dict]            # {"name", "type", "start", "size", "info"}
    n_cells: int
    cell_counts: dict[str, int]
    n_unmatched_faces: int = 0     # どのパッチにも対応付けられなかった境界面
    n_orphan_boundary: int = 0     # どのセル面にも対応しなかった境界要素

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    @property
    def n_internal(self) -> int:
        return int(len(self.neighbour))

    def patch(self, name: str) -> dict:
        for patch in self.patches:
            if patch["name"] == name:
                return patch
        raise KeyError(name)


def orient_cells(points: np.ndarray, kind: str, cells: np.ndarray) -> tuple[np.ndarray, int]:
    """セルの節点順を正の向きにそろえる。(並べ替えたセル, 反転した数) を返す。"""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, CELL_NODE_COUNT[kind])
    if len(cells) == 0:
        return cells, 0
    a, b, c = _ORIENT[kind]
    p0 = points[cells[:, 0]]
    triple = np.einsum("ij,ij->i", np.cross(points[cells[:, a]] - p0, points[cells[:, b]] - p0), points[cells[:, c]] - p0)
    negative = triple < 0
    if negative.any():
        cells = cells.copy()
        cells[negative] = cells[negative][:, _FLIP[kind]]
    return cells, int(negative.sum())


def _pair_faces(faces: np.ndarray, cell_ids: np.ndarray):
    """同じ節点集合を持つ面を対にする。

    戻り値: (内部面の節点, owner, neighbour), (境界面の節点, owner)
    """
    width = faces.shape[1]
    if len(faces) == 0:
        empty = np.empty((0, width), dtype=np.int64)
        none = np.empty(0, dtype=np.int64)
        return (empty, none, none), (empty, none)
    keys = np.sort(faces, axis=1)
    order = np.lexsort(keys.T[::-1])
    sorted_keys = keys[order]
    same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    if np.any(same[1:] & same[:-1]):
        raise JobError("3 個以上のセルが 1 つの面を共有しています (メッシュが非多様体です)")
    is_first = np.concatenate([same, [False]])
    is_second = np.concatenate([[False], same])
    a, b = order[is_first], order[is_second]
    ca, cb = cell_ids[a], cell_ids[b]
    owner_side = np.where(ca < cb, a, b)
    internal = (faces[owner_side], np.minimum(ca, cb), np.maximum(ca, cb))
    single = order[~(is_first | is_second)]
    return internal, (faces[single], cell_ids[single])


def _pad(faces: np.ndarray) -> np.ndarray:
    if faces.shape[1] == 4:
        return faces
    return np.column_stack([faces, np.full(len(faces), -1, dtype=np.int64)])


def build_polymesh(points: np.ndarray, cells: dict[str, np.ndarray], boundary: list[BoundaryGroup]) -> PolyMesh:
    """セルと境界要素から polyMesh を組み立てる。

    cells:    セル種別 → (n, k) の節点番号配列 (0 始まり)。向きは自動で正にそろえる。
    boundary: 境界パッチの定義。ここに無い境界面は ``defaultFaces`` パッチへ入り、
              ``n_unmatched_faces`` に数が記録される (呼び出し側でエラーにすること)。
    """
    points = np.asarray(points, dtype=np.float64)
    tri_faces, tri_cells, quad_faces, quad_cells = [], [], [], []
    offset = 0
    counts: dict[str, int] = {}
    for kind in CELL_ORDER:
        if kind not in cells or len(cells[kind]) == 0:
            continue
        oriented, _ = orient_cells(points, kind, cells[kind])
        n = len(oriented)
        counts[kind] = n
        ids = np.arange(offset, offset + n, dtype=np.int64)
        for template in CELL_FACES[kind]["tri"]:
            tri_faces.append(oriented[:, template])
            tri_cells.append(ids)
        for template in CELL_FACES[kind]["quad"]:
            quad_faces.append(oriented[:, template])
            quad_cells.append(ids)
        offset += n
    n_cells = offset
    if n_cells == 0:
        raise JobError("体積セルがありません")

    def stack(blocks, width):
        return np.vstack(blocks) if blocks else np.empty((0, width), dtype=np.int64)

    def cat(blocks):
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)

    (ti, ti_own, ti_nei), (tb, tb_own) = _pair_faces(stack(tri_faces, 3), cat(tri_cells))
    (qi, qi_own, qi_nei), (qb, qb_own) = _pair_faces(stack(quad_faces, 4), cat(quad_cells))

    internal = np.vstack([_pad(ti), qi])
    own = np.concatenate([ti_own, qi_own])
    nei = np.concatenate([ti_nei, qi_nei])
    order = np.lexsort((nei, own))
    internal, own, nei = internal[order], own[order], nei[order]

    # --- 境界面をパッチへ対応付ける (節点集合が一致する境界要素を探す)
    lookup: dict[tuple, int] = {}
    n_boundary_elements = 0
    for index, group in enumerate(boundary):
        for array in (np.asarray(group.tris, dtype=np.int64).reshape(-1, 3), np.asarray(group.quads, dtype=np.int64).reshape(-1, 4)):
            n_boundary_elements += len(array)
            for key in map(tuple, np.sort(array, axis=1).tolist()):
                lookup[key] = index
    bfaces = np.vstack([_pad(tb), qb])
    bown = np.concatenate([tb_own, qb_own])
    keys = [tuple(k) for k in np.sort(tb, axis=1).tolist()] + [tuple(k) for k in np.sort(qb, axis=1).tolist()]
    n_groups = len(boundary)
    patch_index = np.fromiter((lookup.get(k, n_groups) for k in keys), dtype=np.int64, count=len(keys))
    n_unmatched = int((patch_index == n_groups).sum())
    n_orphan = n_boundary_elements - (len(keys) - n_unmatched)
    order = np.argsort(patch_index, kind="stable")
    bfaces, bown, patch_index = bfaces[order], bown[order], patch_index[order]

    patches = []
    start = len(internal)
    sizes = np.bincount(patch_index, minlength=n_groups + 1)
    for index, group in enumerate(boundary):
        patches.append({"name": group.name, "type": group.type, "start": int(start), "size": int(sizes[index]), "info": dict(group.info)})
        start += int(sizes[index])
    if n_unmatched:
        patches.append({"name": PATCH_DEFAULT, "type": "wall", "start": int(start), "size": n_unmatched, "info": {}})

    return PolyMesh(
        points=points, faces=np.vstack([internal, bfaces]), owner=np.concatenate([own, bown]), neighbour=nei,
        patches=patches, n_cells=n_cells, cell_counts=counts, n_unmatched_faces=n_unmatched, n_orphan_boundary=max(n_orphan, 0),
    )


# --------------------------------------------------------------------------- 書き出し・読み込み

def _header(cls: str, obj: str, note: str = "") -> str:
    note_line = f'    note        "{note}";\n' if note else ""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  Generated by LJM CFD (Claude Edition)\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n{\n    format      ascii;\n"
        f"    class       {cls};\n{note_line}"
        '    location    "constant/polyMesh";\n'
        f"    object      {obj};\n}}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def write_polymesh(mesh: PolyMesh, directory: str | Path) -> Path:
    """``directory`` (= constant/polyMesh に相当) へ points / faces / owner / neighbour / boundary を書く。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    note = f"nPoints: {len(mesh.points)} nCells: {mesh.n_cells} nFaces: {mesh.n_faces} nInternalFaces: {mesh.n_internal}"

    with open(directory / "points", "w", encoding="ascii", newline="\n") as fh:
        fh.write(_header("vectorField", "points"))
        fh.write(f"{len(mesh.points)}\n(\n")
        np.savetxt(fh, mesh.points, fmt="(%.12g %.12g %.12g)")
        fh.write(")\n")

    with open(directory / "faces", "w", encoding="ascii", newline="\n") as fh:
        fh.write(_header("faceList", "faces"))
        fh.write(f"{mesh.n_faces}\n(\n")
        is_tri = mesh.faces[:, 3] < 0
        if is_tri.all():
            np.savetxt(fh, mesh.faces[:, :3], fmt="3(%d %d %d)")
        else:
            lines = [
                f"3({a} {b} {c})" if d < 0 else f"4({a} {b} {c} {d})"
                for a, b, c, d in mesh.faces.tolist()
            ]
            fh.write("\n".join(lines))
            fh.write("\n")
        fh.write(")\n")

    for name, data in (("owner", mesh.owner), ("neighbour", mesh.neighbour)):
        with open(directory / name, "w", encoding="ascii", newline="\n") as fh:
            fh.write(_header("labelList", name, note))
            fh.write(f"{len(data)}\n(\n")
            np.savetxt(fh, data, fmt="%d")
            fh.write(")\n")

    with open(directory / "boundary", "w", encoding="ascii", newline="\n") as fh:
        fh.write(_header("polyBoundaryMesh", "boundary"))
        fh.write(f"{len(mesh.patches)}\n(\n")
        for patch in mesh.patches:
            fh.write(f"    {patch['name']}\n    {{\n        type            {patch['type']};\n")
            if patch["type"] == "wall":
                fh.write("        inGroups        1(wall);\n")
            fh.write(f"        nFaces          {patch['size']};\n        startFace       {patch['start']};\n    }}\n")
        fh.write(")\n")
    return directory


def _read_list_body(path: Path) -> list[str]:
    """FoamFile ヘッダを飛ばし、`N ( … )` の中身の行を返す (自前で書いた形式専用)。"""
    lines = path.read_text(encoding="ascii").splitlines()
    for index, line in enumerate(lines):
        if line.strip().isdigit() and index + 1 < len(lines) and lines[index + 1].strip() == "(":
            count = int(line)
            return lines[index + 2:index + 2 + count]
    raise JobError(f"polyMesh ファイルの形式が不正です: {path}")


def read_polymesh(directory: str | Path) -> PolyMesh:
    """``write_polymesh`` で書いた polyMesh を読み戻す (検証・再利用のため)。"""
    directory = Path(directory)
    points = np.array([line.strip("()").split() for line in _read_list_body(directory / "points")], dtype=np.float64)
    rows = []
    for line in _read_list_body(directory / "faces"):
        nodes = [int(v) for v in line[line.index("(") + 1:-1].split()]
        rows.append(nodes + [-1] * (4 - len(nodes)))
    owner = np.array(_read_list_body(directory / "owner"), dtype=np.int64)
    neighbour = np.array(_read_list_body(directory / "neighbour"), dtype=np.int64)
    patches = []
    text = (directory / "boundary").read_text(encoding="ascii").split("// * *", 1)[-1]
    tokens = text.replace("{", " { ").replace("}", " } ").replace(";", " ").split()
    i = tokens.index("(") + 1
    while tokens[i] != ")":
        name, body = tokens[i], {}
        j = i + 2
        while tokens[j] != "}":
            body[tokens[j]] = tokens[j + 1]
            j += 2
        patches.append({"name": name, "type": body["type"], "start": int(body["startFace"]), "size": int(body["nFaces"]), "info": {}})
        i = j + 1
    return PolyMesh(
        points=points, faces=np.array(rows, dtype=np.int64), owner=owner, neighbour=neighbour, patches=patches,
        # 最後のセルは面の owner にならないことがある (隣接セルの番号がすべて小さい場合) ので neighbour も見る
        n_cells=int(max(owner.max(), neighbour.max(initial=-1))) + 1, cell_counts={},
    )
