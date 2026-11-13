"""形状検査: 開放エッジ・非多様体・面の向き・B-Rep 妥当性。

外部流れの体積メッシュを作るには、車体表面が「閉じた (水密な) 多様体」である
必要がある。ここでは次の 2 段階で検査する。

1. **メッシュ検査** (全形式): 表示用三角形メッシュの節点をマージし、各エッジを
   共有する三角形の数を数える。1 枚 = 開放エッジ、3 枚以上 = 非多様体エッジ。
2. **B-Rep 検査** (STEP / IGES のみ): OpenCASCADE の妥当性検査とシェルの自由エッジ。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

from ..core.jobs import JobContext, JobError

#: 節点マージ許容差 (境界ボックス対角長に対する比)
MERGE_TOLERANCE_REL = 1.0e-6

KIND_OPEN = 1
KIND_NON_MANIFOLD = 2


def _triangles(mesh: pv.PolyData) -> np.ndarray:
    if mesh.n_cells == 0:
        return np.empty((0, 3), dtype=np.int64)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def inspect_surface_mesh(surface: pv.PolyData) -> tuple[dict, pv.PolyData | None]:
    """三角形メッシュのトポロジ検査。

    戻り値: (レポート, 欠陥エッジの PolyData または None)。長さ・面積・体積は
    メッシュの座標単位のまま返す (単位換算は呼び出し側)。
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_input = int(surface.n_cells)
    if n_input == 0:
        raise JobError("検査対象のメッシュが空です")
    bounds = surface.bounds
    diag = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
    tol = max(diag * MERGE_TOLERANCE_REL, 1.0e-12)

    geom = pv.PolyData(surface.points, surface.faces)
    merged = geom.clean(
        tolerance=tol, absolute=True, lines_to_points=False, polys_to_lines=False, strips_to_polys=False,
    )
    tris = _triangles(merged)
    pts = np.asarray(merged.points, dtype=float)
    # マージで縮退した三角形 (clean が除去したもの + 同一節点を含むもの)
    distinct = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[distinct]
    n_degenerate = n_input - len(tris)
    if len(tris) == 0:
        raise JobError("有効な三角形がありません (すべて縮退しています)")

    # 無向エッジごとの共有三角形数
    directed = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    n_pts = len(pts)
    edge_key = undirected[:, 0] * n_pts + undirected[:, 1]
    unique_keys, first_index, counts = np.unique(edge_key, return_index=True, return_counts=True)
    open_mask = counts == 1
    nonmanifold_mask = counts > 2
    n_open = int(open_mask.sum())
    n_nonmanifold = int(nonmanifold_mask.sum())

    # 向きの不整合: 2 枚で共有されるエッジは互いに逆向きに現れるはず
    directed_key = directed[:, 0] * n_pts + directed[:, 1]
    n_inconsistent = int(len(directed_key) - len(np.unique(directed_key)))

    # 連結成分数
    graph = coo_matrix(
        (np.ones(len(undirected), dtype=np.int8), (undirected[:, 0], undirected[:, 1])), shape=(n_pts, n_pts)
    )
    used = np.zeros(n_pts, dtype=bool)
    used[tris.ravel()] = True
    _, labels = connected_components(graph, directed=False)
    n_components = int(len(np.unique(labels[used])))

    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    watertight = n_open == 0 and n_nonmanifold == 0
    volume = None
    if watertight and n_inconsistent == 0:
        # 発散定理による符号付き体積。負なら法線が内向き。
        volume = float(np.einsum("ij,ij->i", p0, cross).sum() / 6.0)

    defects = None
    if n_open or n_nonmanifold:
        bad = np.flatnonzero(open_mask | nonmanifold_mask)
        edges = undirected[first_index[bad]]
        kinds = np.where(counts[bad] == 1, KIND_OPEN, KIND_NON_MANIFOLD).astype(np.int32)
        used_pts, inverse = np.unique(edges.ravel(), return_inverse=True)
        lines = np.empty((len(edges), 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1:] = inverse.reshape(-1, 2)
        defects = pv.PolyData(pts[used_pts].astype(np.float32), lines=lines.ravel())
        defects.cell_data["kind"] = kinds

    open_length = 0.0
    if n_open:
        e = undirected[first_index[open_mask]]
        open_length = float(np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1).sum())

    report = {
        "n_triangles": int(len(tris)),
        "n_degenerate_triangles": int(n_degenerate),
        "n_open_edges": n_open,
        "open_edge_length": open_length,
        "n_non_manifold_edges": n_nonmanifold,
        "n_inconsistent_edges": n_inconsistent,
        "n_components": n_components,
        "watertight": bool(watertight),
        "area": area,
        "volume": volume,
        "merge_tolerance": tol,
    }
    return report, defects


def inspect_brep(shape) -> dict:
    """B-Rep の妥当性とシェルの閉じ具合を検査する (OCP が必要)。"""
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeAnalysis import ShapeAnalysis_Shell
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID

    from .occ_io import count_subshapes, iter_subshapes

    n_solids = count_subshapes(shape, TopAbs_SOLID)
    shells = list(iter_subshapes(shape, TopAbs_SHELL))
    n_faces = count_subshapes(shape, TopAbs_FACE)
    n_free_faces = sum(1 for _ in iter_subshapes(shape, TopAbs_FACE, TopAbs_SHELL))
    n_open_shells = sum(1 for shell in shells if not BRep_Tool.IsClosed_s(shell))

    analyzer = ShapeAnalysis_Shell()
    analyzer.LoadShells(shape)
    analyzer.CheckOrientedShells(shape, True, False)
    n_free_edges = count_subshapes(analyzer.FreeEdges(), TopAbs_EDGE) if analyzer.HasFreeEdges() else 0
    n_bad_edges = count_subshapes(analyzer.BadEdges(), TopAbs_EDGE) if analyzer.HasBadEdges() else 0

    try:
        valid = bool(BRepCheck_Analyzer(shape, True).IsValid())
    except Exception as exc:  # noqa: BLE001 - 検査自体が落ちる形状は「不正」として報告する
        valid = False
        check_error = f"{type(exc).__name__}: {exc}"
    else:
        check_error = None

    return {
        "valid": valid,
        "check_error": check_error,
        "n_solids": n_solids,
        "n_shells": len(shells),
        "n_open_shells": n_open_shells,
        "n_faces": n_faces,
        "n_free_faces": n_free_faces,
        "n_free_edges": n_free_edges,
        "n_bad_edges": n_bad_edges,
    }


def summarize(mesh_report: dict, brep_report: dict | None) -> dict:
    """検査結果から判定 (ok / warning / error) と日本語メッセージを作る。"""
    messages: list[str] = []
    level = "ok"

    def raise_level(new: str) -> None:
        nonlocal level
        order = {"ok": 0, "warning": 1, "error": 2}
        if order[new] > order[level]:
            level = new

    if mesh_report["n_open_edges"]:
        raise_level("error")
        messages.append(
            f"開放エッジが {mesh_report['n_open_edges']:,} 本あります (形状が閉じていません)。"
            "このままでは外部流れ用の体積メッシュを生成できません。"
        )
    if mesh_report["n_non_manifold_edges"]:
        raise_level("error")
        messages.append(
            f"非多様体エッジが {mesh_report['n_non_manifold_edges']:,} 本あります (3 枚以上の面が 1 本のエッジを共有)。"
        )
    if mesh_report["n_inconsistent_edges"]:
        raise_level("warning")
        messages.append(
            f"面の向きが不整合なエッジが {mesh_report['n_inconsistent_edges']:,} 本あります (法線が反転した面を含みます)。"
        )
    if mesh_report["n_degenerate_triangles"]:
        raise_level("warning")
        messages.append(f"縮退した三角形が {mesh_report['n_degenerate_triangles']:,} 個あります。")
    if mesh_report["volume"] is not None and mesh_report["volume"] < 0:
        raise_level("warning")
        messages.append("閉じていますが法線が内向きです (体積が負)。")
    if mesh_report["watertight"] and mesh_report["n_components"] > 1:
        messages.append(f"{mesh_report['n_components']} 個の独立した閉じた形状で構成されています。")

    if brep_report is not None:
        if not brep_report["valid"]:
            raise_level("warning")
            detail = f" ({brep_report['check_error']})" if brep_report.get("check_error") else ""
            messages.append(f"OpenCASCADE の B-Rep 妥当性検査で問題が検出されました{detail}。")
        if brep_report["n_solids"] == 0:
            raise_level("warning")
            messages.append("ソリッドが含まれていません (サーフェスのみのモデルです)。")
        if brep_report["n_bad_edges"]:
            raise_level("warning")
            messages.append(f"シェル内で向きが不正なエッジが {brep_report['n_bad_edges']:,} 本あります。")

    if level == "ok":
        messages.insert(0, "閉じた (水密な) 多様体形状です。体積メッシュ生成に使用できます。")
    return {"level": level, "messages": messages}


def inspect_part_job(args: dict, ctx: JobContext | None = None) -> dict:
    """キャッシュ済みパーツを検査する。

    args:
        cache_dir  インポートキャッシュのディレクトリ
        surface    サーフェス VTP のファイル名
        brep       BREP のファイル名 (無ければ None)
        part_id    結果へそのまま返す識別子 (任意)
    """
    ctx = ctx or JobContext()
    cache_dir = Path(args["cache_dir"])
    surface_path = cache_dir / args["surface"]
    if not surface_path.is_file():
        raise JobError(f"サーフェスキャッシュが見つかりません: {surface_path}")

    ctx.progress(0.1, "メッシュのトポロジを検査中…")
    surface = pv.read(surface_path)
    mesh_report, defects = inspect_surface_mesh(surface)

    defects_name = None
    stem = Path(args["surface"]).stem
    defects_path = cache_dir / f"{stem}_defects.vtp"
    if defects is not None:
        defects.save(defects_path, binary=True)
        defects_name = defects_path.name
    else:
        defects_path.unlink(missing_ok=True)

    brep_report = None
    if args.get("brep"):
        ctx.progress(0.5, "B-Rep の妥当性を検査中…")
        from . import occ_io

        occ_io.silence_occ_messages()
        brep_report = inspect_brep(occ_io.read_brep(cache_dir / args["brep"]))

    result = {
        "part_id": args.get("part_id"),
        "mesh": mesh_report,
        "brep": brep_report,
        "summary": summarize(mesh_report, brep_report),
        "defects": defects_name,
    }
    (cache_dir / f"{stem}_inspect.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.progress(1.0, "検査完了")
    return result
