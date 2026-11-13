from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from ljmcfd.cad.import_job import import_cad_job
from ljmcfd.cad.inspect import KIND_NON_MANIFOLD, KIND_OPEN, inspect_part_job, inspect_surface_mesh


def _inspect(path, cache_root, index=0):
    meta = import_cad_job({"path": str(path), "cache_root": str(cache_root)})
    entry = meta["parts"][index]
    result = inspect_part_job({
        "cache_dir": meta["cache_dir"], "surface": entry["surface"], "brep": entry["brep"], "part_id": "p",
    })
    return meta, result


def test_closed_step_box_is_watertight(box_step, cache_root):
    meta, result = _inspect(box_step, cache_root)
    mesh = result["mesh"]
    assert mesh["watertight"] and mesh["n_open_edges"] == 0 and mesh["n_non_manifold_edges"] == 0
    assert mesh["n_inconsistent_edges"] == 0 and mesh["n_components"] == 1
    assert mesh["volume"] == pytest.approx(4000 * 1800 * 1200, rel=1e-9)   # mm^3、外向き法線で正
    assert mesh["area"] == pytest.approx(2 * (4000 * 1800 + 4000 * 1200 + 1800 * 1200), rel=1e-9)
    brep = result["brep"]
    assert brep["valid"] and brep["n_solids"] == 1 and brep["n_open_shells"] == 0 and brep["n_free_edges"] == 0
    assert result["summary"]["level"] == "ok" and result["defects"] is None
    assert result["part_id"] == "p"
    assert (Path(meta["cache_dir"]) / "part_000_inspect.json").is_file()


def test_open_step_shell_is_reported(open_box_step, cache_root):
    meta, result = _inspect(open_box_step, cache_root)
    mesh = result["mesh"]
    assert not mesh["watertight"] and mesh["n_open_edges"] == 4 and mesh["volume"] is None
    assert result["brep"]["n_solids"] == 0 and result["brep"]["n_open_shells"] >= 1
    assert result["brep"]["n_free_edges"] == 4
    assert result["summary"]["level"] == "error"
    assert any("開放エッジ" in m for m in result["summary"]["messages"])
    defects = pv.read(Path(meta["cache_dir"]) / result["defects"])
    assert defects.n_cells == 4 and set(np.unique(defects.cell_data["kind"])) == {KIND_OPEN}
    # 開放エッジは取り除いた面の周 (1000/500/300 のうち 2 種の辺 x2)
    assert mesh["open_edge_length"] in (pytest.approx(3000.0), pytest.approx(2600.0), pytest.approx(1600.0))


def test_curved_iges_solid_faces(cylinder_iges, cache_root):
    # IGES はトリムドサーフェスの集合になり、面同士が位相的につながらないことが多い。
    # メッシュ検査は節点マージ後に判定するので、幾何学的に閉じていれば水密と判定できる。
    _, result = _inspect(cylinder_iges, cache_root)
    assert result["mesh"]["n_non_manifold_edges"] == 0
    assert result["mesh"]["n_components"] == 1


def test_stl_inspection(cube_stl, open_stl, cache_root):
    _, closed = _inspect(cube_stl, cache_root)
    assert closed["mesh"]["watertight"] and closed["brep"] is None
    assert closed["mesh"]["volume"] == pytest.approx(1.0)
    assert closed["summary"]["level"] == "ok"
    _, opened = _inspect(open_stl, cache_root)
    assert opened["mesh"]["n_open_edges"] == 4 and opened["summary"]["level"] == "error"


def test_mesh_defect_kinds():
    # 3 枚の三角形が 1 本のエッジを共有 → 非多様体
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    faces = np.hstack([[3, 0, 1, 2], [3, 1, 0, 3], [3, 0, 1, 4]])
    report, defects = inspect_surface_mesh(pv.PolyData(pts, faces))
    assert report["n_non_manifold_edges"] == 1 and report["n_open_edges"] == 6
    assert sorted(np.unique(defects.cell_data["kind"])) == [KIND_OPEN, KIND_NON_MANIFOLD]

    # 1 枚だけ裏返した閉じた四面体 → 向き不整合を検出し、体積は報告しない
    tet = pv.PolyData(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.hstack([[3, 0, 2, 1], [3, 0, 1, 3], [3, 1, 2, 3], [3, 0, 2, 3]]),  # 最後の面が逆向き
    )
    report, defects = inspect_surface_mesh(tet)
    assert report["watertight"] and defects is None
    assert report["n_inconsistent_edges"] == 3 and report["volume"] is None

    # 内向き法線の閉じた四面体 → 負の体積
    inward = pv.PolyData(tet.points, np.hstack([[3, 0, 1, 2], [3, 0, 3, 1], [3, 1, 3, 2], [3, 0, 2, 3]]))
    report, _ = inspect_surface_mesh(inward)
    assert report["n_inconsistent_edges"] == 0 and report["volume"] == pytest.approx(-1 / 6)


def test_two_closed_bodies_counted(multi_solid_step, cache_root):
    meta = import_cad_job({"path": str(multi_solid_step), "cache_root": str(cache_root), "max_parts": 1})
    entry = meta["parts"][0]
    result = inspect_part_job({"cache_dir": meta["cache_dir"], "surface": entry["surface"], "brep": entry["brep"]})
    assert result["mesh"]["watertight"] and result["mesh"]["n_components"] == 2
    assert result["brep"]["n_solids"] == 2
