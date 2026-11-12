"""ファイル入出力（日本語パス）・メッシュ編集・プロジェクト保存/再読込・CSV/DXF/Undo."""
import csv
import os

import numpy as np
import pytest

from ljm_re.core import mesh_ops
from ljm_re.core import synthetic as S
from ljm_re.core.coords import Frame
from ljm_re.core.fitting import fit_cylinder, fit_plane
from ljm_re.core.section import section_mesh
from ljm_re.io.export_csv import export_features_csv, export_measurements_csv
from ljm_re.io.mesh_io import export_mesh, guess_unit, load_mesh_file
from ljm_re.model.document import Document
from ljm_re.model.objects import (CoordinateSystemObject, CylinderFeature, HoleFeature, MeasurementObject,
                                  MeshObject, PlaneFeature, PointCloudObject, PointFeature, SectionObject)
from ljm_re.model.project_io import export_project_json, load_project, save_project


@pytest.mark.parametrize("ext", [".stl", ".obj", ".ply"])
def test_mesh_roundtrip_japanese_path(jp_dir, ext):
    m = S.make_plate_with_holes()
    p = str(jp_dir / f"プレート{ext}")
    export_mesh(p, m.vertices, m.faces)
    d = load_mesh_file(p)
    assert d.kind == "mesh"
    assert d.n_faces == len(m.faces)
    assert np.allclose(d.vertices.min(0), m.vertices.min(0), atol=1e-5)
    assert d.unit_guess == "mm" and len(d.sha256) == 64


def test_point_cloud_ply(jp_dir):
    pts, _ = S.sample_surface(S.make_plane_mesh(), 2000)
    p = str(jp_dir / "点群.ply")
    export_mesh(p, pts, None)
    d = load_mesh_file(p)
    assert d.kind == "points" and d.n_vertices == 2000


def test_unit_guess():
    assert guess_unit([0.3, 0.2, 0.1])[0] == "m"
    assert guess_unit([300, 200, 100])[0] == "mm"


def test_mesh_ops():
    m = S.make_plate_with_holes()
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    mask = np.zeros(len(F), bool)
    mask[:10] = True
    V2, F2, fl, _ = mesh_ops.remove_faces(V, F, mask, np.zeros(len(F), np.uint8))
    assert len(F2) == len(F) - 10 and F2.max() < len(V2)
    # 浮遊片除去
    small = S.make_tube_mesh(2, 2, 8, 1, center=(200, 0, 0))
    Vc, Fc, flc, _ = mesh_ops.merge_meshes([(V, F, None, None), (small.vertices, small.faces, None, None)])
    V3, F3, _, _, removed, ncomp = mesh_ops.remove_small_components(Vc, Fc, min_faces=100)
    assert removed == len(small.faces) and ncomp == 1
    parts = mesh_ops.split_components(Vc, Fc)
    assert len(parts) == 2
    # 穴埋め: 生成面は人工面フラグ
    tube = S.make_tube_mesh(50, 50, 64, 5)
    V4, F4, fl4, _, n = mesh_ops.fill_holes(tube.vertices, tube.faces, 100.0)
    assert n > 0 and (fl4 & mesh_ops.FLAG_ARTIFICIAL).sum() == n
    assert np.all(fl4[:len(tube.faces)] == 0)
    assert mesh_ops.boundary_loops(F4) == 0
    # 簡略化
    dense = S.make_tube_mesh(50, 50, 128, 40)
    V5, F5, _ = mesh_ops.decimate(np.asarray(dense.vertices), np.asarray(dense.faces), 0.5)
    assert len(F5) < 0.6 * len(dense.faces)
    # スムージングは頂点数を保つ
    Vs = mesh_ops.smooth_taubin(V, F, 3)
    assert Vs.shape == V.shape
    # 点群外れ値
    pts = np.vstack([np.random.default_rng(0).normal(0, 1, (1000, 3)), [[50, 50, 50]]])
    keep = mesh_ops.statistical_outlier_removal(pts)
    assert not keep[-1]


def _build_doc():
    doc = Document()
    plate = S.make_plate_with_holes()
    mesh = MeshObject("プレート", plate.vertices, plate.faces)
    mesh.source_path = "C:/スキャン/plate.stl"
    doc.add(mesh)
    V = mesh.V
    top = V[:, 2] > 9.999
    pl = PlaneFeature.from_fit(fit_plane(V[top]), "上面")
    pl.critical = True
    pl.meta = {"source_ids": [mesh.id]}
    doc.add(pl)
    r = np.hypot(V[:, 0], V[:, 1])
    wall = np.abs(r - 10) < 0.01
    cyl = CylinderFeature.from_fit(fit_cylinder(V[wall], mesh.vertex_normals[wall]), "穴の内壁")
    doc.add(cyl)
    h = HoleFeature("穴1", [0, 0, 10], [0, 0, 1], 20.0)
    h.critical = True
    h.verified = True
    h.verified_note = "ノギス実測 φ20.02"
    doc.add(h)
    doc.add(PointFeature("基準点A", [1, 2, 3]))
    cs = CoordinateSystemObject("基準座標系", Frame.from_euler([10, 0, 0], 0, 0, 90))
    doc.add(cs)
    doc.set_active_cs(cs.id)
    sec = SectionObject.from_result(section_mesh(V, mesh.F, [0, 0, 5], [0, 0, 1]), "断面Z5")
    from ljm_re.cad.reconstruct import sketch_section
    sec.sketch = sketch_section(sec, 0.05)
    doc.add(sec)
    from ljm_re.core import measure as M
    doc.add(MeasurementObject("距離1", "point_point", M.distance_point_point([0, 0, 0], [3, 4, 0]),
                              [np.zeros(3), np.array([3, 4, 0.0])]))
    pc = PointCloudObject("点群", np.random.default_rng(0).normal(size=(100, 3)))
    doc.add(pc)
    return doc


def test_project_save_load_japanese(jp_dir):
    doc = _build_doc()
    p = str(jp_dir / "作業プロジェクト.ljmre")
    save_project(doc, p, {"length_decimals": 4})
    doc2, settings = load_project(p)
    assert settings["length_decimals"] == 4
    assert [o.kind for o in doc2.objects()] == [o.kind for o in doc.objects()]
    assert doc2.active_cs_id == doc.active_cs_id
    for a, b in zip(doc.objects(), doc2.objects()):
        assert a.name == b.name and a.id == b.id and a.critical == b.critical
    m1, m2 = doc.objects("mesh")[0], doc2.objects("mesh")[0]
    assert np.array_equal(m1.V, m2.V) and np.array_equal(m1.F, m2.F)
    assert np.array_equal(m1.original_V, m2.original_V)
    assert m2.source_path == "C:/スキャン/plate.stl"
    c1, c2 = doc.objects("cylinder")[0], doc2.objects("cylinder")[0]
    assert c2.radius == pytest.approx(c1.radius) and c2.stats == c1.stats
    h2 = doc2.objects("hole")[0]
    assert h2.verified and h2.verified_note == "ノギス実測 φ20.02"
    s1, s2 = doc.objects("section")[0], doc2.objects("section")[0]
    assert len(s2.polylines) == len(s1.polylines) and s2.sketch == s1.sketch
    assert np.allclose(doc2.active_frame.origin, [10, 0, 0])
    j = str(jp_dir / "情報.json")
    export_project_json(doc2, j)
    assert os.path.getsize(j) > 100


def test_undo_redo_document():
    doc = _build_doc()
    n = len(doc)
    pl = doc.objects("plane")[0]
    old = pl.origin.copy()
    T = np.eye(4)
    T[:3, 3] = [5, 0, 0]
    doc.transform(doc.ids(), T, "移動")
    assert np.allclose(pl.origin, old + [5, 0, 0])
    m = doc.objects("mesh")[0]
    assert np.allclose(m.pose[:3, 3], [5, 0, 0])
    doc.undo()
    assert np.allclose(pl.origin, old)
    doc.redo()
    assert np.allclose(pl.origin, old + [5, 0, 0])
    doc.set_property(pl.id, "name", "新しい名前")
    assert pl.name == "新しい名前"
    doc.undo()
    assert pl.name == "上面"
    doc.remove([pl.id])
    assert len(doc) == n - 1
    doc.undo()
    assert len(doc) == n and doc.get(pl.id) is pl
    # メッシュ編集の Undo と オリジナル復元
    F0 = m.F
    doc.modify(m.id, lambda o: o.set_geometry(*mesh_ops.remove_faces(o.V, o.F, np.arange(len(o.F)) < 5)), "削除")
    assert len(m.F) == len(F0) - 5
    doc.undo()
    assert m.F is F0
    doc.modify(m.id, lambda o: o.set_geometry(*mesh_ops.remove_faces(o.V, o.F, np.arange(len(o.F)) < 5)), "削除")
    doc.modify(m.id, lambda o: o.restore_original(), "復元")
    assert len(m.F) == len(F0)
    assert np.allclose(m.V, m.original_V + [5, 0, 0])  # 位置合わせ（pose）は保持


def test_property_edit_in_active_frame():
    doc = _build_doc()
    p = doc.objects("point")[0]
    f = doc.active_frame  # 原点(10,0,0), Z周り90°
    doc.set_property(p.id, "position", [1.0, 0.0, 0.0])
    assert np.allclose(p.position, f.to_world([1, 0, 0]))
    assert np.allclose(p.position, [10, 1, 0])


def test_csv_exports(jp_dir):
    doc = _build_doc()
    p = str(jp_dir / "座標一覧.csv")
    n = export_features_csv(doc, p, doc.active_frame, doc.active_frame_name)
    assert n >= 5
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    text = "\n".join(",".join(r) for r in rows)
    assert "実測未確認" in text and "上面" in text  # 安全上重要・未確認の警告
    hole = next(r for r in rows if r and r[1] == "穴1")
    local = doc.active_frame.to_local([0, 0, 10])
    assert float(hole[3]) == pytest.approx(local[0], abs=1e-4)
    assert hole[19] == "済"
    p2 = str(jp_dir / "測定.csv")
    assert export_measurements_csv(doc, p2) == 1


def test_dxf_export(jp_dir):
    import ezdxf
    from ljm_re.io.dxf_export import export_points_dxf, export_sections_dxf
    doc = _build_doc()
    sec = doc.objects("section")[0]
    p = str(jp_dir / "断面.dxf")
    counts = export_sections_dxf([sec], p, "2d")
    assert counts["line"] == 4 and counts["circle"] == 1
    d = ezdxf.readfile(p)
    msp = d.modelspace()
    circles = list(msp.query("CIRCLE"))
    assert len(circles) == 1 and abs(circles[0].dxf.radius - 10) < 0.01
    assert len(list(msp.query("LINE"))) == 4
    p3 = str(jp_dir / "断面3d.dxf")
    c3 = export_sections_dxf([sec], p3, "3d")
    assert c3["polyline"] == 2
    p4 = str(jp_dir / "点.dxf")
    export_points_dxf([[0, 0, 0], [1, 2, 3]], ["原点", "P1"], p4)
    assert len(list(ezdxf.readfile(p4).modelspace().query("POINT"))) == 2
