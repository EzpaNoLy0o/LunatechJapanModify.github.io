"""CAD 再構築（OCP）: B-Rep 生成・STEP 出力と再読込検証・偏差・自由曲面."""
import numpy as np
import pytest

occ = pytest.importorskip("ljm_re.cad.occ_backend")
if not occ.available():
    pytest.skip("OCP が利用できません", allow_module_level=True)

from ljm_re.cad import reconstruct as rc  # noqa: E402
from ljm_re.core import synthetic as S  # noqa: E402
from ljm_re.core.fitting import fit_cylinder, fit_plane  # noqa: E402
from ljm_re.core.section import section_mesh  # noqa: E402
from ljm_re.model.objects import CylinderFeature, HoleFeature, PlaneFeature, SectionObject  # noqa: E402


def _plate_section():
    m = S.make_plate_with_holes(100, 100, 10, [(0, 0, 20)])
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    sec = SectionObject.from_result(section_mesh(V, F, [0, 0, 5], [0, 0, 1]))
    sec.sketch = rc.sketch_section(sec, 0.05)
    return m, sec


def test_features_to_brep_and_step(jp_dir):
    """フィット結果 → B-Rep → STEP 書込み → 再読込 → 形状・寸法の一致を検証."""
    pts, nrm = S.partial_cylinder_points(50, 80, 360, 4000, 0.0, center=(10, 0, 0), axis=(0, 0, 1))
    cyl = CylinderFeature.from_fit(fit_cylinder(pts, nrm))
    plane = PlaneFeature.from_fit(fit_plane(S.plane_points(2000, 100, 0.0, origin=(0, 0, 50))))
    s_cyl, _ = rc.cylinder_to_shape(cyl, "solid")
    s_face, _ = rc.cylinder_to_shape(cyl, "face")
    s_pl, _ = rc.plane_to_face(plane)
    s_hole, _ = rc.hole_to_shape(HoleFeature("h", [0, 0, 0], [0, 0, 1], 12.0), "edge")
    for s in (s_cyl, s_face, s_pl):
        assert occ.is_valid(s)
    p = str(jp_dir / "再構築モデル.step")
    occ.write_step([(s_cyl, "Cylinder"), (s_face, "円筒面"), (s_pl, "Plane"), (s_hole, "Hole")], p)
    rep = occ.verify_step(p, [s_cyl, s_face, s_pl, s_hole])
    assert rep["ok"], rep["issues"]
    rd = occ.read_step(p)
    faces = occ.analyze_faces(rd)
    radii = sorted(f["radius"] for f in faces if f["type"] == "円筒")
    assert np.allclose(radii, [25, 25], atol=1e-9)
    planes = [f for f in faces if f["type"] == "平面"]
    assert any(abs(f["origin"][2] - 50) < 1e-9 for f in planes)
    s = occ.summarize(rd)
    assert s["n_solids"] == 1
    assert s["volume"] == pytest.approx(np.pi * 25 ** 2 * cyl.length, rel=1e-9)


def test_extrude_section_matches_mesh(jp_dir):
    m, sec = _plate_section()
    sec.origin = sec.origin - np.array([0, 0, 5])
    solid, recipe = rc.extrude_section(sec, 10.0)
    s = occ.summarize(solid)
    assert s["valid"] and s["n_solids"] == 1
    assert s["face_types"] == {"平面": 6, "円筒": 1}
    r = [f["radius"] for f in s["faces"] if "radius" in f][0]
    assert abs(r - 10) < 0.01
    # 元メッシュとの偏差（ほぼゼロ）
    idx, d = rc.deviation_points_to_shape(np.asarray(m.vertices), solid, 0.01)
    assert np.abs(d).max() < 0.01
    p = str(jp_dir / "押し出し.step")
    occ.write_step([(solid, "plate")], p)
    assert occ.verify_step(p, [solid])["ok"]


def test_revolve_bushing():
    V, F = occ.tessellate(occ.boolean(occ.make_cylinder_solid([0, 0, 0], [0, 0, 1], 25, 0, 40),
                                      occ.make_cylinder_solid([0, 0, 0], [0, 0, 1], 12, 0, 40), "cut"), 0.01)
    sec = SectionObject.from_result(section_mesh(V, F, [0, 0, 0], [0, 1, 0]))
    assert len(sec.polylines) == 2 and all(sec.closed)
    solid, _ = rc.revolve_section(sec, [0, 0, 0], [0, 0, 1], loop=0, tol=0.02)
    s = occ.summarize(solid)
    assert s["valid"]
    assert s["volume"] == pytest.approx(np.pi * (25 ** 2 - 12 ** 2) * 40, rel=2e-3)


def test_brep_bytes_roundtrip_and_display_data():
    s = occ.make_cylinder_solid([0, 0, 0], [0, 0, 1], 5, 0, 10)
    V, F = occ.tessellate(s, 0.05)
    assert len(F) > 20 and F.max() < len(V)
    edges = occ.edge_polylines(s)
    assert len(edges) == 3  # 上下の円 + シーム
    s2 = occ.shape_from_brep_bytes(occ.shape_to_brep_bytes(s))
    assert occ.summarize(s2)["volume"] == pytest.approx(np.pi * 25 * 10, rel=1e-9)


def test_transform_and_boolean():
    a = occ.make_cylinder_solid([0, 0, 0], [0, 0, 1], 10, 0, 20)
    T = np.eye(4)
    T[:3, 3] = [5, 0, 0]
    b = occ.transform_shape(a, T)
    assert occ.bbox(b)[0][0] == pytest.approx(-5, abs=1e-6)
    u = occ.boolean(a, b, "fuse")
    assert occ.is_valid(u)
    c = occ.boolean(a, b, "common")
    assert occ.summarize(c)["volume"] < occ.summarize(a)["volume"]


def test_loft_and_bspline_surface():
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    curves = [np.column_stack([(20 + z / 5) * np.cos(t), (20 + z / 5) * np.sin(t), np.full(100, z)])
              for z in (0, 20, 40)]
    lf, _ = rc.loft_curves(curves, True, solid=True)
    s = occ.summarize(lf)
    assert s["valid"] and s["n_solids"] == 1
    # 円錐台の体積
    h, r1, r2 = 40, 20, 28
    assert s["volume"] == pytest.approx(np.pi * h / 3 * (r1 ** 2 + r1 * r2 + r2 ** 2), rel=5e-3)
    rng = np.random.default_rng(0)
    g = rng.uniform(-50, 50, (5000, 2))
    hgt = 0.002 * g[:, 0] ** 2 + 0.001 * g[:, 0] * g[:, 1] + 0.5 * np.sin(g[:, 1] / 15)
    pts = np.column_stack([g, hgt + rng.normal(0, 0.02, len(g))])
    face, info = rc.fit_bspline_surface(pts, 8, 8)
    assert info["rms"] < 0.03 and occ.is_valid(face)
    _, d = rc.deviation_points_to_shape(pts, face, 0.01)
    assert np.sqrt(np.mean(d ** 2)) < 0.035


def test_analytic_deviation():
    pl = PlaneFeature("p", [0, 0, 0], [0, 0, 1])
    d = rc.analytic_deviation(pl, np.array([[0, 0, 1.5], [3, 3, -2]]))
    assert np.allclose(d, [1.5, -2])


def test_partial_cylinder_face_matches_scanned_arc():
    """部分円筒面はスキャンされた円弧の範囲（開始角・範囲）で作られる."""
    pts, nrm = S.partial_cylinder_points(50, 40, 120, 4000, 0.0, axis=(0, 0, 1), seed=11)
    # 円弧を 200°〜320° の位置へ回転
    from ljm_re.core.geometry import rot_z
    R = rot_z(200)
    pts, nrm = pts @ R.T, nrm @ R.T
    cyl = CylinderFeature.from_fit(fit_cylinder(pts, nrm))
    face, _ = rc.cylinder_to_shape(cyl, "partial")
    area = occ.summarize(face)["area"]
    assert area == pytest.approx(2 * np.pi * 25 * cyl.length * cyl.coverage_deg / 360, rel=1e-6)
    # 面上の点は元の点群の近く（偏差ほぼゼロ）
    _, d = rc.deviation_points_to_shape(pts, face, 0.005)
    assert np.abs(d).max() < 0.01
