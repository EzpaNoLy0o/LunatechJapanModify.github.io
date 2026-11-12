"""穴中心認識・断面・スケッチ化."""
import numpy as np
import pytest

from ljm_re.core import synthetic as S
from ljm_re.core.fitting import fit_cylinder, fit_plane
from ljm_re.core.hole import (detect_holes_on_plane, hole_from_cylinder, hole_from_edge_points,
                              hole_from_section)
from ljm_re.core.section import plane_basis, section_mesh, section_series
from ljm_re.core.sketch_fit import fit_spline_2d, segment_polyline

# 128分割の多角形近似による半径の内接誤差: r(1-cos(pi/128)) ≒ 3e-4 r
POLY_TOL = 0.01


def test_plate_hole_center_from_inner_wall():
    m = S.make_plate_with_holes(100, 100, 10, [(0, 0, 20)])
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    N = np.asarray(m.vertex_normals)
    r = np.hypot(V[:, 0], V[:, 1])
    wall = np.abs(r - 10) < 0.01
    cyl = fit_cylinder(V[wall], N[wall])
    top = V[:, 2] > 10 - 1e-9
    pl = fit_plane(V[top & ~wall])
    h = hole_from_cylinder(cyl, (pl.origin, pl.normal))
    assert np.allclose(h.center, [0, 0, 10], atol=1e-6)
    assert abs(h.diameter - 20) < POLY_TOL


def test_plate_hole_section_and_edge():
    m = S.make_plate_with_holes(100, 100, 10, [(0, 0, 20)])
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    h = hole_from_section(V, F, [0, 0, 10], [0, 0, 1], pick_point=[1, 1, 10], depth=0.5)
    assert np.allclose(h.center, [0, 0, 10], atol=1e-6)
    assert abs(h.diameter - 20) < POLY_TOL
    # 縁の点（上面の穴エッジ）
    edge = V[(np.abs(np.hypot(V[:, 0], V[:, 1]) - 10) < 0.01) & (V[:, 2] > 9.99)]
    he = hole_from_edge_points(edge)
    assert np.allclose(he.center, [0, 0, 10], atol=1e-6)
    assert abs(he.diameter - 20) < 1e-6  # 頂点は真円上


def test_partial_hole_section():
    """内周の一部しか無い穴（半円）でも断面の円フィットで中心を求められる."""
    m = S.make_tube_mesh(20, 10, 128, 4, arc_deg=150, inward=True, center=(5, 5, 0))
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    h = hole_from_section(V, F, [5, 5, 5], [0, 0, 1], depth=2.0)
    assert np.allclose(h.center[:2], [5, 5], atol=1e-3)
    assert abs(h.diameter - 20) < 0.01
    assert h.coverage_deg < 160


def test_bracket_auto_detect_holes():
    mesh, truth = S.make_bracket()
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    holes = detect_holes_on_plane(V, F, [0, 0, 8], [0, 0, 1], depth=0.5)
    assert len(holes) == 4
    for t in truth["holes"][:4]:
        c = np.array(t["center_top"])
        best = min(holes, key=lambda h: np.linalg.norm(h.center - c))
        assert np.linalg.norm(best.center - c) < 1e-4
        assert abs(best.diameter - t["diameter"]) < POLY_TOL


def test_section_plate_loops_and_sketch():
    m = S.make_plate_with_holes(100, 100, 10, [(0, 0, 20)])
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    sec = section_mesh(V, F, [0, 0, 5], [0, 0, 1])
    assert len(sec.polylines) == 2 and all(pl.closed for pl in sec.polylines)
    outer = segment_polyline(sec.to_2d(sec.polylines[0].points), True, 0.05)
    assert [e.kind for e in outer] == ["line"] * 4
    L = sorted(np.linalg.norm(e.end - e.start) for e in outer)
    assert np.allclose(L, 100, atol=1e-6)
    hole = segment_polyline(sec.to_2d(sec.polylines[1].points), True, 0.05)
    assert len(hole) == 1 and hole[0].kind == "circle"
    assert abs(hole[0].radius - 10) < POLY_TOL


def test_section_lines_and_arcs_rounded_rect():
    """角R付き矩形のポリライン → 直線4 + 円弧4 に分割される."""
    R, W, H = 5.0, 60.0, 30.0
    pts = []
    corners = [(W / 2 - R, H / 2 - R, 0), (-W / 2 + R, H / 2 - R, 90), (-W / 2 + R, -H / 2 + R, 180),
               (W / 2 - R, -H / 2 + R, 270)]
    for cx, cy, a0 in corners:
        t = np.radians(np.linspace(a0, a0 + 90, 20))
        pts.append(np.column_stack([cx + R * np.cos(t), cy + R * np.sin(t)]))
    P = np.vstack(pts)
    from ljm_re.core.section import resample_polyline
    P = resample_polyline(P, 0.5, closed=True)
    ents = segment_polyline(P, True, 0.02)
    kinds = [e.kind for e in ents]
    assert kinds.count("arc") == 4 and kinds.count("line") == 4, kinds
    for e in ents:
        if e.kind == "arc":
            assert abs(e.radius - R) < 0.05


def test_section_bracket_y_flange_hole():
    mesh, truth = S.make_bracket()
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    sec = section_mesh(V, F, [0, 44, 0], [0, 1, 0])
    circ = []
    for pl in sec.polylines:
        ents = segment_polyline(sec.to_2d(pl.points), pl.closed, 0.05)
        circ += [e for e in ents if e.kind == "circle"]
    assert len(circ) == 1
    c3 = sec.to_3d(circ[0].center[None])[0]
    assert np.allclose(c3[[0, 2]], [0, 38], atol=1e-4)
    assert abs(2 * circ[0].radius - 14) < POLY_TOL


def test_section_series_and_basis():
    m = S.make_tube_mesh(50, 100, 64, 10)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    res = section_series(V, F, [0, 0, 0], [0, 0, 1], np.linspace(-40, 40, 5))
    assert len(res) == 5 and all(len(r.polylines) == 1 for r in res)
    u, v = plane_basis([0, 1, 0])
    assert np.allclose(u, [-1, 0, 0]) or np.allclose(u, [1, 0, 0])
    assert v[2] > 0.99


def test_spline_fit_error():
    t = np.linspace(0, 1, 300)
    xy = np.column_stack([t * 100, 8 * np.sin(t * 7)])
    e = fit_spline_2d(xy, False, 0.01)
    assert e.max_error <= 0.01
    s = e.sample(400)
    assert np.allclose(s[0], xy[0], atol=0.02) and np.allclose(s[-1], xy[-1], atol=0.02)
