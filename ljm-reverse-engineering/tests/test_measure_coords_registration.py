"""測定・座標系・位置合わせ・選択."""
import numpy as np
import pytest

from conftest import angle_deg
from ljm_re.core import measure as M
from ljm_re.core import synthetic as S
from ljm_re.core.coords import (Frame, frame_from_321, frame_from_axis_point, frame_from_plane_axis,
                                frame_from_three_points)
from ljm_re.core.geometry import euler_xyz_to_matrix, matrix_to_euler_xyz
from ljm_re.core.registration import evaluate_alignment, icp, kabsch
from ljm_re.core.selection import MeshTopology, brush_select, rect_select_points, region_grow


def test_point_distance_and_components():
    f = Frame.from_euler([0, 0, 0], 0, 0, 90)
    r = M.distance_point_point([0, 0, 0], [3, 4, 0], f)
    assert r["distance"] == pytest.approx(5)
    assert r["dx"] == pytest.approx(4) and r["dy"] == pytest.approx(-3)


def test_angle_and_point_plane():
    assert M.angle_three_points([1, 0, 0], [0, 0, 0], [0, 1, 0])["angle"] == pytest.approx(90)
    r = M.point_plane([1, 2, 5], [0, 0, 1], [0, 0, 2])
    assert r["signed_distance"] == pytest.approx(4)
    assert np.allclose(r["foot"], [1, 2, 1])


def test_plane_plane():
    r = M.plane_plane([0, 0, 0], [0, 0, 1], [5, 5, 12.5], [0, 0, -1])
    assert r["parallel"] and r["distance"] == pytest.approx(12.5)
    r = M.plane_plane([0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0])
    assert r["perpendicular"] and r["angle"] == pytest.approx(90)


def test_axis_axis_skew_and_coaxiality():
    r = M.axis_axis([0, 0, 0], [1, 0, 0], [0, 0, 10], [0, 1, 0])
    assert r["distance"] == pytest.approx(10) and r["angle"] == pytest.approx(90)
    # 0.1mm オフセット + 傾き: 同軸度 = 2 * 最大偏位
    r = M.axis_axis([0, 0, 0], [0, 0, 1], [0.1, 0, 0], [0, 0, 1], extent2=(-20, 20))
    assert r["coaxiality"] == pytest.approx(0.2)


def test_projected_bbox_scale():
    r = M.projected_distance([0, 0, 0], [3, 4, 5], [0, 0, 1])
    assert r["distance"] == pytest.approx(5) and r["perpendicular"] == pytest.approx(5)
    b = M.bounding_box(np.array([[0, 0, 0], [10, 20, 30]]))
    assert b["size"] == [10, 20, 30]
    s = M.scale_factor(99.5, 100.0)
    assert s["factor"] == pytest.approx(100 / 99.5)


def test_frames():
    f = frame_from_three_points([1, 1, 1], [5, 1, 1], [1, 4, 1])
    assert np.allclose(f.R, np.eye(3))
    assert np.allclose(f.to_world(f.to_local([3, 7, -2])), [3, 7, -2])
    # 3-2-1: Z=上面, X=側面法線, 原点=3平面の交点
    f = frame_from_321(([0, 0, 8], [0, 0, 1]), ([75, 0, 0], [1, 0, 0]), ([0, -40, 0], [0, -1, 0]))
    assert np.allclose(f.origin, [75, -40, 8])
    assert np.allclose(f.z_axis, [0, 0, 1]) and np.allclose(f.x_axis, [1, 0, 0])
    f = frame_from_plane_axis([0, 0, 10], [0, 0, 1], [5, 5, 0], [1, 0, 1])
    assert np.allclose(f.origin, [15, 5, 10])
    assert np.allclose(f.x_axis, [1, 0, 0])
    f = frame_from_axis_point([0, 0, 0], [0, 0, 1], [0, 10, 3])
    assert np.allclose(f.origin, [0, 0, 3]) and np.allclose(f.x_axis, [0, 1, 0])
    assert f.is_valid()


def test_euler_roundtrip():
    for ang in [(10, 20, 30), (-45, 5, 170), (0, 89, 0)]:
        R = euler_xyz_to_matrix(*ang)
        assert np.allclose(euler_xyz_to_matrix(*matrix_to_euler_xyz(R)), R, atol=1e-9)


def test_kabsch_exact():
    rng = np.random.default_rng(0)
    A = rng.uniform(-50, 50, (6, 3))
    R = euler_xyz_to_matrix(12, -30, 45)
    B = A @ R.T + [1, 2, 3]
    T, rms = kabsch(A, B)
    assert rms < 1e-9 and np.allclose(T[:3, :3], R)


@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_icp_recovers_transform(noise):
    mesh, _ = S.make_bracket()
    V = np.asarray(mesh.vertices)
    tgt, _ = S.sample_surface(mesh, 40000, noise, 2)
    from ljm_re.core.mesh_ops import estimate_normals
    tn = estimate_normals(tgt, 20)
    R = euler_xyz_to_matrix(2.0, -1.5, 3.0)
    t = np.array([1.5, -1.0, 0.8])
    src_true, _ = S.sample_surface(mesh, 20000, noise, 5)
    src = (src_true - t) @ R  # src を T^-1 で動かしたもの
    r = icp(src, tgt, tn, max_iterations=100)
    T = r.transform
    assert np.degrees(np.arccos((np.trace(T[:3, :3].T @ R) - 1) / 2)) < 0.05
    assert np.linalg.norm(T[:3, 3] - t) < 0.05
    assert r.fitness > 0.95
    ev = evaluate_alignment(src @ T[:3, :3].T + T[:3, 3], tgt, 1.0)
    assert ev["overlap_ratio"] > 0.95


def test_region_grow_selects_top_face_only():
    mesh, _ = S.make_bracket()
    topo = MeshTopology(mesh.vertices, mesh.faces)
    f = topo.nearest_face([0, -30, 8])
    m = region_grow(topo, f, 5.0)
    z = topo.centroids[m][:, 2]
    assert np.allclose(z, 8.0) and m.sum() > 50
    # 曲面（穴の内壁）は角度許容内で連続に辿れる
    c = topo.centroids
    wall = int(np.argmin(np.abs(np.hypot(c[:, 0] - 30, c[:, 1] - 25) - 8) + np.abs(c[:, 2] - 4)))
    w = region_grow(topo, wall, 5.0)
    rr = np.hypot(c[w][:, 0] - 30, c[w][:, 1] - 25)
    assert np.allclose(rr, 8, atol=0.01)


def test_brush_and_rect_select():
    m = S.make_plane_mesh(100, 50)
    topo = MeshTopology(m.vertices, m.faces)
    b = brush_select(topo, [0, 0, 0], 10.0)
    cr = np.hypot(topo.centroids[:, 0], topo.centroids[:, 1])
    vr = np.hypot(topo.V[:, 0], topo.V[:, 1])[topo.F].min(axis=1)
    assert np.all(b[cr <= 10.0])                 # 重心が半径内の面はすべて選択
    assert np.all(vr[b] <= 10.0 + 1e-9)          # 選択面は半径内に頂点を持つ
    # 正射影の MVP（XY → 画面）
    mvp = np.diag([1 / 50, 1 / 50, -1 / 100, 1.0])
    sel = rect_select_points(m.vertices, mvp, (1000, 1000), (500, 500, 1000, 1000))
    P = np.asarray(m.vertices)[sel]
    assert np.all(P[:, 0] >= 0) and np.all(P[:, 1] >= 0)
