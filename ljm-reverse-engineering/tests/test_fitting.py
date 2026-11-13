"""平面・円筒・円・球フィッティングの精度検証（真値既知のデータ / ノイズ有無）."""
import numpy as np
import pytest

from conftest import angle_deg
from ljm_re.core import synthetic as S
from ljm_re.core.fitting import (fit_circle_2d, fit_circle_3d, fit_cylinder, fit_plane, fit_plane_lsq,
                                 fit_sphere)


# ---------------------------------------------------------------- 平面
def test_plane_mesh_100mm_noiseless():
    m = S.make_plane_mesh(100.0, 50)
    f = fit_plane(np.asarray(m.vertices))
    assert angle_deg(f.normal, [0, 0, 1]) < 1e-9
    assert f.stats.rms < 1e-9
    size = f.extent_uv[1] - f.extent_uv[0]
    assert np.allclose(sorted(size), [100, 100], atol=1e-6)
    assert f.stats.n_used == len(m.vertices)


@pytest.mark.parametrize("noise", [0.02, 0.05])
def test_plane_noisy(noise):
    n = np.array([0.2, -0.1, 1.0])
    p = S.plane_points(20000, 100, noise, origin=(10, 20, 30), normal=n, seed=3)
    f = fit_plane(p)
    assert angle_deg(f.normal, n) < 0.02
    assert abs(f.stats.rms - noise) < 0.25 * noise
    # 平面上の点は真の平面上
    assert abs((f.origin - np.array([10, 20, 30])) @ (n / np.linalg.norm(n))) < 3 * noise / np.sqrt(len(p)) * 10


def test_plane_ransac_rejects_outliers():
    n = np.array([0, 0.3, 1.0])
    p = S.plane_points(10000, 100, 0.03, normal=n, outlier_ratio=0.25, seed=4)
    lsq = fit_plane_lsq(p)
    f = fit_plane(p, method="ransac")
    assert angle_deg(f.normal, n) < 0.05
    assert angle_deg(lsq.normal, n) > angle_deg(f.normal, n)
    assert 0.72 < f.stats.inlier_ratio < 0.78
    assert f.stats.threshold_auto


def test_plane_orient_hint():
    p = S.plane_points(1000, 50, 0.0, normal=(0, 0, 1))
    assert fit_plane(p, orient_hint=(0, 0, -1)).normal[2] < 0
    assert fit_plane(p, orient_hint=(0, 0, 1)).normal[2] > 0


# ---------------------------------------------------------------- 円筒
@pytest.mark.parametrize("use_normals", [True, False])
def test_cylinder_d50_noiseless(use_normals):
    axis = np.array([0.3, -0.2, 1.0])
    pts, nrm = S.partial_cylinder_points(50, 80, 360, 6000, 0.0, center=(5, -3, 12), axis=axis)
    f = fit_cylinder(pts, nrm if use_normals else None)
    assert abs(f.diameter - 50) < 1e-6
    assert angle_deg(f.axis_dir, axis) < 1e-5
    d = np.linalg.norm(np.cross(f.axis_point - np.array([5, -3, 12]), axis / np.linalg.norm(axis)))
    assert d < 1e-5
    assert f.coverage_deg > 350
    assert abs(f.length - 80) < 0.5


def test_cylinder_d50_noisy():
    pts, nrm = S.partial_cylinder_points(50, 80, 360, 8000, 0.05, seed=2)
    f = fit_cylinder(pts, nrm)
    assert abs(f.diameter - 50) < 0.01
    assert angle_deg(f.axis_dir, [0, 0, 1]) < 0.02
    assert abs(f.stats.rms - 0.05) < 0.01
    assert 0 < f.sigma_radius < 0.01


@pytest.mark.parametrize("arc", [180, 120, 90])
def test_partial_cylinder(arc):
    pts, nrm = S.partial_cylinder_points(50, 60, arc, 6000, 0.03, axis=(1, 1, 0.2), seed=5)
    f = fit_cylinder(pts, nrm)
    assert abs(f.diameter - 50) < 0.1
    assert angle_deg(f.axis_dir, [1, 1, 0.2]) < 0.1
    assert abs(f.coverage_deg - arc) < 3
    # 円弧が狭いほど不確かさが大きく、警告が出る
    if arc < 120:
        assert any("スキャン範囲" in w for w in f.warnings)
    assert f.sigma_radius > 0


def test_cylinder_uncertainty_grows_with_smaller_arc():
    s = []
    for arc in (360, 120, 60):
        pts, nrm = S.partial_cylinder_points(50, 60, arc, 5000, 0.05, seed=7)
        s.append(fit_cylinder(pts, nrm).sigma_radius)
    assert s[0] < s[1] < s[2]


def test_cylinder_with_outliers():
    pts, nrm = S.partial_cylinder_points(30, 50, 360, 5000, 0.02, seed=8)
    rng = np.random.default_rng(1)
    out = rng.uniform(-30, 30, (800, 3))
    on = rng.normal(size=(800, 3))
    on /= np.linalg.norm(on, axis=1)[:, None]
    f = fit_cylinder(np.vstack([pts, out]), np.vstack([nrm, on]))
    assert abs(f.diameter - 30) < 0.02
    assert f.stats.inlier_ratio < 0.95


def test_multi_cylinder_part_each_tube():
    mesh, truth = S.make_multi_cylinder_part()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    for t in truth:
        c, a, r, L = np.array(t["center"]), np.array(t["axis"], float), t["diameter"] / 2, t["length"]
        w = V - c
        h = w @ a
        rad = np.linalg.norm(w - np.outer(h, a), axis=1)
        sel = (np.abs(h) < L / 2 - 1e-6) & (np.abs(rad - r) < 1e-6)
        if sel.sum() < 50:
            sel = (np.abs(h) <= L / 2 + 1e-6) & (np.abs(rad - r) < 1e-6)
        f = fit_cylinder(V[sel], N[sel])
        assert abs(f.diameter - t["diameter"]) < 1e-4, t
        assert angle_deg(f.axis_dir, a) < 1e-3


def test_hole_wall_inward_normals():
    m = S.make_tube_mesh(20, 10, 96, 4, inward=True)
    f = fit_cylinder(np.asarray(m.vertices), np.asarray(m.vertex_normals))
    assert abs(f.diameter - 20) < 1e-6


# ---------------------------------------------------------------- 円・球
@pytest.mark.parametrize("arc", [360, 180, 90])
def test_circle_2d_arcs(arc):
    t = np.radians(np.linspace(10, 10 + arc, 200, endpoint=arc < 360))
    xy = np.column_stack([3 + 12.5 * np.cos(t), -4 + 12.5 * np.sin(t)])
    f = fit_circle_2d(xy)
    assert np.allclose(f.center, [3, -4], atol=1e-8)
    assert abs(f.radius - 12.5) < 1e-8


def test_circle_2d_noisy_ransac():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, 500)
    xy = np.column_stack([10 * np.cos(t), 10 * np.sin(t)]) + rng.normal(0, 0.02, (500, 2))
    xy = np.vstack([xy, rng.uniform(-5, 5, (60, 2))])
    f = fit_circle_2d(xy, ransac=True)
    assert abs(f.radius - 10) < 0.01
    assert np.linalg.norm(f.center) < 0.01
    assert f.stats.n_used <= 510


def test_circle_3d_tilted():
    from ljm_re.core.fitting.circle import circle_points_3d
    n = np.array([1.0, 2.0, 3.0])
    P = circle_points_3d([10, 20, 30], n, 7.5, 80)
    f = fit_circle_3d(P)
    assert np.allclose(f.center, [10, 20, 30], atol=1e-8)
    assert abs(f.diameter - 15) < 1e-8
    assert angle_deg(f.normal, n) < 1e-6


def test_sphere():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(3000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v = v[v[:, 2] > 0]  # 半球のみ
    P = v * 12.5 + [1, 2, 3] + rng.normal(0, 0.01, (len(v), 3))
    f = fit_sphere(P)
    assert np.linalg.norm(f.center - [1, 2, 3]) < 0.01
    assert abs(f.radius - 12.5) < 0.01
