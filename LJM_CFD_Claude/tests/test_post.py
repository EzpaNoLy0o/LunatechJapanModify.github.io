"""後処理: 結果の読み込み、断面・流線・等値面の抽出。解析的に答えの分かる合成データで確かめる。"""

from __future__ import annotations

import time

import numpy as np
import pytest
import pyvista as pv

from ljmcfd.post.result_set import SURFACE_FIELDS, VOLUME_FIELDS, ResultSet, SeedSpec, default_seed
from tests.synthetic import DOMAIN, RHO, U0, make_results


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    folder = tmp_path_factory.mktemp("results")
    make_results(folder)
    return ResultSet(folder)


def test_loading_and_field_catalogue(results, tmp_path):
    assert results.has_fields and results.domain == pytest.approx(DOMAIN)
    assert results.available_fields() == ["Umag", "pressure", "Cp", "vorticity_mag", "k", "Q"]
    assert results.axis_range("x") == pytest.approx((-4.0, 8.0)) and results.axis_range("z") == pytest.approx((0.0, 3.0))
    assert results.free_stream_speed == U0
    # 断面・流線用の軽いデータは、必要な配列だけを持つ
    assert set(results.flow.point_data.keys()) == {"U", *VOLUME_FIELDS}
    assert results.flow.n_cells == results.volume.n_cells
    # カラーバーの見出しは VTK の既定フォントで表示できる ASCII のみ
    assert all(info.title.isascii() for info in (*VOLUME_FIELDS.values(), *SURFACE_FIELDS.values()))
    with pytest.raises(FileNotFoundError):
        ResultSet(tmp_path / "nothing")


def test_default_ranges_are_robust_and_sensible(results):
    lo, hi = results.default_range("Umag")
    assert lo == 0.0 and U0 * 1.25 < hi <= U0 * 1.3
    lo, hi = results.default_range("pressure")
    assert lo == pytest.approx(-hi) and 9.0 < hi <= 8.0 * RHO          # 0 を中心に対称
    assert results.default_range("Cp")[1] == 1.0                        # よどみ点 = 1 を上限にする
    lo, hi = results.default_range("yPlus", surface=True)
    assert 99.0 < lo < 105.0 and 115.0 < hi <= 120.0                    # Body の x = 0..2 → y+ = 100..120


def test_slice_is_on_the_plane_with_interpolated_values(results):
    for axis, position in (("x", 1.234), ("y", -0.4), ("z", 1.7)):
        cut = pv.wrap(results.slice(axis, position))
        index = "xyz".index(axis)
        assert cut.n_cells > 100
        assert np.allclose(cut.points[:, index], position, atol=1e-6)
        x, z = cut.points[:, 0], cut.points[:, 2]
        assert np.allclose(cut.point_data["Umag"], U0 * (1 + 0.1 * z), rtol=1e-5)
        assert np.allclose(cut.point_data["pressure"], -x * RHO, atol=1e-4)
        assert np.allclose(cut.point_data["Q"], z - 1.0, atol=1e-5)
        # 断面は領域の端から端まで広がる
        others = [i for i in range(3) if i != index]
        for i in others:
            assert cut.bounds[2 * i] == pytest.approx(DOMAIN[2 * i], abs=1e-6) and cut.bounds[2 * i + 1] == pytest.approx(DOMAIN[2 * i + 1], abs=1e-6)
    assert pv.wrap(results.slice("x", 100.0)).n_cells == 0              # 領域外は空


def test_slice_updates_are_fast_enough_for_a_slider(results):
    results.slice("y", 0.0)                                             # 1 回目は探索木の構築を含む
    positions = np.linspace(-1.9, 1.9, 40)
    started = time.perf_counter()
    for y in positions:
        results.slice("y", float(y))
    per_slice = (time.perf_counter() - started) / len(positions)
    assert per_slice < 0.05, f"断面 1 回 {per_slice * 1000:.1f} ms"


def test_seed_spec_geometry():
    seed = SeedSpec(x=-3.0, y=0.5, z=1.0, width=2.0, height=1.0, n_y=5, n_z=3)
    points = seed.points()
    assert points.shape == (15, 3) and seed.count == 15 and np.allclose(points[:, 0], -3.0)
    assert points[:, 1].min() == pytest.approx(-0.5) and points[:, 1].max() == pytest.approx(1.5)
    assert points[:, 2].min() == pytest.approx(0.5) and points[:, 2].max() == pytest.approx(1.5)
    assert seed.spacing() == pytest.approx((0.5, 0.5))
    assert SeedSpec(n_y=1, n_z=1, y=0.3, z=0.7).points().tolist() == [[0.0, 0.3, 0.7]]
    assert SeedSpec(n_y=1).spacing()[0] == 0.0
    assert SeedSpec.from_dict(seed.to_dict()) == seed and seed.outline().shape == (4, 3)
    # 既定の開始面: 車両前端の少し上流、車幅・車高より少し広く、地面より上
    auto = default_seed((0.0, 4.0, -0.9, 0.9, 0.0, 1.2))
    assert auto.x == pytest.approx(-1.0) and auto.y == 0.0 and auto.width == pytest.approx(2.16)
    assert auto.z - auto.height / 2 > 0.0 and auto.z + auto.height / 2 == pytest.approx(1.32)


def test_streamlines_follow_the_velocity_field(results):
    seed = SeedSpec(x=-3.5, y=0.0, z=1.5, width=2.0, height=2.0, n_y=4, n_z=5)
    lines = results.streamlines(seed)
    assert lines.n_cells == 20
    assert {"Umag", "pressure", "IntegrationTime"} <= set(lines.point_data.keys())
    for i in range(lines.n_cells):
        points = lines.get_cell(i).points
        # 速度は X 成分だけなので、流線は開始点から下流端まで Y・Z が変わらない直線
        assert np.ptp(points[:, 1]) < 1e-6 and np.ptp(points[:, 2]) < 1e-6
        # 終端は流出面の手前、積分の刻み幅 (セル長の半分) 以内で止まる
        assert points[0, 0] == pytest.approx(-3.5, abs=1e-6) and 7.8 < points[-1, 0] <= 8.0
    # 積分時間 = 距離 / 流速 (高い位置ほど速いので短時間で到達する)
    ids = lines.get_cell(0).point_ids
    z = lines.points[ids[0], 2]
    travel = lines.points[ids[-1], 0] - lines.points[ids[0], 0]
    assert lines.point_data["IntegrationTime"][ids[-1]] == pytest.approx(travel / (U0 * (1 + 0.1 * z)), rel=1e-3)
    # 上流側へも延ばすと、開始面より上流の点が現れる
    seed.both_directions = True
    both = results.streamlines(seed)
    assert both.points[:, 0].min() < -3.9
    # 領域外の開始点からは流線が出ない
    assert results.streamlines(SeedSpec(x=-50.0)).n_cells == 0


def test_isosurface(results):
    iso = results.isosurface("Q", 0.0)                                  # Q = z − 1 → 平面 z = 1
    assert iso.n_cells > 100 and np.allclose(iso.points[:, 2], 1.0, atol=1e-6)
    assert np.allclose(iso.point_data["Umag"], U0 * 1.1, rtol=1e-5)      # 他の物理量も補間されている (色付け用)
    speed = results.isosurface("Umag", U0 * 1.2)                        # 流速一定面 → z = 2
    assert np.allclose(speed.points[:, 2], 2.0, atol=1e-5)
    assert results.isosurface("Umag", 1.0e6).n_cells == 0


def test_surfaces(results):
    parts = results.part_surfaces()
    assert list(parts) == ["Body"] and results.ground_surface() is not None
    body = parts["Body"]
    assert {"pressure", "Cp", "wss_mag", "yPlus"} <= set(body.point_data.keys())
    # 壁面せん断応力は密度を掛けて Pa にしてある
    x = body.points[:, 0]
    assert np.allclose(body.point_data["wss_mag"], 0.01 * (1 + x) * RHO, rtol=1e-5)


# ----------------------------------------------------------------------------- 粒子アニメーション

def _animation(results, **kwargs):
    from ljmcfd.post.particles import SteadyPathlines

    seed = SeedSpec(x=-3.5, y=0.0, z=1.5, width=2.0, height=2.0, n_y=3, n_z=3)
    return SteadyPathlines(results.streamlines(seed), **kwargs), seed


def test_particles_move_along_the_steady_velocity_field(results):
    animation, seed = _animation(results, particles_per_line=4)
    assert animation.is_steady                                  # 定常場の軌跡であることをデータ源が明示する
    assert animation.n_lines == 9 and animation.n_particles == 36
    # タイムラインの長さ: 80 % の流線が流出面に届く時間 (低い位置ほど遅い)
    assert 11.3 / (U0 * 1.25) < animation.duration < 11.6 / U0
    # ループなし・時刻 0: 各流線の 1 個目の粒子だけが開始面にいる
    start = animation.frame(0.0, loop=False)
    assert start.points.shape == (36, 3) and start.visible.sum() == 9
    assert np.allclose(start.points[start.visible][:, 0], -3.5, atol=1e-6)
    # 時刻 t: x = x0 + U(z) · t (速度は X 成分のみ、高さで決まる)
    t = 0.05
    frame = animation.frame(t, loop=False)
    first = frame.points.reshape(9, 4, 3)[:, 0, :]              # 各流線の 1 個目の粒子
    z = first[:, 2]
    assert np.allclose(first[:, 0], -3.5 + U0 * (1 + 0.1 * z) * t, atol=2e-3)
    assert np.allclose(first[:, 1:], seed.points()[:, 1:], atol=1e-6)      # Y・Z は変わらない
    assert np.allclose(frame.speed.reshape(9, 4)[:, 0], U0 * (1 + 0.1 * z), rtol=1e-4)
    # 2 個目以降は放出時刻が遅れるので、まだ出ていない (duration/4 > 0.05)
    assert frame.visible.reshape(9, 4)[:, 1:].sum() == 0


def test_timeline_is_reversible_and_loops_seamlessly(results):
    animation, _ = _animation(results, particles_per_line=5)
    duration = animation.duration
    forward = [animation.frame(t, loop=True).points for t in (0.01, 0.02, 0.03)]
    # 同じ時刻には必ず同じ位置 → タイムラインを戻せば粒子は元の位置に戻る
    assert np.array_equal(animation.frame(0.02, loop=True).points, forward[1])
    assert np.array_equal(animation.frame(0.01, loop=True).points, forward[0])
    # 時間を進めると下流 (+X) へ、戻すと上流へ
    visible = animation.frame(0.02, loop=True).visible & animation.frame(0.03, loop=True).visible
    moved = (forward[2][:, 0] - forward[1][:, 0])[visible]
    assert (moved[np.abs(moved) < 1.0] > 0).all()
    # ループ: 時刻 0 と duration は同じ絵 (切れ目なくつながる)
    assert np.allclose(animation.frame(0.0, loop=True).points, animation.frame(duration, loop=True).points, atol=1e-9)
    assert np.allclose(animation.frame(0.3 * duration, loop=True).points, animation.frame(1.3 * duration, loop=True).points, atol=1e-6)
    # ループ時は放出済みの粒子が流線に沿って並ぶ (5 個中、終点に達していないものが見える)
    looped = animation.frame(0.5 * duration, loop=True)
    assert looped.visible.sum() >= 0.6 * animation.n_particles
    # ループなしでは、終了時刻を過ぎた粒子は消える
    late = animation.frame(10 * duration, loop=False)
    assert late.visible.sum() == 0
    assert np.isfinite(late.points).all()                       # 見えない粒子にも非数は入れない
    assert (late.points[:, 0] > 7.5).all() and (late.points[:, 0] <= 8.0 + 1e-6).all()   # 軌跡の終点 (流出面の手前) に留まる


def test_smoke_trails_and_density(results):
    smoke, _ = _animation(results, particles_per_line=6, trail_points=8)
    assert smoke.n_particles == 9 * 6 * 8
    frame = smoke.frame(0.6 * smoke.duration, loop=True)
    points = frame.points.reshape(9, 6, 8, 3)
    fade = frame.fade.reshape(9, 6, 8)
    visible = frame.visible.reshape(9, 6, 8)
    # 尾は先頭より上流側に伸び、先頭ほど濃い
    assert fade[0, 0, 0] == 1.0 and (np.diff(fade[0, 0]) < 0).all() and fade[0, 0, -1] == pytest.approx(1 / 8)
    full = visible.all(axis=2)
    assert full.sum() > 10
    assert (np.diff(points[full][:, :, 0], axis=1) < 0).all()
    # 尾を 1 本の折れ線として描くための接続情報: 粒子ごとに 8 点
    cells = smoke.trail_connectivity().reshape(9 * 6, 9)
    assert (cells[:, 0] == 8).all() and np.array_equal(cells[:, 1:].ravel(), np.arange(9 * 6 * 8))
    # まだ放出されていない尾の点は始点に、流出した先頭は終点に留まる (筋が領域を横切って伸びない)
    hidden_x = points[~visible][:, 0]
    assert ((np.abs(hidden_x + 3.5) < 1e-6) | (hidden_x > 7.5)).all() and points[..., 0].max() <= 8.0 + 1e-6
    step = np.linalg.norm(np.diff(points, axis=2), axis=3)
    assert step.max() < 0.25 * 11.5                              # 1 区間の長さは尾の長さ程度まで (全長 11.5 m)
    # 粒子密度を上げると粒子が増える
    dense, _ = _animation(results, particles_per_line=20)
    assert dense.n_particles == 9 * 20 and len(dense.trail_connectivity()) == 0   # 点表示では線を作らない
    # 流線が無ければ空 (エラーにしない)
    from ljmcfd.post.particles import SteadyPathlines

    empty = SteadyPathlines(results.streamlines(SeedSpec(x=-50.0)))
    assert empty.n_particles == 0 and empty.duration == 0.0 and empty.frame(0.1, True).points.shape == (0, 3)


def test_particle_frames_are_fast(results):
    from ljmcfd.post.particles import SteadyPathlines

    lines = results.streamlines(SeedSpec(x=-3.5, y=0.0, z=1.5, width=3.0, height=2.5, n_y=12, n_z=8))
    animation = SteadyPathlines(lines, particles_per_line=20, trail_points=10)
    assert animation.n_particles == 96 * 20 * 10
    started = time.perf_counter()
    for t in np.linspace(0.0, animation.duration, 60):
        animation.frame(float(t), loop=True)
    per_frame = (time.perf_counter() - started) / 60
    assert per_frame < 0.03, f"1 フレーム {per_frame * 1000:.1f} ms"
