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
