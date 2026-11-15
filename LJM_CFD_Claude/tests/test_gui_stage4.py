"""Stage 4 の GUI 結合テスト: 解析結果の 3D 可視化 (表面分布・断面・流線・等値面・カラースケール)。

解析的に値の分かる合成データを使うので、OpenFOAM が無い環境 (Windows) でも実行できる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("ディスプレイが無いため GUI テストをスキップ (xvfb-run で実行してください)", allow_module_level=True)

from ljmcfd.gui.post_view import SOLID  # noqa: E402
from ljmcfd.solver.environment import native_openfoam_available  # noqa: E402
from tests.synthetic import BODY, U0, make_results  # noqa: E402
from tests.test_gui import TIMEOUT, load, window  # noqa: E402,F401  (window はフィクスチャ)

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")


def with_results(window):
    """合成の解析結果を、アプリの結果フォルダへ置いて読み込ませる。"""
    summary = make_results(window.results_dir())
    window.scene.set_solver_summary(summary)
    return window.post_view, window.post_panel


def actor(window, name):
    return window.viewport.plotter.actors[name]


def dataset(window, name):
    """アクターが描いているデータ。

    滑らかな陰影を付けるレイヤー (表面・等値面) は法線計算のフィルタを挟むため、
    最初の描画まで中身が空に見える。必ず描画してから取り出す。
    """
    window.viewport.plotter.render()
    return actor(window, name).mapper.dataset


def actor_points(window, name) -> np.ndarray:
    points = np.asarray(dataset(window, name).points)
    assert len(points) > 0, f"{name} に点がありません"      # 空の配列だと以降の検査が素通りしてしまう
    return points


def test_panel_is_disabled_without_results(window):
    view, panel = window.post_view, window.post_panel
    assert not view.has_results() and "解析結果がありません" in panel.state_label.text()
    assert not any(box.isEnabled() for box in (panel.surface_box, panel.slice_box, panel.stream_box, panel.iso_box, panel.scale_box))
    assert view.layer_actors() == {"surface": False, "slice": False, "streamlines": False, "iso": False}
    view.update(slice_on=True, surface_on=True)             # 結果が無ければ何も描かない (架空の表示をしない)
    assert view.layer_actors() == {"surface": False, "slice": False, "streamlines": False, "iso": False}


def test_surface_distribution_hides_cad(qtbot, window, box_step):
    part = load(qtbot, window, box_step, 1)[0]
    view, panel = with_results(window)
    assert view.has_results() and panel.surface_box.isEnabled()
    # 合成データは現在のメッシュ・条件で計算したものではないので、その旨が表示される
    assert window.scene.solver_state() == "stale" and "前の条件" in panel.state_label.text()
    cad = window.viewport._actors[part.id]["surface"]
    assert cad.GetVisibility()
    panel.surface_check.setChecked(True)
    assert view.layer_actors()["surface"] and not cad.GetVisibility()       # 重なる CAD は隠す
    mapper = actor(window, "post_surface").mapper
    assert mapper.array_name == "pressure" and tuple(mapper.scalar_range) == pytest.approx(view.color_range("pressure", True))
    # 表面は合成データの直方体そのもの
    points = actor_points(window, "post_surface")
    assert points.min(axis=0) == pytest.approx(BODY[0::2], abs=1e-6) and points.max(axis=0) == pytest.approx(BODY[1::2], abs=1e-6)
    # 物理量の切替、地面の表示
    panel.surface_field.setCurrentIndex(panel.surface_field.findData("yPlus"))
    panel.surface_field.activated.emit(panel.surface_field.currentIndex())
    assert actor(window, "post_surface").mapper.array_name == "yPlus"
    assert "post_surface_ground" not in window.viewport.plotter.actors
    panel.surface_ground.setChecked(True)
    assert "post_surface_ground" in window.viewport.plotter.actors
    panel.surface_check.setChecked(False)
    assert not view.layer_actors()["surface"] and cad.GetVisibility()
    assert "post_surface_ground" not in window.viewport.plotter.actors


def test_slice_fields_axes_and_position(window):
    view, panel = with_results(window)
    panel.slice_check.setChecked(True)
    assert view.layer_actors()["slice"]
    # 既定は Y 断面・流速。位置は領域中央
    s = view.settings
    assert (s.slice_axis, s.slice_field) == ("y", "Umag") and s.slice_position == pytest.approx(0.0)
    points = actor_points(window, "post_slice")
    assert np.allclose(points[:, 1], 0.0, atol=1e-6)
    first_actor = actor(window, "post_slice")

    # 座標の数値入力で断面が動く。アクターは作り直されない (中身だけ差し替え)
    panel.slice_position.setValue(700.0)
    assert s.slice_position == pytest.approx(0.7)
    assert np.allclose(actor_points(window, "post_slice")[:, 1], 0.7, atol=1e-6)
    assert actor(window, "post_slice") is first_actor
    # スライダーは領域の端から端まで。数値欄と連動する
    panel.slice_slider.setValue(250)
    assert s.slice_position == pytest.approx(-1.0) and panel.slice_position.metres() == pytest.approx(-1.0)
    assert np.allclose(actor_points(window, "post_slice")[:, 1], -1.0, atol=1e-6)
    panel.slice_position.setValue(99999.0)                     # 領域外は端で止まる
    assert s.slice_position == pytest.approx(2.0)

    # 向きを変えると、その軸のモデル (領域) 中央へ移る
    panel.slice_axis.setCurrentIndex(panel.slice_axis.findData("x"))
    panel.slice_axis.activated.emit(panel.slice_axis.currentIndex())
    assert s.slice_axis == "x" and s.slice_position == pytest.approx(2.0)      # 領域 X: −4 … 8
    assert np.allclose(actor_points(window, "post_slice")[:, 0], 2.0, atol=1e-6)
    # 物理量: 流速・圧力・Cp・渦度・乱流エネルギー・Q 値
    assert [panel.slice_field.itemData(i) for i in range(panel.slice_field.count())] == ["Umag", "pressure", "Cp", "vorticity_mag", "k", "Q"]
    for key in ("pressure", "Cp", "vorticity_mag", "k", "Q"):
        panel.slice_field.setCurrentIndex(panel.slice_field.findData(key))
        panel.slice_field.activated.emit(panel.slice_field.currentIndex())
        assert actor(window, "post_slice").mapper.array_name == key
    panel.slice_opacity.setValue(50)
    assert actor(window, "post_slice").prop.opacity == pytest.approx(0.5)
    assert not window.jobs.is_busy()                            # 表示の操作で解析ジョブは動かない


def test_streamlines_and_seed_controls(window):
    view, panel = with_results(window)
    seed = view.settings.seed
    # 既定の開始面は上流側で、必ず計算領域の内側にある
    domain = view.results.domain
    assert domain[0] < seed.x < 0.0 and seed.count == 96
    assert domain[2] < seed.y - seed.width / 2 and seed.y + seed.width / 2 < domain[3]
    assert domain[4] < seed.z - seed.height / 2 and seed.z + seed.height / 2 < domain[5]
    panel.stream_check.setChecked(True)
    assert view.layer_actors()["streamlines"] and "post_seed" in window.viewport.plotter.actors
    lines = dataset(window, "post_streamlines")
    assert lines.n_cells == 96 and "96 本" in panel.stream_info.text()
    assert actor(window, "post_streamlines").mapper.array_name == "Umag"

    # 開始位置を動かすと流線が引き直される
    panel.seed_spins["x"].setValue(-3000.0)
    panel.seed_spins["y"].setValue(500.0)
    panel.seed_spins["z"].setValue(1500.0)
    panel.seed_spins["width"].setValue(1000.0)
    panel.seed_spins["height"].setValue(1000.0)
    assert (seed.x, seed.y, seed.z, seed.width, seed.height) == pytest.approx((-3.0, 0.5, 1.5, 1.0, 1.0))
    points = actor_points(window, "post_streamlines")
    assert points[:, 1].min() == pytest.approx(0.0, abs=1e-6) and points[:, 1].max() == pytest.approx(1.0, abs=1e-6)
    assert points[:, 2].min() == pytest.approx(1.0, abs=1e-6) and points[:, 0].min() == pytest.approx(-3.0, abs=1e-6)
    # 本数と間隔
    panel.seed_ny.setValue(5)
    panel.seed_nz.setValue(3)
    assert seed.count == 15 and panel.seed_spacing.metres() == pytest.approx(0.25)
    assert dataset(window, "post_streamlines").n_cells == 15
    panel.seed_spacing.setValue(100.0)                         # 間隔 100 mm → 幅 1000 mm に 11 本
    assert seed.n_y == 11 and "33 本" in panel.stream_info.text()
    panel.seed_both.setChecked(True)
    assert actor_points(window, "post_streamlines")[:, 0].min() < -3.5
    # 線の太さ、単色
    panel.stream_width.setValue(4.0)
    assert actor(window, "post_streamlines").prop.line_width == pytest.approx(4.0)
    panel.stream_color.setCurrentIndex(panel.stream_color.findData(SOLID))
    panel.stream_color.activated.emit(panel.stream_color.currentIndex())
    assert not actor(window, "post_streamlines").mapper.GetScalarVisibility()
    # 車両前面へ戻す
    panel.seed_reset.click()
    assert view.settings.seed.count == 96 and view.settings.seed.x == pytest.approx(-3.76)


def test_isosurface_value_and_colour(window):
    view, panel = with_results(window)
    panel.iso_check.setChecked(True)
    # 既定は流速の等値面で、値は主流の半分 (後流の外形)。合成データの流速は最低でも U0 なので面は無い
    # → 存在しない面は描かない
    assert view.settings.iso_field == "Umag" and view.settings.iso_value == pytest.approx(0.5 * U0)
    assert not view.layer_actors()["iso"]
    panel.iso_value.setValue(U0 * 1.2)                          # 流速 1.2 U0 → 平面 z = 2
    assert view.layer_actors()["iso"]
    assert np.allclose(actor_points(window, "post_iso")[:, 2], 2.0, atol=1e-4)
    assert actor(window, "post_iso").mapper.array_name == "Umag" and actor(window, "post_iso").prop.opacity == pytest.approx(0.6)
    panel.iso_slider.setValue(panel.iso_slider.maximum() // 2)  # スライダーでも値を動かせる
    lo, hi = view.iso_range("Umag")
    assert view.settings.iso_value == pytest.approx((lo + hi) / 2)
    panel.iso_slider.setValue(int(panel.iso_slider.maximum() * 55 / 65))     # 流速 55 → z = 1
    assert np.allclose(actor_points(window, "post_iso")[:, 2], 1.0, atol=0.01)   # スライダーは 1/1000 刻み

    # Q 値 (渦の指標) へ切り替えると、値は 10 (U/H)² へ初期化される。合成データの Q は −1 … 2 なので面は無い
    panel.iso_field.setCurrentIndex(panel.iso_field.findData("Q"))
    panel.iso_field.activated.emit(panel.iso_field.currentIndex())
    assert view.settings.iso_value == pytest.approx(10 * (U0 / 3.0) ** 2) and not view.layer_actors()["iso"]
    assert view.iso_range("Q") == pytest.approx((0.0, 100 * (U0 / 3.0) ** 2))
    panel.iso_value.setValue(0.5)                               # Q = z − 1 = 0.5 → 平面 z = 1.5
    assert np.allclose(actor_points(window, "post_iso")[:, 2], 1.5, atol=1e-5)
    # 色を単色に、不透明度を変更
    panel.iso_color.setCurrentIndex(panel.iso_color.findData(SOLID))
    panel.iso_color.activated.emit(panel.iso_color.currentIndex())
    assert not actor(window, "post_iso").mapper.GetScalarVisibility()
    panel.iso_opacity.setValue(30)
    assert actor(window, "post_iso").prop.opacity == pytest.approx(0.3)


def test_colour_scale_and_scalar_bars(window):
    view, panel = with_results(window)
    plotter = window.viewport.plotter
    assert view.active_color_fields() == [] and not plotter.scalar_bars.keys()
    panel.slice_check.setChecked(True)
    panel.stream_check.setChecked(True)                         # どちらも流速 → カラーバーは 1 本を共有
    assert view.active_color_fields() == [("Umag", False)] and list(plotter.scalar_bars.keys()) == ["Velocity [m/s]"]
    panel.surface_check.setChecked(True)                        # 表面圧力が加わる
    assert set(plotter.scalar_bars.keys()) == {"Velocity [m/s]", "Pressure [Pa]"}

    # 流速の範囲を手動で指定 → 断面と流線の両方に反映
    panel.scale_field.setCurrentIndex(panel.scale_field.findData("Umag"))
    panel.scale_field.activated.emit(panel.scale_field.currentIndex())
    auto = view.color_range("Umag")
    assert panel.scale_auto.isChecked() and not panel.scale_min.isEnabled()
    assert panel.scale_max.value() == pytest.approx(auto[1], abs=1e-3)
    panel.scale_auto.setChecked(False)
    assert panel.scale_min.isEnabled() and view.color_range("Umag") == pytest.approx(auto)   # 手動化の直後は自動の範囲から
    panel.scale_min.setValue(10.0)
    panel.scale_max.setValue(40.0)
    for name in ("post_slice", "post_streamlines"):
        assert tuple(actor(window, name).mapper.scalar_range) == pytest.approx((10.0, 40.0))
    assert tuple(actor(window, "post_surface").mapper.scalar_range) == pytest.approx(view.color_range("pressure", True))
    # カラーマップの変更
    before = actor(window, "post_slice").mapper.lookup_table.cmap.name
    panel.scale_map.setCurrentIndex(panel.scale_map.findData("viridis"))
    panel.scale_map.activated.emit(panel.scale_map.currentIndex())
    assert before == "turbo" and actor(window, "post_slice").mapper.lookup_table.cmap.name == "viridis"
    # 自動へ戻す
    panel.scale_auto.setChecked(True)
    assert tuple(actor(window, "post_slice").mapper.scalar_range) == pytest.approx(auto)
    # レイヤーを消すとカラーバーも消える
    panel.slice_check.setChecked(False)
    panel.stream_check.setChecked(False)
    assert list(plotter.scalar_bars.keys()) == ["Pressure [Pa]"]


def test_layers_change_the_rendered_image_and_survive_theme_change(window, tmp_path):
    import pyvista as pv

    view, panel = with_results(window)
    window.viewport.set_view("iso")
    window.viewport.plotter.reset_camera(bounds=view.results.domain, render=False)
    window.save_screenshot(tmp_path / "empty.png")
    panel.slice_check.setChecked(True)
    panel.stream_check.setChecked(True)
    window.save_screenshot(tmp_path / "layers.png")
    empty = pv.read(tmp_path / "empty.png").point_data.active_scalars.astype(int)
    layers = pv.read(tmp_path / "layers.png").point_data.active_scalars.astype(int)
    changed = (np.abs(empty - layers).sum(axis=1) > 60).mean()
    assert changed > 0.03                                        # 断面と流線が実際に描画されている
    saturation = layers.max(axis=1) - layers.min(axis=1)
    assert (saturation > 100).mean() > 0.01                     # カラーマップの鮮やかな色が画面に出ている
    # 背景を白系にしてもレイヤーは維持され、カラーバーの文字色が背景に合う
    window.display_panel.background_combo.setCurrentIndex(1)
    assert view.layer_actors()["slice"] and view.layer_actors()["streamlines"]
    assert list(window.viewport.plotter.scalar_bars.keys()) == ["Velocity [m/s]"]


def test_post_settings_are_saved_with_the_project(qtbot, window, box_step, tmp_path):
    load(qtbot, window, box_step, 1)
    view, panel = with_results(window)
    panel.slice_check.setChecked(True)
    panel.slice_axis.setCurrentIndex(panel.slice_axis.findData("z"))
    panel.slice_axis.activated.emit(panel.slice_axis.currentIndex())
    panel.slice_position.setValue(1250.0)
    panel.stream_check.setChecked(True)
    panel.seed_ny.setValue(6)
    panel.seed_spins["x"].setValue(-2500.0)
    panel.scale_auto.setChecked(False)
    panel.scale_max.setValue(77.0)
    project_dir = tmp_path / "proj"
    assert window.save_project_to(project_dir)
    assert view.has_results() and view.layer_actors()["slice"]          # 保存先へ移っても表示は続く
    assert str(view.results.results_dir).startswith(str(project_dir))

    window.new_project(confirm=False)
    assert not view.has_results() and view.layer_actors() == {"surface": False, "slice": False, "streamlines": False, "iso": False}
    assert not view.settings.slice_on
    assert window.open_project(project_dir)
    qtbot.waitUntil(lambda: len(window.scene.parts) == 1 and not window._restoring, timeout=TIMEOUT)
    s = view.settings
    assert view.has_results() and s.slice_on and s.streamlines_on and not s.surface_on
    assert (s.slice_axis, s.slice_position) == ("z", pytest.approx(1.25))
    assert s.seed.n_y == 6 and s.seed.x == pytest.approx(-2.5)
    assert view.color_range("Umag")[1] == pytest.approx(77.0)
    assert view.layer_actors()["slice"] and view.layer_actors()["streamlines"]
    assert np.allclose(actor_points(window, "post_slice")[:, 2], 1.25, atol=1e-6)
    assert panel.slice_check.isChecked() and panel.seed_ny.value() == 6
    assert "*" not in window.windowTitle()


@needs_openfoam
def test_visualise_real_openfoam_results(qtbot, window, box_step):
    """実物の OpenFOAM の結果で、全レイヤーが描けることを確かめる。"""
    from tests.test_gui_stage3 import small_mesh, solve

    small_mesh(qtbot, window, box_step)
    window.scene.solver_settings.iterations = 60
    window.scene.solver_settings.n_procs = 2
    window.scene.notify_solver_settings_changed()
    solve(qtbot, window)
    view = window.post_view
    assert view.has_results() and view.results.available_fields() == ["Umag", "pressure", "Cp", "vorticity_mag", "k", "Q"]
    view.update(surface_on=True, slice_on=True, streamlines_on=True, iso_on=True)
    assert view.layer_actors() == {"surface": True, "slice": True, "streamlines": True, "iso": True}
    # 流線は物体を避けて流れる: 物体の内部 (沈み込み後の直方体) を通る点が無い
    points = actor_points(window, "post_streamlines")
    b = view.model_bounds()
    inside = ((points[:, 0] > b[0] + 0.1) & (points[:, 0] < b[1] - 0.1) & (np.abs(points[:, 1]) < 0.8)
              & (points[:, 2] > 0.1) & (points[:, 2] < b[5] - 0.1))
    assert len(points) > 500 and not inside.any()
    # 断面 (車両中心) の流速: よどみ・後流で 0 近く、上面で主流より速い
    cut = np.asarray(dataset(window, "post_slice").point_data["Umag"])
    speed = window.scene.tunnel.flow.speed_ms
    assert cut.min() < 0.3 * speed and speed < cut.max() < 3 * speed
    # 等値面 (主流の半分の流速) は物体の後ろ = 後流に広がる
    iso = actor_points(window, "post_iso")
    assert iso[:, 0].max() > b[1] + 0.5
