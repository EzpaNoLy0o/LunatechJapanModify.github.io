"""ボリューム煙 (Realistic Smoke Visualization) の GUI 結合テスト。

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

from ljmcfd.gui.post_view import SMOKE_NOTICE, STYLE_SMOKE, STYLE_VOLUME, STYLE_VOLUME_HQ  # noqa: E402
from ljmcfd.gui.smoke_view import EMITTER_NAME, VOLUME_NAME  # noqa: E402
from ljmcfd.smoke.engine import TIMELINE_FRAMES  # noqa: E402
from ljmcfd.smoke.settings import COLOR_VELOCITY  # noqa: E402
from ljmcfd.solver.environment import native_openfoam_available  # noqa: E402
from tests.test_gui import TIMEOUT, load, window  # noqa: E402,F401  (window はフィクスチャ)
from tests.test_gui_stage4 import with_results  # noqa: E402
from tests.test_gui_stage5 import choose, notice_text  # noqa: E402

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")


def smoke_on(qtbot, window, style=STYLE_VOLUME):
    """下部のバーでボリューム煙を選び、計算が終わるまで待つ。"""
    bar, view = window.flow_bar, window.post_view
    choose(bar.particle_style, style)
    bar.particle_check.setChecked(True)
    qtbot.waitUntil(lambda: view.particle_duration() > 0 and not view.smoke_layer.building, timeout=TIMEOUT)
    return view.smoke_layer


def shown(layer) -> np.ndarray:
    """いま 3D ビューに渡している煙のデータ (描画用の配列の複製)。"""
    image = layer.renderer._current
    entry = next(e for e in layer.renderer._images.values() if e[0] is image)
    return entry[1].copy()


@pytest.fixture()
def car(qtbot, window, box_step):
    load(qtbot, window, box_step, 1)
    with_results(window)
    return window


def test_volume_smoke_is_built_in_the_background_and_shown(qtbot, car):
    window, bar, view = car, car.flow_bar, car.post_view
    layer = view.smoke_layer
    choose(bar.particle_style, STYLE_VOLUME)
    assert bar.particle_style.currentData() == STYLE_VOLUME
    bar.particle_check.setChecked(True)
    # 計算は別スレッド: 押した直後はまだ準備中で、GUI は止まらない
    assert layer.active and (layer.building or layer.state is not None)
    qtbot.waitUntil(lambda: view.particle_duration() > 0 and not layer.building, timeout=TIMEOUT)
    state = layer.state
    assert state.smoke.tracks.n_tracks > 0 and state.duration > 0
    # 放出源は車両の寸法から自動配置 (Front Flow)。ノズルの位置を 3D ビューに示す
    smoke = view.settings.smoke
    assert smoke.emitters_initialized and smoke.emitters[0].name == "Front Flow"
    actors = window.viewport.plotter.actors
    assert VOLUME_NAME in actors and EMITTER_NAME in actors and view.layer_actors()["particles"]
    assert type(actors[VOLUME_NAME]).__name__ == "vtkVolume"
    # 定常解析の煙であることを、3D ビュー (動画にも残る) とバーに明記する
    assert SMOKE_NOTICE in notice_text(window) and "煙です" in bar.notice.text() and bar.notice.isVisible()
    assert "格子" in bar.smoke_status.text() and "流跡線" in window.smoke_render_panel.status.text()
    # 煙は濃淡のある連続した濃度場 (点の集まりではない)
    bar.time_slider.setValue(600)
    density = shown(layer)
    assert density.ndim == 3 and density.max() > 0 and (density > 0).sum() > 1000
    assert len(np.unique(np.round(density[density > 0], 4))) > 100
    assert not window.jobs.is_busy() and window._solver_job is None


def test_timeline_scrub_is_deterministic_and_previews_while_dragging(qtbot, car):
    view, bar = car.post_view, car.flow_bar
    layer = smoke_on(qtbot, car)
    duration = view.particle_duration()
    bar.time_slider.setValue(300)
    early = shown(layer)
    bar.time_slider.setValue(800)
    late = shown(layer)
    assert late.sum() > early.sum() > 0                              # 放出を続けているので、時間とともに煙が増える
    bar.time_slider.setValue(300)
    assert np.array_equal(shown(layer), early)                       # 戻すと同じ煙
    assert view.settings.particles_time == pytest.approx(0.3 * duration)
    # 操作中: 本来のコマがキャッシュに無ければ、待たずに低解像度の煙を出す。操作が終われば元の解像度へ戻す
    smoke = layer.state.smoke
    smoke.clear_cache()
    view.set_particle_time(0.55 * duration, interactive=True)
    assert layer.showing_preview and shown(layer).size == layer.state.preview.grid.n_voxels
    view.settle_particle_time()
    assert not layer.showing_preview and shown(layer).size == smoke.grid.n_voxels
    # コマ送り: タイムラインの 1/240 ずつ
    bar.step(1)
    assert view.settings.particles_time == pytest.approx(duration * (round(0.55 * (TIMELINE_FRAMES - 1)) + 1) / (TIMELINE_FRAMES - 1))
    bar.step(-1)
    bar.step(-1)
    assert view.settings.particles_time == pytest.approx(duration * (round(0.55 * (TIMELINE_FRAMES - 1)) - 1) / (TIMELINE_FRAMES - 1))
    # 再生・一時停止・ループ (実際のタイマー)
    bar.go_to_start()
    bar.play()
    qtbot.waitUntil(lambda: view.settings.particles_time > 0.05 * duration, timeout=20000)
    bar.pause()
    assert not bar.is_playing() and not layer.showing_preview
    bar.play()
    bar._timer.stop()
    view.set_particle_time(0.98 * duration)
    bar.advance(0.2 * 8.0)
    assert view.settings.particles_time < 0.5 * duration              # ループで先頭側へ回る
    assert not car.jobs.is_busy() and car._solver_job is None        # どの操作でも解析は動かない


def test_appearance_changes_only_touch_rendering(qtbot, car, monkeypatch):
    view = car.post_view
    layer = smoke_on(qtbot, car)
    layer._stop_prefetch()                     # 先のコマを用意する先読み (バックグラウンド) を止めてから数える
    builds, frames = [], []
    original_build, original_frame = layer.engine.build, layer.state.smoke.frame
    monkeypatch.setattr(layer.engine, "build", lambda *a, **k: builds.append(1) or original_build(*a, **k))
    monkeypatch.setattr(layer.state.smoke, "frame", lambda *a, **k: frames.append(1) or original_frame(*a, **k))
    volume = car.viewport.plotter.actors[VOLUME_NAME]

    def opacity_at(rho):
        return volume.GetProperty().GetScalarOpacity().GetValue(rho)

    def color_at(rho):
        return volume.GetProperty().GetRGBTransferFunction().GetColor(rho)

    panel = car.smoke_render_panel
    base = opacity_at(0.5)
    assert color_at(0.5) == pytest.approx((0.97, 0.97, 0.97))       # 既定は白煙
    panel.density_slider.setValue(300)                               # 濃さ × 3
    assert opacity_at(0.5) > base and view.settings.smoke.density == 3.0
    kappa_rho_h = -np.log(1 - base)
    assert opacity_at(0.5) == pytest.approx(1 - np.exp(-3 * kappa_rho_h), rel=0.02)   # α = 1 − exp(−κρΔs)
    panel.opacity_slider.setValue(40)
    assert opacity_at(0.5) == pytest.approx(0.4 * (1 - np.exp(-3 * kappa_rho_h)), rel=0.02)
    choose(panel.color_combo, "dark_gray")
    assert color_at(0.5) == pytest.approx((0.42, 0.43, 0.45))
    panel.lighting.setChecked(True)
    assert volume.GetProperty().GetShade()
    assert car.viewport.plotter.actors[VOLUME_NAME] is volume
    assert builds == [] and frames == []                             # 煙の計算 (Level 2・3) はやり直していない
    # 流速で色付け: 色 (流速) と不透明度 (濃度) を別の成分として渡す
    choose(panel.color_combo, COLOR_VELOCITY)
    data = shown(layer)
    assert data.shape[-1] == 2 and not volume.GetProperty().GetIndependentComponents()
    assert builds == []
    speed, density = data[..., 0], data[..., 1]
    assert speed[density > 0.2 * density.max()].min() > 40.0           # 合成データの流速は 50 m/s 以上
    # 描画方式の切り替え (GPU / CPU) でも計算はやり直さない
    choose(panel.backend, "cpu")
    assert layer.renderer.backend == "cpu" and builds == []


def test_emitter_panel_moves_presets_and_disables_sources(qtbot, car):
    view, panel = car.post_view, car.smoke_emitter_panel
    layer = smoke_on(qtbot, car)
    first = layer.state.smoke
    assert panel.emitter_list.count() == 1 and panel.emitter_list.item(0).checkState().value == 2
    # 位置の変更 → 流跡線だけ作り直す (OpenFOAM は使わない)
    panel.emitter_list.setCurrentRow(0)
    z = view.settings.smoke.emitters[0].z
    panel.length_spins["z"].setValue((z + 0.2) * 1000.0)
    qtbot.waitUntil(lambda: layer.state is not None and layer.state.smoke is not first and not layer.building, timeout=TIMEOUT)
    assert view.settings.smoke.emitters[0].z == pytest.approx(z + 0.2)
    assert layer.state.smoke.grid == first.grid                      # 格子 (流体マスク) は使い回し
    # プリセットの追加: 放出源が 2 つになり、両方の煙が出る
    choose(panel.preset_combo, "underfloor")
    panel.preset_add.click()
    qtbot.waitUntil(lambda: not layer.building and layer.state is not None and len(layer.state.plan.emitters) == 2, timeout=TIMEOUT)
    assert panel.emitter_list.count() == 2 and set(layer.state.smoke.tracks.emitter.tolist()) == {0, 1}
    # すべて無効にすると煙は消え、理由を表示する
    for row in range(2):
        panel.emitter_list.item(row).setCheckState(panel.emitter_list.item(row).checkState().__class__.Unchecked)
    qtbot.waitUntil(lambda: view.particle_duration() == 0, timeout=TIMEOUT)
    assert VOLUME_NAME not in car.viewport.plotter.actors and "有効な放出源がありません" in layer.status_text()
    panel.preset_replace.click()                                     # プリセットで置き直す
    qtbot.waitUntil(lambda: view.particle_duration() > 0 and not layer.building, timeout=TIMEOUT)
    assert len(view.settings.smoke.emitters) == 1 and view.settings.smoke.emitters[0].name == "Underfloor Flow"


def test_switching_modes_keeps_existing_displays_working(qtbot, car):
    view, bar = car.post_view, car.flow_bar
    smoke_on(qtbot, car)

    def actors():
        return car.viewport.plotter.actors           # 呼ぶたびに作り直される一覧なので、毎回取り直す

    for style in (STYLE_SMOKE, "ribbon", "tube", "particle"):
        choose(bar.particle_style, style)
        assert VOLUME_NAME not in actors() and "post_particles" in actors() and not view.smoke_layer.active, style
        assert len(view.particle_positions()) > 0
    # 高品質: High 以上の格子と陰影
    layer = smoke_on(qtbot, car, STYLE_VOLUME_HQ)
    assert layer.high_quality and layer.state.plan.budget >= 192 * 128 * 128
    assert actors()[VOLUME_NAME].GetProperty().GetShade()
    bar.particle_check.setChecked(False)
    assert VOLUME_NAME not in actors() and EMITTER_NAME not in actors() and not layer.active
    # 断面・流線と同時に表示できる
    smoke_on(qtbot, car)
    bar.slice_check.setChecked(True)
    bar.seed_check.setChecked(True)
    assert view.layer_actors() == {"surface": False, "slice": True, "streamlines": True, "iso": False, "particles": True}


def test_export_and_project_round_trip(qtbot, car, tmp_path):
    from PIL import Image

    view, bar = car.post_view, car.flow_bar
    layer = smoke_on(qtbot, car)
    view.update_smoke(density=1.8, lifetime=0.15, quality="low")
    qtbot.waitUntil(lambda: not layer.building and layer.state.plan.lifetime == 0.15, timeout=TIMEOUT)
    bar.time_slider.setValue(600)
    car.viewport.plotter.reset_camera(bounds=view.results.domain, render=False)
    assert car.export_animation(tmp_path / "smoke.gif", n_frames=5, scale=0.3)
    with Image.open(tmp_path / "smoke.gif") as gif:
        frames = []
        for index in range(gif.n_frames):
            gif.seek(index)
            frames.append(np.asarray(gif.convert("RGB")).astype(int))
    assert len(frames) == 5 and np.abs(frames[1] - frames[3]).sum() > 0
    assert not layer.showing_preview                                 # 書き出しは本来の解像度で
    time = view.settings.particles_time
    project = tmp_path / "proj"
    assert car.save_project_to(project)
    data = (project / "project.ljmcfd.json").read_text(encoding="utf-8")
    assert '"smoke"' in data and "Front Flow" in data and len(data) < 200_000     # 濃度場は JSON に入れない
    car.new_project(confirm=False)
    assert not layer.active and VOLUME_NAME not in car.viewport.plotter.actors
    assert car.open_project(project)
    qtbot.waitUntil(lambda: len(car.scene.parts) == 1 and not car._restoring, timeout=TIMEOUT)
    qtbot.waitUntil(lambda: view.particle_duration() > 0 and not layer.building, timeout=TIMEOUT)
    smoke = view.settings.smoke
    assert view.settings.particles_style == STYLE_VOLUME and smoke.density == 1.8 and smoke.lifetime == 0.15 and smoke.quality == "low"
    assert smoke.emitters[0].name == "Front Flow" and view.settings.particles_time == pytest.approx(time)
    assert VOLUME_NAME in car.viewport.plotter.actors


def test_v060_project_without_smoke_settings_opens(qtbot, window, box_step, tmp_path):
    """煙の設定が無い v0.6.0 のプロジェクトも開ける (既定値になる)。"""
    import json

    load(qtbot, window, box_step, 1)
    with_results(window)
    project = tmp_path / "old"
    assert window.save_project_to(project)
    path = project / "project.ljmcfd.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["post"].pop("smoke")
    data["post"]["particles_style"] = "smoke"
    path.write_text(json.dumps(data), encoding="utf-8")
    window.new_project(confirm=False)
    assert window.open_project(project)
    qtbot.waitUntil(lambda: len(window.scene.parts) == 1 and not window._restoring, timeout=TIMEOUT)
    smoke = window.post_view.settings.smoke
    assert smoke.emitters == [] and not smoke.emitters_initialized and smoke.quality == "medium"
    assert window.post_view.settings.particles_style == STYLE_SMOKE


@needs_openfoam
def test_volume_smoke_on_real_openfoam_results_stays_out_of_the_body(qtbot, window, box_step):
    """実物の OpenFOAM の結果で、煙が車体 (直方体) の中へ入らず、車体を避けて下流へ流れることを確かめる。"""
    from tests.test_gui_stage3 import small_mesh, solve

    small_mesh(qtbot, window, box_step)
    window.scene.solver_settings.iterations = 60
    window.scene.solver_settings.n_procs = 2
    window.scene.notify_solver_settings_changed()
    solve(qtbot, window)
    view = window.post_view
    layer = smoke_on(qtbot, window)
    smoke = layer.state.smoke
    x, y, z = smoke.grid.centers()
    b = view.model_bounds()
    inside = ((x[None, None, :] > b[0] + 0.05) & (x[None, None, :] < b[1] - 0.05) & (y[None, :, None] > b[2] + 0.05)
              & (y[None, :, None] < b[3] - 0.05) & (z[:, None, None] > max(b[4], 0.0) + 0.05) & (z[:, None, None] < b[5] - 0.05))
    assert inside.any() and smoke.mask[inside].max() == 0.0          # 流体マスク: 車体の中は固体
    for fraction in (0.3, 0.6, 1.0):
        density = smoke.frame(fraction * layer.state.duration)
        assert density.max() > 0 and density[inside].max() == 0.0, fraction
    # 煙は車体の後方まで届く (流れに乗って下流へ)
    late = smoke.frame(layer.state.duration)
    behind = late[:, :, x > b[1] + 0.3].sum()
    assert behind > 0.05 * late.sum()
    # 未収束の結果には警告を出す (60 反復の短い計算)
    assert "NOT converged" in "\n".join(view._notice_lines()) or view.scene.solver_summary["convergence"]["level"] == "converged"


def test_scalar_transport_never_blocks_the_gui(qtbot, car, tmp_path):
    """受動スカラー方式: コマは先頭から順にしか求まらないので、画面のスレッドでは計算せず、計算が届いたら表示する。"""
    import time as clock

    view, bar = car.post_view, car.flow_bar
    choose(car.smoke_render_panel.transport, "scalar")
    layer = smoke_on(qtbot, car)
    smoke = layer.state.smoke
    assert getattr(smoke, "sequential", False) and "受動スカラー方式" in layer.status_text()
    smoke.clear_cache()
    layer._stop_prefetch()
    started = clock.perf_counter()
    bar.time_slider.setValue(900)                                    # 未計算のコマへ飛ぶ
    assert clock.perf_counter() - started < 0.5                      # 待たされない
    assert layer._wanted is not None and "先頭から順に計算中" in layer.status_text()
    qtbot.waitUntil(lambda: layer._wanted is None, timeout=TIMEOUT)  # 計算が届いたら表示する
    index = smoke.frame_index(view.settings.particles_time, TIMELINE_FRAMES)
    assert np.allclose(shown(layer), smoke.peek_frame(index, TIMELINE_FRAMES).astype(np.float32))
    # 書き出しは、その時刻の煙を必ず計算して使う
    assert car.export_animation(tmp_path / "scalar.gif", n_frames=3, scale=0.25)
    assert (tmp_path / "scalar.gif").stat().st_size > 0
