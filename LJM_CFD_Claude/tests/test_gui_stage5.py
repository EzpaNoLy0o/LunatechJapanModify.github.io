"""Stage 5 の GUI 結合テスト: 3D ビュー下部の操作バー (断面スライダー・流線開始位置スライダー・
粒子/スモークのタイムライン) と、アニメーションの書き出し。

解析的に値の分かる合成データを使うので、OpenFOAM が無い環境 (Windows) でも実行できる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QWidget

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("ディスプレイが無いため GUI テストをスキップ (xvfb-run で実行してください)", allow_module_level=True)

from ljmcfd.gui.flow_bar import BASE_LOOP_SECONDS, SLIDER_STEPS  # noqa: E402
from ljmcfd.gui.post_view import SMOKE_TRAIL_POINTS, STEADY_NOTICE, STYLE_PARTICLE, STYLE_SMOKE  # noqa: E402
from ljmcfd.solver.environment import native_openfoam_available  # noqa: E402
from tests.synthetic import U0  # noqa: E402
from tests.test_gui import TIMEOUT, load, window  # noqa: E402,F401  (window はフィクスチャ)
from tests.test_gui_stage4 import actor, actor_points, with_results  # noqa: E402

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")


def choose(combo, data) -> None:
    """コンボボックスを、利用者が選んだときと同じ経路 (activated) で切り替える。"""
    index = combo.findData(data)
    assert index >= 0, f"選択肢 {data!r} がありません"
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def count_streamline_calls(view, monkeypatch) -> list:
    """流線の積分 (重い処理) が何回呼ばれたかを数える。"""
    calls = []
    original = view.results.streamlines

    def counting(seed):
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(view.results, "streamlines", counting)
    return calls


def small_seed(view) -> None:
    """流線の開始面を小さくして、スライダーで動かしても領域からはみ出さないようにする。"""
    view.set_seed(x=-3.0, y=0.0, z=1.5, width=1.0, height=1.0, n_y=4, n_z=3)


def notice_text(window) -> str:
    annotation = actor(window, "post_notice")
    return " ".join(str(annotation.GetText(index) or "") for index in range(8))


def test_bar_is_hidden_without_results_and_sits_below_the_view(window):
    bar = window.flow_bar
    assert bar.isHidden()                                      # 結果が無い間は出さない (動かせるものが無い)
    view, _panel = with_results(window)
    assert bar.isVisible()
    # 3D ビューの直下にあり、3 本のスライダーが別々に並ぶ
    assert bar.geometry().top() >= window.viewport.geometry().bottom()
    sliders = (bar.slice_slider, bar.seed_slider, bar.time_slider)
    assert len({id(s) for s in sliders}) == 3 and all(s.maximum() == SLIDER_STEPS for s in sliders)
    tops = [s.mapTo(bar, s.rect().topLeft()).y() for s in sliders]
    assert tops == sorted(tops) and len(set(tops)) == 3
    # 粒子表示が無効の間は再生ボタンとタイムラインを押せない。注記も出さない
    assert not any(b.isEnabled() for b in (bar.play_button, bar.rewind_button, bar.pause_button, bar.stop_button, bar.first_button))
    assert not bar.time_slider.isEnabled() and not bar.export_button.isEnabled() and bar.notice.isHidden()
    assert bar.time_label.text() == "—"
    window.new_project(confirm=False)
    assert bar.isHidden() and not view.has_results()


def test_bar_does_not_squeeze_the_side_panels(qtbot, window):
    """バーの最小幅が広すぎると、3D ビューごと中央が広がって左右のパネルが潰れる (実際に起きた不具合)。"""
    bar = window.flow_bar
    qtbot.wait(50)
    before = window.main_splitter.sizes()
    view, _panel = with_results(window)
    qtbot.wait(50)
    assert window.main_splitter.sizes() == before
    bar.slice_check.setChecked(True)
    bar.seed_check.setChecked(True)
    bar.particle_check.setChecked(True)                         # 注記の行が増える
    qtbot.wait(50)
    assert window.main_splitter.sizes() == before and bar.notice.isVisible()
    assert bar.minimumSizeHint().width() <= before[1]           # 幅 1400 px のウィンドウでも中央に収まる
    assert bar.width() == window.viewport.width()
    # どの部品も切れずに表示されている (失敗時は、潰れた部品と寸法を表示する)
    squeezed = []
    for widget in bar.findChildren(QWidget):
        if widget.isVisible() and widget.width() < widget.minimumSizeHint().width():
            squeezed.append(f"{type(widget).__name__}({widget.objectName() or widget.text()!r}): "
                            f"幅 {widget.width()} < 最小 {widget.minimumSizeHint().width()}")
    assert not squeezed, f"バーの部品が潰れています (バー幅 {bar.width()}、ビュー幅 {window.viewport.width()}): {squeezed}"
    for widget in (bar.slice_position, bar.seed_position, bar.time_label, bar.export_button, bar.stop_button, bar.loop_check):
        right = widget.mapTo(bar, widget.rect().bottomRight()).x()
        assert widget.isVisible() and right <= bar.width(), widget.objectName() or widget.text()
    assert window.viewport.height() > 350                       # 3D ビューの高さも十分に残る


def test_section_slider_moves_the_slice_in_real_time(window, monkeypatch):
    view, panel = with_results(window)
    bar = window.flow_bar
    s = view.settings
    bar.slice_check.setChecked(True)
    assert s.slice_on and view.layer_actors()["slice"] and panel.slice_check.isChecked()
    first_actor = actor(window, "post_slice")
    slices = []
    original = view.results.slice
    monkeypatch.setattr(view.results, "slice", lambda axis, position: slices.append((axis, position)) or original(axis, position))

    # スライダーは領域の端から端まで。動かすたびに断面が移る (アクターは作り直さず中身だけ差し替え)
    for step, expected in ((0, -2.0), (250, -1.0), (500, 0.0), (875, 1.5), (SLIDER_STEPS, 2.0)):
        bar.slice_slider.setValue(step)
        assert s.slice_position == pytest.approx(expected)
        if 0 < step < SLIDER_STEPS:
            assert np.allclose(actor_points(window, "post_slice")[:, 1], expected, atol=1e-6)
    assert actor(window, "post_slice") is first_actor
    assert [axis for axis, _ in slices] == ["y"] * 5              # 1 操作につき断面の切り直しは 1 回だけ
    # 座標は表示単位 (mm) で出る。右パネルの数値欄・スライダーとも連動する
    bar.slice_slider.setValue(875)
    assert bar.slice_position.value() == pytest.approx(1500.0) and bar.slice_position.suffix().strip() == "mm"
    assert panel.slice_position.metres() == pytest.approx(1.5) and panel.slice_slider.value() == 875
    window.scene.set_display_unit("m")
    assert bar.slice_position.value() == pytest.approx(1.5) and bar.slice_position.suffix().strip() == "m"
    window.scene.set_display_unit("mm")
    # 数値入力でも動き、スライダーが追従する
    bar.slice_position.setValue(-500.0)
    assert s.slice_position == pytest.approx(-0.5) and bar.slice_slider.value() == 375
    assert np.allclose(actor_points(window, "post_slice")[:, 1], -0.5, atol=1e-6)
    panel.slice_slider.setValue(600)                            # 右パネル側を動かすと下部バーも追従
    assert bar.slice_slider.value() == 600 and bar.slice_position.metres() == pytest.approx(0.4)

    # 向き: X (前後) / Y (左右) / Z (高さ)
    assert [bar.slice_axis.itemData(i) for i in range(bar.slice_axis.count())] == ["x", "y", "z"]
    choose(bar.slice_axis, "x")
    assert s.slice_axis == "x" and panel.slice_axis.currentData() == "x"
    bar.slice_slider.setValue(250)                              # 領域 X: −4 … 8
    assert s.slice_position == pytest.approx(-1.0)
    assert np.allclose(actor_points(window, "post_slice")[:, 0], -1.0, atol=1e-6)
    choose(bar.slice_axis, "z")
    bar.slice_slider.setValue(500)                              # 領域 Z: 0 … 3
    assert np.allclose(actor_points(window, "post_slice")[:, 2], 1.5, atol=1e-6)
    # 物理量: 流速・圧力・Cp・渦度・乱流エネルギー・Q 値
    assert [bar.slice_field.itemData(i) for i in range(bar.slice_field.count())] == ["Umag", "pressure", "Cp", "vorticity_mag", "k", "Q"]
    for key in ("pressure", "Cp", "vorticity_mag", "k", "Q"):
        choose(bar.slice_field, key)
        assert actor(window, "post_slice").mapper.array_name == key and panel.slice_field.currentData() == key
    bar.slice_check.setChecked(False)
    assert not view.layer_actors()["slice"]
    assert not window.jobs.is_busy()                            # スライダー操作で解析ジョブは動かない


def test_seed_slider_moves_the_streamlines_and_coalesces_drags(qtbot, window, monkeypatch):
    view, panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    seed = view.settings.seed
    bar.seed_check.setChecked(True)
    assert view.layer_actors()["streamlines"] and panel.stream_check.isChecked()
    assert bar.seed_axis.currentData() == "y" and bar.seed_slider.value() == 500 and bar.seed_position.metres() == pytest.approx(0.0)
    calls = count_streamline_calls(view, monkeypatch)

    # ドラッグ中に届く途中の値はまとめ、最後の位置だけ流線を引き直す
    for step in (520, 560, 610, 680, 750):
        bar.seed_slider.setValue(step)
    assert len(calls) == 0                                      # まだ引き直していない (操作を待たせない)
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=5000)
    qtbot.wait(100)
    assert len(calls) == 1 and seed.y == pytest.approx(1.0)     # 領域 Y: −2 … 2 の 75 %
    points = actor_points(window, "post_streamlines")
    assert points[:, 1].min() == pytest.approx(0.5, abs=1e-6) and points[:, 1].max() == pytest.approx(1.5, abs=1e-6)
    assert bar.seed_position.value() == pytest.approx(1000.0) and panel.seed_spins["y"].value() == pytest.approx(1000.0)

    # 数値入力
    bar.seed_position.setValue(-800.0)
    assert seed.y == pytest.approx(-0.8) and bar.seed_slider.value() == 300 and len(calls) == 2
    # 動かす軸の切替: X (前後) / Z (高さ)。切り替えただけでは流線は変わらない
    choose(bar.seed_axis, "z")
    assert view.settings.seed_slider_axis == "z" and len(calls) == 2
    assert bar.seed_position.metres() == pytest.approx(1.5) and bar.seed_slider.value() == 500
    bar.seed_slider.setValue(600)                               # 領域 Z: 0 … 3
    qtbot.waitUntil(lambda: len(calls) == 3, timeout=5000)
    assert seed.z == pytest.approx(1.8) and seed.y == pytest.approx(-0.8)
    points = actor_points(window, "post_streamlines")
    assert points[:, 2].min() == pytest.approx(1.3, abs=1e-6) and points[:, 2].max() == pytest.approx(2.3, abs=1e-6)
    choose(bar.seed_axis, "x")
    bar.seed_slider.setValue(250)                               # 領域 X: −4 … 8
    qtbot.waitUntil(lambda: len(calls) == 4, timeout=5000)
    assert seed.x == pytest.approx(-1.0) and actor_points(window, "post_streamlines")[:, 0].min() == pytest.approx(-1.0, abs=1e-6)
    # 右パネルで開始面を動かすと、下部バーの表示も追従する
    panel.seed_spins["x"].setValue(-2500.0)
    assert bar.seed_position.metres() == pytest.approx(-2.5) and bar.seed_slider.value() == 125
    assert not window.jobs.is_busy()


def test_particles_follow_the_timeline_and_rewind_exactly(window, monkeypatch):
    view, _panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    seed = view.settings.seed
    bar.loop_check.setChecked(False)
    choose(bar.particle_style, STYLE_PARTICLE)
    assert not view.layer_actors()["particles"] and "post_notice" not in window.viewport.plotter.actors
    bar.particle_check.setChecked(True)
    assert view.layer_actors()["particles"] and view.settings.particles_on

    # 定常解析の粒子表示であることを、バーと 3D ビュー内の両方に明記する
    assert bar.notice.isVisible() and "定常 RANS" in bar.notice.text() and "非定常解析" in bar.notice.text()
    assert STEADY_NOTICE in notice_text(window) and "not a transient" in STEADY_NOTICE
    assert view.animation().is_steady

    # タイムラインの長さ: 開始面 (x = −3) から出口 (x = 8) まで流れる時間。流速は U0 (1 + 0.1 z)
    duration = view.particle_duration()
    assert 11.0 / (U0 * 1.2) < duration < 11.0 / (U0 * 1.1)
    assert bar.time_slider.isEnabled() and bar.time_label.text() == f"t = 0.0000 / {duration:.4f} s"
    calls = count_streamline_calls(view, monkeypatch)
    source = view.animation()

    # 時刻 0: 放出されたばかりの粒子が開始面にいる
    start = view.particle_positions()
    assert len(start) == seed.count and np.allclose(start[:, 0], -3.0, atol=1e-9)
    # 右へ動かすと下流 (+X) へ進む。進んだ距離は 流速 × 時間 に一致する
    bar.time_slider.setValue(300)
    t1 = view.settings.particles_time
    assert t1 == pytest.approx(0.3 * duration)
    at_300 = view.particle_positions()
    lead = at_300[np.argsort(at_300[:, 0])[-1]]
    assert lead[0] == pytest.approx(-3.0 + U0 * (1 + 0.1 * lead[2]) * t1, rel=1e-3)
    assert len(at_300) > len(start)                             # その間に後続の粒子も放出されている
    assert np.allclose(at_300[:, 1].min(), -0.5, atol=1e-6) and np.allclose(at_300[:, 1].max(), 0.5, atol=1e-6)
    bar.time_slider.setValue(800)
    at_800 = view.particle_positions()
    assert at_800[:, 0].max() > at_300[:, 0].max() + 3.0
    assert bar.time_label.text() == f"t = {0.8 * duration:.4f} / {duration:.4f} s"
    # 左へ戻すと、前の位置へ正確に戻る
    bar.time_slider.setValue(300)
    assert np.array_equal(view.particle_positions(), at_300)
    bar.time_slider.setValue(0)
    assert np.array_equal(view.particle_positions(), start)

    # タイムラインを動かしても流線の積分はやり直さない (事前計算したキャッシュを使う)。解析ジョブも動かない
    assert calls == [] and view.animation() is source
    assert not window.jobs.is_busy() and window._solver_job is None


def test_section_slider_and_timeline_are_independent(window):
    view, _panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    bar.slice_check.setChecked(True)
    bar.particle_check.setChecked(True)
    bar.time_slider.setValue(400)
    time = view.settings.particles_time
    particles = view.particle_positions()
    cloud_actor = actor(window, "post_particles")

    # 断面を動かしても、時刻と粒子はそのまま
    bar.slice_slider.setValue(700)
    assert view.settings.slice_position == pytest.approx(0.8)
    assert view.settings.particles_time == time and bar.time_slider.value() == 400
    assert np.array_equal(view.particle_positions(), particles) and actor(window, "post_particles") is cloud_actor
    # 時刻を動かしても、断面はそのまま
    slice_points = actor_points(window, "post_slice").copy()
    bar.time_slider.setValue(650)
    assert view.settings.slice_position == pytest.approx(0.8) and bar.slice_slider.value() == 700
    assert np.array_equal(actor_points(window, "post_slice"), slice_points)
    assert not np.array_equal(view.particle_positions(), particles)
    # 流線を消しても粒子は残る (粒子は流線の表示に依存しない)
    bar.seed_check.setChecked(True)
    bar.seed_check.setChecked(False)
    assert view.layer_actors()["particles"] and not view.layer_actors()["streamlines"]


def test_playback_controls(qtbot, window):
    view, _panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    s = view.settings
    bar.particle_check.setChecked(True)
    duration = view.particle_duration()
    assert s.particles_loop and bar.loop_check.isChecked() and s.particles_speed == 1.0
    assert bar.play_button.isEnabled() and bar.rewind_button.isEnabled() and not bar.pause_button.isEnabled()

    # 再生: 実際のタイマーで時刻が進み、スライダーと時刻表示が追従する
    bar.play_button.click()
    assert bar.is_playing() and not bar.play_button.isEnabled() and bar.pause_button.isEnabled() and bar.rewind_button.isEnabled()
    qtbot.waitUntil(lambda: s.particles_time > 0.02 * duration, timeout=10000)
    assert bar.time_slider.value() > 0 and not bar.time_label.text().startswith("t = 0.0000")
    # 一時停止: その時刻で止まる
    bar.pause_button.click()
    paused_at = s.particles_time
    assert not bar.is_playing() and bar.play_button.isEnabled() and not bar.pause_button.isEnabled()
    qtbot.wait(150)
    assert s.particles_time == paused_at
    # 巻き戻し: 時刻が戻っていく
    bar.rewind_button.click()
    assert bar.is_playing() and not bar.rewind_button.isEnabled() and bar.play_button.isEnabled()
    qtbot.waitUntil(lambda: s.particles_time < paused_at, timeout=10000)
    # 停止: 止めて先頭へ
    bar.stop_button.click()
    assert not bar.is_playing() and s.particles_time == 0.0 and bar.time_slider.value() == 0

    # 再生速度: × 1 でタイムライン全体を BASE_LOOP_SECONDS 秒。倍率に比例して速くなる
    assert [bar.speed_combo.itemData(i) for i in range(bar.speed_combo.count())] == [0.25, 0.5, 1.0, 2.0, 4.0]
    bar.play()
    bar._timer.stop()                                           # 以降は時間を手で送る (実時間に依存させない)
    bar.advance(1.0)
    assert s.particles_time == pytest.approx(duration / BASE_LOOP_SECONDS)
    choose(bar.speed_combo, 2.0)
    bar.advance(1.0)
    assert s.particles_time == pytest.approx(3 * duration / BASE_LOOP_SECONDS)
    choose(bar.speed_combo, 0.25)
    bar.advance(2.0)
    assert s.particles_time == pytest.approx(3.5 * duration / BASE_LOOP_SECONDS)
    choose(bar.speed_combo, 1.0)
    # 先頭へ: 再生状態は変えずに時刻だけ戻す
    bar.first_button.click()
    assert s.particles_time == 0.0

    # ループ再生: 終端を越えると先頭から続く。逆再生では先頭を越えると終端から続く
    bar.play()
    view.set_particle_time(0.95 * duration)
    bar.advance(0.1 * BASE_LOOP_SECONDS)
    assert bar.is_playing() and s.particles_time == pytest.approx(0.05 * duration)
    bar.rewind()
    bar.advance(0.1 * BASE_LOOP_SECONDS)
    assert bar.is_playing() and s.particles_time == pytest.approx(0.95 * duration)
    # ループなし: 終端で止まる。終端で再生を押すと先頭から始まる
    bar.loop_check.setChecked(False)
    bar.play()
    bar.advance(0.2 * BASE_LOOP_SECONDS)
    assert not bar.is_playing() and s.particles_time == pytest.approx(duration)
    bar.play()
    assert bar.is_playing() and s.particles_time == 0.0
    bar.rewind()                                                # 先頭で巻き戻しを押すと終端から戻る
    assert bar.is_playing() and s.particles_time == pytest.approx(duration)
    bar.advance(0.5 * BASE_LOOP_SECONDS)
    assert bar.is_playing() and s.particles_time == pytest.approx(0.5 * duration)
    bar.advance(0.75 * BASE_LOOP_SECONDS)
    assert not bar.is_playing() and s.particles_time == 0.0     # 先頭で止まる
    # 再生中にスモークを無効にすると再生も止まる
    bar.play()
    bar.particle_check.setChecked(False)
    assert not bar.is_playing() and not view.layer_actors()["particles"] and not bar.play_button.isEnabled()
    # 再生中に結果が無くなっても止まる
    bar.particle_check.setChecked(True)
    bar.play()
    assert bar.is_playing()
    window.new_project(confirm=False)
    assert not bar.is_playing() and bar.isHidden()


def test_smoke_style_density_and_size(window, monkeypatch):
    view, _panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    n_lines = view.settings.seed.count
    bar.particle_check.setChecked(True)
    # 既定はスモーク: 粒子 1 個を、後ろほど薄くなる半透明の筋 (折れ線) で描く
    assert view.settings.particles_style == STYLE_SMOKE and bar.particle_style.currentData() == STYLE_SMOKE
    assert bar.density_spin.value() == 8 and view._cloud.n_points == n_lines * 8 * SMOKE_TRAIL_POINTS
    assert view._cloud.n_lines == n_lines * 8 and view._cloud.n_verts == 0
    rgba = np.asarray(view._cloud.point_data["rgba"])
    assert rgba.dtype == np.uint8 and 0 < rgba[:, 3].max() < 255 and len(np.unique(rgba[:, 3])) > 3   # 尾ほど薄い
    assert actor(window, "post_particles").prop.line_width == pytest.approx(view.settings.particles_size)
    calls = count_streamline_calls(view, monkeypatch)

    # 粒子: 1 個 1 点で不透明
    choose(bar.particle_style, STYLE_PARTICLE)
    assert view._cloud.n_points == n_lines * 8 and view._cloud.n_lines == 0 and view._cloud.n_verts == n_lines * 8
    rgba = np.asarray(view._cloud.point_data["rgba"])
    assert set(np.unique(rgba[:, 3])) <= {0, 255} and rgba[:, 3].max() == 255
    assert actor(window, "post_particles").prop.point_size == pytest.approx(view.settings.particles_size)
    # 色は流速: 高い位置の粒子ほど速い (合成データ) → カラーマップ上で別の色になる
    points = np.asarray(view._cloud.points)
    visible = rgba[:, 3] > 0
    low, high = points[:, 2] < 1.2, points[:, 2] > 1.8
    assert (visible & low).any() and (visible & high).any()
    assert not np.array_equal(rgba[visible & low][0, :3], rgba[visible & high][0, :3])
    # 粒子密度
    bar.density_spin.setValue(20)
    assert view.settings.particles_density == 20.0 and view._cloud.n_points == n_lines * 20
    bar.density_spin.setValue(3)
    assert view._cloud.n_points == n_lines * 3 and len(view.particle_positions()) == n_lines * 3   # ループ再生では常に全粒子が流れている
    # 大きさ
    bar.size_spin.setValue(12.0)
    assert actor(window, "post_particles").prop.point_size == pytest.approx(12.0)
    # ここまで、流線は表示用にも粒子用にも積分し直していない (密度・形式は標本化のやり直しだけ)
    assert calls == []
    # 開始面を動かせば粒子も付いていく
    view.set_seed(y=1.0)
    assert len(calls) == 1
    moved = view.particle_positions()
    assert moved[:, 1].min() == pytest.approx(0.5, abs=1e-6) and moved[:, 1].max() == pytest.approx(1.5, abs=1e-6)
    # 背景テーマを変えても粒子と注記は残る (色だけの変更なので流線は積分し直さない)
    window.display_panel.background_combo.setCurrentIndex(1)
    assert view.layer_actors()["particles"] and STEADY_NOTICE in notice_text(window) and len(calls) == 1


def test_export_animation_gif_and_png_sequence(window, tmp_path):
    from PIL import Image

    view, _panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    # 粒子表示が無効なら書き出さない (空の動画を作らない)
    assert window.act_export_animation.text().startswith("アニメーションを書き出し")
    assert window.export_animation(tmp_path / "none.gif", n_frames=4) is False and not (tmp_path / "none.gif").exists()

    bar.particle_check.setChecked(True)
    window.viewport.plotter.reset_camera(bounds=view.results.domain, render=False)
    bar.time_slider.setValue(420)
    time = view.settings.particles_time
    bar.play()
    assert window.export_animation(tmp_path / "flow.gif", n_frames=6, scale=0.4)
    assert not bar.is_playing() and view.settings.particles_time == time      # 書き出し後は元の時刻へ戻す
    with Image.open(tmp_path / "flow.gif") as gif:
        assert gif.n_frames == 6 and gif.info.get("loop") == 0
        assert gif.info["duration"] == pytest.approx(1000 * BASE_LOOP_SECONDS / 6, abs=11)
        full = window.viewport.capture_frame().shape
        # 撮影のたびに 3D ビューの高さが 1〜2 px 変わることがあるので、許容差で比べる
        assert abs(gif.size[0] - full[1] * 0.4) <= 2 and abs(gif.size[1] - full[0] * 0.4) <= 2
        frames = []
        for index in range(gif.n_frames):
            gif.seek(index)
            frames.append(np.asarray(gif.convert("RGB")).astype(int))
    assert np.abs(frames[0] - frames[3]).sum() > 0                             # コマごとに粒子が動いている
    # PNG 連番
    assert window.export_animation(tmp_path / "seq.png", n_frames=3, scale=0.25)
    assert sorted(p.name for p in tmp_path.glob("seq_*.png")) == ["seq_0001.png", "seq_0002.png", "seq_0003.png"]
    assert not window.jobs.is_busy()


def test_flow_bar_settings_are_saved_with_the_project(qtbot, window, box_step, tmp_path):
    load(qtbot, window, box_step, 1)
    view, _panel = with_results(window)
    bar = window.flow_bar
    small_seed(view)
    bar.slice_check.setChecked(True)
    choose(bar.slice_axis, "x")
    bar.slice_slider.setValue(250)
    choose(bar.seed_axis, "z")
    bar.particle_check.setChecked(True)
    choose(bar.particle_style, STYLE_PARTICLE)
    bar.density_spin.setValue(5)
    bar.size_spin.setValue(9.0)
    choose(bar.speed_combo, 2.0)
    bar.loop_check.setChecked(False)
    bar.time_slider.setValue(350)
    time = view.settings.particles_time
    positions = view.particle_positions()
    assert time > 0 and len(positions) > 0
    project_dir = tmp_path / "proj"
    assert window.save_project_to(project_dir)

    window.new_project(confirm=False)
    assert bar.isHidden() and not view.settings.particles_on
    assert window.open_project(project_dir)
    qtbot.waitUntil(lambda: len(window.scene.parts) == 1 and not window._restoring, timeout=TIMEOUT)
    s = view.settings
    assert bar.isVisible() and view.has_results()
    assert s.slice_on and (s.slice_axis, s.slice_position) == ("x", pytest.approx(-1.0))
    assert s.particles_on and (s.particles_style, s.particles_density, s.particles_size) == (STYLE_PARTICLE, 5.0, 9.0)
    assert (s.particles_speed, s.particles_loop, s.seed_slider_axis) == (2.0, False, "z")
    assert s.particles_time == pytest.approx(time)
    assert bar.slice_slider.value() == 250 and bar.slice_axis.currentData() == "x" and bar.seed_axis.currentData() == "z"
    assert bar.particle_check.isChecked() and bar.particle_style.currentData() == STYLE_PARTICLE
    assert bar.density_spin.value() == 5 and bar.speed_combo.currentData() == 2.0 and not bar.loop_check.isChecked()
    assert bar.time_slider.value() == 350 and not bar.is_playing()               # 再生状態は保存しない (開いた直後は止まっている)
    assert view.layer_actors()["particles"] and np.allclose(view.particle_positions(), positions)
    assert "*" not in window.windowTitle()


@needs_openfoam
def test_particles_on_real_openfoam_results(qtbot, window, box_step):
    """実物の OpenFOAM の結果で、粒子が物体を避けて下流へ流れることを確かめる。"""
    from tests.test_gui_stage3 import small_mesh, solve

    small_mesh(qtbot, window, box_step)
    window.scene.solver_settings.iterations = 60
    window.scene.solver_settings.n_procs = 2
    window.scene.notify_solver_settings_changed()
    solve(qtbot, window)
    view, bar = window.post_view, window.flow_bar
    assert view.has_results() and bar.isVisible()
    bar.loop_check.setChecked(False)
    choose(bar.particle_style, STYLE_PARTICLE)
    bar.particle_check.setChecked(True)
    duration = view.particle_duration()
    speed = window.scene.tunnel.flow.speed_ms
    domain = view.results.domain
    # タイムラインの長さは、開始面から出口まで主流で流れる時間の程度
    transit = (domain[1] - view.settings.seed.x) / speed
    assert 0.5 * transit < duration < 5 * transit
    assert STEADY_NOTICE in notice_text(window) and bar.notice.isVisible()

    b = view.model_bounds()
    means = []
    for step in (0, 200, 400, 600, 800):
        bar.time_slider.setValue(step)
        points = view.particle_positions()
        assert len(points) > 0
        lead = points[points[:, 0] >= np.percentile(points[:, 0], 90)]
        means.append(float(lead[:, 0].mean()))
        # 粒子は物体の内部を通らない
        inside = ((points[:, 0] > b[0] + 0.1) & (points[:, 0] < b[1] - 0.1) & (np.abs(points[:, 1]) < 0.8)
                  & (points[:, 2] > 0.1) & (points[:, 2] < b[5] - 0.1))
        assert not inside.any()
        assert (points[:, 0] >= domain[0] - 1e-6).all() and (points[:, 0] <= domain[1] + 1e-6).all()
    assert means == sorted(means) and means[-1] > means[0] + 0.5 * (domain[1] - view.settings.seed.x)   # 先頭の粒子が下流へ進む
    bar.time_slider.setValue(200)
    again = view.particle_positions()
    bar.time_slider.setValue(900)
    bar.time_slider.setValue(200)
    assert np.array_equal(view.particle_positions(), again)
    assert window._solver_job is None and not window.jobs.is_busy()             # スライダー操作で OpenFOAM は再実行されない
