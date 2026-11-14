"""Stage 2 の GUI 結合テスト: 風洞設定 → メッシュ生成 → 表示 → 保存・再読み込み。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("ディスプレイが無いため GUI テストをスキップ (xvfb-run で実行してください)", allow_module_level=True)

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from ljmcfd.core.project import load_project  # noqa: E402
from ljmcfd.mesh import polymesh as pm  # noqa: E402
from ljmcfd.tunnel import spec as ts  # noqa: E402
from tests.test_gui import TIMEOUT, load, part_named, window  # noqa: E402,F401  (window はフィクスチャ)


def wait_idle(qtbot, window):
    qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=TIMEOUT)


def make_mesh(qtbot, window):
    assert window.generate_mesh()
    qtbot.waitUntil(lambda: window.scene.mesh_report is not None and window._mesh_job is None, timeout=TIMEOUT)
    wait_idle(qtbot, window)
    return window.scene.mesh_report


def coarse(window):
    combo = window.mesh_panel.preset_combo
    combo.setCurrentIndex(combo.findData("coarse"))
    combo.activated.emit(combo.currentIndex())


def test_tunnel_defaults_and_flow_inputs(qtbot, window, box_step):
    panel = window.tunnel_panel
    # パーツが無いうちは生成できない
    assert not panel.generate_button.isEnabled() and "CAD を読み込む" in panel.summary.text()
    load(qtbot, window, box_step, 1)
    tunnel = window.scene.tunnel
    # 既定: 200 km/h、車両モード、地面あり、余白はモデル寸法から自動 (前 3L / 後 6L / 左右 3W / 上 4H)
    assert panel.speed_kmh.value() == pytest.approx(200.0) and panel.speed_ms.value() == pytest.approx(55.556, abs=1e-3)
    assert tunnel.mode == ts.MODE_VEHICLE and tunnel.has_ground and not panel.ground_check.isEnabled()
    assert (tunnel.margin_front, tunnel.margin_rear, tunnel.margin_side, tunnel.margin_top) == pytest.approx((12.0, 24.0, 5.4, 4.8))
    assert panel.margin_spins["margin_front"].value() == pytest.approx(12000.0)        # 表示単位 mm
    assert not panel.margin_spins["margin_bottom"].isEnabled()
    assert "未生成" in panel.summary.text() and "前面投影面積: 未計算" in panel.summary.text()

    # km/h と m/s は連動する
    panel.speed_ms.setValue(50.0)
    assert tunnel.flow.speed_ms == pytest.approx(50.0) and panel.speed_kmh.value() == pytest.approx(180.0)
    panel.speed_kmh.setValue(200.0)
    assert tunnel.flow.speed_ms == pytest.approx(200 / 3.6)
    # 密度: 自動 (温度・気圧から) ⇔ 手動
    assert panel.density_auto.isChecked() and not panel.density.isEnabled()
    panel.temperature.setValue(35.0)
    assert tunnel.flow.density == pytest.approx(101325 / (287.05 * 308.15)) and panel.density.value() == pytest.approx(tunnel.flow.density, abs=1e-4)
    panel.density_auto.setChecked(False)
    panel.density.setValue(1.10)
    assert tunnel.flow.density == pytest.approx(1.10) and "ρ = 1.1000" in panel.flow_info.text()
    panel.density_auto.setChecked(True)
    assert tunnel.flow.density_override is None
    panel.pressure.setValue(1000.0)
    assert tunnel.flow.pressure_pa == pytest.approx(100000.0)
    panel.turbulence.setValue(2.5)
    assert tunnel.flow.turbulence_intensity_pct == pytest.approx(2.5)
    assert "レイノルズ数" in panel.flow_info.text() and "マッハ数" in panel.flow_info.text()

    # 相対風向: 風洞タブの Yaw は車両姿勢そのもの (モデルタブと同じ値)
    panel.attitude["yaw"].setValue(5.0)
    assert window.scene.assembly.pose.yaw == pytest.approx(5.0)
    assert window.model_panel.vehicle_pose.angle_spins["yaw"].value() == pytest.approx(5.0)
    panel.attitude["yaw"].setValue(0.0)

    # 地面との距離を適用するとモデルが持ち上がり、沈み込みは不要になる
    panel.clearance.setValue(80.0)
    panel.clearance_apply.click()
    assert window.scene.analysis_bounds()[4] == pytest.approx(0.08)
    assert tunnel.model_sink(window.scene.analysis_bounds()) == 0.0
    panel.clearance.setValue(0.0)
    panel.clearance_apply.click()
    assert tunnel.model_sink(window.scene.analysis_bounds()) == pytest.approx(0.010)
    assert "沈めます" in panel.issues.text()


def test_generate_tunnel_shows_domain_and_frontal_area(qtbot, window, box_step, monkeypatch):
    load(qtbot, window, box_step, 1)
    window.scene.center_xy()
    viewport, panel = window.viewport, window.tunnel_panel
    assert "tunnel_box" not in viewport.plotter.actors
    panel.generate_button.click()
    wait_idle(qtbot, window)
    tunnel = window.scene.tunnel
    assert tunnel.generated and "tunnel_box" in viewport.plotter.actors and "tunnel_inlet" in viewport.plotter.actors
    assert viewport.tunnel_bounds() == pytest.approx((-14.0, 26.0, -6.3, 6.3, 0.0, 6.0))
    # 前面投影面積 = 1.8 × (1.2 − 沈み込み 0.01)。地面より下は含めない
    assert tunnel.frontal_area == pytest.approx(1.8 * 1.19, rel=3e-3)
    assert "閉塞率: 2.8" in panel.summary.text() and "生成済み" in panel.summary.text()
    assert panel.reference_auto.isChecked() and panel.reference_area.value() == pytest.approx(tunnel.frontal_area, abs=1e-4)
    # 基準面積はユーザーが上書きできる
    panel.reference_auto.setChecked(False)
    panel.reference_area.setValue(2.0)
    assert tunnel.reference_area == pytest.approx(2.0) and tunnel.effective_reference_area() == pytest.approx(2.0)
    panel.reference_auto.setChecked(True)
    assert tunnel.reference_area is None
    # 表示の切替と、モデル移動への追従
    window.mesh_display.tunnel_check.setChecked(False)
    assert "tunnel_box" not in viewport.plotter.actors
    window.mesh_display.tunnel_check.setChecked(True)
    window.scene.set_vehicle_pose(tx=1.0)
    assert viewport.tunnel_bounds()[0] == pytest.approx(-11.0)
    window.mesh_display.fit_tunnel.click()

    # 単体パーツモード: 地面なしになり、下方余白が有効になる
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData(ts.MODE_COMPONENT))
    assert tunnel.mode == ts.MODE_COMPONENT and not tunnel.has_ground
    assert panel.ground_check.isEnabled() and panel.margin_spins["margin_bottom"].isEnabled()
    assert "tunnel_ground" not in viewport.plotter.actors
    # 内部流れモードは選択できない (未実装)
    internal = panel.mode_combo.model().item(panel.mode_combo.findData(ts.MODE_INTERNAL))
    assert not internal.isEnabled()

    # 不正な余白では生成を拒否し、理由を表示する
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]) or QMessageBox.Ok)
    panel.margin_spins["margin_front"].setValue(0.0)
    assert not window.generate_tunnel()
    assert warnings and "前方" in warnings[0]


def test_mesh_generation_display_and_quality_report(qtbot, window, assembly_step):
    load(qtbot, window, assembly_step, 2)
    window.scene.center_xy()
    viewport, panel, display = window.viewport, window.mesh_panel, window.mesh_display
    body, wing = part_named(window, "Body"), part_named(window, "RearWing")
    coarse(window)
    assert window.scene.mesh_settings.preset == "coarse" and panel.size_spins["surface_size"].value() == pytest.approx(80.0)
    # 境界層なしでは、精度の制限が明示される
    assert "境界層メッシュなし" in panel.bl_info.text() and "精度は限定的" in panel.bl_info.text()
    assert "未生成" in panel.state_label.text() and "未生成" in window.mesh_quality.toPlainText()
    assert not display.surface_check.isEnabled()

    report = make_mesh(qtbot, window)
    assert window.scene.tunnel.generated                      # 風洞が未生成なら自動で生成される
    assert window.scene.mesh_state() == "current" and "現在の設定と一致" in panel.state_label.text()
    assert report["quality"]["n_cells"] > 1000 and report["quality"]["n_negative_volumes"] == 0
    assert "境界層プリズム: 未生成" in panel.summary.text() and "精度は限定的" in panel.summary.text()
    quality_text = window.mesh_quality.toPlainText()
    assert "非直交性" in quality_text and "RearWing" in quality_text and "inlet" in quality_text
    assert window.bottom_tabs.currentWidget() is window.mesh_quality
    assert "メッシュ生成完了" in window.log_panel.toPlainText()
    mesh_dir = window.mesh_dir()
    assert "unsaved" in str(mesh_dir) and (mesh_dir / "polyMesh" / "points").is_file()

    # メッシュ表示: 表面メッシュを出すと CAD 側は隠れる
    assert display.surface_check.isEnabled()
    display.surface_check.setChecked(True)
    assert "mesh_surface" in viewport.plotter.actors
    assert not viewport._actors[body.id]["surface"].GetVisibility()
    display.ground_check.setChecked(True)
    assert "mesh_ground" in viewport.plotter.actors
    display.surface_check.setChecked(False)
    assert viewport._actors[body.id]["surface"].GetVisibility() and "mesh_surface" not in viewport.plotter.actors

    # 断面スライダー: 位置を動かすと横切るセルが変わる。メッシュ生成ジョブは走らない
    display.section_check.setChecked(True)
    assert "mesh_section" in viewport.plotter.actors
    assert display.position.metres() == pytest.approx(0.0, abs=1e-6)       # 初期位置はモデル中心 (Y)
    cells_center = viewport.mesh_section_cells("y", 0.0)
    display.slider.setValue(display.slider.maximum() // 4)
    y_quarter = display.position.metres()
    assert y_quarter == pytest.approx(-6.3 + 12.6 / 4, abs=0.02)
    cells_quarter = viewport.mesh_section_cells("y", y_quarter)
    assert len(cells_center) > len(cells_quarter) > 0                       # モデル付近の方が細かい
    assert not window.jobs.is_busy()
    # 断面の全セルが実際に平面をまたいでいる
    volume = viewport._volume()
    section = volume.extract_cells(cells_quarter)
    assert section.bounds[2] <= y_quarter <= section.bounds[3]
    # 座標を数値で直接指定 / 軸の切替
    display.position.setValue(900.0)
    assert display.position.metres() == pytest.approx(0.9) and viewport._section == (True, "y", pytest.approx(0.9))
    display.axis_combo.setCurrentIndex(0)
    assert viewport._section[1] == "x" and viewport.mesh_axis_range("x") == pytest.approx((-14.0, 26.0))

    # パーツ別細分化の倍率を変える → メッシュは「再生成が必要」になる
    spin = panel.factor_spin(wing.id)
    spin.setValue(0.5)
    assert window.scene.mesh_settings.part_size_factors == {wing.id: 0.5}
    assert window.scene.mesh_state() == "stale" and "再生成が必要" in panel.state_label.text()
    spin.setValue(1.0)
    assert window.scene.mesh_settings.part_size_factors == {} and window.scene.mesh_state() == "current"
    # 姿勢・風速以外の風洞寸法を変えても同様
    window.scene.set_vehicle_pose(yaw=2.0)
    assert window.scene.mesh_state() == "stale"


def test_boundary_layer_status_is_visible(qtbot, window, box_step, monkeypatch):
    load(qtbot, window, box_step, 1)
    panel = window.mesh_panel
    coarse(window)
    panel.bl_check.setChecked(True)
    panel.bl_layers.setValue(3)
    panel.bl_first.setValue(4.0)
    assert window.scene.mesh_settings.boundary_layer.first_height == pytest.approx(0.004)
    assert "合計厚さ" in panel.bl_info.text() and "y+" in panel.bl_info.text()

    # 接地したままでは生成できず、理由が表示される (黙って境界層なしにしない)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2]) or QMessageBox.Ok)
    assert window.generate_mesh()
    qtbot.waitUntil(lambda: window._mesh_job is None and not window.jobs.is_busy(), timeout=TIMEOUT)
    assert errors and "接して" in errors[0] and window.scene.mesh_report is None
    assert panel.generate_button.isEnabled() and not panel.stop_button.isEnabled()

    # 地面から離すと生成でき、状態が GUI に表示される
    window.tunnel_panel.clearance.setValue(300.0)
    window.tunnel_panel.clearance_apply.click()
    report = make_mesh(qtbot, window)
    assert report["boundary_layer"]["generated"] and report["quality"]["cell_counts"]["prism"] > 0
    assert "境界層プリズム: 生成済み (3 層" in panel.summary.text()
    assert "生成済み: 3 層" in window.mesh_quality.toPlainText()


def test_mesh_is_kept_across_project_save_and_reopen(qtbot, window, box_step, tmp_path):
    load(qtbot, window, box_step, 1)
    coarse(window)
    window.tunnel_panel.speed_kmh.setValue(160.0)
    report = make_mesh(qtbot, window)
    scratch = window.mesh_dir()
    window.mesh_display.section_check.setChecked(True)
    window.mesh_display.position.setValue(250.0)

    project_dir = tmp_path / "proj"
    assert window.save_project_to(project_dir)
    # 未保存時の作業フォルダからプロジェクトフォルダへ引き継がれ、JSON には埋め込まれない
    assert (project_dir / "mesh" / "polyMesh" / "faces").is_file() and not scratch.exists()
    assert window.mesh_dir() == project_dir / "mesh"
    assert (project_dir / "project.ljmcfd.json").stat().st_size < 30_000
    saved, _ = load_project(project_dir)
    assert saved.sections["tunnel"]["flow"]["speed_ms"] == pytest.approx(160 / 3.6)
    assert saved.sections["mesh"]["settings"]["preset"] == "coarse"
    assert window.viewport._section[0] and "mesh_section" in window.viewport.plotter.actors   # 表示は保存後も有効

    window.new_project(confirm=False)
    assert window.scene.mesh_report is None and not window.scene.tunnel.generated
    assert "mesh_section" not in window.viewport.plotter.actors and "tunnel_box" not in window.viewport.plotter.actors
    assert window.open_project(project_dir)
    qtbot.waitUntil(lambda: len(window.scene.parts) == 1 and not window._restoring, timeout=TIMEOUT)
    assert window.scene.tunnel.generated and window.scene.tunnel.flow.speed_kmh == pytest.approx(160.0)
    assert window.tunnel_panel.speed_kmh.value() == pytest.approx(160.0)
    assert window.scene.mesh_settings.preset == "coarse"
    again = window.scene.mesh_report
    assert again is not None and again["input_hash"] == report["input_hash"]
    assert window.scene.mesh_state() == "current"              # 再読み込みしただけで再生成は不要
    assert window.mesh_display.section_check.isChecked() and window.mesh_display.position.metres() == pytest.approx(0.25)
    assert "tunnel_box" in window.viewport.plotter.actors and "mesh_section" in window.viewport.plotter.actors
    assert pm.read_polymesh(project_dir / "mesh" / "polyMesh").n_cells == report["quality"]["n_cells"]
    assert "*" not in window.windowTitle()


def test_stop_cancels_running_mesh_job(qtbot, window, box_step):
    load(qtbot, window, box_step, 1)
    assert window.generate_mesh()
    assert window.mesh_panel.stop_button.isEnabled() and not window.mesh_panel.generate_button.isEnabled()
    assert not window.generate_mesh()                     # 実行中の二重起動はしない
    window.mesh_panel.stop_button.click()
    qtbot.waitUntil(lambda: window._mesh_job is None and not window.jobs.is_busy(), timeout=TIMEOUT)
    assert window.scene.mesh_report is None and not (window.mesh_dir() / "mesh_report.json").exists()
    assert window.mesh_panel.generate_button.isEnabled() and not window.mesh_panel.stop_button.isEnabled()
    assert "停止" in window.log_panel.toPlainText()
