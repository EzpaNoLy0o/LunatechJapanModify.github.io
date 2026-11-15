"""Stage 3 の GUI 結合テスト: GUI から実物の OpenFOAM を動かし、結果表示・停止・再読み込みを確かめる。"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("ディスプレイが無いため GUI テストをスキップ (xvfb-run で実行してください)", allow_module_level=True)

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from ljmcfd.solver.environment import native_openfoam_available  # noqa: E402
from ljmcfd.solver.run_job import STATUS_COMPLETED, STATUS_FAILED, STATUS_STOPPED  # noqa: E402
from tests.test_gui import TIMEOUT, load, window  # noqa: E402,F401  (window はフィクスチャ)
from tests.test_gui_stage2 import make_mesh, wait_idle  # noqa: E402

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")


def small_mesh(qtbot, window, box_step):
    """解析が数秒で終わる粗いメッシュを用意する。"""
    load(qtbot, window, box_step, 1)
    window.scene.center_xy()
    settings = window.scene.mesh_settings
    settings.preset = "custom"
    settings.surface_size, settings.min_size, settings.far_size = 0.22, 0.11, 3.0
    settings.refine_distance, settings.wake_length = 8.0, 6.0
    window.scene.notify_mesh_settings_changed()
    return make_mesh(qtbot, window)


def solve(qtbot, window):
    assert window.run_solver()
    qtbot.waitUntil(lambda: window._solver_job is None and not window.jobs.is_busy(), timeout=TIMEOUT * 3)
    return window.scene.solver_summary


def test_solver_panel_settings_and_prerequisites(qtbot, window, box_step, monkeypatch):
    panel, results = window.solver_panel, window.results_panel
    settings = window.scene.solver_settings
    # 既定: 500 反復。メッシュが無ければ開始できない
    assert settings.iterations == 500 and panel.iteration_combo.currentData() == 500
    assert not panel.start_button.isEnabled() and "メッシュが未生成" in panel.state_label.text()
    assert all(label.text() == "未計算" for label in results.result_labels.values())
    assert "未計算" in results.state_label.text()
    # 反復数: 選択肢 / 任意の値
    assert [panel.iteration_combo.itemData(i) for i in range(panel.iteration_combo.count())] == [200, 300, 500, 700, 1000, -1]
    panel.iteration_combo.setCurrentIndex(panel.iteration_combo.findData(1000))
    panel.iteration_combo.activated.emit(panel.iteration_combo.currentIndex())
    assert settings.iterations == 1000 and not panel.iteration_spin.isEnabled()
    panel.iteration_combo.setCurrentIndex(panel.iteration_combo.findData(-1))
    panel.iteration_combo.activated.emit(panel.iteration_combo.currentIndex())
    assert panel.iteration_spin.isEnabled()
    panel.iteration_spin.setValue(1234)
    assert settings.iterations == 1234 and panel.iteration_combo.currentData() == -1
    panel.relax_p.setValue(0.2)
    panel.relax_u.setValue(0.5)
    panel.procs.setValue(2)
    panel.correctors.setValue(2)
    panel.continue_check.setChecked(True)
    panel.tolerance_edit.setText("2e-6")
    panel.tolerance_edit.editingFinished.emit()
    assert (settings.relax_p, settings.relax_u, settings.n_procs, settings.non_ortho_correctors) == (0.2, 0.5, 2, 2)
    assert settings.continue_run and settings.residual_tolerance == pytest.approx(2e-6)
    panel.tolerance_edit.setText("abc")                       # 不正な入力は元へ戻す
    panel.tolerance_edit.editingFinished.emit()
    assert settings.residual_tolerance == pytest.approx(2e-6) and panel.tolerance_edit.text() == "2e-06"

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]) or QMessageBox.Ok)
    assert not window.run_solver() and "メッシュが未生成" in warnings[-1]

    # メッシュ生成後は開始可能。メッシュが古くなると開始できない
    small_mesh(qtbot, window, box_step)
    assert panel.start_button.isEnabled()
    window.scene.set_vehicle_pose(yaw=3.0)
    assert not panel.start_button.isEnabled() and "再生成" in panel.state_label.text()
    assert not window.run_solver() and "再生成" in warnings[-1]


@needs_openfoam
def test_environment_check_from_gui(qtbot, window):
    panel = window.solver_panel
    panel.check_button.click()
    wait_idle(qtbot, window)
    assert "OpenFOAM 14 を検出" in panel.env_status.text() and "物理" in panel.env_status.text()
    # 存在しない場所を指定すると、理由つきで表示される
    panel.bashrc_edit.setText("/nonexistent/etc/bashrc")
    panel.bashrc_edit.editingFinished.emit()
    panel.check_button.click()
    wait_idle(qtbot, window)
    assert "OpenFOAM が見つかりません" in panel.env_status.text()
    assert window.settings.value("openfoam/bashrc") == "/nonexistent/etc/bashrc"     # 設定として保存される


@needs_openfoam
def test_run_analysis_from_gui_and_reopen(qtbot, window, box_step, tmp_path):
    small_mesh(qtbot, window, box_step)
    window.tunnel_panel.generate_button.click()               # 前面投影面積 (基準面積) を計算
    wait_idle(qtbot, window)
    area = window.scene.tunnel.frontal_area
    assert area == pytest.approx(1.8 * 1.19, rel=3e-3)
    scene, panel, results, plot = window.scene, window.solver_panel, window.results_panel, window.convergence_plot
    scene.solver_settings.iterations = 150
    scene.solver_settings.n_procs = 2
    scene.notify_solver_settings_changed()

    assert window.run_solver()
    assert panel.stop_button.isEnabled() and not panel.start_button.isEnabled()
    assert not window.run_solver()                            # 実行中の二重起動はしない
    # 実行中に残差・荷重のグラフが増えていく (終了を待たずに確認)
    qtbot.waitUntil(lambda: plot.point_counts().get("drag", 0) >= 10, timeout=TIMEOUT)
    assert window._solver_job is not None and scene.solver_summary is None
    assert "反復" in panel.run_status.text()
    qtbot.waitUntil(lambda: window._solver_job is None and not window.jobs.is_busy(), timeout=TIMEOUT * 3)

    summary = scene.solver_summary
    assert summary["status"] == STATUS_COMPLETED and summary["iterations"]["done"] == 150
    counts = plot.point_counts()
    assert counts["drag"] == 150 and counts["lift"] == 150 and counts["residual_p"] == 150
    assert window.bottom_tabs.currentWidget() is plot and window.right_tabs.currentWidget() is window.results_scroll
    # 結果パネル: 実際の値が入り、Downforce = −Lift、係数は現在の基準面積で換算
    f = summary["forces"]
    text = {key: label.text() for key, label in results.result_labels.items()}
    assert float(text["drag"]) == pytest.approx(f["drag"], abs=0.01) and float(text["drag"]) > 0
    assert float(text["downforce"]) == pytest.approx(-float(text["lift"]), abs=0.02)
    assert float(text["cd"]) == pytest.approx(f["drag"] / (f["dynamic_pressure"] * area), abs=1e-4)
    assert float(text["drag_pressure"]) + float(text["drag_viscous"]) == pytest.approx(float(text["drag"]), abs=0.03)
    assert float(text["speed_max"]) > scene.tunnel.flow.speed_ms and float(text["pressure_min"]) < 0 < float(text["pressure_max"])
    assert "計算完了" in results.status_label.text() and "収束判定:" in results.convergence_label.text()
    assert "抗力の安定" in results.checks_label.text() and "質量収支" in results.checks_label.text()
    assert "境界層メッシュがありません" in results.notes_label.text() and "y+ [box_mm]" in results.notes_label.text()
    assert "直近" in results.basis_label.text() and "Downforce = −Lift" in results.basis_label.text()
    assert "現在の設定の結果" in results.state_label.text() and scene.solver_state() == "current"
    assert "解析結果: Drag" in window.log_panel.toPlainText()

    # 基準面積を変えると係数だけが追従する (力は変わらない)
    window.tunnel_panel.reference_auto.setChecked(False)
    window.tunnel_panel.reference_area.setValue(1.0)
    assert float(results.result_labels["cd"].text()) == pytest.approx(f["drag"] / f["dynamic_pressure"], abs=1e-4)
    assert results.result_labels["drag"].text() == text["drag"] and scene.solver_state() == "current"
    window.tunnel_panel.reference_auto.setChecked(True)
    # 風速を変えると、表示中の結果は「以前の条件のもの」と明示される
    window.tunnel_panel.speed_kmh.setValue(150.0)
    assert scene.solver_state() == "stale" and "以前の条件" in results.state_label.text()
    window.tunnel_panel.speed_kmh.setValue(200.0)
    assert scene.solver_state() == "current"

    # 保存 → 新規 → 再読み込み: 結果とグラフが復元され、再計算は走らない
    project_dir = tmp_path / "proj"
    assert window.save_project_to(project_dir)
    assert (project_dir / "results" / "summary.json").is_file() and (project_dir / "case" / "system" / "controlDict").is_file()
    assert (project_dir / "project.ljmcfd.json").stat().st_size < 40_000
    window.new_project(confirm=False)
    assert scene.solver_summary is None and plot.point_counts() == {}
    assert all(label.text() == "未計算" for label in results.result_labels.values())
    assert window.open_project(project_dir)
    qtbot.waitUntil(lambda: len(scene.parts) == 1 and not window._restoring, timeout=TIMEOUT)
    assert scene.solver_summary["input_hash"] == summary["input_hash"] and scene.solver_state() == "current"
    assert scene.solver_settings.iterations == 150
    assert results.result_labels["drag"].text() == text["drag"]
    assert plot.point_counts()["drag"] == 150
    assert not window.jobs.is_busy()

    # 続きから計算: 反復が追加され、履歴がつながる
    panel.continue_check.setChecked(True)
    scene.solver_settings.iterations = 40
    scene.notify_solver_settings_changed()
    again = solve(qtbot, window)
    assert again["iterations"] == {"start": 150, "requested_end": 190, "done": 190}
    assert plot.point_counts()["drag"] == 190


@needs_openfoam
def test_stop_button_keeps_partial_result(qtbot, window, box_step):
    small_mesh(qtbot, window, box_step)
    window.scene.solver_settings.iterations = 100_000
    window.scene.solver_settings.n_procs = 1
    window.scene.notify_solver_settings_changed()
    assert window.run_solver()
    qtbot.waitUntil(lambda: window.convergence_plot.point_counts().get("drag", 0) >= 15, timeout=TIMEOUT)
    window.solver_panel.stop_button.click()
    qtbot.waitUntil(lambda: window._solver_job is None and not window.jobs.is_busy(), timeout=TIMEOUT)
    summary = window.scene.solver_summary
    assert summary["status"] == STATUS_STOPPED and 15 <= summary["iterations"]["done"] < 100_000
    assert "ユーザーにより停止" in window.results_panel.status_label.text()
    assert window.results_panel.result_labels["drag"].text() not in ("未計算", "計算失敗")
    # 基準面積が無い場合、係数は推測せず「未設定」と表示する (力はそのまま表示)
    window.scene.tunnel.frontal_area = None
    window.scene.notify_tunnel_changed()
    assert window.results_panel.result_labels["cd"].text() == "基準面積が未設定"
    assert window.results_panel.result_labels["cl"].text() == "基準面積が未設定"
    assert window.results_panel.result_labels["drag"].text() not in ("未計算", "計算失敗", "基準面積が未設定")
    assert "停止を要求しました" in window.log_panel.toPlainText()
    assert window.solver_panel.start_button.isEnabled() and not window.solver_panel.stop_button.isEnabled()


@needs_openfoam
def test_solver_failure_shows_failed_not_numbers(qtbot, window, box_step, monkeypatch):
    small_mesh(qtbot, window, box_step)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[1:3]) or QMessageBox.Ok)
    # 緩和係数 1.0・非直交補正 0・初期化なしにして、粗い四面体メッシュで発散させる
    settings = window.scene.solver_settings
    settings.iterations, settings.n_procs = 300, 1
    settings.relax_p = settings.relax_u = settings.relax_turbulence = 1.0
    settings.non_ortho_correctors, settings.potential_init = 0, False
    window.scene.notify_solver_settings_changed()
    summary = solve(qtbot, window)
    if summary["status"] != STATUS_FAILED:
        pytest.skip("この設定では発散しなかったため、失敗表示の確認はスキップ")
    assert errors and errors[0][0] == "計算失敗"
    assert all(label.text() == "計算失敗" for label in window.results_panel.result_labels.values())
    assert "発散" in window.results_panel.convergence_label.text()
    assert "解析に失敗しました" in window.log_panel.toPlainText()
