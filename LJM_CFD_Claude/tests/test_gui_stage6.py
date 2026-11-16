"""Stage 6 の GUI 結合テスト: 解析レポートの書き出しと、指示書の「最終的な完成目標」の一連の操作。"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("ディスプレイが無いため GUI テストをスキップ (xvfb-run で実行してください)", allow_module_level=True)

from ljmcfd.report.export import NOT_COMPUTED  # noqa: E402
from ljmcfd.solver.environment import native_openfoam_available  # noqa: E402
from tests.test_gui import TIMEOUT, load, window  # noqa: E402,F401  (window はフィクスチャ)
from tests.test_gui_stage2 import make_mesh, wait_idle  # noqa: E402
from tests.test_gui_stage4 import actor_points, with_results  # noqa: E402
from tests.test_gui_stage5 import choose  # noqa: E402

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")


def summary_rows(path) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {(row[0], row[1]): row[2] for row in list(csv.reader(fh))[1:]}


def test_report_without_results_from_gui(qtbot, window, box_step, tmp_path):
    """解析前でも書き出せる。結果の欄は「未計算」になり、数値は入らない。"""
    assert window.act_export_report.text().startswith("解析レポートを書き出し") and window.results_panel.report_button.isEnabled()
    load(qtbot, window, box_step, 1)
    files = window.export_report(tmp_path / "before")
    assert files is not None and set(files) == {"html", "summary", "view", "json"}
    values = summary_rows(files["summary"])
    assert values[("空力結果", "Drag (抗力)")] == NOT_COMPUTED and values[("空力結果", "Cd")] == NOT_COMPUTED
    assert values[("メッシュ", "メッシュ")] == "未生成" and values[("仮想風洞", "計算領域 X (流れ方向)")] == "風洞が未生成"
    assert values[("解析条件", "風速")] == "200.0"
    html = files["html"].read_text(encoding="utf-8")
    assert "未実行" in html and "box" in html.lower() and files["view"].stat().st_size > 1000


def test_report_from_gui_with_results(qtbot, window, box_step, tmp_path):
    from PIL import Image

    load(qtbot, window, box_step, 1)
    window.scene.drop_to_ground()
    assert window.generate_tunnel()
    wait_idle(qtbot, window)
    qtbot.waitUntil(lambda: window.scene.tunnel.frontal_area is not None, timeout=TIMEOUT)
    view, _panel = with_results(window)
    view.update(slice_on=True)
    requested = []
    window.results_panel.reportRequested.connect(lambda: requested.append(True))
    window.results_panel.reportRequested.disconnect(window._export_report_dialog)    # フォルダ選択ダイアログは開かない
    window.results_panel.report_button.click()
    assert requested == [True]

    files = window.export_report(tmp_path / "report")
    assert set(files) >= {"html", "summary", "view", "json"}
    values = summary_rows(files["summary"])
    # 画面の空力結果タブと同じ値 (係数は現在の基準面積で換算)
    labels = window.results_panel.result_labels
    assert float(values[("空力結果", "Drag (抗力)")]) == pytest.approx(float(labels["drag"].text()))
    assert float(values[("空力結果", "Downforce (下向き正)")]) == pytest.approx(float(labels["downforce"].text()))
    assert float(values[("空力結果", "Cd")]) == pytest.approx(float(labels["cd"].text()), abs=1e-4)
    assert values[("空力結果", "基準面積 (Cd・Cl の換算に使用)")] == f"{window.scene.tunnel.effective_reference_area():.4f}"
    # 合成データは現在のメッシュで計算したものではない → その旨がレポートにも残る
    assert "前の条件で計算" in values[("計算の状態と収束", "結果と現在の設定の関係")]
    html = files["html"].read_text(encoding="utf-8")
    assert "前の条件で計算したものです" in html and "計算領域 X" in html
    with Image.open(files["view"]) as image:                                  # 3D ビューの画像は実際の表示
        assert image.size[0] == window.viewport.width() and np.asarray(image.convert("RGB")).std() > 5
    assert not window.jobs.is_busy()


@needs_openfoam
def test_complete_workflow_from_step_to_saved_results(qtbot, window, box_step, tmp_path):
    """指示書 13 章の一連の操作を、GUI の部品を使って最初から最後まで通す。"""
    scene, view, bar = window.scene, window.post_view, window.flow_bar
    # 1. STEP ファイルを読み込む
    part = load(qtbot, window, box_step, 1)[0]
    # 2. モデルの角度・位置を調整する
    pose = window.model_panel.vehicle_pose
    pose.angle_spins["yaw"].setValue(4.0)
    window.model_panel.center_button.click()
    window.model_panel.ground_button.click()
    assert scene.assembly.pose.yaw == pytest.approx(4.0) and scene.analysis_bounds()[4] == pytest.approx(0.0, abs=1e-9)
    # 3. 仮想風洞を生成する  4. 風速 200 km/h を設定する
    tunnel_panel = window.tunnel_panel
    tunnel_panel.speed_kmh.setValue(180.0)
    tunnel_panel.speed_kmh.setValue(200.0)
    tunnel_panel.generate_button.click()
    qtbot.waitUntil(lambda: scene.tunnel.frontal_area is not None and not window.jobs.is_busy(), timeout=TIMEOUT)
    assert scene.tunnel.generated and scene.tunnel.flow.speed_kmh == pytest.approx(200.0)
    # 5. メッシュを生成する (テストが短時間で済む粗さにする)
    settings = scene.mesh_settings
    settings.preset = "custom"
    settings.surface_size, settings.min_size, settings.far_size = 0.22, 0.11, 3.0
    settings.refine_distance, settings.wake_length = 8.0, 6.0
    scene.notify_mesh_settings_changed()
    report = make_mesh(qtbot, window)
    assert report["quality"]["n_cells"] > 1000 and scene.mesh_state() == "current"
    # 6. CFD 解析を実行する (反復数は選択肢から 200 を選ぶ)
    solver_panel = window.solver_panel
    choose(solver_panel.iteration_combo, 200)
    solver_panel.procs.setValue(2)
    assert scene.solver_settings.iterations == 200 and solver_panel.start_button.isEnabled()
    solver_panel.start_button.click()
    qtbot.waitUntil(lambda: window._solver_job is not None, timeout=TIMEOUT)
    qtbot.waitUntil(lambda: window._solver_job is None and not window.jobs.is_busy(), timeout=TIMEOUT * 3)
    summary = scene.solver_summary
    assert summary is not None and summary["status"] in ("completed", "residual_stop") and scene.solver_state() == "current"
    # 7. ダウンフォースとドラッグを確認する
    labels = window.results_panel.result_labels
    drag, downforce, cd = float(labels["drag"].text()), float(labels["downforce"].text()), float(labels["cd"].text())
    assert drag == pytest.approx(summary["forces"]["drag"], abs=0.01) and drag > 0
    assert downforce == pytest.approx(summary["forces"]["downforce"], abs=0.01)
    assert 0.3 < cd < 3.0                                                    # 地面に置いた直方体として妥当な範囲
    assert "収束判定" in window.results_panel.convergence_label.text()
    # 8. 車両周囲の流線を 3D 表示する
    bar.seed_check.setChecked(True)
    assert view.layer_actors()["streamlines"] and len(actor_points(window, "post_streamlines")) > 500
    # 9. スライダーを動かして空気の流れの断面を確認する
    bar.slice_check.setChecked(True)
    for step in (300, 450, 600):
        bar.slice_slider.setValue(step)
        lo, hi = view.results.axis_range("y")
        assert np.allclose(actor_points(window, "post_slice")[:, 1], lo + (hi - lo) * step / 1000, atol=1e-6)
    # 10. スモークアニメーションを再生し、タイムラインで前後に移動する
    bar.particle_check.setChecked(True)
    bar.play_button.click()
    qtbot.waitUntil(lambda: view.settings.particles_time > 0, timeout=10000)
    bar.pause_button.click()
    bar.time_slider.setValue(700)
    forward = view.particle_positions()
    bar.time_slider.setValue(200)
    earlier = view.particle_positions()
    bar.time_slider.setValue(700)
    assert len(forward) > 0 and np.array_equal(view.particle_positions(), forward) and not np.array_equal(forward, earlier)
    assert window._solver_job is None and not window.jobs.is_busy()          # 表示の操作で解析は再実行されない
    # 11. 解析結果を保存する (プロジェクト + レポート)
    project_dir = tmp_path / "car_project"
    assert window.save_project_to(project_dir)
    files = window.export_report(project_dir / "reports" / "r1")
    assert set(files) - {"parts"} == {"html", "summary", "history", "view", "json"}
    assert ("parts" in files) == bool(summary.get("parts"))                  # パーツ別の荷重は、解析がパーツ別に出したときだけ
    values = summary_rows(files["summary"])
    assert float(values[("空力結果", "Drag (抗力)")]) == pytest.approx(drag, abs=0.01)
    assert len(files["history"].read_text(encoding="utf-8").splitlines()) == summary["iterations"]["done"] + 1
    assert values[("計算の状態と収束", "結果と現在の設定の関係")] == "現在の設定で計算した結果"
    assert values[("メッシュ", "セル数")] == f"{report['quality']['n_cells']:,}".replace(",", "")
    assert "境界層メッシュがありません" in files["html"].read_text(encoding="utf-8")
    for name in ("project.ljmcfd.json", "mesh/polyMesh/points", "results/summary.json", "results/fields.vtu", "results/history.csv"):
        assert (project_dir / name).is_file(), name
    assert (project_dir / "project.ljmcfd.json").stat().st_size < 100_000     # 大きなデータは JSON に埋め込まない

    # 保存したプロジェクトを開き直すと、結果・表示設定・スライダー位置が戻る (再計算なし)
    window.new_project(confirm=False)
    assert scene.solver_summary is None and labels["drag"].text() == "未計算" and bar.isHidden()
    assert window.open_project(project_dir)
    qtbot.waitUntil(lambda: len(scene.parts) == 1 and not window._restoring, timeout=TIMEOUT)
    assert scene.assembly.pose.yaw == pytest.approx(4.0) and scene.tunnel.generated
    assert scene.mesh_state() == "current" and scene.solver_state() == "current"
    assert float(labels["drag"].text()) == pytest.approx(drag) and float(labels["downforce"].text()) == pytest.approx(downforce)
    s = view.settings
    assert s.slice_on and s.streamlines_on and s.particles_on and bar.slice_slider.value() == 600 and bar.time_slider.value() == 700
    assert np.allclose(view.particle_positions(), forward) and not window.jobs.is_busy()
    assert part.name == scene.parts[0].name
