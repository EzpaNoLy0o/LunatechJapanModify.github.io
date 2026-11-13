"""GUI の結合テスト: 実際にウィンドウを開き、読み込みから保存までを操作する。

ディスプレイが必要 (Linux では ``xvfb-run -a python -m pytest``)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("ディスプレイが無いため GUI テストをスキップ (xvfb-run で実行してください)", allow_module_level=True)

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from ljmcfd.core.pose import Pose  # noqa: E402
from ljmcfd.core.project import load_project  # noqa: E402
from ljmcfd.gui import theme  # noqa: E402
from ljmcfd.gui.viewport import MODE_FACE, MODE_MEASURE, MODE_PART  # noqa: E402

TIMEOUT = 120_000


@pytest.fixture()
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("LJMCFD_HOME", str(tmp_path / "home"))
    from ljmcfd.gui.main_window import MainWindow

    win = MainWindow()
    win.setStyleSheet(theme.STYLESHEET)
    # qtbot.addWidget は使わない: pytest-qt がフィクスチャ後始末より先に close() を呼び、
    # 「未保存の変更」確認ダイアログ (モーダル) でテストが止まってしまうため。
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win._modified = False   # 終了時の保存確認ダイアログを出さない
    win.close()


def load(qtbot, window, path, n_parts):
    window.import_files([str(path)])
    qtbot.waitUntil(lambda: len(window.scene.parts) == n_parts and not window.jobs.is_busy(), timeout=TIMEOUT)
    window.viewport.plotter.render()
    return window.scene.parts


def click(qtbot, window, world_point, modifier=Qt.NoModifier):
    """ワールド座標の点を 3D ビュー上で実際にマウスクリックする。"""
    viewport = window.viewport
    viewport.plotter.render()
    x, y = viewport.world_to_widget(world_point)
    qtbot.mouseClick(viewport.plotter, Qt.LeftButton, modifier, QPoint(x, y))
    qtbot.wait(50)


def part_named(window, name):
    return next(p for p in window.scene.parts if p.name == name)


def test_layout_is_resizable_and_japanese(window):
    assert window.main_splitter.count() == 3 and window.outer_splitter.count() == 2
    assert window.main_splitter.widget(1) is window.viewport          # 中央が 3D ビュー
    assert window.outer_splitter.widget(1) is window.bottom_tabs      # 下部がログ
    assert [window.bottom_tabs.tabText(i) for i in range(window.bottom_tabs.count())] == ["ログ", "形状検査"]
    before = window.main_splitter.sizes()
    window.main_splitter.setSizes([500, 600, 300])
    assert window.main_splitter.sizes() != before
    assert "起動しました" in window.log_panel.toPlainText()
    # CFD 未実装の段階で数値を表示しない
    assert all(label.text() == "未計算" for label in window.display_panel.result_labels.values())


def test_import_select_and_visibility(qtbot, window, assembly_step):
    parts = load(qtbot, window, assembly_step, 2)
    assert sorted(p.name for p in parts) == ["Body", "RearWing"]
    tree = window.part_tree
    group = tree.topLevelItem(0)
    assert group.text(0) == "car_assembly.step" and group.childCount() == 2
    body, wing = part_named(window, "Body"), part_named(window, "RearWing")
    assert window.viewport._actors[wing.id]["surface"].prop.color.float_rgb == pytest.approx((1.0, 0.0, 0.0))

    # 3D ビュー上のクリックでパーツ選択 → ツリーと配色が連動
    click(qtbot, window, (2.0, 0.9, 1.2))            # ボディ上面の中央
    assert window.scene.selected_ids == [body.id]
    assert tree._items[body.id].isSelected() and not tree._items[wing.id].isSelected()
    assert window.viewport._actors[body.id]["surface"].prop.color.float_rgb == pytest.approx(theme.SELECT_RGB, abs=0.01)
    assert window.model_panel.name_label.text() == "Body"
    assert "4000.0 × 1800.0 × 1200.0 mm" in window.model_panel.size_label.text()

    # Ctrl+クリックで追加選択、背景クリックで解除
    click(qtbot, window, (3.85, 0.9, 1.33), Qt.ControlModifier)
    assert set(window.scene.selected_ids) == {body.id, wing.id}
    window.viewport.pick_at(3, 3)
    assert window.scene.selected_ids == []
    assert window.viewport._actors[body.id]["surface"].prop.color.float_rgb == pytest.approx(body.color, abs=0.01)

    # ツリーのチェックで非表示 → 隠れたパーツはピックされず、奥のパーツが選ばれる
    tree._items[wing.id].setCheckState(0, Qt.Unchecked)
    assert wing.visible is False and not window.viewport._actors[wing.id]["surface"].GetVisibility()
    assert window.viewport.pick(*window.viewport.world_to_widget((3.85, 0.9, 1.33))) is None
    tree._items[wing.id].setCheckState(0, Qt.Checked)
    assert window.viewport._actors[wing.id]["surface"].GetVisibility()

    # ツリー側の選択 → シーンへ反映
    tree._items[wing.id].setSelected(True)
    assert window.scene.selected_ids == [wing.id]
    window.scene.remove_parts([wing.id])
    assert group.childCount() == 1 and wing.id not in window.viewport._actors


def test_face_selection_and_measurement(qtbot, window, box_step):
    load(qtbot, window, box_step, 1)
    viewport = window.viewport
    window.mode_actions[MODE_FACE].trigger()
    assert viewport.mode == MODE_FACE
    click(qtbot, window, (2.0, 0.9, 1.2))            # 上面
    info = window.scene.face_info()
    assert info is not None and info["surface_type"] == "plane"
    assert info["area"] == pytest.approx(4.0 * 1.8, rel=1e-6)
    assert info["normal"] == pytest.approx([0, 0, 1], abs=1e-6)
    assert "面積: 7.2 m²" in window.display_panel.selection_label.text()
    assert "face_highlight" in viewport.plotter.actors

    window.mode_actions[MODE_MEASURE].trigger()
    assert window.scene.selected_face is None and "face_highlight" not in viewport.plotter.actors
    # 上面の対角の頂点付近をクリック → 頂点へスナップして正確な対角長になる
    click(qtbot, window, (0.02, 0.02, 1.2))
    assert viewport.measurement()["distance"] is None
    click(qtbot, window, (3.98, 1.78, 1.2))
    result = viewport.measurement()
    assert result["distance"] == pytest.approx(np.hypot(4.0, 1.8), abs=1e-6)
    assert result["delta"] == pytest.approx([4.0, 1.8, 0.0], abs=1e-6)
    assert f"距離: {np.hypot(4000.0, 1800.0):.1f} mm" in window.display_panel.measure_label.text()
    # スナップを切るとクリック位置そのもの
    window.display_panel.snap_check.setChecked(False)
    click(qtbot, window, (1.0, 0.5, 1.2))
    assert viewport.measurement()["p1"] == pytest.approx([1.0, 0.5, 1.2], abs=0.02)
    window.display_panel.measure_clear.click()
    assert viewport.measurement() is None
    window.mode_actions[MODE_PART].trigger()
    assert viewport.mode == MODE_PART


def test_pose_editing_units_and_ground(qtbot, window, assembly_step):
    load(qtbot, window, assembly_step, 2)
    scene, panel = window.scene, window.model_panel
    wing = part_named(window, "RearWing")
    scene.set_selection([wing.id])
    # パーツ配置: Z を +100 mm、Yaw 90°
    panel.part_pose.length_spins["tz"].setValue(100.0)
    panel.part_pose.angle_spins["yaw"].setValue(90.0)
    assert wing.pose == Pose(tz=0.1, yaw=90.0)
    b = scene.assembly.world_bounds([wing])
    assert (b[1] - b[0], b[3] - b[2], b[4]) == pytest.approx((1.6, 0.3, 1.4), abs=1e-6)
    matrix = window.viewport._actors[wing.id]["surface"].user_matrix
    assert matrix == pytest.approx(scene.assembly.world_matrix(wing))
    panel.part_reset.click()
    assert wing.pose == Pose()

    # 表示単位 m へ切り替え → 入力欄の単位と値が追従
    window.display_panel.unit_combo.setCurrentText("m")
    spin = panel.vehicle_pose.length_spins["tz"]
    assert spin.suffix() == " m"
    spin.setValue(0.25)
    assert scene.assembly.pose.tz == pytest.approx(0.25)
    assert "4.0000 × 1.8000 × 1.3300 m" in panel.total_label.text()

    # 車両全体: Pitch を付けて接地
    panel.vehicle_pose.angle_spins["pitch"].setValue(3.0)
    assert scene.assembly.world_bounds()[4] != pytest.approx(0.0, abs=1e-6)
    panel.ground_button.click()
    assert scene.assembly.world_bounds()[4] == pytest.approx(0.0, abs=1e-9)
    panel.center_button.click()
    b = scene.assembly.world_bounds()
    assert (b[0] + b[1], b[2] + b[3]) == pytest.approx((0.0, 0.0), abs=1e-9)
    panel.vehicle_reset.click()
    assert scene.assembly.pose == Pose()


def test_stl_unit_is_flagged_and_editable(qtbot, window, cube_stl):
    part = load(qtbot, window, cube_stl, 1)[0]
    assert part.unit == "m" and part.unit_source == "heuristic"
    assert "推定" in window.log_panel.toPlainText()
    window.scene.set_selection([part.id])
    assert "推定" in window.model_panel.unit_note.text()
    assert window.part_tree._items[part.id].text(1) == "m?"
    combo = window.model_panel.unit_combo
    combo.setCurrentText("mm")
    combo.activated.emit(combo.currentIndex())
    assert part.unit == "mm" and part.unit_source == "user"
    assert "2.0 × 1.0 × 0.5 mm" in window.model_panel.size_label.text()
    assert window.part_tree._items[part.id].text(1) == "mm"


def test_inspection_reports_open_geometry(qtbot, window, open_box_step, box_step):
    load(qtbot, window, open_box_step, 1)
    load(qtbot, window, box_step, 2)
    opened, closed = window.scene.parts
    window.inspect_selected()                        # 未選択 → 全パーツ
    qtbot.waitUntil(lambda: all(p.inspection for p in window.scene.parts) and not window.jobs.is_busy(), timeout=TIMEOUT)
    assert opened.inspection["summary"]["level"] == "error"
    assert closed.inspection["summary"]["level"] == "ok"
    assert window.viewport._actors[opened.id]["defects"] is not None   # 開放エッジを 3D に表示
    assert window.viewport._actors[closed.id]["defects"] is None
    assert window.part_tree._items[opened.id].text(2) == "不良"
    assert window.part_tree._items[closed.id].text(2) == "OK"
    table = window.inspection_table
    assert table.rowCount() == 2 and table.item(0, 2).text() == "4"
    assert table.item(1, 6).text() == f"{4.0 * 1.8 * 1.2:.6g}"         # 体積 [m³]
    assert "開放エッジ" in window.log_panel.toPlainText()
    table.selectRow(1)
    assert window.scene.selected_ids == [closed.id]


def test_display_settings_and_screenshot(qtbot, window, assembly_step, tmp_path):
    load(qtbot, window, assembly_step, 2)
    viewport, panel = window.viewport, window.display_panel
    dark = tmp_path / "dark.png"
    window.save_screenshot(dark)
    panel.background_combo.setCurrentIndex(panel.background_combo.findData("light"))
    assert viewport.theme_name == "light"
    light = tmp_path / "light.png"
    window.save_screenshot(light)
    import pyvista as pv

    img_dark, img_light = pv.read(dark).point_data.active_scalars, pv.read(light).point_data.active_scalars
    assert img_dark.std() > 5 and img_light.std() > 5          # 何かが描画されている
    assert img_light.mean() > img_dark.mean() + 40             # 白系背景は明るい

    # 半透明にしても面が描画され続ける (赤いウイングの画素が残る)。
    # MSAA と depth peeling を併用すると半透明面が消える不具合の回帰テスト。
    def red_fraction():
        img = viewport.plotter.screenshot().astype(int)
        return ((img[..., 0] - img[..., 2]) > 40).mean()

    viewport.set_view("top")
    viewport.plotter.render()
    opaque_red = red_fraction()
    assert opaque_red > 0.002
    panel.opacity_slider.setValue(40)
    assert all(e["surface"].prop.opacity == pytest.approx(0.4) for e in viewport._actors.values())
    viewport.plotter.render()
    assert red_fraction() > 0.5 * opaque_red
    panel.edges_check.setChecked(False)
    assert not any(e["edges"].GetVisibility() for e in viewport._actors.values())
    panel.bbox_check.setChecked(False)
    assert "bbox" not in viewport.plotter.actors
    panel.bbox_check.setChecked(True)
    assert "bbox" in viewport.plotter.actors
    panel.ground_check.setChecked(False)
    assert "ground" not in viewport.plotter.actors
    panel.parallel_check.setChecked(True)
    assert viewport.plotter.camera.parallel_projection
    for name in ("front", "rear", "left", "right", "top", "bottom", "iso"):
        viewport.set_view(name)
    direction = np.array(viewport.plotter.camera.position) - np.array(viewport.plotter.camera.focal_point)
    assert direction[0] < 0 and direction[1] < 0 and direction[2] > 0   # 等角: 左前方上から


def test_project_save_and_reopen(qtbot, window, assembly_step, tmp_path):
    load(qtbot, window, assembly_step, 2)
    scene = window.scene
    body, wing = part_named(window, "Body"), part_named(window, "RearWing")
    scene.set_part_pose([wing.id], tz=0.05, pitch=-8.0)
    scene.set_vehicle_pose(yaw=4.0)
    scene.drop_to_ground()
    scene.set_visible([body.id], False)
    window.display_panel.unit_combo.setCurrentText("m")
    window.display_panel.background_combo.setCurrentIndex(1)
    window.display_panel.opacity_slider.setValue(55)
    window.viewport.set_view("left")
    window.viewport.plotter.render()
    camera_before = window.viewport.camera_state()
    bounds_before = scene.assembly.world_bounds()
    assert "*" in window.windowTitle()

    project_dir = tmp_path / "projects" / "car01"
    assert window.save_project_to(project_dir)
    assert "*" not in window.windowTitle() and window.windowTitle().startswith("car01")
    saved, _ = load_project(project_dir)
    assert len(saved.parts) == 2 and saved.view["background"] == "light"
    assert (project_dir / "project.ljmcfd.json").stat().st_size < 20_000   # メッシュを埋め込んでいない

    window.new_project(confirm=False)
    assert not scene.parts and not window.viewport._actors
    assert window.open_project(project_dir)
    qtbot.waitUntil(lambda: len(scene.parts) == 2 and not window._restoring, timeout=TIMEOUT)
    body2, wing2 = part_named(window, "Body"), part_named(window, "RearWing")
    assert (body2.id, wing2.id) == (body.id, wing.id)
    assert wing2.pose == Pose(tz=0.05, pitch=-8.0) and body2.visible is False
    assert scene.assembly.pose.yaw == pytest.approx(4.0)
    assert scene.assembly.world_bounds() == pytest.approx(bounds_before)
    assert scene.display_unit == "m" and window.viewport.theme_name == "light"
    assert window.display_panel.opacity_slider.value() == 55
    assert window.viewport.camera_state()["position"] == pytest.approx(camera_before["position"], rel=1e-6)
    assert "*" not in window.windowTitle()


def test_project_reopen_without_source_uses_cache_or_reports(qtbot, window, box_step, tmp_path, monkeypatch):
    import shutil

    src = tmp_path / "cad" / "body.step"
    src.parent.mkdir()
    shutil.copyfile(box_step, src)
    load(qtbot, window, src, 1)
    project_dir = tmp_path / "proj"
    window.save_project_to(project_dir)
    src.unlink()

    # 元ファイルが無くてもキャッシュが残っていれば復元できる
    window.new_project(confirm=False)
    window.open_project(project_dir)
    qtbot.waitUntil(lambda: not window._restoring, timeout=TIMEOUT)
    assert len(window.scene.parts) == 1
    assert "キャッシュから復元" in window.log_panel.toPlainText()

    # キャッシュも無ければ、黙って空にせず問題を報告する
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kw: warnings.append(args[2]) or QMessageBox.Ok)
    shutil.rmtree(tmp_path / "home" / "cache")
    window.new_project(confirm=False)
    window.open_project(project_dir)
    qtbot.waitUntil(lambda: not window._restoring, timeout=TIMEOUT)
    assert not window.scene.parts
    assert warnings and "body.step" in warnings[0]


def test_import_failure_is_reported(qtbot, window, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kw: errors.append(args[2]) or QMessageBox.Ok)
    garbage = tmp_path / "broken.step"
    garbage.write_text("not a step file")
    window.import_files([str(garbage), str(tmp_path / "model.obj")])
    qtbot.waitUntil(lambda: bool(errors) and not window.jobs.is_busy(), timeout=TIMEOUT)
    assert "broken.step" in errors[0]
    log = window.log_panel.toPlainText()
    assert "未対応のファイル形式" in log and "読み込み broken.step" in log
    assert not window.scene.parts


def test_vtk_default_keys_are_disabled(qtbot, window, box_step):
    """3D ビューにフォーカスがある状態で q / e / w などを押しても、VTK 既定の動作をしない。"""
    part = load(qtbot, window, box_step, 1)[0]
    viewport = window.viewport
    viewport.plotter.setFocus()
    for key in ("q", "e", "w", "s", "3", "p", "r"):
        qtbot.keyClick(viewport.plotter, key)
    qtbot.wait(50)
    assert window.isVisible() and viewport.plotter.render_window is not None
    actor = viewport._actors[part.id]["surface"]
    assert actor.prop.style.lower() == "surface"                      # w でワイヤーフレームにならない
    assert not viewport.plotter.render_window.GetStereoRender()  # 3 でステレオ表示にならない
    viewport.plotter.render()
    assert viewport.pick(*viewport.world_to_widget((2.0, 0.9, 1.2))) is not None
