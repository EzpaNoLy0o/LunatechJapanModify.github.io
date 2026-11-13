"""GUI テスト: 起動・ファイル読込・主要ワークフロー・全ボタンの動作確認・保存/再読込.

ダイアログはテスト用に自動応答させる（キャンセル / 既定値で OK）。
Linux ではディスプレイ（xvfb-run 等）が必要。
"""
import logging
import os

import numpy as np
import pytest

from conftest import has_display

if not has_display():
    pytest.skip("GUI テストにはディスプレイが必要です（xvfb-run を使用してください）", allow_module_level=True)

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from ljm_re.core import synthetic as S  # noqa: E402
from ljm_re.io.mesh_io import export_mesh  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ljm_re.app import create_app
    a = create_app()
    return a


class DialogBot:
    """ダイアログの自動応答とエラー記録."""

    def __init__(self, monkeypatch, tmpdir, accept=False):
        self.errors = []
        self.warnings = []
        self.tmpdir = str(tmpdir)
        self.accept = accept
        from ljm_re.gui import dialogs, main_window
        bot = self

        def ask(parent, title, fields, description=""):
            if not bot.accept:
                return None
            d = dialogs.ParamDialog(title, fields, parent, description)
            return d.values()
        monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(ask))
        monkeypatch.setattr(dialogs.ReportDialog, "exec", lambda self: 0)
        monkeypatch.setattr(dialogs.MountPointsDialog, "exec", lambda self: 0)
        monkeypatch.setattr(dialogs.ImportUnitDialog, "exec", lambda self: QtWidgets.QDialog.Accepted)
        W = main_window.MainWindow
        monkeypatch.setattr(W, "info", lambda self, t, x: None)
        monkeypatch.setattr(W, "warn", lambda self, t, x: bot.warnings.append((t, x)))
        monkeypatch.setattr(W, "error", lambda self, t, x, d="": bot.errors.append((t, x, d)))
        monkeypatch.setattr(W, "confirm", lambda self, t, x: bot.accept)

        def ask_save(self_, title, filt, default_name):
            return os.path.join(bot.tmpdir, default_name) if bot.accept else None
        monkeypatch.setattr(W, "ask_save", ask_save)
        monkeypatch.setattr(W, "ask_open", lambda self_, *a, **k: [] if k.get("multi") else None)
        monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
        monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", staticmethod(lambda *a, **k: QtGui.QColor()))


@pytest.fixture
def win(app, monkeypatch, tmp_path):
    from ljm_re.gui.main_window import MainWindow
    from ljm_re.utils.settings import AppSettings
    monkeypatch.setattr(AppSettings, "save", lambda self: None)
    s = AppSettings()
    w = MainWindow(s)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    w.doc.undo_stack.set_clean()
    w.runner.wait(30000)
    w.close()
    app.processEvents()


def _import_bracket(w, d):
    mesh, truth = S.make_bracket()
    p = os.path.join(str(d), "ブラケット.stl")
    export_mesh(p, mesh.vertices, mesh.faces)
    w.import_files([p], unit="mm", ask_unit=False)
    assert w.runner.wait(30000)
    return w.doc.objects("mesh")[0], truth


def test_startup_and_all_actions_connected(win):
    from PySide6.QtCore import QMetaMethod
    assert win.windowTitle().startswith("LJM Reverse Engineering")
    assert len(win.actions) >= 120
    for key, a in win.actions.items():
        sig = QMetaMethod.fromSignal(a.toggled if a.isCheckable() else a.triggered)
        assert a.isSignalConnected(sig), f"未接続のアクション: {key}"
    # 指定されたツールバーカテゴリ
    tabs = [win.ribbon.tabText(i) for i in range(win.ribbon.count())]
    assert tabs == ["ファイル", "編集", "表示", "スキャンデータ", "測定", "位置合わせ", "形状認識", "CAD再構築",
                    "エクスポート", "設定"]


def test_workflow_bracket(win, monkeypatch, jp_dir, app, caplog):
    caplog.set_level(logging.INFO)
    bot = DialogBot(monkeypatch, jp_dir, accept=True)
    m, truth = _import_bracket(win, jp_dir)
    from ljm_re.core.selection import region_grow
    # 上面: スマート選択 → 平面フィット
    f = m.topology.nearest_face([0, -30, 8])
    win.viewer.set_selection(m.id, region_grow(m.topology, f, 5.0), "replace")
    win.cmd_fit_plane()
    assert win.runner.wait(30000)
    top = win.doc.objects("plane")[-1]
    assert abs(top.normal[2]) > 0.999999 and abs(top.origin[2] - 8) < 1e-6
    # 側面 2 面 → 3-2-1 座標系用
    topo = m.topology
    for nrm in ([1, 0, 0], [0, -1, 0]):
        cand = np.nonzero(topo.face_normals @ np.array(nrm, float) > 0.999)[0]
        f = int(cand[np.argmax(topo.centroids[cand] @ np.array(nrm, float))])
        win.viewer.set_selection(m.id, region_grow(topo, f, 5.0), "replace")
        win.cmd_fit_plane()
        assert win.runner.wait(30000)
    planes = win.doc.objects("plane")
    assert len(planes) == 3
    # 穴の内壁 → 穴中心（内壁から, 基準平面 = 上面）
    c = m.topology.centroids
    wall = (np.hypot(c[:, 0] - 30, c[:, 1] - 25) < 8.5) & (c[:, 2] > 0.5) & (c[:, 2] < 7.5)
    win.viewer.set_selection(m.id, wall, "replace")
    win.tree.select_ids([top.id])
    win.cmd_hole_from_cylinder()
    assert win.runner.wait(30000)
    hole = win.doc.objects("hole")[-1]
    assert np.allclose(hole.center, [30, 25, 8], atol=1e-5) and abs(hole.diameter - 16) < 0.01
    # 円筒フィット
    win.cmd_fit_cylinder()
    assert win.runner.wait(30000)
    cyl = win.doc.objects("cylinder")[-1]
    assert abs(cyl.diameter - 16) < 0.01
    # 平面上の穴の自動検出（4個）
    win.tree.select_ids([top.id])
    win.cmd_detect_holes()
    assert win.runner.wait(30000)
    assert len(win.doc.objects("hole")) == 5
    # 要素間測定: 穴中心間
    holes = [h for h in win.doc.objects("hole") if h.meta.get("params", {}).get("auto")]
    win.tree.select_ids([holes[0].id, holes[1].id])
    win.cmd_measure_between()
    meas = win.doc.objects("measurement")[-1]
    assert meas.mtype == "hole_hole"
    exp = np.linalg.norm(holes[0].center - holes[1].center)
    assert meas.result["distance"] == pytest.approx(exp)
    # 平面間（平行でない → 角度90°）
    win.tree.select_ids([planes[0].id, planes[1].id])
    win.cmd_measure_between()
    assert win.doc.objects("measurement")[-1].result["angle"] == pytest.approx(90, abs=1e-6)
    # 3-2-1 座標系
    win.tree.select_ids([p.id for p in planes])
    win.cmd_cs_321()
    assert win.doc.objects("cs"), bot.warnings
    cs = win.doc.objects("cs")[-1]
    assert np.allclose(cs.frame.origin, [75, -40, 8], atol=1e-6)
    win.cmd_activate_cs_id(cs.id)
    assert win.doc.active_frame_name == cs.name
    # 2点間距離（ピックツール経由）
    win.cmd_measure_distance()
    tool = win.viewer.tool
    tool.points = [(np.array([0, 0, 0.0]), m.id, "a")]
    tool.callback(tool.points + [(np.array([3, 4, 0.0]), m.id, "b")])
    assert win.doc.objects("measurement")[-1].result["distance"] == pytest.approx(5)
    win.viewer.set_tool(None)
    # 断面 → スケッチ化 → 押し出し → STEP 出力（検証付き）
    win.tree.select_ids([m.id])
    win.cmd_section_axis()
    sec = win.doc.objects("section")[-1]
    assert len(sec.polylines) >= 1
    win.tree.select_ids([sec.id])
    win.cmd_sketch_lines_arcs()
    assert any(len(s) for s in sec.sketch)
    win.tree.select_ids([top.id, cyl.id])
    win.cmd_cad_from_features()
    assert len(win.doc.objects("cad")) == 2
    win.tree.select_ids([])
    win.cmd_export_step()
    assert win.runner.wait(60000)
    assert os.path.exists(os.path.join(str(jp_dir), "reconstruction.step"))
    assert os.path.exists(os.path.join(str(jp_dir), "reconstruction_datums.csv"))
    # CSV
    win.cmd_export_csv_features(None)
    assert os.path.exists(os.path.join(str(jp_dir), "coordinates.csv"))
    # Undo / Redo
    n = len(win.doc)
    win.cmd_undo()
    assert len(win.doc) == n - 1 or win.doc.undo_stack.can_redo
    win.cmd_redo()
    # 保存 → 新規 → 再読込
    p = os.path.join(str(jp_dir), "ワーク.ljmre")
    assert win.save_project(p)
    win.set_document(type(win.doc)())
    win.open_project(p)
    assert win.runner.wait(30000)
    assert len(win.doc.objects("hole")) == 5 and win.doc.active_cs_id == cs.id
    win.viewer.set_view("iso")
    win.viewer.screenshot(os.path.join(str(jp_dir), "screen.png"))
    assert not bot.errors, bot.errors
    errs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errs, errs


@pytest.mark.parametrize("accept", [False, True])
def test_every_action_runs_without_error(win, monkeypatch, jp_dir, app, accept, caplog):
    caplog.set_level(logging.INFO)
    """全アクションを実行（空のドキュメント / データ読込後）して例外が出ないことを確認."""
    bot = DialogBot(monkeypatch, jp_dir, accept=accept)
    skip = {"exit", "new", "open"}
    failed = []
    for key, a in win.actions.items():
        if key in skip or not a.isEnabled():
            continue
        n0 = len(bot.errors)
        a.trigger()
        win.runner.wait(60000)
        win.viewer.set_tool(None)
        app.processEvents()
        if len(bot.errors) > n0:
            failed.append(("empty", key, bot.errors[-1][1]))
    m, _ = _import_bracket(win, jp_dir)
    # 選択状態と各種要素を用意
    from ljm_re.core.selection import region_grow
    win.viewer.set_selection(m.id, region_grow(m.topology, m.topology.nearest_face([0, -30, 8]), 5.0), "replace")
    win.cmd_fit_plane()
    win.runner.wait(30000)
    for key, a in win.actions.items():
        if key in skip or not a.isEnabled():
            continue
        if key == "align_to_cs":
            continue
        win.tree.select_ids([o.id for o in win.doc.objects()[:2]])
        n0 = len(bot.errors)
        a.trigger()
        win.runner.wait(60000)
        if len(bot.errors) > n0:
            failed.append(("data", key, bot.errors[-1][1]))
        if win.viewer.tool is not None:
            win.viewer.set_tool(None)
        app.processEvents()
        if win.doc.get(m.id) is None:  # 削除系アクションの後は元に戻す
            win.cmd_undo()
    assert not failed, failed
    errs = [r.getMessage()[:300] for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errs, errs


def test_viewer_picking_and_selection(win, monkeypatch, jp_dir, app):
    """画面座標からのピック・スマート選択・矩形選択（深度バッファで前面のみ）・スナップ."""
    from ljm_re.core.selection import project_to_display
    bot = DialogBot(monkeypatch, jp_dir, accept=True)
    m, _ = _import_bracket(win, jp_dir)
    v = win.viewer
    v.set_view("top")
    app.processEvents()
    mvp, size, cam = v._display_info()

    def disp(p):
        d = project_to_display(np.array([p], float), mvp, size)[0]
        return int(round(d[0])), int(round(d[1]))

    # ピック: 上面の点
    x, y = disp([0, -30, 8])
    r = v.pick_point(x, y, snap=False)
    assert r is not None and r[1] == m.id
    assert abs(r[0][2] - 8) < 0.5 and np.linalg.norm(r[0][:2] - [0, -30]) < 2
    # スマート選択 → 上面のみ（z=8）
    v.smart_select(x, y, "replace")
    sel = v.selection[m.id]
    assert np.allclose(m.topology.centroids[sel][:, 2], 8)
    # 矩形選択（前面のみ）: 上から見てベース全体を囲む → 上面の面は選ばれ、底面（z=0）は選ばれない
    x0, y0 = disp([-80, -45, 8])
    x1, y1 = disp([80, 35, 8])
    win.settings.rect_front_only = True
    v.rect_select((x0, y0, x1, y1), "replace")
    sel = v.selection[m.id]
    z = m.topology.centroids[sel][:, 2]
    assert (z > 7.9).sum() > 0 and (z < 0.1).sum() == 0
    win.settings.rect_front_only = False
    v.rect_select((x0, y0, x1, y1), "replace")
    z2 = m.topology.centroids[v.selection[m.id]][:, 2]
    assert (z2 < 0.1).sum() > 0  # 裏側も選択される
    # スナップ: 穴中心の近くをクリックすると穴中心に吸着
    from ljm_re.model.objects import HoleFeature
    h = HoleFeature("穴S", [30, 25, 8], [0, 0, 1], 16)
    win.doc.add(h)
    app.processEvents()
    hx, hy = disp([30, 25, 8])
    r = v.pick_point(hx + 4, hy + 3, snap=True)
    assert r is not None and r[1] == h.id and np.allclose(r[0], [30, 25, 8])
    # ブラシ
    v.brush_select(x, y, "replace")
    assert v.selection[m.id].sum() > 0
    v.clear_selection()
    assert not v.selection
    assert not bot.errors
