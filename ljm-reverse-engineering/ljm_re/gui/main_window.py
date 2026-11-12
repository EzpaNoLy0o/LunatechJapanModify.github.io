"""メインウィンドウ: メニュー/リボン・ドック・ステータスバー・ファイル/編集/表示コマンド."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __app_name__, __version__
from ..model.document import Document
from ..model.formatting import NumberFormat
from ..model.objects import KIND_LABELS, CadObject, MeshObject, PointCloudObject
from ..utils.logging_setup import CallbackHandler
from ..utils.paths import log_dir
from .cmds_cad import CadCommands
from .cmds_fit import FitCommands
from .cmds_measure import MeasureCommands
from .cmds_scan import ScanCommands
from .dialogs import ClipDialog, ImportUnitDialog, ParamDialog, ReportDialog
from .model_tree import ModelTree
from .property_panel import PropertyPanel
from .viewer import Viewer3D
from .workers import TaskRunner

log = logging.getLogger(__name__)

PROJECT_FILTER = "LJM プロジェクト (*.ljmre)"
MESH_FILTER = "メッシュ/点群 (*.stl *.obj *.ply);;STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;すべて (*.*)"


class _LogBridge(QtCore.QObject):
    message = QtCore.Signal(int, str)


class MainWindow(QtWidgets.QMainWindow, ScanCommands, MeasureCommands, FitCommands, CadCommands):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.doc = Document()
        self.runner = TaskRunner(self)
        self.actions: dict[str, QtGui.QAction] = {}
        self._warn_count = 0
        self._clip_dialog = None
        self.setWindowTitle(f"{__app_name__} {__version__}")
        self.resize(1680, 1000)

        # ---- 中央: 3Dビューア ----
        self.viewer = Viewer3D(self.doc, settings, self)
        self.setCentralWidget(self.viewer)
        # ---- 左: モデルツリー ----
        self.tree = ModelTree(self.doc)
        d1 = QtWidgets.QDockWidget("モデルツリー", self)
        d1.setObjectName("dock_tree")
        d1.setWidget(self.tree)
        d1.setMinimumWidth(280)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, d1)
        # ---- 右: プロパティ ----
        self.props = PropertyPanel(self.doc, self.fmt)
        d2 = QtWidgets.QDockWidget("プロパティ", self)
        d2.setObjectName("dock_props")
        d2.setWidget(self.props)
        d2.setMinimumWidth(380)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d2)
        # ---- 下: ログ ----
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        d3 = QtWidgets.QDockWidget("ログ / 警告", self)
        d3.setObjectName("dock_log")
        d3.setWidget(self.log_view)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d3)
        self.log_dock = d3
        self.resizeDocks([d3], [140], QtCore.Qt.Vertical)
        self._bridge = _LogBridge()
        self._bridge.message.connect(self._append_log, QtCore.Qt.QueuedConnection)
        self._log_handler = CallbackHandler(lambda lv, m: self._bridge.message.emit(lv, m))
        logging.getLogger().addHandler(self._log_handler)

        self._build_statusbar()
        self._build_actions()
        self._build_menus_and_ribbon()

        # ---- 接続 ----
        self.tree.selectionIdsChanged.connect(self._tree_selection)
        self.tree.requestDelete.connect(self.cmd_delete_ids)
        self.tree.requestActivateCS.connect(self.cmd_activate_cs_id)
        self.tree.requestZoom.connect(self.zoom_to)
        self.tree.requestColor.connect(self.change_color)
        self.viewer.selectionChanged.connect(self._update_status)
        self.viewer.statusMessage.connect(self.set_guide)
        self.runner.started.connect(self._task_started)
        self.runner.progress.connect(self._task_progress)
        self.runner.ended.connect(self._task_ended)
        self.doc.undo_stack.listeners.append(self._update_undo)
        self.doc.listeners.append(self._doc_event)
        self._update_undo()
        self._update_status()
        self.set_guide("右ドラッグ: 回転 / 中ドラッグ: パン / ホイール: ズーム / 左ドラッグ: 選択ツール未使用時は回転。"
                       "まず「ファイル > メッシュ/点群をインポート」でスキャンデータを読み込んでください。")
        log.info("%s %s 起動", __app_name__, __version__)

    # ------------------------------------------------------------------
    # 共通ヘルパ
    # ------------------------------------------------------------------
    def fmt(self) -> NumberFormat:
        return NumberFormat(self.settings.length_decimals, self.settings.angle_decimals)

    def info(self, title, text):
        QtWidgets.QMessageBox.information(self, title, text)

    def warn(self, title, text):
        log.warning("%s: %s", title, text)
        QtWidgets.QMessageBox.warning(self, title, text)

    def error(self, title, text, details=""):
        log.error("%s: %s", title, text)
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, title, text, parent=self)
        if details:
            box.setDetailedText(details)
        box.exec()

    def confirm(self, title, text) -> bool:
        r = QtWidgets.QMessageBox.question(self, title, text,
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        return r == QtWidgets.QMessageBox.Yes

    def set_guide(self, text):
        self.guide_label.setText(text)

    def _start_dir(self):
        d = self.settings.last_dir
        return d if d and os.path.isdir(d) else os.path.expanduser("~")

    def ask_open(self, title, filt, multi=False):
        if multi:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, self._start_dir(), filt)
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, self._start_dir(), filt)
            paths = [p] if p else []
        if paths:
            self.settings.last_dir = os.path.dirname(paths[0])
        return paths if multi else (paths[0] if paths else None)

    def ask_save(self, title, filt, default_name):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(self._start_dir(), default_name), filt)
        if p:
            self.settings.last_dir = os.path.dirname(p)
        return p or None

    def run(self, title, fn, *args, on_done=None, **kwargs) -> bool:
        if self.runner.busy:
            self.warn("処理中", "別の処理を実行中です。完了またはキャンセルしてから実行してください。")
            return False

        def on_error(msg, tb):
            self.error(f"{title} に失敗しました", msg, tb)

        def on_cancel():
            self.set_guide(f"{title}: キャンセルしました")
            log.info("%s をキャンセルしました", title)

        return self.runner.run(title, fn, *args, on_done=on_done, on_error=on_error, on_cancel=on_cancel, **kwargs)

    def selected_objects(self, kinds=None):
        objs = [self.doc.get(i) for i in self.tree.selected_ids()]
        objs = [o for o in objs if o is not None]
        if kinds:
            kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
            objs = [o for o in objs if o.kind in kinds]
        return objs

    def target_scan(self, kinds=("mesh", "points"), ask=True):
        """操作対象のメッシュ/点群: ツリー選択 → 選択領域の持ち主 → 唯一のメッシュ."""
        o = self._target_scan(kinds, ask)
        if o is not None:
            n = len(o.F) if o.kind == "mesh" else len(o.P) if o.kind == "points" else 1
            if n == 0:
                if ask:
                    self.warn("空のデータ", f"{o.name} には面/点がありません。")
                return None
        return o

    def _target_scan(self, kinds, ask):
        objs = self.selected_objects(kinds)
        if objs:
            return objs[0]
        for oid in self.viewer.selection:
            o = self.doc.get(oid)
            if o is not None and o.kind in kinds:
                return o
        cands = self.doc.objects(kinds)
        if len(cands) == 1:
            return cands[0]
        if ask:
            if not cands:
                self.warn("対象なし", "対象となる" + "/".join(KIND_LABELS[k] for k in kinds) + "がありません。")
            else:
                self.warn("対象を選択", "モデルツリーで対象の" + "/".join(KIND_LABELS[k] for k in kinds) + "を選択してください。")
        return None

    def feature_meta(self, source_ids, params=None):
        names, files = [], []
        for oid in source_ids:
            o = self.doc.get(oid)
            if o is not None:
                names.append(o.name)
                sp = getattr(o, "source_path", "")
                if sp:
                    files.append(os.path.basename(sp))
        import datetime as _dt
        return {"source_object_name": ", ".join(names), "source_ids": list(source_ids),
                "source_file": ", ".join(files), "frame_name": self.doc.active_frame_name,
                "created": _dt.datetime.now().isoformat(timespec="seconds"), "params": params or {}}

    def add_object(self, obj, label=None, select=True):
        self.doc.add(obj, label)
        if select:
            self.tree.select_ids([obj.id])
        return obj

    def zoom_to(self, oid):
        o = self.doc.get(oid)
        if o is None:
            return
        pts = None
        if o.kind == "mesh" and len(o.V):
            pts = np.vstack([o.V.min(0), o.V.max(0)])
        elif o.kind == "points" and len(o.P):
            pts = np.vstack([o.P.min(0), o.P.max(0)])
        else:
            b = o.bounds_points()
            if b is not None:
                pts = np.atleast_2d(b)
        if pts is None:
            return
        lo, hi = pts.min(0), pts.max(0)
        pad = max(float(np.max(hi - lo)) * 0.3, 10.0)
        bounds = [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad]
        self.viewer.plotter.reset_camera(bounds=bounds)
        self.viewer.render()

    def change_color(self, oid):
        o = self.doc.get(oid)
        if o is None:
            return
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*o.color), self, "色の変更")
        if c.isValid():
            col = (c.redF(), c.greenF(), c.blueF())
            self.doc.modify(oid, lambda ob: setattr(ob, "color", col), "色の変更")

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        sb = self.statusBar()
        self.guide_label = QtWidgets.QLabel("")
        self.guide_label.setMinimumWidth(400)
        sb.addWidget(self.guide_label, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.cancel_btn = QtWidgets.QPushButton("キャンセル")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self.runner.cancel)
        self.sel_label = QtWidgets.QLabel("")
        self.model_label = QtWidgets.QLabel("")
        self.time_label = QtWidgets.QLabel("")
        self.cs_label = QtWidgets.QLabel("")
        self.warn_btn = QtWidgets.QPushButton("警告 0")
        self.warn_btn.setFlat(True)
        self.warn_btn.clicked.connect(lambda: (self.log_dock.show(), self.log_dock.raise_()))
        for w in (self.progress, self.cancel_btn, self.sel_label, self.model_label, self.time_label,
                  self.cs_label, self.warn_btn):
            sb.addPermanentWidget(w)

    def _act(self, key, text, slot, shortcut=None, tip=None, checkable=False, checked=False):
        a = QtGui.QAction(text, self)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
            a.setShortcutContext(QtCore.Qt.WindowShortcut)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        if checkable:
            a.setChecked(checked)
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda _=False, f=slot: self._safe(f))
        self.actions[key] = a
        return a

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log.exception("コマンド実行エラー")
            self.error("エラー", str(e), traceback.format_exc())

    def _build_actions(self):
        A = self._act
        s = self.settings
        # ファイル
        A("new", "新規プロジェクト", self.cmd_new, "Ctrl+N")
        A("open", "プロジェクトを開く…", self.cmd_open_project, "Ctrl+O")
        A("save", "プロジェクトを保存", self.cmd_save, "Ctrl+S")
        A("save_as", "名前を付けて保存…", self.cmd_save_as, "Ctrl+Shift+S")
        A("import", "メッシュ/点群をインポート…", self.cmd_import_mesh, "Ctrl+I",
          "STL / OBJ / PLY を読み込みます（Revo Scan の出力など）")
        A("import_step", "STEP をインポート…", self.cmd_import_step)
        A("samples", "サンプルデータを生成して開く", self.cmd_open_samples,
          tip="真値が既知のテスト用ブラケット・円筒部品などを生成して読み込みます")
        A("exit", "終了", self.close, "Ctrl+Q")
        # 編集
        A("undo", "元に戻す", self.cmd_undo, "Ctrl+Z")
        A("redo", "やり直し", self.cmd_redo, "Ctrl+Y")
        A("delete", "削除", self.cmd_delete, "Delete")
        A("rename", "名前の変更", self.cmd_rename, "F2")
        A("tool_nav", "ナビゲーション", self.cmd_tool_nav, "Esc", "選択ツールを解除（左ドラッグで回転）")
        A("tool_rect", "矩形選択", self.cmd_tool_rect, "R", "左ドラッグの矩形で頂点/面を選択（前面のみ: 設定）")
        A("tool_smart", "スマート選択", self.cmd_tool_smart, "G", "クリックした面から滑らかに連続する領域を選択")
        A("tool_brush", "ブラシ選択", self.cmd_tool_brush, "B", "ドラッグで半径内を選択")
        A("sel_clear", "選択解除", self.cmd_clear_selection, "Ctrl+D")
        A("sel_grow", "選択拡大", self.cmd_grow_selection, "Ctrl+Up")
        A("sel_shrink", "選択縮小", self.cmd_shrink_selection, "Ctrl+Down")
        A("sel_invert", "選択反転", self.cmd_invert_selection, "Ctrl+Shift+I")
        A("sel_all", "全面を選択", self.cmd_select_all, "Ctrl+A")
        # 表示
        for key, label, sc in (("front", "正面", "1"), ("back", "背面", "Ctrl+1"), ("left", "左側面", "3"),
                               ("right", "右側面", "Ctrl+3"), ("top", "上面", "7"), ("bottom", "下面", "Ctrl+7"),
                               ("iso", "アイソメ", "0")):
            A("view_" + key, label, lambda k=key: self.viewer.set_view(k), sc)
        A("view_fit", "全体表示", self.viewer.fit_view, "F")
        self.mode_group = QtGui.QActionGroup(self)
        for key, label in (("shaded", "シェーディング"), ("edges", "シェーディング+エッジ"),
                           ("wire", "ワイヤーフレーム"), ("points", "点群表示")):
            a = A("mode_" + key, label, lambda on, k=key: on and self.viewer.set_display_mode(k),
                  checkable=True, checked=(key == "shaded"))
            self.mode_group.addAction(a)
        A("transparent", "透明表示", self.viewer.set_transparent, "T", checkable=True)
        A("normals", "法線表示", self.viewer.set_show_normals, checkable=True)
        A("grid", "グリッド表示", self._toggle_grid, checkable=True, checked=s.show_grid)
        A("waxes", "ワールド座標軸表示", self._toggle_waxes, checkable=True, checked=s.show_world_axes)
        A("parallel", "平行投影", self._toggle_parallel, "P", checkable=True, checked=s.parallel_projection)
        A("clip", "断面表示（クリッピング）…", self.cmd_clip_view)
        A("art_hi", "人工面（穴埋め）を強調", self._toggle_art, checkable=True, checked=s.highlight_artificial)
        A("colors", "スキャンカラー表示切替", self.cmd_toggle_colors)
        A("dev_clear", "偏差カラーマップを消去", self.viewer.clear_deviation)
        A("show_all", "すべて表示", self.cmd_show_all)
        self._build_scan_actions(A)
        self._build_measure_actions(A)
        self._build_fit_actions(A)
        self._build_cad_actions(A)
        # 設定
        A("settings", "設定…", self.cmd_settings, "Ctrl+,")
        A("log_folder", "ログフォルダを開く", self.cmd_open_log_folder)
        A("shortcuts", "操作ガイド", self.cmd_shortcuts, "F1")
        A("licenses", "ライブラリとライセンス", self.cmd_licenses)
        A("about", "バージョン情報", self.cmd_about)

    MENU_LAYOUT = [
        ("ファイル", ["new", "open", "save", "save_as", None, "import", "import_step", "samples", None, "@recent", None, "exit"]),
        ("編集", ["undo", "redo", None, "delete", "rename", None, "tool_nav", "tool_rect", "tool_smart", "tool_brush", None,
                "sel_clear", "sel_grow", "sel_shrink", "sel_invert", "sel_all"]),
        ("表示", ["view_front", "view_back", "view_left", "view_right", "view_top", "view_bottom", "view_iso", "view_fit", None,
                "mode_shaded", "mode_edges", "mode_wire", "mode_points", None, "transparent", "normals", "grid", "waxes",
                "parallel", "art_hi", "colors", None, "clip", "dev_clear", "show_all"]),
        ("スキャンデータ", ["mesh_del_sel", "mesh_keep_sel", "mesh_noise", "mesh_smooth", "mesh_decimate", "mesh_fix_normals",
                     "mesh_flip", "mesh_fill", None, "mesh_split", "mesh_separate", "mesh_merge", "mesh_restore", None,
                     "pc_outlier", "pc_normals", "pc_poisson", "mesh_to_points", None, "rescale"]),
        ("測定", ["m_dist", "m_angle", "m_coord", "m_point_plane", "m_between", "m_projected", "m_bbox", "m_diameter", None,
                "m_scale", None, "mount_points"]),
        ("位置合わせ", ["cs_activate", "cs_world", None, "cs_3pt", "cs_plane_axis", "cs_321", "cs_axis_point", "cs_numeric", None,
                   "align_to_cs", None, "reg_points", "reg_icp", "reg_eval"]),
        ("形状認識", ["fit_plane", "fit_cylinder", "fit_sphere", "fit_circle", None, "hole_cyl", "hole_section", "hole_auto", None,
                  "pt_create", "pt_3planes", "axis_feature", "axis_2pt", "axis_2planes", None, "fit_options"]),
        ("CAD再構築", ["sec_axis", "sec_plane", "sec_series", "sec_sketch", "sec_spline", None, "cad_features", "cad_extrude",
                    "cad_revolve", "cad_loft", "cad_bspline", "cad_boolean", None, "cad_deviation", "feat_deviation"]),
        ("エクスポート", ["exp_step", "exp_dxf", "exp_csv", "exp_csv_meas", "exp_dist", "exp_json", "exp_mesh", "exp_points_dxf"]),
        ("設定", ["settings", None, "shortcuts", "log_folder", "licenses", "about"]),
    ]

    RIBBON = {
        "ファイル": ["import", "open", "save", "samples", "import_step"],
        "編集": ["undo", "redo", "|", "tool_nav", "tool_rect", "tool_smart", "tool_brush", "|", "sel_clear", "sel_grow",
               "sel_shrink", "sel_invert", "|", "delete"],
        "表示": ["view_fit", "view_front", "view_top", "view_right", "view_iso", "|", "mode_shaded", "mode_edges", "mode_wire",
               "mode_points", "|", "transparent", "normals", "grid", "parallel", "clip"],
        "スキャンデータ": ["mesh_del_sel", "mesh_keep_sel", "mesh_noise", "mesh_smooth", "mesh_decimate", "mesh_fill",
                    "mesh_fix_normals", "|", "mesh_separate", "mesh_merge", "mesh_restore", "|", "pc_outlier", "pc_normals"],
        "測定": ["m_dist", "m_angle", "m_coord", "m_point_plane", "m_between", "m_projected", "m_bbox", "m_diameter", "|",
               "m_scale", "mount_points"],
        "位置合わせ": ["cs_3pt", "cs_plane_axis", "cs_321", "cs_numeric", "cs_activate", "cs_world", "|", "align_to_cs", "|",
                  "reg_points", "reg_icp", "reg_eval"],
        "形状認識": ["tool_smart", "tool_rect", "|", "fit_plane", "fit_cylinder", "fit_sphere", "fit_circle", "|", "hole_cyl",
                 "hole_section", "hole_auto", "|", "pt_create", "axis_feature", "fit_options"],
        "CAD再構築": ["sec_axis", "sec_series", "sec_plane", "sec_sketch", "|", "cad_features", "cad_extrude", "cad_revolve",
                   "cad_loft", "cad_bspline", "cad_boolean", "|", "cad_deviation"],
        "エクスポート": ["exp_step", "exp_dxf", "exp_csv", "exp_csv_meas", "exp_json", "exp_mesh"],
        "設定": ["settings", "shortcuts", "licenses", "about"],
    }

    def _build_menus_and_ribbon(self):
        mb = self.menuBar()
        for title, keys in self.MENU_LAYOUT:
            m = mb.addMenu(title)
            for k in keys:
                if k is None:
                    m.addSeparator()
                elif k == "@recent":
                    self.recent_menu = m.addMenu("最近使ったファイル")
                    self._update_recent_menu()
                else:
                    m.addAction(self.actions[k])
        # リボン（カテゴリ別ツールバー）
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        for title, keys in self.RIBBON.items():
            tb = QtWidgets.QToolBar()
            tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            for k in keys:
                if k == "|":
                    tb.addSeparator()
                else:
                    tb.addAction(self.actions[k])
            tabs.addTab(tb, title)
        tabs.setCurrentIndex(0)
        tabs.setMaximumHeight(72)
        rib = QtWidgets.QToolBar("リボン")
        rib.setObjectName("ribbon")
        rib.setMovable(False)
        rib.addWidget(tabs)
        self.addToolBar(QtCore.Qt.TopToolBarArea, rib)
        self.ribbon = tabs

    def _update_recent_menu(self):
        self.recent_menu.clear()
        for p in self.settings.recent_files:
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda _=False, path=p: self._open_recent(path))
        if not self.settings.recent_files:
            a = self.recent_menu.addAction("（なし）")
            a.setEnabled(False)

    def _open_recent(self, path):
        if not os.path.exists(path):
            self.warn("ファイルがありません", path)
            return
        if path.lower().endswith(".ljmre"):
            self.open_project(path)
        else:
            self.import_files([path])

    # ------------------------------------------------------------------
    # ステータス・ログ
    # ------------------------------------------------------------------
    def _append_log(self, level, msg):
        self.log_view.appendPlainText(msg)
        if level >= logging.WARNING:
            self._warn_count += 1
            self.warn_btn.setText(f"警告 {self._warn_count}")
            self.warn_btn.setStyleSheet("color: #f0b35a;")

    def _update_status(self):
        nf, npt = self.viewer.selection_count()
        self.sel_label.setText(f"選択: {nf:,} 面 / {npt:,} 点")
        meshes = self.doc.objects("mesh")
        clouds = self.doc.objects("points")
        nface = sum(len(o.F) for o in meshes)
        self.model_label.setText(f"メッシュ {len(meshes)} ({nface:,} 面) / 点群 {len(clouds)} / 要素 {len(self.doc)}")
        self.cs_label.setText(f"作業座標系: {self.doc.active_frame_name}")

    def _doc_event(self, event, ids):
        self._update_status()
        title = f"{__app_name__} {__version__}"
        if self.doc.path:
            title += " - " + os.path.basename(self.doc.path)
        if self.doc.modified:
            title += " *"
        self.setWindowTitle(title)

    def _update_undo(self):
        st = self.doc.undo_stack
        self.actions["undo"].setEnabled(st.can_undo)
        self.actions["redo"].setEnabled(st.can_redo)
        self.actions["undo"].setText("元に戻す" + (f"：{st.undo_label}" if st.can_undo else ""))
        self.actions["redo"].setText("やり直し" + (f"：{st.redo_label}" if st.can_redo else ""))
        self._doc_event("", [])

    def _task_started(self, title):
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_btn.show()
        self.set_guide(f"{title} を実行中…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)

    def _task_progress(self, f, msg):
        self.progress.setValue(int(f * 1000))
        if msg:
            self.set_guide(msg)

    def _task_ended(self, title, ok, secs):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.progress.hide()
        self.cancel_btn.hide()
        if ok:
            self.time_label.setText(f"処理時間: {secs:.2f} 秒")
            log.info("%s 完了 (%.2f 秒)", title, secs)

    def _tree_selection(self, ids):
        self.props.show_object(ids[0] if ids else None)
        self.viewer.set_highlight(ids)

    # ------------------------------------------------------------------
    # ファイル
    # ------------------------------------------------------------------
    def maybe_save(self) -> bool:
        if not self.doc.modified or len(self.doc) == 0:
            return True
        r = QtWidgets.QMessageBox.question(
            self, "保存の確認", "変更が保存されていません。保存しますか？",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.cmd_save()
        return r == QtWidgets.QMessageBox.Discard

    def set_document(self, doc):
        try:
            self.doc.listeners.remove(self._doc_event)
        except ValueError:
            pass
        self.doc = doc
        doc.undo_stack.listeners.append(self._update_undo)
        doc.listeners.append(self._doc_event)
        self.viewer.set_document(doc)
        self.tree.set_document(doc)
        self.props.set_document(doc)
        self._update_undo()
        self._update_status()

    def cmd_new(self):
        if not self.maybe_save():
            return
        self.viewer.set_tool(None)
        self.set_document(Document())

    def cmd_open_project(self):
        if not self.maybe_save():
            return
        p = self.ask_open("プロジェクトを開く", PROJECT_FILTER)
        if p:
            self.open_project(p)

    def open_project(self, path):
        from ..model.project_io import load_project

        def work(progress, p):
            progress.report(0.1, "プロジェクトを読込中")
            return load_project(p)

        def done(res):
            doc, saved = res
            self.set_document(doc)
            self.viewer.fit_view()
            self.settings.add_recent(path)
            self._update_recent_menu()
            self.set_guide(f"プロジェクトを開きました: {path}")
        self.run("プロジェクトを開く", work, path, on_done=done)

    def cmd_save(self) -> bool:
        if not self.doc.path:
            return self.cmd_save_as()
        return self.save_project(self.doc.path)

    def cmd_save_as(self) -> bool:
        p = self.ask_save("名前を付けて保存", PROJECT_FILTER, "project.ljmre")
        if not p:
            return False
        if not p.lower().endswith(".ljmre"):
            p += ".ljmre"
        return self.save_project(p)

    def save_project(self, path) -> bool:
        from ..model.project_io import save_project
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            save_project(self.doc, path, self.settings.to_dict())
        except Exception as e:  # noqa: BLE001
            self.error("保存に失敗しました", str(e), traceback.format_exc())
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.settings.add_recent(path)
        self._update_recent_menu()
        self._doc_event("", [])
        self.set_guide(f"保存しました: {path}")
        log.info("プロジェクト保存: %s", path)
        return True

    def cmd_import_mesh(self):
        paths = self.ask_open("メッシュ/点群をインポート", MESH_FILTER, multi=True)
        if paths:
            self.import_files(paths)

    def import_files(self, paths, unit=None, ask_unit=True):
        from ..io.mesh_io import load_mesh_file

        def work(progress, ps):
            out = []
            for i, p in enumerate(ps):
                progress.report(i / len(ps), f"読込中: {os.path.basename(p)}")
                out.append(load_mesh_file(p))
            return out

        def done(datas):
            for data in datas:
                u = unit or data.unit_guess
                if ask_unit and unit is None:
                    dlg = ImportUnitDialog(data, self)
                    if dlg.exec() != QtWidgets.QDialog.Accepted:
                        continue
                    u = dlg.unit()
                self._add_loaded(data, u)
            self.viewer.fit_view()
        self.run("インポート", work, list(paths), on_done=done)

    def _add_loaded(self, data, unit):
        from ..io.mesh_io import UNIT_TO_MM
        s = UNIT_TO_MM[unit]
        name = os.path.splitext(os.path.basename(data.source_path))[0]
        V = data.vertices * s
        if data.kind == "mesh":
            o = MeshObject(self.doc.unique_name(name), V, data.faces, data.vertex_colors)
        else:
            o = PointCloudObject(self.doc.unique_name(name), V, data.vertex_normals, data.vertex_colors)
        o.source_path = data.source_path
        o.source_sha256 = data.sha256
        o.source_unit = unit
        o.unit_scale = s
        o.meta["source_file"] = os.path.basename(data.source_path)
        o.meta["import_warnings"] = data.warnings
        self.doc.add(o, f"{name} をインポート")
        for w in data.warnings:
            log.warning("%s: %s", name, w)
        self.settings.add_recent(data.source_path)
        self._update_recent_menu()
        log.info("インポート: %s (%s, 単位 %s, %d 頂点, %d 面)", data.source_path, data.kind, unit,
                 data.n_vertices, data.n_faces)
        self.tree.select_ids([o.id])
        return o

    def cmd_import_step(self):
        from ..cad import occ_backend as occ
        if not occ.available():
            self.warn("CADカーネルなし", "OCP (OpenCASCADE) がインストールされていません。\n" + occ.import_error())
            return
        p = self.ask_open("STEP をインポート", "STEP (*.step *.stp)")
        if not p:
            return

        def work(progress, path):
            shape = occ.read_step(path)
            return shape, occ.summarize(shape)

        def done(res):
            shape, summ = res
            o = CadObject(self.doc.unique_name(os.path.splitext(os.path.basename(p))[0]), shape,
                          {"type": "step_import", "path": p})
            o.summary = summ
            self.add_object(o, "STEP インポート")
            self.viewer.fit_view()
        self.run("STEP インポート", work, p, on_done=done)

    def cmd_open_samples(self):
        from ..core import synthetic as S
        from ..io.mesh_io import export_mesh
        import json
        d = os.path.join(tempfile.gettempdir(), "LJM_RE_samples")
        os.makedirs(d, exist_ok=True)
        choice = ParamDialog.ask(self, "サンプルデータ", [
            ("model", "モデル", "choice", "bracket", [("bracket", "穴が複数あるブラケット"),
                                                    ("multi", "複数円筒の部品（段付きシャフト）"),
                                                    ("plate", "穴径20mmのプレート"),
                                                    ("plane", "100×100mmの平面"),
                                                    ("cylinder", "直径50mmの円筒"),
                                                    ("partial", "部分的に欠損した円筒(120°)"),
                                                    ("noisy", "ノイズ付き点群（ブラケット, σ=0.05mm）")]),
        ], f"真値が既知のテスト用データを生成して読み込みます。\n保存先: {d}")
        if not choice:
            return
        m = choice["model"]
        truth = {}
        if m == "bracket":
            mesh, truth = S.make_bracket()
        elif m == "multi":
            mesh, truth = S.make_multi_cylinder_part()
        elif m == "plate":
            mesh = S.make_plate_with_holes()
            truth = {"hole": {"center": [0, 0, 10], "diameter": 20}}
        elif m == "plane":
            mesh = S.make_plane_mesh()
        elif m == "cylinder":
            mesh = S.make_tube_mesh(50, 100, 128, 20)
        elif m == "partial":
            mesh = S.make_tube_mesh(50, 60, 128, 12, arc_deg=120)
        else:
            mesh, truth = S.make_bracket()
            pts, nrm = S.sample_surface(mesh, 60000, 0.05, 1)
            path = os.path.join(d, "sample_noisy_points.ply")
            export_mesh(path, pts, None)
            self.import_files([path], unit="mm", ask_unit=False)
            return
        path = os.path.join(d, f"sample_{m}.stl")
        export_mesh(path, mesh.vertices, mesh.faces)
        if truth:
            with open(os.path.join(d, f"sample_{m}_truth.json"), "w", encoding="utf-8") as f:
                json.dump(truth, f, ensure_ascii=False, indent=1)
        self.import_files([path], unit="mm", ask_unit=False)

    # ------------------------------------------------------------------
    # 編集
    # ------------------------------------------------------------------
    def cmd_undo(self):
        lab = self.doc.undo()
        if lab:
            self.set_guide(f"元に戻しました: {lab}")

    def cmd_redo(self):
        lab = self.doc.redo()
        if lab:
            self.set_guide(f"やり直しました: {lab}")

    def cmd_delete(self):
        ids = self.tree.selected_ids()
        if ids:
            self.cmd_delete_ids(ids)
        elif self.viewer.selection:
            self.cmd_mesh_delete_selected()

    def cmd_delete_ids(self, ids):
        if not ids:
            return
        names = ", ".join(self.doc.get(i).name for i in ids[:5] if self.doc.get(i)) + (" …" if len(ids) > 5 else "")
        if any(self.doc.get(i) is not None and self.doc.get(i).kind in ("mesh", "points") for i in ids):
            if not self.confirm("削除の確認", f"次のオブジェクトを削除しますか？（元に戻す で復元できます）\n{names}"):
                return
        self.doc.remove(ids, f"{len(ids)} 個を削除")

    def cmd_rename(self):
        ids = self.tree.selected_ids()
        if not ids:
            return
        o = self.doc.get(ids[0])
        name, ok = QtWidgets.QInputDialog.getText(self, "名前の変更", "新しい名前:", text=o.name)
        if ok and name.strip():
            self.doc.rename(o.id, name.strip())

    def cmd_show_all(self):
        for o in self.doc.objects():
            self.doc.set_visible(o.id, True)

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------
    def _toggle_grid(self, on):
        self.settings.show_grid = bool(on)
        self.viewer.set_show_grid(on)

    def _toggle_waxes(self, on):
        self.settings.show_world_axes = bool(on)
        self.viewer.set_show_world_axes(on)

    def _toggle_parallel(self, on):
        self.settings.parallel_projection = bool(on)
        self.viewer.set_parallel(on)

    def _toggle_art(self, on):
        self.settings.highlight_artificial = bool(on)
        self.viewer.rebuild_all()

    def cmd_toggle_colors(self):
        o = self.target_scan()
        if o is None:
            return
        if o.colors is None:
            self.info("スキャンカラー", "このデータには頂点カラーがありません。")
            return
        self.doc.set_property(o.id, "show_colors", not o.show_colors, "カラー表示切替")

    def cmd_clip_view(self):
        if self._clip_dialog is None or not self._clip_dialog.isVisible():
            self._clip_dialog = ClipDialog(self.viewer, self.doc, self)
        self._clip_dialog.show()
        self._clip_dialog.raise_()

    # ------------------------------------------------------------------
    # 設定・ヘルプ
    # ------------------------------------------------------------------
    def cmd_settings(self):
        s = self.settings
        v = ParamDialog.ask(self, "設定", [
            ("length_decimals", "長さの表示桁数（小数点以下）", "int", s.length_decimals, {"min": 0, "max": 8}),
            ("angle_decimals", "角度の表示桁数（小数点以下）", "int", s.angle_decimals, {"min": 0, "max": 8}),
            ("lod_max_faces", "表示LODを使う面数の閾値", "int", s.lod_max_faces, {"min": 10000, "max": 100_000_000}),
            ("smart_angle_deg", "スマート選択: 隣接面の角度許容", "float", s.smart_angle_deg, {"min": 0.1, "max": 90, "suffix": "°"}),
            ("smart_seed_dev_deg", "スマート選択: 開始面からの最大偏差 (0=無制限)", "float", s.smart_seed_dev_deg, {"min": 0, "max": 180, "suffix": "°"}),
            ("brush_radius_mm", "ブラシ半径", "float", s.brush_radius_mm, {"min": 0.01, "max": 1e4, "suffix": "mm"}),
            ("rect_front_only", "矩形選択は前面に見える部分のみ", "bool", s.rect_front_only, None),
            ("exclude_artificial_in_fit", "フィット・選択で人工面（穴埋め）を除外", "bool", s.exclude_artificial_in_fit, None),
            ("sketch_tol_mm", "断面スケッチ化の許容誤差", "float", s.sketch_tol_mm, {"min": 0.001, "max": 10, "suffix": "mm"}),
            ("tess_deflection_mm", "CAD表示のテッセレーション精度", "float", s.tess_deflection_mm, {"min": 0.001, "max": 5, "suffix": "mm"}),
            ("deviation_tol_mm", "偏差解析の許容値", "float", s.deviation_tol_mm, {"min": 0.001, "max": 100, "suffix": "mm"}),
            ("theme", "テーマ（再起動後に反映）", "choice", s.theme, [("dark", "ダーク"), ("light", "ライト")]),
        ], "表示桁数は表示上の丸めであり、測定精度を表すものではありません。")
        if not v:
            return
        for k, val in v.items():
            setattr(s, k, val)
        s.save()
        self.viewer.rebuild_all()
        self.props.show_object(self.props.oid)

    def cmd_open_log_folder(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(log_dir()))

    def cmd_shortcuts(self):
        text = """■ マウス操作
  右ドラッグ: 回転（常に有効） / 中ドラッグ: パン / ホイール: ズーム
  左ドラッグ: 選択ツール未使用時は回転、選択ツール使用時は選択

■ 選択ツール（形状認識の前に、フィットしたい面を選択します）
  R: 矩形選択   G: スマート選択（クリックした面から連続する滑らかな領域）   B: ブラシ選択
  Shift+クリック: 追加   Ctrl+クリック: 解除   Esc: ツール解除   Ctrl+D: 選択解除

■ 基本ワークフロー
  1. ファイル > メッシュ/点群をインポート（単位を確認）
  2. G キー → 平面にしたい面をクリック → 形状認識 > 平面フィット
  3. G キー → 円筒面（ボルト穴の内壁など）をクリック → 円筒フィット / 穴中心（内壁）
  4. 位置合わせ > 座標系（3平面 / 平面+軸 / 3点）→ 作業座標系に設定
  5. 測定 > 2点間距離 など（特徴点の近くをクリックするとスナップ）
  6. エクスポート > CSV（座標一覧）/ ファイル > プロジェクトを保存

■ 表示
  F: 全体表示  1/3/7/0: 正面/右/上/アイソメ（Ctrl で反対側）  T: 透明  P: 平行投影
  Ctrl+Z / Ctrl+Y: 元に戻す / やり直し   F2: 名前の変更   Delete: 削除
"""
        ReportDialog("操作ガイド", text, self).exec()

    def cmd_licenses(self):
        from ..utils.versions import library_report
        ReportDialog("ライブラリとライセンス", library_report(), self).exec()

    def cmd_about(self):
        self.info("バージョン情報", f"{__app_name__}\nバージョン {__version__}\n\n"
                  "3Dスキャンデータ（Revopoint POP 4 / Revo Scan 6 出力の STL/OBJ/PLY）からの\n"
                  "形状認識・寸法測定・CAD再構築を支援するソフトウェアです。\n\n"
                  "注意: 安全上重要な寸法（サスペンション取付点など）は必ず実測で確認してください。")

    # ------------------------------------------------------------------
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.cmd_tool_nav()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.runner.busy:
            if not self.confirm("処理中", "処理を実行中です。キャンセルして終了しますか？"):
                e.ignore()
                return
            self.runner.cancel()
            self.runner.pool.waitForDone(5000)
        if not self.maybe_save():
            e.ignore()
            return
        self.settings.save()
        logging.getLogger().removeHandler(self._log_handler)
        self.viewer.close_viewer()
        e.accept()
