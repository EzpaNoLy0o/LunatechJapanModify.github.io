"""メインウィンドウ: レイアウト、メニュー、CAD 読み込み、形状検査、プロジェクト保存。"""

from __future__ import annotations

import logging
import platform
import shutil
from datetime import datetime
from pathlib import Path

import pyvista as pv
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .. import APP_EDITION, APP_NAME, IMPLEMENTED_STAGE, __version__
from ..cad.import_job import SUPPORTED_SUFFIXES, load_cached_meta
from ..core import paths
from ..core.pose import Pose
from ..core.project import PROJECT_FILE, Project, ProjectError, load_project, save_project
from ..mesh.gmsh_job import load_report
from ..mesh.request import part_payloads
from ..mesh.settings import MeshSettings
from ..tunnel.spec import TunnelSpec
from .analysis_panels import MeshDisplayPanel, MeshPanel, MeshQualityView, TunnelPanel
from .job_runner import JobManager
from .panels import DisplayPanel, InspectionTable, LogPanel, ModelPanel, PartTree, scrollable
from .scene import Scene
from .viewport import MODE_FACE, MODE_MEASURE, MODE_PART, Viewport

logger = logging.getLogger("ljmcfd.gui")

CAD_FILTER = "CAD ファイル (*.step *.stp *.iges *.igs *.stl);;STEP (*.step *.stp);;IGES (*.iges *.igs);;STL (*.stl);;すべてのファイル (*)"
QUALITY_LABELS = {"coarse": "粗い (高速)", "standard": "標準", "fine": "高精細"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings(str(paths.workspace_root() / "settings.ini"), QSettings.IniFormat)
        self.scene = Scene(self)
        self.jobs = JobManager(self)
        self.project: Project = Project()
        self.project_dir: Path | None = None
        self._modified = False
        self._restoring = False
        self._pending_restore = 0
        self._restore_problems: list[str] = []
        self._restore_view: dict | None = None
        self._scratch_dir: Path | None = None     # 未保存プロジェクトの作業フォルダ
        self._mesh_job = None

        self.setAcceptDrops(True)
        self.resize(1600, 960)
        self._build_layout()
        self._build_actions()
        self._build_status_bar()
        self._connect()
        self._restore_window_state()
        self._update_title()
        logger.info(
            "%s %s v%s を起動しました (Stage %d ビルド)", APP_NAME, APP_EDITION, __version__, IMPLEMENTED_STAGE
        )
        logger.info("作業フォルダ: %s", paths.workspace_root())

    # ------------------------------------------------------------------ レイアウト
    def _build_layout(self) -> None:
        self.viewport = Viewport(self.scene)
        self.part_tree = PartTree(self.scene)
        self.model_panel = ModelPanel(self.scene)
        self.display_panel = DisplayPanel(self.scene, self.viewport)
        self.log_panel = LogPanel()
        self.inspection_table = InspectionTable(self.scene)

        # 左: パーツツリー (上) + 設定タブ (下)
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(scrollable(self.model_panel), "モデル")
        self.tunnel_panel = TunnelPanel(self.scene)
        self.mesh_panel = MeshPanel(self.scene)
        self.left_tabs.addTab(scrollable(self.tunnel_panel), "風洞")
        self.left_tabs.addTab(scrollable(self.mesh_panel), "メッシュ")
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.addWidget(self._titled("モデル・パーツ", self.part_tree))
        self.left_splitter.addWidget(self.left_tabs)
        self.left_splitter.setSizes([260, 440])

        # 右: 可視化設定・結果
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(scrollable(self.display_panel), "表示・結果")
        self.mesh_display = MeshDisplayPanel(self.scene, self.viewport)
        self.right_tabs.addTab(scrollable(self.mesh_display), "風洞・メッシュ")

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.viewport)
        self.main_splitter.addWidget(self.right_tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([390, 880, 330])

        # 下: ログ・検査結果・メッシュ品質 (Stage 3 以降で残差グラフ・スライダーが加わる)
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.log_panel, "ログ")
        self.bottom_tabs.addTab(self.inspection_table, "形状検査")
        self.mesh_quality = MeshQualityView(self.scene)
        self.bottom_tabs.addTab(self.mesh_quality, "メッシュ品質")

        self.outer_splitter = QSplitter(Qt.Vertical)
        self.outer_splitter.addWidget(self.main_splitter)
        self.outer_splitter.addWidget(self.bottom_tabs)
        self.outer_splitter.setStretchFactor(0, 1)
        self.outer_splitter.setStretchFactor(1, 0)
        self.outer_splitter.setSizes([740, 190])
        for splitter in (self.left_splitter, self.main_splitter, self.outer_splitter):
            splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.outer_splitter)

    @staticmethod
    def _titled(title: str, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(title)
        label.setProperty("role", "headline")
        label.setContentsMargins(6, 4, 0, 0)
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return box

    def _action(self, text: str, slot, shortcut=None, tip: str = "", checkable: bool = False) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        action.setCheckable(checkable)
        if slot is not None:
            action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        a = self._action
        self.act_new = a("新規プロジェクト", self.new_project, "Ctrl+N")
        self.act_open = a("プロジェクトを開く…", self._open_project_dialog, "Ctrl+O")
        self.act_save = a("保存", self._save_project_dialog, "Ctrl+S", "プロジェクトを保存")
        self.act_save_as = a("名前を付けて保存…", lambda: self._save_project_dialog(True), "Ctrl+Shift+S")
        self.act_import = a("CAD 読み込み", self._import_dialog, "Ctrl+I", "STEP / IGES / STL を読み込む")
        self.act_screenshot = a("スクリーンショット", self._screenshot_dialog, "F12", "3D ビューを PNG で保存")
        self.act_quit = a("終了", self.close, "Ctrl+Q")
        self.act_fit = a("フィット", self.viewport.fit, "F", "モデル全体を表示")
        self.act_inspect = a("形状検査", self.inspect_selected, "Ctrl+Shift+I",
                             "選択パーツ (未選択なら全パーツ) の開放エッジ・不良面を検査")
        self.act_remove = a("選択パーツを削除", self._remove_selected, QKeySequence.Delete)
        self.act_clear_measure = a("計測をクリア", self.viewport.clear_measurement, "Esc")
        self.act_about = a("このビルドについて", self._show_about)

        view_actions = [
            a(text, lambda _=False, n=name: self.viewport.set_view(n), key, tip)
            for name, text, key, tip in (
                ("iso", "等角", "0", "斜め前方から"), ("front", "前", "1", "上流 (車両前方) から"),
                ("rear", "後", "2", "下流 (車両後方) から"), ("left", "左", "3", "車両左側面から"),
                ("right", "右", "4", "車両右側面から"), ("top", "上", "5", "真上から"),
                ("bottom", "下", "6", "真下 (床下) から"),
            )
        ]

        self.mode_group = QActionGroup(self)
        self.mode_actions = {}
        for mode, text, key, tip in (
            (MODE_PART, "パーツ選択", "Q", "クリックでパーツを選択 (Ctrl+クリックで追加)"),
            (MODE_FACE, "面選択", "W", "クリックで面を選択し、面積・法線を表示"),
            (MODE_MEASURE, "寸法計測", "E", "2 点をクリックして距離を計測"),
        ):
            action = a(text, lambda _=False, m=mode: self.viewport.set_mode(m), key, tip, checkable=True)
            self.mode_group.addAction(action)
            self.mode_actions[mode] = action
        self.mode_actions[MODE_PART].setChecked(True)

        self.quality_group = QActionGroup(self)
        current_quality = self.import_quality()
        self.quality_actions = {}
        for key, text in QUALITY_LABELS.items():
            action = a(text, lambda _=False, k=key: self.settings.setValue("import/quality", k), checkable=True)
            action.setChecked(key == current_quality)
            self.quality_group.addAction(action)
            self.quality_actions[key] = action

        menu = self.menuBar()
        file_menu = menu.addMenu("ファイル")
        for item in (self.act_new, self.act_open, self.act_save, self.act_save_as, None, self.act_import, None,
                     self.act_screenshot, None, self.act_quit):
            file_menu.addSeparator() if item is None else file_menu.addAction(item)
        view_menu = menu.addMenu("表示")
        view_menu.addAction(self.act_fit)
        view_menu.addSeparator()
        view_menu.addActions(view_actions)
        tool_menu = menu.addMenu("ツール")
        tool_menu.addActions(list(self.mode_actions.values()))
        tool_menu.addAction(self.act_clear_measure)
        tool_menu.addSeparator()
        tool_menu.addAction(self.act_inspect)
        tool_menu.addAction(self.act_remove)
        tool_menu.addSeparator()
        quality_menu = tool_menu.addMenu("CAD 読み込み品質 (次回の読み込みから)")
        quality_menu.addActions(list(self.quality_actions.values()))
        menu.addMenu("ヘルプ").addAction(self.act_about)

        bar = self.addToolBar("メイン")
        bar.setObjectName("main_toolbar")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for group in ((self.act_open, self.act_save, self.act_import), (self.act_fit, *view_actions),
                      tuple(self.mode_actions.values()), (self.act_inspect, self.act_screenshot)):
            bar.addActions(list(group))
            bar.addSeparator()

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.job_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        stage = QLabel(f"Stage {IMPLEMENTED_STAGE} ビルド  |  風向 +X  |  地面 Z=0")
        stage.setProperty("role", "dim")
        bar.addPermanentWidget(self.job_label)
        bar.addPermanentWidget(self.progress_bar)
        bar.addPermanentWidget(stage)
        bar.showMessage("CAD ファイルをドラッグ＆ドロップ、または「CAD 読み込み」で開始します。")

    def _connect(self) -> None:
        self.viewport.partClicked.connect(self._on_part_clicked)
        self.viewport.backgroundClicked.connect(self._on_background_clicked)
        self.part_tree.inspectRequested.connect(self.inspect_parts)
        self.part_tree.removeRequested.connect(self.scene.remove_parts)
        self.tunnel_panel.generateRequested.connect(self.generate_tunnel)
        self.mesh_panel.generateRequested.connect(self.generate_mesh)
        self.mesh_panel.stopRequested.connect(self.stop_mesh)
        self.scene.modified.connect(self._on_modified)
        self.jobs.progress.connect(self._on_job_progress)
        self.jobs.busyChanged.connect(self._on_busy_changed)
        self.jobs.logLine.connect(lambda level, text: (logger.error if level == "error" else logger.info)(text))

    # ------------------------------------------------------------------ ウィンドウ状態
    def _restore_window_state(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        for name in ("left_splitter", "main_splitter", "outer_splitter"):
            state = self.settings.value(f"window/{name}")
            if state is not None:
                getattr(self, name).restoreState(state)

    def _save_window_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        for name in ("left_splitter", "main_splitter", "outer_splitter"):
            self.settings.setValue(f"window/{name}", getattr(self, name).saveState())
        self.settings.sync()

    def _update_title(self) -> None:
        name = self.project_dir.name if self.project_dir else "無題"
        mark = " *" if self._modified else ""
        self.setWindowTitle(f"{name}{mark} — {APP_NAME} {APP_EDITION} v{__version__}")

    def _on_modified(self) -> None:
        if not self._restoring and not self._modified:
            self._modified = True
            self._update_title()

    def _set_clean(self) -> None:
        self._modified = False
        self._update_title()

    def import_quality(self) -> str:
        quality = str(self.settings.value("import/quality", "standard"))
        return quality if quality in QUALITY_LABELS else "standard"

    # ------------------------------------------------------------------ ジョブ表示
    def _on_job_progress(self, label: str, value: float, message: str) -> None:
        self.progress_bar.setValue(int(value * 1000))
        self.job_label.setText(f"{label}: {message}" if message else label)

    def _on_busy_changed(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        if not busy:
            self.progress_bar.setValue(0)
            self.job_label.setText("")

    # ------------------------------------------------------------------ 3D ビューのクリック
    def _on_part_clicked(self, part_id: str, face_id: int, ctrl: bool) -> None:
        if self.viewport.mode == MODE_FACE:
            self.scene.set_selection([part_id])
            self.scene.select_face(part_id, face_id)
        elif ctrl:
            self.scene.toggle_selection(part_id)
        else:
            self.scene.set_selection([part_id])

    def _on_background_clicked(self) -> None:
        self.scene.select_face(None)
        self.scene.set_selection([])

    # ------------------------------------------------------------------ CAD 読み込み
    def _import_dialog(self) -> None:
        start = str(self.settings.value("paths/cad", str(Path.home())))
        files, _ = QFileDialog.getOpenFileNames(self, "CAD ファイルを読み込む", start, CAD_FILTER)
        if files:
            self.settings.setValue("paths/cad", str(Path(files[0]).parent))
            self.import_files(files)

    def import_files(self, files) -> None:
        """CAD ファイルを読み込む (変換は別プロセス。完了するとシーンへ追加される)。"""
        for file in files:
            path = Path(file)
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                logger.error("未対応のファイル形式です: %s (対応: STEP / STP / IGES / IGS / STL)", path.name)
                continue
            self.jobs.submit(
                "cad-import", {"path": str(path), "quality": self.import_quality()},
                label=f"読み込み {path.name}",
                on_success=self._on_import_done,
                on_failure=lambda msg, detail, p=path: self._on_import_failed(p, msg, detail),
            )

    def _on_import_done(self, meta: dict) -> None:
        parts = self.scene.add_import(meta)
        if self.scene.tunnel.margin_front <= 0:
            self.scene.auto_size_tunnel()      # 最初の読み込み時に、風洞とメッシュの寸法をモデルへ合わせる
        name = Path(meta["source"]["path"]).name
        logger.info("%s: %d パーツを追加しました", name, len(parts))
        if meta["unit_source"] == "heuristic":
            logger.warning(
                "%s は単位情報を持たないため、寸法から単位を「%s」と推定しました。"
                "モデルタブの「単位」で必ず確認してください。", name, meta["geometry_unit"])
        elif meta["unit_source"] == "occt":
            logger.warning("%s から単位を検出できませんでした。CAD カーネルの換算結果 (mm) を使用します。", name)
        elif meta["file_unit"] != meta["geometry_unit"]:
            logger.info("%s: 元の単位 %s を %s へ自動換算しました", name, meta["file_unit"], meta["geometry_unit"])
        self.statusBar().showMessage(f"{name} を読み込みました ({len(parts)} パーツ)", 8000)

    def _on_import_failed(self, path: Path, message: str, detail: str) -> None:
        if detail:
            logger.error("詳細:\n%s", detail)
        QMessageBox.critical(self, "読み込み失敗", f"{path.name} を読み込めませんでした。\n\n{message}")

    def _remove_selected(self) -> None:
        if self.scene.selected_ids:
            self.scene.remove_parts(list(self.scene.selected_ids))

    # ------------------------------------------------------------------ 形状検査
    def inspect_selected(self) -> None:
        ids = list(self.scene.selected_ids) or [p.id for p in self.scene.parts]
        self.inspect_parts(ids)

    def inspect_parts(self, part_ids: list[str]) -> None:
        if not part_ids:
            self.statusBar().showMessage("検査するパーツがありません。", 5000)
            return
        self.bottom_tabs.setCurrentWidget(self.inspection_table)
        for part_id in part_ids:
            part = self.scene.assembly.get(part_id)
            self.jobs.submit(
                "cad-inspect",
                {"cache_dir": part.cache_dir, "surface": part.entry["surface"], "brep": part.entry["brep"],
                 "part_id": part.id},
                label=f"形状検査 {part.name}",
                on_success=self._on_inspect_done,
            )

    def _on_inspect_done(self, result: dict) -> None:
        part_id = result.get("part_id")
        try:
            part = self.scene.assembly.get(part_id)
        except KeyError:
            return   # 検査中に削除された
        self.scene.set_inspection(part_id, result)
        defects = None
        if result.get("defects"):
            defects = pv.read(Path(part.cache_dir) / result["defects"])
        self.viewport.show_defects(part_id, defects)
        summary = result["summary"]
        log = {"ok": logger.info, "warning": logger.warning, "error": logger.error}[summary["level"]]
        log("形状検査 [%s]: %s", part.name, " ".join(summary["messages"]))

    # ------------------------------------------------------------------ 仮想風洞
    def generate_tunnel(self) -> bool:
        """風洞を確定して 3D に表示し、前面投影面積 (閉塞率・基準面積用) を計算する。"""
        issues = self.scene.tunnel_issues()
        errors = [text for level, text in issues if level == "error"]
        if errors:
            for text in errors:
                logger.error("風洞を生成できません: %s", text)
            QMessageBox.warning(self, "風洞を生成できません", "\n".join(errors))
            return False
        tunnel = self.scene.tunnel
        tunnel.generated = True
        tunnel.frontal_area = None          # 姿勢が変わっている可能性があるので計算し直す
        self.scene.notify_tunnel_changed()
        bounds = self.scene.analysis_bounds()
        d = tunnel.domain_bounds(bounds)
        logger.info("風洞を生成しました: %.2f × %.2f × %.2f m、風速 %.1f km/h (%.2f m/s)",
                    d[1] - d[0], d[3] - d[2], d[5] - d[4], tunnel.flow.speed_kmh, tunnel.flow.speed_ms)
        for level, text in issues:
            (logger.warning if level == "warning" else logger.info)("風洞: %s", text)
        self.jobs.submit(
            "frontal-area",
            {"parts": part_payloads(self.scene.assembly),
             "clip_below_z": tunnel.model_sink(bounds) if tunnel.has_ground else None},
            label="前面投影面積",
            on_success=self._on_frontal_area,
        )
        return True

    def _on_frontal_area(self, result: dict) -> None:
        tunnel = self.scene.tunnel
        tunnel.frontal_area = float(result["area"])
        self.scene.notify_tunnel_changed()
        bounds = self.scene.analysis_bounds()
        blockage = tunnel.blockage_ratio(bounds) if bounds else None
        logger.info("前面投影面積: %.4f m²%s", tunnel.frontal_area,
                    "" if blockage is None else f" (閉塞率 {blockage * 100:.2f} %)")

    # ------------------------------------------------------------------ メッシュ
    def work_dir(self) -> Path:
        """メッシュ・解析結果を置くフォルダ。未保存の間は作業フォルダ内の一時的な場所を使う。"""
        if self.project_dir is not None:
            return self.project_dir
        if self._scratch_dir is None:
            self._scratch_dir = paths.workspace_root() / "unsaved" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._scratch_dir.mkdir(parents=True, exist_ok=True)
        return self._scratch_dir

    def mesh_dir(self) -> Path:
        return self.work_dir() / "mesh"

    def _move_results_into(self, project_dir: Path) -> None:
        """保存先が変わるとき、生成済みメッシュを新しいプロジェクトフォルダへ引き継ぐ。"""
        report = self.scene.mesh_report
        if report is None:
            return
        source, target = Path(report["out_dir"]), project_dir / "mesh"
        if source.resolve() == target.resolve():
            return
        project_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        if self._scratch_dir is not None and self._scratch_dir in source.parents:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = None
        report["out_dir"] = str(target)
        self.viewport._mesh = None      # 読み込み元のパスが変わったので表示データを読み直させる

    def generate_mesh(self) -> bool:
        """現在の形状・姿勢・風洞・メッシュ設定でメッシュを生成する (別プロセス)。"""
        if self._mesh_job is not None:
            return False
        if not self.scene.parts:
            QMessageBox.information(self, "メッシュ生成", "先に CAD を読み込んでください。")
            return False
        if not self.scene.tunnel.generated and not self.generate_tunnel():
            return False
        problems = [text for level, text in self.scene.tunnel_issues() if level == "error"]
        problems += self.scene.mesh_settings.validate()
        if problems:
            for text in problems:
                logger.error("メッシュを生成できません: %s", text)
            QMessageBox.warning(self, "メッシュを生成できません", "\n".join(problems))
            return False
        request = self.scene.mesh_request(self.mesh_dir())
        logger.info("メッシュ生成を開始します (表面サイズ %.1f mm、遠方 %.0f mm、境界層 %s)",
                    self.scene.mesh_settings.surface_size * 1000, self.scene.mesh_settings.far_size * 1000,
                    "あり" if self.scene.mesh_settings.boundary_layer.enabled else "なし")
        self.mesh_panel.set_running(True)
        self._mesh_job = self.jobs.submit(
            "mesh-generate", request, label="メッシュ生成",
            on_success=self._on_mesh_done, on_failure=self._on_mesh_failed,
        )
        return True

    def stop_mesh(self) -> None:
        if self._mesh_job is not None:
            logger.warning("メッシュ生成を停止します")
            self.jobs.cancel(self._mesh_job)

    def _on_mesh_done(self, report: dict) -> None:
        self._mesh_job = None
        self.mesh_panel.set_running(False)
        self.scene.set_mesh_report(report)
        quality = report["quality"]
        summary = quality["summary"]
        log = {"ok": logger.info, "warning": logger.warning, "error": logger.error}[summary["level"]]
        log("メッシュ品質: %s", " ".join(summary["messages"]))
        if not report["boundary_layer"]["generated"]:
            logger.warning("境界層プリズムは生成されていません。粘性抗力と Cd の精度は限定的です。")
        self.bottom_tabs.setCurrentWidget(self.mesh_quality)
        self.statusBar().showMessage(f"メッシュ生成完了: {quality['n_cells']:,} セル", 10000)
        self._on_modified()

    def _on_mesh_failed(self, message: str, detail: str) -> None:
        self._mesh_job = None
        self.mesh_panel.set_running(False)
        if detail:
            logger.error("詳細:\n%s", detail)
        if message != "キャンセルされました":
            QMessageBox.critical(self, "メッシュ生成失敗", message)

    # ------------------------------------------------------------------ プロジェクト
    def _confirm_discard(self) -> bool:
        if not self._modified:
            return True
        answer = QMessageBox.question(
            self, "未保存の変更", "保存していない変更があります。保存しますか？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self._save_project_dialog()
        return True

    def new_project(self, _checked: bool = False, confirm: bool = True) -> None:
        if confirm and not self._confirm_discard():
            return
        self.jobs.cancel_all()
        self.scene.clear()
        self.project = Project()
        self.project_dir = None
        self._scratch_dir = None
        self._set_clean()

    def _projects_root(self) -> Path:
        root = paths.install_root() / "projects"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _save_project_dialog(self, save_as: bool = False) -> bool:
        target = self.project_dir
        if target is None or save_as:
            chosen = QFileDialog.getExistingDirectory(
                self, "プロジェクトフォルダを選択 (新しい空フォルダを作成して選んでください)",
                str(self.project_dir or self._projects_root()),
            )
            if not chosen:
                return False
            target = Path(chosen)
        return self.save_project_to(target)

    def save_project_to(self, project_dir: str | Path) -> bool:
        project_dir = Path(project_dir)
        project = self.project
        project.name = project_dir.name
        project.display_unit = self.scene.display_unit
        project.parts = self.scene.to_records()
        project.vehicle_pose = Pose(**self.scene.assembly.pose.to_dict())
        project.view = self.display_panel.view_settings()
        project.sections["tunnel"] = self.scene.tunnel.to_dict()
        project.sections["mesh"] = {
            "settings": self.scene.mesh_settings.to_dict(), "display": self.mesh_display.settings(),
        }
        try:
            self._move_results_into(project_dir)
            file = save_project(project, project_dir)
        except OSError as exc:
            logger.exception("プロジェクトを保存できません")
            QMessageBox.critical(self, "保存失敗", f"プロジェクトを保存できませんでした。\n\n{exc}")
            return False
        self.project_dir = project_dir
        self._set_clean()
        logger.info("プロジェクトを保存しました: %s", file)
        self.statusBar().showMessage(f"保存しました: {file}", 8000)
        return True

    def _open_project_dialog(self) -> None:
        if not self._confirm_discard():
            return
        file, _ = QFileDialog.getOpenFileName(
            self, "プロジェクトを開く", str(self._projects_root()), f"LJM CFD プロジェクト ({PROJECT_FILE})",
        )
        if file:
            self.open_project(file)

    def open_project(self, path: str | Path) -> bool:
        """プロジェクトを読み込む。CAD の再変換はキャッシュがあれば即座に終わる。"""
        try:
            project, project_dir = load_project(path)
        except ProjectError as exc:
            logger.error("プロジェクトを開けません: %s", exc)
            QMessageBox.critical(self, "プロジェクトを開けません", str(exc))
            return False
        self.jobs.cancel_all()
        self._restoring = True
        self.scene.clear()
        self.project = project
        self.project_dir = project_dir
        self.scene.assembly.pose = Pose(**project.vehicle_pose.to_dict())
        self.scene.set_display_unit(project.display_unit)
        self.scene.tunnel = TunnelSpec.from_dict(project.sections.get("tunnel"))
        self.scene.mesh_settings = MeshSettings.from_dict(project.sections.get("mesh", {}).get("settings"))
        self.scene.tunnelChanged.emit()
        self.scene.meshSettingsChanged.emit()
        self._scratch_dir = None
        self._restore_view = dict(project.view)
        self._restore_problems = []

        groups: dict[tuple[str, str], list] = {}
        for record in project.parts:
            groups.setdefault((record.cache_key, record.source_path), []).append(record)
        self._pending_restore = len(groups)
        if not groups:
            self._finish_restore()
            return True
        for (cache_key, source_path), records in groups.items():
            source = records[0].resolve_source(project_dir)
            if source is not None:
                self.jobs.submit(
                    "cad-import", {"path": str(source), "quality": self.import_quality()},
                    label=f"復元 {source.name}",
                    on_success=lambda meta, r=records: self._on_restore_group(meta, r),
                    on_failure=lambda msg, _detail, s=source_path: self._on_restore_group(None, [], f"{s}: {msg}"),
                )
                continue
            # 元ファイルが無い場合は、残っているキャッシュから復元を試みる
            cache_dir = paths.cad_cache_root() / cache_key
            cached = load_cached_meta(cache_dir) if cache_key else None
            if cached is not None:
                cached["cache_dir"] = str(cache_dir)
                logger.warning("元ファイルが見つからないためキャッシュから復元します: %s", source_path)
                self._on_restore_group(cached, records)
            else:
                self._on_restore_group(None, [], f"元の CAD ファイルが見つかりません: {source_path}")
        return True

    def _on_restore_group(self, meta: dict | None, records: list, problem: str | None = None) -> None:
        if meta is not None:
            self.scene.add_import(meta, records)
        if problem:
            self._restore_problems.append(problem)
            logger.error("復元できません: %s", problem)
        self._pending_restore -= 1
        if self._pending_restore <= 0:
            self._finish_restore()

    def _finish_restore(self) -> None:
        # 保存済みメッシュがあれば再読み込みする (再生成はしない)
        self.scene.set_mesh_report(load_report(self.mesh_dir()) if self.project_dir else None)
        self.mesh_display.apply_settings(self.project.sections.get("mesh", {}).get("display", {}))
        self.scene.tunnelChanged.emit()
        if self._restore_view:
            self.display_panel.apply_view_settings(self._restore_view)
        self._restore_view = None
        self._restoring = False
        self._set_clean()
        logger.info("プロジェクトを開きました: %s (%d パーツ)", self.project_dir, len(self.scene.parts))
        if self._restore_problems:
            QMessageBox.warning(
                self, "一部を復元できませんでした",
                "次の CAD データを復元できませんでした。\n\n" + "\n".join(self._restore_problems),
            )

    # ------------------------------------------------------------------ その他
    def _screenshot_dialog(self) -> None:
        folder = Path(self.project_dir or paths.workspace_root()) / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        default = folder / f"ljmcfd_{datetime.now():%Y%m%d_%H%M%S}.png"
        file, _ = QFileDialog.getSaveFileName(self, "スクリーンショットを保存", str(default), "PNG (*.png)")
        if file:
            self.save_screenshot(file)

    def save_screenshot(self, path: str | Path) -> None:
        try:
            self.viewport.screenshot(path)
        except Exception:
            logger.exception("スクリーンショットを保存できません")
            return
        logger.info("スクリーンショットを保存しました: %s", path)

    def _show_about(self) -> None:
        import PySide6
        import vtk

        lines = [
            f"<b>{APP_NAME} {APP_EDITION}</b> v{__version__}",
            f"実装済みステージ: Stage {IMPLEMENTED_STAGE} / 6 (CAD 読み込み・モデル操作・仮想風洞・メッシュ生成)",
            "未実装: OpenFOAM 解析 (Stage 3)、結果可視化 (Stage 4-5)",
            "",
            f"Python {platform.python_version()} / PySide6 {PySide6.__version__} / "
            f"VTK {vtk.vtkVersion.GetVTKVersion()} / PyVista {pv.__version__}",
            f"作業フォルダ: {paths.workspace_root()}",
        ]
        QMessageBox.about(self, "このビルドについて", "<br>".join(lines))

    # ------------------------------------------------------------------ Qt イベント
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(Path(u.toLocalFile()).suffix.lower() in SUPPORTED_SUFFIXES or u.toLocalFile().endswith(PROJECT_FILE)
               for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        projects = [f for f in files if f.endswith(PROJECT_FILE)]
        if projects:
            if self._confirm_discard():
                self.open_project(projects[0])
            return
        self.import_files(files)

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._confirm_discard():
            event.ignore()
            return
        self.jobs.cancel_all()
        self._save_window_state()
        self.log_panel.detach()
        self.viewport.shutdown()
        event.accept()
