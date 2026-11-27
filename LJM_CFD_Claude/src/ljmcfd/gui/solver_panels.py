"""解析 (OpenFOAM) のパネル: 設定と実行、残差・荷重のグラフ、空力結果。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import Qt, Signal  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..solver.convergence import (  # noqa: E402
    LEVEL_CONVERGED, LEVEL_DIVERGED, LEVEL_MOSTLY, LEVEL_NOT_CONVERGED, LEVEL_UNDETERMINED,
)
from ..solver.environment import KIND_WSL, FoamEnvironment  # noqa: E402
from ..solver.results import axle_loads, read_history  # noqa: E402
from ..solver.run_job import STATUS_FAILED  # noqa: E402
from ..solver.settings import ITERATION_CHOICES  # noqa: E402
from . import theme  # noqa: E402
from .analysis_panels import _set, _spin  # noqa: E402
from .panels import NOT_COMPUTED, _label  # noqa: E402
from .scene import Scene  # noqa: E402

FAILED_TEXT = "計算失敗"
CUSTOM_ITERATIONS = -1
LEVEL_COLORS = {
    LEVEL_CONVERGED: theme.OK, LEVEL_MOSTLY: theme.WARN, LEVEL_NOT_CONVERGED: theme.ERROR,
    LEVEL_UNDETERMINED: theme.TEXT_DIM, LEVEL_DIVERGED: theme.ERROR,
}
SOLVER_STATE_TEXT = {
    "none": ("解析結果: 未計算", theme.TEXT_DIM),
    "current": ("解析結果: 現在の設定の結果です", theme.OK),
    "stale": ("解析結果: 設定が変更されています — 表示中の値は以前の条件のものです", theme.WARN),
}
RESIDUAL_COLORS = {"p": "#22b8e6", "Ux": "#ff9f1c", "Uy": "#4cc38a", "Uz": "#e5b93c", "k": "#c77dff", "omega": "#ef5f5f"}


# ============================================================================ 解析設定・実行

class SolverPanel(QWidget):
    """OpenFOAM の実行環境、反復数などの設定、開始・停止。"""

    startRequested = Signal()
    stopRequested = Signal()
    checkEnvironmentRequested = Signal()
    environmentEdited = Signal()

    def __init__(self, scene: Scene, environment: FoamEnvironment, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._running = False
        self._custom_iterations = False     # 「任意の値」を選んでいるか
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- 実行環境 -----------------------------------------------------
        env_box = QGroupBox("OpenFOAM 実行環境")
        env_form = QFormLayout(env_box)
        self.kind = environment.kind
        self.distro_edit = QLineEdit(environment.distro)
        self.distro_edit.setToolTip("OpenFOAM を入れた WSL のディストリビューション名 (wsl -l -v で確認できます)。")
        self.bashrc_edit = QLineEdit(environment.bashrc)
        self.bashrc_edit.setToolTip("OpenFOAM の環境設定ファイル (WSL / Linux 側のパス)。")
        if self.kind == KIND_WSL:
            env_form.addRow("WSL ディストリビューション", self.distro_edit)
        env_form.addRow("OpenFOAM bashrc", self.bashrc_edit)
        self.check_button = QPushButton("環境を確認")
        self.env_status = _label("未確認 — 「環境を確認」を押すと OpenFOAM を検出します。", "dim", wrap=True)
        env_form.addRow(self.check_button)
        env_form.addRow(self.env_status)
        layout.addWidget(env_box)

        # --- 解析設定 -----------------------------------------------------
        run_box = QGroupBox("解析設定 (定常 RANS・k-ω SST)")
        form = QFormLayout(run_box)
        self.iteration_combo = QComboBox()
        for value in ITERATION_CHOICES:
            self.iteration_combo.addItem(f"{value}" + (" (推奨)" if value == 500 else ""), value)
        self.iteration_combo.addItem("任意の値", CUSTOM_ITERATIONS)
        self.iteration_spin = QSpinBox()
        self.iteration_spin.setRange(1, 1_000_000)
        self.iteration_spin.setKeyboardTracking(False)
        self.procs = QSpinBox()
        self.procs.setRange(0, 256)
        self.procs.setSpecialValueText("自動")
        self.procs.setKeyboardTracking(False)
        self.procs.setToolTip("並列プロセス数。自動 = 物理コア数とセル数から決めます。")
        self.continue_check = QCheckBox("前回の続きから反復を追加する")
        self.continue_check.setToolTip("外すと最初から再計算します。")
        form.addRow("反復数", self.iteration_combo)
        form.addRow("", self.iteration_spin)
        form.addRow("並列数", self.procs)
        form.addRow(self.continue_check)
        layout.addWidget(run_box)

        # --- 詳細 (安定化) ------------------------------------------------
        detail_box = QGroupBox("詳細設定 (数値安定化)")
        detail = QFormLayout(detail_box)
        self.relax_p = _spin(0.05, 1.0, 2, 0.05)
        self.relax_u = _spin(0.05, 1.0, 2, 0.05)
        self.relax_t = _spin(0.05, 1.0, 2, 0.05)
        self.correctors = QSpinBox()
        self.correctors.setRange(-1, 5)
        self.correctors.setSpecialValueText("自動")
        self.correctors.setToolTip("非直交補正の回数。自動 = メッシュの最大非直交性から決めます。")
        self.potential_check = QCheckBox("ポテンシャル流れで初期化する")
        self.tolerance_edit = QLineEdit()
        self.tolerance_edit.setToolTip("すべての残差がこの値を下回ると、指定反復数より前に終了します。")
        detail.addRow("緩和係数 (圧力)", self.relax_p)
        detail.addRow("緩和係数 (速度)", self.relax_u)
        detail.addRow("緩和係数 (乱流)", self.relax_t)
        detail.addRow("非直交補正", self.correctors)
        detail.addRow("残差の打ち切り基準", self.tolerance_edit)
        detail.addRow(self.potential_check)
        detail.addRow(_label("発散する場合は緩和係数を下げてください (例: 圧力 0.2、速度 0.5)。", "dim", wrap=True))
        layout.addWidget(detail_box)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("解析開始")
        self.start_button.setProperty("role", "primary")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("現在の反復の結果を保存して終了します (途中結果として読み込めます)。")
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        status_box = QGroupBox("解析の状態")
        status_layout = QVBoxLayout(status_box)
        self.state_label = _label("")
        self.run_status = _label("", "value", wrap=True)
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.run_status)
        layout.addWidget(status_box)
        layout.addStretch(1)

        self.iteration_combo.activated.connect(self._on_iteration_choice)
        self.iteration_spin.valueChanged.connect(lambda v: self._set("iterations", int(v)))
        self.procs.valueChanged.connect(lambda v: self._set("n_procs", int(v)))
        self.continue_check.toggled.connect(lambda on: self._set("continue_run", bool(on)))
        self.relax_p.valueChanged.connect(lambda v: self._set("relax_p", v))
        self.relax_u.valueChanged.connect(lambda v: self._set("relax_u", v))
        self.relax_t.valueChanged.connect(lambda v: self._set("relax_turbulence", v))
        self.correctors.valueChanged.connect(lambda v: self._set("non_ortho_correctors", int(v)))
        self.potential_check.toggled.connect(lambda on: self._set("potential_init", bool(on)))
        self.tolerance_edit.editingFinished.connect(self._on_tolerance)
        self.start_button.clicked.connect(self.startRequested)
        self.stop_button.clicked.connect(self.stopRequested)
        self.check_button.clicked.connect(self.checkEnvironmentRequested)
        self.distro_edit.editingFinished.connect(self.environmentEdited)
        self.bashrc_edit.editingFinished.connect(self.environmentEdited)
        for signal in (scene.solverSettingsChanged, scene.solverChanged, scene.meshChanged, scene.meshSettingsChanged,
                       scene.tunnelChanged, scene.transformsChanged, scene.cleared):
            signal.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ 環境
    def environment(self) -> FoamEnvironment:
        return FoamEnvironment(kind=self.kind, bashrc=self.bashrc_edit.text().strip() or FoamEnvironment().bashrc,
                               distro=self.distro_edit.text().strip() or FoamEnvironment().distro)

    def show_environment(self, info: dict | None, error: str = "") -> None:
        if info is None:
            self.env_status.setText(f"✖ {error}")
            self.env_status.setStyleSheet(f"color: {theme.ERROR};")
            return
        where = f"WSL「{info['distro']}」" if info["kind"] == KIND_WSL else "この PC (Linux)"
        text = (f"✔ OpenFOAM {info['version']} を検出 ({where})\n"
                f"コア数: 物理 {info['physical_cores']} / 論理 {info['logical_cores']}　"
                f"並列実行: {'可' if info['has_mpirun'] else '不可 (mpirun なし)'}")
        color = theme.OK
        if not info["version_ok"]:
            text += "\n⚠ 本ソフトは OpenFOAM 14 で検証しています。他の版では辞書の書式が合わない可能性があります。"
            color = theme.WARN
        self.env_status.setText(text)
        self.env_status.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------------ 入力 → 設定
    def _set(self, name: str, value) -> None:
        setattr(self.scene.solver_settings, name, value)
        self.scene.notify_solver_settings_changed()

    def _on_iteration_choice(self, _index: int) -> None:
        value = self.iteration_combo.currentData()
        self._custom_iterations = value == CUSTOM_ITERATIONS
        if not self._custom_iterations:
            self._set("iterations", int(value))
        else:
            self.refresh()
            self.iteration_spin.setFocus()

    def _on_tolerance(self) -> None:
        try:
            value = float(self.tolerance_edit.text())
            if value <= 0:
                raise ValueError
        except ValueError:
            self.refresh()      # 不正な入力は元の値へ戻す
            return
        self._set("residual_tolerance", value)

    def set_running(self, running: bool, text: str = "") -> None:
        self._running = running
        self.stop_button.setEnabled(running)
        if text:
            self.run_status.setText(text)
        self.refresh()

    def set_progress_text(self, text: str) -> None:
        self.run_status.setText(text)

    # ------------------------------------------------------------------ 設定 → 表示
    def refresh(self) -> None:
        settings = self.scene.solver_settings
        custom = self._custom_iterations or settings.iterations not in ITERATION_CHOICES
        _set(self.iteration_combo, CUSTOM_ITERATIONS if custom else settings.iterations)
        _set(self.iteration_spin, settings.iterations)
        self.iteration_spin.setEnabled(custom)
        _set(self.procs, settings.n_procs)
        _set(self.continue_check, settings.continue_run)
        _set(self.relax_p, settings.relax_p)
        _set(self.relax_u, settings.relax_u)
        _set(self.relax_t, settings.relax_turbulence)
        _set(self.correctors, settings.non_ortho_correctors)
        _set(self.potential_check, settings.potential_init)
        if not self.tolerance_edit.hasFocus():
            self.tolerance_edit.setText(f"{settings.residual_tolerance:g}")

        mesh_state = self.scene.mesh_state()
        ready = mesh_state == "current"
        self.start_button.setEnabled(ready and not self._running)
        text, color = SOLVER_STATE_TEXT[self.scene.solver_state()]
        if not self._running:
            if mesh_state == "none":
                text, color = "メッシュが未生成です。先に「メッシュ」タブで生成してください。", theme.WARN
            elif mesh_state == "stale":
                text, color = "メッシュが現在の設定と一致しません。メッシュを再生成してください。", theme.WARN
        else:
            text, color = "解析を実行中…", theme.ACCENT
        self.state_label.setText(text)
        self.state_label.setWordWrap(True)
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")


# ============================================================================ 残差・荷重グラフ

class ConvergencePlot(QWidget):
    """下部タブ: 残差 (対数) と荷重の履歴。解析中は逐次追加される。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        pg.setConfigOptions(antialias=True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(theme.PANEL)
        layout.addWidget(self.graphics)
        axis_pen = pg.mkPen(theme.TEXT_DIM)

        def make(title: str, column: int, left: str):
            plot = self.graphics.addPlot(row=0, col=column, title=title)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "反復")
            plot.setLabel("left", left)
            plot.addLegend(offset=(-8, 8), labelTextColor=theme.TEXT)
            for name in ("bottom", "left"):
                plot.getAxis(name).setPen(axis_pen)
                plot.getAxis(name).setTextPen(axis_pen)
            return plot

        self.residual_plot = make("残差 (初期残差)", 0, "残差")
        self.residual_plot.setLogMode(y=True)
        self.force_plot = make("荷重", 1, "力 [N]")
        self._residuals: dict[str, tuple[list, list]] = {}
        self._residual_curves: dict[str, pg.PlotDataItem] = {}
        self._forces: dict[str, tuple[list, list]] = {}
        self._force_curves: dict[str, pg.PlotDataItem] = {}
        self._force_style = {"drag": ("Drag", "#ff9f1c"), "lift": ("Lift (＋上向き)", "#22b8e6"), "side": ("Side", "#4cc38a")}

    def clear(self) -> None:
        for plot in (self.residual_plot, self.force_plot):
            plot.clear()
            if plot.legend is not None:
                plot.legend.clear()
        self._residuals, self._residual_curves, self._forces, self._force_curves = {}, {}, {}, {}

    def append_residuals(self, data: dict) -> None:
        iterations = data.get("iteration", [])
        for name, values in data.get("values", {}).items():
            xs, ys = self._residuals.setdefault(name, ([], []))
            for i, value in zip(iterations, values):
                if value is not None and value > 0:
                    xs.append(i)
                    ys.append(value)
            if name not in self._residual_curves:
                pen = pg.mkPen(RESIDUAL_COLORS.get(name, theme.TEXT), width=1.5)
                self._residual_curves[name] = self.residual_plot.plot([], [], pen=pen, name=name)
            self._residual_curves[name].setData(xs, ys)

    def append_forces(self, data: dict) -> None:
        iterations = data.get("iteration", [])
        for key, (label, color) in self._force_style.items():
            if key not in data:
                continue
            xs, ys = self._forces.setdefault(key, ([], []))
            xs.extend(iterations)
            ys.extend(data[key])
            if key not in self._force_curves:
                self._force_curves[key] = self.force_plot.plot([], [], pen=pg.mkPen(color, width=1.5), name=label)
            self._force_curves[key].setData(xs, ys)
        self._fit_force_range()

    def _fit_force_range(self) -> None:
        """立ち上がりの大きな振れに縦軸を奪われないよう、後半 60 % の範囲に合わせる。"""
        values = []
        for xs, ys in self._forces.values():
            start = int(len(ys) * 0.4)
            values.extend(ys[start:] if len(ys) > 20 else ys)
        if not values:
            return
        lo, hi = float(np.min(values)), float(np.max(values))
        pad = max(0.15 * (hi - lo), 1e-6 + 0.02 * max(abs(lo), abs(hi)))
        self.force_plot.setYRange(lo - pad, hi + pad, padding=0)

    def load_history(self, path: str | Path) -> None:
        """保存済みの履歴 (history.csv) を読み込んで表示する。"""
        self.clear()
        history = read_history(path)
        iterations = history["iteration"].tolist()
        self.append_residuals({
            "iteration": iterations,
            "values": {n: [None if np.isnan(v) else float(v) for v in values] for n, values in history["residuals"].items()},
        })
        valid = ~np.isnan(history["drag"])
        self.append_forces({
            "iteration": history["iteration"][valid].tolist(), "drag": history["drag"][valid].tolist(),
            "lift": history["lift"][valid].tolist(), "side": history["side"][valid].tolist(),
        })

    def point_counts(self) -> dict[str, int]:
        """系列ごとの点数 (テスト・確認用)。"""
        return {**{f"residual_{n}": len(xs) for n, (xs, _) in self._residuals.items()},
                **{k: len(xs) for k, (xs, _) in self._forces.items()}}


# ============================================================================ 空力結果

class ResultsPanel(QWidget):
    """空力結果。計算していない値は「未計算」、失敗した場合は「計算失敗」と表示し、推測値は出さない。"""

    reportRequested = Signal()

    ROWS = (
        ("drag", "Drag (抗力) [N]"), ("downforce", "Downforce [N]"), ("lift", "Lift (揚力・上向き正) [N]"),
        ("side", "Side Force (横力) [N]"), ("cd", "Cd"), ("cl", "Cl (上向き正)"),
        ("drag_pressure", "抗力の圧力成分 [N]"), ("drag_viscous", "抗力の粘性成分 [N]"),
        ("downforce_front", "前輪側ダウンフォース [N]"), ("downforce_rear", "後輪側ダウンフォース [N]"),
        ("front_share", "前後配分 (前輪側)"), ("cl_front", "Cl 前 / 後 (上向き正)"),
        ("speed_max", "最大流速 [m/s]"), ("pressure_max", "最大圧力 [Pa] (ゲージ)"), ("pressure_min", "最小圧力 [Pa] (ゲージ)"),
    )
    #: 前後の荷重配分の行 (ホイールベースが無いと計算できない)
    AXLE_ROWS = ("downforce_front", "downforce_rear", "front_share", "cl_front")

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        status_box = QGroupBox("計算の状態と収束")
        status_layout = QVBoxLayout(status_box)
        self.state_label = _label("", wrap=True)
        self.status_label = _label("", "value", wrap=True)
        self.convergence_label = _label("", wrap=True)
        self.checks_label = _label("", wrap=True)
        self.checks_label.setTextFormat(Qt.RichText)
        for widget in (self.state_label, self.status_label, self.convergence_label, self.checks_label):
            status_layout.addWidget(widget)
        layout.addWidget(status_box)

        value_box = QGroupBox("空力結果")
        form = QFormLayout(value_box)
        self.result_labels: dict[str, QLabel] = {}
        for key, text in self.ROWS:
            label = _label(NOT_COMPUTED, "dim")
            form.addRow(text, label)
            self.result_labels[key] = label
        self.basis_label = _label("", "dim", wrap=True)
        form.addRow(self.basis_label)
        layout.addWidget(value_box)

        part_box = QGroupBox("パーツ別の荷重")
        part_layout = QVBoxLayout(part_box)
        self.part_table = QTableWidget(0, 3)
        self.part_table.setHorizontalHeaderLabels(["パーツ", "Drag [N]", "Downforce [N]"])
        self.part_table.verticalHeader().setVisible(False)
        self.part_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.part_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.part_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.part_table.setMinimumHeight(110)
        part_layout.addWidget(self.part_table)
        layout.addWidget(part_box)
        self.part_box = part_box

        note_box = QGroupBox("精度に関する注意")
        note_layout = QVBoxLayout(note_box)
        self.notes_label = _label("", wrap=True)
        self.notes_label.setTextFormat(Qt.RichText)
        note_layout.addWidget(self.notes_label)
        layout.addWidget(note_box)
        self.report_button = QPushButton("解析レポートを書き出し…")
        self.report_button.setToolTip("条件・メッシュ・荷重・収束判定・履歴グラフを HTML と CSV で保存します。")
        self.report_button.clicked.connect(self.reportRequested)
        layout.addWidget(self.report_button)
        layout.addStretch(1)

        for signal in (scene.solverChanged, scene.tunnelChanged, scene.meshChanged, scene.meshSettingsChanged,
                       scene.solverSettingsChanged, scene.transformsChanged, scene.cleared):
            signal.connect(self.refresh)
        self.refresh()

    def _set_all(self, text: str, color: str) -> None:
        for label in self.result_labels.values():
            label.setText(text)
            label.setStyleSheet(f"color: {color};")

    def refresh(self) -> None:
        summary = self.scene.solver_summary
        text, color = SOLVER_STATE_TEXT[self.scene.solver_state()]
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.part_table.setRowCount(0)
        self.part_box.setVisible(False)
        if summary is None:
            self._set_all(NOT_COMPUTED, theme.TEXT_DIM)
            self.status_label.setText("CFD 解析はまだ実行されていません。")
            self.convergence_label.setText("")
            self.checks_label.setText("")
            self.basis_label.setText("")
            self.notes_label.setText("")
            return

        iterations = summary["iterations"]
        self.status_label.setText(
            f"{summary['status_label']}\n{iterations['done']} 反復 / ソルバー {summary['seconds_solver']:.0f} 秒 / "
            f"OpenFOAM {summary['openfoam']['version']} ({summary['openfoam']['n_procs']} プロセス)\n{summary['created']}")
        convergence = summary["convergence"]
        self.convergence_label.setText(f"収束判定: {convergence['label']}\n{convergence['message']}")
        self.convergence_label.setStyleSheet(f"color: {LEVEL_COLORS[convergence['level']]}; font-weight: bold;")
        marks = {True: ("✔", theme.OK), False: ("✖", theme.ERROR), None: ("—", theme.WARN)}
        self.checks_label.setText("<br>".join(
            f'<span style="color:{marks[c["ok"]][1]}">{marks[c["ok"]][0]} {c["name"]}</span>: {c["text"]}'
            for c in convergence["checks"]))

        if summary["status"] == STATUS_FAILED or summary["forces"] is None:
            self._set_all(FAILED_TEXT, theme.ERROR)
            self.basis_label.setText("")
            error = (summary.get("error") or "").strip().splitlines()
            self.notes_label.setText(f'<span style="color:{theme.ERROR}">ソルバーのエラー: {" ".join(error[:6])}</span>')
            return

        forces, ranges = summary["forces"], summary["ranges"] or {}
        # 係数は「現在の」基準面積で換算する (解析後に基準面積を変えたら係数も追従する)
        area = self.scene.tunnel.effective_reference_area()
        q = forces["dynamic_pressure"]
        values = {
            "drag": f"{forces['drag']:.2f}", "downforce": f"{forces['downforce']:.2f}", "lift": f"{forces['lift']:.2f}",
            "side": f"{forces['side']:.2f}",
            "cd": f"{forces['drag'] / (q * area):.4f}" if area else None,
            "cl": f"{forces['lift'] / (q * area):.4f}" if area else None,
            "drag_pressure": f"{forces['drag_pressure']:.2f}", "drag_viscous": f"{forces['drag_viscous']:.2f}",
            "speed_max": None if ranges.get("speed_max") is None else f"{ranges['speed_max']:.2f}",
            "pressure_max": None if ranges.get("pressure_max") is None else f"{ranges['pressure_max']:.1f}",
            "pressure_min": None if ranges.get("pressure_min") is None else f"{ranges['pressure_min']:.1f}",
        }
        tunnel = self.scene.tunnel
        axle = axle_loads(forces, tunnel.front_axle_x, tunnel.reference_length, q, area)
        if axle is not None:
            values.update(
                downforce_front=f"{axle['front']:.2f}", downforce_rear=f"{axle['rear']:.2f}",
                front_share=None if axle["front_share"] is None else f"{axle['front_share'] * 100:.1f} %",
                cl_front=f"{axle['cl_front']:.4f} / {axle['cl_rear']:.4f}" if "cl_front" in axle else None)
        else:
            values.update(downforce_front=None, downforce_rear=None, front_share=None, cl_front=None)
        for key, label in self.result_labels.items():
            value = values[key]
            if value is None:
                if key in self.AXLE_ROWS and forces.get("moment") is None:
                    text = "この結果には無し (v0.8 以降の解析で計算)"
                elif key in self.AXLE_ROWS and not tunnel.reference_length:
                    text = "ホイールベースが未設定"
                elif key in ("cd", "cl", "cl_front"):
                    text = "基準面積が未設定"
                else:
                    text = NOT_COMPUTED
                label.setText(text)
                label.setStyleSheet(f"color: {theme.WARN};")
            else:
                label.setText(value)
                label.setStyleSheet(f"color: {theme.TEXT}; font-family: Consolas, monospace; font-weight: bold;")
        area_text = f"{area:.4f} m²" if area else "未設定 (風洞タブで風洞を生成してください)"
        self.basis_label.setText(
            f"荷重は直近 {forces['averaged_over']} 反復の平均です (変動幅: Drag ±{forces['drag_range'] / 2:.2f} N、"
            f"Lift ±{forces['lift_range'] / 2:.2f} N)。\n"
            f"係数 = 力 / (動圧 {q:.1f} Pa × 基準面積 {area_text})。Downforce = −Lift。力は風洞座標 (風向 +X) で表します。")

        parts = summary.get("parts") or []
        self.part_box.setVisible(bool(parts))
        self.part_table.setRowCount(len(parts))
        for row, part in enumerate(parts):
            for column, text in enumerate((part["part_name"], f"{part['drag']:.2f}", f"{part['downforce']:.2f}")):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.part_table.setItem(row, column, item)

        notes = list(summary.get("notes") or [])
        yplus = summary.get("yplus") or {}
        lines = [f'<span style="color:{theme.WARN}">⚠ {note}</span>' for note in notes]
        lines += [f'<span style="color:{theme.TEXT_DIM}">y+ [{name}]: 平均 {v["average"]:.0f} / 最小 {v["min"]:.1f} / 最大 {v["max"]:.0f}</span>'
                  for name, v in yplus.items()]
        mass = summary.get("mass") or {}
        if mass.get("imbalance") is not None:
            lines.append(f'<span style="color:{theme.TEXT_DIM}">質量収支: 流入出の差 {mass["imbalance"] * 100:.4f} %</span>')
        self.notes_label.setText("<br>".join(lines))
