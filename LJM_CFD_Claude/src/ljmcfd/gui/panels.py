"""左右・下部のパネル: パーツツリー、モデル配置、表示設定、ログ、形状検査結果。"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMenu, QPlainTextEdit, QPushButton, QScrollArea, QSlider,
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .. import IMPLEMENTED_STAGE
from ..core.units import DISPLAY_UNITS, format_length, from_display, to_display
from . import theme
from .scene import Scene
from .viewport import Viewport

PART_ID_ROLE = Qt.UserRole + 1
PART_UNITS = ("mm", "cm", "m", "inch")
LEVEL_LABELS = {"ok": ("OK", theme.OK), "warning": ("警告", theme.WARN), "error": ("不良", theme.ERROR)}
NOT_COMPUTED = "未計算"


def _label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if role:
        label.setProperty("role", role)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def scrollable(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return area


class LengthSpin(QDoubleSpinBox):
    """値を常にメートルで受け渡しし、表示だけを表示単位に合わせるスピンボックス。"""

    metresChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._unit = "mm"
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignRight)
        self._configure()
        self.valueChanged.connect(lambda v: self.metresChanged.emit(from_display(v, self._unit)))

    def _configure(self) -> None:
        if self._unit == "mm":
            self.setDecimals(2)
            self.setRange(-1.0e6, 1.0e6)
            self.setSingleStep(1.0)
        else:
            self.setDecimals(5)
            self.setRange(-1.0e3, 1.0e3)
            self.setSingleStep(0.001)
        self.setSuffix(f" {self._unit}")

    def set_unit(self, unit: str) -> None:
        metres = self.metres()
        self._unit = unit
        blocked = self.blockSignals(True)
        self._configure()
        self.setValue(to_display(metres, unit))
        self.blockSignals(blocked)

    def metres(self) -> float:
        return from_display(self.value(), self._unit)

    def set_metres(self, value: float) -> None:
        blocked = self.blockSignals(True)
        self.setValue(to_display(value, self._unit))
        self.blockSignals(blocked)


class AngleSpin(QDoubleSpinBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignRight)
        self.setDecimals(2)
        self.setRange(-360.0, 360.0)
        self.setSingleStep(0.5)
        self.setSuffix(" °")

    def set_silent(self, value: float) -> None:
        blocked = self.blockSignals(True)
        self.setValue(value)
        self.blockSignals(blocked)


# ============================================================================ パーツツリー

class PartTree(QTreeWidget):
    """読み込んだファイルとパーツの一覧。チェックで表示/非表示、選択は 3D ビューと連動。"""

    inspectRequested = Signal(list)   # part ids
    removeRequested = Signal(list)

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._syncing = False
        self._items: dict[str, QTreeWidgetItem] = {}
        self._groups: dict[str, QTreeWidgetItem] = {}
        self.setHeaderLabels(["パーツ", "単位", "検査"])
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self._on_tree_selection)
        self.customContextMenuRequested.connect(self._show_menu)
        scene.partsAdded.connect(self._on_parts_added)
        scene.partRemoved.connect(self._on_part_removed)
        scene.partChanged.connect(self._refresh_part)
        scene.selectionChanged.connect(self._on_scene_selection)
        scene.cleared.connect(self._on_cleared)

    # ------------------------------------------------------------------ 同期 (シーン → ツリー)
    def _on_parts_added(self, parts) -> None:
        self._syncing = True
        try:
            for part in parts:
                key = f"{part.cache_key}|{part.source_path}"
                group = self._groups.get(key)
                if group is None:
                    group = QTreeWidgetItem(self, [Path(part.source_path).name, part.fmt.upper(), ""])
                    group.setFlags(group.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                    group.setToolTip(0, part.source_path)
                    group.setData(0, Qt.UserRole, key)
                    group.setExpanded(True)
                    self._groups[key] = group
                item = QTreeWidgetItem(group)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setData(0, PART_ID_ROLE, part.id)
                self._items[part.id] = item
                self._fill(item, part)
        finally:
            self._syncing = False

    def _fill(self, item: QTreeWidgetItem, part) -> None:
        item.setText(0, part.name)
        item.setCheckState(0, Qt.Checked if part.visible else Qt.Unchecked)
        item.setText(1, part.unit + ("?" if part.unit_source == "heuristic" else ""))
        item.setToolTip(1, "寸法から推定した単位です。モデルタブで確認してください。" if part.unit_source == "heuristic" else "")
        item.setToolTip(0, f"{part.name}\n面 {part.entry['n_faces']:,} / 三角形 {part.entry['n_triangles']:,}")
        if part.inspection is None:
            item.setText(2, "—")
            item.setForeground(2, QBrush(QColor(theme.TEXT_DIM)))
            item.setToolTip(2, "未検査 (右クリック → 形状検査)")
        else:
            text, color = LEVEL_LABELS[part.inspection["summary"]["level"]]
            item.setText(2, text)
            item.setForeground(2, QBrush(QColor(color)))
            item.setToolTip(2, "\n".join(part.inspection["summary"]["messages"]))

    def _refresh_part(self, part_id: str) -> None:
        item = self._items.get(part_id)
        if item is None:
            return
        self._syncing = True
        try:
            self._fill(item, self.scene.assembly.get(part_id))
        finally:
            self._syncing = False

    def _on_part_removed(self, part_id: str) -> None:
        item = self._items.pop(part_id, None)
        if item is None:
            return
        self._syncing = True
        try:
            group = item.parent()
            group.removeChild(item)
            if group.childCount() == 0:
                self._groups.pop(group.data(0, Qt.UserRole), None)
                self.takeTopLevelItem(self.indexOfTopLevelItem(group))
        finally:
            self._syncing = False

    def _on_cleared(self) -> None:
        self._syncing = True
        try:
            self.clear()
            self._items.clear()
            self._groups.clear()
        finally:
            self._syncing = False

    def _on_scene_selection(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.clearSelection()
            for part_id in self.scene.selected_ids:
                item = self._items.get(part_id)
                if item is not None:
                    item.setSelected(True)
            if self.scene.selected_ids and self.scene.selected_ids[-1] in self._items:
                self.scrollToItem(self._items[self.scene.selected_ids[-1]])
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ 同期 (ツリー → シーン)
    def _ids_of(self, items) -> list[str]:
        ids: list[str] = []
        for item in items:
            part_id = item.data(0, PART_ID_ROLE)
            if part_id:
                ids.append(part_id)
            else:
                ids.extend(item.child(i).data(0, PART_ID_ROLE) for i in range(item.childCount()))
        return list(dict.fromkeys(ids))

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        part_id = item.data(0, PART_ID_ROLE)
        if part_id:
            self.scene.set_visible([part_id], item.checkState(0) == Qt.Checked)

    def _on_tree_selection(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.scene.set_selection(self._ids_of(self.selectedItems()))
        finally:
            self._syncing = False

    def _show_menu(self, pos) -> None:
        ids = self._ids_of(self.selectedItems())
        menu = QMenu(self)
        show = menu.addAction("表示")
        hide = menu.addAction("非表示")
        isolate = menu.addAction("選択パーツのみ表示")
        show_all = menu.addAction("すべて表示")
        menu.addSeparator()
        inspect = menu.addAction("形状検査 (開放エッジ・不良面)")
        menu.addSeparator()
        remove = menu.addAction("削除")
        for action in (show, hide, isolate, inspect, remove):
            action.setEnabled(bool(ids))
        show_all.setEnabled(bool(self.scene.parts))
        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        all_ids = [p.id for p in self.scene.parts]
        if chosen is show:
            self.scene.set_visible(ids, True)
        elif chosen is hide:
            self.scene.set_visible(ids, False)
        elif chosen is isolate:
            self.scene.set_visible([i for i in all_ids if i not in ids], False)
            self.scene.set_visible(ids, True)
        elif chosen is show_all:
            self.scene.set_visible(all_ids, True)
        elif chosen is inspect:
            self.inspectRequested.emit(ids)
        elif chosen is remove:
            self.removeRequested.emit(ids)


# ============================================================================ モデル配置パネル

class _PoseEditor(QWidget):
    """X/Y/Z と Yaw/Pitch/Roll の 6 入力欄。"""

    changed = Signal(str, float)   # フィールド名 (tx …), 値 (m または deg)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.length_spins: dict[str, LengthSpin] = {}
        self.angle_spins: dict[str, AngleSpin] = {}
        for row, (key, text) in enumerate((("tx", "X"), ("ty", "Y"), ("tz", "Z"))):
            spin = LengthSpin()
            spin.metresChanged.connect(lambda v, k=key: self.changed.emit(k, v))
            grid.addWidget(_label(text, "dim"), row, 0)
            grid.addWidget(spin, row, 1)
            self.length_spins[key] = spin
        for row, (key, text, tip) in enumerate((
            ("yaw", "Yaw", "Z 軸まわりの回転 (+ で反時計回り・上から見て)"),
            ("pitch", "Pitch", "Y 軸まわりの回転 (右手系)"),
            ("roll", "Roll", "X 軸まわりの回転 (右手系)"),
        )):
            spin = AngleSpin()
            spin.setToolTip(tip)
            spin.valueChanged.connect(lambda v, k=key: self.changed.emit(k, v))
            grid.addWidget(_label(text, "dim"), row, 2)
            grid.addWidget(spin, row, 3)
            self.angle_spins[key] = spin
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

    def set_unit(self, unit: str) -> None:
        for spin in self.length_spins.values():
            spin.set_unit(unit)

    def set_pose(self, pose) -> None:
        for key, spin in self.length_spins.items():
            spin.set_metres(getattr(pose, key))
        for key, spin in self.angle_spins.items():
            spin.set_silent(getattr(pose, key))


class ModelPanel(QWidget):
    """選択パーツの情報・単位・配置と、車両全体の姿勢。"""

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- 選択パーツ ---------------------------------------------------
        info_box = QGroupBox("選択パーツ")
        form = QFormLayout(info_box)
        form.setLabelAlignment(Qt.AlignLeft)
        self.name_label = _label("—", "headline", wrap=True)
        self.file_label = _label("", "dim", wrap=True)
        self.size_label = _label("", "value")
        self.stats_label = _label("", "value")
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(PART_UNITS)
        self.unit_combo.setToolTip("キャッシュ座標の単位。STL など単位情報が無いファイルで誤っていれば変更します。")
        self.unit_note = _label("", "dim", wrap=True)
        form.addRow(self.name_label)
        form.addRow(self.file_label)
        form.addRow("寸法 X×Y×Z", self.size_label)
        form.addRow("面 / 三角形", self.stats_label)
        form.addRow("単位", self.unit_combo)
        form.addRow(self.unit_note)
        layout.addWidget(info_box)

        # --- パーツ配置 ---------------------------------------------------
        part_box = QGroupBox("パーツ配置 (選択パーツ)")
        part_layout = QVBoxLayout(part_box)
        self.part_pose = _PoseEditor()
        part_layout.addWidget(self.part_pose)
        part_layout.addWidget(_label("回転中心は各パーツの境界ボックス中心です。", "dim", wrap=True))
        self.part_reset = QPushButton("配置をリセット")
        part_layout.addWidget(self.part_reset)
        layout.addWidget(part_box)
        self.part_box = part_box

        # --- 車両全体 -----------------------------------------------------
        vehicle_box = QGroupBox("車両全体の姿勢 (全パーツ)")
        vehicle_layout = QVBoxLayout(vehicle_box)
        self.vehicle_pose = _PoseEditor()
        vehicle_layout.addWidget(self.vehicle_pose)
        vehicle_layout.addWidget(_label(
            "風向は常に +X です。Yaw/Pitch/Roll で車両側を回して相対風向を与えます。"
            "回転中心は全体境界の XY 中心・最下点です。", "dim", wrap=True))
        buttons = QHBoxLayout()
        self.ground_button = QPushButton("地面に接地 (Z=0)")
        self.center_button = QPushButton("XY 中心を原点へ")
        buttons.addWidget(self.ground_button)
        buttons.addWidget(self.center_button)
        vehicle_layout.addLayout(buttons)
        self.vehicle_reset = QPushButton("姿勢をリセット")
        vehicle_layout.addWidget(self.vehicle_reset)
        self.total_label = _label("", "value", wrap=True)
        vehicle_layout.addWidget(self.total_label)
        layout.addWidget(vehicle_box)
        layout.addStretch(1)

        self.unit_combo.activated.connect(self._on_unit_chosen)
        self.part_pose.changed.connect(lambda key, v: scene.set_part_pose(scene.selected_ids, **{key: v}))
        self.part_reset.clicked.connect(lambda: scene.reset_part_pose(scene.selected_ids))
        self.vehicle_pose.changed.connect(lambda key, v: scene.set_vehicle_pose(**{key: v}))
        self.vehicle_reset.clicked.connect(scene.reset_vehicle_pose)
        self.ground_button.clicked.connect(scene.drop_to_ground)
        self.center_button.clicked.connect(scene.center_xy)

        for signal in (scene.selectionChanged, scene.transformsChanged, scene.cleared):
            signal.connect(self.refresh)
        scene.partChanged.connect(lambda _id: self.refresh())
        scene.displayUnitChanged.connect(self._on_display_unit)
        self._on_display_unit(scene.display_unit)

    def _on_display_unit(self, unit: str) -> None:
        self.part_pose.set_unit(unit)
        self.vehicle_pose.set_unit(unit)
        self.refresh()

    def _on_unit_chosen(self, _index: int) -> None:
        self.scene.set_part_unit(self.scene.selected_ids, self.unit_combo.currentText())

    def _size_text(self, bounds) -> str:
        if bounds is None:
            return "—"
        unit = self.scene.display_unit
        dims = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        decimals = 1 if unit == "mm" else 4
        return " × ".join(f"{to_display(d, unit):.{decimals}f}" for d in dims) + f" {unit}"

    def refresh(self) -> None:
        scene = self.scene
        selected = scene.selected_parts()
        has_selection = bool(selected)
        self.part_box.setEnabled(has_selection)
        self.unit_combo.setEnabled(has_selection)
        if has_selection:
            part = selected[0]
            extra = f"  (+{len(selected) - 1})" if len(selected) > 1 else ""
            self.name_label.setText(part.name + extra)
            self.file_label.setText(Path(part.source_path).name)
            self.size_label.setText(self._size_text(scene.assembly.world_bounds(selected)))
            faces = sum(p.entry["n_faces"] for p in selected)
            tris = sum(p.entry["n_triangles"] for p in selected)
            self.stats_label.setText(f"{faces:,} / {tris:,}")
            self.unit_combo.setCurrentText(part.unit)
            self.unit_note.setText(self._unit_note(part))
            self.unit_note.setProperty("role", "notice" if part.unit_source == "heuristic" else "dim")
            self.unit_note.style().unpolish(self.unit_note)
            self.unit_note.style().polish(self.unit_note)
            self.part_pose.set_pose(part.pose)
        else:
            self.name_label.setText("— (パーツ未選択)")
            self.file_label.setText("ツリーまたは 3D ビューでパーツをクリックして選択します。")
            self.size_label.setText("—")
            self.stats_label.setText("—")
            self.unit_note.setText("")
        self.vehicle_pose.set_pose(scene.assembly.pose)
        self.vehicle_pose.setEnabled(bool(scene.parts))
        for button in (self.ground_button, self.center_button, self.vehicle_reset):
            button.setEnabled(bool(scene.parts))
        total = scene.assembly.world_bounds()
        if total is None:
            self.total_label.setText("全体寸法: —")
        else:
            unit = scene.display_unit
            self.total_label.setText(
                f"全体寸法: {self._size_text(total)}\n最下点 Z: {format_length(total[4], unit)}"
            )

    @staticmethod
    def _unit_note(part) -> str:
        if part.unit_source == "file":
            if part.file_unit and part.file_unit != part.unit:
                return f"元ファイルの単位は {part.file_unit}。CAD カーネルが {part.unit} へ自動換算済みです。"
            return f"ファイルに記録された単位 ({part.file_unit}) を使用しています。"
        if part.unit_source == "occt":
            return "ファイルから単位を検出できませんでした。CAD カーネルの換算結果 (mm) を使用しています。"
        if part.unit_source == "heuristic":
            return "⚠ このファイル形式は単位を持ちません。寸法から推定した値です。必ず確認してください。"
        return "ユーザーが指定した単位です。"


# ============================================================================ 右側: 表示設定・選択情報・計測・結果

class DisplayPanel(QWidget):
    """表示設定 / 選択情報 / 寸法計測 / 空力結果。"""

    def __init__(self, scene: Scene, viewport: Viewport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.viewport = viewport
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- 表示設定 -----------------------------------------------------
        view_box = QGroupBox("表示設定")
        form = QFormLayout(view_box)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(DISPLAY_UNITS)
        self.background_combo = QComboBox()
        self.background_combo.addItem("黒系 (ダーク)", "dark")
        self.background_combo.addItem("白系 (ライト)", "light")
        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(100)
        self.opacity_value = _label("100 %", "value")
        self.opacity_value.setMinimumWidth(44)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_value)
        self.edges_check = QCheckBox("面の輪郭線")
        self.bbox_check = QCheckBox("境界ボックスと寸法")
        self.ground_check = QCheckBox("地面グリッド・風向")
        self.parallel_check = QCheckBox("平行投影")
        for check in (self.edges_check, self.bbox_check, self.ground_check):
            check.setChecked(True)
        form.addRow("表示単位", self.unit_combo)
        form.addRow("背景", self.background_combo)
        form.addRow("モデル不透明度", opacity_row)
        form.addRow(self.edges_check)
        form.addRow(self.bbox_check)
        form.addRow(self.ground_check)
        form.addRow(self.parallel_check)
        layout.addWidget(view_box)

        # --- 選択情報 -----------------------------------------------------
        select_box = QGroupBox("選択情報")
        select_layout = QVBoxLayout(select_box)
        self.selection_label = _label("", "value", wrap=True)
        select_layout.addWidget(self.selection_label)
        layout.addWidget(select_box)

        # --- 寸法計測 -----------------------------------------------------
        measure_box = QGroupBox("寸法計測")
        measure_layout = QVBoxLayout(measure_box)
        self.measure_label = _label("", "value", wrap=True)
        self.snap_check = QCheckBox("頂点にスナップ")
        self.snap_check.setChecked(True)
        self.measure_clear = QPushButton("計測をクリア")
        measure_layout.addWidget(self.measure_label)
        measure_layout.addWidget(self.snap_check)
        measure_layout.addWidget(self.measure_clear)
        layout.addWidget(measure_box)

        # --- 空力結果 -----------------------------------------------------
        result_box = QGroupBox("空力結果")
        result_form = QFormLayout(result_box)
        self.result_labels: dict[str, QLabel] = {}
        for key, text in (
            ("drag", "Drag [N]"), ("downforce", "Downforce [N]"), ("lift", "Lift [N]"),
            ("side", "Side Force [N]"), ("cd", "Cd"), ("cl", "Cl"),
        ):
            value = _label(NOT_COMPUTED, "dim")
            result_form.addRow(text, value)
            self.result_labels[key] = value
        result_form.addRow(_label(
            f"このビルドは Stage {IMPLEMENTED_STAGE} (メッシュ生成まで) です。"
            "CFD 解析は Stage 3 で実装されるため、結果は常に「未計算」と表示されます。", "dim", wrap=True))
        layout.addWidget(result_box)
        layout.addStretch(1)

        self.unit_combo.currentTextChanged.connect(scene.set_display_unit)
        self.background_combo.currentIndexChanged.connect(
            lambda _i: viewport.set_theme(self.background_combo.currentData()))
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        self.edges_check.toggled.connect(viewport.set_show_edges)
        self.bbox_check.toggled.connect(viewport.set_show_bbox)
        self.ground_check.toggled.connect(viewport.set_show_ground)
        self.parallel_check.toggled.connect(viewport.set_parallel_projection)
        self.snap_check.toggled.connect(lambda on: setattr(viewport, "snap_to_vertex", bool(on)))
        self.measure_clear.clicked.connect(viewport.clear_measurement)
        viewport.measureChanged.connect(self._on_measure)
        for signal in (scene.selectionChanged, scene.faceSelectionChanged, scene.transformsChanged):
            signal.connect(self._refresh_selection)
        scene.displayUnitChanged.connect(self._on_display_unit)
        for check in (self.edges_check, self.bbox_check, self.ground_check, self.parallel_check):
            check.toggled.connect(lambda _on: scene.modified.emit())
        self._refresh_selection()
        self._on_measure(None)

    # ------------------------------------------------------------------ 表示設定
    def _on_opacity(self, value: int) -> None:
        self.opacity_value.setText(f"{value} %")
        self.viewport.set_model_opacity(value / 100.0)
        self.scene.modified.emit()

    def _on_display_unit(self, unit: str) -> None:
        if self.unit_combo.currentText() != unit:
            self.unit_combo.setCurrentText(unit)
        self._refresh_selection()
        self._on_measure(self.viewport.measurement())

    def view_settings(self) -> dict:
        """プロジェクトへ保存する表示設定。"""
        return {
            "background": self.background_combo.currentData(),
            "opacity": self.opacity_slider.value() / 100.0,
            "show_edges": self.edges_check.isChecked(),
            "show_bbox": self.bbox_check.isChecked(),
            "show_ground": self.ground_check.isChecked(),
            "snap_to_vertex": self.snap_check.isChecked(),
            "camera": self.viewport.camera_state(),
        }

    def apply_view_settings(self, settings: dict) -> None:
        index = self.background_combo.findData(settings.get("background", "dark"))
        self.background_combo.setCurrentIndex(max(index, 0))
        self.opacity_slider.setValue(int(round(float(settings.get("opacity", 1.0)) * 100)))
        self.edges_check.setChecked(bool(settings.get("show_edges", True)))
        self.bbox_check.setChecked(bool(settings.get("show_bbox", True)))
        self.ground_check.setChecked(bool(settings.get("show_ground", True)))
        self.snap_check.setChecked(bool(settings.get("snap_to_vertex", True)))
        camera = settings.get("camera")
        if camera:
            self.parallel_check.setChecked(bool(camera.get("parallel", False)))
            self.viewport.restore_camera(camera)

    # ------------------------------------------------------------------ 選択・計測
    def _refresh_selection(self) -> None:
        unit = self.scene.display_unit
        info = self.scene.face_info()
        if info is not None:
            n = info["normal"]
            c = info["center"]
            kind = info["surface_type"] or "(STL 領域)"
            self.selection_label.setText(
                f"面 #{info['face_id']}  [{info['part'].name}]\n"
                f"種別: {kind}\n"
                f"面積: {info['area']:.6g} m²\n"
                f"平均法線: ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})\n"
                f"中心: ({format_length(c[0], unit)}, {format_length(c[1], unit)}, {format_length(c[2], unit)})\n"
                f"三角形: {info['n_triangles']:,}"
            )
            return
        selected = self.scene.selected_parts()
        if not selected:
            self.selection_label.setText("何も選択されていません。")
            return
        names = "、".join(p.name for p in selected[:4]) + (" …" if len(selected) > 4 else "")
        b = self.scene.assembly.world_bounds(selected)
        self.selection_label.setText(
            f"パーツ {len(selected)} 個: {names}\n"
            f"X: {format_length(b[0], unit)} 〜 {format_length(b[1], unit)}\n"
            f"Y: {format_length(b[2], unit)} 〜 {format_length(b[3], unit)}\n"
            f"Z: {format_length(b[4], unit)} 〜 {format_length(b[5], unit)}"
        )

    def _on_measure(self, data) -> None:
        unit = self.scene.display_unit
        if not data:
            self.measure_label.setText("ツールバーの「寸法計測」を選び、モデル上の 2 点をクリックします。")
            return
        p1 = data["p1"]
        text = f"点 1: ({format_length(p1[0], unit)}, {format_length(p1[1], unit)}, {format_length(p1[2], unit)})"
        if data["p2"] is None:
            self.measure_label.setText(text + "\n2 点目をクリックしてください。")
            return
        p2, d = data["p2"], data["delta"]
        self.measure_label.setText(
            text + f"\n点 2: ({format_length(p2[0], unit)}, {format_length(p2[1], unit)}, {format_length(p2[2], unit)})\n"
            f"距離: {format_length(data['distance'], unit)}\n"
            f"ΔX: {format_length(d[0], unit)}\nΔY: {format_length(d[1], unit)}\nΔZ: {format_length(d[2], unit)}"
        )


# ============================================================================ 下部: ログ・形状検査

class _LogBridge(QObject):
    record = Signal(str, str)   # レベル名, 整形済みテキスト


class QtLogHandler(logging.Handler):
    """logging のレコードをシグナル経由で GUI スレッドへ渡す (他スレッドからでも安全)。"""

    def __init__(self) -> None:
        super().__init__()
        self.bridge = _LogBridge()
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            if record.exc_info and not record.exc_text:
                text += "\n" + logging.Formatter().formatException(record.exc_info)
            self.bridge.record.emit(record.levelname, text)
        except RuntimeError:   # アプリ終了後に Qt オブジェクトが破棄済み
            pass


class LogPanel(QPlainTextEdit):
    COLORS = {"DEBUG": theme.TEXT_DIM, "INFO": theme.TEXT, "WARNING": theme.WARN, "ERROR": theme.ERROR, "CRITICAL": theme.ERROR}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.handler = QtLogHandler()
        self.handler.bridge.record.connect(self.append_record)
        root = logging.getLogger("ljmcfd")
        if root.level == logging.NOTSET:   # setup_logging() を通らない起動経路でも INFO 以上を表示する
            root.setLevel(logging.INFO)
        root.addHandler(self.handler)

    def append_record(self, level: str, text: str) -> None:
        color = self.COLORS.get(level, theme.TEXT)
        body = html.escape(text).replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
        self.appendHtml(f'<span style="color:{color}">{body}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def detach(self) -> None:
        logging.getLogger("ljmcfd").removeHandler(self.handler)


class InspectionTable(QTableWidget):
    """形状検査の結果一覧。行を選ぶと該当パーツが選択される。"""

    COLUMNS = ["パーツ", "判定", "開放エッジ", "非多様体", "向き不整合", "ソリッド", "体積 [m³]", "表面積 [m²]", "詳細"]

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self.COLUMNS), parent)
        self.scene = scene
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(self.COLUMNS) - 1, QHeaderView.Stretch)
        self.itemSelectionChanged.connect(self._on_row_selected)
        scene.partChanged.connect(lambda _id: self.rebuild())
        scene.partRemoved.connect(lambda _id: self.rebuild())
        scene.transformsChanged.connect(self.rebuild)   # 単位変更で体積・面積が変わる
        scene.cleared.connect(self.rebuild)

    def rebuild(self) -> None:
        parts = [p for p in self.scene.parts if p.inspection is not None]
        blocked = self.blockSignals(True)
        self.setRowCount(len(parts))
        for row, part in enumerate(parts):
            mesh = part.inspection["mesh"]
            brep = part.inspection["brep"]
            summary = part.inspection["summary"]
            text, color = LEVEL_LABELS[summary["level"]]
            scale = part.scale
            volume = "—" if mesh["volume"] is None else f"{mesh['volume'] * scale ** 3:.6g}"
            cells = [
                part.name, text, f"{mesh['n_open_edges']:,}", f"{mesh['n_non_manifold_edges']:,}",
                f"{mesh['n_inconsistent_edges']:,}", "—" if brep is None else str(brep["n_solids"]),
                volume, f"{mesh['area'] * scale ** 2:.6g}", " ".join(summary["messages"]),
            ]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, part.id)
                if column == 1:
                    item.setForeground(QBrush(QColor(color)))
                if 2 <= column <= 7:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setToolTip("\n".join(summary["messages"]))
                self.setItem(row, column, item)
        self.blockSignals(blocked)

    def _on_row_selected(self) -> None:
        items = self.selectedItems()
        if items:
            self.scene.set_selection([items[0].data(Qt.UserRole)])
