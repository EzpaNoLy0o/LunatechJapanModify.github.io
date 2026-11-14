"""解析設定のパネル: 仮想風洞、メッシュ、メッシュ品質、風洞・メッシュの表示切替。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..core.units import format_length
from ..mesh.settings import PRESET_CUSTOM, PRESETS
from ..tunnel import spec as ts
from . import theme
from .panels import AngleSpin, LengthSpin, _label
from .scene import Scene
from .viewport import Viewport

LEVEL_COLORS = {"error": theme.ERROR, "warning": theme.WARN, "info": theme.TEXT_DIM, "ok": theme.OK}
LEVEL_PREFIX = {"error": "✖", "warning": "⚠", "info": "・", "ok": "✔"}
MESH_STATE_TEXT = {
    "none": ("メッシュ: 未生成", theme.TEXT_DIM),
    "current": ("メッシュ: 生成済み (現在の設定と一致)", theme.OK),
    "stale": ("メッシュ: 設定が変更されています — 再生成が必要です", theme.WARN),
}


def _spin(minimum: float, maximum: float, decimals: int, step: float, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    spin.setAlignment(Qt.AlignRight)
    return spin


def _set(spin, value) -> None:
    """シグナルを出さずに値を入れる。"""
    blocked = spin.blockSignals(True)
    if isinstance(spin, QCheckBox):
        spin.setChecked(bool(value))
    elif isinstance(spin, QComboBox):
        spin.setCurrentIndex(max(spin.findData(value), 0))
    else:
        spin.setValue(value)
    spin.blockSignals(blocked)


def _issues_html(issues) -> str:
    return "<br>".join(
        f'<span style="color:{LEVEL_COLORS[level]}">{LEVEL_PREFIX[level]} {text}</span>' for level, text in issues
    )


# ============================================================================ 仮想風洞

class TunnelPanel(QWidget):
    """解析モード、流れの条件、相対風向 (車両姿勢)、風洞寸法。"""

    generateRequested = Signal()

    MARGINS = (("margin_front", "前方余白"), ("margin_rear", "後方余白"), ("margin_side", "左右余白 (片側)"),
               ("margin_top", "上方余白"), ("margin_bottom", "下方余白"))

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- 解析モード ---------------------------------------------------
        mode_box = QGroupBox("解析モード")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_combo = QComboBox()
        for mode, text in ts.MODE_LABELS.items():
            self.mode_combo.addItem(text, mode)
        index = self.mode_combo.findData(ts.MODE_INTERNAL)
        item = self.mode_combo.model().item(index)
        item.setEnabled(False)      # 内部流れは将来対応 (選択不可)
        item.setToolTip("内部流れ (ダクト等) の解析は未実装です。")
        mode_layout.addWidget(self.mode_combo)
        layout.addWidget(mode_box)

        # --- 流れの条件 ---------------------------------------------------
        flow_box = QGroupBox("流れの条件")
        form = QFormLayout(flow_box)
        self.speed_kmh = _spin(0.1, 2000.0, 2, 5.0, " km/h")
        self.speed_ms = _spin(0.03, 560.0, 3, 1.0, " m/s")
        self.speed_ms.setToolTip("km/h と連動します。どちらで入力しても構いません。")
        self.temperature = _spin(-50.0, 80.0, 1, 1.0, " °C")
        self.pressure = _spin(300.0, 1200.0, 2, 1.0, " hPa")
        self.density = _spin(0.1, 5.0, 4, 0.01, " kg/m³")
        self.density_auto = QCheckBox("密度を温度・気圧から自動計算")
        self.density_auto.setToolTip("温度と気圧から乾燥空気の密度を算出します。外すと手入力できます。")
        self.turbulence = _spin(0.01, 30.0, 2, 0.1, " %")
        self.flow_info = _label("", "dim", wrap=True)
        form.addRow("風速", self.speed_kmh)
        form.addRow("", self.speed_ms)
        form.addRow("温度", self.temperature)
        form.addRow("大気圧", self.pressure)
        form.addRow("空気密度", self.density)
        form.addRow(self.density_auto)
        form.addRow("乱流強度", self.turbulence)
        form.addRow(self.flow_info)
        layout.addWidget(flow_box)

        # --- 相対風向 -----------------------------------------------------
        attitude_box = QGroupBox("相対風向 (車両姿勢)")
        attitude_form = QFormLayout(attitude_box)
        self.attitude: dict[str, AngleSpin] = {}
        for key, text in (("yaw", "Yaw"), ("pitch", "Pitch"), ("roll", "Roll")):
            spin = AngleSpin()
            spin.valueChanged.connect(lambda v, k=key: scene.set_vehicle_pose(**{k: v}))
            attitude_form.addRow(text, spin)
            self.attitude[key] = spin
        attitude_form.addRow(_label("風向は常に +X です。車両側を回して相対風向を与えます。", "dim", wrap=True))
        layout.addWidget(attitude_box)

        # --- 風洞寸法 -----------------------------------------------------
        size_box = QGroupBox("風洞寸法")
        size_form = QFormLayout(size_box)
        self.margin_spins: dict[str, LengthSpin] = {}
        for key, text in self.MARGINS:
            spin = LengthSpin()
            spin.metresChanged.connect(lambda v, k=key: self._set_spec(k, v))
            size_form.addRow(text, spin)
            self.margin_spins[key] = spin
        self.ground_check = QCheckBox("地面あり (下面を壁として扱う)")
        self.moving_check = QCheckBox("Moving Ground (地面を風速で動かす)")
        clearance_row = QHBoxLayout()
        self.clearance = LengthSpin()
        self.clearance_apply = QPushButton("適用")
        self.clearance_apply.setToolTip("モデル全体を上下に動かし、最下点をこの高さに合わせます。")
        clearance_row.addWidget(self.clearance, 1)
        clearance_row.addWidget(self.clearance_apply)
        self.sink = LengthSpin()
        self.sink.setToolTip("接地しているモデルを、メッシュ生成時にこの量だけ沈めて接地面を作ります (車高もその分下がります)。")
        size_form.addRow(self.ground_check)
        size_form.addRow(self.moving_check)
        size_form.addRow("地面との距離", clearance_row)
        size_form.addRow("接地部の沈み込み", self.sink)
        self.auto_button = QPushButton("モデル寸法から自動設定")
        size_form.addRow(self.auto_button)
        layout.addWidget(size_box)

        self.generate_button = QPushButton("風洞を生成 / 更新")
        self.generate_button.setProperty("role", "primary")
        layout.addWidget(self.generate_button)

        # --- 結果・基準面積 -----------------------------------------------
        result_box = QGroupBox("風洞の状態")
        result_form = QFormLayout(result_box)
        self.summary = _label("", "value", wrap=True)
        self.reference_area = _spin(0.0001, 1000.0, 4, 0.01, " m²")
        self.reference_auto = QCheckBox("基準面積に前面投影面積を使用")
        self.issues = _label("", wrap=True)
        self.issues.setTextFormat(Qt.RichText)
        result_form.addRow(self.summary)
        result_form.addRow("基準面積 (Cd・Cl)", self.reference_area)
        result_form.addRow(self.reference_auto)
        result_form.addRow(self.issues)
        layout.addWidget(result_box)
        layout.addStretch(1)

        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        self.speed_kmh.valueChanged.connect(lambda v: self._set_flow("speed_ms", v / 3.6))
        self.speed_ms.valueChanged.connect(lambda v: self._set_flow("speed_ms", v))
        self.temperature.valueChanged.connect(lambda v: self._set_flow("temperature_c", v))
        self.pressure.valueChanged.connect(lambda v: self._set_flow("pressure_pa", v * 100.0))
        self.density.valueChanged.connect(lambda v: self._set_flow("density_override", v))
        self.density_auto.toggled.connect(self._on_density_auto)
        self.turbulence.valueChanged.connect(lambda v: self._set_flow("turbulence_intensity_pct", v))
        self.ground_check.toggled.connect(lambda on: self._set_spec("has_ground", bool(on)))
        self.moving_check.toggled.connect(lambda on: self._set_spec("moving_ground", bool(on)))
        self.clearance.metresChanged.connect(lambda v: self._set_spec("ground_clearance", v))
        self.clearance_apply.clicked.connect(self._apply_clearance)
        self.sink.metresChanged.connect(lambda v: self._set_spec("contact_sink", max(v, 0.0)))
        self.auto_button.clicked.connect(scene.auto_size_tunnel)
        self.generate_button.clicked.connect(self.generateRequested)
        self.reference_area.valueChanged.connect(lambda v: self._set_spec("reference_area", v))
        self.reference_auto.toggled.connect(self._on_reference_auto)
        for signal in (scene.tunnelChanged, scene.transformsChanged, scene.cleared):
            signal.connect(self.refresh)
        scene.displayUnitChanged.connect(self._on_display_unit)
        self._on_display_unit(scene.display_unit)

    # ------------------------------------------------------------------ 入力 → 仕様
    def _set_spec(self, name: str, value) -> None:
        setattr(self.scene.tunnel, name, value)
        self.scene.notify_tunnel_changed()

    def _set_flow(self, name: str, value) -> None:
        setattr(self.scene.tunnel.flow, name, value)
        self.scene.notify_tunnel_changed()

    def _on_mode(self, _index: int) -> None:
        mode = self.mode_combo.currentData()
        if mode == self.scene.tunnel.mode:
            return
        self.scene.tunnel.mode = mode
        if not self.scene.auto_size_tunnel():    # パーツが無ければ寸法は決められないのでモードだけ反映
            self.scene.tunnel.has_ground = mode == ts.MODE_VEHICLE
            self.scene.notify_tunnel_changed()

    def _on_density_auto(self, auto: bool) -> None:
        flow = self.scene.tunnel.flow
        flow.density_override = None if auto else flow.density
        self.scene.notify_tunnel_changed()

    def _on_reference_auto(self, auto: bool) -> None:
        tunnel = self.scene.tunnel
        tunnel.reference_area = None if auto else (tunnel.frontal_area or self.reference_area.value())
        self.scene.notify_tunnel_changed()

    def _apply_clearance(self) -> None:
        if self.scene.parts:
            self.scene.assembly.drop_to_ground(self.scene.tunnel.ground_clearance)
            self.scene.transformsChanged.emit()
            self.scene.modified.emit()

    def _on_display_unit(self, unit: str) -> None:
        for spin in (*self.margin_spins.values(), self.clearance, self.sink):
            spin.set_unit(unit)
        self.refresh()

    # ------------------------------------------------------------------ 仕様 → 表示
    def refresh(self) -> None:
        tunnel, flow = self.scene.tunnel, self.scene.tunnel.flow
        unit = self.scene.display_unit
        _set(self.mode_combo, tunnel.mode)
        _set(self.speed_kmh, flow.speed_kmh)
        _set(self.speed_ms, flow.speed_ms)
        _set(self.temperature, flow.temperature_c)
        _set(self.pressure, flow.pressure_pa / 100.0)
        _set(self.density_auto, flow.density_override is None)
        _set(self.density, flow.density)
        self.density.setEnabled(flow.density_override is not None)
        _set(self.turbulence, flow.turbulence_intensity_pct)
        for key, spin in self.attitude.items():
            spin.set_silent(getattr(self.scene.assembly.pose, key))
        for key, spin in self.margin_spins.items():
            spin.set_metres(getattr(tunnel, key))
        vehicle = tunnel.mode == ts.MODE_VEHICLE
        _set(self.ground_check, tunnel.has_ground)
        self.ground_check.setEnabled(not vehicle)        # 車両モードでは地面が必須
        _set(self.moving_check, tunnel.moving_ground)
        self.moving_check.setEnabled(tunnel.has_ground)
        self.margin_spins["margin_bottom"].setEnabled(not tunnel.has_ground)
        self.clearance.set_metres(tunnel.ground_clearance)
        self.sink.set_metres(tunnel.contact_sink)
        for widget in (self.clearance, self.clearance_apply, self.sink):
            widget.setEnabled(tunnel.has_ground)

        bounds = self.scene.analysis_bounds()
        length = (bounds[1] - bounds[0]) if bounds else None
        info = (f"密度 ρ = {flow.density:.4f} kg/m³　動粘性 ν = {flow.nu:.3e} m²/s\n"
                f"マッハ数 = {flow.mach:.3f}　動圧 q = {flow.dynamic_pressure:.1f} Pa")
        if length:
            info += f"\nレイノルズ数 = {flow.reynolds(length):.3e} (基準長 = 全長 {format_length(length, unit)})"
        self.flow_info.setText(info)

        has_parts = bounds is not None
        for widget in (self.auto_button, self.generate_button):
            widget.setEnabled(has_parts)
        _set(self.reference_auto, tunnel.reference_area is None)
        effective = tunnel.effective_reference_area()
        self.reference_area.setEnabled(tunnel.reference_area is not None)
        if effective is not None:
            _set(self.reference_area, effective)
        if not has_parts:
            self.summary.setText("CAD を読み込むと風洞寸法を設定できます。")
            self.issues.setText("")
            return
        d = tunnel.domain_bounds(bounds)
        state = "生成済み" if tunnel.generated else "未生成 (「風洞を生成 / 更新」を押してください)"
        frontal = "未計算" if tunnel.frontal_area is None else f"{tunnel.frontal_area:.4f} m²"
        blockage = tunnel.blockage_ratio(bounds)
        self.summary.setText(
            f"状態: {state}\n"
            f"風洞 L×W×H: {d[1] - d[0]:.2f} × {d[3] - d[2]:.2f} × {d[5] - d[4]:.2f} m\n"
            f"断面積: {tunnel.cross_section_area(bounds):.2f} m²\n"
            f"前面投影面積: {frontal}\n"
            f"閉塞率: {'未計算' if blockage is None else f'{blockage * 100:.2f} %'}\n"
            f"モデル最下点 Z: {format_length(bounds[4], unit)}"
        )
        self.issues.setText(_issues_html(self.scene.tunnel_issues()))


# ============================================================================ メッシュ

class MeshPanel(QWidget):
    """メッシュサイズ、局所細分化、境界層、生成の実行と状態表示。"""

    generateRequested = Signal()
    stopRequested = Signal()

    SIZES = (("surface_size", "表面サイズ"), ("min_size", "最小サイズ (曲率)"), ("far_size", "遠方サイズ"),
             ("refine_distance", "細分化距離"))

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._filling_table = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        size_box = QGroupBox("メッシュサイズ (四面体)")
        form = QFormLayout(size_box)
        self.preset_combo = QComboBox()
        for key, (_, text) in PRESETS.items():
            self.preset_combo.addItem(text, key)
        self.preset_combo.addItem("カスタム", PRESET_CUSTOM)
        form.addRow("プリセット", self.preset_combo)
        self.size_spins: dict[str, LengthSpin] = {}
        for key, text in self.SIZES:
            spin = LengthSpin()
            spin.metresChanged.connect(lambda v, k=key: self._set_custom(k, v))
            form.addRow(text, spin)
            self.size_spins[key] = spin
        self.curvature = QSpinBox()
        self.curvature.setRange(0, 200)
        self.curvature.setKeyboardTracking(False)
        self.curvature.setSuffix(" 要素 / 周")
        self.curvature.setToolTip("曲率ベース細分化: 円 1 周あたりの要素数 (0 で無効)。")
        form.addRow("曲率細分化", self.curvature)
        self.wake_check = QCheckBox("後流領域を細分化")
        self.wake_factor = _spin(1.0, 20.0, 2, 0.5, " ×")
        self.wake_factor.setToolTip("後流領域の要素サイズ = 表面サイズ × この倍率")
        self.wake_length = LengthSpin()
        form.addRow(self.wake_check)
        form.addRow("後流サイズ倍率", self.wake_factor)
        form.addRow("後流領域の長さ", self.wake_length)
        layout.addWidget(size_box)

        part_box = QGroupBox("パーツ近傍の局所細分化")
        part_layout = QVBoxLayout(part_box)
        self.part_table = QTableWidget(0, 2)
        self.part_table.setHorizontalHeaderLabels(["パーツ", "サイズ倍率"])
        self.part_table.verticalHeader().setVisible(False)
        self.part_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.part_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.part_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.part_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.part_table.setMinimumHeight(120)
        part_layout.addWidget(self.part_table)
        part_layout.addWidget(_label("倍率 0.5 = そのパーツの表面を標準の半分のサイズで分割します。", "dim", wrap=True))
        layout.addWidget(part_box)

        bl_box = QGroupBox("境界層プリズム (実験的)")
        bl_form = QFormLayout(bl_box)
        self.bl_check = QCheckBox("プリズム層を生成する")
        self.bl_layers = QSpinBox()
        self.bl_layers.setRange(1, 30)
        self.bl_layers.setKeyboardTracking(False)
        self.bl_first = _spin(0.001, 100.0, 3, 0.1, " mm")
        self.bl_ratio = _spin(1.0, 2.0, 2, 0.05)
        self.bl_info = _label("", wrap=True)
        self.bl_info.setTextFormat(Qt.RichText)
        bl_form.addRow(self.bl_check)
        bl_form.addRow("層数", self.bl_layers)
        bl_form.addRow("第 1 層の厚さ", self.bl_first)
        bl_form.addRow("成長率", self.bl_ratio)
        bl_form.addRow(self.bl_info)
        bl_form.addRow(_label(
            "モデルが地面・風洞壁に接していない場合のみ生成できます。形状によっては失敗します。"
            "接地した車両の境界層メッシュ (snappyHexMesh) は今後のステージで対応予定です。", "dim", wrap=True))
        layout.addWidget(bl_box)

        buttons = QHBoxLayout()
        self.generate_button = QPushButton("メッシュ生成")
        self.generate_button.setProperty("role", "primary")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.generate_button, 1)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        status_box = QGroupBox("メッシュの状態")
        status_layout = QVBoxLayout(status_box)
        self.state_label = _label("")
        self.summary = _label("", "value", wrap=True)
        self.summary.setTextFormat(Qt.RichText)
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.summary)
        layout.addWidget(status_box)
        layout.addStretch(1)

        self.preset_combo.activated.connect(self._on_preset)
        self.curvature.valueChanged.connect(lambda v: self._set_custom("curvature_elements", int(v)))
        self.wake_check.toggled.connect(lambda on: self._set("wake_enabled", bool(on)))
        self.wake_factor.valueChanged.connect(lambda v: self._set("wake_size_factor", v))
        self.wake_length.metresChanged.connect(lambda v: self._set("wake_length", v))
        self.bl_check.toggled.connect(lambda on: self._set_bl("enabled", bool(on)))
        self.bl_layers.valueChanged.connect(lambda v: self._set_bl("layers", int(v)))
        self.bl_first.valueChanged.connect(lambda v: self._set_bl("first_height", v / 1000.0))
        self.bl_ratio.valueChanged.connect(lambda v: self._set_bl("growth_ratio", v))
        self.generate_button.clicked.connect(self.generateRequested)
        self.stop_button.clicked.connect(self.stopRequested)
        for signal in (scene.meshSettingsChanged, scene.meshChanged, scene.tunnelChanged, scene.transformsChanged, scene.cleared):
            signal.connect(self.refresh)
        for signal in (scene.partsAdded, scene.partRemoved, scene.partChanged, scene.cleared):
            signal.connect(lambda *_: self._rebuild_part_table())
        scene.displayUnitChanged.connect(self._on_display_unit)
        self._on_display_unit(scene.display_unit)

    # ------------------------------------------------------------------ 入力 → 設定
    def _set(self, name: str, value) -> None:
        setattr(self.scene.mesh_settings, name, value)
        self.scene.notify_mesh_settings_changed()

    def _set_custom(self, name: str, value) -> None:
        """サイズを手で変えたらプリセットを「カスタム」にする。"""
        self.scene.mesh_settings.preset = PRESET_CUSTOM
        self._set(name, value)

    def _set_bl(self, name: str, value) -> None:
        setattr(self.scene.mesh_settings.boundary_layer, name, value)
        self.scene.notify_mesh_settings_changed()

    def _on_preset(self, _index: int) -> None:
        preset = self.preset_combo.currentData()
        bounds = self.scene.analysis_bounds()
        if bounds is not None:
            self.scene.mesh_settings.apply_preset(preset, bounds)
        else:
            self.scene.mesh_settings.preset = preset
        self.scene.notify_mesh_settings_changed()

    def _on_display_unit(self, unit: str) -> None:
        for spin in (*self.size_spins.values(), self.wake_length):
            spin.set_unit(unit)
        self.refresh()

    def set_running(self, running: bool) -> None:
        self.generate_button.setEnabled(not running and bool(self.scene.parts))
        self.stop_button.setEnabled(running)

    # ------------------------------------------------------------------ パーツ別倍率
    def _rebuild_part_table(self) -> None:
        self._filling_table = True
        try:
            parts = [p for p in self.scene.parts if p.visible]
            self.part_table.setRowCount(len(parts))
            factors = self.scene.mesh_settings.part_size_factors
            for row, part in enumerate(parts):
                item = QTableWidgetItem(part.name)
                item.setData(Qt.UserRole, part.id)
                self.part_table.setItem(row, 0, item)
                spin = _spin(0.05, 5.0, 2, 0.05)
                spin.setValue(float(factors.get(part.id, 1.0)))
                spin.valueChanged.connect(lambda v, pid=part.id: self._on_factor(pid, v))
                self.part_table.setCellWidget(row, 1, spin)
        finally:
            self._filling_table = False

    def _on_factor(self, part_id: str, value: float) -> None:
        if self._filling_table:
            return
        factors = self.scene.mesh_settings.part_size_factors
        if abs(value - 1.0) < 1e-9:
            factors.pop(part_id, None)
        else:
            factors[part_id] = float(value)
        self.scene.notify_mesh_settings_changed()

    def factor_spin(self, part_id: str) -> QDoubleSpinBox | None:
        for row in range(self.part_table.rowCount()):
            if self.part_table.item(row, 0).data(Qt.UserRole) == part_id:
                return self.part_table.cellWidget(row, 1)
        return None

    # ------------------------------------------------------------------ 設定 → 表示
    def refresh(self) -> None:
        settings = self.scene.mesh_settings
        bl = settings.boundary_layer
        _set(self.preset_combo, settings.preset)
        for key, spin in self.size_spins.items():
            spin.set_metres(getattr(settings, key))
        _set(self.curvature, settings.curvature_elements)
        _set(self.wake_check, settings.wake_enabled)
        _set(self.wake_factor, settings.wake_size_factor)
        self.wake_length.set_metres(settings.wake_length)
        self.wake_factor.setEnabled(settings.wake_enabled)
        self.wake_length.setEnabled(settings.wake_enabled)
        _set(self.bl_check, bl.enabled)
        _set(self.bl_layers, bl.layers)
        _set(self.bl_first, bl.first_height * 1000.0)
        _set(self.bl_ratio, bl.growth_ratio)
        for widget in (self.bl_layers, self.bl_first, self.bl_ratio):
            widget.setEnabled(bl.enabled)
        if not self.stop_button.isEnabled():
            self.generate_button.setEnabled(bool(self.scene.parts))
        self.bl_info.setText(self._bl_text())
        self._refresh_status()

    def _bl_text(self) -> str:
        settings = self.scene.mesh_settings
        bl = settings.boundary_layer
        bounds = self.scene.analysis_bounds()
        if bounds is None:
            return ""
        flow = self.scene.tunnel.flow
        length = bounds[1] - bounds[0]
        if bl.enabled:
            yplus = ts.estimate_yplus(bl.first_height, flow, length)
            color = theme.OK if 30 <= yplus <= 300 or yplus <= 5 else theme.WARN
            return (f'合計厚さ {bl.total_thickness() * 1000:.2f} mm　'
                    f'<span style="color:{color}">第 1 セルの y+ ≈ {yplus:.0f} (目安)</span><br>'
                    f'<span style="color:{theme.TEXT_DIM}">壁関数を使う場合は 30〜300 が適正範囲です。</span>')
        # 境界層なし: 壁面に接する四面体の高さ (正四面体で 0.816 × 辺長) を第 1 層とみなす
        yplus = ts.estimate_yplus(0.816 * settings.surface_size, flow, length)
        return (f'<span style="color:{theme.WARN}">境界層メッシュなし: 壁面第 1 セルの y+ ≈ {yplus:,.0f} (目安)。'
                f'壁関数の適正範囲 (30〜300) を外れるため、粘性抗力・剥離位置・Cd の精度は限定的です。</span>')

    def _refresh_status(self) -> None:
        state = self.scene.mesh_state()
        text, color = MESH_STATE_TEXT[state]
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        report = self.scene.mesh_report
        if report is None:
            self.summary.setText("")
            return
        quality = report["quality"]
        bl = report["boundary_layer"]
        level = quality["summary"]["level"]
        level_text = {"ok": "良好", "warning": "注意", "error": "不良"}[level]
        cells = " / ".join(f"{name} {count:,}" for name, count in quality["cell_counts"].items())
        if bl["generated"]:
            bl_line = (f'<span style="color:{theme.OK}">境界層プリズム: 生成済み ({bl["layers"]} 層、'
                       f'第 1 層 {bl["first_height"] * 1000:.2f} mm、合計 {bl["total_thickness"] * 1000:.2f} mm)</span>')
        else:
            bl_line = (f'<span style="color:{theme.WARN}">境界層プリズム: 未生成 — 粘性抗力と Cd の精度は限定的です</span>')
        self.summary.setText(
            f"セル数: {quality['n_cells']:,} ({cells})<br>節点数: {quality['n_points']:,}<br>"
            f'品質: <span style="color:{LEVEL_COLORS[level]}">{level_text}</span> '
            f"(非直交性 最大 {quality['non_ortho_max']:.1f}°、歪度 最大 {quality['skewness_max']:.2f})<br>"
            f"{bl_line}<br>生成日時: {report['created']} ({report['seconds_total']:.1f} 秒)"
        )


class MeshQualityView(QTextEdit):
    """下部タブ: メッシュ統計と品質レポート。"""

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.setReadOnly(True)
        scene.meshChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        report = self.scene.mesh_report
        if report is None:
            self.setHtml(f'<p style="color:{theme.TEXT_DIM}">メッシュは未生成です。「メッシュ」タブで生成すると、統計と品質がここに表示されます。</p>')
            return
        q = report["quality"]
        summary = q["summary"]
        cell = f'style="padding:2px 14px 2px 0"'
        rows = [
            ("セル数", f"{q['n_cells']:,}"), ("セルの内訳", "、".join(f"{k} {v:,}" for k, v in q["cell_counts"].items())),
            ("節点数", f"{q['n_points']:,}"), ("面数 (内部 / 境界)", f"{q['n_internal_faces']:,} / {q['n_boundary_faces']:,}"),
            ("流体体積", f"{q['volume_total']:.4f} m³ (CAD との差 {report['volume_error']:.1e})"),
            ("セル体積 最小 / 最大", f"{q['volume_min']:.3e} / {q['volume_max']:.3e} m³"),
            ("非直交性 最大 / 平均", f"{q['non_ortho_max']:.1f}° / {q['non_ortho_mean']:.1f}°  (70° 超: {q['n_non_ortho_over_70']:,} 面)"),
            ("歪度 最大", f"{q['skewness_max']:.2f}  (4 超: {q['n_skewness_over_4']:,} 面)"),
            ("隣接セル体積比 最大", f"{q['volume_ratio_max']:.1f}"),
            ("負体積セル / 開いたセル", f"{q['n_negative_volumes']:,} / {q['n_open_cells']:,}"),
        ]
        gamma = report.get("tet_gamma")
        if gamma:
            rows.append(("四面体の形状品質 γ 最小 / 平均", f"{gamma['min']:.3f} / {gamma['mean']:.3f}  (1 が正四面体)"))
        bl = report["boundary_layer"]
        rows.append(("境界層プリズム", f"生成済み: {bl['layers']} 層 / {bl['n_prisms']:,} セル" if bl["generated"] else f"未生成 — {bl['reason']}"))
        rows.append(("メッシャー", f"Gmsh {report['gmsh_version']} ({report['algorithm']})、{report['seconds_total']:.1f} 秒"))
        html = [f'<p style="color:{LEVEL_COLORS[summary["level"]]}"><b>{" ".join(summary["messages"])}</b></p>', "<table>"]
        html += [f"<tr><td {cell}>{k}</td><td {cell}><b>{v}</b></td></tr>" for k, v in rows]
        html.append("</table><p><b>境界パッチ</b></p><table>")
        html.append(f"<tr><td {cell}>名前</td><td {cell}>種別</td><td {cell}>面数</td><td {cell}>面積 [m²]</td><td {cell}>対応パーツ</td></tr>")
        for patch in report["patches"]:
            area = q["patch_areas"].get(patch["name"], 0.0)
            html.append(
                f"<tr><td {cell}>{patch['name']}</td><td {cell}>{patch['type']}</td><td {cell} align='right'>{patch['n_faces']:,}</td>"
                f"<td {cell} align='right'>{area:.4f}</td><td {cell}>{patch.get('part_name', '')}</td></tr>")
        html.append("</table><p><b>非直交性の分布 (内部面)</b></p><table>")
        histogram = q["non_ortho_histogram"]
        total = max(sum(histogram), 1)
        labels = [f"{i * 10}–{i * 10 + 10}°" for i in range(9)] + ["90° 以上"]
        for label, count in zip(labels, histogram):
            bar = "█" * int(round(40 * count / total))
            html.append(f"<tr><td {cell}>{label}</td><td {cell} align='right'>{count:,}</td><td style='color:{theme.ACCENT}'>{bar}</td></tr>")
        html.append("</table>")
        self.setHtml("".join(html))


# ============================================================================ 風洞・メッシュの表示切替 (右パネル)

class MeshDisplayPanel(QWidget):
    """3D ビューでの風洞・メッシュの表示切替と、メッシュ断面の位置スライダー。"""

    AXES = (("x", "X (前後)"), ("y", "Y (左右)"), ("z", "Z (高さ)"))
    SLIDER_STEPS = 1000

    def __init__(self, scene: Scene, viewport: Viewport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.viewport = viewport
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        tunnel_box = QGroupBox("風洞の表示")
        tunnel_layout = QVBoxLayout(tunnel_box)
        self.tunnel_check = QCheckBox("風洞を表示 (青 = 流入、赤 = 流出)")
        self.tunnel_check.setChecked(True)
        self.fit_tunnel = QPushButton("風洞全体を表示")
        tunnel_layout.addWidget(self.tunnel_check)
        tunnel_layout.addWidget(self.fit_tunnel)
        layout.addWidget(tunnel_box)

        mesh_box = QGroupBox("メッシュの表示")
        mesh_layout = QVBoxLayout(mesh_box)
        self.surface_check = QCheckBox("モデル表面のメッシュ")
        self.ground_check = QCheckBox("地面のメッシュ")
        self.section_check = QCheckBox("断面のメッシュ (体積セル)")
        mesh_layout.addWidget(self.surface_check)
        mesh_layout.addWidget(self.ground_check)
        mesh_layout.addWidget(self.section_check)
        form = QFormLayout()
        self.axis_combo = QComboBox()
        for key, text in self.AXES:
            self.axis_combo.addItem(text, key)
        self.axis_combo.setCurrentIndex(1)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.SLIDER_STEPS)
        self.position = LengthSpin()
        form.addRow("断面の向き", self.axis_combo)
        form.addRow("位置", self.slider)
        form.addRow("座標", self.position)
        mesh_layout.addLayout(form)
        self.note = _label("", "dim", wrap=True)
        mesh_layout.addWidget(self.note)
        layout.addWidget(mesh_box)
        layout.addStretch(1)

        self.tunnel_check.toggled.connect(viewport.set_show_tunnel)
        self.fit_tunnel.clicked.connect(viewport.fit_tunnel)
        self.surface_check.toggled.connect(viewport.set_show_mesh_surface)
        self.ground_check.toggled.connect(viewport.set_show_mesh_ground)
        self.section_check.toggled.connect(self._apply_section)
        self.axis_combo.currentIndexChanged.connect(self._on_axis)
        self.slider.valueChanged.connect(self._on_slider)
        self.position.metresChanged.connect(self._on_position)
        scene.meshChanged.connect(self.refresh)
        scene.displayUnitChanged.connect(self.position.set_unit)
        self.position.set_unit(scene.display_unit)
        self.refresh()

    def _range(self) -> tuple[float, float] | None:
        return self.viewport.mesh_axis_range(self.axis_combo.currentData())

    def refresh(self) -> None:
        has_mesh = self.scene.mesh_report is not None
        for widget in (self.surface_check, self.ground_check, self.section_check, self.axis_combo, self.slider, self.position):
            widget.setEnabled(has_mesh)
        if not has_mesh:
            for check in (self.surface_check, self.ground_check, self.section_check):
                _set(check, False)
            self.note.setText("メッシュを生成すると表示できます。")
            return
        self.note.setText("スライダーで断面を動かせます (メッシュの再生成は行いません)。")
        has_ground = any(p["name"] == "ground" for p in self.scene.mesh_report["patches"])
        self.ground_check.setEnabled(has_ground)
        self._center_position()

    def _center_position(self) -> None:
        """断面の初期位置をモデルの中心にする。"""
        limits = self._range()
        bounds = self.scene.mesh_report["model_bounds"] if self.scene.mesh_report else None
        if limits is None or bounds is None:
            return
        index = "xyz".index(self.axis_combo.currentData())
        self._move_to((bounds[2 * index] + bounds[2 * index + 1]) / 2)

    def _move_to(self, value: float) -> None:
        limits = self._range()
        if limits is None:
            return
        lo, hi = limits
        value = min(max(value, lo), hi)
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(round((value - lo) / (hi - lo) * self.SLIDER_STEPS)) if hi > lo else 0)
        self.slider.blockSignals(blocked)
        self.position.set_metres(value)
        self._apply_section()

    def _on_axis(self, _index: int) -> None:
        self._center_position()

    def _on_slider(self, step: int) -> None:
        limits = self._range()
        if limits is not None:
            lo, hi = limits
            self._move_to(lo + (hi - lo) * step / self.SLIDER_STEPS)

    def _on_position(self, value: float) -> None:
        self._move_to(value)

    def _apply_section(self, *_args) -> None:
        self.viewport.set_mesh_section(self.section_check.isChecked(), self.axis_combo.currentData(), self.position.metres())

    def settings(self) -> dict:
        return {
            "show_tunnel": self.tunnel_check.isChecked(), "mesh_surface": self.surface_check.isChecked(),
            "mesh_ground": self.ground_check.isChecked(), "mesh_section": self.section_check.isChecked(),
            "section_axis": self.axis_combo.currentData(), "section_position": self.position.metres(),
        }

    def apply_settings(self, data: dict) -> None:
        self.tunnel_check.setChecked(bool(data.get("show_tunnel", True)))
        if self.scene.mesh_report is None:
            return
        self.axis_combo.setCurrentIndex(max(self.axis_combo.findData(data.get("section_axis", "y")), 0))
        if "section_position" in data:
            self._move_to(float(data["section_position"]))
        self.surface_check.setChecked(bool(data.get("mesh_surface", False)))
        self.ground_check.setChecked(bool(data.get("mesh_ground", False)) and self.ground_check.isEnabled())
        self.section_check.setChecked(bool(data.get("mesh_section", False)))
