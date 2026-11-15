"""右パネル「可視化」: 解析結果の表示レイヤー (表面分布・断面・流線・等値面) とカラースケールの設定。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..post.result_set import COLORMAPS, SURFACE_FIELDS, VOLUME_FIELDS
from . import theme
from .analysis_panels import _set, _spin
from .panels import LengthSpin, _label
from .post_view import SOLID, PostView
from .scene import Scene

AXIS_LABELS = (("x", "X (前後位置)"), ("y", "Y (左右位置)"), ("z", "Z (高さ)"))
SLIDER_STEPS = 1000


def _field_combo(fields: dict, with_solid: bool = False) -> QComboBox:
    combo = QComboBox()
    for key, info in fields.items():
        combo.addItem(info.label, key)
    if with_solid:
        combo.addItem("単色 (色分けしない)", SOLID)
    return combo


def _percent_slider(minimum: int = 5) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, 100)
    return slider


class PostPanel(QWidget):
    def __init__(self, scene: Scene, view: PostView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.view = view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.state_label = _label("", wrap=True)
        layout.addWidget(self.state_label)

        # --- 表面分布 -----------------------------------------------------
        self.surface_box = QGroupBox("表面分布 (Surface Pressure など)")
        form = QFormLayout(self.surface_box)
        self.surface_check = QCheckBox("車体表面に分布を表示")
        self.surface_field = _field_combo(SURFACE_FIELDS)
        self.surface_ground = QCheckBox("地面にも表示")
        form.addRow(self.surface_check)
        form.addRow("物理量", self.surface_field)
        form.addRow(self.surface_ground)
        layout.addWidget(self.surface_box)

        # --- 断面 ---------------------------------------------------------
        self.slice_box = QGroupBox("断面 (流速・圧力・渦度など)")
        form = QFormLayout(self.slice_box)
        self.slice_check = QCheckBox("断面を表示")
        self.slice_field = _field_combo(VOLUME_FIELDS)
        self.slice_axis = QComboBox()
        for key, text in AXIS_LABELS:
            self.slice_axis.addItem(text, key)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, SLIDER_STEPS)
        self.slice_position = LengthSpin()
        self.slice_opacity = _percent_slider()
        form.addRow(self.slice_check)
        form.addRow("物理量", self.slice_field)
        form.addRow("断面の向き", self.slice_axis)
        form.addRow("位置", self.slice_slider)
        form.addRow("座標", self.slice_position)
        form.addRow("不透明度", self.slice_opacity)
        layout.addWidget(self.slice_box)

        # --- 流線 ---------------------------------------------------------
        self.stream_box = QGroupBox("流線 (Streamlines)")
        form = QFormLayout(self.stream_box)
        self.stream_check = QCheckBox("流線を表示")
        self.stream_color = _field_combo(VOLUME_FIELDS, with_solid=True)
        self.stream_width = _spin(0.5, 10.0, 1, 0.5, " px")
        self.seed_spins: dict[str, LengthSpin] = {}
        form.addRow(self.stream_check)
        form.addRow("色", self.stream_color)
        form.addRow("線の太さ", self.stream_width)
        for key, text in (("x", "開始位置 X"), ("y", "開始位置 Y"), ("z", "開始位置 Z"), ("width", "開始面の幅 (Y)"), ("height", "開始面の高さ (Z)")):
            spin = LengthSpin()
            spin.metresChanged.connect(lambda v, k=key: self.view.set_seed(**{k: v}))
            form.addRow(text, spin)
            self.seed_spins[key] = spin
        self.seed_ny = QSpinBox()
        self.seed_nz = QSpinBox()
        for spin in (self.seed_ny, self.seed_nz):
            spin.setRange(1, 200)
            spin.setKeyboardTracking(False)
        self.seed_spacing = LengthSpin()
        self.seed_spacing.setToolTip("横方向の間隔。変更すると、開始面の幅に合わせて横の本数が決まります。")
        self.seed_both = QCheckBox("上流側へも延ばす")
        self.seed_reset = QPushButton("車両前面に配置")
        self.stream_info = _label("", "dim", wrap=True)
        form.addRow("本数 (横)", self.seed_ny)
        form.addRow("本数 (縦)", self.seed_nz)
        form.addRow("間隔 (横)", self.seed_spacing)
        form.addRow(self.seed_both)
        form.addRow(self.seed_reset)
        form.addRow(self.stream_info)
        layout.addWidget(self.stream_box)

        # --- 等値面 -------------------------------------------------------
        self.iso_box = QGroupBox("等値面 (ISO Surface / Q 値)")
        form = QFormLayout(self.iso_box)
        self.iso_check = QCheckBox("等値面を表示")
        self.iso_field = _field_combo(VOLUME_FIELDS)
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(0, SLIDER_STEPS)
        self.iso_value = _spin(-1.0e12, 1.0e12, 4, 1.0)
        self.iso_color = _field_combo(VOLUME_FIELDS, with_solid=True)
        self.iso_opacity = _percent_slider()
        form.addRow(self.iso_check)
        form.addRow("物理量", self.iso_field)
        form.addRow("値", self.iso_slider)
        form.addRow("", self.iso_value)
        form.addRow("色", self.iso_color)
        form.addRow("不透明度", self.iso_opacity)
        form.addRow(_label(
            "流速の等値面は後流・剥離域の外形を示します。Q 値は渦の中心部を示しますが、"
            "境界層メッシュの無い粗いメッシュでは壁面近くにノイズが出ます。", "dim", wrap=True))
        layout.addWidget(self.iso_box)

        # --- カラースケール -----------------------------------------------
        self.scale_box = QGroupBox("カラースケール")
        form = QFormLayout(self.scale_box)
        self.scale_field = QComboBox()
        for key, info in {**VOLUME_FIELDS, **SURFACE_FIELDS}.items():
            self.scale_field.addItem(info.label, key)
        self.scale_auto = QCheckBox("範囲を自動で決める")
        self.scale_min = _spin(-1.0e12, 1.0e12, 4, 1.0)
        self.scale_max = _spin(-1.0e12, 1.0e12, 4, 1.0)
        self.scale_map = QComboBox()
        self.scale_map.addItem("既定", "")
        for name in COLORMAPS:
            self.scale_map.addItem(name, name)
        form.addRow("物理量", self.scale_field)
        form.addRow(self.scale_auto)
        form.addRow("最小値", self.scale_min)
        form.addRow("最大値", self.scale_max)
        form.addRow("カラーマップ", self.scale_map)
        layout.addWidget(self.scale_box)

        layout.addWidget(_label(
            "表示は計算済みの結果から抽出します (解析は再実行しません)。"
            "定常 RANS の結果なので、流線は定常の速度場に沿った軌跡です。", "dim", wrap=True))
        layout.addStretch(1)

        u = view.update
        self.surface_check.toggled.connect(lambda on: u(surface_on=bool(on)))
        self.surface_field.activated.connect(lambda _i: u(surface_field=self.surface_field.currentData()))
        self.surface_ground.toggled.connect(lambda on: u(surface_ground=bool(on)))
        self.slice_check.toggled.connect(lambda on: u(slice_on=bool(on)))
        self.slice_field.activated.connect(lambda _i: u(slice_field=self.slice_field.currentData()))
        self.slice_axis.activated.connect(lambda _i: u(slice_axis=self.slice_axis.currentData()))
        self.slice_slider.valueChanged.connect(self._on_slice_slider)
        self.slice_position.metresChanged.connect(view.set_slice_position)
        self.slice_opacity.valueChanged.connect(lambda v: u(slice_opacity=v / 100.0))
        self.stream_check.toggled.connect(lambda on: u(streamlines_on=bool(on)))
        self.stream_color.activated.connect(lambda _i: u(streamlines_color=self.stream_color.currentData()))
        self.stream_width.valueChanged.connect(lambda v: u(streamlines_width=float(v)))
        self.seed_ny.valueChanged.connect(lambda v: view.set_seed(n_y=int(v)))
        self.seed_nz.valueChanged.connect(lambda v: view.set_seed(n_z=int(v)))
        self.seed_spacing.metresChanged.connect(self._on_spacing)
        self.seed_both.toggled.connect(lambda on: view.set_seed(both_directions=bool(on)))
        self.seed_reset.clicked.connect(view.reset_seed)
        self.iso_check.toggled.connect(lambda on: u(iso_on=bool(on)))
        self.iso_field.activated.connect(lambda _i: u(iso_field=self.iso_field.currentData()))
        self.iso_slider.valueChanged.connect(self._on_iso_slider)
        self.iso_value.valueChanged.connect(lambda v: u(iso_value=float(v)))
        self.iso_color.activated.connect(lambda _i: u(iso_color=self.iso_color.currentData()))
        self.iso_opacity.valueChanged.connect(lambda v: u(iso_opacity=v / 100.0))
        self.scale_field.activated.connect(lambda _i: self.refresh())
        self.scale_auto.toggled.connect(lambda on: view.set_color_scale(self.scale_field.currentData(), auto=bool(on)))
        self.scale_min.valueChanged.connect(lambda v: view.set_color_scale(self.scale_field.currentData(), vmin=float(v)))
        self.scale_max.valueChanged.connect(lambda v: view.set_color_scale(self.scale_field.currentData(), vmax=float(v)))
        self.scale_map.activated.connect(lambda _i: view.set_color_scale(self.scale_field.currentData(), colormap=self.scale_map.currentData()))

        view.changed.connect(self.refresh)
        for signal in (scene.solverChanged, scene.tunnelChanged, scene.meshChanged, scene.solverSettingsChanged, scene.transformsChanged):
            signal.connect(self._refresh_state)
        scene.displayUnitChanged.connect(self._on_display_unit)
        self._on_display_unit(scene.display_unit)

    # ------------------------------------------------------------------ 入力
    def _on_display_unit(self, unit: str) -> None:
        for spin in (self.slice_position, self.seed_spacing, *self.seed_spins.values()):
            spin.set_unit(unit)
        self.refresh()

    def _on_slice_slider(self, step: int) -> None:
        if self.view.results is None:
            return
        lo, hi = self.view.results.axis_range(self.view.settings.slice_axis)
        self.view.set_slice_position(lo + (hi - lo) * step / SLIDER_STEPS)

    def _on_iso_slider(self, step: int) -> None:
        lo, hi = self.view.iso_range(self.view.settings.iso_field)
        self.view.update(iso_value=lo + (hi - lo) * step / SLIDER_STEPS)

    def _on_spacing(self, spacing: float) -> None:
        """横の間隔を指定 → 開始面の幅に収まる本数へ換算する。"""
        seed = self.view.settings.seed
        if spacing > 0:
            self.view.set_seed(n_y=int(max(1, min(200, round(seed.width / spacing) + 1))))
        else:
            self.refresh()

    # ------------------------------------------------------------------ 表示
    def _refresh_state(self) -> None:
        state = self.scene.solver_state()
        if not self.view.has_results():
            text, color = "表示できる解析結果がありません。「解析」タブで解析を実行してください。", theme.TEXT_DIM
        elif state == "stale":
            text, color = "表示中の結果は、現在の設定より前の条件で計算したものです。", theme.WARN
        else:
            text, color = "解析結果を表示できます。", theme.OK
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        for box in (self.surface_box, self.slice_box, self.stream_box, self.iso_box, self.scale_box):
            box.setEnabled(self.view.has_results())

    def refresh(self) -> None:
        s = self.view.settings
        self._refresh_state()
        _set(self.surface_check, s.surface_on)
        _set(self.surface_field, s.surface_field)
        _set(self.surface_ground, s.surface_ground)
        _set(self.slice_check, s.slice_on)
        _set(self.slice_field, s.slice_field)
        _set(self.slice_axis, s.slice_axis)
        self.slice_position.set_metres(s.slice_position)
        _set(self.slice_opacity, int(round(s.slice_opacity * 100)))
        _set(self.stream_check, s.streamlines_on)
        _set(self.stream_color, s.streamlines_color)
        _set(self.stream_width, s.streamlines_width)
        seed = s.seed
        for key, spin in self.seed_spins.items():
            spin.set_metres(getattr(seed, key))
        _set(self.seed_ny, seed.n_y)
        _set(self.seed_nz, seed.n_z)
        self.seed_spacing.set_metres(seed.spacing()[0])
        _set(self.seed_both, seed.both_directions)
        self.stream_info.setText(f"流線 {seed.count} 本 (横 {seed.n_y} × 縦 {seed.n_z})")
        _set(self.iso_check, s.iso_on)
        _set(self.iso_field, s.iso_field)
        _set(self.iso_color, s.iso_color)
        _set(self.iso_opacity, int(round(s.iso_opacity * 100)))
        _set(self.iso_value, s.iso_value)

        if self.view.results is not None:
            lo, hi = self.view.results.axis_range(s.slice_axis)
            _set(self.slice_slider, int(round((s.slice_position - lo) / (hi - lo) * SLIDER_STEPS)) if hi > lo else 0)
            lo, hi = self.view.iso_range(s.iso_field)
            self.iso_value.setSingleStep(max((hi - lo) / 100.0, 1e-9))
            position = (s.iso_value - lo) / (hi - lo) if hi > lo else 0.0
            _set(self.iso_slider, int(round(min(max(position, 0.0), 1.0) * SLIDER_STEPS)))

        key = self.scale_field.currentData()
        scale = self.view.scale(key)
        _set(self.scale_auto, scale.auto)
        if self.view.results is not None:
            lo, hi = self.view.color_range(key, surface=key in SURFACE_FIELDS)
            step = max((hi - lo) / 50.0, 1e-9)
            for spin, value in ((self.scale_min, lo), (self.scale_max, hi)):
                spin.setSingleStep(step)
                _set(spin, value)
        self.scale_min.setEnabled(not scale.auto)
        self.scale_max.setEnabled(not scale.auto)
        _set(self.scale_map, scale.colormap)
        info = VOLUME_FIELDS.get(key) or SURFACE_FIELDS[key]
        self.scale_min.setSuffix(f" {info.unit}" if info.unit else "")
        self.scale_max.setSuffix(f" {info.unit}" if info.unit else "")
