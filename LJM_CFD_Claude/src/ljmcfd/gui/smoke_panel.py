"""ボリューム煙の設定画面。

* ``SmokeEmitterPanel`` (左のタブ「スモーク」)      … 放出源の配置 (プリセット・複数・個別の調整)
* ``SmokeRenderPanel``  (右のタブ「スモーク表示」)  … 表示方式・品質・色・濃さ・不透明度・広がり・寿命と詳細設定

どちらも ``PostView`` の設定を変えるだけで、OpenFOAM は再実行しない。
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from ..smoke.density import QUALITY_LABELS
from ..smoke.emitter import KIND_LABELS, KIND_LINE, KINDS, PRESET_LABELS, Emitter
from ..smoke.settings import BACKEND_LABELS, COLOR_CUSTOM, COLOR_LABELS, QUALITIES, TRANSPORT_LABELS
from . import theme
from .analysis_panels import _set, _spin
from .panels import LengthSpin, _label
from .post_view import STYLE_LABELS, VOLUME_STYLES, PostView
from .scene import Scene


def _int_spin(minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    spin.setAlignment(Qt.AlignRight)
    return spin


class SmokeEmitterPanel(QWidget):
    """煙の放出源。放出源は煙の出る位置・量・時間を決めるだけで、気流は変えない。"""

    def __init__(self, scene: Scene, view: PostView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.view = view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        self.state_label = _label("", wrap=True)
        layout.addWidget(self.state_label)

        preset_box = QGroupBox("配置のプリセット (実車向け)")
        preset_layout = QVBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        for key, text in PRESET_LABELS.items():
            self.preset_combo.addItem(text, key)
        buttons = QHBoxLayout()
        self.preset_replace = QPushButton("置き換える")
        self.preset_add = QPushButton("追加する")
        self.preset_replace.setToolTip("今の放出源をすべて消して、プリセットの位置に置きます。")
        self.preset_add.setToolTip("今の放出源に、プリセットの放出源を加えます。")
        buttons.addWidget(self.preset_replace)
        buttons.addWidget(self.preset_add)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addLayout(buttons)
        preset_layout.addWidget(_label("位置は車両の外形寸法から決めます。気流は変えません。", "dim", wrap=True))
        layout.addWidget(preset_box)

        list_box = QGroupBox("放出源 (チェックで有効)")
        list_layout = QVBoxLayout(list_box)
        self.emitter_list = QListWidget()
        self.emitter_list.setMinimumHeight(90)
        list_layout.addWidget(self.emitter_list)
        row = QHBoxLayout()
        self.add_button = QPushButton("追加")
        self.copy_button = QPushButton("複製")
        self.remove_button = QPushButton("削除")
        for button in (self.add_button, self.copy_button, self.remove_button):
            row.addWidget(button)
        list_layout.addLayout(row)
        layout.addWidget(list_box)

        self.edit_box = QGroupBox("選択した放出源")
        form = QFormLayout(self.edit_box)
        self.name_edit = QLineEdit()
        self.kind_combo = QComboBox()
        for key in KINDS:
            self.kind_combo.addItem(KIND_LABELS[key], key)
        self.length_spins = {key: LengthSpin() for key in ("x", "y", "z", "width", "height", "radius")}
        self.n_y = _int_spin(1, 200)
        self.n_z = _int_spin(1, 200)
        self.rate = _spin(0.0, 100.0, 2, 0.1)
        self.start = _spin(0.0, 100.0, 4, 0.005, " s")
        self.stop = _spin(-1.0, 100.0, 4, 0.005, " s")
        self.stop.setSpecialValueText("止めない")
        self.pulse = _spin(0.0, 10.0, 4, 0.005, " s")
        self.pulse.setSpecialValueText("連続")
        self.duty = _spin(0.05, 1.0, 2, 0.05)
        form.addRow("名前", self.name_edit)
        form.addRow("形", self.kind_combo)
        form.addRow("位置 X (前後)", self.length_spins["x"])
        form.addRow("位置 Y (左右)", self.length_spins["y"])
        form.addRow("位置 Z (高さ)", self.length_spins["z"])
        form.addRow("幅 (Y 方向)", self.length_spins["width"])
        form.addRow("高さ (Z 方向)", self.length_spins["height"])
        form.addRow("ノズル数 Y", self.n_y)
        form.addRow("ノズル数 Z", self.n_z)
        form.addRow("初期の太さ (半径)", self.length_spins["radius"])
        form.addRow("放出量 (相対)", self.rate)
        form.addRow("放出開始", self.start)
        form.addRow("放出終了", self.stop)
        form.addRow("間欠放出の周期", self.pulse)
        form.addRow("間欠放出の割合", self.duty)
        self.length_spins["radius"].setToolTip("煙の筋の初期の太さ (Kernel Radius)。")
        self.pulse.setToolTip("0 = 連続して放出します。周期を入れると、煙のかたまりが流れていく様子が見やすくなります。")
        layout.addWidget(self.edit_box)
        layout.addStretch(1)

        self.preset_replace.clicked.connect(lambda: view.apply_smoke_preset(self.preset_combo.currentData()))
        self.preset_add.clicked.connect(lambda: view.apply_smoke_preset(self.preset_combo.currentData(), add=True))
        self.add_button.clicked.connect(self._add)
        self.copy_button.clicked.connect(self._copy)
        self.remove_button.clicked.connect(self._remove)
        self.emitter_list.currentRowChanged.connect(lambda _row: self._refresh_editor())
        self.emitter_list.itemChanged.connect(self._on_item_changed)
        self.name_edit.editingFinished.connect(lambda: self._edit(name=self.name_edit.text().strip() or "Emitter"))
        self.kind_combo.activated.connect(lambda _i: self._edit(kind=self.kind_combo.currentData()))
        for key, spin in self.length_spins.items():
            spin.metresChanged.connect(lambda value, k=key: self._edit(**{k: value}))
        self.n_y.valueChanged.connect(lambda v: self._edit(n_y=int(v)))
        self.n_z.valueChanged.connect(lambda v: self._edit(n_z=int(v)))
        self.rate.valueChanged.connect(lambda v: self._edit(rate=float(v)))
        self.start.valueChanged.connect(lambda v: self._edit(start=float(v)))
        self.stop.valueChanged.connect(lambda v: self._edit(stop=float(v)))
        self.pulse.valueChanged.connect(lambda v: self._edit(pulse_period=float(v)))
        self.duty.valueChanged.connect(lambda v: self._edit(pulse_duty=float(v)))
        view.changed.connect(self.refresh)
        view.resultsChanged.connect(self.refresh)
        scene.displayUnitChanged.connect(self._on_unit)
        self._on_unit(scene.display_unit)

    # ------------------------------------------------------------------ 操作
    def emitters(self) -> list[Emitter]:
        return self.view.settings.smoke.emitters

    def current_index(self) -> int:
        row = self.emitter_list.currentRow()
        return row if 0 <= row < len(self.emitters()) else -1

    def _edit(self, **values) -> None:
        index = self.current_index()
        if index < 0:
            return
        emitter = self.emitters()[index]
        if all(getattr(emitter, key) == value for key, value in values.items()):
            return
        updated = list(self.emitters())
        updated[index] = replace(emitter, **values)
        self.view.set_emitters(updated)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        row = self.emitter_list.row(item)
        if 0 <= row < len(self.emitters()):
            enabled = item.checkState() == Qt.Checked
            if self.emitters()[row].enabled != enabled:
                updated = list(self.emitters())
                updated[row] = replace(updated[row], enabled=enabled)
                self.view.set_emitters(updated)

    def _add(self) -> None:
        bounds = self.view.model_bounds()
        if bounds is None:
            return
        x0, x1, y0, y1, z0, z1 = bounds
        new = Emitter(name=f"Emitter {len(self.emitters()) + 1}", kind=KIND_LINE, x=x0 - 0.25 * (x1 - x0), y=0.5 * (y0 + y1),
                      z=z0 + 0.5 * (z1 - z0), width=0.6 * (y1 - y0), height=0.0, n_y=7)
        self.view.set_emitters(self.emitters() + [new])
        self.emitter_list.setCurrentRow(len(self.emitters()) - 1)

    def _copy(self) -> None:
        index = self.current_index()
        if index >= 0:
            source = self.emitters()[index]
            self.view.set_emitters(self.emitters() + [replace(source, name=f"{source.name} (複製)")])
            self.emitter_list.setCurrentRow(len(self.emitters()) - 1)

    def _remove(self) -> None:
        index = self.current_index()
        if index >= 0:
            updated = list(self.emitters())
            del updated[index]
            self.view.set_emitters(updated)

    # ------------------------------------------------------------------ 表示
    def _on_unit(self, unit: str) -> None:
        for spin in self.length_spins.values():
            spin.set_unit(unit)
        self.refresh()

    def refresh(self) -> None:
        has = self.view.has_results()
        for widget in (self.preset_combo, self.preset_replace, self.preset_add, self.add_button):
            widget.setEnabled(has)
        if not has:
            self.state_label.setText("解析結果がありません。煙は計算済みの速度場から作るので、先に CFD 解析を実行してください。")
        else:
            self.state_label.setText("煙は解析済みの速度場に沿って流れます (OpenFOAM は再実行しません)。"
                                     "表示は下部のバーまたは「スモーク表示」タブで「ボリューム煙」を選びます。")
        smoke = self.view.settings.smoke
        _set(self.preset_combo, smoke.preset)
        row = self.emitter_list.currentRow()
        blocked = self.emitter_list.blockSignals(True)
        self.emitter_list.clear()
        for emitter in smoke.emitters:
            item = QListWidgetItem(f"{emitter.name}  ({KIND_LABELS.get(emitter.kind, emitter.kind)}・ノズル {len(emitter.nozzle_points())})")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if emitter.enabled else Qt.Unchecked)
            self.emitter_list.addItem(item)
        if smoke.emitters:
            self.emitter_list.setCurrentRow(min(max(row, 0), len(smoke.emitters) - 1))
        self.emitter_list.blockSignals(blocked)
        self._refresh_editor()

    def _refresh_editor(self) -> None:
        index = self.current_index()
        self.edit_box.setEnabled(index >= 0)
        self.copy_button.setEnabled(index >= 0)
        self.remove_button.setEnabled(index >= 0)
        if index < 0:
            return
        e = self.emitters()[index]
        widgets = [self.name_edit, self.kind_combo, self.n_y, self.n_z, self.rate, self.start, self.stop, self.pulse, self.duty,
                   *self.length_spins.values()]
        states = [w.blockSignals(True) for w in widgets]
        self.name_edit.setText(e.name)
        _set(self.kind_combo, e.kind)
        for key, spin in self.length_spins.items():
            spin.set_metres(getattr(e, key))
        self.n_y.setValue(e.n_y)
        self.n_z.setValue(e.n_z)
        self.rate.setValue(e.rate)
        self.start.setValue(e.start)
        self.stop.setValue(e.stop)
        self.pulse.setValue(e.pulse_period)
        self.duty.setValue(e.pulse_duty)
        for w, state in zip(widgets, states):
            w.blockSignals(state)
        # 形によって使わない項目は触れないようにする
        self.n_z.setEnabled(e.kind == "grid")
        self.n_y.setEnabled(e.kind in ("line", "grid"))
        self.length_spins["width"].setEnabled(e.kind != "point")
        self.length_spins["height"].setEnabled(e.kind != "point")
        self.duty.setEnabled(e.pulse_period > 0)


class SmokeRenderPanel(QWidget):
    """煙の見た目と計算の設定。見た目の変更は描画だけをやり直す (煙の計算はやり直さない)。"""

    def __init__(self, scene: Scene, view: PostView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.view = view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        basic = QGroupBox("煙の表示")
        form = QFormLayout(basic)
        self.show_check = QCheckBox("煙を表示")
        self.mode_combo = QComboBox()
        for key, text in STYLE_LABELS.items():
            self.mode_combo.addItem(text, key)
        self.quality_combo = QComboBox()
        for key in QUALITIES:
            self.quality_combo.addItem(QUALITY_LABELS[key], key)
        self.quality_combo.setToolTip("煙の格子の細かさ。Low 約 26 万・Medium 約 118 万・High 約 315 万・Ultra 約 1060 万ボクセル。")
        self.color_combo = QComboBox()
        for key, text in COLOR_LABELS.items():
            self.color_combo.addItem(text, key)
        self.color_button = QPushButton("色を選ぶ…")
        self.density_slider = self._slider(5, 400)
        self.density_slider.setToolTip("光学的な濃さ (吸収係数の倍率)。表示だけの設定で、煙の量・流れは変わりません。")
        self.opacity_slider = self._slider(0, 100)
        self.spread = _spin(0.0, 1.0, 3, 0.005, " m²/s")
        self.spread.setMinimumWidth(0)
        self.spread.setToolTip("表示用の広がり係数 (σ² = σ0² + 2·D·t)。OpenFOAM が計算した乱流拡散ではありません。")
        self.lifetime = _spin(0.0, 10.0, 3, 0.01, " s")
        self.lifetime.setSpecialValueText("なし")
        self.lifetime.setToolTip("この時間より古い煙は消えます。なし = 計算領域から出るまで表示します。")
        self.lighting = QCheckBox("陰影を付ける")
        self.lighting.setToolTip("密度の勾配による柔らかい陰影。高品質ボリューム煙では常に付けます。CPU 描画では少し遅くなります。")
        self.light = self._slider(0, 200)
        form.addRow(self.show_check)
        form.addRow("表示方式", self.mode_combo)
        form.addRow("品質", self.quality_combo)
        form.addRow("色", self.color_combo)
        form.addRow("", self.color_button)
        form.addRow("濃さ", self.density_slider)
        form.addRow("不透明度", self.opacity_slider)
        form.addRow("広がり", self.spread)
        form.addRow("寿命", self.lifetime)
        form.addRow(self.lighting)
        form.addRow("光の強さ", self.light)
        layout.addWidget(basic)

        status_box = QGroupBox("煙の計算")
        status_layout = QVBoxLayout(status_box)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = _label("", wrap=True)
        self.cancel_button = QPushButton("計算を中止")
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.status)
        status_layout.addWidget(self.cancel_button)
        layout.addWidget(status_box)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("詳細設定")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        adv = QFormLayout(self.advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        self.voxel_budget = _int_spin(0, 50_000_000)
        self.voxel_budget.setSingleStep(100_000)
        self.voxel_budget.setSpecialValueText("品質から決める")
        self.seeds = _int_spin(1, 64)
        self.sampling = _spin(0.1, 4.0, 2, 0.05, " × ボクセル")
        self.sample_distance = LengthSpin()
        self.sample_distance.setSpecialValueText("自動")
        self.cache_mb = _int_spin(32, 16384, " MB")
        self.max_tracks = _int_spin(100, 500_000)
        self.backend = QComboBox()
        for key, text in BACKEND_LABELS.items():
            self.backend.addItem(text, key)
        self.transport = QComboBox()
        for key, text in TRANSPORT_LABELS.items():
            self.transport.addItem(text, key)
        # 長い選択肢でパネルが横に広がらないようにする (開いた一覧には全文が出る)
        for combo in (self.transport, self.backend, self.mode_combo, self.color_combo, self.quality_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.timeline = _spin(0.0, 60.0, 3, 0.01, " s")
        self.timeline.setSpecialValueText("自動")
        self.noise = QCheckBox("視覚効果のゆらぎ")
        self.noise.setToolTip("煙の筋に流れに沿った濃淡を付ける表示だけの効果です。CFD の結果ではありません (既定は OFF)。")
        adv.addRow("ボクセル数", self.voxel_budget)
        adv.addRow("ノズル 1 本の流跡線数", self.seeds)
        adv.addRow("流跡線の標本間隔", self.sampling)
        adv.addRow("レイキャスティングの間隔", self.sample_distance)
        adv.addRow("キャッシュの上限", self.cache_mb)
        adv.addRow("流跡線の上限本数", self.max_tracks)
        adv.addRow("描画", self.backend)
        adv.addRow("輸送モデル", self.transport)
        adv.addRow("タイムラインの長さ", self.timeline)
        adv.addRow(self.noise)
        adv.addRow(_label("ゆらぎは CFD の結果ではない表示だけの効果です。", "dim", wrap=True))
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)
        note = _label("煙は定常 RANS の速度場に沿った可視化です。時間変化する流れ (非定常解析) の結果ではありません。"
                      "広がり・ゆらぎは表示用の効果で、解析の精度は変わりません。", "dim", wrap=True)
        layout.addWidget(note)
        layout.addStretch(1)

        u = view.update_smoke
        self.show_check.toggled.connect(lambda on: view.update(particles_on=bool(on)))
        self.mode_combo.activated.connect(lambda _i: view.update(particles_style=self.mode_combo.currentData()))
        self.quality_combo.activated.connect(lambda _i: u(quality=self.quality_combo.currentData()))
        self.color_combo.activated.connect(lambda _i: u(color=self.color_combo.currentData()))
        self.color_button.clicked.connect(self._pick_color)
        self.density_slider.valueChanged.connect(lambda v: u(density=v / 100.0))
        self.opacity_slider.valueChanged.connect(lambda v: u(opacity=v / 100.0))
        self.spread.valueChanged.connect(lambda v: u(spread=float(v)))
        self.lifetime.valueChanged.connect(lambda v: u(lifetime=float(v)))
        self.lighting.toggled.connect(lambda on: u(lighting=bool(on)))
        self.light.valueChanged.connect(lambda v: u(light_intensity=v / 100.0))
        self.cancel_button.clicked.connect(view.smoke_layer.cancel)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.voxel_budget.valueChanged.connect(lambda v: u(voxel_budget=int(v)))
        self.seeds.valueChanged.connect(lambda v: u(seeds_per_nozzle=int(v)))
        self.sampling.valueChanged.connect(lambda v: u(sampling_step=float(v)))
        self.sample_distance.metresChanged.connect(lambda v: u(sample_distance=float(v)))
        self.cache_mb.valueChanged.connect(lambda v: u(max_cache_mb=int(v)))
        self.max_tracks.valueChanged.connect(lambda v: u(max_tracks=int(v)))
        self.backend.activated.connect(lambda _i: u(backend=self.backend.currentData()))
        self.transport.activated.connect(lambda _i: u(transport=self.transport.currentData()))
        self.timeline.valueChanged.connect(lambda v: u(timeline_length=float(v)))
        self.noise.toggled.connect(lambda on: u(visual_noise=bool(on)))
        view.changed.connect(self.refresh)
        view.resultsChanged.connect(self.refresh)
        view.smoke_layer.stateChanged.connect(self._refresh_status)
        scene.displayUnitChanged.connect(lambda unit: (self.sample_distance.set_unit(unit), self.refresh()))
        self.sample_distance.set_unit(scene.display_unit)
        self.refresh()

    @staticmethod
    def _slider(lo: int, hi: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        return slider

    def _toggle_advanced(self, on: bool) -> None:
        self.advanced.setVisible(on)
        self.advanced_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def _pick_color(self) -> None:
        r, g, b = self.view.settings.smoke.rgb()
        color = QColorDialog.getColor(QColor.fromRgbF(r, g, b), self, "煙の色")
        if color.isValid():
            self.view.update_smoke(custom_color=(color.redF(), color.greenF(), color.blueF()), color=COLOR_CUSTOM)

    def refresh(self) -> None:
        s = self.view.settings
        smoke = s.smoke
        has = self.view.has_results()
        self.setEnabled(has)
        _set(self.show_check, s.particles_on)
        _set(self.mode_combo, s.particles_style)
        _set(self.quality_combo, smoke.quality)
        _set(self.color_combo, smoke.color)
        _set(self.density_slider, int(round(smoke.density * 100)))
        _set(self.opacity_slider, int(round(smoke.opacity * 100)))
        _set(self.spread, smoke.spread)
        _set(self.lifetime, smoke.lifetime)
        _set(self.lighting, smoke.lighting)
        _set(self.light, int(round(smoke.light_intensity * 100)))
        _set(self.voxel_budget, smoke.voxel_budget)
        _set(self.seeds, smoke.seeds_per_nozzle)
        _set(self.sampling, smoke.sampling_step)
        blocked = self.sample_distance.blockSignals(True)
        self.sample_distance.set_metres(smoke.sample_distance)
        self.sample_distance.blockSignals(blocked)
        _set(self.cache_mb, smoke.max_cache_mb)
        _set(self.max_tracks, smoke.max_tracks)
        _set(self.backend, smoke.backend)
        _set(self.transport, smoke.transport)
        _set(self.timeline, smoke.timeline_length)
        _set(self.noise, smoke.visual_noise)
        volume = s.particles_style in VOLUME_STYLES
        for widget in (self.quality_combo, self.color_combo, self.density_slider, self.opacity_slider, self.spread, self.lifetime,
                       self.lighting, self.light, self.advanced):
            widget.setEnabled(volume)
        self.color_button.setEnabled(volume and smoke.color == COLOR_CUSTOM)
        self._refresh_status()

    def _refresh_status(self) -> None:
        layer = self.view.smoke_layer
        self.progress.setVisible(layer.building)
        self.cancel_button.setVisible(layer.building)
        if layer.building:
            self.progress.setValue(int(layer.progress[0] * 100))
        text = layer.status_text() if layer.active else "「ボリューム煙」を選んで「煙を表示」にすると、煙を計算して表示します。"
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {theme.ERROR};" if layer.error else "")
