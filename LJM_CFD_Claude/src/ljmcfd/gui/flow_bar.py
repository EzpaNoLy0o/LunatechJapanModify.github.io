"""3D ビュー下部の操作バー: 断面スライダー、流線開始位置スライダー、粒子アニメーションのタイムライン。

3 本のスライダーは互いに独立している。どれを動かしても、計算済みの結果から表示を
作り直すだけで、CFD 解析は再実行しない。
"""

from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..post.result_set import AXES, VOLUME_FIELDS
from . import theme
from .analysis_panels import _set, _spin
from .panels import LengthSpin, _label
from .post_view import STYLE_PARTICLE, STYLE_SMOKE, PostView
from .scene import Scene

SLIDER_STEPS = 1000
AXIS_LABELS = (("x", "X 前後"), ("y", "Y 左右"), ("z", "Z 高さ"))
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
#: 再生速度 ×1 のとき、タイムライン全体を実時間で何秒かけて再生するか
BASE_LOOP_SECONDS = 8.0
FRAME_INTERVAL_MS = 33
#: 右端の数値欄・時刻表示の幅 [px]
VALUE_WIDTH = 132


def _slider() -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, SLIDER_STEPS)
    slider.setPageStep(SLIDER_STEPS // 20)
    slider.setMinimumWidth(50)        # 3D ビューが狭いときも左右のパネルを押し潰さない
    return slider


def _axis_combo() -> QComboBox:
    combo = QComboBox()
    for key, text in AXIS_LABELS:
        combo.addItem(text, key)
    return combo


class FlowControlBar(QWidget):
    exportRequested = Signal()

    def __init__(self, scene: Scene, view: PostView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.view = view
        self._direction = 1
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        outer.addLayout(grid)

        # --- 1 行目: 断面 -------------------------------------------------
        self.slice_check = QCheckBox("断面")
        self.slice_axis = _axis_combo()
        self.slice_field = QComboBox()
        for key, info in VOLUME_FIELDS.items():
            self.slice_field.addItem(info.label, key)
        self.slice_slider = _slider()
        self.slice_slider.setToolTip("ドラッグで断面を移動します (解析は再実行しません)。")
        self.slice_position = LengthSpin()
        grid.addWidget(self.slice_check, 0, 0)
        grid.addWidget(self.slice_axis, 0, 1)
        grid.addWidget(self.slice_field, 0, 2)
        grid.addWidget(self.slice_slider, 0, 3)
        grid.addWidget(self.slice_position, 0, 4)

        # --- 2 行目: 流線の開始位置 ---------------------------------------
        self.seed_check = QCheckBox("流線")
        self.seed_axis = _axis_combo()
        self.seed_label = _label("開始位置", "dim")
        self.seed_slider = _slider()
        self.seed_slider.setToolTip("ドラッグで流線の開始位置を移動します。")
        self.seed_position = LengthSpin()
        grid.addWidget(self.seed_check, 1, 0)
        grid.addWidget(self.seed_axis, 1, 1)
        grid.addWidget(self.seed_label, 1, 2)
        grid.addWidget(self.seed_slider, 1, 3)
        grid.addWidget(self.seed_position, 1, 4)

        # --- 3 行目: 粒子・スモークのタイムライン -------------------------
        self.particle_check = QCheckBox("スモーク")
        self.particle_style = QComboBox()
        self.particle_style.addItem("スモーク", STYLE_SMOKE)
        self.particle_style.addItem("粒子", STYLE_PARTICLE)
        self.density_spin = QSpinBox()
        self.density_spin.setRange(1, 60)
        self.density_spin.setPrefix("密度 ")
        self.density_spin.setKeyboardTracking(False)
        self.density_spin.setToolTip("粒子密度: 流線 1 本あたりの粒子数。")
        self.size_spin = _spin(1.0, 30.0, 0, 1.0, " px")
        self.size_spin.setToolTip("粒子の大きさ。")
        look = QHBoxLayout()
        look.setContentsMargins(0, 0, 0, 0)
        look.setSpacing(4)
        look.addWidget(self.density_spin)
        look.addWidget(self.size_spin)
        look_box = QWidget()
        look_box.setLayout(look)
        self.time_slider = _slider()
        self.time_slider.setToolTip("タイムライン: 右へ動かすと粒子が進み、左へ戻すと前の位置に戻ります。")
        self.time_label = QLabel("")
        self.time_label.setMinimumWidth(VALUE_WIDTH)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setProperty("role", "value")
        grid.addWidget(self.particle_check, 2, 0)
        grid.addWidget(self.particle_style, 2, 1)
        grid.addWidget(look_box, 2, 2)
        grid.addWidget(self.time_slider, 2, 3)
        grid.addWidget(self.time_label, 2, 4)
        grid.setColumnStretch(3, 1)

        # --- 4 行目: 再生の操作 -------------------------------------------
        transport = QHBoxLayout()
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(4)
        self.first_button = QPushButton("先頭")
        self.rewind_button = QPushButton("◀ 巻き戻し")
        self.play_button = QPushButton("▶ 再生")
        self.pause_button = QPushButton("一時停止")
        self.stop_button = QPushButton("■ 停止")
        self.first_button.setToolTip("タイムラインの先頭へ移動します。")
        self.rewind_button.setToolTip("時間を逆向きに再生します (粒子が上流へ戻ります)。")
        self.stop_button.setToolTip("再生を止めて先頭へ戻します。")
        self.speed_combo = QComboBox()
        for value in SPEEDS:
            self.speed_combo.addItem(f"× {value:g}", value)
        self.speed_combo.setToolTip("再生速度。")
        self.loop_check = QCheckBox("ループ")
        self.loop_check.setToolTip("ループ再生: 終端まで再生したら先頭から続けます。")
        self.export_button = QPushButton("書き出し…")
        self.export_button.setToolTip("タイムライン 1 周分をアニメーション GIF または PNG 連番として保存します。")
        for button in (self.first_button, self.rewind_button, self.play_button, self.pause_button, self.stop_button,
                       self.export_button):
            button.setFocusPolicy(Qt.NoFocus)
            button.setStyleSheet("padding: 4px 8px;")
        for button in (self.first_button, self.rewind_button, self.play_button, self.pause_button, self.stop_button):
            transport.addWidget(button)
        transport.addSpacing(8)
        transport.addWidget(_label("速度", "dim"))
        transport.addWidget(self.speed_combo)
        transport.addWidget(self.loop_check)
        transport.addStretch(1)
        transport.addWidget(self.export_button)
        outer.addLayout(transport)

        # --- 注記: 定常解析の粒子表示であること (粒子表示中だけ出す) -------
        self.notice = QLabel("定常 RANS の速度場に沿った粒子軌跡です。時間変化する流れ (非定常解析) の結果ではありません。")
        self.notice.setWordWrap(True)          # 幅が足りなければ折り返す (バーの最小幅を広げて左右のパネルを押し潰さない)
        self.notice.setStyleSheet(f"color: {theme.WARN};")
        outer.addWidget(self.notice)

        u = view.update
        self.slice_check.toggled.connect(lambda on: u(slice_on=bool(on)))
        self.slice_axis.activated.connect(lambda _i: u(slice_axis=self.slice_axis.currentData()))
        self.slice_field.activated.connect(lambda _i: u(slice_field=self.slice_field.currentData()))
        self.slice_slider.valueChanged.connect(self._on_slice_slider)
        self.slice_position.metresChanged.connect(view.set_slice_position)
        self.seed_check.toggled.connect(lambda on: u(streamlines_on=bool(on)))
        self.seed_axis.activated.connect(lambda _i: u(seed_slider_axis=self.seed_axis.currentData()))
        self.seed_slider.valueChanged.connect(self._on_seed_slider)
        self.seed_position.metresChanged.connect(lambda v: view.set_seed(**{view.settings.seed_slider_axis: v}))
        self.particle_check.toggled.connect(self._on_particles_toggled)
        self.particle_style.activated.connect(lambda _i: u(particles_style=self.particle_style.currentData()))
        self.time_slider.valueChanged.connect(self._on_time_slider)
        self.first_button.clicked.connect(self.go_to_start)
        self.rewind_button.clicked.connect(self.rewind)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.stop_button.clicked.connect(self.stop)
        self.speed_combo.activated.connect(lambda _i: u(particles_speed=float(self.speed_combo.currentData())))
        self.loop_check.toggled.connect(lambda on: u(particles_loop=bool(on)))
        self.density_spin.valueChanged.connect(lambda v: u(particles_density=float(v)))
        self.size_spin.valueChanged.connect(lambda v: u(particles_size=float(v)))
        self.export_button.clicked.connect(self.exportRequested)

        view.changed.connect(self.refresh)
        view.resultsChanged.connect(self._on_results_changed)
        view.timeChanged.connect(lambda _t: self._refresh_time())
        scene.displayUnitChanged.connect(self._on_display_unit)
        self._on_display_unit(scene.display_unit)
        self._on_results_changed()

    # ------------------------------------------------------------------ 断面・開始位置
    def _on_display_unit(self, unit: str) -> None:
        self.slice_position.set_unit(unit)
        self.seed_position.set_unit(unit)
        self.refresh()

    def _on_results_changed(self) -> None:
        if not self.view.has_results():
            self.pause()
        self.setVisible(self.view.has_results())
        self.refresh()

    def _on_slice_slider(self, step: int) -> None:
        if self.view.results is None:
            return
        lo, hi = self.view.results.axis_range(self.view.settings.slice_axis)
        self.view.set_slice_position(lo + (hi - lo) * step / SLIDER_STEPS)

    def _on_seed_slider(self, step: int) -> None:
        if self.view.results is None:
            return
        axis = self.view.settings.seed_slider_axis
        lo, hi = self.view.results.axis_range(axis)
        # ドラッグ中の途中の値はまとめて、最新の位置だけ流線を引き直す
        self.view.request_seed(**{axis: lo + (hi - lo) * step / SLIDER_STEPS})

    # ------------------------------------------------------------------ 再生
    def is_playing(self) -> bool:
        return self._timer.isActive()

    def play(self) -> None:
        self._start(1)

    def rewind(self) -> None:
        self._start(-1)

    def _start(self, direction: int) -> None:
        if self.view.particle_duration() <= 0:
            return
        duration = self.view.particle_duration()
        time = self.view.settings.particles_time
        if not self.view.settings.particles_loop:
            # ループなしで端に居るときは、反対の端から始める (押しても何も起きない状態を避ける)
            if direction > 0 and time >= duration:
                self.view.set_particle_time(0.0)
            elif direction < 0 and time <= 0.0:
                self.view.set_particle_time(duration)
        self._direction = direction
        self._clock.restart()
        self._timer.start()
        self._refresh_buttons()

    def pause(self) -> None:
        self._timer.stop()
        self._refresh_buttons()

    def stop(self) -> None:
        self._timer.stop()
        self.view.set_particle_time(0.0)
        self._refresh_buttons()

    def go_to_start(self) -> None:
        self.view.set_particle_time(0.0)

    def advance(self, wall_seconds: float) -> None:
        """実時間 ``wall_seconds`` ぶんタイムラインを進める (逆再生中は戻す)。"""
        duration = self.view.particle_duration()
        if duration <= 0:
            self.pause()
            return
        s = self.view.settings
        step = self._direction * wall_seconds * s.particles_speed * duration / BASE_LOOP_SECONDS
        time = s.particles_time + step
        if s.particles_loop:
            time = time % duration
        elif time >= duration or time <= 0.0:
            time = min(max(time, 0.0), duration)
            self.view.set_particle_time(time)
            self.pause()                       # ループなしでは端で止まる
            return
        self.view.set_particle_time(time)

    def _tick(self) -> None:
        elapsed = self._clock.restart() / 1000.0
        self.advance(min(elapsed, 0.25))       # 描画が詰まっても時刻が大きく飛ばないようにする

    def _on_particles_toggled(self, on: bool) -> None:
        if not on:
            self.pause()
        self.view.update(particles_on=bool(on))

    def _on_time_slider(self, step: int) -> None:
        duration = self.view.particle_duration()
        if duration > 0:
            self.view.set_particle_time(duration * step / SLIDER_STEPS)

    # ------------------------------------------------------------------ 表示
    def _refresh_buttons(self) -> None:
        enabled = self.view.particle_duration() > 0
        playing = self.is_playing()
        self.play_button.setEnabled(enabled and not (playing and self._direction > 0))
        self.rewind_button.setEnabled(enabled and not (playing and self._direction < 0))
        self.pause_button.setEnabled(enabled and playing)
        self.stop_button.setEnabled(enabled)
        self.first_button.setEnabled(enabled)
        self.time_slider.setEnabled(enabled)
        self.export_button.setEnabled(enabled)

    def _refresh_time(self) -> None:
        duration = self.view.particle_duration()
        time = self.view.settings.particles_time
        _set(self.time_slider, int(round(time / duration * SLIDER_STEPS)) if duration > 0 else 0)
        if duration > 0:
            self.time_label.setText(f"t = {time:.4f} / {duration:.4f} s")
        else:
            self.time_label.setText("—")

    def refresh(self) -> None:
        s = self.view.settings
        results = self.view.results
        _set(self.slice_check, s.slice_on)
        _set(self.slice_axis, s.slice_axis)
        _set(self.slice_field, s.slice_field)
        self.slice_position.set_metres(s.slice_position)
        _set(self.seed_check, s.streamlines_on)
        _set(self.seed_axis, s.seed_slider_axis)
        axis = s.seed_slider_axis if s.seed_slider_axis in AXES else "y"
        seed_value = getattr(s.seed, axis)
        self.seed_position.set_metres(seed_value)
        if results is not None:
            lo, hi = results.axis_range(s.slice_axis)
            _set(self.slice_slider, int(round((s.slice_position - lo) / (hi - lo) * SLIDER_STEPS)) if hi > lo else 0)
            lo, hi = results.axis_range(axis)
            position = (seed_value - lo) / (hi - lo) if hi > lo else 0.0
            _set(self.seed_slider, int(round(min(max(position, 0.0), 1.0) * SLIDER_STEPS)))
        _set(self.particle_check, s.particles_on)
        _set(self.particle_style, s.particles_style)
        _set(self.speed_combo, s.particles_speed)
        _set(self.loop_check, s.particles_loop)
        _set(self.density_spin, int(round(s.particles_density)))
        _set(self.size_spin, s.particles_size)
        self.notice.setVisible(s.particles_on)
        if not s.particles_on and self.is_playing():
            self._timer.stop()
        self._refresh_time()
        self._refresh_buttons()
