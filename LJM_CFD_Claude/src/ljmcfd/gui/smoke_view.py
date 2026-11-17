"""ボリューム煙の表示 (Level 4) と、煙の計算のバックグラウンド実行。

* ``VolumeRenderer``  … 濃度場を ``vtkVolume`` としてレイキャスティングで描く。色・濃さ・不透明度・陰影は
  伝達関数を変えるだけで、煙の計算はやり直さない。
* ``SmokeLayer``      … ``smoke.engine`` をバックグラウンドのスレッドで動かし、できた煙をタイムラインの時刻に
  合わせて表示する。操作中 (スライダーのドラッグ・再生でコマが間に合わない時) は低解像度の煙を出し、
  操作が終われば本来の解像度へ戻す。

不透明度は Beer–Lambert の形 α = 1 − exp(−κ ρ Δs)。VTK の「単位距離あたりの不透明度」で与えるので、
描画のサンプル間隔 Δs を変えても煙の濃さは変わらない。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

import numpy as np
import vtk
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from vtkmodules.util import numpy_support
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper, vtkGPUVolumeRayCastMapper
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401  (GPU のレイキャスティングを有効にする)

from ..smoke.engine import TIMELINE_FRAMES, Cancelled, SmokeEngine, SmokeState
from ..smoke.settings import BACKEND_CPU, BACKEND_GPU, COLOR_VELOCITY, QUALITIES, SmokeSettings

logger = logging.getLogger("ljmcfd.smoke")

VOLUME_NAME = "post_smoke"
EMITTER_NAME = "post_smoke_emitters"
#: 吸収係数 κ の基準値 [1/m] (濃さ 1.0 のとき)。放出直後の細い煙がほぼ不透明、広がった煙が淡く見える値
KAPPA = 60.0
#: 伝達関数で扱う濃度の上限 (これより濃い所は同じ不透明度)
RHO_TOP = 3.0
RHO_FLOOR = 2e-3        # これより薄い煙は描かない (遠くの裾が霞のように残らないようにする)
SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "gdi generic", "microsoft basic render", "swiftshader", "software")


def opengl_info(render_window) -> dict:
    """OpenGL の実装 (GPU の名前など)。描画後に呼ぶ。"""
    try:
        text = render_window.ReportCapabilities() or ""
    except Exception:  # noqa: BLE001 - 情報が取れなくても表示は続ける
        text = ""
    info = {}
    for line in text.splitlines():
        for key, label in (("vendor", "OpenGL vendor string:"), ("renderer", "OpenGL renderer string:"), ("version", "OpenGL version string:")):
            if line.strip().startswith(label):
                info[key] = line.split(":", 1)[1].strip()
    info["software"] = any(name in info.get("renderer", "").lower() for name in SOFTWARE_RENDERERS)
    return info


class VolumeRenderer:
    """濃度場 (と、流速で色付けする場合は流速場) を 1 つの ``vtkVolume`` で描く。"""

    def __init__(self, plotter) -> None:
        self.plotter = plotter
        self.volume: vtk.vtkVolume | None = None
        self.mapper = None
        self.backend = ""
        self.gl = {}
        self._images: dict = {}       # 格子の識別子 → (vtkImageData, numpy の書き込み先)
        self._current = None
        self._components = 1

    # ------------------------------------------------------------------ バックエンド
    def choose_backend(self, requested: str) -> str:
        """GPU が使えればレイキャスティングを GPU で、使えなければ CPU で行う。"""
        render_window = self.plotter.render_window
        if not self.gl and render_window is not None:
            self.gl = opengl_info(render_window)
        if requested == BACKEND_CPU:
            return BACKEND_CPU
        if requested == BACKEND_GPU:
            return BACKEND_GPU
        if self.gl.get("software"):
            # ソフトウェアの OpenGL (GPU なし) では、GPU 用の描画のほうが CPU 用より遅い (開発機で 160 ms 対 31 ms)
            return BACKEND_CPU
        probe = vtkGPUVolumeRayCastMapper()
        try:
            supported = bool(probe.IsRenderSupported(render_window, vtk.vtkVolumeProperty()))
        except Exception:  # noqa: BLE001
            supported = False
        return BACKEND_GPU if supported else BACKEND_CPU

    def _make_mapper(self, backend: str):
        if backend == BACKEND_GPU:
            mapper = vtkGPUVolumeRayCastMapper()
            mapper.SetAutoAdjustSampleDistances(True)
            mapper.SetUseJittering(True)          # サンプル間隔による縞模様を目立たなくする
        else:
            mapper = vtkFixedPointVolumeRayCastMapper()
            mapper.SetAutoAdjustSampleDistances(True)
            mapper.SetLockSampleDistanceToInputSpacing(False)
        mapper.SetBlendModeToComposite()
        return mapper

    # ------------------------------------------------------------------ データ
    def _image(self, grid, components: int):
        key = (grid.origin, grid.spacing, grid.dims, components)
        entry = self._images.get(key)
        if entry is None:
            image = vtk.vtkImageData()
            image.SetDimensions(*grid.dims)
            image.SetSpacing(grid.spacing, grid.spacing, grid.spacing)
            image.SetOrigin(*grid.origin)
            shape = grid.shape + ((components,) if components > 1 else ())
            buffer = np.zeros(shape, dtype=np.float32)
            array = numpy_support.numpy_to_vtk(buffer.reshape(grid.n_voxels, components) if components > 1 else buffer.ravel(),
                                               deep=False, array_type=vtk.VTK_FLOAT)
            array.SetName("smoke")
            image.GetPointData().SetScalars(array)
            entry = (image, buffer, array)
            if len(self._images) >= 4:
                self._images.pop(next(iter(self._images)))
            self._images[key] = entry
        return entry

    def show(self, grid, density, speed, settings: SmokeSettings, color_range, backend_request: str, high_quality: bool = False) -> None:
        """濃度場を表示する。同じ格子なら中身を書き換えるだけ (アクターも伝達関数も作り直さない)。"""
        components = 2 if speed is not None else 1
        image, buffer, array = self._image(grid, components)
        if components == 2:
            np.copyto(buffer[..., 0], speed, casting="unsafe")
            np.copyto(buffer[..., 1], density, casting="unsafe")
        else:
            np.copyto(buffer, density, casting="unsafe")
        array.Modified()
        image.Modified()
        backend = self.choose_backend(backend_request)
        if self.volume is None or backend != self.backend:
            self.remove()
            self.backend = backend
            self.mapper = self._make_mapper(backend)
            self.volume = vtk.vtkVolume()
            self.volume.SetMapper(self.mapper)
            self.volume.SetPickable(False)
            self.volume.SetProperty(vtk.vtkVolumeProperty())
            self.plotter.renderer.add_actor(self.volume, name=VOLUME_NAME, reset_camera=False, render=False, pickable=False)
            self._current = None
            logger.info("煙の描画: %s (%s)", "GPU レイキャスティング" if backend == BACKEND_GPU else "CPU レイキャスティング",
                        self.gl.get("renderer", "OpenGL の情報なし"))
        changed = self._current is not image
        if changed:
            self.mapper.SetInputData(image)
            self._current = image
        if changed or components != self._components or getattr(self, "_spacing", None) != grid.spacing:
            self._components = components
            self._spacing = grid.spacing
            self.apply(settings, color_range, grid.spacing, high_quality)

    def apply(self, settings: SmokeSettings, color_range, spacing: float | None = None, high_quality: bool = False) -> None:
        """色・濃さ・不透明度・陰影 (Level 4)。煙の計算はやり直さない。"""
        if self.volume is None:
            return
        spacing = float(spacing or getattr(self, "_spacing", 0.05))
        prop = self.volume.GetProperty()
        color = vtk.vtkColorTransferFunction()
        if self._components == 2:
            from matplotlib import colormaps

            lo, hi = color_range
            table = colormaps["turbo"](np.linspace(0.0, 1.0, 16))
            for value, rgba in zip(np.linspace(lo, hi, 16), table):
                color.AddRGBPoint(float(value), *map(float, rgba[:3]))
            prop.IndependentComponentsOff()
        else:
            r, g, b = settings.rgb()
            color.AddRGBPoint(0.0, r, g, b)
            color.AddRGBPoint(RHO_TOP, r, g, b)
            prop.IndependentComponentsOn()
        opacity = vtk.vtkPiecewiseFunction()
        kappa = KAPPA * max(float(settings.density), 0.0)
        scale = min(max(float(settings.opacity), 0.0), 1.0)
        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(RHO_FLOOR * 0.5, 0.0)
        for rho in RHO_TOP * np.linspace(0.0, 1.0, 40)[1:] ** 2:
            if rho < RHO_FLOOR:
                continue
            opacity.AddPoint(float(rho), float(scale * (1.0 - np.exp(-kappa * rho * spacing))))
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.SetScalarOpacityUnitDistance(spacing)     # 上の不透明度は「ボクセル 1 個ぶんの距離」あたり
        prop.SetInterpolationTypeToLinear()
        if settings.lighting or high_quality:
            # 密度の勾配による柔らかい陰影。環境光を強めにして、白い煙が灰色に沈まないようにする
            intensity = max(float(settings.light_intensity), 0.0)
            prop.ShadeOn()
            prop.SetAmbient(0.75)
            prop.SetDiffuse(0.4 * intensity)
            prop.SetSpecular(0.0)
        else:
            prop.ShadeOff()
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
        # サンプル間隔: GPU はボクセル幅の半分、CPU はボクセル幅 (見た目はほぼ同じで約 2 倍速い)。高品質ではさらに半分
        factor = (0.5 if self.backend == BACKEND_GPU else 1.0) * (0.5 if high_quality else 1.0)
        sample = float(settings.sample_distance) if settings.sample_distance > 0 else spacing * factor
        if self.mapper is not None:
            self.mapper.SetSampleDistance(sample)
            if isinstance(self.mapper, vtkFixedPointVolumeRayCastMapper):
                self.mapper.SetInteractiveSampleDistance(max(sample * 4, spacing))

    def remove(self) -> None:
        if self.volume is not None:
            self.plotter.remove_actor(VOLUME_NAME, reset_camera=False, render=False)
        self.volume = None
        self.mapper = None
        self._current = None

    @property
    def visible(self) -> bool:
        return self.volume is not None


class _BuildThread(QThread):
    progressed = Signal(int, float, str)
    built = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, engine: SmokeEngine, plan, generation: int, cancel: threading.Event) -> None:
        super().__init__()
        self.engine, self.plan, self.generation, self.cancel = engine, plan, generation, cancel

    def run(self) -> None:  # noqa: D401 - QThread の入口
        try:
            state = self.engine.build(self.plan, progress=lambda f, m: self.progressed.emit(self.generation, f, m), cancel=self.cancel)
        except Cancelled:
            return
        except Exception as exc:  # noqa: BLE001 - GUI へ伝えてログに残す
            logger.exception("煙を計算できません")
            self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")
            return
        self.built.emit(self.generation, state)


class SmokeLayer(QObject):
    """ボリューム煙のレイヤー。``PostView`` が持ち、タイムラインの時刻を渡す。"""

    stateChanged = Signal()      # 計算の開始・進捗・完了・失敗 (状態の表示を更新する)
    ready = Signal()             # 煙ができた (タイムラインの長さが決まった)

    def __init__(self, view) -> None:
        super().__init__(view)
        self.view = view
        self.plotter = view.plotter
        self.engine = SmokeEngine()
        self.renderer = VolumeRenderer(self.plotter)
        self.state: SmokeState | None = None
        self.active = False
        self.high_quality = False
        self.building = False
        self.progress = (0.0, "")
        self.error = ""
        self.last_frame_ms = 0.0
        self.showing_preview = False
        self._generation = 0
        self._threads: list[_BuildThread] = []
        self._cancel: threading.Event | None = None
        self._prefetch_stop: threading.Event | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._time = 0.0
        self._wanted: tuple | None = None      # 順に計算する方式で、まだ計算できていないコマ (index, 流速つきか)
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._check_wanted)

    # ------------------------------------------------------------------ 設定
    @property
    def settings(self) -> SmokeSettings:
        return self.view.settings.smoke

    def set_results(self, results) -> None:
        """解析結果が変わった (読み込み・破棄)。速度場から作り直す。"""
        self._stop_background()
        self.state = None
        self.error = ""
        if results is None:
            self.engine.set_flow(None, 1.0, None)
        else:
            self.engine.set_flow(results.flow, results.free_stream_speed, results.domain)
        self.renderer.remove()
        if self.active:
            self.request_build()

    def ensure_emitters(self) -> bool:
        """放出源が未設定なら、車両の寸法からプリセットで配置する。"""
        settings = self.settings
        if settings.emitters_initialized and settings.emitters:
            return False
        bounds = self.view.model_bounds()
        if bounds is None or not self.engine.has_flow():
            return False
        settings.emitters = self.engine.default_emitters(settings, bounds)
        settings.emitters_initialized = True
        return True

    # ------------------------------------------------------------------ 有効・無効
    def activate(self, high_quality: bool = False) -> None:
        self.active = True
        self.high_quality = bool(high_quality)
        self.ensure_emitters()
        self.request_build()

    def deactivate(self) -> None:
        self.active = False
        self._stop_background()
        self.renderer.remove()
        self.plotter.remove_actor(EMITTER_NAME, reset_camera=False, render=False)
        self.stateChanged.emit()

    def duration(self) -> float:
        return self.state.duration if (self.active and self.state is not None) else 0.0

    # ------------------------------------------------------------------ 計算 (バックグラウンド)
    def request_build(self) -> None:
        """設定に合わせて煙を作り直す。必要な階層だけを、別スレッドで計算する。"""
        if not self.active:
            return
        self._show_emitters()
        settings = self.settings
        if self.high_quality and QUALITIES.index(settings.quality) < QUALITIES.index("high"):
            settings = replace(settings, quality="high")          # 高品質ボリューム煙は High 以上の格子
        plan = self.engine.plan(settings, self.view.model_bounds())
        if plan is None:
            self._stop_background()
            self.state = None
            self.renderer.remove()
            self.error = "" if self.engine.has_flow() else "解析結果がありません"
            if self.engine.has_flow() and not any(e.enabled for e in self.settings.emitters):
                self.error = "有効な放出源がありません"
            self.stateChanged.emit()
            self.view.smoke_ready()
            return
        if self.state is not None and self.state.plan == plan and not self.building:
            self.show_time(self._time)
            return
        self._stop_background()
        self._generation += 1
        self._cancel = threading.Event()
        thread = _BuildThread(self.engine, plan, self._generation, self._cancel)
        thread.progressed.connect(self._on_progress)
        thread.built.connect(self._on_built)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(self._on_thread_finished)
        self._threads.append(thread)
        self.building = True
        self.error = ""
        self.progress = (0.0, "準備中…")
        self.stateChanged.emit()
        thread.start()

    def _on_progress(self, generation: int, fraction: float, message: str) -> None:
        if generation == self._generation:
            self.progress = (fraction, message)
            self.stateChanged.emit()

    def _on_built(self, generation: int, state: SmokeState) -> None:
        if generation != self._generation or not self.active:
            return
        self.building = False
        self.state = state
        self.error = ""
        stats = state.stats
        logger.info("煙を準備しました: 格子 %s (%.0f mm)、流跡線 %d 本、%.2f 秒", "×".join(map(str, state.grid.dims)),
                    state.grid.spacing * 1000, stats.get("tracks", 0), stats.get("seconds_total", 0.0))
        self._time = min(max(self.view.settings.particles_time, 0.0), state.duration)
        self.view.settings.particles_time = self._time
        self.show_time(self._time)
        self._start_prefetch()
        self.stateChanged.emit()
        self.ready.emit()
        self.view.smoke_ready()

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.building = False
        self.state = None
        self.error = message
        self.renderer.remove()
        self.stateChanged.emit()
        self.view.smoke_ready()

    def _on_thread_finished(self) -> None:
        self._threads = [t for t in self._threads if t.isRunning()]

    def _stop_background(self) -> None:
        self._wanted = None
        self._poll.stop()
        if self._cancel is not None:
            self._cancel.set()
        self._generation += 1
        self.building = False
        self._stop_prefetch()

    def cancel(self) -> None:
        """計算中の煙を中止する (表示中の煙はそのまま)。"""
        if self.building:
            self._stop_background()
            self.error = "計算を中止しました"
            self.stateChanged.emit()
            self.view.smoke_ready()

    def shutdown(self) -> None:
        """終了時: バックグラウンドの計算を止め、終わるのを待つ。"""
        self._stop_background()
        for thread in list(self._threads):
            thread.wait(30000)
        self._threads.clear()

    # ------------------------------------------------------------------ 先読み
    def _start_prefetch(self, start: int | None = None) -> None:
        """タイムラインのコマを、今の時刻 (``start``) から先へ順にキャッシュへ用意する。

        どの時刻も直接求まる方式 (流跡線) はキャッシュに収まる数まで。順にしか求まらない方式 (受動スカラー) は
        全コマを順に計算する (古いコマはキャッシュから外れても、保存した状態から速く計算し直せる)。
        """
        self._stop_prefetch()
        state = self.state
        if state is None:
            return
        smoke = state.smoke
        frame_bytes = smoke.grid.n_voxels * 2 * (2 if self._with_speed() else 1)
        sequential = getattr(smoke, "sequential", False)
        count = TIMELINE_FRAMES if sequential else int(min(TIMELINE_FRAMES, max(smoke.cache_bytes // max(frame_bytes, 1) - 2, 0)))
        if count <= 0:
            return
        if start is None:
            start = smoke.frame_index(self._time, TIMELINE_FRAMES)
        stop = threading.Event()
        with_speed = self._with_speed()

        def run() -> None:
            for step in range(count):
                if stop.is_set():
                    return
                smoke.cached_frame((start + step) % TIMELINE_FRAMES, TIMELINE_FRAMES, with_speed)

        self._prefetch_stop = stop
        self._prefetch_thread = threading.Thread(target=run, name="smoke-prefetch", daemon=True)
        self._prefetch_thread.start()

    def _stop_prefetch(self) -> None:
        if self._prefetch_stop is not None:
            self._prefetch_stop.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
        self._prefetch_stop = None
        self._prefetch_thread = None

    # ------------------------------------------------------------------ 表示
    def _with_speed(self) -> bool:
        return self.settings.color == COLOR_VELOCITY

    def show_time(self, time_value: float, interactive: bool = False, block: bool = False) -> None:
        """時刻 ``time_value`` の煙を表示する。

        interactive: 操作中 (ドラッグ・再生)。本来の解像度のコマがキャッシュに無ければ、待たずに低解像度の煙を出す。
        block:       必ずその時刻のコマを計算して表示する (動画の書き出し用)。
        """
        self._time = float(time_value)
        state = self.state
        if not self.active or state is None:
            return
        started = time.perf_counter()
        with_speed = self._with_speed()
        index = state.smoke.frame_index(self._time, TIMELINE_FRAMES)
        item = state.smoke.peek_frame(index, TIMELINE_FRAMES, with_speed)
        source = state.smoke
        if item is None and getattr(state.smoke, "sequential", False) and not block:
            # 受動スカラー方式は先頭から順にしか計算できない。画面のスレッドでは計算せず、
            # バックグラウンドの計算がそのコマに届いたら表示する (それまでは今の表示のまま)
            if self._wanted != (index, with_speed):
                self._wanted = (index, with_speed)
                self._start_prefetch(index)            # 求められたコマから計算する
            self._poll.start()
            self.stateChanged.emit()
            return
        self._wanted = None
        if item is None and interactive:
            source = state.preview
            item = source.cached_frame(source.frame_index(self._time, TIMELINE_FRAMES), TIMELINE_FRAMES, with_speed)
        elif item is None:
            item = state.smoke.cached_frame(index, TIMELINE_FRAMES, with_speed)
        density, speed = item if with_speed else (item, None)
        self.showing_preview = source is state.preview
        self.renderer.show(source.grid, density, speed, self.settings, self._speed_range(), self.settings.backend, self.high_quality)
        self.last_frame_ms = (time.perf_counter() - started) * 1000.0

    def _check_wanted(self) -> None:
        if self._wanted is None or self.state is None or not self.active:
            self._poll.stop()
            return
        index, with_speed = self._wanted
        if self.state.smoke.peek_frame(index, TIMELINE_FRAMES, with_speed) is not None:
            self._poll.stop()
            self.show_time(self._time)
            self.view.viewport.request_render()
            self.stateChanged.emit()

    def refresh_appearance(self) -> None:
        """色・濃さ・不透明度・陰影の変更 (Level 4 だけ)。流速で色付けに切り替えたときは、流速場のコマに替える。"""
        if self.state is None or not self.active:
            return
        if self.renderer._components != (2 if self._with_speed() else 1):
            self._start_prefetch()
            self.show_time(self._time)
            return
        grid = self.state.preview.grid if self.showing_preview else self.state.grid
        self.renderer.apply(self.settings, self._speed_range(), grid.spacing, self.high_quality)

    def _speed_range(self) -> tuple[float, float]:
        lo, hi = self.view.color_range("Umag") if self.view.results is not None else (0.0, 1.0)
        return (float(lo), float(hi) if hi > lo else float(lo) + 1.0)

    def _show_emitters(self) -> None:
        """放出源のノズルの位置を小さな点で示す (配置の確認用。煙の計算には影響しない)。"""
        import pyvista as pv

        self.plotter.remove_actor(EMITTER_NAME, reset_camera=False, render=False)
        points = [e.nozzle_points() for e in self.settings.emitters if e.enabled]
        if not points or not self.active:
            return
        cloud = pv.PolyData(np.vstack(points))
        self.plotter.add_mesh(cloud, name=EMITTER_NAME, color=self.view.viewport.colors["measure"], point_size=6.0,
                              render_points_as_spheres=True, pickable=False, reset_camera=False, render=False)

    def status_text(self) -> str:
        if not self.active:
            return ""
        if self.error:
            return f"煙を表示できません: {self.error}"
        if self.building:
            fraction, message = self.progress
            return f"{message} ({fraction * 100:.0f} %)"
        if self.state is None:
            return "—"
        if self._wanted is not None:
            done = sum(1 for k in range(TIMELINE_FRAMES) if self.state.smoke.peek_frame(k, TIMELINE_FRAMES, self._wanted[1]) is not None)
            return f"受動スカラー方式: 先頭から順に計算中 ({done} / {TIMELINE_FRAMES} コマ)。計算が届くと表示します。"
        grid = self.state.grid
        info = self.state.smoke.cache_info()
        backend = "GPU" if self.renderer.backend == BACKEND_GPU else "CPU"
        stats = self.state.stats
        if "cfl" in stats:
            method = f"受動スカラー方式 (CFL {stats['cfl']:.2f}、数値拡散の目安 {stats['numerical_diffusion']:.3f} m²/s)"
        else:
            method = f"流跡線 {stats.get('tracks', 0):,} 本"
        return (f"格子 {'×'.join(map(str, grid.dims))} (ボクセル {grid.spacing * 1000:.0f} mm)、{method}、"
                f"描画 {backend}、キャッシュ {info['frames']} コマ / {info['bytes'] / 1e6:.0f} MB")
