"""解析結果の 3D 表示: 表面分布・断面・流線・等値面のレイヤーを 3D ビューに重ねる。

表示の更新は、読み込み済みの結果からの抽出だけで行う (CFD は再実行しない)。
断面は位置を変えるたびにアクターを作り直さず、同じメッシュの中身だけを差し替えるので、
スライダーへ滑らかに追従する。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QTimer, Signal

from ..post.particles import AnimationSource, SteadyPathlines
from ..post.result_set import AXES, COLORMAPS, SURFACE_FIELDS, VOLUME_FIELDS, FieldInfo, ResultSet, SeedSpec, default_seed
from .scene import Scene
from .viewport import Viewport

logger = logging.getLogger("ljmcfd.post")

SOLID = "solid"          # 流線・等値面を物理量で色分けせず単色で描く
LAYERS = ("surface", "slice", "streamlines", "iso", "particles")
STYLE_PARTICLE = "particle"   # 点の粒子
STYLE_SMOKE = "smoke"         # 尾を引く半透明の筋 (煙のように見える)
SMOKE_TRAIL_POINTS = 12
#: スモークの筋 1 本の長さ (タイムライン全体に対する比)
SMOKE_TRAIL_FRACTION = 0.07
#: 定常解析の粒子表示に必ず添える注記 (3D ビュー内。スクリーンショットや動画にも残る)
STEADY_NOTICE = "Steady RANS result: particles follow the steady velocity field (not a transient simulation)"


@dataclass
class ColorScale:
    """物理量ごとのカラースケール。自動のときは結果から決めた範囲を使う。"""

    auto: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    colormap: str = ""      # 空なら物理量の既定のカラーマップ


@dataclass
class PostSettings:
    surface_on: bool = False
    surface_field: str = "pressure"
    surface_ground: bool = False
    slice_on: bool = False
    slice_field: str = "Umag"
    slice_axis: str = "y"
    slice_position: float = 0.0
    slice_opacity: float = 1.0
    streamlines_on: bool = False
    streamlines_color: str = "Umag"
    streamlines_width: float = 2.0
    seed: SeedSpec = field(default_factory=SeedSpec)
    seed_initialized: bool = False      # 開始面を車両前面へ自動配置済みか
    iso_on: bool = False
    iso_field: str = "Umag"         # 既定は流速 (主流の半分の速さ = 後流の外形)。Q 値は粗いメッシュでは壁面近くにノイズが出やすい
    iso_value: float = 0.0
    iso_color: str = "Umag"
    iso_opacity: float = 0.6
    iso_initialized: bool = False
    particles_on: bool = False
    particles_style: str = STYLE_SMOKE
    particles_density: float = 8.0      # 流線 1 本あたりの粒子数
    particles_size: float = 7.0         # 点の大きさ [px]
    particles_time: float = 0.0         # タイムラインの現在時刻 [s] (流れの時間)
    particles_loop: bool = True
    particles_speed: float = 1.0        # 再生速度の倍率
    seed_slider_axis: str = "y"         # 下部スライダーで動かす開始位置の軸
    scales: dict[str, ColorScale] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seed"] = self.seed.to_dict()
        data["scales"] = {key: asdict(scale) for key, scale in self.scales.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "PostSettings":
        data = dict(data or {})
        settings = cls()
        seed = data.pop("seed", None)
        scales = data.pop("scales", None) or {}
        for name, default in asdict(settings).items():
            if name in ("seed", "scales") or name not in data:
                continue
            value = data[name]
            value = bool(value) if isinstance(default, bool) else float(value) if isinstance(default, float) else str(value)
            setattr(settings, name, value)
        settings.seed = SeedSpec.from_dict(seed)
        for key, raw in scales.items():
            settings.scales[str(key)] = ColorScale(
                auto=bool(raw.get("auto", True)), vmin=float(raw.get("vmin", 0.0)), vmax=float(raw.get("vmax", 1.0)),
                colormap=str(raw.get("colormap", "")),
            )
        return settings


class PostView(QObject):
    """結果表示の制御。設定を持ち、変更に応じて 3D ビューのアクターを更新する。"""

    changed = Signal()            # 設定が変わった (パネルの再表示・保存対象)
    resultsChanged = Signal()     # 表示対象の結果が変わった
    timeChanged = Signal(float)   # 粒子アニメーションの時刻が変わった (毎フレーム出るので軽い受け手だけがつなぐ)

    def __init__(self, scene: Scene, viewport: Viewport, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.viewport = viewport
        self.plotter = viewport.plotter
        self.settings = PostSettings()
        self.results: ResultSet | None = None
        self._slice_mesh: pv.PolyData | None = None
        self._slice_key: tuple | None = None
        self._streamlines: pv.PolyData | None = None
        self._streamlines_seed: SeedSpec | None = None
        self._animation: AnimationSource | None = None
        self._animation_key: tuple | None = None
        self._pathlines: pv.PolyData | None = None
        self._pathlines_key: tuple | None = None
        self._cloud: pv.PolyData | None = None
        self._pending_seed: dict = {}
        self._bars: list[str] = []
        scene.solverChanged.connect(self.reload)
        scene.cleared.connect(self._on_cleared)
        viewport.themeChanged.connect(lambda: self.refresh_all(recompute=False))     # 色だけの変更

    # ------------------------------------------------------------------ 結果の読み込み
    def reload(self) -> None:
        """シーンの解析結果に合わせて表示対象を切り替える。"""
        summary = self.scene.solver_summary
        self.results = None
        self._slice_mesh = self._slice_key = self._streamlines = self._streamlines_seed = None
        self._animation = self._animation_key = self._cloud = None
        self._pathlines = self._pathlines_key = None
        if summary and summary.get("files", {}).get("fields"):
            try:
                candidate = ResultSet(Path(summary["results_dir"]), summary)
                self.results = candidate if candidate.has_fields else None
            except (OSError, KeyError) as exc:
                logger.error("解析結果を読み込めません: %s", exc)
        if self.results is not None:
            self._initialize_defaults()
        self.refresh_all()
        self.resultsChanged.emit()
        self.changed.emit()

    def _on_cleared(self) -> None:
        self.settings = PostSettings()
        self.reload()

    def has_results(self) -> bool:
        return self.results is not None

    def model_bounds(self) -> tuple[float, ...] | None:
        """流れ場の座標系でのモデルの境界 (メッシュ生成時の沈み込みを反映)。"""
        report = self.scene.mesh_report
        if report is None:
            return self.scene.analysis_bounds()
        b = list(report["model_bounds"])
        b[4] -= report.get("sink", 0.0)
        b[5] -= report.get("sink", 0.0)
        return tuple(b)

    def _initialize_defaults(self) -> None:
        """初めて結果を表示するとき、断面位置・流線の開始面・等値面の値を妥当な位置に置く。"""
        s = self.settings
        bounds = self.model_bounds() or self.results.domain
        if not s.seed_initialized:
            s.seed = self._default_seed()
            index = AXES.index(s.slice_axis)
            s.slice_position = (bounds[2 * index] + bounds[2 * index + 1]) / 2
            s.seed_initialized = True
        if not s.iso_initialized:
            s.iso_value = self.default_iso_value(s.iso_field)
            s.iso_initialized = True

    def _default_seed(self) -> SeedSpec:
        """流線の開始面の既定位置: 車両前面の少し上流。必ず計算領域の内側に収める。"""
        domain = self.results.domain
        seed = default_seed(self.model_bounds() or domain, ground=bool(self.scene.tunnel.has_ground))
        margin = 0.02 * (domain[1] - domain[0])
        seed.x = min(max(seed.x, domain[0] + margin), domain[1] - margin)
        # 開始面が領域からはみ出す場合は、領域に収まる大きさへ縮める
        y_lo, y_hi = max(seed.y - seed.width / 2, domain[2]), min(seed.y + seed.width / 2, domain[3])
        z_lo, z_hi = max(seed.z - seed.height / 2, domain[4]), min(seed.z + seed.height / 2, domain[5])
        pad_y, pad_z = 0.02 * (y_hi - y_lo), 0.02 * (z_hi - z_lo)
        seed.y, seed.width = (y_lo + y_hi) / 2, (y_hi - y_lo) - 2 * pad_y
        seed.z, seed.height = (z_lo + z_hi) / 2, (z_hi - z_lo) - 2 * pad_z
        return seed

    def default_iso_value(self, field_key: str) -> float:
        if field_key == "Q":
            # Q 値は (代表速度 / 代表長さ)² の桁。車高 H を代表長さとして 10 (U/H)² を初期値にする
            # (デモ車両の実データで、車体まわりの渦構造が見え、遠方のノイズが消える値)
            bounds = self.model_bounds() or self.results.domain
            height = max(bounds[5] - bounds[4], 1e-6)
            return float(10.0 * (self.results.free_stream_speed / height) ** 2)
        lo, hi = self.results.default_range(field_key)
        if field_key == "Umag":
            return 0.5 * self.results.free_stream_speed      # 主流の半分の速さ = 後流・剥離域の外形
        return 0.5 * (lo + hi)

    def iso_range(self, field_key: str) -> tuple[float, float]:
        """等値面の値を動かせる範囲。"""
        if self.results is None:
            return (0.0, 1.0)
        if field_key == "Q":
            return (0.0, 10.0 * self.default_iso_value("Q"))
        return self.results.default_range(field_key)

    # ------------------------------------------------------------------ カラースケール
    def scale(self, key: str, surface: bool = False) -> ColorScale:
        return self.settings.scales.setdefault(key, ColorScale())

    def color_range(self, key: str, surface: bool = False) -> tuple[float, float]:
        scale = self.scale(key)
        if scale.auto or scale.vmax <= scale.vmin:
            return self.results.default_range(key, surface=surface and key not in VOLUME_FIELDS) if self.results else (0.0, 1.0)
        return (scale.vmin, scale.vmax)

    def colormap(self, key: str) -> str:
        info = VOLUME_FIELDS.get(key) or SURFACE_FIELDS[key]
        chosen = self.scale(key).colormap
        return chosen if chosen in COLORMAPS else info.colormap

    def set_color_scale(self, key: str, auto: bool | None = None, vmin: float | None = None, vmax: float | None = None,
                        colormap: str | None = None) -> None:
        scale = self.scale(key)
        if auto is not None:
            if scale.auto and not auto:      # 手動へ切り替えた瞬間は、現在の自動範囲から始める
                scale.vmin, scale.vmax = self.color_range(key, surface=key in SURFACE_FIELDS)
            scale.auto = bool(auto)
        if vmin is not None:
            scale.vmin = float(vmin)
        if vmax is not None:
            scale.vmax = float(vmax)
        if colormap is not None:
            scale.colormap = colormap
        self.refresh_all()
        self.changed.emit()

    def _info(self, key: str) -> FieldInfo:
        return VOLUME_FIELDS.get(key) or SURFACE_FIELDS[key]

    def _mapping(self, key: str, surface: bool = False) -> dict:
        return dict(scalars=key, cmap=self.colormap(key), clim=self.color_range(key, surface), show_scalar_bar=False,
                    interpolate_before_map=True)

    # ------------------------------------------------------------------ 設定の変更
    def update(self, **values) -> None:
        """設定を変更し、関係するレイヤーだけを更新する。"""
        s = self.settings
        touched: set[str] = set()
        for name, value in values.items():
            if not hasattr(s, name):
                raise AttributeError(name)
            if getattr(s, name) == value:
                continue
            setattr(s, name, value)
            touched.add(name.split("_")[0])
            if name == "iso_field" and self.results is not None:
                s.iso_value = self.default_iso_value(value)
            if name == "slice_axis":
                bounds = self.model_bounds() or (self.results.domain if self.results else None)
                if bounds:
                    index = AXES.index(value)
                    s.slice_position = (bounds[2 * index] + bounds[2 * index + 1]) / 2
        if not touched:
            return
        if "surface" in touched:
            self._refresh_surface()
        if "slice" in touched:
            self._refresh_slice()
        if "streamlines" in touched:
            self._refresh_streamlines(recompute=False)
        if "iso" in touched:
            self._refresh_iso()
        if "particles" in touched:
            if set(values) <= {"particles_loop", "particles_speed"}:
                self._update_particle_frame()         # 再生方法だけの変更では作り直さない
            else:
                self._refresh_particles()
        self._rebuild_scalar_bars()
        self.viewport.request_render()
        self.changed.emit()

    def set_slice_position(self, position: float) -> None:
        """断面を動かす。スライダーから高頻度で呼ばれるので、メッシュの中身だけを差し替える。"""
        if self.results is not None:
            lo, hi = self.results.axis_range(self.settings.slice_axis)
            position = min(max(float(position), lo), hi)
        if position == self.settings.slice_position:
            return
        self.settings.slice_position = float(position)
        if self.settings.slice_on and self.results is not None:
            self._update_slice_geometry()
            self.viewport.request_render()
        self.changed.emit()

    def set_seed(self, **values) -> None:
        """流線の開始面を変更して流線を引き直す。"""
        seed = self.settings.seed
        changed = False
        for name, value in values.items():
            if getattr(seed, name) != value:
                setattr(seed, name, value)
                changed = True
        if not changed:
            return
        self._refresh_streamlines(recompute=True)
        self._refresh_particles()
        self.viewport.request_render()
        self.changed.emit()

    def request_seed(self, **values) -> None:
        """スライダー用: 開始面の変更を予約する。

        流線の引き直しには 0.2 秒程度かかるので、ドラッグ中に届いた途中の値はまとめ、
        最新の値だけを処理する (操作に遅れず追従させるため)。
        """
        first = not self._pending_seed
        self._pending_seed.update(values)
        if first:
            QTimer.singleShot(0, self._apply_pending_seed)

    def _apply_pending_seed(self) -> None:
        values, self._pending_seed = self._pending_seed, {}
        if values:
            self.set_seed(**values)

    def reset_seed(self) -> None:
        """開始面を車両前面の既定位置へ戻す。"""
        if self.results is None:
            return
        self.settings.seed = self._default_seed()
        self._refresh_streamlines(recompute=True)
        self._refresh_particles()
        self.viewport.request_render()
        self.changed.emit()

    def apply_settings(self, settings: PostSettings) -> None:
        self.settings = settings
        self._slice_mesh = self._slice_key = self._streamlines = self._streamlines_seed = None
        self._animation = self._animation_key = self._cloud = None
        self._pathlines = self._pathlines_key = None
        if self.results is not None:
            self._initialize_defaults()
        self.refresh_all()
        self.changed.emit()

    # ------------------------------------------------------------------ 描画
    def refresh_all(self, recompute: bool = True) -> None:
        """全レイヤーを描き直す。``recompute=False`` なら流線は積分し直さない (配色の変更など)。"""
        self._refresh_surface()
        self._refresh_slice()
        self._refresh_streamlines(recompute=recompute)
        self._refresh_iso()
        self._refresh_particles()
        self._rebuild_scalar_bars()
        self.viewport.request_render()

    def _remove(self, *names: str) -> None:
        for name in names:
            self.plotter.remove_actor(name, reset_camera=False, render=False)

    def _common(self) -> dict:
        return dict(pickable=False, reset_camera=False, render=False)

    def _refresh_surface(self) -> None:
        self._remove("post_surface", "post_surface_ground")
        s = self.settings
        active = s.surface_on and self.results is not None
        self.viewport.set_cad_hidden(active)      # 表面分布の表示中は、重なって見える CAD 側を隠す
        if not active:
            return
        parts = [surface for surface in self.results.part_surfaces().values() if s.surface_field in surface.point_data]
        if parts:
            merged = parts[0] if len(parts) == 1 else pv.merge(parts, merge_points=False)
            self.plotter.add_mesh(merged, name="post_surface", smooth_shading=True, **self._mapping(s.surface_field, True),
                                  **self._common())
        ground = self.results.ground_surface() if s.surface_ground else None
        if ground is not None and s.surface_field in ground.point_data:
            self.plotter.add_mesh(ground, name="post_surface_ground", **self._mapping(s.surface_field, True), **self._common())

    def _refresh_slice(self) -> None:
        self._remove("post_slice")
        self._slice_mesh = self._slice_key = None
        s = self.settings
        if not (s.slice_on and self.results is not None):
            return
        self._slice_mesh = pv.PolyData()
        self._slice_mesh.shallow_copy(self.results.slice(s.slice_axis, s.slice_position))
        if self._slice_mesh.n_points == 0:
            # 空のメッシュからは色付けの設定を作れないので、1 点だけのダミーで初期化する
            self._slice_mesh = pv.PolyData(np.zeros((1, 3)))
            self._slice_mesh.point_data[s.slice_field] = np.zeros(1)
        self.plotter.add_mesh(self._slice_mesh, name="post_slice", lighting=False, opacity=s.slice_opacity,
                              **self._mapping(s.slice_field), **self._common())
        self._slice_key = (s.slice_axis, s.slice_field)

    def _update_slice_geometry(self) -> None:
        s = self.settings
        if self._slice_mesh is None or self._slice_key != (s.slice_axis, s.slice_field):
            self._refresh_slice()
            return
        self._slice_mesh.shallow_copy(self.results.slice(s.slice_axis, s.slice_position))
        if s.slice_field in self._slice_mesh.point_data:
            self._slice_mesh.set_active_scalars(s.slice_field)

    def _refresh_streamlines(self, recompute: bool) -> None:
        self._remove("post_streamlines", "post_seed")
        s = self.settings
        if not (s.streamlines_on and self.results is not None):
            return
        if recompute or self._streamlines is None:
            self._streamlines = self.results.streamlines(s.seed)
            self._streamlines_seed = replace(s.seed)
        corners = s.seed.outline()
        outline = pv.PolyData(corners, lines=np.array([5, 0, 1, 2, 3, 0]))
        self.plotter.add_mesh(outline, name="post_seed", color=self.viewport.colors["measure"], line_width=2.0, lighting=False,
                              **self._common())
        lines = self._streamlines
        if lines.n_points == 0:
            return
        if s.streamlines_color == SOLID or s.streamlines_color not in lines.point_data:
            self.plotter.add_mesh(lines, name="post_streamlines", color=self.viewport.colors["text"], line_width=s.streamlines_width,
                                  lighting=False, scalars=None, **self._common())
        else:
            self.plotter.add_mesh(lines, name="post_streamlines", line_width=s.streamlines_width, lighting=False,
                                  **self._mapping(s.streamlines_color), **self._common())

    def _refresh_iso(self) -> None:
        self._remove("post_iso")
        s = self.settings
        if not (s.iso_on and self.results is not None):
            return
        surface = self.results.isosurface(s.iso_field, s.iso_value)
        if surface.n_points == 0:
            return
        if s.iso_color == SOLID or s.iso_color not in surface.point_data:
            self.plotter.add_mesh(surface, name="post_iso", color=(0.75, 0.85, 0.95), opacity=s.iso_opacity, smooth_shading=True,
                                  scalars=None, **self._common())
        else:
            self.plotter.add_mesh(surface, name="post_iso", opacity=s.iso_opacity, smooth_shading=True,
                                  **self._mapping(s.iso_color), **self._common())

    def active_color_fields(self) -> list[tuple[str, bool]]:
        """いま色分けに使っている物理量 (重複なし、表示順)。(キー, 壁面量か) のリスト。"""
        s = self.settings
        if self.results is None:
            return []
        fields: list[tuple[str, bool]] = []
        if s.surface_on:
            fields.append((s.surface_field, True))
        if s.slice_on:
            fields.append((s.slice_field, False))
        if s.streamlines_on and s.streamlines_color != SOLID:
            fields.append((s.streamlines_color, False))
        if s.iso_on and s.iso_color != SOLID:
            fields.append((s.iso_color, False))
        if s.particles_on:
            fields.append(("Umag", False))
        unique: list[tuple[str, bool]] = []
        for key, surface in fields:
            if key not in [k for k, _ in unique]:
                unique.append((key, surface))
        return unique

    def _rebuild_scalar_bars(self) -> None:
        """色分け中の物理量ごとにカラーバーを 1 本ずつ、画面右側に並べる。"""
        for title in self._bars:
            try:
                self.plotter.remove_scalar_bar(title, render=False)
            except (KeyError, ValueError):
                pass
        self._bars = []
        text = self.viewport.colors["text"]
        for index, (key, surface) in enumerate(self.active_color_fields()[:3]):
            info = self._info(key)
            lo, hi = self.color_range(key, surface)
            # カラーバー用に、同じ範囲・カラーマップを持つ見えないメッシュを使う (どのレイヤーが消えても残るように)
            dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
            dummy.point_data[key] = np.array([lo, hi])
            actor = self.plotter.add_mesh(dummy, name=f"post_bar_{index}", scalars=key, cmap=self.colormap(key), clim=(lo, hi),
                                          show_scalar_bar=False, opacity=0.0, point_size=0.0, **self._common())
            self.plotter.add_scalar_bar(
                title=info.title, mapper=actor.mapper, vertical=True, n_labels=5, fmt="%.3g", color=text,
                position_x=0.88 - 0.15 * index, position_y=0.22, width=0.05, height=0.55,
                title_font_size=12, label_font_size=11, render=False,
            )
            self._bars.append(info.title)
        for index in range(len(self._bars), 3):
            self._remove(f"post_bar_{index}")

    # ------------------------------------------------------------------ 確認用
    def layer_actors(self) -> dict[str, bool]:
        """レイヤーごとに、アクターが 3D ビューにあるか (テスト・確認用)。"""
        actors = self.plotter.actors
        return {"surface": "post_surface" in actors, "slice": "post_slice" in actors,
                "streamlines": "post_streamlines" in actors, "iso": "post_iso" in actors,
                "particles": "post_particles" in actors}

    # ------------------------------------------------------------------ 粒子・スモーク
    def animation(self) -> AnimationSource | None:
        """粒子アニメーションのデータ (事前計算のキャッシュ)。

        2 段階で使い回す:

        * 流跡線 (流線の積分。重い) … 結果と開始面が同じ間
        * 粒子の標本化 (軽い)        … さらに密度・表示形式が同じ間

        時刻・再生速度・ループの変更ではどちらも作り直さない。
        """
        if self.results is None:
            return None
        s = self.settings
        seed = replace(s.seed, both_directions=False)       # 粒子は開始面から下流へ流す
        density = max(int(round(s.particles_density)), 1)
        trail = SMOKE_TRAIL_POINTS if s.particles_style == STYLE_SMOKE else 1
        lines_key = (id(self.results), tuple(asdict(seed).values()))
        key = (lines_key, density, trail)
        if self._animation is None or self._animation_key != key:
            if self._pathlines is None or self._pathlines_key != lines_key:
                if self._streamlines is not None and self._streamlines_seed == seed:
                    self._pathlines = self._streamlines      # 表示中の流線をそのまま使う
                else:
                    self._pathlines = self.results.streamlines(seed)
                self._pathlines_key = lines_key
            self._animation = SteadyPathlines(self._pathlines, particles_per_line=density, trail_points=trail,
                                              trail_fraction=SMOKE_TRAIL_FRACTION)
            self._animation_key = key
        return self._animation

    def particle_duration(self) -> float:
        """タイムラインの長さ [s]。粒子表示が無効・結果なしのときは 0。"""
        if not self.settings.particles_on:
            return 0.0
        source = self.animation()
        return source.duration if source is not None else 0.0

    def set_particle_time(self, time: float) -> None:
        """タイムラインの時刻を設定する。OpenFOAM も流線の積分も走らず、キャッシュから粒子位置を求めるだけ。"""
        duration = self.particle_duration()
        time = min(max(float(time), 0.0), duration) if duration > 0 else 0.0
        if time == self.settings.particles_time:
            return
        self.settings.particles_time = time
        self._update_particle_frame()
        self.viewport.request_render()
        self.timeChanged.emit(time)

    def _refresh_particles(self) -> None:
        self._remove("post_particles", "post_notice")
        self._cloud = None
        s = self.settings
        if not (s.particles_on and self.results is not None):
            return
        source = self.animation()
        if source is None or source.n_particles == 0:
            return
        s.particles_time = min(max(s.particles_time, 0.0), source.duration)
        frame = source.frame(s.particles_time, s.particles_loop)
        if s.particles_style == STYLE_SMOKE:
            # スモーク: 粒子ごとの尾を、後ろほど薄くなる半透明の筋として描く
            self._cloud = pv.PolyData(frame.points.copy(), lines=source.trail_connectivity())
            self._cloud.point_data["rgba"] = self._particle_colors(frame)
            self.plotter.add_mesh(self._cloud, name="post_particles", scalars="rgba", rgba=True, line_width=s.particles_size,
                                  lighting=False, **self._common())
        else:
            self._cloud = pv.PolyData(frame.points.copy())
            self._cloud.point_data["rgba"] = self._particle_colors(frame)
            self.plotter.add_mesh(self._cloud, name="post_particles", scalars="rgba", rgba=True, point_size=s.particles_size,
                                  render_points_as_spheres=True, **self._common())
        if source.is_steady:
            # 定常解析の結果を、時間変化する流れの記録と誤解されないようにする
            self.plotter.add_text(STEADY_NOTICE, position="upper_left", font_size=8, color=self.viewport.colors["text"],
                                  name="post_notice", render=False)

    def _particle_colors(self, frame) -> np.ndarray:
        """粒子の色 (RGBA)。流速でカラーマップを引き、尾ほど・見えない粒子ほど透明にする。"""
        from matplotlib import colormaps

        lo, hi = self.color_range("Umag")
        table = (colormaps[self.colormap("Umag")](np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)
        index = np.clip((frame.speed - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.int64)
        rgba = table[index]
        base = 0.75 if self.settings.particles_style == STYLE_SMOKE else 1.0
        rgba[:, 3] = (255 * base * frame.fade * frame.visible).astype(np.uint8)
        return rgba

    def _update_particle_frame(self) -> None:
        s = self.settings
        if self._cloud is None or not s.particles_on:
            return
        source = self.animation()
        if source is None or source.n_particles != self._cloud.n_points:
            self._refresh_particles()
            return
        frame = source.frame(s.particles_time, s.particles_loop)
        self._cloud.points[:] = frame.points
        self._cloud.point_data["rgba"][:] = self._particle_colors(frame)
        self._cloud.GetPoints().Modified()
        self._cloud.GetPointData().GetArray("rgba").Modified()
        self._cloud.Modified()

    def particle_positions(self) -> np.ndarray:
        """いま見えている粒子の座標 (テスト・確認用)。"""
        if self._cloud is None:
            return np.zeros((0, 3))
        visible = np.asarray(self._cloud.point_data["rgba"])[:, 3] > 0
        return np.asarray(self._cloud.points)[visible].copy()
