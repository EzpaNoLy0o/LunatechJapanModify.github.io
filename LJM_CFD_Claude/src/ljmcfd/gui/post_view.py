"""解析結果の 3D 表示: 表面分布・断面・流線・等値面のレイヤーを 3D ビューに重ねる。

表示の更新は、読み込み済みの結果からの抽出だけで行う (CFD は再実行しない)。
断面は位置を変えるたびにアクターを作り直さず、同じメッシュの中身だけを差し替えるので、
スライダーへ滑らかに追従する。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, Signal

from ..post.result_set import AXES, COLORMAPS, SURFACE_FIELDS, VOLUME_FIELDS, FieldInfo, ResultSet, SeedSpec, default_seed
from .scene import Scene
from .viewport import Viewport

logger = logging.getLogger("ljmcfd.post")

SOLID = "solid"          # 流線・等値面を物理量で色分けせず単色で描く
LAYERS = ("surface", "slice", "streamlines", "iso")


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
        self._bars: list[str] = []
        scene.solverChanged.connect(self.reload)
        scene.cleared.connect(self._on_cleared)
        viewport.themeChanged.connect(self.refresh_all)

    # ------------------------------------------------------------------ 結果の読み込み
    def reload(self) -> None:
        """シーンの解析結果に合わせて表示対象を切り替える。"""
        summary = self.scene.solver_summary
        self.results = None
        self._slice_mesh = self._slice_key = self._streamlines = None
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
        self.viewport.request_render()
        self.changed.emit()

    def reset_seed(self) -> None:
        """開始面を車両前面の既定位置へ戻す。"""
        if self.results is None:
            return
        self.settings.seed = self._default_seed()
        self._refresh_streamlines(recompute=True)
        self.viewport.request_render()
        self.changed.emit()

    def apply_settings(self, settings: PostSettings) -> None:
        self.settings = settings
        self._slice_mesh = self._slice_key = self._streamlines = None
        if self.results is not None:
            self._initialize_defaults()
        self.refresh_all()
        self.changed.emit()

    # ------------------------------------------------------------------ 描画
    def refresh_all(self) -> None:
        self._refresh_surface()
        self._refresh_slice()
        self._refresh_streamlines(recompute=True)
        self._refresh_iso()
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
                "streamlines": "post_streamlines" in actors, "iso": "post_iso" in actors}
