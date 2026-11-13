"""中央の 3D ビュー (PyVista / VTK)。

シーンのシグナルを受けてアクターを同期する。パーツの移動・回転は VTK アクターの
変換行列だけを書き換えるので、メッシュを作り直さず即座に反映される。

マウス操作: 左ドラッグ = 回転 / 中ドラッグ (Shift+左) = 移動 / ホイール・右ドラッグ = ズーム。
ドラッグを伴わない左クリックが「ピック」(パーツ選択・面選択・計測点) になる。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from ..cad.assembly import Part
from ..cad.inspect import KIND_OPEN
from ..cad.surface import FACE_ID
from ..core.pose import transform_points
from ..core.units import format_length
from . import theme
from .scene import Scene

logger = logging.getLogger("ljmcfd.viewport")

MODE_PART = "part"
MODE_FACE = "face"
MODE_MEASURE = "measure"

#: ビュー名 → (注視点から見たカメラ方向, 上方向)。車両はノーズを -X へ向ける前提。
VIEW_DIRECTIONS = {
    "iso": ((-1.0, -1.0, 0.65), (0.0, 0.0, 1.0)),
    "front": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),    # 上流 (車両前方) から
    "rear": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),      # 下流 (車両後方) から
    "left": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),     # 車両左側面 (-Y 側) から
    "right": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),       # 真上から (風は画面左 → 右)
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}
DEFAULT_BOUNDS = (-2.5, 2.5, -1.5, 1.5, 0.0, 1.5)
CLICK_TOLERANCE_PX = 4


class Viewport(QWidget):
    partClicked = Signal(str, int, bool)     # part id, face id, Ctrl 押下
    backgroundClicked = Signal()
    measureChanged = Signal(object)          # dict (p1, p2, distance, delta) または None

    def __init__(self, scene: Scene, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # MSAA (multi_samples > 0) は半透明表示用の depth peeling と両立せず、半透明の面が
        # 一切描画されなくなる。そのため MSAA は使わず、アンチエイリアスは FXAA で行う。
        self.plotter = QtInteractor(self, auto_update=False, multi_samples=0)
        self.plotter.enable_anti_aliasing("fxaa")
        layout.addWidget(self.plotter)

        self.mode = MODE_PART
        self.theme_name = "dark"
        self.model_opacity = 1.0
        self.show_edges = True
        self.show_bbox = True
        self.show_ground = True
        self.snap_to_vertex = True
        self._actors: dict[str, dict] = {}
        self._actor_to_part: dict[str, str] = {}
        self._measure_points: list[np.ndarray] = []
        self._press_pos: QPoint | None = None
        self._render_pending = False
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._picker.PickFromListOn()

        self.plotter.installEventFilter(self)
        self.plotter.add_axes(interactive=False, line_width=2, labels_off=False)
        self._apply_theme()

        scene.partsAdded.connect(self._on_parts_added)
        scene.partRemoved.connect(self._on_part_removed)
        scene.partChanged.connect(self._on_part_changed)
        scene.transformsChanged.connect(self._on_transforms_changed)
        scene.selectionChanged.connect(self._on_selection_changed)
        scene.faceSelectionChanged.connect(self._update_face_highlight)
        scene.displayUnitChanged.connect(self._on_display_unit_changed)
        scene.cleared.connect(self._on_cleared)

        self._update_ground()
        self.set_view("iso")

    # ------------------------------------------------------------------ 描画要求
    def request_render(self) -> None:
        """同一イベント内の複数変更を 1 回の描画にまとめる。"""
        if not self._render_pending:
            self._render_pending = True
            QTimer.singleShot(0, self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        if getattr(self.plotter, "_closed", False) or self.plotter.render_window is None:
            return
        self.plotter.render()

    def shutdown(self) -> None:
        self.plotter.close()

    # ------------------------------------------------------------------ テーマ・表示設定
    @property
    def colors(self) -> dict:
        return theme.VIEW_THEMES[self.theme_name]

    def _apply_theme(self) -> None:
        c = self.colors
        self.plotter.set_background(c["bottom"], top=c["top"])
        for entry in self._actors.values():
            entry["edges"].prop.color = c["edge"]
        self._update_ground()
        self._update_bbox()
        self._update_measure_overlay()
        self.request_render()

    def set_theme(self, name: str) -> None:
        if name not in theme.VIEW_THEMES:
            raise ValueError(f"未知の背景テーマ: {name}")
        self.theme_name = name
        self._apply_theme()

    def set_model_opacity(self, opacity: float) -> None:
        self.model_opacity = float(np.clip(opacity, 0.05, 1.0))
        for entry in self._actors.values():
            entry["surface"].prop.opacity = self.model_opacity
        # 半透明を正しい前後関係で描くには depth peeling が必要
        if self.model_opacity < 0.999:
            self.plotter.enable_depth_peeling(number_of_peels=6, occlusion_ratio=0.0)
        else:
            self.plotter.disable_depth_peeling()
        self.request_render()

    def set_show_edges(self, show: bool) -> None:
        self.show_edges = bool(show)
        for part_id, entry in self._actors.items():
            entry["edges"].SetVisibility(self.show_edges and self.scene.assembly.get(part_id).visible)
        self.request_render()

    def set_show_bbox(self, show: bool) -> None:
        self.show_bbox = bool(show)
        self._update_bbox()
        self.request_render()

    def set_show_ground(self, show: bool) -> None:
        self.show_ground = bool(show)
        self._update_ground()
        self.request_render()

    def set_parallel_projection(self, enabled: bool) -> None:
        if enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.request_render()

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_PART, MODE_FACE, MODE_MEASURE):
            raise ValueError(f"未知の操作モード: {mode}")
        self.mode = mode
        self.plotter.setCursor(Qt.CrossCursor if mode == MODE_MEASURE else Qt.ArrowCursor)
        if mode != MODE_FACE:
            self.scene.select_face(None)

    # ------------------------------------------------------------------ カメラ
    def _scene_bounds(self) -> tuple[float, ...]:
        return self.scene.assembly.world_bounds(visible_only=True) or DEFAULT_BOUNDS

    def _fit_bounds(self) -> tuple[float, ...]:
        """フィット対象の範囲。寸法ラベルや風向矢印が収まるよう、モデル境界を少し広げる。"""
        b = self._scene_bounds()
        pad = 0.12 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        return (b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad, b[4], b[5] + pad)

    def set_view(self, name: str) -> None:
        direction, up = VIEW_DIRECTIONS[name]
        b = self._scene_bounds()
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        radius = max(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]) / 2, 1e-6)
        direction = np.asarray(direction) / np.linalg.norm(direction)
        camera = self.plotter.camera
        camera.focal_point = tuple(center)
        camera.position = tuple(center + direction * radius * 3.0)
        camera.up = up
        self.plotter.reset_camera(bounds=self._fit_bounds(), render=False)
        self.request_render()

    def fit(self) -> None:
        self.plotter.reset_camera(bounds=self._fit_bounds(), render=False)
        self.request_render()

    def camera_state(self) -> dict:
        position, focal, up = self.plotter.camera_position
        return {
            "position": list(position), "focal_point": list(focal), "up": list(up),
            "parallel": bool(self.plotter.camera.parallel_projection),
            "parallel_scale": float(self.plotter.camera.parallel_scale),
        }

    def restore_camera(self, state: dict) -> None:
        try:
            self.plotter.camera_position = [tuple(state["position"]), tuple(state["focal_point"]), tuple(state["up"])]
            self.set_parallel_projection(bool(state.get("parallel", False)))
            if "parallel_scale" in state:
                self.plotter.camera.parallel_scale = float(state["parallel_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("保存されたカメラ位置を復元できません: %s", exc)
        self.plotter.reset_camera_clipping_range()
        self.request_render()

    def screenshot(self, path: str | Path) -> None:
        self.plotter.render()
        self.plotter.screenshot(str(path))

    # ------------------------------------------------------------------ シーン同期
    def _on_parts_added(self, parts: list[Part]) -> None:
        first = not self._actors
        for part in parts:
            surface = self.plotter.add_mesh(
                part.surface, color=part.color, smooth_shading=True, scalars=None,
                specular=0.25, specular_power=18, ambient=0.18, diffuse=0.85,
                opacity=self.model_opacity, pickable=True, reset_camera=False, render=False,
                name=f"surface:{part.id}",
            )
            edges = self.plotter.add_mesh(
                part.edges, color=self.colors["edge"], line_width=1.0, lighting=False,
                pickable=False, reset_camera=False, render=False, name=f"edges:{part.id}",
            )
            surface.SetVisibility(part.visible)
            edges.SetVisibility(part.visible and self.show_edges)
            self._actors[part.id] = {"surface": surface, "edges": edges, "defects": None}
            self._actor_to_part[surface.GetAddressAsString("")] = part.id
        self._update_matrices()
        if first:
            self.set_view("iso")

    def _remove_part_actors(self, part_id: str) -> None:
        entry = self._actors.pop(part_id, None)
        if entry is None:
            return
        self._actor_to_part.pop(entry["surface"].GetAddressAsString(""), None)
        for key in ("surface", "edges", "defects"):
            if entry[key] is not None:
                self.plotter.remove_actor(entry[key], reset_camera=False, render=False)

    def _on_part_removed(self, part_id: str) -> None:
        self._remove_part_actors(part_id)
        self.request_render()

    def _on_cleared(self) -> None:
        for part_id in list(self._actors):
            self._remove_part_actors(part_id)
        self.clear_measurement()
        self.request_render()

    def _on_part_changed(self, part_id: str) -> None:
        entry = self._actors.get(part_id)
        if entry is None:
            return
        part = self.scene.assembly.get(part_id)
        entry["surface"].SetVisibility(part.visible)
        entry["edges"].SetVisibility(part.visible and self.show_edges)
        if entry["defects"] is not None:
            entry["defects"].SetVisibility(part.visible)
        self._apply_part_color(part)
        self.request_render()

    def _apply_part_color(self, part: Part) -> None:
        entry = self._actors.get(part.id)
        if entry is not None:
            selected = part.id in self.scene.selected_ids
            entry["surface"].prop.color = theme.SELECT_RGB if selected else part.color

    def _update_matrices(self) -> None:
        attitude = self.scene.assembly.attitude_matrix()
        for part in self.scene.parts:
            entry = self._actors.get(part.id)
            if entry is None:
                continue
            matrix = self.scene.assembly.world_matrix(part, attitude)
            for key in ("surface", "edges", "defects"):
                if entry[key] is not None:
                    entry[key].user_matrix = matrix

    def _on_transforms_changed(self) -> None:
        self._update_matrices()
        self._update_bbox()
        self._update_ground()
        self._update_face_highlight()
        self.request_render()

    def _on_selection_changed(self) -> None:
        for part in self.scene.parts:
            self._apply_part_color(part)
        self._update_bbox()
        self.request_render()

    def _on_display_unit_changed(self, _unit: str) -> None:
        self._update_bbox()
        self._update_measure_overlay()
        self.request_render()

    # ------------------------------------------------------------------ オーバーレイ
    def _label_kwargs(self) -> dict:
        return dict(
            font_size=12, text_color=self.colors["text"], shape=None, show_points=False,
            always_visible=True, pickable=False, reset_camera=False, render=False,
        )

    def _update_bbox(self) -> None:
        self.plotter.remove_actor("bbox", reset_camera=False, render=False)
        self.plotter.remove_actor("bbox_labels", reset_camera=False, render=False)
        if not self.show_bbox:
            return
        selected = [p for p in self.scene.selected_parts() if p.visible]
        bounds = self.scene.assembly.world_bounds(selected or None, visible_only=True)
        if bounds is None:
            return
        self.plotter.add_mesh(
            pv.Box(bounds=bounds).outline(), color=self.colors["bbox"], line_width=1.5, lighting=False,
            pickable=False, reset_camera=False, render=False, name="bbox",
        )
        x0, x1, y0, y1, z0, z1 = bounds
        unit = self.scene.display_unit
        anchors = np.array([[(x0 + x1) / 2, y0, z0], [x1, (y0 + y1) / 2, z0], [x1, y1, (z0 + z1) / 2]])
        labels = [
            f"X {format_length(x1 - x0, unit)}",
            f"Y {format_length(y1 - y0, unit)}",
            f"Z {format_length(z1 - z0, unit)}",
        ]
        self.plotter.add_point_labels(anchors, labels, name="bbox_labels", **self._label_kwargs())

    def _update_ground(self) -> None:
        """地面 (Z = 0) のグリッドと風向矢印。"""
        for name in ("ground", "ground_major", "wind_arrow", "wind_label"):
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        if not self.show_ground:
            return
        b = self.scene.assembly.world_bounds(visible_only=True) or DEFAULT_BOUNDS
        size = max(b[1] - b[0], b[3] - b[2], 1e-3)
        step = 1.0 if size > 3.0 else 0.5 if size > 1.0 else 0.1 if size > 0.2 else 0.02
        margin = max(0.35 * size, 2 * step)
        x0, x1 = np.floor((b[0] - margin) / step) * step, np.ceil((b[1] + margin) / step) * step
        y0, y1 = np.floor((b[2] - margin) / step) * step, np.ceil((b[3] + margin) / step) * step
        for name, spacing, color, width in (
            ("ground", step, self.colors["grid"], 1.0),
            ("ground_major", step * 5, self.colors["grid_major"], 1.0),
        ):
            xs = np.arange(np.ceil(x0 / spacing) * spacing, x1 + spacing * 1e-6, spacing)
            ys = np.arange(np.ceil(y0 / spacing) * spacing, y1 + spacing * 1e-6, spacing)
            if len(xs) + len(ys) == 0:
                continue
            starts = [(x, y0, 0.0) for x in xs] + [(x0, y, 0.0) for y in ys]
            ends = [(x, y1, 0.0) for x in xs] + [(x1, y, 0.0) for y in ys]
            pts = np.empty((2 * len(starts), 3))
            pts[0::2], pts[1::2] = starts, ends
            cells = np.column_stack([np.full(len(starts), 2), np.arange(0, len(pts), 2), np.arange(1, len(pts), 2)])
            self.plotter.add_mesh(
                pv.PolyData(pts, lines=cells.ravel()), color=color, line_width=width, lighting=False,
                pickable=False, reset_camera=False, render=False, name=name,
            )
        # 風向 (+X) の矢印: モデル上流側の地面に小さく置く
        length = 0.4 * margin
        start = np.array([b[0] - 0.25 * margin - length, (b[2] + b[3]) / 2, 0.0])
        arrow = pv.Arrow(start=start, direction=(1, 0, 0), scale=length, tip_length=0.3, tip_radius=0.08, shaft_radius=0.025)
        self.plotter.add_mesh(
            arrow, color=theme.ACCENT, lighting=False, pickable=False, reset_camera=False, render=False,
            name="wind_arrow",
        )
        self.plotter.add_point_labels(
            np.array([start + [length / 2, 0.0, 0.2 * length]]), ["WIND +X"], name="wind_label", **self._label_kwargs()
        )

    def _update_face_highlight(self) -> None:
        self.plotter.remove_actor("face_highlight", reset_camera=False, render=False)
        selection = self.scene.selected_face
        if selection is None:
            self.request_render()
            return
        part = self.scene.assembly.get(selection[0])
        surface = part.surface
        mask = np.asarray(surface.cell_data[FACE_ID]) == selection[1]
        cells = np.asarray(surface.faces).reshape(-1, 4)[mask]
        if len(cells) and part.visible:
            actor = self.plotter.add_mesh(
                pv.PolyData(surface.points, cells.ravel()), color=theme.FACE_RGB, lighting=False,
                pickable=False, reset_camera=False, render=False, name="face_highlight",
            )
            actor.user_matrix = self.scene.assembly.world_matrix(part)
            # 元の面と同じ位置に重ねるため、手前へオフセットして Z ファイティングを防ぐ
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
        self.request_render()

    def show_defects(self, part_id: str, defects: pv.PolyData | None) -> None:
        """形状検査で見つかった開放エッジ (赤) / 非多様体エッジ (マゼンタ) を重ねて表示する。"""
        entry = self._actors.get(part_id)
        if entry is None:
            return
        if entry["defects"] is not None:
            self.plotter.remove_actor(entry["defects"], reset_camera=False, render=False)
            entry["defects"] = None
        if defects is not None and defects.n_cells:
            kinds = np.asarray(defects.cell_data["kind"])
            colors = np.where(
                (kinds == KIND_OPEN)[:, None],
                np.array(theme.OPEN_EDGE_RGB) * 255, np.array(theme.NON_MANIFOLD_RGB) * 255,
            ).astype(np.uint8)
            defects = defects.copy()
            defects.cell_data["rgb"] = colors
            part = self.scene.assembly.get(part_id)
            actor = self.plotter.add_mesh(
                defects, scalars="rgb", rgb=True, line_width=4.0, lighting=False, render_lines_as_tubes=True,
                pickable=False, reset_camera=False, render=False, name=f"defects:{part_id}",
            )
            actor.user_matrix = self.scene.assembly.world_matrix(part)
            actor.SetVisibility(part.visible)
            entry["defects"] = actor
        self.request_render()

    # ------------------------------------------------------------------ 計測
    def clear_measurement(self) -> None:
        self._measure_points = []
        self._update_measure_overlay()
        self.measureChanged.emit(None)
        self.request_render()

    def add_measure_point(self, point) -> None:
        """計測点を追加する。2 点そろうと距離を確定し、3 点目で新しい計測を始める。"""
        if len(self._measure_points) >= 2:
            self._measure_points = []
        self._measure_points.append(np.asarray(point, dtype=float))
        self._update_measure_overlay()
        self.measureChanged.emit(self.measurement())
        self.request_render()

    def measurement(self) -> dict | None:
        if not self._measure_points:
            return None
        p1 = self._measure_points[0]
        if len(self._measure_points) == 1:
            return {"p1": p1, "p2": None, "distance": None, "delta": None}
        p2 = self._measure_points[1]
        return {"p1": p1, "p2": p2, "distance": float(np.linalg.norm(p2 - p1)), "delta": p2 - p1}

    def _update_measure_overlay(self) -> None:
        for name in ("measure_points", "measure_line", "measure_label"):
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        if not self._measure_points:
            return
        color = self.colors["measure"]
        pts = np.array(self._measure_points)
        self.plotter.add_mesh(
            pv.PolyData(pts), color=color, point_size=11, render_points_as_spheres=True, lighting=False,
            pickable=False, reset_camera=False, render=False, name="measure_points",
        )
        if len(pts) == 2:
            self.plotter.add_mesh(
                pv.Line(pts[0], pts[1]), color=color, line_width=2.5, lighting=False,
                pickable=False, reset_camera=False, render=False, name="measure_line",
            )
            distance = float(np.linalg.norm(pts[1] - pts[0]))
            kwargs = self._label_kwargs()
            kwargs["text_color"] = color
            self.plotter.add_point_labels(
                np.array([(pts[0] + pts[1]) / 2]), [format_length(distance, self.scene.display_unit)],
                name="measure_label", **kwargs,
            )

    # ------------------------------------------------------------------ ピック
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt の命名
        if obj is self.plotter:
            kind = event.type()
            if kind in (QEvent.KeyPress, QEvent.KeyRelease):
                # VTK 既定のキー操作 (q/e = 終了, w = ワイヤーフレーム, 3 = ステレオ表示 など) を無効化する。
                # アプリのショートカットは Qt 側で先に処理されるため、ここで捨てても影響しない。
                return True
            if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._press_pos = event.position().toPoint()
            elif kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                press, self._press_pos = self._press_pos, None
                release = event.position().toPoint()
                if press is not None and (release - press).manhattanLength() <= CLICK_TOLERANCE_PX:
                    ctrl = bool(event.modifiers() & Qt.ControlModifier)
                    # VTK 側のマウス処理 (回転終了) が済んでからピックする
                    QTimer.singleShot(0, lambda p=release, c=ctrl: self.pick_at(p.x(), p.y(), c))
        return False

    def pick_at(self, x: int, y: int, ctrl: bool = False) -> dict | None:
        """ウィジェット座標 (左上原点・論理ピクセル) でピックし、モードに応じて処理する。"""
        hit = self.pick(x, y)
        if hit is None:
            if self.mode != MODE_MEASURE:
                self.backgroundClicked.emit()
            return None
        if self.mode == MODE_MEASURE:
            self.add_measure_point(hit["snapped"] if self.snap_to_vertex else hit["point"])
        else:
            self.partClicked.emit(hit["part_id"], hit["face_id"], ctrl)
        return hit

    def pick(self, x: int, y: int) -> dict | None:
        """ピック結果 (part_id, face_id, point, snapped) を返す。何も無ければ ``None``。"""
        ratio = self.plotter.devicePixelRatioF()
        vx = x * ratio
        vy = (self.plotter.height() - y - 1) * ratio
        self._picker.InitializePickList()
        for part in self.scene.parts:
            entry = self._actors.get(part.id)
            if entry is not None and part.visible:
                self._picker.AddPickList(entry["surface"])
        if not self._picker.Pick(vx, vy, 0.0, self.plotter.renderer):
            return None
        actor = self._picker.GetActor()
        part_id = self._actor_to_part.get(actor.GetAddressAsString("")) if actor is not None else None
        cell_id = self._picker.GetCellId()
        if part_id is None or cell_id < 0:
            return None
        part = self.scene.assembly.get(part_id)
        point = np.array(self._picker.GetPickPosition())
        snapped = point
        point_id = self._picker.GetPointId()
        if point_id >= 0:
            local = np.asarray(part.surface.points[point_id], dtype=float)
            snapped = transform_points(self.scene.assembly.world_matrix(part), local[None, :])[0]
        return {
            "part_id": part_id,
            "face_id": int(part.surface.cell_data[FACE_ID][cell_id]),
            "point": point,
            "snapped": snapped,
        }

    def world_to_widget(self, point) -> tuple[int, int]:
        """ワールド座標 → ウィジェット座標 (テスト・オーバーレイ用)。"""
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        dx, dy, _ = renderer.GetDisplayPoint()
        ratio = self.plotter.devicePixelRatioF()
        return int(round(dx / ratio)), int(round(self.plotter.height() - 1 - dy / ratio))
