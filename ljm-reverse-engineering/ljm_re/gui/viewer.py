"""3Dビューア（PyVistaQt / VTK）.

マウス操作（CADソフト風）:
  右ドラッグ  : 回転（常に有効）
  中ドラッグ  : パン
  ホイール    : ズーム
  左ドラッグ  : ツールなし → 回転 / 選択ツール → 選択・ピック
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pyvista as pv
import vtk
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from ..core import mesh_ops
from ..core.selection import project_to_display, region_grow, brush_select
from ..model.objects import SceneObject

log = logging.getLogger(__name__)

SEL_COLOR = (1.0, 0.55, 0.1)
ART_COLOR = (0.95, 0.25, 0.75)
HILITE_COLOR = (1.0, 0.95, 0.35)

vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()


def _faces_vtk(F):
    F = np.asarray(F, np.int64)
    return np.column_stack([np.full(len(F), 3, np.int64), F]).ravel()


def make_polydata(V, F) -> pv.PolyData:
    V = np.asarray(V, float)
    if F is None or len(F) == 0:
        return pv.PolyData(V)
    if hasattr(pv.PolyData, "from_regular_faces"):
        return pv.PolyData.from_regular_faces(V, np.asarray(F, np.int64))
    return pv.PolyData(V, _faces_vtk(F))


def lines_polydata(polylines, closed=None) -> pv.PolyData:
    pts, cells = [], []
    off = 0
    for i, pl in enumerate(polylines):
        pl = np.asarray(pl, float)
        if len(pl) < 2:
            continue
        idx = list(range(off, off + len(pl)))
        if closed is not None and closed[i]:
            idx.append(off)
        cells.append([len(idx)] + idx)
        pts.append(pl)
        off += len(pl)
    if not pts:
        return pv.PolyData()
    pd = pv.PolyData(np.vstack(pts))
    pd.lines = np.concatenate([np.asarray(c, np.int64) for c in cells])
    return pd


# ---------------------------------------------------------------------------
# インタラクタスタイル
# ---------------------------------------------------------------------------

class LJMInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, viewer: "Viewer3D"):
        super().__init__()
        self.viewer = viewer
        self._tool_drag = False
        self.AddObserver("LeftButtonPressEvent", self._ldown)
        self.AddObserver("LeftButtonReleaseEvent", self._lup)
        self.AddObserver("MouseMoveEvent", self._move)
        self.AddObserver("RightButtonPressEvent", self._rdown)
        self.AddObserver("RightButtonReleaseEvent", self._rup)
        self.AddObserver("CharEvent", lambda o, e: None)       # VTK既定のキー操作を無効化
        self.AddObserver("KeyPressEvent", lambda o, e: None)

    def _pos(self):
        iren = self.GetInteractor()
        x, y = iren.GetEventPosition()
        return x, y, bool(iren.GetShiftKey()), bool(iren.GetControlKey())

    def _ldown(self, obj, ev):
        tool = self.viewer.tool
        if tool is not None and tool.wants_left:
            x, y, sh, ct = self._pos()
            self._tool_drag = True
            tool.on_press(x, y, sh, ct)
        else:
            self.OnLeftButtonDown()

    def _lup(self, obj, ev):
        if self._tool_drag:
            self._tool_drag = False
            x, y, sh, ct = self._pos()
            if self.viewer.tool is not None:
                self.viewer.tool.on_release(x, y, sh, ct)
        else:
            self.OnLeftButtonUp()

    def _move(self, obj, ev):
        if self._tool_drag and self.viewer.tool is not None:
            x, y, _, _ = self._pos()
            self.viewer.tool.on_move(x, y)
        else:
            self.OnMouseMove()

    def _rdown(self, obj, ev):
        x, y, _, _ = self._pos()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is not None:
            self.StartRotate()

    def _rup(self, obj, ev):
        self.EndRotate()


# ---------------------------------------------------------------------------
# ツール
# ---------------------------------------------------------------------------

class Tool:
    wants_left = True
    hint = ""

    def __init__(self, viewer: "Viewer3D"):
        self.v = viewer

    def activate(self):
        pass

    def deactivate(self):
        pass

    def on_press(self, x, y, shift, ctrl):
        pass

    def on_move(self, x, y):
        pass

    def on_release(self, x, y, shift, ctrl):
        pass


def _sel_mode(shift, ctrl):
    return "remove" if ctrl else ("add" if shift else "replace")


class RectSelectTool(Tool):
    hint = "矩形選択: 左ドラッグで範囲を選択（Shift: 追加 / Ctrl: 解除）。右ドラッグで回転。"

    def __init__(self, viewer):
        super().__init__(viewer)
        self.p0 = None
        self.band = None

    def on_press(self, x, y, shift, ctrl):
        self.p0 = (x, y)
        self.v.show_rubber_band(x, y, x, y)

    def on_move(self, x, y):
        if self.p0:
            self.v.show_rubber_band(self.p0[0], self.p0[1], x, y)

    def on_release(self, x, y, shift, ctrl):
        self.v.hide_rubber_band()
        if self.p0 is None:
            return
        x0, y0 = self.p0
        self.p0 = None
        if abs(x - x0) < 3 and abs(y - y0) < 3:
            # クリック: その位置の面だけ（スマート選択と同様の挙動はしない）
            self.v.pick_select_face(x, y, _sel_mode(shift, ctrl))
            return
        self.v.rect_select((x0, y0, x, y), _sel_mode(shift, ctrl))


class SmartSelectTool(Tool):
    hint = "スマート選択: クリックした面から滑らかに連続する領域を選択（Shift: 追加 / Ctrl: 解除）。角度許容は設定で変更。"

    def on_release(self, x, y, shift, ctrl):
        self.v.smart_select(x, y, _sel_mode(shift, ctrl))


class BrushSelectTool(Tool):
    hint = "ブラシ選択: クリック/ドラッグで半径内の面を選択（Shift: 追加 / Ctrl: 解除）。半径は設定で変更。"

    def __init__(self, viewer):
        super().__init__(viewer)
        self.mode = "add"
        self.active = False
        self._first = True

    def on_press(self, x, y, shift, ctrl):
        self.active = True
        self.mode = _sel_mode(shift, ctrl)
        self.v.brush_select(x, y, self.mode)
        if self.mode == "replace":
            self.mode = "add"

    def on_move(self, x, y):
        if self.active:
            self.v.brush_select(x, y, self.mode, render=True)

    def on_release(self, x, y, shift, ctrl):
        self.active = False


class PickPointsTool(Tool):
    """クリックで点を取得（特徴点へのスナップ付き）。n 点そろうと callback(list)."""

    def __init__(self, viewer, n: int, callback: Callable, hints: list, snap: bool = True,
                 allowed_kinds=None):
        super().__init__(viewer)
        self.n = n
        self.callback = callback
        self.hints = hints
        self.snap = snap
        self.allowed_kinds = allowed_kinds
        self.points = []
        self._down = None

    @property
    def hint(self):
        i = min(len(self.points), len(self.hints) - 1)
        return f"[{len(self.points) + 1}/{self.n}] {self.hints[i]}（Esc で中止）"

    def on_press(self, x, y, shift, ctrl):
        self._down = (x, y)

    def on_release(self, x, y, shift, ctrl):
        if self._down is None or abs(x - self._down[0]) > 4 or abs(y - self._down[1]) > 4:
            self._down = None
            return
        self._down = None
        res = self.v.pick_point(x, y, snap=self.snap, allowed_kinds=self.allowed_kinds)
        if res is None:
            self.v.status("モデル上をクリックしてください。")
            return
        self.points.append(res)
        self.v.add_temp_marker(res[0])
        if len(self.points) >= self.n:
            pts = list(self.points)
            self.v.set_tool(None)
            self.callback(pts)
        else:
            self.v.status(self.hint)

    def deactivate(self):
        self.v.clear_temp_markers()


# ---------------------------------------------------------------------------
# ビューア本体
# ---------------------------------------------------------------------------

class Viewer3D(QtWidgets.QWidget):
    selectionChanged = QtCore.Signal()
    statusMessage = QtCore.Signal(str)
    toolChanged = QtCore.Signal(str)

    def __init__(self, doc, settings, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.settings = settings
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self, auto_update=False)
        lay.addWidget(self.plotter.interactor)
        self.plotter.enable_depth_peeling(4)
        self._apply_background()
        try:
            self.plotter.add_axes(interactive=False, line_width=2, color="white",
                                  xlabel="X", ylabel="Y", zlabel="Z")
        except Exception:
            log.debug("方向マーカーの追加に失敗")
        self.style = LJMInteractorStyle(self)
        iren = self.plotter.iren.interactor
        iren.SetInteractorStyle(self.style)
        self.style.SetDefaultRenderer(self.plotter.renderer)
        for evname in ("KeyPressEvent", "CharEvent", "KeyReleaseEvent"):
            try:
                iren.RemoveObservers(evname)
            except Exception:
                pass
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0008)

        self.actors: dict[str, list] = {}
        self.owner: dict = {}
        self.lod_used: dict[str, bool] = {}
        self.selection: dict[str, np.ndarray] = {}
        self.sel_actors: dict[str, object] = {}
        self.highlight: set = set()
        self.deviation: dict[str, tuple] = {}
        self.display_mode = "shaded"      # shaded / wire / edges / points
        self.transparent = False
        self.show_normals = False
        self.show_grid = settings.show_grid
        self.show_world_axes = settings.show_world_axes
        self.clip_plane: Optional[vtk.vtkPlane] = None
        self.tool: Optional[Tool] = None
        self._temp_markers = []
        self._band_actor = None
        self._aux_actors = {"grid": [], "axes": [], "normals": [], "scalarbar": []}
        self._first_fit = True
        if settings.parallel_projection:
            self.plotter.enable_parallel_projection()
        doc.listeners.append(self.on_doc_event)
        self.rebuild_all()

    # ------------------------------------------------------------------
    def status(self, msg: str):
        self.statusMessage.emit(msg)

    def render(self):
        try:
            self.plotter.render()
        except Exception:
            log.exception("描画エラー")

    def _apply_background(self):
        self.plotter.set_background(self.settings.background_bottom, top=self.settings.background_top)

    def set_document(self, doc):
        try:
            self.doc.listeners.remove(self.on_doc_event)
        except ValueError:
            pass
        self.doc = doc
        doc.listeners.append(self.on_doc_event)
        self.selection.clear()
        self.deviation.clear()
        self.highlight.clear()
        self._first_fit = True
        self.rebuild_all()

    # ------------------------------------------------------------------
    # ドキュメント同期
    # ------------------------------------------------------------------
    def on_doc_event(self, event, ids):
        if event == "cleared":
            for oid in list(self.actors):
                self._remove_actors(oid)
            self.selection.clear()
            self.deviation.clear()
            self._update_aux()
            self.render()
            return
        if event in ("added", "changed", "renamed"):
            for oid in ids:
                o = self.doc.get(oid)
                if o is None:
                    continue
                if event == "changed":
                    self._validate_selection(o)
                    if o.kind == "mesh" and oid in self.deviation:
                        dv = self.deviation[oid]
                        if dv[3] != len(o.V):
                            self.deviation.pop(oid, None)
                self._rebuild(o)
            if event == "added" and self._first_fit and any(
                    self.doc.get(i) is not None and self.doc.get(i).kind in ("mesh", "points") for i in ids):
                self._first_fit = False
                self._update_aux()
                self.fit_view()
                return
        elif event == "removed":
            for oid in ids:
                self._remove_actors(oid)
                self.selection.pop(oid, None)
                self._remove_sel_actor(oid)
                self.deviation.pop(oid, None)
            self.selectionChanged.emit()
        elif event == "visibility":
            for oid in ids:
                o = self.doc.get(oid)
                if o is not None:
                    for a in self.actors.get(oid, []):
                        a.SetVisibility(bool(o.visible))
                    sa = self.sel_actors.get(oid)
                    if sa is not None:
                        sa.SetVisibility(bool(o.visible))
        elif event == "active_cs":
            pass
        self._update_aux()
        self.render()

    def rebuild_all(self):
        for oid in list(self.actors):
            self._remove_actors(oid)
        for o in self.doc.objects():
            self._rebuild(o)
        self._update_aux()
        self.render()

    def _remove_actors(self, oid):
        for a in self.actors.pop(oid, []):
            self.owner.pop(a, None)
            try:
                self.plotter.remove_actor(a, render=False)
            except Exception:
                pass

    def _add(self, oid, actor):
        if actor is None:
            return
        self.actors.setdefault(oid, []).append(actor)
        self.owner[actor] = oid

    def _mesh_kw(self, o, color):
        opacity = o.opacity * (0.4 if self.transparent else 1.0)
        kw = dict(color=color, opacity=opacity, reset_camera=False, render=False,
                  smooth_shading=False, specular=0.25, specular_power=20, ambient=0.08,
                  diffuse=0.85, pickable=True)
        if self.display_mode == "wire":
            kw["style"] = "wireframe"
        elif self.display_mode == "edges":
            kw["show_edges"] = True
            kw["edge_color"] = (0.25, 0.27, 0.3)
        elif self.display_mode == "points":
            kw["style"] = "points"
            kw["point_size"] = 2
        return kw

    def _rebuild(self, o: SceneObject):
        self._remove_actors(o.id)
        try:
            builder = getattr(self, f"_build_{o.kind}", None)
            if builder is not None:
                builder(o)
        except Exception:
            log.exception("表示の生成に失敗: %s", o.name)
        for a in self.actors.get(o.id, []):
            a.SetVisibility(bool(o.visible))
        self._update_sel_actor(o.id)
        self._apply_clip_to(o.id)

    # ---- 各オブジェクトの表示 -------------------------------------------
    def _color(self, o):
        return HILITE_COLOR if o.id in self.highlight else o.color

    def _build_mesh(self, o):
        if len(o.F) == 0:
            return
        lod = None
        if len(o.F) > self.settings.lod_max_faces:
            lod = mesh_ops.lod_decimate_for_display(o.V, o.F, self.settings.lod_max_faces)
        self.lod_used[o.id] = lod is not None
        pd = lod if lod is not None else make_polydata(o.V, o.F)
        kw = self._mesh_kw(o, self._color(o))
        dev = self.deviation.get(o.id)
        if dev is not None and lod is None:
            vals = np.full(len(o.V), np.nan)
            vals[dev[0]] = dev[1]
            pd.point_data["deviation"] = vals
            kw.pop("color")
            r = dev[2]
            kw.update(scalars="deviation", cmap="jet", clim=[-r, r], nan_color=(0.55, 0.55, 0.58),
                      show_scalar_bar=True,
                      scalar_bar_args={"title": "Deviation [mm]", "vertical": True,
                                       "position_x": 0.9, "position_y": 0.15, "height": 0.7,
                                       "fmt": "%.3f", "color": "white"})
        elif o.show_colors and o.colors is not None and lod is None:
            pd.point_data["rgb"] = o.colors
            kw.pop("color")
            kw.update(scalars="rgb", rgb=True)
        a = self.plotter.add_mesh(pd, name=f"{o.id}:mesh", **kw)
        self._add(o.id, a)
        if self.settings.highlight_artificial and o.face_flags is not None and len(o.face_flags) == len(o.F):
            am = o.artificial_mask
            if am.any():
                apd = make_polydata(o.V, o.F[am])
                aa = self.plotter.add_mesh(apd, name=f"{o.id}:art", color=ART_COLOR, opacity=0.9,
                                           reset_camera=False, render=False, pickable=False)
                aa.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.5, -1.5)
                self._add(o.id, aa)

    def _build_points(self, o):
        if len(o.P) == 0:
            return
        pd = pv.PolyData(o.P)
        kw = dict(color=self._color(o), point_size=o.point_size, render_points_as_spheres=False,
                  reset_camera=False, render=False, opacity=o.opacity)
        if o.show_colors and o.colors is not None:
            pd.point_data["rgb"] = o.colors
            kw.pop("color")
            kw.update(scalars="rgb", rgb=True)
        a = self.plotter.add_mesh(pd, name=f"{o.id}:pts", **kw)
        self._add(o.id, a)

    def _scene_size(self):
        pts = self.doc.all_points_for_bounds()
        if pts is None:
            return 100.0
        return float(np.linalg.norm(pts.max(0) - pts.min(0))) or 100.0

    def _build_plane(self, o):
        c = o.corners()
        quad = pv.PolyData(c, np.array([4, 0, 1, 2, 3]))
        col = self._color(o)
        a = self.plotter.add_mesh(quad, name=f"{o.id}:quad", color=col, opacity=o.opacity,
                                  reset_camera=False, render=False, show_edges=False)
        self._add(o.id, a)
        border = lines_polydata([c], [True])
        self._add(o.id, self.plotter.add_mesh(border, name=f"{o.id}:edge", color=col, line_width=2,
                                              reset_camera=False, render=False))
        size = float(np.max(o.extent[1] - o.extent[0])) or 10.0
        ctr = o.center()
        arrow = pv.Arrow(start=ctr, direction=o.normal, scale=0.25 * size, tip_length=0.25, shaft_radius=0.03)
        self._add(o.id, self.plotter.add_mesh(arrow, name=f"{o.id}:n", color=col, reset_camera=False,
                                              render=False))

    def _build_cylinder(self, o):
        a, b = o.end_points()
        h = max(np.linalg.norm(b - a), 1e-3)
        cyl = pv.Cylinder(center=0.5 * (a + b), direction=o.axis_dir, radius=o.radius, height=h,
                          resolution=64, capping=False)
        col = self._color(o)
        self._add(o.id, self.plotter.add_mesh(cyl, name=f"{o.id}:cyl", color=col, opacity=o.opacity,
                                              reset_camera=False, render=False))
        ext = 0.15 * h + 0.2 * o.radius
        ax = lines_polydata([[a - ext * o.axis_dir, b + ext * o.axis_dir]])
        self._add(o.id, self.plotter.add_mesh(ax, name=f"{o.id}:axis", color=col, line_width=2,
                                              reset_camera=False, render=False))

    def _build_hole(self, o):
        from ..core.fitting.circle import circle_points_3d
        col = self._color(o)
        circ = circle_points_3d(o.center, o.axis, o.diameter / 2, 96)
        self._add(o.id, self.plotter.add_mesh(lines_polydata([circ], [True]), name=f"{o.id}:c",
                                              color=col, line_width=3, reset_camera=False, render=False))
        L = 0.75 * o.diameter
        ax = lines_polydata([[o.center - L * o.axis, o.center + L * o.axis]])
        self._add(o.id, self.plotter.add_mesh(ax, name=f"{o.id}:ax", color=col, line_width=2,
                                              reset_camera=False, render=False))
        self._add(o.id, self.plotter.add_mesh(pv.PolyData(o.center[None]), name=f"{o.id}:p", color=col,
                                              point_size=10, render_points_as_spheres=True,
                                              reset_camera=False, render=False))

    def _build_sphere(self, o):
        sp = pv.Sphere(radius=o.radius, center=o.center, theta_resolution=48, phi_resolution=24)
        self._add(o.id, self.plotter.add_mesh(sp, name=f"{o.id}:s", color=self._color(o),
                                              opacity=o.opacity, reset_camera=False, render=False))

    def _build_point(self, o):
        self._add(o.id, self.plotter.add_mesh(pv.PolyData(o.position[None]), name=f"{o.id}:p",
                                              color=self._color(o), point_size=12,
                                              render_points_as_spheres=True, reset_camera=False,
                                              render=False))

    def _build_axis(self, o):
        a, b = o.end_points()
        self._add(o.id, self.plotter.add_mesh(lines_polydata([[a, b]]), name=f"{o.id}:l",
                                              color=self._color(o), line_width=3, reset_camera=False,
                                              render=False))

    def _build_cs(self, o):
        f = o.frame
        s = o.display_size
        cols = [(0.95, 0.3, 0.3), (0.35, 0.9, 0.35), (0.35, 0.55, 1.0)]
        active = self.doc.active_cs_id == o.id
        for i, (ax, col) in enumerate(zip((f.x_axis, f.y_axis, f.z_axis), cols)):
            arr = pv.Arrow(start=f.origin, direction=ax, scale=s, tip_length=0.15, tip_radius=0.05, shaft_radius=0.015)
            self._add(o.id, self.plotter.add_mesh(arr, name=f"{o.id}:{i}", color=col, reset_camera=False,
                                                  render=False, opacity=1.0 if active else 0.75))
        if o.id in self.highlight or active:
            self._add(o.id, self.plotter.add_mesh(pv.PolyData(f.origin[None]), name=f"{o.id}:o",
                                                  color=HILITE_COLOR if o.id in self.highlight else (1, 1, 1),
                                                  point_size=10, render_points_as_spheres=True,
                                                  reset_camera=False, render=False))

    def _build_section(self, o):
        col = self._color(o)
        if o.polylines:
            pd = lines_polydata(o.polylines, o.closed)
            self._add(o.id, self.plotter.add_mesh(pd, name=f"{o.id}:raw", color=col, line_width=2,
                                                  reset_camera=False, render=False))
        sk = []
        for i in range(len(o.sketch)):
            for e in o.sketch_entities(i):
                sk.append(o.to_3d(e.sample(48)))
        if sk:
            self._add(o.id, self.plotter.add_mesh(lines_polydata(sk), name=f"{o.id}:sk",
                                                  color=(0.3, 1.0, 0.9), line_width=3,
                                                  reset_camera=False, render=False))

    def _build_cad(self, o):
        if o.shape is None:
            return
        from ..cad import occ_backend as occ
        if o._tess is None:
            V, F = occ.tessellate(o.shape, self.settings.tess_deflection_mm, 0.3)
            E = occ.edge_polylines(o.shape, self.settings.tess_deflection_mm)
            o._tess = (V, F, E)
        V, F, E = o._tess
        col = self._color(o)
        if len(F):
            self._add(o.id, self.plotter.add_mesh(make_polydata(V, F), name=f"{o.id}:f", color=col,
                                                  opacity=o.opacity * (0.4 if self.transparent else 1.0),
                                                  smooth_shading=True, specular=0.4, reset_camera=False,
                                                  render=False))
        if E:
            self._add(o.id, self.plotter.add_mesh(lines_polydata(E), name=f"{o.id}:e",
                                                  color=(0.08, 0.1, 0.14) if len(F) else col,
                                                  line_width=1.5 if len(F) else 3,
                                                  reset_camera=False, render=False))

    def _build_measurement(self, o):
        from ..model.formatting import NumberFormat
        fmt = NumberFormat(self.settings.length_decimals, self.settings.angle_decimals)
        col = self._color(o)
        A = o.anchors
        if len(A) >= 2:
            if o.mtype == "angle_3pt" and len(A) == 3:
                segs = [[A[1], A[0]], [A[1], A[2]]]
            else:
                segs = [[A[i], A[i + 1]] for i in range(len(A) - 1)]
            self._add(o.id, self.plotter.add_mesh(lines_polydata(segs), name=f"{o.id}:l", color=col,
                                                  line_width=2, reset_camera=False, render=False))
        if A:
            self._add(o.id, self.plotter.add_mesh(pv.PolyData(np.array(A)), name=f"{o.id}:p", color=col,
                                                  point_size=8, render_points_as_spheres=True,
                                                  reset_camera=False, render=False))
            label = o.short_label(fmt)
            if label:
                pos = A[1] if (o.mtype == "angle_3pt" and len(A) == 3) else np.mean(A[:2] if len(A) >= 2 else A, axis=0)
                try:
                    la = self.plotter.add_point_labels([pos], [label], font_size=13, text_color="white",
                                                       shape_color=(0.17, 0.31, 0.53), shape_opacity=0.85,
                                                       always_visible=True, show_points=False,
                                                       name=f"{o.id}:lab", reset_camera=False, render=False)
                    self._add(o.id, la)
                except TypeError:
                    la = self.plotter.add_point_labels([pos], [label], font_size=13, text_color="white",
                                                       shape_color=(0.17, 0.31, 0.53), shape_opacity=0.85,
                                                       always_visible=True, show_points=False,
                                                       name=f"{o.id}:lab")
                    self._add(o.id, la)

    # ------------------------------------------------------------------
    # 補助表示（グリッド・ワールド座標軸・法線）
    # ------------------------------------------------------------------
    def _clear_aux(self, key):
        for a in self._aux_actors[key]:
            try:
                self.plotter.remove_actor(a, render=False)
            except Exception:
                pass
        self._aux_actors[key] = []

    def _update_aux(self):
        pts = self.doc.all_points_for_bounds()
        self._clear_aux("grid")
        self._clear_aux("axes")
        self._clear_aux("normals")
        if pts is not None:
            lo, hi = pts.min(0), pts.max(0)
        else:
            lo, hi = np.array([-50.0, -50, 0]), np.array([50.0, 50, 0])
        size = float(np.max(hi - lo)) or 100.0
        if self.show_grid:
            sp = 10 ** np.floor(np.log10(max(size, 1e-3) / 4))
            if size / sp > 30:
                sp *= 5
            ext_lo = np.floor(min(lo[0], lo[1], 0) / sp - 1) * sp
            ext_hi = np.ceil(max(hi[0], hi[1], 0) / sp + 1) * sp
            ticks = np.arange(ext_lo, ext_hi + 0.5 * sp, sp)
            segs = [[(t, ext_lo, 0), (t, ext_hi, 0)] for t in ticks] + [[(ext_lo, t, 0), (ext_hi, t, 0)] for t in ticks]
            a = self.plotter.add_mesh(lines_polydata(segs), name="__grid", color=(0.30, 0.34, 0.40),
                                      line_width=1, pickable=False, reset_camera=False, render=False,
                                      lighting=False)
            self._aux_actors["grid"].append(a)
        if self.show_world_axes:
            L = 0.2 * size
            for i, col in enumerate([(0.95, 0.3, 0.3), (0.35, 0.9, 0.35), (0.35, 0.55, 1.0)]):
                e = np.zeros(3)
                e[i] = L
                a = self.plotter.add_mesh(lines_polydata([[np.zeros(3), e]]), name=f"__wax{i}", color=col,
                                          line_width=3, pickable=False, reset_camera=False, render=False)
                self._aux_actors["axes"].append(a)
        if self.show_normals:
            segs = []
            for o in self.doc.objects("mesh"):
                if not o.visible or len(o.V) == 0:
                    continue
                n = o.vertex_normals
                idx = np.arange(len(o.V))
                if len(idx) > 6000:
                    idx = np.random.default_rng(0).choice(len(o.V), 6000, replace=False)
                L = 0.02 * size
                P = o.V[idx]
                Q = P + n[idx] * L
                segs.extend(np.stack([P, Q], axis=1))
            for o in self.doc.objects("points"):
                if not o.visible or o.normals is None:
                    continue
                idx = np.arange(len(o.P))
                if len(idx) > 6000:
                    idx = np.random.default_rng(0).choice(len(o.P), 6000, replace=False)
                L = 0.02 * size
                segs.extend(np.stack([o.P[idx], o.P[idx] + o.normals[idx] * L], axis=1))
            if segs:
                a = self.plotter.add_mesh(lines_polydata(segs), name="__normals", color=(0.4, 0.85, 1.0),
                                          line_width=1, pickable=False, reset_camera=False, render=False)
                self._aux_actors["normals"].append(a)

    # ------------------------------------------------------------------
    # 表示設定
    # ------------------------------------------------------------------
    def set_display_mode(self, mode):
        self.display_mode = mode
        for o in self.doc.objects(("mesh",)):
            self._rebuild(o)
        self.render()

    def set_transparent(self, on):
        self.transparent = bool(on)
        for o in self.doc.objects(("mesh", "cad")):
            self._rebuild(o)
        self.render()

    def set_show_normals(self, on):
        self.show_normals = bool(on)
        self._update_aux()
        self.render()

    def set_show_grid(self, on):
        self.show_grid = bool(on)
        self._update_aux()
        self.render()

    def set_show_world_axes(self, on):
        self.show_world_axes = bool(on)
        self._update_aux()
        self.render()

    def set_parallel(self, on):
        if on:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.render()

    def set_highlight(self, ids):
        old = self.highlight
        self.highlight = set(ids)
        for oid in old ^ self.highlight:
            o = self.doc.get(oid)
            if o is not None and o.kind not in ("mesh", "points"):
                self._rebuild(o)
        self.render()

    def refresh_object(self, oid):
        o = self.doc.get(oid)
        if o is not None:
            self._rebuild(o)
            self.render()

    def set_deviation(self, mesh_id, idx, values, clim):
        o = self.doc.get(mesh_id)
        if o is None:
            return
        self.deviation[mesh_id] = (np.asarray(idx), np.asarray(values), float(clim), len(o.V))
        self._rebuild(o)
        self.render()

    def clear_deviation(self):
        ids = list(self.deviation)
        self.deviation.clear()
        for i in ids:
            o = self.doc.get(i)
            if o is not None:
                self._rebuild(o)
        try:
            self.plotter.remove_scalar_bar()
        except Exception:
            pass
        self.render()

    # ---- 断面表示（クリッピング） ----------------------------------------
    def set_clip(self, origin, normal):
        if origin is None:
            self.clip_plane = None
        else:
            p = vtk.vtkPlane()
            p.SetOrigin(*map(float, origin))
            p.SetNormal(*map(float, normal))
            self.clip_plane = p
        for oid in self.actors:
            self._apply_clip_to(oid)
        self.render()

    def _apply_clip_to(self, oid):
        o = self.doc.get(oid)
        if o is None or o.kind not in ("mesh", "points", "cad"):
            return
        acts = list(self.actors.get(oid, []))
        if oid in self.sel_actors:
            acts.append(self.sel_actors[oid])
        for a in acts:
            m = a.GetMapper()
            if m is None or not hasattr(m, "RemoveAllClippingPlanes"):
                continue
            m.RemoveAllClippingPlanes()
            if self.clip_plane is not None:
                m.AddClippingPlane(self.clip_plane)

    # ---- カメラ ------------------------------------------------------------
    def fit_view(self):
        self.plotter.reset_camera()
        self.render()

    def set_view(self, name: str):
        frame = self.doc.active_frame
        X, Y, Z = frame.x_axis, frame.y_axis, frame.z_axis
        dirs = {  # (カメラ位置方向, 上方向)
            "front": (-Y, Z), "back": (Y, Z), "left": (-X, Z), "right": (X, Z),
            "top": (Z, Y), "bottom": (-Z, -Y), "iso": (X - Y + Z, Z),
        }
        d, up = dirs[name]
        d = d / np.linalg.norm(d)
        cam = self.plotter.camera
        fp = np.array(cam.focal_point)
        dist = np.linalg.norm(np.array(cam.position) - fp) or 100.0
        cam.position = tuple(fp + d * dist)
        cam.up = tuple(up)
        self.plotter.reset_camera()
        self.render()

    # ------------------------------------------------------------------
    # 選択
    # ------------------------------------------------------------------
    def _validate_selection(self, o):
        m = self.selection.get(o.id)
        if m is None:
            return
        n = len(o.F) if o.kind == "mesh" else len(o.P) if o.kind == "points" else 0
        if len(m) != n:
            self.selection.pop(o.id, None)
            self._remove_sel_actor(o.id)
            self.selectionChanged.emit()

    def _remove_sel_actor(self, oid):
        a = self.sel_actors.pop(oid, None)
        if a is not None:
            try:
                self.plotter.remove_actor(a, render=False)
            except Exception:
                pass

    def _update_sel_actor(self, oid):
        self._remove_sel_actor(oid)
        o = self.doc.get(oid)
        m = self.selection.get(oid)
        if o is None or m is None or not m.any():
            return
        if o.kind == "mesh":
            pd = make_polydata(o.V, o.F[m])
            a = self.plotter.add_mesh(pd, name=f"{oid}:sel", color=SEL_COLOR, opacity=1.0,
                                      reset_camera=False, render=False, pickable=False, ambient=0.3)
            a.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.5, -2.5)
        elif o.kind == "points":
            a = self.plotter.add_mesh(pv.PolyData(o.P[m]), name=f"{oid}:sel", color=SEL_COLOR,
                                      point_size=o.point_size + 2, reset_camera=False, render=False,
                                      pickable=False)
        else:
            return
        a.SetVisibility(bool(o.visible))
        self.sel_actors[oid] = a
        if self.clip_plane is not None:
            a.GetMapper().AddClippingPlane(self.clip_plane)

    def set_selection(self, oid, mask, mode="replace", render=True):
        o = self.doc.get(oid)
        if o is None:
            return
        mask = np.asarray(mask, bool)
        if mode == "replace":
            for other in list(self.selection):
                if other != oid:
                    self.selection.pop(other)
                    self._remove_sel_actor(other)
            new = mask
        else:
            cur = self.selection.get(oid)
            if cur is None or len(cur) != len(mask):
                cur = np.zeros(len(mask), bool)
            new = (cur | mask) if mode == "add" else (cur & ~mask)
        if new.any():
            self.selection[oid] = new
        else:
            self.selection.pop(oid, None)
        self._update_sel_actor(oid)
        self.selectionChanged.emit()
        if render:
            self.render()

    def clear_selection(self):
        for oid in list(self.selection):
            self._remove_sel_actor(oid)
        self.selection.clear()
        self.selectionChanged.emit()
        self.render()

    def selection_count(self):
        nf = sum(int(m.sum()) for oid, m in self.selection.items()
                 if self.doc.get(oid) is not None and self.doc.get(oid).kind == "mesh")
        npt = sum(int(m.sum()) for oid, m in self.selection.items()
                  if self.doc.get(oid) is not None and self.doc.get(oid).kind == "points")
        return nf, npt

    def selected_points(self, exclude_artificial=True):
        """選択領域の点と法線。戻り値 (P, N, [oid...], n_excluded_artificial)."""
        Ps, Ns, ids = [], [], []
        excluded = 0
        for oid, m in self.selection.items():
            o = self.doc.get(oid)
            if o is None or not m.any():
                continue
            if o.kind == "mesh":
                fm = m.copy()
                if exclude_artificial and len(o.face_flags) == len(o.F):
                    art = o.artificial_mask & fm
                    excluded += int(art.sum())
                    fm &= ~o.artificial_mask
                vid = np.unique(o.F[fm].ravel())
                if len(vid) == 0:
                    continue
                Ps.append(o.V[vid])
                Ns.append(o.vertex_normals[vid])
                ids.append(oid)
            elif o.kind == "points":
                Ps.append(o.P[m])
                Ns.append(o.normals[m] if o.normals is not None else np.full((int(m.sum()), 3), np.nan))
                ids.append(oid)
        if not Ps:
            return None, None, [], excluded
        P = np.vstack(Ps)
        N = np.vstack(Ns)
        if np.isnan(N).any():
            N = None
        return P, N, ids, excluded

    # ---- 選択の実行 ------------------------------------------------------
    def _display_info(self):
        ren = self.plotter.renderer
        w, h = self.plotter.render_window.GetSize()
        cam = ren.GetActiveCamera()
        aspect = ren.GetTiledAspectRatio()
        M = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
        mvp = np.array([[M.GetElement(i, j) for j in range(4)] for i in range(4)])
        return mvp, (w, h), cam

    def _zbuffer(self):
        """メッシュ/点群のみ表示した状態の深度バッファ (h, w)."""
        hidden = []
        for oid, acts in self.actors.items():
            o = self.doc.get(oid)
            if o is not None and o.kind not in ("mesh", "points"):
                for a in acts:
                    if a.GetVisibility():
                        a.SetVisibility(False)
                        hidden.append(a)
        for key in ("grid", "axes", "normals"):
            for a in self._aux_actors[key]:
                if a.GetVisibility():
                    a.SetVisibility(False)
                    hidden.append(a)
        for a in self.sel_actors.values():
            if a.GetVisibility():
                a.SetVisibility(False)
                hidden.append(a)
        self.plotter.render()
        rw = self.plotter.render_window
        w, h = rw.GetSize()
        arr = vtk.vtkFloatArray()
        rw.GetZbufferData(0, 0, w - 1, h - 1, arr)
        from vtkmodules.util.numpy_support import vtk_to_numpy
        z = vtk_to_numpy(arr).reshape(h, w).copy()
        for a in hidden:
            a.SetVisibility(True)
        return z

    def _linear_depth(self, zb, cam):
        n, f = cam.GetClippingRange()
        if cam.GetParallelProjection():
            return n + zb * (f - n)
        zn = 2.0 * zb - 1.0
        return 2.0 * n * f / (f + n - zn * (f - n))

    def rect_select(self, rect, mode="replace"):
        mvp, size, cam = self._display_info()
        zb = self._zbuffer() if self.settings.rect_front_only else None
        cpos = np.array(cam.GetPosition())
        cdir = np.array(cam.GetDirectionOfProjection())
        tol = 0.004 * self._scene_size()
        if mode == "replace":
            for oid in list(self.selection):
                self._remove_sel_actor(oid)
            self.selection.clear()
            mode = "add"
        x0, x1 = sorted((rect[0], rect[2]))
        y0, y1 = sorted((rect[1], rect[3]))

        def visible(Q):
            """深度バッファと比較して前面に見えている点か."""
            dq = project_to_display(Q, mvp, size)
            H, W = zb.shape
            xi = np.clip(dq[:, 0].astype(int), 0, W - 1)
            yi = np.clip(dq[:, 1].astype(int), 0, H - 1)
            zmax = zb[yi, xi]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                zmax = np.maximum(zmax, zb[np.clip(yi + dy, 0, H - 1), np.clip(xi + dx, 0, W - 1)])
            buf_depth = self._linear_depth(zmax, cam)
            return (Q - cpos) @ cdir <= buf_depth + tol

        for o in self.doc.objects(("mesh", "points")):
            if not o.visible:
                continue
            P = o.V if o.kind == "mesh" else o.P
            if len(P) == 0:
                continue
            d = project_to_display(P, mvp, size)
            m = (d[:, 0] >= x0) & (d[:, 0] <= x1) & (d[:, 1] >= y0) & (d[:, 1] <= y1) & (d[:, 2] >= 0) & (d[:, 2] <= 1)
            if o.kind == "mesh":
                mask = m[o.F].all(axis=1)
                if zb is not None and mask.any():
                    # 面の可視判定は重心で行う（穴越しに頂点だけ見える裏面を除外するため）
                    idx = np.nonzero(mask)[0]
                    mask[idx[~visible(o.topology.centroids[idx])]] = False
            else:
                mask = m
                if zb is not None and mask.any():
                    idx = np.nonzero(mask)[0]
                    mask[idx[~visible(P[idx])]] = False
            if mask.any():
                self.set_selection(o.id, mask, mode, render=False)
        self.selectionChanged.emit()
        self.render()

    def _pick(self, x, y):
        """表示座標で拾った (world_point, object_id, cell_id)。何もなければ None."""
        self.picker.Pick(x, y, 0, self.plotter.renderer)
        a = self.picker.GetActor()
        if a is None:
            return None
        oid = self.owner.get(a)
        if oid is None:
            return None
        return np.array(self.picker.GetPickPosition()), oid, self.picker.GetCellId()

    def _mesh_face_at(self, o, world_pt, cell_id):
        if not self.lod_used.get(o.id) and 0 <= cell_id < len(o.F):
            return int(cell_id)
        return o.topology.nearest_face(world_pt)

    def pick_select_face(self, x, y, mode):
        r = self._pick(x, y)
        if r is None:
            if mode == "replace":
                self.clear_selection()
            return
        p, oid, cid = r
        o = self.doc.get(oid)
        if o.kind == "mesh":
            f = self._mesh_face_at(o, p, cid)
            m = np.zeros(len(o.F), bool)
            m[f] = True
            self.set_selection(oid, m, mode)
        elif o.kind == "points":
            _, i = o.tree.query(p)
            m = np.zeros(len(o.P), bool)
            m[i] = True
            self.set_selection(oid, m, mode)

    def smart_select(self, x, y, mode):
        r = self._pick(x, y)
        if r is None:
            self.status("メッシュ上をクリックしてください。")
            return
        p, oid, cid = r
        o = self.doc.get(oid)
        if o.kind != "mesh":
            self.status("スマート選択はメッシュに対して使用します。点群は矩形/ブラシ選択を使ってください。")
            return
        f = self._mesh_face_at(o, p, cid)
        seed_dev = self.settings.smart_seed_dev_deg or None
        allowed = ~o.artificial_mask if (self.settings.exclude_artificial_in_fit and len(o.face_flags) == len(o.F)) else None
        m = region_grow(o.topology, f, self.settings.smart_angle_deg, seed_dev, allowed=allowed)
        self.set_selection(oid, m, mode)
        self.status(f"スマート選択: {int(m.sum()):,} 面（角度許容 {self.settings.smart_angle_deg}°）")

    def brush_select(self, x, y, mode, render=True):
        r = self._pick(x, y)
        if r is None:
            return
        p, oid, cid = r
        o = self.doc.get(oid)
        rad = self.settings.brush_radius_mm
        if o.kind == "mesh":
            f = self._mesh_face_at(o, p, cid)
            m = brush_select(o.topology, p, rad, connected=True, seed_face=f)
        elif o.kind == "points":
            idx = o.tree.query_ball_point(p, rad)
            m = np.zeros(len(o.P), bool)
            m[idx] = True
        else:
            return
        self.set_selection(oid, m, mode, render=render)

    # ---- 点のピック（スナップ付き） ------------------------------------------
    def snap_candidates(self):
        out = []
        for o in self.doc.objects(("point", "hole", "cs", "cylinder", "plane", "sphere", "axis")):
            if not o.visible:
                continue
            for lab, p in o.key_points():
                out.append((np.asarray(p, float), o.id, lab))
        return out

    def pick_point(self, x, y, snap=True, allowed_kinds=None, snap_px=12.0):
        """クリック位置の3D点。特徴点が近ければスナップ。戻り値 (point, oid, label) または None."""
        if snap:
            cands = self.snap_candidates()
            if cands:
                mvp, size, _ = self._display_info()
                P = np.array([c[0] for c in cands])
                d = project_to_display(P, mvp, size)
                dist = np.hypot(d[:, 0] - x, d[:, 1] - y)
                i = int(np.argmin(dist))
                if dist[i] <= snap_px:
                    p, oid, lab = cands[i]
                    o = self.doc.get(oid)
                    return p, oid, f"{o.name}:{lab}"
        r = self._pick(x, y)
        if r is None:
            return None
        p, oid, cid = r
        o = self.doc.get(oid)
        if allowed_kinds and o.kind not in allowed_kinds:
            return None
        return p, oid, o.name

    def add_temp_marker(self, p):
        a = self.plotter.add_mesh(pv.PolyData(np.asarray(p, float)[None]), color=(1.0, 0.3, 0.3), point_size=12,
                                  render_points_as_spheres=True, reset_camera=False, render=False,
                                  pickable=False, name=f"__tmp{len(self._temp_markers)}")
        self._temp_markers.append(a)
        self.render()

    def clear_temp_markers(self):
        for a in self._temp_markers:
            try:
                self.plotter.remove_actor(a, render=False)
            except Exception:
                pass
        self._temp_markers = []
        self.render()

    # ---- ラバーバンド --------------------------------------------------------
    def show_rubber_band(self, x0, y0, x1, y1):
        if self._band_actor is None:
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(4)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(5)
            for i in (0, 1, 2, 3, 0):
                lines.InsertCellPoint(i)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetLines(lines)
            m = vtk.vtkPolyDataMapper2D()
            m.SetInputData(pd)
            a = vtk.vtkActor2D()
            a.SetMapper(m)
            a.GetProperty().SetColor(1.0, 0.65, 0.2)
            a.GetProperty().SetLineWidth(1.5)
            self.plotter.renderer.AddActor2D(a)
            self._band_actor = (a, pts, pd)
        a, pts, pd = self._band_actor
        for i, (px, py) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
            pts.SetPoint(i, px, py, 0)
        pts.Modified()
        pd.Modified()
        a.SetVisibility(True)
        self.render()

    def hide_rubber_band(self):
        if self._band_actor is not None:
            self._band_actor[0].SetVisibility(False)
            self.render()

    # ---- ツール ------------------------------------------------------------
    def set_tool(self, tool: Optional[Tool]):
        if self.tool is not None:
            try:
                self.tool.deactivate()
            except Exception:
                log.exception("ツール終了処理エラー")
        self.tool = tool
        if tool is not None:
            tool.activate()
            self.status(tool.hint)
            self.toolChanged.emit(type(tool).__name__)
        else:
            self.toolChanged.emit("")

    def screenshot(self, path):
        self.render()
        self.plotter.screenshot(path)

    def close_viewer(self):
        try:
            self.plotter.close()
        except Exception:
            pass
