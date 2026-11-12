"""形状認識コマンド（平面・円筒・球・穴中心・基準点/軸）."""
from __future__ import annotations

import logging

import numpy as np

from ..core.fitting import fit_circle_3d, fit_cylinder, fit_plane, fit_sphere
from ..core.geometry import normalize, plane_plane_intersection, three_plane_intersection
from ..core.hole import detect_holes_on_plane, hole_from_cylinder, hole_from_edge_points, hole_from_section
from ..model.objects import (AxisFeature, CylinderFeature, HoleFeature, PlaneFeature, PointFeature,
                             SphereFeature)
from .dialogs import ParamDialog
from .viewer import PickPointsTool

log = logging.getLogger(__name__)


class FitCommands:
    def _build_fit_actions(self, A):
        A("fit_plane", "平面フィット", self.cmd_fit_plane, "Ctrl+1",
          "選択領域に平面をフィット（RANSAC + 最小二乗）")
        A("fit_cylinder", "円筒フィット", self.cmd_fit_cylinder, "Ctrl+2",
          "選択領域に円筒をフィット（法線RANSAC + 非線形最小二乗）。部分円筒にも対応")
        A("fit_sphere", "球フィット", self.cmd_fit_sphere)
        A("fit_circle", "円フィット（穴の縁）", self.cmd_fit_circle, tip="穴の縁（円形エッジ）付近を選択して円を求める")
        A("hole_cyl", "穴中心（内壁から）", self.cmd_hole_from_cylinder, "Ctrl+3",
          "穴の内壁を選択 → 円筒フィット → 軸と基準平面（ツリーで選択）の交点")
        A("hole_section", "穴中心（平面断面から）", self.cmd_hole_section,
          tip="ツリーで穴のある平面を選択し、穴の近くをクリック。内周が不完全でも有効")
        A("hole_auto", "平面上の穴を自動検出", self.cmd_detect_holes, tip="ツリーで平面を選択して実行")
        A("pt_create", "基準点を作成（クリック）", self.cmd_create_point)
        A("pt_3planes", "基準点: 3平面の交点", self.cmd_point_3planes)
        A("axis_feature", "基準軸: 円筒/穴の軸から", self.cmd_axis_from_feature)
        A("axis_2pt", "基準軸: 2点から", self.cmd_axis_2points)
        A("axis_2planes", "基準軸: 2平面の交線", self.cmd_axis_2planes)
        A("fit_options", "フィッティング設定…", self.cmd_fit_options)

    def _selection_or_warn(self):
        P, N, ids, excl = self.viewer.selected_points(self.settings.exclude_artificial_in_fit)
        if P is None or len(P) == 0:
            self.warn("選択がありません", "フィットする領域を選択してください。\n"
                      "G: スマート選択（面をクリック）/ R: 矩形選択 / B: ブラシ選択")
            return None
        if excl:
            log.info("人工面 %d 面をフィット対象から除外しました", excl)
        return P, N, ids, excl

    def _report_fit(self, feat, extra=""):
        st = feat.stats or {}
        L = self.fmt().length
        msg = f"{feat.name}: RMS {L(st.get('rms'))} mm / 最大 {L(st.get('max_abs'))} mm / " \
              f"使用 {st.get('n_used', 0):,}/{st.get('n_points', 0):,} 点" + extra
        self.set_guide(msg)
        log.info(msg)
        for w in feat.warnings:
            log.warning("%s: %s", feat.name, w)

    def _thr(self, v):
        return None if not v or v <= 0 else float(v)

    # ------------------------------------------------------------------
    def cmd_fit_plane(self):
        sel = self._selection_or_warn()
        if sel is None:
            return
        P, N, ids, _ = sel
        s = self.settings
        hint = None if N is None else N.mean(axis=0)
        params = {"method": s.plane_method, "threshold": self._thr(s.plane_threshold_mm), "max_iter": s.ransac_max_iter}

        def work(progress):
            return fit_plane(P, method=params["method"], threshold=params["threshold"],
                             max_iter=params["max_iter"], orient_hint=hint, progress=progress)

        def done(fit):
            f = PlaneFeature.from_fit(fit, self.doc.auto_name("plane"))
            f.meta = self.feature_meta(ids, params)
            self.add_object(f, "平面フィット")
            self._report_fit(f, f" / 平面度 {self.fmt().length(fit.flatness)} mm")
        self.run("平面フィット", work, on_done=done)

    def _cylinder_work(self, P, N):
        s = self.settings
        thr = self._thr(s.cylinder_threshold_mm)
        params = {"threshold": thr, "normals": N is not None}

        def work(progress):
            return fit_cylinder(P, N, threshold=thr, progress=progress)
        return work, params

    def cmd_fit_cylinder(self):
        sel = self._selection_or_warn()
        if sel is None:
            return
        P, N, ids, _ = sel
        work, params = self._cylinder_work(P, N)

        def done(fit):
            f = CylinderFeature.from_fit(fit, self.doc.auto_name("cylinder"))
            f.meta = self.feature_meta(ids, params)
            self.add_object(f, "円筒フィット")
            L = self.fmt().length
            self._report_fit(f, f" / φ{L(fit.diameter)} (1σ {L(2 * fit.sigma_radius)}) / 円周カバー {fit.coverage_deg:.0f}°")
        self.run("円筒フィット", work, on_done=done)

    def cmd_fit_sphere(self):
        sel = self._selection_or_warn()
        if sel is None:
            return
        P, N, ids, _ = sel

        def work(progress):
            return fit_sphere(P)

        def done(fit):
            f = SphereFeature.from_fit(fit, self.doc.auto_name("sphere"))
            f.meta = self.feature_meta(ids)
            self.add_object(f, "球フィット")
            self._report_fit(f, f" / φ{self.fmt().length(fit.diameter)}")
        self.run("球フィット", work, on_done=done)

    def cmd_fit_circle(self):
        sel = self._selection_or_warn()
        if sel is None:
            return
        P, N, ids, _ = sel
        hint = None if N is None else N.mean(axis=0)

        def work(progress):
            return hole_from_edge_points(P, normal_hint=hint, ransac=True)

        def done(h):
            f = HoleFeature.from_result(h, self.doc.auto_name("hole"))
            f.meta = self.feature_meta(ids, {"method": "circle3d"})
            self.add_object(f, "円フィット")
            self._report_fit(f, f" / φ{self.fmt().length(h.diameter)} / 円周カバー {h.coverage_deg:.0f}°")
        self.run("円フィット", work, on_done=done)

    def cmd_hole_from_cylinder(self):
        sel = self._selection_or_warn()
        if sel is None:
            return
        P, N, ids, _ = sel
        planes = self.selected_objects("plane")
        ref = (planes[0].origin, planes[0].normal) if planes else None
        work, params = self._cylinder_work(P, N)

        def done(fit):
            h = hole_from_cylinder(fit, ref)
            f = HoleFeature.from_result(h, self.doc.auto_name("hole"))
            params["ref_plane"] = planes[0].name if planes else None
            f.meta = self.feature_meta(ids + ([planes[0].id] if planes else []), params)
            f.stats = fit.stats.to_dict()
            if not planes:
                f.warnings.append("基準平面が指定されていないため、内壁の軸方向端を穴中心としました。"
                                  "穴のある平面をツリーで選択してから実行すると、軸と平面の交点を求めます。")
            self.add_object(f, "穴中心（内壁）")
            L = self.fmt().length
            self._report_fit(f, f" / φ{L(h.diameter)} / 円周カバー {h.coverage_deg:.0f}°")
        self.run("穴中心（内壁）", work, on_done=done)

    def _plane_and_mesh(self):
        planes = self.selected_objects("plane")
        if not planes:
            self.warn("平面を選択", "モデルツリーで穴のある平面（平面フィットの結果）を選択してください。")
            return None, None
        pl = planes[0]
        src = [self.doc.get(i) for i in pl.meta.get("source_ids", [])]
        mesh = next((o for o in src if o is not None and o.kind == "mesh"), None)
        if mesh is None:
            mesh = self.target_scan(("mesh",))
        return pl, mesh

    def cmd_hole_section(self):
        pl, mesh = self._plane_and_mesh()
        if pl is None or mesh is None:
            return
        depth = self.settings.hole_section_depth_mm

        def cb(pts):
            p = pts[0][0]

            def work(progress):
                return hole_from_section(mesh.V, mesh.F, pl.origin, pl.normal, p, depth)

            def done(h):
                f = HoleFeature.from_result(h, self.doc.auto_name("hole"))
                f.meta = self.feature_meta([mesh.id, pl.id], {"section_depth": depth})
                self.add_object(f, "穴中心（断面）")
                self._report_fit(f, f" / φ{self.fmt().length(h.diameter)} / 円周カバー {h.coverage_deg:.0f}°")
            self.run("穴中心（断面）", work, on_done=done)
        self.viewer.set_tool(PickPointsTool(self.viewer, 1, cb, [f"『{pl.name}』上の穴の近くをクリック（断面深さ {depth} mm）"],
                                            snap=False))

    def cmd_detect_holes(self):
        pl, mesh = self._plane_and_mesh()
        if pl is None or mesh is None:
            return
        v = ParamDialog.ask(self, "平面上の穴を自動検出", [
            ("depth", "断面深さ（平面から材料側へ）", "float", self.settings.hole_section_depth_mm,
             {"min": 0.01, "max": 1000, "suffix": "mm"}),
            ("dmin", "最小穴径", "float", 2.0, {"min": 0.01, "max": 1e4, "suffix": "mm"}),
            ("dmax", "最大穴径", "float", 200.0, {"min": 0.01, "max": 1e5, "suffix": "mm"}),
            ("tol", "真円からの許容偏差", "float", 0.2, {"min": 0.001, "max": 10, "suffix": "mm"}),
            ("crit", "検出した穴を『安全上重要』にする", "bool", False, None),
        ], f"対象: {mesh.name} / 平面: {pl.name}\n平面の少し内側で断面を取り、円形の閉ループを穴として検出します。")
        if not v:
            return

        def work(progress):
            return detect_holes_on_plane(mesh.V, mesh.F, pl.origin, pl.normal, v["depth"], v["dmin"], v["dmax"],
                                         abs_tol=v["tol"])

        def done(holes):
            if not holes:
                self.info("穴の自動検出", "穴が見つかりませんでした。断面深さ・許容偏差を調整してください。")
                return
            objs = []
            for h in holes:
                f = HoleFeature.from_result(h, self.doc.auto_name("hole") if not objs else "")
                f.name = self.doc.unique_name(f"穴 φ{h.diameter:.2f}")
                f.meta = self.feature_meta([mesh.id, pl.id], {"auto": True, **v})
                f.critical = v["crit"]
                objs.append(f)
            self.doc.add(objs, f"穴の自動検出 ({len(objs)} 個)")
            self.tree.select_ids([o.id for o in objs])
            self.set_guide(f"{len(objs)} 個の穴を検出しました")
            log.info("穴の自動検出: %d 個 (%s)", len(objs), ", ".join(f"φ{h.diameter:.3f}" for h in holes))
        self.run("穴の自動検出", work, on_done=done)

    # ------------------------------------------------------------------
    def cmd_create_point(self):
        def cb(pts):
            p, oid, lab = pts[0]
            f = PointFeature(self.doc.auto_name("point"), p)
            f.meta = self.feature_meta([oid], {"picked_on": lab})
            self.add_object(f, "基準点作成")
        self.viewer.set_tool(PickPointsTool(self.viewer, 1, cb, ["基準点を作る位置をクリック（特徴点にスナップ）"]))

    def cmd_point_3planes(self):
        planes = self.selected_objects("plane")
        if len(planes) != 3:
            self.warn("3平面を選択", "ツリーで平面を3つ選択してください。")
            return
        p = three_plane_intersection([(o.origin, o.normal) for o in planes])
        if p is None:
            self.warn("交点なし", "3平面の交点が定まりません（平行な平面が含まれています）。")
            return
        f = PointFeature(self.doc.auto_name("point"), p)
        f.meta = self.feature_meta([o.id for o in planes], {"method": "3平面の交点"})
        self.add_object(f, "3平面の交点")

    def cmd_axis_from_feature(self):
        objs = self.selected_objects(("cylinder", "hole"))
        if not objs:
            self.warn("要素を選択", "ツリーで円筒または穴を選択してください。")
            return
        new = []
        for o in objs:
            if o.kind == "cylinder":
                a, b = o.end_points()
                ax = AxisFeature(self.doc.unique_name(o.name + "_軸"), 0.5 * (a + b), o.axis_dir,
                                 max(o.length, o.diameter))
            else:
                ax = AxisFeature(self.doc.unique_name(o.name + "_軸"), o.center, o.axis, 2 * o.diameter)
            ax.stats = o.stats
            ax.meta = self.feature_meta([o.id])
            ax.critical = o.critical
            new.append(ax)
        self.doc.add(new, "基準軸作成")
        self.tree.select_ids([a.id for a in new])

    def cmd_axis_2points(self):
        def cb(pts):
            (p1, o1, _), (p2, o2, _) = pts
            d = p2 - p1
            if np.linalg.norm(d) < 1e-9:
                self.warn("基準軸", "2点が一致しています。")
                return
            ax = AxisFeature(self.doc.auto_name("axis"), 0.5 * (p1 + p2), normalize(d), float(np.linalg.norm(d)))
            ax.meta = self.feature_meta([o1, o2])
            self.add_object(ax, "基準軸作成（2点）")
        self.viewer.set_tool(PickPointsTool(self.viewer, 2, cb, ["軸の1点目をクリック", "軸の2点目をクリック"]))

    def cmd_axis_2planes(self):
        planes = self.selected_objects("plane")
        if len(planes) != 2:
            self.warn("2平面を選択", "ツリーで平面を2つ選択してください。")
            return
        a, b = planes
        r = plane_plane_intersection(a.origin, a.normal, b.origin, b.normal)
        if r is None:
            self.warn("交線なし", "2平面が平行なため交線がありません。")
            return
        p, d = r
        c = 0.5 * (a.center() + b.center())
        p = p + np.dot(c - p, d) * d
        size = float(max(np.max(a.extent[1] - a.extent[0]), np.max(b.extent[1] - b.extent[0])))
        ax = AxisFeature(self.doc.auto_name("axis"), p, d, size)
        ax.meta = self.feature_meta([a.id, b.id], {"method": "2平面の交線"})
        self.add_object(ax, "2平面の交線")

    def cmd_fit_options(self):
        s = self.settings
        v = ParamDialog.ask(self, "フィッティング設定", [
            ("plane_method", "平面フィットの手法", "choice", s.plane_method,
             [("ransac", "RANSAC + 最小二乗（外れ値に強い）"), ("lsq", "最小二乗のみ")]),
            ("plane_threshold_mm", "平面のインライア閾値 (0=自動)", "float", s.plane_threshold_mm,
             {"min": 0, "max": 100, "suffix": "mm", "decimals": 4}),
            ("cylinder_threshold_mm", "円筒のインライア閾値 (0=自動)", "float", s.cylinder_threshold_mm,
             {"min": 0, "max": 100, "suffix": "mm", "decimals": 4}),
            ("ransac_max_iter", "RANSAC 最大反復", "int", s.ransac_max_iter, {"min": 10, "max": 100000}),
            ("hole_section_depth_mm", "穴断面の深さ", "float", s.hole_section_depth_mm, {"min": 0.01, "max": 1000, "suffix": "mm"}),
            ("exclude_artificial_in_fit", "人工面（穴埋め）を除外", "bool", s.exclude_artificial_in_fit, None),
            ("smart_angle_deg", "スマート選択の角度許容", "float", s.smart_angle_deg, {"min": 0.1, "max": 90, "suffix": "°"}),
        ], "自動閾値は残差の中央絶対偏差（MAD）から決定します。実際に使用した閾値は結果に記録されます。")
        if v:
            for k, val in v.items():
                setattr(s, k, val)
            s.save()
