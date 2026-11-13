"""断面・CAD再構築・偏差解析・エクスポートのコマンド."""
from __future__ import annotations

import json
import logging
import os

import numpy as np
from PySide6 import QtWidgets

from ..core.section import section_mesh, section_series
from ..model.objects import CadObject, SectionObject
from .dialogs import ParamDialog, ReportDialog

log = logging.getLogger(__name__)


def _occ():
    from ..cad import occ_backend as occ
    return occ


class CadCommands:
    def _build_cad_actions(self, A):
        A("sec_axis", "断面作成: X/Y/Z", self.cmd_section_axis, tip="作業座標系の X/Y/Z 方向の断面")
        A("sec_plane", "断面作成: 要素基準（平面/軸）", self.cmd_section_feature,
          tip="選択した平面に平行な断面、または軸を含む断面")
        A("sec_series", "連続断面", self.cmd_section_series)
        A("sec_sketch", "断面をスケッチ化（直線・円弧）", self.cmd_sketch_lines_arcs,
          tip="断面曲線を直線・円弧・円に自動分割（CADで使えるスケッチ要素）")
        A("sec_spline", "断面をスケッチ化（スプライン）", self.cmd_sketch_spline)
        A("cad_features", "要素から CAD 形状を作成", self.cmd_cad_from_features,
          tip="平面→面, 円筒→円筒面/ソリッド, 穴→円エッジ/円柱, 球→ソリッド, 軸→エッジ")
        A("cad_extrude", "押し出し（閉じた断面）", self.cmd_cad_extrude)
        A("cad_revolve", "回転体（断面＋軸）", self.cmd_cad_revolve)
        A("cad_loft", "ロフト曲面（複数断面）", self.cmd_cad_loft)
        A("cad_bspline", "Bスプライン曲面フィット（選択領域）", self.cmd_cad_bspline)
        A("cad_boolean", "ブーリアン（結合/差/共通）", self.cmd_cad_boolean)
        A("cad_deviation", "CAD-メッシュ偏差解析", self.cmd_cad_deviation)
        A("feat_deviation", "要素-メッシュ偏差（平面/円筒/球）", self.cmd_feature_deviation)
        A("exp_step", "STEP 出力（再読込検証付き）…", self.cmd_export_step)
        A("exp_dxf", "DXF 出力（断面曲線）…", self.cmd_export_dxf)
        A("exp_csv", "CSV 出力（座標一覧・穴中心）…", lambda: self.cmd_export_csv_features(None))
        A("exp_csv_meas", "CSV 出力（測定結果）…", self.cmd_export_csv_measurements)
        A("exp_dist", "CSV 出力（点間距離マトリクス）…", lambda: self.cmd_export_distance_matrix(None))
        A("exp_json", "JSON 出力（プロジェクト情報）…", self.cmd_export_json)
        A("exp_mesh", "メッシュ/点群を出力（STL/OBJ/PLY）…", self.cmd_export_mesh)
        A("exp_points_dxf", "基準点・穴中心を DXF 出力…", self.cmd_export_points_dxf)
        from ..cad import occ_backend as occ
        if not occ.available():
            for k in ("cad_features", "cad_extrude", "cad_revolve", "cad_loft", "cad_bspline", "cad_boolean",
                      "cad_deviation", "exp_step", "import_step"):
                if k in self.actions:
                    self.actions[k].setEnabled(False)
                    self.actions[k].setToolTip("OCP (OpenCASCADE) が利用できないため無効: " + occ.import_error())

    # ------------------------------------------------------------------
    # 断面
    # ------------------------------------------------------------------
    def _section_target(self):
        o = self.target_scan(("mesh",))
        return o

    def _add_section(self, res, mesh, name):
        s = SectionObject.from_result(res, self.doc.unique_name(name))
        s.source_mesh_id = mesh.id
        s.meta = self.feature_meta([mesh.id])
        return s

    def cmd_section_axis(self):
        mesh = self._section_target()
        if mesh is None:
            return
        f = self.doc.active_frame
        L = f.to_local(mesh.V)
        v = ParamDialog.ask(self, "断面作成", [
            ("axis", "断面の法線方向（作業座標系）", "choice", "z", [("x", "X"), ("y", "Y"), ("z", "Z")]),
            ("pos", "位置", "float", float(np.round(L[:, 2].mean(), 3)), {"decimals": 4, "suffix": "mm"}),
        ], f"対象: {mesh.name}\n範囲: X {L[:, 0].min():.2f}〜{L[:, 0].max():.2f} / Y {L[:, 1].min():.2f}〜{L[:, 1].max():.2f}"
           f" / Z {L[:, 2].min():.2f}〜{L[:, 2].max():.2f} mm")
        if not v:
            return
        k = "xyz".index(v["axis"])
        n = f.R[:, k]
        q = np.zeros(3)
        q[k] = v["pos"]
        o = f.to_world(q)
        res = section_mesh(mesh.V, mesh.F, o, n)
        if not res.polylines:
            self.warn("断面", "指定位置に断面がありません。")
            return
        s = self._add_section(res, mesh, f"断面 {v['axis'].upper()}={v['pos']:g}")
        self.add_object(s, "断面作成")
        self.set_guide(f"{s.name}: {len(s.polylines)} 曲線（閉 {sum(s.closed)}）")

    def cmd_section_feature(self):
        mesh = self._section_target()
        if mesh is None:
            return
        pl = self.selected_objects("plane")
        from .cmds_measure import axis_of, AXIS_KINDS
        ax = self.selected_objects(AXIS_KINDS)
        if pl:
            p = pl[0]
            v = ParamDialog.ask(self, "平面に平行な断面", [
                ("off", "平面からのオフセット（法線方向）", "float", -0.5, {"decimals": 4, "suffix": "mm", "min": -1e5, "max": 1e5})])
            if not v:
                return
            res = section_mesh(mesh.V, mesh.F, p.origin + v["off"] * p.normal, p.normal, p.u_axis, p.v_axis)
            name = f"断面 {p.name}{v['off']:+g}"
        elif ax:
            a = ax[0]
            ap, ad, _ = axis_of(a)
            v = ParamDialog.ask(self, "軸を含む断面", [
                ("ang", "軸周りの角度", "float", 0.0, {"min": -360, "max": 360, "suffix": "°"})],
                "回転体の作成に使います。角度 0° は作業座標系の X 軸側。")
            if not v:
                return
            from ..core.geometry import rotation_about_axis, orthonormal_basis
            ref = self.doc.active_frame.x_axis
            u = ref - np.dot(ref, ad) * ad
            if np.linalg.norm(u) < 1e-6:
                u, _ = orthonormal_basis(ad)
            u = u / np.linalg.norm(u)
            u = rotation_about_axis(ad, v["ang"]) @ u
            n = np.cross(ad, u)
            res = section_mesh(mesh.V, mesh.F, ap, n, ad, u)
            name = f"断面 {a.name} {v['ang']:g}°"
        else:
            self.warn("要素を選択", "ツリーで平面（平行断面）または円筒/穴/軸（軸を含む断面）を選択してください。")
            return
        if not res.polylines:
            self.warn("断面", "断面がありません。")
            return
        s = self._add_section(res, mesh, name)
        self.add_object(s, "断面作成")

    def cmd_section_series(self):
        mesh = self._section_target()
        if mesh is None:
            return
        f = self.doc.active_frame
        L = f.to_local(mesh.V)
        v = ParamDialog.ask(self, "連続断面", [
            ("axis", "法線方向（作業座標系）", "choice", "x", [("x", "X"), ("y", "Y"), ("z", "Z")]),
            ("start", "開始位置", "float", float(np.round(L[:, 0].min() + 1, 2)), {"decimals": 3, "suffix": "mm"}),
            ("end", "終了位置", "float", float(np.round(L[:, 0].max() - 1, 2)), {"decimals": 3, "suffix": "mm"}),
            ("count", "断面数", "int", 5, {"min": 2, "max": 500}),
        ], "エアロパーツ等のロフト用に、等間隔の断面を作成します。")
        if not v:
            return
        k = "xyz".index(v["axis"])
        n = f.R[:, k]
        offsets = np.linspace(v["start"], v["end"], v["count"])
        origin = f.origin

        def work(progress):
            return section_series(mesh.V, mesh.F, origin, n, offsets, progress)

        def done(results):
            objs = []
            for off, r in zip(offsets, results):
                if r.polylines:
                    objs.append(self._add_section(r, mesh, f"断面 {v['axis'].upper()}={off:.2f}"))
            if objs:
                self.doc.add(objs, f"連続断面 ({len(objs)})")
                self.tree.select_ids([o.id for o in objs])
            self.set_guide(f"{len(objs)} 個の断面を作成しました")
        self.run("連続断面", work, on_done=done)

    def _selected_sections(self):
        secs = self.selected_objects("section")
        if not secs:
            self.warn("断面を選択", "モデルツリーで断面曲線を選択してください。")
        return secs

    def _sketch(self, mode):
        secs = self._selected_sections()
        if not secs:
            return
        v = ParamDialog.ask(self, "スケッチ化", [
            ("tol", "許容誤差", "float", self.settings.sketch_tol_mm, {"min": 0.001, "max": 10, "suffix": "mm", "decimals": 4})],
            "断面曲線からの最大偏差がこの値以内になるよう要素を作成します。スキャンノイズより大きめに設定してください。")
        if not v:
            return
        from ..cad.reconstruct import sketch_section
        tol = v["tol"]
        for s in secs:
            try:
                new = sketch_section(s, tol, mode)
            except Exception as e:  # noqa: BLE001
                self.warn("スケッチ化", f"{s.name}: {e}")
                continue

            def apply(o, new=new):
                o.sketch = new
                o.sketch_tol = tol
            self.doc.modify(s.id, apply, "断面スケッチ化")
            kinds = {}
            for ents in new:
                for d in ents:
                    kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
            log.info("%s をスケッチ化: %s", s.name, kinds)
            self.set_guide(f"{s.name}: " + ", ".join(f"{k} {n}" for k, n in kinds.items()))

    def cmd_sketch_lines_arcs(self):
        self._sketch("lines_arcs")

    def cmd_sketch_spline(self):
        self._sketch("spline")

    # ------------------------------------------------------------------
    # CAD 形状
    # ------------------------------------------------------------------
    def _add_cad(self, shape, recipe, name, source_ids=()):
        occ = _occ()
        o = CadObject(self.doc.unique_name(name), shape, recipe)
        o.summary = occ.summarize(shape)
        o.meta = self.feature_meta(list(source_ids), recipe)
        if not o.summary.get("valid"):
            log.warning("%s: BRepCheck で無効な形状と判定されました", o.name)
        return o

    def cmd_cad_from_features(self):
        from ..cad import reconstruct as rc
        feats = self.selected_objects(("plane", "cylinder", "hole", "sphere", "axis"))
        if not feats:
            self.warn("要素を選択", "ツリーで平面・円筒・穴・球・軸を選択してください。")
            return
        fields = []
        if any(f.kind == "cylinder" for f in feats):
            fields.append(("cyl", "円筒の形式", "choice", "face",
                           [("face", "円筒面（360°）"), ("solid", "円柱ソリッド"), ("partial", "スキャン範囲の部分円筒面")]))
            fields.append(("ext", "円筒の延長（両端）", "float", 0.0, {"min": 0, "max": 1e4, "suffix": "mm"}))
        if any(f.kind == "hole" for f in feats):
            fields.append(("hole", "穴の形式", "choice", "edge", [("edge", "円エッジ"), ("cylinder", "円柱ソリッド（カット用）")]))
        if any(f.kind == "plane" for f in feats):
            fields.append(("margin", "平面の余白", "float", 0.0, {"min": 0, "max": 1e4, "suffix": "mm"}))
        v = ParamDialog.ask(self, "要素から CAD 形状を作成", fields) if fields else {}
        if v is None:
            return
        objs = []
        for f in feats:
            try:
                if f.kind == "plane":
                    s, r = rc.plane_to_face(f, v.get("margin", 0.0))
                elif f.kind == "cylinder":
                    s, r = rc.cylinder_to_shape(f, v.get("cyl", "face"), v.get("ext", 0.0))
                elif f.kind == "hole":
                    s, r = rc.hole_to_shape(f, v.get("hole", "edge"))
                elif f.kind == "sphere":
                    s, r = rc.sphere_to_solid(f)
                else:
                    s, r = rc.axis_to_edge(f)
                objs.append(self._add_cad(s, r, f"CAD_{f.name}", [f.id]))
            except Exception as e:  # noqa: BLE001
                self.warn("CAD 形状作成", f"{f.name}: {e}")
        if objs:
            self.doc.add(objs, f"CAD 形状作成 ({len(objs)})")
            self.tree.select_ids([o.id for o in objs])

    def cmd_cad_extrude(self):
        from ..cad import reconstruct as rc
        secs = self._selected_sections()
        if not secs:
            return
        s = secs[0]
        if not any(s.closed):
            self.warn("押し出し", "閉じた断面曲線がありません。")
            return
        v = ParamDialog.ask(self, "押し出し", [
            ("dist", "距離", "float", 10.0, {"min": -1e5, "max": 1e5, "suffix": "mm", "decimals": 4}),
            ("sym", "両側に対称", "bool", False, None),
            ("tol", "未スケッチ曲線の自動スケッチ化許容誤差", "float", self.settings.sketch_tol_mm,
             {"min": 0.001, "max": 10, "suffix": "mm", "decimals": 4}),
        ], f"{s.name}: 閉じた曲線 {sum(s.closed)} 本。最大の曲線を外周、内側の曲線を穴として押し出します（方向は断面法線）。")
        if not v:
            return
        dist = v["dist"]
        normal = s.normal if dist >= 0 else -s.normal

        def work(progress):
            return rc.extrude_section(s, abs(dist), symmetric=v["sym"], tol=v["tol"], direction=normal)

        def done(res):
            shape, recipe = res
            o = self._add_cad(shape, recipe, f"押し出し_{s.name}", [s.id])
            self.add_object(o, "押し出し")
        self.run("押し出し", work, on_done=done)

    def cmd_cad_revolve(self):
        from ..cad import reconstruct as rc
        from .cmds_measure import axis_of, AXIS_KINDS
        secs = self._selected_sections()
        if not secs:
            return
        s = secs[0]
        ax = self.selected_objects(AXIS_KINDS)
        if ax:
            ap, ad, _ = axis_of(ax[0])
            axname = ax[0].name
        else:
            f = self.doc.active_frame
            ap, ad, axname = f.origin, f.z_axis, "作業座標系 Z 軸"
        closed_idx = [i for i, c in enumerate(s.closed) if c]
        if not closed_idx:
            self.warn("回転体", "閉じた断面曲線がありません。")
            return
        v = ParamDialog.ask(self, "回転体", [
            ("loop", "使用する曲線", "choice", closed_idx[0], [(i, f"曲線 {i + 1}（点 {len(s.polylines[i])}）") for i in closed_idx]),
            ("ang", "回転角度", "float", 360.0, {"min": 0.1, "max": 360, "suffix": "°"}),
            ("half", "軸の片側だけをプロファイルにする", "bool", True, None),
            ("tol", "スケッチ化許容誤差", "float", self.settings.sketch_tol_mm, {"min": 0.001, "max": 10, "suffix": "mm"}),
        ], f"回転軸: {axname}\n軸を含む断面（『断面作成: 要素基準』で軸を選択）を使用してください。")
        if not v:
            return

        def work(progress):
            return rc.revolve_section(s, ap, ad, v["loop"], v["ang"], v["tol"], v["half"])

        def done(res):
            shape, recipe = res
            o = self._add_cad(shape, recipe, f"回転体_{s.name}", [s.id] + ([ax[0].id] if ax else []))
            self.add_object(o, "回転体")
        self.run("回転体", work, on_done=done)

    def cmd_cad_loft(self):
        from ..cad import reconstruct as rc
        secs = self.selected_objects("section")
        if len(secs) < 2:
            self.warn("ロフト", "ツリーで2つ以上の断面曲線を選択してください（連続断面が便利です）。")
            return
        curves, closed_flags = [], []
        for s in secs:
            i = int(np.argmax([len(p) for p in s.polylines]))
            curves.append(s.polylines[i])
            closed_flags.append(s.closed[i])
        closed = all(closed_flags)
        v = ParamDialog.ask(self, "ロフト曲面", [
            ("solid", "ソリッドにする（閉じた断面のみ）", "bool", closed, None),
            ("ruled", "直線補間（ルールド）", "bool", False, None),
            ("n", "断面あたりのサンプル点数", "int", 80, {"min": 8, "max": 1000}),
        ], f"{len(secs)} 断面（{'閉じた曲線' if closed else '開いた曲線を含む'}）から曲面を作成します。"
           "各断面の最長曲線を使用し、ツリーの選択順に接続します。")
        if not v:
            return

        def work(progress):
            return rc.loft_curves(curves, closed, v["solid"], v["ruled"], v["n"])

        def done(res):
            shape, recipe = res
            o = self._add_cad(shape, recipe, "ロフト", [s.id for s in secs])
            self.add_object(o, "ロフト")
        self.run("ロフト", work, on_done=done)

    def cmd_cad_bspline(self):
        from ..cad import reconstruct as rc
        P, N, ids, _ = self.viewer.selected_points(self.settings.exclude_artificial_in_fit)
        if P is None:
            self.warn("選択がありません", "曲面をフィットする領域を選択してください。")
            return
        v = ParamDialog.ask(self, "Bスプライン曲面フィット", [
            ("nu", "制御点数 U", "int", 10, {"min": 4, "max": 80}),
            ("nv", "制御点数 V", "int", 10, {"min": 4, "max": 80}),
            ("deg", "次数", "int", 3, {"min": 2, "max": 5}),
            ("smooth", "平滑化（0=純粋な最小二乗）", "float", 0.001, {"min": 0, "max": 10, "decimals": 5}),
            ("tol", "許容誤差（判定用）", "float", self.settings.deviation_tol_mm, {"min": 0.001, "max": 100, "suffix": "mm"}),
        ], f"選択点 {len(P):,} 点。最小二乗平面から見て一価（オーバーハングなし）の領域に対応します。\n"
           "結果の残差が許容誤差を超える場合は制御点数を増やしてください。")
        if not v:
            return
        hint = None if N is None else N.mean(axis=0)

        def work(progress):
            return rc.fit_bspline_surface(P, v["nu"], v["nv"], v["deg"], v["smooth"], hint)

        def done(res):
            face, info = res
            o = self._add_cad(face, info, "Bスプライン曲面", ids)
            o.deviation = {"mesh_name": "選択領域", "mean": 0.0, "rms": info["rms"], "max_abs": info["max_abs"],
                           "tol": v["tol"], "within_tol": None, "n": info["n_points"]}
            self.add_object(o, "Bスプライン曲面")
            ok = info["max_abs"] <= v["tol"]
            msg = (f"制御点 {info['n_ctrl_u']}×{info['n_ctrl_v']} / 次数 {info['degree']}\n"
                   f"残差 RMS {info['rms']:.4f} mm / 最大 {info['max_abs']:.4f} mm / 95% {info['p95']:.4f} mm\n"
                   + ("許容誤差内です。" if ok else "⚠ 許容誤差を超えています。制御点数を増やすか領域を分割してください。")
                   + "\n\n制約: 曲面は矩形範囲で作成されます（輪郭トリムは未実装）。")
            self.info("Bスプライン曲面", msg)
        self.run("Bスプライン曲面フィット", work, on_done=done)

    def cmd_cad_boolean(self):
        occ = _occ()
        cads = self.selected_objects("cad")
        if len(cads) != 2:
            self.warn("ブーリアン", "ツリーで CAD 形状を2つ選択してください（1つ目が基準）。")
            return
        v = ParamDialog.ask(self, "ブーリアン", [
            ("op", "演算", "choice", "fuse", [("fuse", "結合 (A ∪ B)"), ("cut", "差 (A − B)"), ("common", "共通 (A ∩ B)")])],
            f"A = {cads[0].name}\nB = {cads[1].name}")
        if not v:
            return
        a, b = cads

        def work(progress):
            return occ.boolean(a.shape, b.shape, v["op"])

        def done(shape):
            o = self._add_cad(shape, {"type": "boolean", "op": v["op"], "a": a.id, "b": b.id},
                              f"{a.name}_{v['op']}_{b.name}", [a.id, b.id])
            self.add_object(o, "ブーリアン")
            self.doc.set_visible(a.id, False)
            self.doc.set_visible(b.id, False)
        self.run("ブーリアン", work, on_done=done)

    # ------------------------------------------------------------------
    # 偏差
    # ------------------------------------------------------------------
    def cmd_cad_deviation(self):
        from ..cad import reconstruct as rc
        cads = self.selected_objects("cad")
        if not cads:
            self.warn("CAD を選択", "ツリーで CAD 形状を選択してください。")
            return
        meshes = self.doc.objects("mesh")
        if not meshes:
            self.warn("メッシュなし", "比較するメッシュがありません。")
            return
        sel_mesh = self.selected_objects("mesh")
        dm = sel_mesh[0].id if sel_mesh else meshes[0].id
        v = ParamDialog.ask(self, "CAD-メッシュ偏差解析", [
            ("mesh", "比較するメッシュ", "choice", dm, [(m.id, m.name) for m in meshes]),
            ("tol", "許容値", "float", self.settings.deviation_tol_mm, {"min": 0.001, "max": 100, "suffix": "mm"}),
            ("range", "カラーマップ範囲 ±", "float", 3 * self.settings.deviation_tol_mm, {"min": 0.001, "max": 1000, "suffix": "mm"}),
            ("use_sel", "メッシュの選択領域のみ評価", "bool", False, None),
            ("maxd", "これより遠い点は対象外（0=すべて）", "float", 5.0, {"min": 0, "max": 1e5, "suffix": "mm"}),
        ], "メッシュ頂点から CAD 面までの符号付き距離を計算します（CAD はテッセレーション精度で近似）。")
        if not v:
            return
        mesh = self.doc.get(v["mesh"])
        shape = cads[0].shape if len(cads) == 1 else _occ().make_compound([c.shape for c in cads])
        P = mesh.V
        base_idx = np.arange(len(P))
        if v["use_sel"] and mesh.id in self.viewer.selection:
            from ..core.selection import vertices_from_face_mask
            vm = vertices_from_face_mask(mesh.F, self.viewer.selection[mesh.id], len(mesh.V))
            base_idx = np.nonzero(vm)[0]

        def work(progress):
            idx, d = rc.deviation_points_to_shape(P[base_idx], shape, min(0.01, self.settings.tess_deflection_mm))
            gi = base_idx[idx]
            if v["maxd"] > 0:
                keep = np.abs(d) <= v["maxd"]
                gi, d = gi[keep], d[keep]
            return gi, d

        def done(res):
            gi, d = res
            if len(d) == 0:
                self.warn("偏差解析", "評価できる点がありませんでした。")
                return
            st = rc.deviation_stats(d, v["tol"])
            st["mesh_name"] = mesh.name
            for c in cads:
                self.doc.modify(c.id, lambda o, st=st: setattr(o, "deviation", dict(st)), "偏差解析")
            self.viewer.set_deviation(mesh.id, gi, d, v["range"])
            self.info("偏差解析", f"比較メッシュ: {mesh.name}\n評価点数: {st['n']:,}\n平均（符号付き）: {st['mean']:+.4f} mm\n"
                      f"RMS: {st['rms']:.4f} mm / 最大: {st['max_abs']:.4f} mm / 95%: {st['p95']:.4f} mm\n"
                      f"許容 ±{v['tol']} mm 内: {st['within_tol'] * 100:.1f} %\n\n"
                      "符号: CAD 面の法線方向が正（外側）。灰色はスキップされた頂点。表示 > 偏差カラーマップを消去 で解除。")
        self.run("偏差解析", work, on_done=done)

    def cmd_feature_deviation(self):
        from ..cad.reconstruct import analytic_deviation, deviation_stats
        feats = self.selected_objects(("plane", "cylinder", "sphere"))
        if not feats:
            self.warn("要素を選択", "ツリーで平面・円筒・球を選択してください。")
            return
        f = feats[0]
        src = [self.doc.get(i) for i in f.meta.get("source_ids", [])]
        mesh = next((o for o in src if o is not None and o.kind == "mesh"), None) or self.target_scan(("mesh",))
        if mesh is None:
            return
        tol = self.settings.deviation_tol_mm
        d_all = analytic_deviation(f, mesh.V)
        if mesh.id in self.viewer.selection:
            from ..core.selection import vertices_from_face_mask
            idx = np.nonzero(vertices_from_face_mask(mesh.F, self.viewer.selection[mesh.id], len(mesh.V)))[0]
        else:
            idx = np.nonzero(np.abs(d_all) <= max(5 * tol, 3 * (f.stats or {}).get("max_abs", tol)))[0]
        if len(idx) == 0:
            self.warn("偏差", "評価する頂点がありません。")
            return
        d = d_all[idx]
        st = deviation_stats(d, tol)
        self.viewer.set_deviation(mesh.id, idx, d, 3 * tol)
        self.info("要素-メッシュ偏差", f"{f.name} ⇔ {mesh.name}\n評価点数: {st['n']:,}（選択領域、または要素近傍の頂点）\n"
                  f"平均 {st['mean']:+.4f} / RMS {st['rms']:.4f} / 最大 {st['max_abs']:.4f} mm\n"
                  f"許容 ±{tol} mm 内: {st['within_tol'] * 100:.1f} %")

    # ------------------------------------------------------------------
    # エクスポート
    # ------------------------------------------------------------------
    def _check_critical(self) -> bool:
        from ..io.export_csv import unverified_critical
        crit = unverified_critical(self.doc)
        if not crit:
            return True
        names = "\n".join("・" + o.name for o in crit[:15])
        r = QtWidgets.QMessageBox.warning(
            self, "実測確認のお願い",
            f"安全上重要に設定された要素のうち、実測確認が済んでいないものが {len(crit)} 件あります。\n{names}\n\n"
            "スキャンデータの座標は実測値ではありません。サスペンション取付点などは必ず実測で確認してください。\n"
            "このまま出力しますか？（出力ファイルにも未確認である旨を記録します）",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        return r == QtWidgets.QMessageBox.Yes

    def cmd_export_step(self):
        occ = _occ()
        cads = self.selected_objects("cad") or self.doc.objects("cad")
        axes = self.doc.objects("axis")
        holes = self.doc.objects("hole")
        if not cads and not axes and not holes:
            self.warn("STEP 出力", "出力する CAD 形状がありません。先に『CAD再構築』で形状を作成してください。\n"
                      "（メッシュをそのまま STEP に保存する機能はありません）")
            return
        v = ParamDialog.ask(self, "STEP 出力", [
            ("info", "出力する CAD 形状", "info", ", ".join(c.name for c in cads) or "（なし）", None),
            ("axes", f"基準軸をエッジとして含める（{len(axes)}）", "bool", False, None),
            ("holes", f"穴中心を円エッジとして含める（{len(holes)}）", "bool", False, None),
            ("datum", "基準点・穴中心・軸を CSV/JSON でも出力", "bool", True, None),
            ("schema", "スキーマ", "choice", self.settings.step_schema, [("AP214IS", "AP214"), ("AP242DIS", "AP242")]),
        ], "STEP には再構築した幾何形状（B-Rep）のみを出力します。出力後に再読込して形状と寸法を検証します。")
        if not v or not self._check_critical():
            return
        p = self.ask_save("STEP 出力", "STEP (*.step *.stp)", "reconstruction.step")
        if not p:
            return
        if not p.lower().endswith((".step", ".stp")):
            p += ".step"
        items = [(c.shape, c.name) for c in cads if c.shape is not None]
        if v["axes"]:
            from ..cad.reconstruct import axis_to_edge
            items += [(axis_to_edge(a)[0], a.name) for a in axes]
        if v["holes"]:
            from ..cad.reconstruct import hole_to_shape
            items += [(hole_to_shape(h, "edge")[0], h.name) for h in holes]
        if not items:
            self.warn("STEP 出力", "出力できる形状がありません。")
            return
        frame, fname = self.doc.active_frame, self.doc.active_frame_name
        schema = v["schema"]

        def work(progress):
            progress.report(0.2, "STEP 書込み中")
            occ.write_step(items, p, schema)
            progress.report(0.6, "STEP 再読込検証中")
            return occ.verify_step(p, [s for s, _ in items])

        def done(rep):
            extra = ""
            if v["datum"]:
                base = os.path.splitext(p)[0]
                from ..io.export_csv import export_features_csv
                n = export_features_csv(self.doc, base + "_datums.csv", frame, fname, 4)
                self._export_datum_json(base + "_datums.json")
                extra = f"\n基準要素 {n} 件を {os.path.basename(base)}_datums.csv / .json に出力しました。"
            ok = rep["ok"]
            text = (f"出力: {p}\n\n検証結果: {'✔ 一致（有効な形状）' if ok else '⚠ 不一致あり'}\n"
                    + "".join(f"  - {s}\n" for s in rep["issues"])
                    + f"\n元形状: {json.dumps(rep['reference'], ensure_ascii=False)}\n"
                    f"再読込: {json.dumps(rep['reloaded'], ensure_ascii=False)}\n半径(再読込): {rep['radii'][:20]}" + extra
                    + "\n\n注: STEP の座標はワールド座標系です。作業座標系で出力したい場合は"
                    "『位置合わせ > 選択座標系をワールドに合わせる』を先に実行してください。")
            log.info("STEP 出力 %s 検証=%s %s", p, ok, rep["issues"])
            ReportDialog("STEP 出力・検証", text, self).exec()
        self.run("STEP 出力", work, on_done=done)

    def _export_datum_json(self, path):
        f = self.doc.active_frame
        out = {"frame": self.doc.active_frame_name, "frame_matrix_local_to_world": f.matrix().tolist(),
               "units": "mm", "note": "座標は記載の座標系。安全上重要な値は必ず実測で確認すること。", "items": []}
        for o in self.doc.objects(("point", "hole", "axis", "cylinder", "plane", "cs")):
            d = {"name": o.name, "kind": o.kind, "critical": o.critical, "verified": o.verified,
                 "verified_note": o.verified_note, "category": getattr(o, "category", "")}
            if o.kind == "point":
                d["position"] = f.to_local(o.position).tolist()
            elif o.kind == "hole":
                d.update(center=f.to_local(o.center).tolist(), axis=f.dir_to_local(o.axis).tolist(), diameter=o.diameter)
            elif o.kind == "axis":
                d.update(point=f.to_local(o.point).tolist(), direction=f.dir_to_local(o.direction).tolist())
            elif o.kind == "cylinder":
                a, b = o.end_points()
                d.update(end1=f.to_local(a).tolist(), end2=f.to_local(b).tolist(),
                         direction=f.dir_to_local(o.axis_dir).tolist(), diameter=o.diameter)
            elif o.kind == "plane":
                d.update(center=f.to_local(o.center()).tolist(), normal=f.dir_to_local(o.normal).tolist())
            elif o.kind == "cs":
                d.update(matrix_local_to_world=o.frame.matrix().tolist())
            if getattr(o, "stats", None):
                d["fit"] = o.stats
            out["items"].append(d)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)

    def cmd_export_dxf(self):
        from ..io.dxf_export import export_sections_dxf
        secs = self.selected_objects("section") or self.doc.objects("section")
        if not secs:
            self.warn("DXF 出力", "断面曲線がありません。")
            return
        v = ParamDialog.ask(self, "DXF 出力", [
            ("mode", "座標", "choice", "2d", [("2d", "2D（断面平面の座標, スケッチ要素）"), ("3d", "3D（ワールド座標のポリライン）")]),
            ("raw", "スケッチ化済みでも元の曲線を別レイヤーに含める", "bool", False, None),
        ], f"{len(secs)} 断面を出力します。")
        if not v:
            return
        p = self.ask_save("DXF 出力", "DXF (*.dxf)", "sections.dxf")
        if not p:
            return
        if not p.lower().endswith(".dxf"):
            p += ".dxf"
        counts = export_sections_dxf(secs, p, v["mode"], include_raw=v["raw"])
        self.info("DXF 出力", f"出力しました: {p}\n要素数: {counts}")
        log.info("DXF 出力 %s %s", p, counts)

    def cmd_export_csv_features(self, ids=None):
        from ..io.export_csv import export_features_csv
        if not self._check_critical():
            return
        p = self.ask_save("CSV 出力（座標一覧）", "CSV (*.csv)", "coordinates.csv")
        if not p:
            return
        if not p.lower().endswith(".csv"):
            p += ".csv"
        n = export_features_csv(self.doc, p, self.doc.active_frame, self.doc.active_frame_name,
                                max(4, self.settings.length_decimals), ids)
        self.set_guide(f"CSV 出力: {n} 件 → {p}")
        log.info("CSV 出力 %s (%d 件)", p, n)

    def cmd_export_csv_measurements(self):
        from ..io.export_csv import export_measurements_csv
        p = self.ask_save("CSV 出力（測定結果）", "CSV (*.csv)", "measurements.csv")
        if not p:
            return
        if not p.lower().endswith(".csv"):
            p += ".csv"
        n = export_measurements_csv(self.doc, p, max(4, self.settings.length_decimals))
        self.set_guide(f"測定結果 CSV: {n} 件 → {p}")

    def cmd_export_distance_matrix(self, ids=None):
        from ..io.export_csv import export_distance_matrix_csv
        if ids is None:
            sel = self.selected_objects(("point", "hole", "sphere", "cs", "cylinder", "axis"))
            objs = sel or self.doc.objects(("point", "hole"))
            ids = [o.id for o in objs]
        if len(ids) < 2:
            self.warn("距離マトリクス", "2つ以上の点要素（基準点・穴など）が必要です。")
            return
        p = self.ask_save("点間距離マトリクス", "CSV (*.csv)", "distance_matrix.csv")
        if not p:
            return
        if not p.lower().endswith(".csv"):
            p += ".csv"
        n = export_distance_matrix_csv(self.doc, p, ids, self.doc.active_frame, max(3, self.settings.length_decimals))
        self.set_guide(f"距離マトリクス: {n}×{n} → {p}")

    def cmd_export_json(self):
        from ..model.project_io import export_project_json
        p = self.ask_save("JSON 出力", "JSON (*.json)", "project.json")
        if not p:
            return
        if not p.lower().endswith(".json"):
            p += ".json"
        export_project_json(self.doc, p, self.settings.to_dict())
        self.set_guide(f"JSON 出力: {p}")

    def cmd_export_mesh(self):
        from ..io.mesh_io import export_mesh
        o = self.target_scan(("mesh", "points"))
        if o is None:
            return
        if o.kind == "mesh":
            p = self.ask_save("メッシュ出力", "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)", o.name + ".stl")
        else:
            p = self.ask_save("点群出力", "PLY (*.ply)", o.name + ".ply")
        if not p:
            return
        if os.path.splitext(p)[1].lower() not in (".stl", ".obj", ".ply"):
            p += ".stl" if o.kind == "mesh" else ".ply"
        if o.kind == "mesh":
            export_mesh(p, o.V, o.F, o.colors)
        else:
            export_mesh(p, o.P, None, o.colors)
        self.set_guide(f"出力しました: {p}")

    def cmd_export_points_dxf(self):
        from ..io.dxf_export import export_points_dxf
        objs = self.doc.objects(("point", "hole"))
        if not objs:
            self.warn("DXF 出力", "基準点・穴中心がありません。")
            return
        p = self.ask_save("基準点 DXF", "DXF (*.dxf)", "datum_points.dxf")
        if not p:
            return
        if not p.lower().endswith(".dxf"):
            p += ".dxf"
        pts = [o.position if o.kind == "point" else o.center for o in objs]
        export_points_dxf(pts, [o.name for o in objs], p)  # DXF R2018 は UTF-8 文字列に対応
        self.set_guide(f"DXF 出力: {p}（ワールド座標）")
