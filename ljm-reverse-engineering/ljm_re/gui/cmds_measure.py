"""測定・座標系・位置合わせコマンド."""
from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtWidgets

from ..core import measure as M
from ..core.coords import (Frame, frame_from_321, frame_from_axis_point, frame_from_plane_axis,
                           frame_from_three_points)
from ..core.registration import evaluate_alignment, icp, kabsch
from ..model.objects import CoordinateSystemObject, MeasurementObject
from .dialogs import MountPointsDialog, ParamDialog, ReportDialog
from .viewer import PickPointsTool

log = logging.getLogger(__name__)

AXIS_KINDS = ("cylinder", "hole", "axis")
POINT_KINDS = ("point", "hole", "sphere", "cs")


def axis_of(o):
    if o.kind == "cylinder":
        return o.axis_point, o.axis_dir, o.t_range
    if o.kind == "hole":
        L = (o.depth or o.diameter) / 2
        return o.center, o.axis, (-L, L)
    if o.kind == "axis":
        return o.point, o.direction, (-o.length / 2, o.length / 2)
    raise ValueError(f"{o.name} は軸を持ちません")


def point_of(o):
    if o.kind == "point":
        return o.position
    if o.kind in ("hole", "sphere"):
        return o.center
    if o.kind == "cs":
        return o.frame.origin
    raise ValueError(f"{o.name} は点を持ちません")


class MeasureCommands:
    def _build_measure_actions(self, A):
        A("m_dist", "2点間距離", self.cmd_measure_distance, "D", "モデル上の2点をクリック（特徴点にスナップ）")
        A("m_angle", "3点による角度", self.cmd_measure_angle, tip="点1 → 頂点 → 点2 の順にクリック")
        A("m_coord", "XYZ座標の取得", self.cmd_measure_coordinate, "C")
        A("m_point_plane", "点と平面の距離", self.cmd_measure_point_plane, tip="ツリーで平面を選択してから点をクリック")
        A("m_between", "選択要素間の測定", self.cmd_measure_between, "M",
          "ツリーで2要素を選択: 平面間距離/角度・軸間距離/同軸度・穴中心間距離・点-平面・点-軸・軸-平面")
        A("m_projected", "投影距離（方向指定）", self.cmd_measure_projected)
        A("m_bbox", "バウンディングボックス", self.cmd_measure_bbox)
        A("m_diameter", "直径・半径", self.cmd_measure_diameter)
        A("m_scale", "スケール確認・補正（既知寸法）", self.cmd_scale_check)
        A("mount_points", "取付点一覧（サブフレーム等）", self.cmd_mount_points)
        A("cs_activate", "作業座標系に設定", self.cmd_activate_cs)
        A("cs_world", "ワールド座標系に戻す", self.cmd_world_cs)
        A("cs_3pt", "座標系: 3点から", self.cmd_cs_3points, tip="原点 → X軸方向 → XY平面上の点 をクリック")
        A("cs_plane_axis", "座標系: 平面＋軸", self.cmd_cs_plane_axis, tip="Z=平面法線, X=軸の投影, 原点=交点")
        A("cs_321", "座標系: 3平面 (3-2-1)", self.cmd_cs_321)
        A("cs_axis_point", "座標系: 軸＋点", self.cmd_cs_axis_point)
        A("cs_numeric", "座標系: 数値入力", self.cmd_cs_numeric)
        A("align_to_cs", "選択座標系をワールドに合わせる", self.cmd_align_to_cs,
          tip="全オブジェクトを移動し、選択した座標系をワールド座標系に一致させる")
        A("reg_points", "対応点による位置合わせ", self.cmd_register_points)
        A("reg_icp", "ICP 精密位置合わせ", self.cmd_register_icp)
        A("reg_eval", "位置合わせ評価（重複・残差）", self.cmd_register_eval)

    # ------------------------------------------------------------------
    def _add_measurement(self, mtype, result, anchors, refs=(), name=None):
        result = dict(result)
        result["frame_name"] = self.doc.active_frame_name
        m = MeasurementObject(name or self.doc.auto_name("measurement"), mtype, result, anchors, list(refs))
        from ..model.objects import MEASURE_LABELS
        m.name = name or self.doc.unique_name(MEASURE_LABELS.get(mtype, "測定"))
        self.add_object(m, "測定")
        txt = m.summary_text(self.fmt())
        self.set_guide(f"{m.name}: {txt}")
        log.info("測定 %s: %s", m.name, txt)
        return m

    def _pick(self, n, hints, cb, allowed_kinds=None):
        self.viewer.set_tool(PickPointsTool(self.viewer, n, cb, hints, allowed_kinds=allowed_kinds))

    def cmd_measure_distance(self):
        def cb(pts):
            (p1, o1, l1), (p2, o2, l2) = pts
            r = M.distance_point_point(p1, p2, self.doc.active_frame)
            r["labels"] = [l1, l2]
            self._add_measurement("point_point", r, [p1, p2], {o1, o2})
        self._pick(2, ["1点目をクリック", "2点目をクリック"], cb)

    def cmd_measure_angle(self):
        def cb(pts):
            a, v, b = (p[0] for p in pts)
            r = M.angle_three_points(a, v, b)
            self._add_measurement("angle_3pt", r, [a, v, b], {p[1] for p in pts})
        self._pick(3, ["点1をクリック", "頂点をクリック", "点2をクリック"], cb)

    def cmd_measure_coordinate(self):
        def cb(pts):
            p, oid, lab = pts[0]
            r = M.coordinate(p, self.doc.active_frame)
            r["label"] = lab
            self._add_measurement("coordinate", r, [p], {oid})
        self._pick(1, ["座標を取得する点をクリック"], cb)

    def cmd_measure_point_plane(self):
        planes = self.selected_objects("plane")
        if not planes:
            self.warn("平面を選択", "モデルツリーで基準となる平面を選択してから実行してください。")
            return
        pl = planes[0]

        def cb(pts):
            p, oid, _ = pts[0]
            r = M.point_plane(p, pl.origin, pl.normal)
            self._add_measurement("point_plane", r, [p, np.array(r["foot"])], {oid, pl.id})
        self._pick(1, [f"『{pl.name}』までの距離を測る点をクリック"], cb)

    def cmd_measure_between(self):
        objs = self.selected_objects()
        if len(objs) == 1 and objs[0].kind in ("cylinder", "hole", "sphere"):
            return self.cmd_measure_diameter()
        if len(objs) != 2:
            self.warn("2要素を選択", "モデルツリーで2つの要素（平面・円筒・穴・基準点・基準軸など）を Ctrl+クリックで選択してください。")
            return
        a, b = objs
        ka, kb = a.kind, b.kind
        refs = [a.id, b.id]
        if ka == "plane" and kb == "plane":
            r = M.plane_plane(a.origin, a.normal, b.origin, b.normal)
            c = a.center()
            foot = c - np.dot(c - b.origin, b.normal) * b.normal
            return self._add_measurement("plane_plane", r, [c, foot], refs, f"{a.name}-{b.name}")
        if ka in AXIS_KINDS and kb in AXIS_KINDS and not (ka == "hole" and kb == "hole"):
            p1, d1, _ = axis_of(a)
            p2, d2, ext2 = axis_of(b)
            r = M.axis_axis(p1, d1, p2, d2, extent2=ext2)
            return self._add_measurement("axis_axis", r, [np.array(r["closest_1"]), np.array(r["closest_2"])],
                                         refs, f"{a.name}-{b.name}")
        if ka in POINT_KINDS and kb in POINT_KINDS:
            p1, p2 = point_of(a), point_of(b)
            r = M.distance_point_point(p1, p2, self.doc.active_frame)
            return self._add_measurement("hole_hole", r, [p1, p2], refs, f"{a.name}-{b.name}")
        for x, y in ((a, b), (b, a)):
            if x.kind in POINT_KINDS and y.kind == "plane":
                p = point_of(x)
                r = M.point_plane(p, y.origin, y.normal)
                return self._add_measurement("point_plane", r, [p, np.array(r["foot"])], refs, f"{x.name}-{y.name}")
            if x.kind in POINT_KINDS and y.kind in AXIS_KINDS:
                p = point_of(x)
                ap, ad, _ = axis_of(y)
                r = M.point_axis(p, ap, ad)
                foot = ap + np.dot(p - ap, ad) * ad
                return self._add_measurement("point_axis", r, [p, foot], refs, f"{x.name}-{y.name}")
            if x.kind in AXIS_KINDS and y.kind == "plane":
                ap, ad, _ = axis_of(x)
                r = M.axis_plane(ap, ad, y.origin, y.normal)
                anchors = [ap, np.array(r["intersection"])] if r["intersection"] is not None else [ap]
                return self._add_measurement("axis_plane", r, anchors, refs, f"{x.name}-{y.name}")
        self.warn("未対応の組み合わせ", f"{a.name}（{ka}）と {b.name}（{kb}）の測定には対応していません。")

    def cmd_measure_projected(self):
        dirs = [("x", "作業座標系 X"), ("y", "作業座標系 Y"), ("z", "作業座標系 Z")]
        feats = self.selected_objects(("plane",) + AXIS_KINDS)
        for f in feats:
            dirs.append((f.id, f"{f.name} の{'法線' if f.kind == 'plane' else '軸'}方向"))
        v = ParamDialog.ask(self, "投影距離", [("dir", "投影方向", "choice", "x", dirs)],
                            "2点をクリックし、指定方向へ投影した距離を測定します。要素の方向を使う場合はツリーで選択してから実行してください。")
        if not v:
            return
        key = v["dir"]
        f = self.doc.active_frame
        if key in ("x", "y", "z"):
            d = f.R[:, "xyz".index(key)]
        else:
            o = self.doc.get(key)
            d = o.normal if o.kind == "plane" else axis_of(o)[1]

        def cb(pts):
            (p1, o1, _), (p2, o2, _) = pts
            r = M.projected_distance(p1, p2, d)
            self._add_measurement("projected", r, [p1, p2], {o1, o2})
        self._pick(2, ["1点目をクリック", "2点目をクリック"], cb)

    def cmd_measure_bbox(self):
        o = self.target_scan(("mesh", "points"))
        if o is None:
            return
        P = o.V if o.kind == "mesh" else o.P
        f = self.doc.active_frame
        r = M.bounding_box(P, f)
        lo, hi = f.to_world(np.array(r["min"])), f.to_world(np.array(r["max"]))
        self._add_measurement("bbox", r, [lo, hi], [o.id], f"BBox {o.name}")

    def cmd_measure_diameter(self):
        objs = self.selected_objects(("cylinder", "hole", "sphere"))
        if not objs:
            self.warn("要素を選択", "ツリーで円筒・穴・球を選択してください。")
            return
        for o in objs:
            if o.kind == "cylinder":
                d, c, n = o.diameter, 0.5 * sum(o.end_points()), o.axis_dir
            elif o.kind == "hole":
                d, c, n = o.diameter, o.center, o.axis
            else:
                d, c, n = 2 * o.radius, o.center, np.array([0, 0, 1.0])
            from ..core.geometry import orthonormal_basis
            u, _ = orthonormal_basis(n)
            r = {"type": "diameter", "diameter": d, "radius": d / 2}
            self._add_measurement("diameter", r, [c - u * d / 2, c + u * d / 2], [o.id], f"φ {o.name}")

    def cmd_scale_check(self):
        cands = [m for m in self.selected_objects("measurement") if m.result.get("distance")]
        measured = cands[0].result["distance"] if cands else 100.0
        v = ParamDialog.ask(self, "スケール確認・補正", [
            ("measured", "スキャン上の測定値", "float", measured, {"min": 1e-6, "max": 1e7, "decimals": 4, "suffix": "mm"}),
            ("reference", "既知の実寸（ノギス等の実測値）", "float", measured, {"min": 1e-6, "max": 1e7, "decimals": 4, "suffix": "mm"}),
            ("apply", "スケール補正を全オブジェクトに適用する", "bool", False, None),
        ], "既知寸法（例: ブロックゲージ、ノギスで測った穴間距離）とスキャン上の測定値を比較します。\n"
           "測定値は、ツリーで距離測定を選択してから実行すると自動入力されます。")
        if not v:
            return
        r = M.scale_factor(v["measured"], v["reference"])
        msg = (f"スケール係数: {r['factor']:.6f}\n誤差: {r['error_mm']:+.4f} mm ({r['error_percent']:+.4f} %)\n\n"
               "注意: 1か所の比較だけでは精度保証になりません。離れた複数箇所で確認してください。")
        self._add_measurement("scale", r, [], [c.id for c in cands[:1]], "スケール確認")
        if v["apply"] and abs(r["factor"] - 1) > 1e-12:
            pts = self.doc.all_points_for_bounds()
            center = pts.mean(axis=0) if pts is not None else np.zeros(3)
            self.doc.scale(self.doc.ids(), r["factor"], center, f"スケール補正 ×{r['factor']:.6f}")
            msg += f"\n\n全オブジェクトにスケール補正を適用しました（中心 {np.round(center, 3).tolist()}）。"
        self.info("スケール確認", msg)

    def cmd_mount_points(self):
        MountPointsDialog(self, self).exec()

    # ------------------------------------------------------------------
    # 座標系
    # ------------------------------------------------------------------
    def cmd_activate_cs(self):
        cs = self.selected_objects("cs")
        if not cs:
            self.warn("座標系を選択", "モデルツリーで座標系を選択してください。")
            return
        self.cmd_activate_cs_id(cs[0].id)

    def cmd_activate_cs_id(self, oid):
        self.doc.set_active_cs(oid)
        self.viewer.refresh_object(oid)
        for o in self.doc.objects("cs"):
            self.viewer.refresh_object(o.id)
        self.set_guide(f"作業座標系: {self.doc.active_frame_name}（座標表示・CSV出力・ビュー方向に使用）")

    def cmd_world_cs(self):
        self.doc.set_active_cs(None)
        for o in self.doc.objects("cs"):
            self.viewer.refresh_object(o.id)

    def _add_cs(self, frame, method, refs=()):
        pts = self.doc.all_points_for_bounds()
        size = 0.15 * float(np.linalg.norm(pts.max(0) - pts.min(0))) if pts is not None else 50.0
        cs = CoordinateSystemObject(self.doc.auto_name("cs"), frame)
        cs.method = method
        cs.display_size = max(size, 5.0)
        cs.meta = self.feature_meta(refs)
        self.add_object(cs, f"座標系作成 ({method})")
        if self.confirm("作業座標系", f"{cs.name} を作業座標系に設定しますか？"):
            self.cmd_activate_cs_id(cs.id)
        return cs

    def cmd_cs_3points(self):
        def cb(pts):
            try:
                f = frame_from_three_points(pts[0][0], pts[1][0], pts[2][0])
            except ValueError as e:
                self.warn("座標系", str(e))
                return
            self._add_cs(f, "3点", [p[1] for p in pts])
        self._pick(3, ["原点をクリック", "X軸方向の点をクリック", "XY平面上（Y正側）の点をクリック"], cb)

    def cmd_cs_plane_axis(self):
        pl = self.selected_objects("plane")
        ax = self.selected_objects(AXIS_KINDS)
        if not pl or not ax:
            self.warn("要素を選択", "ツリーで平面1つと軸（円筒/穴/基準軸）1つを選択してください。")
            return
        p, d, _ = axis_of(ax[0])
        f = frame_from_plane_axis(pl[0].origin, pl[0].normal, p, d)
        self._add_cs(f, "平面+軸", [pl[0].id, ax[0].id])

    def cmd_cs_321(self):
        planes = self.doc.objects("plane")
        if len(planes) < 3:
            self.warn("平面が不足", "3平面座標系には平面が3つ以上必要です。")
            return
        sel = [o.id for o in self.selected_objects("plane")]
        ch = [(o.id, o.name) for o in planes]
        dflt = (sel + [o.id for o in planes if o.id not in sel])[:3]
        v = ParamDialog.ask(self, "3平面座標系 (3-2-1)", [
            ("p1", "第1平面（Z軸 = 法線）", "choice", dflt[0], ch),
            ("p2", "第2平面（X軸 = 法線を第1平面へ投影）", "choice", dflt[1], ch),
            ("p3", "第3平面（原点を決める）", "choice", dflt[2], ch),
        ], "原点は3平面の交点です。自動車部品の基準面（取付面・側面・端面）による位置決めに使います。")
        if not v:
            return
        a, b, c = (self.doc.get(v[k]) for k in ("p1", "p2", "p3"))
        try:
            f = frame_from_321((a.origin, a.normal), (b.origin, b.normal), (c.origin, c.normal))
        except ValueError as e:
            self.warn("座標系", str(e))
            return
        self._add_cs(f, "3平面(3-2-1)", [a.id, b.id, c.id])

    def cmd_cs_axis_point(self):
        ax = self.selected_objects(AXIS_KINDS)
        if not ax:
            self.warn("軸を選択", "ツリーで軸（円筒/穴/基準軸）を選択してから実行してください。")
            return
        p, d, _ = axis_of(ax[0])

        def cb(pts):
            f = frame_from_axis_point(p, d, pts[0][0])
            self._add_cs(f, "軸+点", [ax[0].id, pts[0][1]])
        self._pick(1, ["X軸方向を決める点をクリック（Z=軸）"], cb)

    def cmd_cs_numeric(self):
        v = ParamDialog.ask(self, "座標系: 数値入力（ワールド基準）", [
            ("x", "原点 X", "float", 0.0, {"decimals": 4, "suffix": "mm"}),
            ("y", "原点 Y", "float", 0.0, {"decimals": 4, "suffix": "mm"}),
            ("z", "原点 Z", "float", 0.0, {"decimals": 4, "suffix": "mm"}),
            ("rx", "回転 X", "float", 0.0, {"decimals": 4, "suffix": "°", "min": -360, "max": 360}),
            ("ry", "回転 Y", "float", 0.0, {"decimals": 4, "suffix": "°", "min": -360, "max": 360}),
            ("rz", "回転 Z", "float", 0.0, {"decimals": 4, "suffix": "°", "min": -360, "max": 360}),
        ], "回転は固定軸 X → Y → Z の順に適用します。作成後もプロパティで編集できます。")
        if v:
            self._add_cs(Frame.from_euler([v["x"], v["y"], v["z"]], v["rx"], v["ry"], v["rz"]), "数値入力")

    def cmd_align_to_cs(self):
        cs = self.selected_objects("cs")
        cs = cs[0] if cs else (self.doc.get(self.doc.active_cs_id) if self.doc.active_cs_id else None)
        if cs is None:
            self.warn("座標系を選択", "ワールドに合わせる座標系をツリーで選択してください。")
            return
        if not self.confirm("位置合わせ", f"全オブジェクトを移動して『{cs.name}』をワールド座標系に一致させます。\n"
                                       "（元に戻す で取り消せます。元のスキャンデータは保持されます）"):
            return
        T = cs.frame.inverse_matrix()
        self.doc.transform(self.doc.ids(), T, f"{cs.name} をワールドに合わせる")
        self.doc.set_active_cs(None)
        self.viewer.fit_view()

    # ------------------------------------------------------------------
    # 位置合わせ
    # ------------------------------------------------------------------
    def _choose_pair(self, title, extra=None, desc=""):
        scans = self.doc.objects(("mesh", "points"))
        if len(scans) < 2:
            self.warn(title, "位置合わせには2つ以上のメッシュ/点群が必要です。")
            return None
        ch = [(o.id, o.name) for o in scans]
        sel = [o.id for o in self.selected_objects(("mesh", "points"))]
        mv = sel[0] if sel else scans[-1].id
        fx = sel[1] if len(sel) > 1 else next(o.id for o in scans if o.id != mv)
        fields = [("moving", "移動するデータ", "choice", mv, ch), ("fixed", "基準（固定）データ", "choice", fx, ch)]
        fields += extra or []
        v = ParamDialog.ask(self, title, fields, desc)
        if not v:
            return None
        if v["moving"] == v["fixed"]:
            self.warn(title, "移動側と固定側に同じデータは指定できません。")
            return None
        return v

    def _related_ids(self, oid):
        return [o.id for o in self.doc.objects() if oid in o.meta.get("source_ids", [])]

    def cmd_register_points(self):
        v = self._choose_pair("対応点による位置合わせ", [
            ("n", "対応点の組数", "int", 4, {"min": 3, "max": 20}),
            ("with_related", "移動データから作成した要素も一緒に移動", "bool", True, None),
        ], "移動側 → 固定側 の順に、同じ特徴（角・穴中心など）を交互にクリックします。")
        if not v:
            return
        mov, fix = self.doc.get(v["moving"]), self.doc.get(v["fixed"])
        n = v["n"]
        hints = []
        for i in range(n):
            hints += [f"移動側『{mov.name}』の点 {i + 1}", f"固定側『{fix.name}』の点 {i + 1}"]

        def cb(pts):
            src = np.array([pts[2 * i][0] for i in range(n)])
            dst = np.array([pts[2 * i + 1][0] for i in range(n)])
            try:
                T, rms = kabsch(src, dst)
            except ValueError as e:
                self.warn("位置合わせ", str(e))
                return
            ids = [mov.id] + (self._related_ids(mov.id) if v["with_related"] else [])
            self.doc.transform(ids, T, "対応点による位置合わせ")
            res = np.linalg.norm((src @ T[:3, :3].T + T[:3, 3]) - dst, axis=1)
            self.info("対応点による位置合わせ", f"対応点 RMS: {rms:.4f} mm / 最大: {res.max():.4f} mm\n"
                      "続けて ICP 精密位置合わせを実行すると精度が向上します。")
            log.info("対応点位置合わせ %s→%s RMS %.4f", mov.name, fix.name, rms)
        self._pick(2 * n, hints, cb)

    def cmd_register_icp(self):
        s = self.settings
        v = self._choose_pair("ICP 精密位置合わせ", [
            ("maxd", "対応点の最大距離 (0=自動)", "float", 0.0, {"min": 0, "max": 1e5, "suffix": "mm"}),
            ("iters", "最大反復回数", "int", s.icp_max_iter, {"min": 1, "max": 1000}),
            ("sample", "使用点数（移動側）", "int", s.icp_sample, {"min": 1000, "max": 2_000_000}),
            ("method", "手法", "choice", "point_to_plane", [("point_to_plane", "点-面 (推奨)"), ("point_to_point", "点-点")]),
            ("use_sel", "移動側の選択領域のみ使用", "bool", False, None),
            ("with_related", "移動データから作成した要素も一緒に移動", "bool", True, None),
        ], "事前に対応点などでおおまかに位置を合わせておく必要があります（局所最適化）。")
        if not v:
            return
        mov, fix = self.doc.get(v["moving"]), self.doc.get(v["fixed"])
        S = mov.V if mov.kind == "mesh" else mov.P
        if v["use_sel"] and mov.id in self.viewer.selection:
            P, _, _, _ = self.viewer.selected_points(False)
            if P is not None:
                S = P
        Tg = fix.V if fix.kind == "mesh" else fix.P
        TN = fix.vertex_normals if fix.kind == "mesh" else fix.normals

        def work(progress):
            return icp(S, Tg, TN, None, v["iters"], v["maxd"] or None, v["method"], v["sample"],
                       progress=progress)

        def done(r):
            ids = [mov.id] + (self._related_ids(mov.id) if v["with_related"] else [])
            self.doc.transform(ids, r.transform, "ICP 位置合わせ")
            msg = (f"RMS 残差: {r.rms:.4f} mm（{'点-面' if TN is not None and v['method'] == 'point_to_plane' else '点-点'}）\n"
                   f"重複率（閾値 {r.max_distance:.3f} mm 以内の点）: {r.fitness * 100:.1f} %\n"
                   f"反復: {r.iterations} 回 / 収束: {'はい' if r.converged else 'いいえ'}\n\n"
                   "注意: 位置合わせの残差が小さくても測定精度が保証されるわけではありません。\n"
                   "基準寸法（スケール確認）で検証してください。")
            self.info("ICP 位置合わせ", msg)
            log.info("ICP %s→%s: RMS %.4f, overlap %.3f, iter %d", mov.name, fix.name, r.rms, r.fitness, r.iterations)
        self.run("ICP 位置合わせ", work, on_done=done)

    def cmd_register_eval(self):
        v = self._choose_pair("位置合わせ評価", [
            ("thr", "重複判定距離", "float", 0.5, {"min": 0.001, "max": 1e4, "suffix": "mm"})])
        if not v:
            return
        mov, fix = self.doc.get(v["moving"]), self.doc.get(v["fixed"])
        S = mov.V if mov.kind == "mesh" else mov.P
        Tg = fix.V if fix.kind == "mesh" else fix.P

        def work(progress):
            return evaluate_alignment(S, Tg, v["thr"])

        def done(r):
            ReportDialog("位置合わせ評価", (
                f"移動: {mov.name} / 固定: {fix.name}\n"
                f"重複判定距離: {r['threshold']} mm\n重複率: {r['overlap_ratio'] * 100:.2f} %\n"
                f"重複領域の RMS 距離: {r['rms']:.4f} mm（最近傍頂点距離。点密度の影響を含む）\n"
                f"平均: {r['mean']:.4f} mm / 95%: {r['p95']:.4f} mm\n評価点数: {r['n']:,}\n\n"
                "注意: これは2つのスキャン間の一致度であり、寸法精度の保証ではありません。"), self).exec()
        self.run("位置合わせ評価", work, on_done=done)
