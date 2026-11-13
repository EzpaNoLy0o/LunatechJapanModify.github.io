"""選択ツールとスキャンデータ（メッシュ/点群）編集コマンド."""
from __future__ import annotations

import logging

import numpy as np

from ..core import mesh_ops
from ..core.selection import grow_selection, shrink_selection
from ..model.objects import MeshObject, PointCloudObject
from .dialogs import ParamDialog
from .viewer import BrushSelectTool, RectSelectTool, SmartSelectTool

log = logging.getLogger(__name__)


class ScanCommands:
    def _build_scan_actions(self, A):
        A("mesh_del_sel", "選択面を削除", self.cmd_mesh_delete_selected, tip="選択した面を削除（不要領域の削除）")
        A("mesh_keep_sel", "選択以外を削除（トリミング）", self.cmd_mesh_keep_selected)
        A("mesh_noise", "ノイズ除去（浮遊片）", self.cmd_mesh_remove_small, tip="小さな浮遊片（連結成分）を削除")
        A("mesh_smooth", "スムージング", self.cmd_mesh_smooth, tip="Taubin スムージング（選択があれば選択範囲のみ）")
        A("mesh_decimate", "簡略化", self.cmd_mesh_decimate)
        A("mesh_fix_normals", "法線修正（向きの統一）", self.cmd_mesh_fix_normals)
        A("mesh_flip", "法線反転", self.cmd_mesh_flip)
        A("mesh_fill", "穴埋め", self.cmd_mesh_fill, tip="生成した面は『人工面』として実測面と区別されます")
        A("mesh_split", "連結成分で分割", self.cmd_mesh_split)
        A("mesh_separate", "選択領域を別メッシュに分離", self.cmd_mesh_separate)
        A("mesh_merge", "メッシュ結合", self.cmd_mesh_merge, tip="ツリーで選択した複数メッシュを1つに結合")
        A("mesh_restore", "オリジナルに戻す", self.cmd_mesh_restore, tip="読込時のスキャンデータに戻す（位置合わせは保持）")
        A("pc_outlier", "点群: 外れ値除去", self.cmd_pc_outliers)
        A("pc_normals", "点群: 法線推定", self.cmd_pc_normals)
        A("pc_poisson", "点群: メッシュ化（Poisson, Open3D）", self.cmd_pc_poisson)
        A("mesh_to_points", "メッシュから点群を作成", self.cmd_mesh_to_points)
        A("rescale", "単位/スケール変更…", self.cmd_rescale)
        try:
            from ..core.mesh_ops import open3d_available
            if not open3d_available():
                self.actions["pc_poisson"].setEnabled(False)
                self.actions["pc_poisson"].setToolTip("Open3D がインストールされていないため使用できません（pip install open3d）")
        except Exception:
            pass

    # ---- 選択ツール ---------------------------------------------------------
    def cmd_tool_nav(self):
        self.viewer.set_tool(None)
        self.set_guide("ナビゲーション: 左/右ドラッグで回転、中ドラッグでパン、ホイールでズーム")

    def cmd_tool_rect(self):
        self.viewer.set_tool(RectSelectTool(self.viewer))

    def cmd_tool_smart(self):
        self.viewer.set_tool(SmartSelectTool(self.viewer))

    def cmd_tool_brush(self):
        self.viewer.set_tool(BrushSelectTool(self.viewer))

    def cmd_clear_selection(self):
        self.viewer.clear_selection()

    def _sel_items(self):
        out = []
        for oid, m in list(self.viewer.selection.items()):
            o = self.doc.get(oid)
            if o is not None:
                out.append((o, m))
        return out

    def cmd_grow_selection(self):
        for o, m in self._sel_items():
            if o.kind == "mesh":
                self.viewer.set_selection(o.id, grow_selection(o.topology, m), "add", render=False)
        self.viewer.render()

    def cmd_shrink_selection(self):
        for o, m in self._sel_items():
            if o.kind == "mesh":
                self.viewer.selection[o.id] = shrink_selection(o.topology, m)
                self.viewer._update_sel_actor(o.id)
        self.viewer.selectionChanged.emit()
        self.viewer.render()

    def cmd_invert_selection(self):
        items = self._sel_items()
        if not items:
            o = self.target_scan()
            if o is None:
                return
            n = len(o.F) if o.kind == "mesh" else len(o.P)
            items = [(o, np.zeros(n, bool))]
        for o, m in items:
            self.viewer.selection[o.id] = ~m
            self.viewer._update_sel_actor(o.id)
        self.viewer.selectionChanged.emit()
        self.viewer.render()

    def cmd_select_all(self):
        o = self.target_scan()
        if o is None:
            return
        n = len(o.F) if o.kind == "mesh" else len(o.P)
        self.viewer.set_selection(o.id, np.ones(n, bool), "replace")

    def _require_selection(self, kinds=("mesh", "points")):
        items = [(o, m) for o, m in self._sel_items() if o.kind in kinds and m.any()]
        if not items:
            self.warn("選択がありません", "先に選択ツール（R: 矩形 / G: スマート / B: ブラシ）で領域を選択してください。")
        return items

    # ---- メッシュ編集（すべて Undo 可能） -------------------------------------
    def _mesh_edit(self, o, label, fn):
        """fn(o) -> (V, F, flags, colors) を適用."""
        def apply(obj):
            V, F, fl, c = fn(obj)
            obj.set_geometry(V, F, fl, c)
        self.doc.modify(o.id, apply, label)

    def cmd_mesh_delete_selected(self):
        items = self._require_selection()
        if any(m.all() for _, m in items):
            self.warn("選択面を削除", "すべての面/点が選択されています。オブジェクト全体を削除する場合は"
                      "モデルツリーで選択して Delete キーを押してください。")
            return
        for o, m in items:
            if o.kind == "mesh":
                self._mesh_edit(o, "選択面を削除", lambda ob, m=m: mesh_ops.remove_faces(ob.V, ob.F, m, ob.face_flags, ob.colors))
            else:
                keep = ~m
                self.doc.modify(o.id, lambda ob, k=keep: ob.set_points(
                    ob.P[k], None if ob.normals is None else ob.normals[k], None if ob.colors is None else ob.colors[k]),
                    "選択点を削除")
        self.viewer.clear_selection()

    def cmd_mesh_keep_selected(self):
        items = self._require_selection()
        for o, m in items:
            if o.kind == "mesh":
                self._mesh_edit(o, "トリミング", lambda ob, m=m: mesh_ops.keep_faces(ob.V, ob.F, m, ob.face_flags, ob.colors))
            else:
                self.doc.modify(o.id, lambda ob, k=m: ob.set_points(
                    ob.P[k], None if ob.normals is None else ob.normals[k], None if ob.colors is None else ob.colors[k]),
                    "トリミング")
        self.viewer.clear_selection()

    def cmd_mesh_remove_small(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return
        v = ParamDialog.ask(self, "ノイズ除去（浮遊片の削除）", [
            ("ratio", "最大成分に対する面数比がこれ未満を削除", "float", 1.0, {"min": 0.0, "max": 100, "suffix": "%"}),
            ("min_faces", "かつ面数がこれ未満を削除", "int", 100, {"min": 0, "max": 10 ** 8}),
        ])
        if not v:
            return

        def work(progress, V, F, fl, c):
            return mesh_ops.remove_small_components(V, F, v["min_faces"], v["ratio"] / 100.0, fl, c)

        def done(res):
            V, F, fl, c, removed, ncomp = res
            if removed == 0:
                self.info("ノイズ除去", "削除対象の浮遊片はありませんでした。")
                return
            self.doc.modify(o.id, lambda ob: ob.set_geometry(V, F, fl, c), "ノイズ除去")
            self.set_guide(f"ノイズ除去: {ncomp} 個の浮遊片（{removed:,} 面）を削除しました")
        self.run("ノイズ除去", work, o.V, o.F, o.face_flags, o.colors, on_done=done)

    def cmd_mesh_smooth(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return
        v = ParamDialog.ask(self, "スムージング (Taubin)", [
            ("iterations", "反復回数", "int", 10, {"min": 1, "max": 500}),
            ("lamb", "λ", "float", 0.5, {"min": 0.01, "max": 1.0}),
            ("only_sel", "選択範囲のみ", "bool", o.id in self.viewer.selection, None),
        ], "注意: スムージングは形状を変化させます。測定・フィットには元データの使用を推奨します。")
        if not v:
            return
        vm = None
        if v["only_sel"] and o.id in self.viewer.selection:
            from ..core.selection import vertices_from_face_mask
            vm = vertices_from_face_mask(o.F, self.viewer.selection[o.id], len(o.V))

        def work(progress, V, F):
            return mesh_ops.smooth_taubin(V, F, v["iterations"], v["lamb"], -0.53, vm)

        def done(V2):
            self.doc.modify(o.id, lambda ob: ob.set_geometry(V2, ob.F, ob.face_flags, ob.colors), "スムージング")
        self.run("スムージング", work, o.V, o.F, on_done=done)

    def cmd_mesh_decimate(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return
        v = ParamDialog.ask(self, "メッシュ簡略化", [
            ("red", "削減率", "float", 50.0, {"min": 1, "max": 99, "suffix": "%"}),
        ], f"現在の面数: {len(o.F):,}。人工面フラグは失われます。")
        if not v:
            return

        def work(progress, V, F):
            return mesh_ops.decimate(V, F, v["red"] / 100.0)

        def done(res):
            V, F, fl = res
            self.doc.modify(o.id, lambda ob: ob.set_geometry(V, F, fl, None), "簡略化")
            self.set_guide(f"簡略化: {len(F):,} 面になりました")
        self.run("簡略化", work, o.V, o.F, on_done=done)

    def cmd_mesh_fix_normals(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return

        def work(progress, V, F):
            return mesh_ops.fix_normals(V, F)

        def done(F2):
            self.doc.modify(o.id, lambda ob: ob.set_geometry(ob.V, F2, ob.face_flags, ob.colors), "法線修正")
        self.run("法線修正", work, o.V, o.F, on_done=done)

    def cmd_mesh_flip(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return
        m = self.viewer.selection.get(o.id)
        self.doc.modify(o.id, lambda ob: ob.set_geometry(ob.V, mesh_ops.flip_faces(ob.F, m), ob.face_flags, ob.colors),
                        "法線反転")

    def cmd_mesh_fill(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return
        size = float(np.linalg.norm(o.V.max(0) - o.V.min(0))) if len(o.V) else 10.0
        v = ParamDialog.ask(self, "穴埋め", [
            ("size", "埋める穴の最大サイズ（外接球半径）", "float", round(0.05 * size, 3), {"min": 0.001, "max": 1e6, "suffix": "mm"}),
        ], f"境界エッジ数: {mesh_ops.boundary_loops(o.F):,}。\n生成された面は『人工面』として記録され、フィットから除外できます（実測面と区別）。")
        if not v:
            return

        def work(progress, V, F, fl, c):
            return mesh_ops.fill_holes(V, F, v["size"], fl, c)

        def done(res):
            V, F, fl, c, n = res
            if n == 0:
                self.info("穴埋め", "埋められる穴はありませんでした。サイズを大きくしてください。")
                return
            self.doc.modify(o.id, lambda ob: ob.set_geometry(V, F, fl, c), "穴埋め")
            self.set_guide(f"穴埋め: {n:,} 面（人工面）を追加しました")
            log.info("%s: 穴埋めで人工面 %d 面を追加", o.name, n)
        self.run("穴埋め", work, o.V, o.F, o.face_flags, o.colors, on_done=done)

    def cmd_mesh_split(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return

        def work(progress, V, F, fl, c):
            return mesh_ops.split_components(V, F, fl, c)

        def done(parts):
            if len(parts) <= 1:
                self.info("分割", "連結成分は1つだけです。")
                return
            objs = []
            for i, (V, F, fl, c) in enumerate(parts, 1):
                m = MeshObject(self.doc.unique_name(f"{o.name}_{i}"), V, F, c)
                m.face_flags = fl
                m.source_path, m.source_unit, m.unit_scale = o.source_path, o.source_unit, o.unit_scale
                m.meta = dict(o.meta)
                objs.append(m)
            self.doc.add(objs, "連結成分で分割")
            self.doc.set_visible(o.id, False)
            self.set_guide(f"{len(objs)} 個に分割しました（元のメッシュは非表示）")
        self.run("連結成分で分割", work, o.V, o.F, o.face_flags, o.colors, on_done=done)

    def cmd_mesh_separate(self):
        items = self._require_selection(("mesh",))
        for o, m in items:
            V, F, fl, c = mesh_ops.keep_faces(o.V, o.F, m, o.face_flags, o.colors)
            new = MeshObject(self.doc.unique_name(o.name + "_分離"), V, F, c)
            new.face_flags = fl
            new.source_path, new.source_unit, new.unit_scale = o.source_path, o.source_unit, o.unit_scale
            new.meta = dict(o.meta)
            self.doc.add(new, "選択領域を分離")
            self._mesh_edit(o, "選択領域を分離", lambda ob, m=m: mesh_ops.remove_faces(ob.V, ob.F, m, ob.face_flags, ob.colors))
        self.viewer.clear_selection()

    def cmd_mesh_merge(self):
        objs = self.selected_objects("mesh")
        if len(objs) < 2:
            self.warn("メッシュ結合", "モデルツリーで2つ以上のメッシュを選択してください（Ctrl+クリック）。")
            return
        V, F, fl, c = mesh_ops.merge_meshes([(o.V, o.F, o.face_flags, o.colors) for o in objs])
        m = MeshObject(self.doc.unique_name("結合メッシュ"), V, F, c)
        m.face_flags = fl
        m.meta = {"merged_from": [o.name for o in objs]}
        self.doc.add(m, "メッシュ結合")
        for o in objs:
            self.doc.set_visible(o.id, False)

    def cmd_mesh_restore(self):
        o = self.target_scan()
        if o is None:
            return
        if self.confirm("オリジナルに戻す", f"{o.name} の編集を破棄し、読込時のデータに戻しますか？\n（位置合わせの移動は保持されます。元に戻す で取り消し可能）"):
            self.doc.modify(o.id, lambda ob: ob.restore_original(), "オリジナルに戻す")

    # ---- 点群 -------------------------------------------------------------
    def cmd_pc_outliers(self):
        o = self.target_scan(("points",))
        if o is None:
            return
        v = ParamDialog.ask(self, "外れ値除去（統計的）", [
            ("k", "近傍点数", "int", 16, {"min": 3, "max": 200}),
            ("std", "標準偏差倍率", "float", 2.0, {"min": 0.1, "max": 10}),
        ])
        if not v:
            return

        def work(progress, P):
            return mesh_ops.statistical_outlier_removal(P, v["k"], v["std"])

        def done(keep):
            n = int((~keep).sum())
            self.doc.modify(o.id, lambda ob: ob.set_points(
                ob.P[keep], None if ob.normals is None else ob.normals[keep],
                None if ob.colors is None else ob.colors[keep]), "外れ値除去")
            self.set_guide(f"外れ値除去: {n:,} 点を削除しました")
        self.run("外れ値除去", work, o.P, on_done=done)

    def cmd_pc_normals(self):
        o = self.target_scan(("points",))
        if o is None:
            return
        v = ParamDialog.ask(self, "法線推定", [("k", "近傍点数", "int", 20, {"min": 5, "max": 200})],
                            "法線の向きは点群の重心から外向きに揃えます。")
        if not v:
            return

        def work(progress, P):
            return mesh_ops.estimate_normals(P, v["k"], progress=progress)

        def done(N):
            self.doc.modify(o.id, lambda ob: ob.set_points(ob.P, N, ob.colors), "法線推定")
        self.run("法線推定", work, o.P, on_done=done)

    def cmd_pc_poisson(self):
        o = self.target_scan(("points",))
        if o is None:
            return
        if o.normals is None:
            self.warn("法線がありません", "先に『点群: 法線推定』を実行してください。")
            return
        v = ParamDialog.ask(self, "Poisson メッシュ化", [("depth", "オクツリー深さ", "int", 9, {"min": 5, "max": 12})],
                            "注意: 再構成された面は実測点そのものではありません。")
        if not v:
            return

        def work(progress, P, N):
            return mesh_ops.poisson_reconstruction(P, N, v["depth"])

        def done(res):
            V, F = res
            m = MeshObject(self.doc.unique_name(o.name + "_mesh"), V, F)
            m.meta = {"generated": "poisson", "source_points": o.name}
            self.add_object(m, "Poisson メッシュ化")
        self.run("Poisson メッシュ化", work, o.P, o.normals, on_done=done)

    def cmd_mesh_to_points(self):
        o = self.target_scan(("mesh",))
        if o is None:
            return
        pc = PointCloudObject(self.doc.unique_name(o.name + "_点群"), o.V.copy(), o.vertex_normals.copy(),
                              None if o.colors is None else o.colors.copy())
        pc.source_path = o.source_path
        self.add_object(pc, "メッシュから点群")

    def cmd_rescale(self):
        o = self.target_scan()
        if o is None:
            return
        v = ParamDialog.ask(self, "単位/スケール変更", [
            ("factor", "倍率", "float", 1.0, {"min": 1e-6, "max": 1e6, "decimals": 6}),
            ("all", "すべてのオブジェクトに適用", "bool", False, None),
        ], "例: cm で読み込んだデータを mm にするには 10。原点を中心に拡大縮小します。")
        if not v or v["factor"] == 1.0:
            return
        ids = self.doc.ids() if v["all"] else [o.id]
        self.doc.scale(ids, v["factor"], (0, 0, 0), f"スケール ×{v['factor']:g}")
