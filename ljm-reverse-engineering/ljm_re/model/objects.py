"""シーンオブジェクト（メッシュ・点群・幾何要素・CAD形状・測定）.

設計方針:
- 幾何要素の座標はすべてワールド座標 [mm] で保持する。表示/編集時に作業座標系へ変換する。
- メッシュ編集は新しい配列を生成し、既存配列を書き換えない（Undo のスナップショットを軽量化）。
- 元のスキャンデータ (original_V/original_F) は常に保持し、復元可能にする（非破壊）。
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.coords import Frame
from ..core.geometry import (apply_rotation, apply_transform, normalize, orthonormal_basis)

KIND_LABELS = {
    "mesh": "メッシュ", "points": "点群", "cs": "座標系", "point": "基準点", "axis": "基準軸",
    "plane": "平面", "cylinder": "円筒", "hole": "穴/円", "sphere": "球", "section": "断面曲線",
    "cad": "CAD形状", "measurement": "測定",
}

DEFAULT_COLORS = {
    "mesh": (0.72, 0.74, 0.78), "points": (0.55, 0.75, 0.95), "cs": (1.0, 1.0, 1.0),
    "point": (1.0, 0.85, 0.2), "axis": (1.0, 0.45, 0.2), "plane": (0.35, 0.65, 1.0),
    "cylinder": (0.3, 0.9, 0.6), "hole": (1.0, 0.3, 0.6), "sphere": (0.8, 0.5, 1.0),
    "section": (1.0, 0.9, 0.3), "cad": (0.45, 0.8, 0.95), "measurement": (0.95, 0.95, 0.5),
}

POINT_CATEGORIES = ["", "車体取付穴", "アームピボット", "デフマウント", "エンジンマウント",
                    "排気ポート", "その他取付点"]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Prop:
    """プロパティパネルの1行."""
    key: str
    label: str
    value: Any
    kind: str = "str"        # 'float' | 'vec3' | 'str' | 'bool' | 'int' | 'choice' | 'info'
    editable: bool = False
    unit: str = ""
    choices: Optional[list] = None
    group: str = ""


def _tolist(a):
    return None if a is None else np.asarray(a).tolist()


class SceneObject:
    kind = "base"

    def __init__(self, name: str = "", oid: Optional[str] = None):
        self.id = oid or new_id()
        self.name = name or KIND_LABELS.get(self.kind, self.kind)
        self.visible = True
        self.color = DEFAULT_COLORS.get(self.kind, (0.8, 0.8, 0.8))
        self.opacity = 1.0
        self.critical = False      # 安全上重要（実測確認を促す）
        self.verified = False      # 実測確認済み
        self.verified_note = ""    # 実測値・確認者・日付など
        self.note = ""
        self.meta: dict = {}       # 生成条件・元ファイルなど
        self.revision = 0          # 変更カウンタ（表示更新用）

    # ---- 永続化 -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "id": self.id, "name": self.name, "visible": self.visible,
                "color": list(self.color), "opacity": self.opacity, "critical": self.critical,
                "verified": self.verified, "verified_note": self.verified_note,
                "note": self.note, "meta": self.meta, "data": self._data_dict()}

    def _data_dict(self) -> dict:
        return {}

    def _load_data(self, d: dict):
        pass

    def arrays(self) -> dict:
        """npz に保存する大きな配列."""
        return {}

    def set_arrays(self, arrs: dict):
        pass

    @classmethod
    def from_dict(cls, d: dict, arrays: Optional[dict] = None) -> "SceneObject":
        o = cls.__new__(cls)
        SceneObject.__init__(o, d.get("name", ""), d.get("id"))
        o._init_defaults()
        o.visible = d.get("visible", True)
        o.color = tuple(d.get("color", o.color))
        o.opacity = d.get("opacity", 1.0)
        o.critical = d.get("critical", False)
        o.verified = d.get("verified", False)
        o.verified_note = d.get("verified_note", "")
        o.note = d.get("note", "")
        o.meta = d.get("meta", {})
        o._load_data(d.get("data", {}))
        if arrays:
            o.set_arrays(arrays)
        return o

    def _init_defaults(self):
        pass

    # ---- Undo 用スナップショット -----------------------------------------
    def get_state(self) -> dict:
        st = {"name": self.name, "visible": self.visible, "color": self.color,
              "opacity": self.opacity, "critical": self.critical, "verified": self.verified,
              "verified_note": self.verified_note, "note": self.note,
              "meta": copy.deepcopy(self.meta), "data": copy.deepcopy(self._data_dict())}
        st["arrays"] = dict(self.arrays())  # 配列は不変として扱う（参照のみ保持）
        return st

    def set_state(self, st: dict):
        for k in ("name", "visible", "color", "opacity", "critical", "verified",
                  "verified_note", "note"):
            setattr(self, k, st[k])
        self.meta = copy.deepcopy(st["meta"])
        self._load_data(copy.deepcopy(st["data"]))
        self.set_arrays(st["arrays"])
        self.revision += 1

    # ---- 幾何操作 ---------------------------------------------------------
    def apply_transform(self, T: np.ndarray):
        """剛体変換をワールド座標に適用（位置合わせ・座標系アライン）."""

    def apply_scale(self, s: float, center=(0, 0, 0)):
        """スケール補正."""

    def key_points(self) -> list:
        """スナップ・座標出力用の代表点 [(ラベル, xyz)]."""
        return []

    def bounds_points(self) -> Optional[np.ndarray]:
        kp = self.key_points()
        return np.array([p for _, p in kp]) if kp else None

    def properties(self, frame: Frame, fmt) -> list:
        return []

    def set_property(self, key: str, value, frame: Frame):
        if key == "name":
            self.name = str(value)
        elif key == "critical":
            self.critical = bool(value)
        elif key == "verified":
            self.verified = bool(value)
        elif key == "verified_note":
            self.verified_note = str(value)
        elif key == "note":
            self.note = str(value)
        elif key == "opacity":
            self.opacity = float(min(1.0, max(0.0, value)))
        else:
            raise KeyError(key)
        self.revision += 1

    def common_props(self) -> list:
        return [
            Prop("name", "名前", self.name, "str", True, group="基本"),
            Prop("kind", "種類", KIND_LABELS.get(self.kind, self.kind), "info", group="基本"),
        ]

    def safety_props(self) -> list:
        return [
            Prop("critical", "安全上重要", self.critical, "bool", True, group="精度管理"),
            Prop("verified", "実測確認済み", self.verified, "bool", True, group="精度管理"),
            Prop("verified_note", "実測値・確認メモ", self.verified_note, "str", True, group="精度管理"),
            Prop("note", "備考", self.note, "str", True, group="精度管理"),
        ]


# ---------------------------------------------------------------------------
# プロパティ表示用ヘルパ
# ---------------------------------------------------------------------------

def frame_pos(frame: Frame, p):
    return frame.to_local(np.asarray(p, float)).tolist()


def frame_dir(frame: Frame, d):
    return frame.dir_to_local(np.asarray(d, float)).tolist()


def stats_props(stats: Optional[dict], extra: Optional[list] = None) -> list:
    if not stats:
        return []
    out = [
        Prop("st_n", "入力点数", stats.get("n_points"), "info", group="フィッティング結果"),
        Prop("st_used", "使用点数（インライア）", stats.get("n_used"), "info", group="フィッティング結果"),
        Prop("st_ratio", "インライア率", f"{stats.get('inlier_ratio', 0) * 100:.1f} %", "info",
             group="フィッティング結果"),
        Prop("st_rms", "RMS残差", stats.get("rms"), "float", unit="mm", group="フィッティング結果"),
        Prop("st_max", "最大残差", stats.get("max_abs"), "float", unit="mm", group="フィッティング結果"),
        Prop("st_method", "手法", stats.get("method"), "info", group="フィッティング条件"),
        Prop("st_thr", "インライア閾値" + ("（自動）" if stats.get("threshold_auto") else ""),
             stats.get("threshold"), "float", unit="mm", group="フィッティング条件"),
    ]
    if extra:
        out.extend(extra)
    return out


def source_props(meta: dict) -> list:
    out = []
    if meta.get("source_object_name"):
        out.append(Prop("src_obj", "元のオブジェクト", meta.get("source_object_name"), "info",
                        group="フィッティング条件"))
    if meta.get("source_file"):
        out.append(Prop("src_file", "元のスキャンファイル", meta.get("source_file"), "info",
                        group="フィッティング条件"))
    if meta.get("frame_name"):
        out.append(Prop("src_frame", "作成時の作業座標系", meta.get("frame_name"), "info",
                        group="フィッティング条件"))
    return out


# ---------------------------------------------------------------------------
# メッシュ・点群
# ---------------------------------------------------------------------------

class MeshObject(SceneObject):
    kind = "mesh"

    def __init__(self, name="", V=None, F=None, colors=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if V is not None:
            self.original_V = np.asarray(V, float)
            self.original_F = np.asarray(F, np.int64)
            self.V = self.original_V
            self.F = self.original_F
            self.face_flags = np.zeros(len(self.F), np.uint8)
            self.colors = None if colors is None else np.asarray(colors, np.uint8)
            self.original_colors = self.colors

    def _init_defaults(self):
        self.original_V = np.zeros((0, 3))
        self.original_F = np.zeros((0, 3), np.int64)
        self.original_colors = None
        self.V = self.original_V
        self.F = self.original_F
        self.face_flags = np.zeros(0, np.uint8)
        self.colors = None
        self.pose = np.eye(4)              # 読込後に適用された剛体変換の累積
        self.source_path = ""
        self.source_sha256 = ""
        self.source_unit = "mm"
        self.unit_scale = 1.0
        self.show_colors = False
        self._topo = None
        self._vnormals = None

    # ---- データ -----------------------------------------------------------
    def set_geometry(self, V, F, flags=None, colors=None):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, np.int64)
        self.face_flags = np.zeros(len(self.F), np.uint8) if flags is None else np.asarray(flags, np.uint8)
        self.colors = None if colors is None else np.asarray(colors, np.uint8)
        self.invalidate()

    def invalidate(self):
        self._topo = None
        self._vnormals = None
        self.revision += 1

    @property
    def topology(self):
        if self._topo is None:
            from ..core.selection import MeshTopology
            self._topo = MeshTopology(self.V, self.F)
        return self._topo

    @property
    def vertex_normals(self):
        if self._vnormals is None:
            from ..core.mesh_ops import vertex_normals
            self._vnormals = vertex_normals(self.V, self.F) if len(self.F) else np.zeros_like(self.V)
        return self._vnormals

    @property
    def artificial_mask(self):
        from ..core.mesh_ops import FLAG_ARTIFICIAL
        return (self.face_flags & FLAG_ARTIFICIAL) > 0

    def restore_original(self):
        V = apply_transform(self.original_V, self.pose)
        self.set_geometry(V, self.original_F, None, self.original_colors)

    def _data_dict(self):
        return {"pose": self.pose.tolist(), "source_path": self.source_path,
                "source_sha256": self.source_sha256, "source_unit": self.source_unit,
                "unit_scale": self.unit_scale, "show_colors": self.show_colors}

    def _load_data(self, d):
        self.pose = np.array(d.get("pose", np.eye(4).tolist()), float)
        self.source_path = d.get("source_path", "")
        self.source_sha256 = d.get("source_sha256", "")
        self.source_unit = d.get("source_unit", "mm")
        self.unit_scale = d.get("unit_scale", 1.0)
        self.show_colors = d.get("show_colors", False)

    def arrays(self):
        a = {"original_V": self.original_V, "original_F": self.original_F, "V": self.V,
             "F": self.F, "face_flags": self.face_flags}
        if self.colors is not None:
            a["colors"] = self.colors
        if self.original_colors is not None:
            a["original_colors"] = self.original_colors
        return a

    def set_arrays(self, a):
        self.original_V = a.get("original_V", self.original_V)
        self.original_F = a.get("original_F", self.original_F)
        self.original_colors = a.get("original_colors", None)
        self.V = a.get("V", self.V)
        self.F = a.get("F", self.F)
        self.face_flags = a.get("face_flags", np.zeros(len(self.F), np.uint8))
        self.colors = a.get("colors", None)
        self.invalidate()

    def apply_transform(self, T):
        self.V = apply_transform(self.V, T)
        self.pose = np.asarray(T, float) @ self.pose
        self.invalidate()

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.V = (self.V - c) * s + c
        S = np.eye(4)
        S[:3, :3] *= s
        S[:3, 3] = c - s * c
        self.pose = S @ self.pose
        self.invalidate()

    def key_points(self):
        if len(self.V) == 0:
            return []
        lo, hi = self.V.min(0), self.V.max(0)
        return [("min", lo), ("max", hi)]

    def properties(self, frame, fmt):
        lo = frame.to_local(self.V).min(0) if len(self.V) else np.zeros(3)
        hi = frame.to_local(self.V).max(0) if len(self.V) else np.zeros(3)
        n_art = int(self.artificial_mask.sum())
        props = self.common_props() + [
            Prop("n_vertices", "頂点数", f"{len(self.V):,}", "info", group="メッシュ"),
            Prop("n_faces", "面数", f"{len(self.F):,}", "info", group="メッシュ"),
            Prop("n_artificial", "人工面（穴埋め）", f"{n_art:,}", "info", group="メッシュ"),
            Prop("size", "外形寸法 (作業座標系)", (hi - lo).tolist(), "vec3", unit="mm", group="メッシュ"),
            Prop("bmin", "最小座標", lo.tolist(), "vec3", unit="mm", group="メッシュ"),
            Prop("bmax", "最大座標", hi.tolist(), "vec3", unit="mm", group="メッシュ"),
            Prop("opacity", "不透明度", self.opacity, "float", True, group="表示"),
            Prop("show_colors", "スキャンカラー表示", self.show_colors, "bool",
                 self.colors is not None, group="表示"),
            Prop("source_path", "元ファイル", self.source_path, "info", group="元データ"),
            Prop("source_unit", "元データの単位", self.source_unit, "info", group="元データ"),
            Prop("unit_scale", "mm への換算係数", self.unit_scale, "float", group="元データ"),
            Prop("source_sha256", "SHA-256", self.source_sha256[:16] + "…" if self.source_sha256 else "",
                 "info", group="元データ"),
        ]
        return props

    def set_property(self, key, value, frame):
        if key == "show_colors":
            self.show_colors = bool(value)
            self.revision += 1
        else:
            super().set_property(key, value, frame)


class PointCloudObject(SceneObject):
    kind = "points"

    def __init__(self, name="", P=None, normals=None, colors=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if P is not None:
            self.original_P = np.asarray(P, float)
            self.P = self.original_P
            self.normals = None if normals is None else np.asarray(normals, float)
            self.colors = None if colors is None else np.asarray(colors, np.uint8)
            self.original_colors = self.colors

    def _init_defaults(self):
        self.original_P = np.zeros((0, 3))
        self.P = self.original_P
        self.normals = None
        self.colors = None
        self.original_colors = None
        self.pose = np.eye(4)
        self.source_path = ""
        self.source_sha256 = ""
        self.source_unit = "mm"
        self.unit_scale = 1.0
        self.point_size = 3.0
        self.show_colors = False
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.P)
        return self._tree

    def set_points(self, P, normals=None, colors=None):
        self.P = np.asarray(P, float)
        self.normals = normals
        self.colors = colors
        self._tree = None
        self.revision += 1

    def _data_dict(self):
        return {"pose": self.pose.tolist(), "source_path": self.source_path,
                "source_sha256": self.source_sha256, "source_unit": self.source_unit,
                "unit_scale": self.unit_scale, "point_size": self.point_size,
                "show_colors": self.show_colors}

    def _load_data(self, d):
        self.pose = np.array(d.get("pose", np.eye(4).tolist()), float)
        self.source_path = d.get("source_path", "")
        self.source_sha256 = d.get("source_sha256", "")
        self.source_unit = d.get("source_unit", "mm")
        self.unit_scale = d.get("unit_scale", 1.0)
        self.point_size = d.get("point_size", 3.0)
        self.show_colors = d.get("show_colors", False)

    def arrays(self):
        a = {"original_P": self.original_P, "P": self.P}
        if self.normals is not None:
            a["normals"] = self.normals
        if self.colors is not None:
            a["colors"] = self.colors
        if self.original_colors is not None:
            a["original_colors"] = self.original_colors
        return a

    def set_arrays(self, a):
        self.original_P = a.get("original_P", self.original_P)
        self.P = a.get("P", self.P)
        self.normals = a.get("normals", None)
        self.colors = a.get("colors", None)
        self.original_colors = a.get("original_colors", None)
        self._tree = None
        self.revision += 1

    def restore_original(self):
        self.set_points(apply_transform(self.original_P, self.pose), None, self.original_colors)

    def apply_transform(self, T):
        self.P = apply_transform(self.P, T)
        if self.normals is not None:
            self.normals = apply_rotation(self.normals, T)
        self.pose = np.asarray(T, float) @ self.pose
        self._tree = None
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.P = (self.P - c) * s + c
        S = np.eye(4)
        S[:3, :3] *= s
        S[:3, 3] = c - s * c
        self.pose = S @ self.pose
        self._tree = None
        self.revision += 1

    def key_points(self):
        if len(self.P) == 0:
            return []
        return [("min", self.P.min(0)), ("max", self.P.max(0))]

    def properties(self, frame, fmt):
        L = frame.to_local(self.P) if len(self.P) else np.zeros((1, 3))
        return self.common_props() + [
            Prop("n_points", "点数", f"{len(self.P):,}", "info", group="点群"),
            Prop("has_normals", "法線", "あり" if self.normals is not None else "なし", "info", group="点群"),
            Prop("size", "外形寸法 (作業座標系)", (L.max(0) - L.min(0)).tolist(), "vec3", unit="mm", group="点群"),
            Prop("point_size", "表示点サイズ", self.point_size, "float", True, unit="px", group="表示"),
            Prop("source_path", "元ファイル", self.source_path, "info", group="元データ"),
            Prop("source_unit", "元データの単位", self.source_unit, "info", group="元データ"),
        ]

    def set_property(self, key, value, frame):
        if key == "point_size":
            self.point_size = float(max(1.0, min(20.0, value)))
            self.revision += 1
        else:
            super().set_property(key, value, frame)


# ---------------------------------------------------------------------------
# 幾何要素
# ---------------------------------------------------------------------------

class FeatureObject(SceneObject):
    """フィッティング等で作られる幾何要素の共通部分."""

    def _init_defaults(self):
        self.stats = None          # FitStats.to_dict()
        self.warnings = []

    def _base_data(self):
        return {"stats": self.stats, "warnings": self.warnings}

    def _load_base(self, d):
        self.stats = d.get("stats")
        self.warnings = d.get("warnings", [])

    def warning_props(self):
        return [Prop(f"warn{i}", "⚠ 警告", w, "info", group="警告") for i, w in enumerate(self.warnings)]


class CoordinateSystemObject(SceneObject):
    kind = "cs"

    def __init__(self, name="", frame: Optional[Frame] = None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if frame is not None:
            self.frame = frame.copy()

    def _init_defaults(self):
        self.frame = Frame()
        self.display_size = 50.0
        self.method = ""

    def _data_dict(self):
        return {"frame": self.frame.to_dict(), "display_size": self.display_size,
                "method": self.method}

    def _load_data(self, d):
        self.frame = Frame.from_dict(d["frame"]) if "frame" in d else Frame()
        self.display_size = d.get("display_size", 50.0)
        self.method = d.get("method", "")

    def apply_transform(self, T):
        M = np.asarray(T, float) @ self.frame.matrix()
        # スケールを含まない剛体変換を前提に正規直交化
        U, _, Vt = np.linalg.svd(M[:3, :3])
        self.frame = Frame(M[:3, 3], U @ Vt)
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.frame = Frame((self.frame.origin - c) * s + c, self.frame.R)
        self.revision += 1

    def key_points(self):
        return [("原点", self.frame.origin)]

    def properties(self, frame, fmt):
        # 座標系そのものはワールド基準で表示・編集する
        rx, ry, rz = self.frame.euler_xyz()
        return self.common_props() + [
            Prop("origin", "原点 (ワールド)", self.frame.origin.tolist(), "vec3", True, "mm", group="座標系"),
            Prop("euler", "回転 X→Y→Z (ワールド)", [rx, ry, rz], "vec3", True, "deg", group="座標系"),
            Prop("x_axis", "X軸方向", self.frame.x_axis.tolist(), "vec3", group="座標系"),
            Prop("y_axis", "Y軸方向", self.frame.y_axis.tolist(), "vec3", group="座標系"),
            Prop("z_axis", "Z軸方向", self.frame.z_axis.tolist(), "vec3", group="座標系"),
            Prop("method", "作成方法", self.method, "info", group="座標系"),
            Prop("display_size", "表示サイズ", self.display_size, "float", True, "mm", group="表示"),
        ]

    def set_property(self, key, value, frame):
        if key == "origin":
            self.frame = Frame(np.asarray(value, float), self.frame.R)
        elif key == "euler":
            self.frame = Frame.from_euler(self.frame.origin, *[float(v) for v in value])
        elif key == "display_size":
            self.display_size = max(1.0, float(value))
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


class PointFeature(FeatureObject):
    kind = "point"

    def __init__(self, name="", position=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if position is not None:
            self.position = np.asarray(position, float)

    def _init_defaults(self):
        super()._init_defaults()
        self.position = np.zeros(3)
        self.category = ""
        self.display_size = 2.0

    def _data_dict(self):
        d = self._base_data()
        d.update({"position": _tolist(self.position), "category": self.category,
                  "display_size": self.display_size})
        return d

    def _load_data(self, d):
        self._load_base(d)
        self.position = np.array(d.get("position", [0, 0, 0]), float)
        self.category = d.get("category", "")
        self.display_size = d.get("display_size", 2.0)

    def apply_transform(self, T):
        self.position = apply_transform(self.position, T)[0]
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.position = (self.position - c) * s + c
        self.revision += 1

    def key_points(self):
        return [("点", self.position)]

    def properties(self, frame, fmt):
        return self.common_props() + [
            Prop("position", "座標", frame_pos(frame, self.position), "vec3", True, "mm", group="基準点"),
            Prop("category", "分類", self.category, "choice", True, choices=POINT_CATEGORIES, group="基準点"),
        ] + stats_props(self.stats) + source_props(self.meta) + self.safety_props() + self.warning_props()

    def set_property(self, key, value, frame):
        if key == "position":
            self.position = frame.to_world(np.asarray(value, float))
        elif key == "category":
            self.category = str(value)
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


class AxisFeature(FeatureObject):
    kind = "axis"

    def __init__(self, name="", point=None, direction=None, length=100.0, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if point is not None:
            self.point = np.asarray(point, float)
            self.direction = normalize(np.asarray(direction, float))
            self.length = float(length)

    def _init_defaults(self):
        super()._init_defaults()
        self.point = np.zeros(3)
        self.direction = np.array([0, 0, 1.0])
        self.length = 100.0
        self.category = ""

    def _data_dict(self):
        d = self._base_data()
        d.update({"point": _tolist(self.point), "direction": _tolist(self.direction),
                  "length": self.length, "category": self.category})
        return d

    def _load_data(self, d):
        self._load_base(d)
        self.point = np.array(d.get("point", [0, 0, 0]), float)
        self.direction = np.array(d.get("direction", [0, 0, 1]), float)
        self.length = d.get("length", 100.0)
        self.category = d.get("category", "")

    def end_points(self):
        h = 0.5 * self.length * self.direction
        return self.point - h, self.point + h

    def apply_transform(self, T):
        self.point = apply_transform(self.point, T)[0]
        self.direction = normalize(apply_rotation(self.direction, T)[0])
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.point = (self.point - c) * s + c
        self.length *= s
        self.revision += 1

    def key_points(self):
        a, b = self.end_points()
        return [("中点", self.point), ("端点1", a), ("端点2", b)]

    def properties(self, frame, fmt):
        return self.common_props() + [
            Prop("point", "軸上の点（中点）", frame_pos(frame, self.point), "vec3", True, "mm", group="基準軸"),
            Prop("direction", "方向ベクトル", frame_dir(frame, self.direction), "vec3", True, group="基準軸"),
            Prop("length", "表示長さ", self.length, "float", True, "mm", group="基準軸"),
            Prop("category", "分類", self.category, "choice", True, choices=POINT_CATEGORIES, group="基準軸"),
        ] + stats_props(self.stats) + source_props(self.meta) + self.safety_props() + self.warning_props()

    def set_property(self, key, value, frame):
        if key == "point":
            self.point = frame.to_world(np.asarray(value, float))
        elif key == "direction":
            self.direction = normalize(frame.dir_to_world(np.asarray(value, float)))
        elif key == "length":
            self.length = max(1e-3, float(value))
        elif key == "category":
            self.category = str(value)
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


class PlaneFeature(FeatureObject):
    kind = "plane"

    def __init__(self, name="", origin=None, normal=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if origin is not None:
            self.origin = np.asarray(origin, float)
            self.normal = normalize(np.asarray(normal, float))
            self.u_axis, self.v_axis = orthonormal_basis(self.normal)

    def _init_defaults(self):
        super()._init_defaults()
        self.origin = np.zeros(3)
        self.normal = np.array([0, 0, 1.0])
        self.u_axis = np.array([1.0, 0, 0])
        self.v_axis = np.array([0, 1.0, 0])
        self.extent = np.array([[-25.0, -25.0], [25.0, 25.0]])  # u,v の範囲
        self.flatness = None
        self.sigma_normal_deg = None
        self.opacity = 0.45

    @staticmethod
    def from_fit(fit, name="平面") -> "PlaneFeature":
        p = PlaneFeature(name, fit.origin, fit.normal)
        p.u_axis, p.v_axis = fit.u_axis, fit.v_axis
        p.extent = np.asarray(fit.extent_uv, float)
        p.stats = fit.stats.to_dict()
        p.flatness = fit.flatness
        p.sigma_normal_deg = fit.sigma_normal_deg
        p.warnings = list(fit.warnings)
        return p

    def _data_dict(self):
        d = self._base_data()
        d.update({"origin": _tolist(self.origin), "normal": _tolist(self.normal),
                  "u_axis": _tolist(self.u_axis), "v_axis": _tolist(self.v_axis),
                  "extent": _tolist(self.extent), "flatness": self.flatness,
                  "sigma_normal_deg": self.sigma_normal_deg})
        return d

    def _load_data(self, d):
        self._load_base(d)
        self.origin = np.array(d.get("origin", [0, 0, 0]), float)
        self.normal = np.array(d.get("normal", [0, 0, 1]), float)
        self.u_axis = np.array(d.get("u_axis", [1, 0, 0]), float)
        self.v_axis = np.array(d.get("v_axis", [0, 1, 0]), float)
        self.extent = np.array(d.get("extent", [[-25, -25], [25, 25]]), float)
        self.flatness = d.get("flatness")
        self.sigma_normal_deg = d.get("sigma_normal_deg")

    def corners(self):
        (u0, v0), (u1, v1) = self.extent
        o, u, v = self.origin, self.u_axis, self.v_axis
        return np.array([o + u0 * u + v0 * v, o + u1 * u + v0 * v, o + u1 * u + v1 * v, o + u0 * u + v1 * v])

    def center(self):
        (u0, v0), (u1, v1) = self.extent
        return self.origin + 0.5 * (u0 + u1) * self.u_axis + 0.5 * (v0 + v1) * self.v_axis

    def apply_transform(self, T):
        self.origin = apply_transform(self.origin, T)[0]
        R = apply_rotation(np.vstack([self.normal, self.u_axis, self.v_axis]), T)
        self.normal, self.u_axis, self.v_axis = (normalize(r) for r in R)
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.origin = (self.origin - c) * s + c
        self.extent = self.extent * s
        self.revision += 1

    def key_points(self):
        return [("中心", self.center())]

    def properties(self, frame, fmt):
        size = self.extent[1] - self.extent[0]
        extra = []
        if self.flatness is not None:
            extra.append(Prop("flatness", "平面度（インライア幅）", self.flatness, "float", unit="mm",
                              group="フィッティング結果"))
        if self.sigma_normal_deg is not None:
            extra.append(Prop("sigma_n", "法線の推定不確かさ (1σ)", self.sigma_normal_deg, "float",
                              unit="deg", group="フィッティング結果"))
        return self.common_props() + [
            Prop("origin", "平面上の点（重心）", frame_pos(frame, self.origin), "vec3", True, "mm", group="平面"),
            Prop("normal", "法線ベクトル", frame_dir(frame, self.normal), "vec3", True, group="平面"),
            Prop("center", "表示範囲の中心", frame_pos(frame, self.center()), "vec3", unit="mm", group="平面"),
            Prop("size", "表示範囲 (U×V)", [float(size[0]), float(size[1]), 0.0], "vec3", unit="mm", group="平面"),
            Prop("opacity", "不透明度", self.opacity, "float", True, group="表示"),
        ] + stats_props(self.stats, extra) + source_props(self.meta) + self.safety_props() + self.warning_props()

    def set_property(self, key, value, frame):
        if key == "origin":
            self.origin = frame.to_world(np.asarray(value, float))
        elif key == "normal":
            n = normalize(frame.dir_to_world(np.asarray(value, float)))
            from ..core.geometry import rotation_between
            R = rotation_between(self.normal, n)
            self.normal = n
            self.u_axis = normalize(R @ self.u_axis)
            self.v_axis = np.cross(n, self.u_axis)
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


class CylinderFeature(FeatureObject):
    kind = "cylinder"

    def __init__(self, name="", axis_point=None, axis_dir=None, radius=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if axis_point is not None:
            self.axis_point = np.asarray(axis_point, float)
            self.axis_dir = normalize(np.asarray(axis_dir, float))
            self.radius = float(radius)

    def _init_defaults(self):
        super()._init_defaults()
        self.axis_point = np.zeros(3)
        self.axis_dir = np.array([0, 0, 1.0])
        self.radius = 10.0
        self.t_range = (-10.0, 10.0)
        self.coverage_deg = 360.0
        self.sigma_radius = None
        self.sigma_axis_deg = None
        self.sigma_axis_pos = None
        self.arc_ref = None          # スキャン円弧の角度基準方向
        self.arc_start_deg = 0.0
        self.opacity = 0.5

    @staticmethod
    def from_fit(fit, name="円筒") -> "CylinderFeature":
        c = CylinderFeature(name, fit.axis_point, fit.axis_dir, fit.radius)
        c.t_range = tuple(fit.t_range)
        c.coverage_deg = fit.coverage_deg
        c.sigma_radius = fit.sigma_radius
        c.sigma_axis_deg = fit.sigma_axis_deg
        c.sigma_axis_pos = fit.sigma_axis_pos
        c.arc_ref = None if fit.arc_ref is None else np.asarray(fit.arc_ref, float)
        c.arc_start_deg = float(fit.arc_start_deg)
        c.stats = fit.stats.to_dict()
        c.warnings = list(fit.warnings)
        return c

    @property
    def diameter(self):
        return 2 * self.radius

    @property
    def length(self):
        return float(self.t_range[1] - self.t_range[0])

    def end_points(self):
        return (self.axis_point + self.t_range[0] * self.axis_dir,
                self.axis_point + self.t_range[1] * self.axis_dir)

    def _data_dict(self):
        d = self._base_data()
        d.update({"axis_point": _tolist(self.axis_point), "axis_dir": _tolist(self.axis_dir),
                  "radius": self.radius, "t_range": list(self.t_range),
                  "coverage_deg": self.coverage_deg, "sigma_radius": self.sigma_radius,
                  "sigma_axis_deg": self.sigma_axis_deg, "sigma_axis_pos": self.sigma_axis_pos,
                  "arc_ref": _tolist(self.arc_ref), "arc_start_deg": self.arc_start_deg})
        return d

    def _load_data(self, d):
        self._load_base(d)
        self.axis_point = np.array(d.get("axis_point", [0, 0, 0]), float)
        self.axis_dir = np.array(d.get("axis_dir", [0, 0, 1]), float)
        self.radius = d.get("radius", 10.0)
        self.t_range = tuple(d.get("t_range", (-10, 10)))
        self.coverage_deg = d.get("coverage_deg", 360.0)
        self.sigma_radius = d.get("sigma_radius")
        self.sigma_axis_deg = d.get("sigma_axis_deg")
        self.sigma_axis_pos = d.get("sigma_axis_pos")
        self.arc_ref = None if d.get("arc_ref") is None else np.array(d["arc_ref"], float)
        self.arc_start_deg = d.get("arc_start_deg", 0.0)

    def apply_transform(self, T):
        self.axis_point = apply_transform(self.axis_point, T)[0]
        self.axis_dir = normalize(apply_rotation(self.axis_dir, T)[0])
        if self.arc_ref is not None:
            self.arc_ref = normalize(apply_rotation(self.arc_ref, T)[0])
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.axis_point = (self.axis_point - c) * s + c
        self.radius *= s
        self.t_range = (self.t_range[0] * s, self.t_range[1] * s)
        self.revision += 1

    def key_points(self):
        a, b = self.end_points()
        return [("軸中点", 0.5 * (a + b)), ("軸端1", a), ("軸端2", b)]

    def properties(self, frame, fmt):
        a, b = self.end_points()
        extra = [Prop("coverage", "円周方向カバー範囲", self.coverage_deg, "float", unit="deg",
                      group="フィッティング結果")]
        if self.sigma_radius is not None:
            extra += [
                Prop("sig_d", "直径の推定不確かさ (1σ)", 2 * self.sigma_radius, "float", unit="mm",
                     group="フィッティング結果"),
                Prop("sig_a", "軸方向の推定不確かさ (1σ)", self.sigma_axis_deg, "float", unit="deg",
                     group="フィッティング結果"),
                Prop("sig_p", "軸位置の推定不確かさ (1σ)", self.sigma_axis_pos, "float", unit="mm",
                     group="フィッティング結果"),
            ]
        return self.common_props() + [
            Prop("diameter", "直径", self.diameter, "float", True, "mm", group="円筒"),
            Prop("radius", "半径", self.radius, "float", True, "mm", group="円筒"),
            Prop("axis_point", "軸上の点", frame_pos(frame, self.axis_point), "vec3", True, "mm", group="円筒"),
            Prop("axis_dir", "軸方向ベクトル", frame_dir(frame, self.axis_dir), "vec3", True, group="円筒"),
            Prop("end1", "軸端点1", frame_pos(frame, a), "vec3", unit="mm", group="円筒"),
            Prop("end2", "軸端点2", frame_pos(frame, b), "vec3", unit="mm", group="円筒"),
            Prop("length", "軸方向長さ", self.length, "float", unit="mm", group="円筒"),
            Prop("opacity", "不透明度", self.opacity, "float", True, group="表示"),
        ] + stats_props(self.stats, extra) + source_props(self.meta) + self.safety_props() + self.warning_props()

    def set_property(self, key, value, frame):
        if key == "diameter":
            self.radius = max(1e-6, float(value) / 2)
        elif key == "radius":
            self.radius = max(1e-6, float(value))
        elif key == "axis_point":
            self.axis_point = frame.to_world(np.asarray(value, float))
        elif key == "axis_dir":
            self.axis_dir = normalize(frame.dir_to_world(np.asarray(value, float)))
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


class HoleFeature(FeatureObject):
    """穴中心 / 円（円形エッジ）."""
    kind = "hole"

    def __init__(self, name="", center=None, axis=None, diameter=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if center is not None:
            self.center = np.asarray(center, float)
            self.axis = normalize(np.asarray(axis, float))
            self.diameter = float(diameter)

    def _init_defaults(self):
        super()._init_defaults()
        self.center = np.zeros(3)
        self.axis = np.array([0, 0, 1.0])
        self.diameter = 10.0
        self.method = ""
        self.depth = None
        self.coverage_deg = 360.0
        self.sigma_diameter = None
        self.sigma_center = None
        self.category = ""

    @staticmethod
    def from_result(h, name="穴") -> "HoleFeature":
        f = HoleFeature(name, h.center, h.axis, h.diameter)
        f.method = h.method
        f.depth = h.depth
        f.coverage_deg = h.coverage_deg
        f.sigma_diameter = h.sigma_diameter
        f.sigma_center = h.sigma_center
        f.stats = {"n_points": h.n_points, "n_used": h.n_points, "rms": h.rms, "max_abs": h.max_abs,
                   "mean_abs": None, "inlier_ratio": 1.0, "method": h.method, "threshold": None,
                   "iterations": None, "threshold_auto": False}
        f.warnings = list(h.warnings)
        return f

    def _data_dict(self):
        d = self._base_data()
        d.update({"center": _tolist(self.center), "axis": _tolist(self.axis),
                  "diameter": self.diameter, "method": self.method, "depth": self.depth,
                  "coverage_deg": self.coverage_deg, "sigma_diameter": self.sigma_diameter,
                  "sigma_center": self.sigma_center, "category": self.category})
        return d

    def _load_data(self, d):
        self._load_base(d)
        self.center = np.array(d.get("center", [0, 0, 0]), float)
        self.axis = np.array(d.get("axis", [0, 0, 1]), float)
        self.diameter = d.get("diameter", 10.0)
        self.method = d.get("method", "")
        self.depth = d.get("depth")
        self.coverage_deg = d.get("coverage_deg", 360.0)
        self.sigma_diameter = d.get("sigma_diameter")
        self.sigma_center = d.get("sigma_center")
        self.category = d.get("category", "")

    def apply_transform(self, T):
        self.center = apply_transform(self.center, T)[0]
        self.axis = normalize(apply_rotation(self.axis, T)[0])
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.center = (self.center - c) * s + c
        self.diameter *= s
        if self.depth:
            self.depth *= s
        self.revision += 1

    def key_points(self):
        return [("中心", self.center)]

    def properties(self, frame, fmt):
        extra = [Prop("coverage", "円周方向カバー範囲", self.coverage_deg, "float", unit="deg",
                      group="フィッティング結果")]
        if self.sigma_diameter is not None:
            extra += [Prop("sig_d", "直径の推定不確かさ (1σ)", self.sigma_diameter, "float", unit="mm",
                           group="フィッティング結果"),
                      Prop("sig_c", "中心の推定不確かさ (1σ)", self.sigma_center, "float", unit="mm",
                           group="フィッティング結果")]
        props = self.common_props() + [
            Prop("center", "穴中心", frame_pos(frame, self.center), "vec3", True, "mm", group="穴"),
            Prop("diameter", "穴径", self.diameter, "float", True, "mm", group="穴"),
            Prop("axis", "中心軸方向", frame_dir(frame, self.axis), "vec3", True, group="穴"),
            Prop("dist_origin", "作業座標系原点からの距離",
                 float(np.linalg.norm(frame.to_local(self.center))), "float", unit="mm", group="穴"),
            Prop("method", "算出方法", self.method, "info", group="穴"),
            Prop("category", "分類", self.category, "choice", True, choices=POINT_CATEGORIES, group="穴"),
        ]
        if self.depth:
            props.append(Prop("depth", "内壁の軸方向長さ", self.depth, "float", unit="mm", group="穴"))
        return props + stats_props(self.stats, extra) + source_props(self.meta) + self.safety_props() + self.warning_props()

    def set_property(self, key, value, frame):
        if key == "center":
            self.center = frame.to_world(np.asarray(value, float))
        elif key == "diameter":
            self.diameter = max(1e-6, float(value))
        elif key == "axis":
            self.axis = normalize(frame.dir_to_world(np.asarray(value, float)))
        elif key == "category":
            self.category = str(value)
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


class SphereFeature(FeatureObject):
    kind = "sphere"

    def __init__(self, name="", center=None, radius=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        if center is not None:
            self.center = np.asarray(center, float)
            self.radius = float(radius)

    def _init_defaults(self):
        super()._init_defaults()
        self.center = np.zeros(3)
        self.radius = 10.0
        self.sigma_radius = None
        self.sigma_center = None
        self.opacity = 0.5

    @staticmethod
    def from_fit(fit, name="球"):
        s = SphereFeature(name, fit.center, fit.radius)
        s.stats = fit.stats.to_dict()
        s.sigma_radius = fit.sigma_radius
        s.sigma_center = fit.sigma_center
        s.warnings = list(fit.warnings)
        return s

    def _data_dict(self):
        d = self._base_data()
        d.update({"center": _tolist(self.center), "radius": self.radius,
                  "sigma_radius": self.sigma_radius, "sigma_center": self.sigma_center})
        return d

    def _load_data(self, d):
        self._load_base(d)
        self.center = np.array(d.get("center", [0, 0, 0]), float)
        self.radius = d.get("radius", 10.0)
        self.sigma_radius = d.get("sigma_radius")
        self.sigma_center = d.get("sigma_center")

    def apply_transform(self, T):
        self.center = apply_transform(self.center, T)[0]
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.center = (self.center - c) * s + c
        self.radius *= s
        self.revision += 1

    def key_points(self):
        return [("中心", self.center)]

    def properties(self, frame, fmt):
        extra = []
        if self.sigma_radius is not None:
            extra.append(Prop("sig_d", "直径の推定不確かさ (1σ)", 2 * self.sigma_radius, "float", unit="mm",
                              group="フィッティング結果"))
        return self.common_props() + [
            Prop("center", "中心", frame_pos(frame, self.center), "vec3", True, "mm", group="球"),
            Prop("diameter", "直径", 2 * self.radius, "float", True, "mm", group="球"),
            Prop("opacity", "不透明度", self.opacity, "float", True, group="表示"),
        ] + stats_props(self.stats, extra) + source_props(self.meta) + self.safety_props() + self.warning_props()

    def set_property(self, key, value, frame):
        if key == "center":
            self.center = frame.to_world(np.asarray(value, float))
        elif key == "diameter":
            self.radius = max(1e-6, float(value) / 2)
        else:
            return super().set_property(key, value, frame)
        self.revision += 1


# ---------------------------------------------------------------------------
# 断面曲線（CADで使えるスケッチとして保持）
# ---------------------------------------------------------------------------

class SectionObject(SceneObject):
    kind = "section"

    def __init__(self, name="", oid=None):
        super().__init__(name, oid)
        self._init_defaults()

    def _init_defaults(self):
        self.origin = np.zeros(3)
        self.normal = np.array([0, 0, 1.0])
        self.u_axis = np.array([1.0, 0, 0])
        self.v_axis = np.array([0, 1.0, 0])
        self.polylines = []    # list of (k,3) arrays (world)
        self.closed = []       # list of bool
        self.sketch = []       # list of list[SketchEntity dict] (2D, 平面座標)
        self.source_mesh_id = ""
        self.sketch_tol = None

    @staticmethod
    def from_result(res, name="断面"):
        s = SectionObject(name)
        s.origin, s.normal, s.u_axis, s.v_axis = res.origin, res.normal, res.u_axis, res.v_axis
        s.polylines = [pl.points for pl in res.polylines]
        s.closed = [pl.closed for pl in res.polylines]
        s.sketch = [[] for _ in res.polylines]
        return s

    def to_2d(self, pts):
        rel = np.asarray(pts, float) - self.origin
        return np.column_stack([rel @ self.u_axis, rel @ self.v_axis])

    def to_3d(self, uv):
        uv = np.asarray(uv, float)
        return self.origin + np.outer(uv[:, 0], self.u_axis) + np.outer(uv[:, 1], self.v_axis)

    def sketch_entities(self, i):
        from ..core.sketch_fit import SketchEntity
        return [SketchEntity.from_dict(d) for d in self.sketch[i]] if i < len(self.sketch) else []

    def _data_dict(self):
        return {"origin": _tolist(self.origin), "normal": _tolist(self.normal),
                "u_axis": _tolist(self.u_axis), "v_axis": _tolist(self.v_axis),
                "closed": list(map(bool, self.closed)), "sketch": self.sketch,
                "source_mesh_id": self.source_mesh_id, "sketch_tol": self.sketch_tol,
                "n_polylines": len(self.polylines)}

    def _load_data(self, d):
        self.origin = np.array(d.get("origin", [0, 0, 0]), float)
        self.normal = np.array(d.get("normal", [0, 0, 1]), float)
        self.u_axis = np.array(d.get("u_axis", [1, 0, 0]), float)
        self.v_axis = np.array(d.get("v_axis", [0, 1, 0]), float)
        self.closed = list(d.get("closed", []))
        self.sketch = d.get("sketch", [[] for _ in self.closed])
        self.source_mesh_id = d.get("source_mesh_id", "")
        self.sketch_tol = d.get("sketch_tol")

    def arrays(self):
        return {f"pl{i}": p for i, p in enumerate(self.polylines)}

    def set_arrays(self, a):
        keys = sorted([k for k in a if k.startswith("pl")], key=lambda k: int(k[2:]))
        self.polylines = [np.asarray(a[k], float) for k in keys]
        self.revision += 1

    def apply_transform(self, T):
        self.origin = apply_transform(self.origin, T)[0]
        R = apply_rotation(np.vstack([self.normal, self.u_axis, self.v_axis]), T)
        self.normal, self.u_axis, self.v_axis = (normalize(r) for r in R)
        self.polylines = [apply_transform(p, T) for p in self.polylines]
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.origin = (self.origin - c) * s + c
        self.polylines = [(p - c) * s + c for p in self.polylines]
        # スケッチは平面座標なので相似拡大
        from ..core.sketch_fit import SketchEntity
        new = []
        for ents in self.sketch:
            ne = []
            for d in ents:
                e = SketchEntity.from_dict(d)
                for k in ("start", "end", "mid", "center", "ctrl", "fit_points"):
                    v = getattr(e, k)
                    if v is not None:
                        setattr(e, k, v * s)
                if e.radius is not None:
                    e.radius *= s
                if e.knots is not None and e.kind != "spline":
                    pass
                ne.append(e.to_dict())
            new.append(ne)
        self.sketch = new
        self.revision += 1

    def key_points(self):
        return [("断面原点", self.origin)]

    def bounds_points(self):
        return np.vstack(self.polylines) if self.polylines else None

    def properties(self, frame, fmt):
        n_ent = sum(len(s) for s in self.sketch)
        kinds = {}
        for ents in self.sketch:
            for d in ents:
                kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        total_len = 0.0
        for p, c in zip(self.polylines, self.closed):
            if len(p) > 1:
                total_len += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
                if c:
                    total_len += float(np.linalg.norm(p[0] - p[-1]))
        return self.common_props() + [
            Prop("origin", "断面平面の原点", frame_pos(frame, self.origin), "vec3", unit="mm", group="断面"),
            Prop("normal", "断面平面の法線", frame_dir(frame, self.normal), "vec3", group="断面"),
            Prop("n_loops", "曲線数", f"{len(self.polylines)}（閉 {sum(self.closed)} / 開 {len(self.closed) - sum(self.closed)}）",
                 "info", group="断面"),
            Prop("total_len", "曲線の総長", total_len, "float", unit="mm", group="断面"),
            Prop("n_ent", "スケッチ要素数", f"{n_ent} " + " ".join(f"{k}:{v}" for k, v in kinds.items()),
                 "info", group="スケッチ"),
            Prop("sketch_tol", "スケッチ化の許容誤差", self.sketch_tol, "float", unit="mm", group="スケッチ"),
        ]


# ---------------------------------------------------------------------------
# CAD形状
# ---------------------------------------------------------------------------

class CadObject(SceneObject):
    kind = "cad"

    def __init__(self, name="", shape=None, recipe=None, oid=None):
        super().__init__(name, oid)
        self._init_defaults()
        self.shape = shape
        self.recipe = recipe or {}

    def _init_defaults(self):
        self.shape = None          # OCC TopoDS_Shape
        self.recipe = {}           # 再生成用パラメータ
        self.summary = {}          # 面種別数・体積など
        self.deviation = None      # 偏差解析結果
        self.opacity = 0.85
        self._tess = None

    def invalidate(self):
        self._tess = None
        self.revision += 1

    def _data_dict(self):
        return {"recipe": self.recipe, "summary": self.summary, "deviation": self.deviation}

    def _load_data(self, d):
        self.recipe = d.get("recipe", {})
        self.summary = d.get("summary", {})
        self.deviation = d.get("deviation")

    def get_state(self):
        st = super().get_state()
        st["shape"] = self.shape
        return st

    def set_state(self, st):
        super().set_state(st)
        self.shape = st.get("shape", self.shape)
        self.invalidate()

    def apply_transform(self, T):
        if self.shape is not None:
            from ..cad import occ_backend as occ
            self.shape = occ.transform_shape(self.shape, T)
        self.invalidate()

    def apply_scale(self, s, center=(0, 0, 0)):
        if self.shape is not None:
            from ..cad import occ_backend as occ
            self.shape = occ.scale_shape(self.shape, s, center)
        self.invalidate()

    def key_points(self):
        b = self.summary.get("bbox")
        if b:
            return [("min", np.array(b[0])), ("max", np.array(b[1]))]
        return []

    def properties(self, frame, fmt):
        s = self.summary
        props = self.common_props() + [
            Prop("cad_type", "生成元", self.recipe.get("type", ""), "info", group="CAD"),
            Prop("valid", "形状の妥当性 (BRepCheck)", "有効" if s.get("valid") else "無効/未確認", "info", group="CAD"),
            Prop("n_solids", "ソリッド数", s.get("n_solids"), "info", group="CAD"),
            Prop("n_faces", "面数", s.get("n_faces"), "info", group="CAD"),
            Prop("face_types", "面の種類", s.get("face_types_text", ""), "info", group="CAD"),
            Prop("volume", "体積", s.get("volume"), "float", unit="mm³", group="CAD"),
            Prop("area", "表面積", s.get("area"), "float", unit="mm²", group="CAD"),
            Prop("opacity", "不透明度", self.opacity, "float", True, group="表示"),
        ]
        if self.deviation:
            d = self.deviation
            props += [
                Prop("dev_mesh", "偏差: 比較メッシュ", d.get("mesh_name", ""), "info", group="偏差解析"),
                Prop("dev_mean", "平均偏差（符号付き）", d.get("mean"), "float", unit="mm", group="偏差解析"),
                Prop("dev_rms", "RMS偏差", d.get("rms"), "float", unit="mm", group="偏差解析"),
                Prop("dev_max", "最大偏差（絶対値）", d.get("max_abs"), "float", unit="mm", group="偏差解析"),
                Prop("dev_in", f"許容 ±{d.get('tol', 0):g} mm 内の割合", f"{d.get('within_tol', 0) * 100:.1f} %",
                     "info", group="偏差解析"),
                Prop("dev_n", "評価点数", d.get("n"), "info", group="偏差解析"),
            ]
        return props


# ---------------------------------------------------------------------------
# 測定結果
# ---------------------------------------------------------------------------

MEASURE_LABELS = {
    "point_point": "2点間距離", "angle_3pt": "3点角度", "point_plane": "点と平面の距離",
    "plane_plane": "平面間", "axis_axis": "軸間", "point_axis": "点と軸の距離",
    "axis_plane": "軸と平面", "projected": "投影距離", "bbox": "バウンディングボックス",
    "coordinate": "座標", "diameter": "直径", "hole_hole": "穴中心間距離", "scale": "スケール確認",
}


class MeasurementObject(SceneObject):
    kind = "measurement"

    def __init__(self, name="", mtype="", result=None, anchors=None, refs=None, oid=None):
        super().__init__(name or MEASURE_LABELS.get(mtype, "測定"), oid)
        self._init_defaults()
        self.mtype = mtype
        self.result = result or {}
        self.anchors = [] if anchors is None else [np.asarray(a, float) for a in anchors]
        self.refs = refs or []

    def _init_defaults(self):
        self.mtype = ""
        self.result = {}
        self.anchors = []      # 寸法線表示用のワールド点
        self.refs = []         # 参照したオブジェクトID
        self.label = ""

    def _data_dict(self):
        return {"mtype": self.mtype, "result": self.result,
                "anchors": [a.tolist() for a in self.anchors], "refs": self.refs,
                "label": self.label}

    def _load_data(self, d):
        self.mtype = d.get("mtype", "")
        self.result = d.get("result", {})
        self.anchors = [np.array(a, float) for a in d.get("anchors", [])]
        self.refs = d.get("refs", [])
        self.label = d.get("label", "")

    def apply_transform(self, T):
        self.anchors = [apply_transform(a, T)[0] for a in self.anchors]
        self.revision += 1

    def apply_scale(self, s, center=(0, 0, 0)):
        c = np.asarray(center, float)
        self.anchors = [(a - c) * s + c for a in self.anchors]
        self.revision += 1

    def key_points(self):
        return [(f"P{i + 1}", a) for i, a in enumerate(self.anchors)]

    def main_value(self):
        """(値, 単位) の主表示値."""
        r = self.result
        t = self.mtype
        if t in ("angle_3pt",):
            return r.get("angle"), "°"
        if t == "plane_plane":
            return (r.get("distance"), "mm") if r.get("parallel") else (r.get("angle"), "°")
        if t == "axis_axis":
            return r.get("distance"), "mm"
        if t == "axis_plane":
            return r.get("angle_to_plane"), "°"
        if t == "bbox":
            return r.get("diagonal"), "mm"
        if t == "coordinate":
            return None, ""
        if t == "scale":
            return r.get("factor"), ""
        if t == "diameter":
            return r.get("diameter"), "mm"
        return r.get("distance"), "mm"

    def summary_text(self, fmt) -> str:
        r = self.result
        t = self.mtype
        L = fmt.length
        A = fmt.angle
        if t == "point_point" or t == "hole_hole":
            return f"{L(r['distance'])} mm (dX {L(r['dx'])}, dY {L(r['dy'])}, dZ {L(r['dz'])})"
        if t == "angle_3pt":
            return f"{A(r['angle'])}°"
        if t == "point_plane":
            return f"{L(r['distance'])} mm (符号付 {L(r['signed_distance'])})"
        if t == "plane_plane":
            s = f"角度 {A(r['angle'])}°"
            if r.get("parallel"):
                s += f" / 平行: 距離 {L(r['distance'])} mm"
            elif r.get("perpendicular"):
                s += " / 直角"
            return s
        if t == "axis_axis":
            s = f"距離 {L(r['distance'])} mm / 角度 {A(r['angle'])}°"
            if "coaxiality" in r:
                s += f" / 同軸度 φ{L(r['coaxiality'])}"
            return s
        if t == "point_axis":
            return f"{L(r['distance'])} mm"
        if t == "axis_plane":
            return f"軸-平面角度 {A(r['angle_to_plane'])}°"
        if t == "projected":
            return f"{L(r['distance'])} mm (投影方向) / 直交 {L(r['perpendicular'])} mm"
        if t == "bbox":
            s = r["size"]
            return f"{L(s[0])} × {L(s[1])} × {L(s[2])} mm"
        if t == "coordinate":
            return f"X {L(r['x'])}, Y {L(r['y'])}, Z {L(r['z'])}"
        if t == "diameter":
            return f"φ{L(r['diameter'])} mm (R {L(r['diameter'] / 2)})"
        if t == "scale":
            return f"係数 {r['factor']:.6f} (誤差 {L(r['error_mm'])} mm, {r['error_percent']:.3f} %)"
        return str(r)

    def short_label(self, fmt) -> str:
        """3D表示用のASCIIラベル（VTKの標準フォントは日本語非対応）."""
        r = self.result
        t = self.mtype
        L = fmt.length
        A = fmt.angle
        if t in ("point_point", "hole_hole", "point_plane", "point_axis", "projected"):
            return f"{L(r['distance'])} mm"
        if t == "angle_3pt":
            return f"{A(r['angle'])} deg"
        if t == "plane_plane":
            return f"{L(r['distance'])} mm" if r.get("parallel") else f"{A(r['angle'])} deg"
        if t == "axis_axis":
            return f"{L(r['distance'])} mm / {A(r['angle'])} deg"
        if t == "axis_plane":
            return f"{A(r['angle_to_plane'])} deg"
        if t == "coordinate":
            return f"({L(r['x'])}, {L(r['y'])}, {L(r['z'])})"
        if t == "diameter":
            return f"D{L(r['diameter'])}"
        if t == "bbox":
            s = r["size"]
            return f"{L(s[0])} x {L(s[1])} x {L(s[2])}"
        return ""

    def properties(self, frame, fmt):
        props = self.common_props() + [
            Prop("mtype", "測定種類", MEASURE_LABELS.get(self.mtype, self.mtype), "info", group="測定"),
            Prop("summary", "結果", self.summary_text(fmt), "info", group="測定"),
        ]
        r = self.result
        for k, lab, unit in (("distance", "距離", "mm"), ("angle", "角度", "deg"),
                             ("coaxiality", "同軸度", "mm"), ("signed_distance", "符号付距離", "mm"),
                             ("factor", "スケール係数", "")):
            if k in r and r[k] is not None:
                props.append(Prop("r_" + k, lab, r[k], "float", unit=unit, group="測定"))
        if "frame_name" in r:
            props.append(Prop("frame", "測定時の作業座標系", r["frame_name"], "info", group="測定"))
        for i, a in enumerate(self.anchors):
            props.append(Prop(f"a{i}", f"参照点 {i + 1}", frame_pos(frame, a), "vec3", unit="mm", group="参照点"))
        return props + self.safety_props()


OBJECT_CLASSES = {c.kind: c for c in (
    MeshObject, PointCloudObject, CoordinateSystemObject, PointFeature, AxisFeature, PlaneFeature,
    CylinderFeature, HoleFeature, SphereFeature, SectionObject, CadObject, MeasurementObject)}
