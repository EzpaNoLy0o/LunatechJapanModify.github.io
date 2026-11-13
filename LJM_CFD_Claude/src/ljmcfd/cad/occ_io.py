"""OpenCASCADE (OCP) による STEP / IGES 読み込みとテッセレーション。

このモジュールは OCP を import するため重い。必ずジョブ (子プロセス) 側から使い、
GUI プロセスからは import しないこと。

OpenCASCADE は STEP / IGES を読み込む際に長さを mm へ換算する。したがって
ここで扱う B-Rep 形状と三角形メッシュの座標単位は常に **mm** である。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Reader
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import (
    TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool, XCAFDoc_DocumentTool

import OCP.TopoDS as _topods

from ..core.jobs import JobContext, JobError

try:  # OCP 8 系
    from OCP.collections import Sequence_TDF_Label as _LabelSequence
except ImportError:  # OCP 7 系
    from OCP.TDF import TDF_LabelSequence as _LabelSequence  # type: ignore[attr-defined,no-redef]

# OCP 7 系では静的メソッドに ``_s`` が付く (TopoDS.Face_s)。8 系では名前空間関数。
_to_face = getattr(_topods.TopoDS, "Face_s", None) or _topods.TopoDS.Face

#: テッセレーション品質 → (線形偏差の対角長比, 角度偏差 [rad])
TESSELLATION_QUALITY: dict[str, tuple[float, float]] = {
    "coarse": (2.0e-3, 0.50),
    "standard": (5.0e-4, 0.35),
    "fine": (1.5e-4, 0.20),
}

_SURFACE_TYPE_NAMES = {
    GeomAbs_SurfaceType.GeomAbs_Plane: "plane",
    GeomAbs_SurfaceType.GeomAbs_Cylinder: "cylinder",
    GeomAbs_SurfaceType.GeomAbs_Cone: "cone",
    GeomAbs_SurfaceType.GeomAbs_Sphere: "sphere",
    GeomAbs_SurfaceType.GeomAbs_Torus: "torus",
    GeomAbs_SurfaceType.GeomAbs_BezierSurface: "bezier",
    GeomAbs_SurfaceType.GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_SurfaceType.GeomAbs_OffsetSurface: "offset",
    GeomAbs_SurfaceType.GeomAbs_OtherSurface: "other",
}


@dataclass
class OccPart:
    """読み込んだ 1 パーツ (位置適用済みの B-Rep, mm)。"""

    name: str
    shape: TopoDS_Shape
    color: tuple[float, float, float] | None = None


@dataclass
class Tessellation:
    points: np.ndarray      # (N, 3) float64, mm
    triangles: np.ndarray   # (M, 3) int64
    face_ids: np.ndarray    # (M,) int32  — 三角形が属する B-Rep 面の番号 (0 始まり)
    face_types: list[str] = field(default_factory=list)  # 面番号 → 曲面種別
    skipped_faces: int = 0  # 三角形化できなかった面の数


def silence_occ_messages() -> None:
    """OpenCASCADE の情報メッセージ (転送統計など) を抑制し、警告以上のみ出す。"""
    try:
        from OCP.Message import Message, Message_Gravity

        for printer in list(Message.DefaultMessenger_s().Printers()):
            printer.SetTraceLevel(Message_Gravity.Message_Warning)
    except Exception:  # noqa: BLE001 - 抑制できなくても機能に影響しない
        pass


# --------------------------------------------------------------------------- 形状ユーティリティ

def shape_bounds(shape: TopoDS_Shape) -> tuple[float, float, float, float, float, float] | None:
    """B-Rep の境界 (xmin, xmax, ymin, ymax, zmin, zmax)。空形状は ``None``。"""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    lo, hi = box.CornerMin(), box.CornerMax()
    return (lo.X(), hi.X(), lo.Y(), hi.Y(), lo.Z(), hi.Z())


def iter_subshapes(shape: TopoDS_Shape, kind, avoid=None):
    explorer = TopExp_Explorer(shape, kind) if avoid is None else TopExp_Explorer(shape, kind, avoid)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


def count_subshapes(shape: TopoDS_Shape, kind) -> int:
    return sum(1 for _ in iter_subshapes(shape, kind))


def make_compound(shapes) -> TopoDS_Compound:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shp in shapes:
        builder.Add(compound, shp)
    return compound


def write_brep(shape: TopoDS_Shape, path: str | Path) -> None:
    if not BRepTools.Write_s(shape, str(path)):
        raise JobError(f"BREP キャッシュを書き込めません: {path}")


def read_brep(path: str | Path) -> TopoDS_Shape:
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()) or shape.IsNull():
        raise JobError(f"BREP キャッシュを読み込めません: {path}")
    return shape


# --------------------------------------------------------------------------- STEP / IGES 読み込み

#: 書き出しソフトが自動で付ける、意味を持たない製品名
_PLACEHOLDER_NAME_PREFIXES = ("open cascade step translator", "=>")
_PLACEHOLDER_NAMES = {"", "none", "unnamed", "noname", "unknown"}


def _label_name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return ""
    name = str(attr.Get().ToExtString()).strip()
    lowered = name.lower()
    # 数字だけの名前は STEP のインスタンス番号 (NAUO id) であり、パーツ名ではない
    if lowered in _PLACEHOLDER_NAMES or lowered.startswith(_PLACEHOLDER_NAME_PREFIXES) or lowered.isdigit():
        return ""
    return name


def _label_color(color_tool, label: TDF_Label) -> tuple[float, float, float] | None:
    color = Quantity_Color()
    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        try:
            if XCAFDoc_ColorTool.GetColor_s(label, color_type, color):
                return (float(color.Red()), float(color.Green()), float(color.Blue()))
        except Exception:  # noqa: BLE001 - 色は必須情報ではない
            return None
    return None


def _split_multi_solid(part: OccPart) -> list[OccPart]:
    """アセンブリ構造を持たないマルチソリッド形状をソリッド単位へ分割する。"""
    if part.shape.ShapeType() not in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
        return [part]
    solids = list(iter_subshapes(part.shape, TopAbs_SOLID))
    if len(solids) < 2:
        return [part]
    result = [OccPart(f"{part.name}_solid{idx + 1}", solid, part.color) for idx, solid in enumerate(solids)]
    loose = list(iter_subshapes(part.shape, TopAbs_SHELL, TopAbs_SOLID))
    loose += list(iter_subshapes(part.shape, TopAbs_FACE, TopAbs_SHELL))
    if loose:
        result.append(OccPart(f"{part.name}_surfaces", make_compound(loose), part.color))
    return result


def _read_step_xcaf(path: str, ctx: JobContext) -> list[OccPart]:
    """XCAF でアセンブリ構造・名前・色を保ったまま STEP を読む。"""
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise JobError(f"STEP ファイルを解析できません: {path}")
    ctx.progress(0.25, "STEP 形状を変換中…")
    if not reader.Transfer(doc):
        raise JobError("STEP 形状の変換 (Transfer) に失敗しました")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    parts: list[OccPart] = []

    def walk(label: TDF_Label, location: TopLoc_Location, inherited_name: str, inherited_color) -> None:
        target = label
        if shape_tool.IsReference_s(label):
            target = TDF_Label()
            shape_tool.GetReferredShape_s(label, target)
            location = location.Multiplied(shape_tool.GetLocation_s(label))
        name = _label_name(target) or _label_name(label) or inherited_name
        color = _label_color(color_tool, label) or _label_color(color_tool, target) or inherited_color
        if shape_tool.IsAssembly_s(target):
            components = _LabelSequence()
            shape_tool.GetComponents_s(target, components, False)
            for i in range(1, components.Length() + 1):
                walk(components.Value(i), location, name, color)
            return
        shape = shape_tool.GetShape_s(target)
        if shape.IsNull():
            return
        parts.append(OccPart(name, shape.Moved(location), color))

    free = _LabelSequence()
    shape_tool.GetFreeShapes(free)
    for i in range(1, free.Length() + 1):
        walk(free.Value(i), TopLoc_Location(), "", None)
    return parts


def _read_step_plain(path: str) -> list[OccPart]:
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise JobError(f"STEP ファイルを解析できません: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        return []
    return [OccPart("", shape)]


def _finalize_parts(parts: list[OccPart], default_name: str, max_parts: int, ctx: JobContext) -> list[OccPart]:
    # 面を 1 枚も持たない形状 (曲線のみ等) は解析対象にならないので除外
    parts = [p for p in parts if count_subshapes(p.shape, TopAbs_FACE) > 0]
    if not parts:
        raise JobError("ファイルに面 (サーフェス) を持つ形状が含まれていません")
    expanded: list[OccPart] = []
    for part in parts:
        expanded.extend(_split_multi_solid(part))
    if len(expanded) > max_parts:
        ctx.log(
            f"パーツ数が {len(expanded)} 個と多いため、表示性能を保つために 1 パーツへ統合しました "
            f"(上限 {max_parts})"
        )
        expanded = [OccPart(default_name, make_compound([p.shape for p in expanded]))]
    # 名前の補完と重複解消
    seen: dict[str, int] = {}
    for part in expanded:
        base = part.name or default_name
        seen[base] = seen.get(base, 0) + 1
        part.name = base if seen[base] == 1 else f"{base} ({seen[base]})"
    return expanded


def read_step(path: str | Path, ctx: JobContext | None = None, max_parts: int = 300,
              default_name: str | None = None) -> list[OccPart]:
    """STEP を読み、パーツ (mm, 位置適用済み) のリストを返す。"""
    ctx = ctx or JobContext()
    path = str(path)
    try:
        parts = _read_step_xcaf(path, ctx)
    except JobError:
        raise
    except Exception as exc:  # noqa: BLE001 - XCAF で読めないファイルは素の reader で再試行
        ctx.log(f"XCAF での読み込みに失敗したため標準リーダーで再試行します ({type(exc).__name__}: {exc})")
        parts = []
    if not parts:
        parts = _read_step_plain(path)
    return _finalize_parts(parts, default_name or Path(path).stem, max_parts, ctx)


def read_iges(path: str | Path, ctx: JobContext | None = None, max_parts: int = 300,
              default_name: str | None = None) -> list[OccPart]:
    """IGES を読み、パーツ (mm) のリストを返す。IGES は開いたサーフェス集合であることが多い。"""
    ctx = ctx or JobContext()
    path = str(path)
    reader = IGESControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise JobError(f"IGES ファイルを解析できません: {path}")
    ctx.progress(0.25, "IGES 形状を変換中…")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise JobError("IGES ファイルに変換可能な形状がありません")
    return _finalize_parts([OccPart("", shape)], default_name or Path(path).stem, max_parts, ctx)


# --------------------------------------------------------------------------- テッセレーション

def _trsf_matrix(trsf) -> tuple[np.ndarray, np.ndarray]:
    rot = np.array([[trsf.Value(r, c) for c in (1, 2, 3)] for r in (1, 2, 3)], dtype=float)
    trans = np.array([trsf.Value(r, 4) for r in (1, 2, 3)], dtype=float)
    return rot, trans


def tessellate(shape: TopoDS_Shape, quality: str = "standard") -> Tessellation:
    """B-Rep を三角形メッシュ化する。三角形ごとに所属面番号を保持する。

    面ごとに独立した節点を持つ (面境界で節点を共有しない) ため、面の境界で
    法線が折れ、CAD らしいシェーディングになる。水密性の検査時は節点を
    マージしてから判定する (``ljmcfd.cad.inspect``)。
    """
    if quality not in TESSELLATION_QUALITY:
        raise JobError(f"未知のテッセレーション品質です: {quality}")
    rel_deflection, angular = TESSELLATION_QUALITY[quality]
    bounds = shape_bounds(shape)
    if bounds is None:
        raise JobError("形状が空のためメッシュ化できません")
    diag = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
    linear = max(diag * rel_deflection, 1.0e-6)
    BRepMesh_IncrementalMesh(shape, linear, False, angular, True)

    point_blocks: list[np.ndarray] = []
    tri_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    face_types: list[str] = []
    offset = 0
    skipped = 0
    for face_index, raw in enumerate(iter_subshapes(shape, TopAbs_FACE)):
        face = _to_face(raw)
        try:
            face_types.append(_SURFACE_TYPE_NAMES.get(BRepAdaptor_Surface(face, False).GetType(), "other"))
        except Exception:  # noqa: BLE001 - 種別が取れなくても三角形化は続行
            face_types.append("other")
        location = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, location)
        if tri is None or tri.NbTriangles() == 0:
            skipped += 1
            continue
        n_nodes = tri.NbNodes()
        pts = np.empty((n_nodes, 3), dtype=float)
        for i in range(1, n_nodes + 1):
            p = tri.Node(i)
            pts[i - 1] = (p.X(), p.Y(), p.Z())
        if not location.IsIdentity():
            rot, trans = _trsf_matrix(location.Transformation())
            pts = pts @ rot.T + trans
        n_tris = tri.NbTriangles()
        tris = np.array([tri.Triangle(i).Get() for i in range(1, n_tris + 1)], dtype=np.int64).reshape(-1, 3)
        tris += offset - 1
        if face.Orientation() == TopAbs_REVERSED:
            tris = tris[:, ::-1]
        point_blocks.append(pts)
        tri_blocks.append(tris)
        id_blocks.append(np.full(n_tris, face_index, dtype=np.int32))
        offset += n_nodes

    if not tri_blocks:
        raise JobError("どの面も三角形化できませんでした (形状が壊れている可能性があります)")
    return Tessellation(
        points=np.vstack(point_blocks),
        triangles=np.vstack(tri_blocks),
        face_ids=np.concatenate(id_blocks),
        face_types=face_types,
        skipped_faces=skipped,
    )
