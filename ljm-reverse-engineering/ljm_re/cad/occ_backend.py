"""OpenCASCADE (OCP) による B-Rep 生成・STEP 入出力・検証.

重要: STL 全体を STEP 拡張子で保存するような「偽の CAD 変換」は行わない。
ここで出力するのは、フィッティング結果から再構築した解析曲面（平面・円筒・球）や
スケッチ輪郭からの押し出し/回転体など、実際の幾何形状のみである。
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sys
import tempfile
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_GTransform, BRepBuilderAPI_MakeEdge,
                                    BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire,
                                    BRepBuilderAPI_Transform)
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism,
                                 BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere)
    from OCP.BRepTools import BRepTools
    from OCP.Bnd import Bnd_Box
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.GCPnts import GCPnts_TangentialDeflection
    from OCP.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder,
                             GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_C2)
    from OCP.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_Interpolate
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCP.TopAbs import (TopAbs_COMPOUND, TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED,
                            TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE)
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
    from OCP.gp import (gp_Ax1, gp_Ax2, gp_Ax3, gp_Circ, gp_Cylinder, gp_Dir, gp_GTrsf, gp_Pln,
                        gp_Pnt, gp_Trsf, gp_Vec)
    _AVAILABLE = True
    _IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover - OCP 未インストール環境
    _AVAILABLE = False
    _IMPORT_ERROR = str(_e)


def available() -> bool:
    return _AVAILABLE


def import_error() -> str:
    return _IMPORT_ERROR


def _require():
    if not _AVAILABLE:
        raise RuntimeError("CADカーネル (OCP / OpenCASCADE) が利用できません: " + _IMPORT_ERROR)


def _quiet_occ():
    """OpenCASCADE の標準出力メッセージを抑制する."""
    try:
        from OCP.Message import Message
        msgr = Message.DefaultMessenger_s()
        printers = msgr.Printers()
        for i in range(printers.Length(), 0, -1):
            msgr.RemovePrinter(printers.Value(i))
    except Exception:
        pass


if _AVAILABLE:
    _quiet_occ()


# ---------------------------------------------------------------------------
# 互換ヘルパ（OCP 7.x は TopoDS.Face_s、8.x は TopoDS.Face）
# ---------------------------------------------------------------------------

def _arr1_pnt(n):
    """点配列（OCP 8: OCP.collections / OCP 7: TColgp）."""
    try:
        from OCP.collections import Array1_gp_Pnt
        return Array1_gp_Pnt(1, n)
    except ImportError:
        from OCP.TColgp import TColgp_Array1OfPnt
        return TColgp_Array1OfPnt(1, n)


def _harr1_pnt(n):
    try:
        from OCP.collections import HArray1_gp_Pnt
        return HArray1_gp_Pnt(1, n)
    except ImportError:
        from OCP.TColgp import TColgp_HArray1OfPnt
        return TColgp_HArray1OfPnt(1, n)


def _arr2_pnt(n, m):
    try:
        from OCP.collections import Array2_gp_Pnt
        return Array2_gp_Pnt(1, n, 1, m)
    except ImportError:
        from OCP.TColgp import TColgp_Array2OfPnt
        return TColgp_Array2OfPnt(1, n, 1, m)


def _arr1_real(vals):
    try:
        from OCP.collections import Array1_double
        a = Array1_double(1, len(vals))
    except ImportError:
        from OCP.TColStd import TColStd_Array1OfReal
        a = TColStd_Array1OfReal(1, len(vals))
    for i, v in enumerate(vals, 1):
        a.SetValue(i, float(v))
    return a


def _arr1_int(vals):
    try:
        from OCP.collections import Array1_int
        a = Array1_int(1, len(vals))
    except ImportError:
        from OCP.TColStd import TColStd_Array1OfInteger
        a = TColStd_Array1OfInteger(1, len(vals))
    for i, v in enumerate(vals, 1):
        a.SetValue(i, int(v))
    return a


def _cast(kind: str, shape):
    f = getattr(TopoDS, kind, None) or getattr(TopoDS, kind + "_s")
    return f(shape)


def _pnt(p):
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    return gp_Dir(float(d[0]), float(d[1]), float(d[2]))


def _vec(v):
    return gp_Vec(float(v[0]), float(v[1]), float(v[2]))


def _np(p):
    return np.array([p.X(), p.Y(), p.Z()])


def _explore(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        yield ex.Current()
        ex.Next()


@contextlib.contextmanager
def occ_safe_path(path: str, mode: str = "w"):
    """OCC のネイティブ I/O に渡すパス.

    Windows で非ASCII（日本語）パスを OCC が扱えない場合に備え、ASCII の一時パスを経由する。
    mode='w': 一時ファイルに書かせ、終了後に目的のパスへ移動。
    mode='r': 目的のファイルを一時パスへコピーして読ませる。
    """
    path = os.fspath(path)
    if path.isascii() or sys.platform != "win32":
        yield path
        return
    base = tempfile.gettempdir()
    if not base.isascii():
        base = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "ProgramData", "LJM_RE", "tmp")
        os.makedirs(base, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="ljm_", dir=base)
    tmp = os.path.join(tmpdir, "occ_io" + os.path.splitext(path)[1])
    try:
        if mode == "r":
            shutil.copyfile(path, tmp)
        yield tmp
        if mode == "w" and os.path.exists(tmp):
            shutil.move(tmp, path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 形状生成
# ---------------------------------------------------------------------------

def make_vertex(p):
    _require()
    return BRepBuilderAPI_MakeVertex(_pnt(p)).Vertex()


def make_line_edge(p0, p1):
    _require()
    return BRepBuilderAPI_MakeEdge(_pnt(p0), _pnt(p1)).Edge()


def make_circle_edge(center, normal, radius):
    _require()
    circ = gp_Circ(gp_Ax2(_pnt(center), _dir(normal)), float(radius))
    return BRepBuilderAPI_MakeEdge(circ).Edge()


def make_polygon_face(points3d):
    """平面多角形（閉）から面."""
    _require()
    poly = BRepBuilderAPI_MakePolygon()
    for p in points3d:
        poly.Add(_pnt(p))
    poly.Close()
    mk = BRepBuilderAPI_MakeFace(poly.Wire(), True)
    if not mk.IsDone():
        raise RuntimeError("多角形面の生成に失敗しました")
    return mk.Face()


def make_plane_face(origin, normal, u_axis, extent):
    """有限平面（矩形）。extent=[[umin,vmin],[umax,vmax]]（u_axis基準）."""
    _require()
    ax3 = gp_Ax3(_pnt(origin), _dir(normal), _dir(u_axis))
    pln = gp_Pln(ax3)
    (u0, v0), (u1, v1) = np.asarray(extent, float)
    mk = BRepBuilderAPI_MakeFace(pln, float(u0), float(u1), float(v0), float(v1))
    if not mk.IsDone():
        raise RuntimeError("平面の生成に失敗しました")
    return mk.Face()


def make_cylinder_face(axis_point, axis_dir, radius, t0, t1, ang0=0.0, ang1=2 * np.pi, x_dir=None):
    """円筒面（部分円筒も可）。t0,t1 は axis_point 基準の軸方向範囲."""
    _require()
    a = np.asarray(axis_dir, float)
    base = np.asarray(axis_point, float) + float(t0) * a / np.linalg.norm(a)
    if x_dir is None:
        from ..core.geometry import orthonormal_basis
        x_dir, _ = orthonormal_basis(a)
    ax3 = gp_Ax3(_pnt(base), _dir(a), _dir(x_dir))
    cyl = gp_Cylinder(ax3, float(radius))
    mk = BRepBuilderAPI_MakeFace(cyl, float(ang0), float(ang1), 0.0, float(t1 - t0))
    if not mk.IsDone():
        raise RuntimeError("円筒面の生成に失敗しました")
    return mk.Face()


def make_cylinder_solid(axis_point, axis_dir, radius, t0, t1):
    _require()
    a = np.asarray(axis_dir, float)
    a = a / np.linalg.norm(a)
    base = np.asarray(axis_point, float) + float(t0) * a
    return BRepPrimAPI_MakeCylinder(gp_Ax2(_pnt(base), _dir(a)), float(radius), float(t1 - t0)).Shape()


def make_sphere_solid(center, radius):
    _require()
    return BRepPrimAPI_MakeSphere(_pnt(center), float(radius)).Shape()


def _uv_to_3d(frame, p2):
    o, u, v = frame
    return o + p2[0] * u + p2[1] * v


def _sketch_edges(entities, frame):
    """2Dスケッチ要素 → 3D エッジ列。frame=(origin, u_axis, v_axis)."""
    edges = []
    for e in entities:
        if e.kind == "line":
            p0, p1 = _uv_to_3d(frame, e.start), _uv_to_3d(frame, e.end)
            if np.linalg.norm(p1 - p0) < 1e-9:
                continue
            edges.append(BRepBuilderAPI_MakeEdge(GC_MakeSegment(_pnt(p0), _pnt(p1)).Value()).Edge())
        elif e.kind == "arc":
            p0, pm, p1 = (_uv_to_3d(frame, x) for x in (e.start, e.mid, e.end))
            arc = GC_MakeArcOfCircle(_pnt(p0), _pnt(pm), _pnt(p1))
            if not arc.IsDone():
                edges.append(BRepBuilderAPI_MakeEdge(_pnt(p0), _pnt(p1)).Edge())
            else:
                edges.append(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
        elif e.kind == "circle":
            o, u, v = frame
            n = np.cross(u, v)
            c = _uv_to_3d(frame, e.center)
            edges.append(BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(_pnt(c), _dir(n), _dir(u)), float(e.radius))).Edge())
        elif e.kind == "spline":
            pts = e.sample(max(32, 4 * len(e.ctrl)))
            if e.closed:
                pts = pts[:-1]
            P3 = np.array([_uv_to_3d(frame, p) for p in pts])
            edges.append(make_bspline_edge(P3, closed=e.closed))
        else:
            raise ValueError(f"未対応のスケッチ要素: {e.kind}")
    return edges


def make_bspline_edge(points3d, closed=False, tol: float = 1e-3):
    _require()
    P = np.asarray(points3d, float)
    if closed:
        arr = _harr1_pnt(len(P))
        for i, p in enumerate(P, 1):
            arr.SetValue(i, _pnt(p))
        it = GeomAPI_Interpolate(arr, True, 1e-6)
        it.Perform()
        if not it.IsDone():
            raise RuntimeError("閉じたスプラインの補間に失敗しました")
        return BRepBuilderAPI_MakeEdge(it.Curve()).Edge()
    arr = _arr1_pnt(len(P))
    for i, p in enumerate(P, 1):
        arr.SetValue(i, _pnt(p))
    approx = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, float(tol))
    if not approx.IsDone():
        raise RuntimeError("スプライン近似に失敗しました")
    return BRepBuilderAPI_MakeEdge(approx.Curve()).Edge()


def make_bspline_surface_face(poles, uknots, vknots, udeg: int, vdeg: int):
    """Bスプライン曲面の面。poles=(nu,nv,3), u/vknots は重複を含む完全なノット列."""
    _require()
    from OCP.Geom import Geom_BSplineSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace as _MF
    poles = np.asarray(poles, float)
    nu, nv = poles.shape[:2]
    arr = _arr2_pnt(nu, nv)
    for i in range(nu):
        for j in range(nv):
            arr.SetValue(i + 1, j + 1, _pnt(poles[i, j]))

    def km(k):
        k = np.asarray(k, float)
        uniq, mult = [], []
        for x in k:
            if uniq and abs(x - uniq[-1]) < 1e-12:
                mult[-1] += 1
            else:
                uniq.append(x)
                mult.append(1)
        return _arr1_real(uniq), _arr1_int(mult)

    uk, um = km(uknots)
    vk, vm = km(vknots)
    surf = Geom_BSplineSurface(arr, uk, vk, um, vm, int(udeg), int(vdeg), False, False)
    return _MF(surf, 1e-6).Face()


def make_wire(entities, frame):
    _require()
    edges = _sketch_edges(entities, frame)
    if not edges:
        raise ValueError("ワイヤーを構成する要素がありません")
    mk = BRepBuilderAPI_MakeWire()
    for ed in edges:
        mk.Add(ed)
        if not mk.IsDone():
            raise RuntimeError("ワイヤーの連結に失敗しました（要素が連続していません）")
    return mk.Wire()


def make_planar_face(outer_wire, inner_wires=()):
    _require()
    mk = BRepBuilderAPI_MakeFace(outer_wire, True)
    if not mk.IsDone():
        raise RuntimeError("閉じた輪郭から面を作れません（輪郭が閉じていない/平面でない可能性）")
    for w in inner_wires:
        mk.Add(_cast("Wire", w.Reversed()))
    face = mk.Face()
    return face


def profile_face_from_loops(loops_entities, frame):
    """複数の閉ループ（2Dスケッチ要素）から穴付きの面を作る。最大面積のループを外周とする."""
    from ..core.sketch_fit import entities_bbox
    if not loops_entities:
        raise ValueError("閉じた輪郭がありません")
    areas = []
    for ents in loops_entities:
        pts = np.vstack([e.sample(32) for e in ents])
        x, y = pts[:, 0], pts[:, 1]
        areas.append(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    order = np.argsort(areas)[::-1]
    outer = make_wire(loops_entities[order[0]], frame)
    face = make_planar_face(outer)
    if len(order) > 1:
        # 内側ループは穴としてカット（向きの問題を避けるためブーリアンで処理）
        for k in order[1:]:
            w = make_wire(loops_entities[k], frame)
            hole = make_planar_face(w)
            face = BRepAlgoAPI_Cut(face, hole).Shape()
    return face


def extrude(face_or_shape, direction, distance: float, symmetric: bool = False):
    _require()
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    shape = face_or_shape
    if symmetric:
        shape = transform_shape(shape, _translation(-0.5 * distance * d))
    return BRepPrimAPI_MakePrism(shape, _vec(d * distance)).Shape()


def revolve(face_or_shape, axis_point, axis_dir, angle_deg: float = 360.0):
    _require()
    ax = gp_Ax1(_pnt(axis_point), _dir(axis_dir))
    if abs(angle_deg) >= 360.0 - 1e-9:
        return BRepPrimAPI_MakeRevol(face_or_shape, ax).Shape()
    return BRepPrimAPI_MakeRevol(face_or_shape, ax, float(np.radians(angle_deg))).Shape()


def boolean(a, b, op: str):
    _require()
    cls = {"fuse": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut, "common": BRepAlgoAPI_Common}[op]
    alg = cls(a, b)
    if not alg.IsDone():
        raise RuntimeError(f"ブーリアン演算 ({op}) に失敗しました")
    return alg.Shape()


def make_compound(shapes: Iterable):
    _require()
    b = BRep_Builder()
    c = TopoDS_Compound()
    b.MakeCompound(c)
    for s in shapes:
        if s is not None:
            b.Add(c, s)
    return c


def _translation(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def transform_shape(shape, T):
    """剛体変換（4x4）を適用した新しい形状."""
    _require()
    T = np.asarray(T, float)
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    tr = gp_Trsf()
    tr.SetValues(R[0, 0], R[0, 1], R[0, 2], T[0, 3],
                 R[1, 0], R[1, 1], R[1, 2], T[1, 3],
                 R[2, 0], R[2, 1], R[2, 2], T[2, 3])
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def scale_shape(shape, s: float, center=(0, 0, 0)):
    _require()
    tr = gp_Trsf()
    tr.SetScale(_pnt(center), float(s))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

_FACE_TYPE_NAMES = {}
if _AVAILABLE:
    _FACE_TYPE_NAMES = {GeomAbs_Plane: "平面", GeomAbs_Cylinder: "円筒", GeomAbs_Sphere: "球",
                        GeomAbs_Cone: "円錐", GeomAbs_Torus: "円環", GeomAbs_BSplineSurface: "Bスプライン"}


def is_valid(shape) -> bool:
    _require()
    return bool(BRepCheck_Analyzer(shape).IsValid())


def bbox(shape):
    _require()
    b = Bnd_Box()
    # テッセレーションの有無に依存しない厳密な外形（三角形・形状公差を使わない）
    try:
        BRepBndLib.AddOptimal_s(shape, b, False, False)
    except Exception:
        BRepBndLib.Add_s(shape, b, False)
    if b.IsVoid():
        return None
    try:
        lo, hi = b.CornerMin(), b.CornerMax()
        return [[lo.X(), lo.Y(), lo.Z()], [hi.X(), hi.Y(), hi.Z()]]
    except Exception:
        x0, y0, z0, x1, y1, z1 = b.Get()
        return [[x0, y0, z0], [x1, y1, z1]]


def analyze_faces(shape) -> list:
    out = []
    for f in _explore(shape, TopAbs_FACE):
        face = _cast("Face", f)
        ad = BRepAdaptor_Surface(face)
        t = ad.GetType()
        rec = {"type": _FACE_TYPE_NAMES.get(t, str(t).split(".")[-1])}
        if t == GeomAbs_Cylinder:
            c = ad.Cylinder()
            ax = c.Axis()
            rec.update(radius=c.Radius(), axis_point=_np(ax.Location()).tolist(),
                       axis_dir=[ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        elif t == GeomAbs_Plane:
            p = ad.Plane()
            ax = p.Axis()
            rec.update(origin=_np(p.Location()).tolist(),
                       normal=[ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        elif t == GeomAbs_Sphere:
            sp = ad.Sphere()
            rec.update(radius=sp.Radius(), center=_np(sp.Location()).tolist())
        out.append(rec)
    return out


def summarize(shape) -> dict:
    _require()
    faces = analyze_faces(shape)
    types = {}
    for f in faces:
        types[f["type"]] = types.get(f["type"], 0) + 1
    # 体積はソリッドのみで計算（開いた面を含む複合形状の「体積」は無意味なため）
    volume = 0.0
    n_solids = 0
    for so in _explore(shape, TopAbs_SOLID):
        vp = GProp_GProps()
        BRepGProp.VolumeProperties_s(so, vp)
        volume += abs(float(vp.Mass()))
        n_solids += 1
    sp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, sp)
    n_edges_free = 0
    return {
        "valid": is_valid(shape), "n_solids": n_solids,
        "n_shells": sum(1 for _ in _explore(shape, TopAbs_SHELL)),
        "n_faces": len(faces), "n_edges": sum(1 for _ in _explore(shape, TopAbs_EDGE)),
        "face_types": types,
        "face_types_text": ", ".join(f"{k}×{v}" for k, v in types.items()),
        "volume": volume,
        "area": float(sp.Mass()), "bbox": bbox(shape), "faces": faces[:200],
        "n_edges_free": n_edges_free,
    }


# ---------------------------------------------------------------------------
# テッセレーション（表示・偏差計算用）
# ---------------------------------------------------------------------------

def tessellate(shape, linear_deflection: float = 0.05, angular_deflection: float = 0.3,
               merge: bool = True):
    """形状を三角形メッシュ化。戻り値 (V, F)。merge=True で面間の重複頂点を統合."""
    _require()
    BRepMesh_IncrementalMesh(shape, float(linear_deflection), False, float(angular_deflection), True)
    Vs, Fs = [], []
    off = 0
    for f in _explore(shape, TopAbs_FACE):
        face = _cast("Face", f)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nn = tri.NbNodes()
        V = np.empty((nn, 3))
        for i in range(1, nn + 1):
            p = tri.Node(i).Transformed(trsf)
            V[i - 1] = (p.X(), p.Y(), p.Z())
        nt = tri.NbTriangles()
        F = np.empty((nt, 3), np.int64)
        for i in range(1, nt + 1):
            F[i - 1] = tri.Triangle(i).Get()
        F -= 1
        if face.Orientation() == TopAbs_REVERSED:
            F = F[:, ::-1]
        Vs.append(V)
        Fs.append(F + off)
        off += nn
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    V, F = np.vstack(Vs), np.vstack(Fs)
    if merge and len(V):
        scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
        q = np.round(V / (scale * 1e-9)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        V = V[first]
        F = inv.reshape(-1)[F]
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V, F


def edge_polylines(shape, deflection: float = 0.05) -> list:
    """全エッジ（重複なし）を折れ線化（ワイヤー表示用）."""
    _require()
    from OCP.TopExp import TopExp
    try:  # OCP 8
        from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as _IMap
    except ImportError:  # OCP 7.x
        from OCP.TopTools import TopTools_IndexedMapOfShape as _IMap
    emap = _IMap()
    mapper = getattr(TopExp, "MapShapes_s", None) or getattr(TopExp, "MapShapes")
    mapper(shape, TopAbs_EDGE, emap)
    out = []
    for i in range(1, emap.Extent() + 1):
        edge = _cast("Edge", emap.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        try:
            c = BRepAdaptor_Curve(edge)
            disc = GCPnts_TangentialDeflection(c, 0.2, float(deflection))
            pts = np.array([_np(disc.Value(k)) for k in range(1, disc.NbPoints() + 1)])
            if len(pts) >= 2:
                out.append(pts)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# STEP / BRep 入出力
# ---------------------------------------------------------------------------

def _ascii_name(name: str, fallback: str) -> str:
    s = re.sub(r"[^\x20-\x7e]", "", name or "").strip()
    return s or fallback


def write_step(items, path: str, schema: str = "AP214IS") -> str:
    """STEP 出力。items = [(shape, name), ...]（名前付きで XCAF 経由、失敗時は通常出力）.

    単位は mm。名前はASCII以外を除去（STEPの互換性のため）。
    """
    _require()
    items = [(s, n) for s, n in items if s is not None]
    if not items:
        raise ValueError("出力するCAD形状がありません")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", schema)
    Interface_Static.SetCVal_s("write.step.product.name", "LJM_RE")
    with occ_safe_path(path, "w") as p:
        try:
            _write_step_xcaf(items, p)
        except Exception as e:
            log.warning("XCAF STEP 出力に失敗したため通常出力に切替: %s", e)
            w = STEPControl_Writer()
            comp = make_compound([s for s, _ in items]) if len(items) > 1 else items[0][0]
            if w.Transfer(comp, STEPControl_AsIs) != IFSelect_RetDone:
                raise RuntimeError("STEP 変換に失敗しました")
            if w.Write(p) != IFSelect_RetDone:
                raise RuntimeError("STEP ファイルの書込みに失敗しました")
    return path


def _write_step_xcaf(items, path):
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for i, (shape, name) in enumerate(items, 1):
        lab = st.AddShape(shape, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(_ascii_name(name, f"Body_{i}")))
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    if not w.Transfer(doc, STEPControl_AsIs):
        raise RuntimeError("XCAF transfer failed")
    if w.Write(path) != IFSelect_RetDone:
        raise RuntimeError("STEP write failed")


def read_step(path: str):
    _require()
    with occ_safe_path(path, "r") as p:
        r = STEPControl_Reader()
        if r.ReadFile(p) != IFSelect_RetDone:
            raise RuntimeError("STEP ファイルを読み込めません")
        r.TransferRoots()
        shape = r.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError("STEP ファイルに形状がありません")
    return shape


def shape_to_brep_bytes(shape) -> bytes:
    _require()
    tmpdir = tempfile.mkdtemp(prefix="ljm_brep_")
    try:
        p = os.path.join(tmpdir, "s.brep")
        with occ_safe_path(p, "w") as q:
            BRepTools.Write_s(shape, q)
        with open(p, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def shape_from_brep_bytes(data: bytes):
    _require()
    tmpdir = tempfile.mkdtemp(prefix="ljm_brep_")
    try:
        p = os.path.join(tmpdir, "s.brep")
        with open(p, "wb") as f:
            f.write(data)
        shape = TopoDS_Shape()
        with occ_safe_path(p, "r") as q:
            BRepTools.Read_s(shape, q, BRep_Builder())
        return shape
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def verify_step(path: str, reference_shapes: list, tol: float = 1e-4) -> dict:
    """STEP を再読込し、幾何形状と寸法が元形状と一致するか検証する."""
    _require()
    rd = read_step(path)
    ref = make_compound(reference_shapes) if len(reference_shapes) > 1 else reference_shapes[0]
    a, b = summarize(ref), summarize(rd)
    issues = []
    if not b["valid"]:
        issues.append("再読込した形状が BRepCheck で無効と判定されました")
    if a["face_types"] != b["face_types"]:
        issues.append(f"面の種類が一致しません: 元 {a['face_types']} / 再読込 {b['face_types']}")
    if a["n_solids"] != b["n_solids"]:
        issues.append(f"ソリッド数が一致しません: {a['n_solids']} / {b['n_solids']}")
    if a["volume"] > 0:
        rel = abs(a["volume"] - b["volume"]) / a["volume"]
        if rel > 1e-6:
            issues.append(f"体積差 {rel * 100:.6f} %")
    if a["area"] > 0:
        rel = abs(a["area"] - b["area"]) / a["area"]
        if rel > 1e-6:
            issues.append(f"表面積差 {rel * 100:.6f} %")
    if a["bbox"] and b["bbox"]:
        db = float(np.max(np.abs(np.array(a["bbox"]) - np.array(b["bbox"]))))
        if db > max(tol, 1e-3):
            issues.append(f"外形の差 {db:.6f} mm")
    else:
        db = None
    ra = sorted(round(f["radius"], 6) for f in a["faces"] if "radius" in f)
    rb = sorted(round(f["radius"], 6) for f in b["faces"] if "radius" in f)
    if len(ra) != len(rb) or (ra and np.max(np.abs(np.array(ra) - np.array(rb))) > tol):
        issues.append(f"円筒/球の半径が一致しません: {ra[:10]} / {rb[:10]}")
    return {"ok": not issues, "issues": issues, "reference": {k: a[k] for k in ("n_faces", "n_solids", "face_types", "volume", "area", "bbox")},
            "reloaded": {k: b[k] for k in ("n_faces", "n_solids", "face_types", "volume", "area", "bbox")},
            "radii": rb, "bbox_diff": db, "path": path}
