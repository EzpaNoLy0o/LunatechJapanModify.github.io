"""動作確認用のデモ CAD モデルを生成する。

    python scripts/make_demo_models.py [出力フォルダ]

* ``ahmed_body.step``  Ahmed ボディ (スラント角 25°)。自動車空力の標準ベンチマーク形状で、
                       Stage 3 以降の CFD 検証 (文献値 Cd ≈ 0.285) にも使う。
* ``demo_car.step``    ボディ + リアウイング + 4 輪のアセンブリ (パーツ名・色つき)。
* ``open_panel.step``  1 面を取り除いた開いたシェル (形状検査の確認用)。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shell
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_DocumentTool
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
import OCP.TopoDS as _topods

_to_edge = getattr(_topods.TopoDS, "Edge_s", None) or _topods.TopoDS.Edge


def _edge_midpoint_and_length(edge):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint

    curve = BRepAdaptor_Curve(edge)
    mid = curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2)
    return mid, GCPnts_AbscissaPoint.Length_s(curve)


def ahmed_body(slant_deg: float = 25.0, clearance: float = 50.0):
    """Ahmed ボディ [mm]。ノーズは -X 向き、後端が X = 0、地面 Z = 0。"""
    length, width, height, nose_r, slant_len = 1044.0, 389.0, 288.0, 100.0, 222.0
    box = BRepPrimAPI_MakeBox(gp_Pnt(-length, -width / 2, clearance), length, width, height).Shape()
    # 前面を囲む 4 辺を R100 で丸める
    fillet = BRepFilletAPI_MakeFillet(box)
    explorer = TopExp_Explorer(box, TopAbs_EDGE)
    seen = []
    while explorer.More():
        edge = _to_edge(explorer.Current())
        mid, _ = _edge_midpoint_and_length(edge)
        key = (round(mid.X(), 3), round(mid.Y(), 3), round(mid.Z(), 3))
        if abs(mid.X() + length) < 1e-6 and key not in seen:
            seen.append(key)
            fillet.Add(nose_r, edge)
        explorer.Next()
    body = fillet.Shape()
    # 後部スラント: XZ 平面の三角形を Y 方向へ押し出して切り落とす
    dx = slant_len * math.cos(math.radians(slant_deg))
    dz = slant_len * math.sin(math.radians(slant_deg))
    top = clearance + height
    polygon = BRepBuilderAPI_MakePolygon(
        gp_Pnt(-dx, -width, top), gp_Pnt(10.0, -width, top + 10.0 * dz / dx), gp_Pnt(10.0, -width, top - dz - 10.0 * dz / dx), True,
    )
    wedge = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(polygon.Wire()).Face(), gp_Vec(0, 2 * width, 0)).Shape()
    return BRepAlgoAPI_Cut(body, wedge).Shape()


def _moved(shape, dx=0.0, dy=0.0, dz=0.0):
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def write_step(shape, path: Path) -> None:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP を書き込めません: {path}")


def write_demo_car(path: Path) -> None:
    """簡易車両アセンブリ [mm]: 全長 4.2 m、ノーズ -X 向き。"""
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    root = shapes.NewShape()
    TDataStd_Name.Set_s(root, TCollection_ExtendedString("DemoCar"))

    def add(name, shape, rgb, location=None):
        label = shapes.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        colors.SetColor(label, Quantity_Color(*rgb, Quantity_TOC_RGB), XCAFDoc_ColorGen)
        # 同じ形状 (ホイール) を複数配置するので、名前はインスタンス側にも付ける
        component = shapes.AddComponent(root, label, location or TopLoc_Location())
        TDataStd_Name.Set_s(component, TCollection_ExtendedString(name))

    # ボディ: Ahmed 形状を 4 倍に拡大 (全長 4176 mm)
    scale = gp_Trsf()
    scale.SetScale(gp_Pnt(0, 0, 0), 4.0)
    body = BRepBuilderAPI_Transform(ahmed_body(clearance=30.0), scale, True).Shape()
    add("Body", body, (0.75, 0.78, 0.82))
    wing = BRepPrimAPI_MakeBox(gp_Pnt(-350, -700, 1360), 320, 1400, 25).Shape()
    add("RearWing", wing, (0.85, 0.12, 0.12))
    for side, y in (("L", -800.0), ("R", 800.0 - 60.0)):
        add(f"WingEndplate_{side}", BRepPrimAPI_MakeBox(gp_Pnt(-380, y, 1150), 380, 60, 280).Shape(), (0.2, 0.2, 0.22))
    wheel = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), 330.0, 245.0).Shape()
    for name, x, y in (("Wheel_FL", -3300.0, -860.0), ("Wheel_FR", -3300.0, 615.0),
                       ("Wheel_RL", -800.0, -860.0), ("Wheel_RR", -800.0, 615.0)):
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(x, y, 330.0))
        add(name, wheel, (0.12, 0.12, 0.13), TopLoc_Location(trsf))
    shapes.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    if not writer.Transfer(doc, STEPControl_AsIs) or writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP を書き込めません: {path}")


def open_panel():
    faces = []
    explorer = TopExp_Explorer(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 800, 400, 60).Shape(), TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces[:-1]:
        builder.Add(shell, face)
    return shell


def main(out_dir: str = "demo") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_step(ahmed_body(), out / "ahmed_body.step")
    write_demo_car(out / "demo_car.step")
    write_step(open_panel(), out / "open_panel.step")
    for file in sorted(out.glob("*.step")):
        print(f"{file}  ({file.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
