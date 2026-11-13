"""テスト用の試験形状を OCP / PyVista で生成するフィクスチャ。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")


def _write_step(shape, path: Path) -> Path:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    from ljmcfd.cad.occ_io import silence_occ_messages

    silence_occ_messages()
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def make_box(dx: float, dy: float, dz: float, origin=(0.0, 0.0, 0.0)):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    return BRepPrimAPI_MakeBox(gp_Pnt(*origin), dx, dy, dz).Shape()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cad")


@pytest.fixture(scope="session")
def box_step(data_dir) -> Path:
    """4000 x 1800 x 1200 mm の直方体 (mm 単位の STEP)。"""
    return _write_step(make_box(4000, 1800, 1200), data_dir / "box_mm.step")


@pytest.fixture(scope="session")
def box_step_metre(data_dir) -> Path:
    """4 x 1.8 x 1.2 m の直方体を **メートル単位** で記述した STEP。"""
    src = _write_step(make_box(4, 1.8, 1.2), data_dir / "box_tmp.step")
    text = src.read_text()
    assert "SI_UNIT(.MILLI.,.METRE.)" in text
    out = data_dir / "box_m.step"
    out.write_text(text.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)"))
    return out


@pytest.fixture(scope="session")
def assembly_step(data_dir) -> Path:
    """Body (4000x1800x1200) と RearWing (300x1600x30, 赤, 平行移動つき) のアセンブリ STEP。"""
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_DocumentTool
    from OCP.gp import gp_Trsf, gp_Vec

    from ljmcfd.cad.occ_io import silence_occ_messages

    silence_occ_messages()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    asm = shape_tool.NewShape()
    TDataStd_Name.Set_s(asm, TCollection_ExtendedString("Car"))
    body = shape_tool.AddShape(make_box(4000, 1800, 1200), False)
    TDataStd_Name.Set_s(body, TCollection_ExtendedString("Body"))
    wing = shape_tool.AddShape(make_box(300, 1600, 30), False)
    TDataStd_Name.Set_s(wing, TCollection_ExtendedString("RearWing"))
    color_tool.SetColor(wing, Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB), XCAFDoc_ColorGen)
    move = gp_Trsf()
    move.SetTranslation(gp_Vec(3700, 100, 1300))
    shape_tool.AddComponent(asm, body, TopLoc_Location())
    shape_tool.AddComponent(asm, wing, TopLoc_Location(move))
    shape_tool.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    assert writer.Transfer(doc, STEPControl_AsIs)
    path = data_dir / "car_assembly.step"
    writer.Write(str(path))
    return path


@pytest.fixture(scope="session")
def open_box_step(data_dir) -> Path:
    """直方体から 1 面を取り除いた開いたシェルの STEP。"""
    from OCP.BRep import BRep_Builder
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS_Shell

    from ljmcfd.cad.occ_io import iter_subshapes

    faces = list(iter_subshapes(make_box(1000, 500, 300), TopAbs_FACE))
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces[:-1]:
        builder.Add(shell, face)
    return _write_step(shell, data_dir / "open_box.step")


@pytest.fixture(scope="session")
def multi_solid_step(data_dir) -> Path:
    """アセンブリ構造なしで 2 ソリッドを 1 つのコンパウンドに入れた STEP。"""
    from ljmcfd.cad.occ_io import make_compound

    compound = make_compound([make_box(100, 100, 100), make_box(50, 50, 50, origin=(300, 0, 0))])
    return _write_step(compound, data_dir / "two_solids.step")


@pytest.fixture(scope="session")
def cylinder_iges(data_dir) -> Path:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.IGESControl import IGESControl_Writer

    writer = IGESControl_Writer("MM", 0)
    writer.AddShape(BRepPrimAPI_MakeCylinder(50.0, 200.0).Shape())
    writer.ComputeModel()
    path = data_dir / "cylinder.igs"
    assert writer.Write(str(path))
    return path


@pytest.fixture(scope="session")
def cube_stl(data_dir) -> Path:
    """2 x 1 x 0.5 の直方体 STL (バイナリ、メートル相当の寸法)。"""
    import pyvista as pv

    path = data_dir / "cube.stl"
    pv.Cube(x_length=2.0, y_length=1.0, z_length=0.5).triangulate().save(path)
    return path


@pytest.fixture(scope="session")
def sphere_stl_mm(data_dir) -> Path:
    """半径 500 の球 STL (mm 相当の寸法)。"""
    import pyvista as pv

    path = data_dir / "sphere_mm.stl"
    pv.Sphere(radius=500.0, theta_resolution=24, phi_resolution=24).save(path)
    return path


@pytest.fixture(scope="session")
def open_stl(data_dir) -> Path:
    """上面を取り除いた立方体 STL (開放エッジ 4 本 = 三角形エッジ 4 本)。"""
    import numpy as np
    import pyvista as pv

    cube = pv.Cube().triangulate()
    centers = cube.cell_centers().points
    keep = np.flatnonzero(centers[:, 2] < 0.49)
    path = data_dir / "open_cube.stl"
    cube.extract_cells(keep).extract_surface(algorithm=None).save(path)
    return path


@pytest.fixture()
def cache_root(tmp_path) -> Path:
    root = tmp_path / "cache"
    root.mkdir()
    return root


@pytest.fixture(scope="session")
def instanced_step(data_dir) -> Path:
    """同じ形状 (ホイール) を別名で 2 か所に配置したアセンブリ STEP。"""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.gp import gp_Trsf, gp_Vec

    from ljmcfd.cad.occ_io import silence_occ_messages

    silence_occ_messages()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    asm = shape_tool.NewShape()
    TDataStd_Name.Set_s(asm, TCollection_ExtendedString("Axle"))
    wheel = shape_tool.AddShape(BRepPrimAPI_MakeCylinder(300.0, 200.0).Shape(), False)
    TDataStd_Name.Set_s(wheel, TCollection_ExtendedString("Wheel"))
    for name, x in (("Wheel_FL", 0.0), ("Wheel_RL", 2600.0)):
        move = gp_Trsf()
        move.SetTranslation(gp_Vec(x, 0, 0))
        component = shape_tool.AddComponent(asm, wheel, TopLoc_Location(move))
        TDataStd_Name.Set_s(component, TCollection_ExtendedString(name))
    shape_tool.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    assert writer.Transfer(doc, STEPControl_AsIs)
    path = data_dir / "instanced.step"
    writer.Write(str(path))
    return path
