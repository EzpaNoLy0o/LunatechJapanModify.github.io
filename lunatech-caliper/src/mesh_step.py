"""
mesh_step.py - convert a closed triangle mesh into a faceted B-Rep solid
(one planar face per triangle, shared vertices/edges, consistent
orientation) so that the organic SDF geometry can be written to STEP.

The result is a valid closed solid that CAD systems can import, measure
and section; it is *faceted* (not NURBS) - see reports/Geometry_Validation.md.
"""
import os
import tempfile
import numpy as np
import trimesh
import cadquery as cq
from OCP.StlAPI import StlAPI_Reader
from OCP.TopoDS import TopoDS_Shape, TopoDS, TopoDS_Solid
from OCP.BRep import BRep_Builder
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.Interface import Interface_Static
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing


def step_compact_mode():
    """Do not write 2D parameter-space curves (p-curves) -> ~4x smaller files."""
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.unit", "MM")


def mesh_to_solid(V, F):
    tm = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, np.int64), process=False)
    fd, path = tempfile.mkstemp(suffix=".stl")
    os.close(fd)
    tm.export(path)
    shape = TopoDS_Shape()
    StlAPI_Reader().Read(shape, path)
    os.remove(path)
    sew = BRepBuilderAPI_Sewing(1e-4)
    sew.Load(shape)
    sew.Perform()
    sewed = sew.SewedShape()
    exp = TopExp_Explorer(sewed, TopAbs_SHELL)
    shell = TopoDS.Shell_s(exp.Current())
    bld = BRep_Builder()
    solid = TopoDS_Solid()
    bld.MakeSolid(solid)
    bld.Add(solid, shell)
    s = cq.Solid(solid)
    if s.Volume() < 0:
        s = cq.Solid(TopoDS.Solid_s(solid.Reversed()))
    return s
