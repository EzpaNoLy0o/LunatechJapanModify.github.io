# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 設定: LJM Reverse Engineering（onedir 形式）
#   python -m PyInstaller --noconfirm --clean ljm_re.spec
#
# onedir を採用する理由:
#   - PySide6 (LGPL-3.0) の DLL を利用者が差し替え可能な形で同梱するため
#   - VTK / OpenCASCADE の大量の DLL を起動ごとに展開しないため（起動が速い）
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None
here = os.path.abspath(SPECPATH)

hiddenimports = []
hiddenimports += collect_submodules("vtkmodules")
hiddenimports += collect_submodules("pyvista", filter=lambda n: not n.startswith("pyvista.trame"))
hiddenimports += collect_submodules("pyvistaqt")
hiddenimports += collect_submodules("OCP")
hiddenimports += collect_submodules("ljm_re")
hiddenimports += ["shapely", "mapbox_earcut", "rtree", "ezdxf", "trimesh", "qtpy"]

datas = []
datas += collect_data_files("pyvista")
datas += collect_data_files("trimesh")
datas += collect_data_files("ezdxf")
datas += [(os.path.join(here, "docs"), "docs"), (os.path.join(here, "README.md"), ".")]

binaries = []
binaries += collect_dynamic_libs("OCP")
binaries += collect_dynamic_libs("vtkmodules")
binaries += collect_dynamic_libs("rtree")
binaries += collect_dynamic_libs("shapely")
# delvewheel / auditwheel で同梱された依存 DLL (cadquery_ocp.libs 等)
import site
for sp in site.getsitepackages():
    for libdir in ("cadquery_ocp.libs", "vtk.libs", "shapely.libs", "scipy.libs", "numpy.libs"):
        p = os.path.join(sp, libdir)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                binaries.append((os.path.join(p, fn), libdir))

excludes = ["tkinter", "IPython", "jupyter", "jupyter_client", "jupyter_core", "notebook", "ipywidgets",
            "dash", "flask", "plotly", "open3d", "PyQt5", "PyQt6", "PySide2",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
            "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "pytest"]

a = Analysis(
    [os.path.join(here, "main.py")],
    pathex=[here],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LJM_ReverseEngineering",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="LJM_ReverseEngineering")
