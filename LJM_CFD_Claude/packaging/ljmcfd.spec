# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の定義。``scripts\\build_windows.bat`` から使う (手動なら ``pyinstaller packaging\\ljmcfd.spec``)。

出力: ``dist/LJM_CFD_Claude/`` フォルダ (onedir)。フォルダごとコピーすれば Python なしで起動できる。
OpenFOAM は含まない (WSL2 側に別途インストールする)。
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

hiddenimports = (
    collect_submodules("ljmcfd")                 # ジョブは名前で遅延 import されるので、解析に任せず全て含める
    + collect_submodules("vtkmodules")           # pyvista は VTK のモジュールを実行時に名前で読み込む
    + ["OCP", "gmsh", "pyvistaqt", "pyqtgraph", "scipy.spatial", "PIL.Image", "matplotlib.cm"]
)
datas = collect_data_files("pyvista") + collect_data_files("pyvistaqt") + collect_data_files("matplotlib", includes=["mpl-data/**"])
binaries = collect_dynamic_libs("OCP") + collect_dynamic_libs("gmsh")

excludes = [
    "tkinter", "IPython", "jupyter", "notebook", "pytest", "PyQt5", "PyQt6", "PySide2", "trame", "wx",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore",
    "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization",
]

analysis = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(analysis.pure)

gui = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="LJM_CFD_Claude", console=False, upx=False)
worker = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="ljmcfd-worker", console=True, upx=False)
COLLECT(gui, worker, analysis.binaries, analysis.datas, upx=False, name="LJM_CFD_Claude")
