"""依存ライブラリのバージョンとライセンス情報."""
from __future__ import annotations

import platform
import sys

LIBS = [
    # (import名, 表示名, ライセンス, 備考)
    ("PySide6", "PySide6 (Qt for Python)", "LGPL-3.0", "動的リンクで利用。EXE配布時は Qt DLL を差し替え可能な onedir 形式で同梱"),
    ("vtk", "VTK", "BSD-3-Clause", ""),
    ("pyvista", "PyVista", "MIT", ""),
    ("pyvistaqt", "PyVistaQt", "MIT", ""),
    ("numpy", "NumPy", "BSD-3-Clause", ""),
    ("scipy", "SciPy", "BSD-3-Clause", ""),
    ("trimesh", "Trimesh", "MIT", ""),
    ("shapely", "Shapely", "BSD-3-Clause", "GEOS (LGPL-2.1) を含む"),
    ("mapbox_earcut", "mapbox-earcut", "ISC", ""),
    ("rtree", "Rtree", "MIT", ""),
    ("networkx", "NetworkX", "BSD-3-Clause", ""),
    ("ezdxf", "ezdxf", "MIT", ""),
    ("OCP", "OCP (cadquery-ocp) / OpenCASCADE", "Apache-2.0 / OCCT: LGPL-2.1 + 例外", "CADカーネル"),
    ("open3d", "Open3D（任意）", "MIT", "未インストールでも基本機能は動作"),
    ("matplotlib", "Matplotlib", "PSF/Matplotlib License", "pyvista の依存"),
]


def _version(mod):
    try:
        m = __import__(mod)
    except Exception:
        return None
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(m, attr, None)
        if isinstance(v, str):
            return v
    try:
        from importlib.metadata import version
        name = {"OCP": "cadquery-ocp", "mapbox_earcut": "mapbox_earcut", "PySide6": "PySide6"}.get(mod, mod)
        return version(name)
    except Exception:
        return "インストール済み"


def library_report() -> str:
    lines = [f"Python {sys.version.split()[0]} ({platform.architecture()[0]}) / {platform.platform()}", ""]
    for mod, name, lic, note in LIBS:
        v = _version(mod)
        st = v if v else "未インストール"
        lines.append(f"{name:36s} {st:16s} ライセンス: {lic}" + (f"  ※{note}" if note else ""))
    lines += ["", "各ライブラリのライセンス全文は docs/LICENSES.md を参照してください。"]
    return "\n".join(lines)
