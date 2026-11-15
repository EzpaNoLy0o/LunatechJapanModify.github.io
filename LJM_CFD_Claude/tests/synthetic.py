"""解析結果の合成データ。OpenFOAM が無い環境 (Windows など) でも可視化機能をテストできるようにする。

流れ場は解析的に値が分かる単純なもの:

* 速度   U = (U0 · (1 + 0.1 z), 0, 0)       … 流線は X 方向の直線、流速は高さとともに増える
* 圧力   p/ρ = −x                          … 圧力は X 方向に直線的に下がる
* k      = y²
* Q      = z − 1                           … Q = 0 の等値面は平面 z = 1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

from ljmcfd.solver.results import FIELDS_FILE, SUMMARY_FILE, SUMMARY_VERSION, _save_uncompressed

U0 = 50.0
RHO = 1.2
DOMAIN = (-4.0, 8.0, -2.0, 2.0, 0.0, 3.0)
BODY = (0.0, 2.0, -0.5, 0.5, 0.2, 1.0)


def _fields(points: np.ndarray) -> dict[str, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = np.column_stack([U0 * (1 + 0.1 * z), np.zeros_like(x), np.zeros_like(x)])
    p = -x
    vorticity = np.column_stack([np.zeros_like(x), np.full_like(x, 0.1 * U0), np.zeros_like(x)])
    return {
        "U": u, "p": p, "k": y ** 2, "Q": z - 1.0, "vorticity": vorticity, "nut": np.full_like(x, 1e-4),
        "Umag": np.linalg.norm(u, axis=1), "pressure": p * RHO, "Cp": p / (0.5 * U0 ** 2),
        "vorticity_mag": np.linalg.norm(vorticity, axis=1),
    }


def make_results(results_dir: str | Path, input_hash: str = "synthetic", status: str = "completed") -> dict:
    """合成の解析結果フォルダを作り、サマリーを返す。"""
    results_dir = Path(results_dir)
    (results_dir / "surface").mkdir(parents=True, exist_ok=True)
    grid = pv.ImageData(dimensions=(41, 17, 13), spacing=(0.3, 0.25, 0.25), origin=(DOMAIN[0], DOMAIN[2], DOMAIN[4]))
    volume = grid.cast_to_unstructured_grid().triangulate()
    for name, values in _fields(np.asarray(volume.points, dtype=float)).items():
        volume.point_data[name] = values
    volume.cell_data.clear()
    _save_uncompressed(volume, results_dir / FIELDS_FILE)

    surfaces = {}
    body = pv.Box(bounds=BODY, level=6).triangulate()
    ground = pv.Plane(center=(2.0, 0.0, 0.0), direction=(0, 0, 1), i_size=12.0, j_size=4.0, i_resolution=24, j_resolution=8).triangulate()
    for name, surface in (("Body", body), ("ground", ground)):
        pts = np.asarray(surface.points, dtype=float)
        values = _fields(pts)
        surface.point_data["pressure"] = values["pressure"]
        surface.point_data["Cp"] = values["Cp"]
        surface.point_data["wallShearStress"] = np.column_stack([-0.01 * (1 + pts[:, 0]), np.zeros(len(pts)), np.zeros(len(pts))])
        surface.point_data["yPlus"] = 100.0 + 10.0 * pts[:, 0]
        _save_uncompressed(surface, results_dir / "surface" / f"{name}.vtp")
        surfaces[name] = f"surface/{name}.vtp"

    q = 0.5 * RHO * U0 ** 2
    summary = {
        "summary_version": SUMMARY_VERSION, "status": status,
        "status_label": "計算完了 (指定反復数に到達)", "created": "2026-01-01 00:00:00", "input_hash": input_hash,
        "iterations": {"start": 0, "requested_end": 500, "done": 500},
        "openfoam": {"version": "14", "kind": "native", "n_procs": 4},
        "flow": {"speed": U0, "nu": 1.5e-5, "rho": RHO, "k": 0.4, "omega": 2000.0}, "moving_ground": True,
        "mesh": {"n_cells": int(volume.n_cells), "boundary_layer": {"generated": False}, "input_hash": "mesh", "checkmesh": {"ok": True, "messages": []}},
        "non_ortho_correctors": 1, "seconds_solver": 12.0, "seconds_total": 15.0, "error": None,
        "forces": {"averaged_over": 100, "drag": 100.0, "side": 2.0, "lift": -300.0, "downforce": 300.0,
                   "drag_pressure": 85.0, "drag_viscous": 15.0, "lift_pressure": -298.0, "lift_viscous": -2.0,
                   "drag_range": 0.4, "lift_range": 1.0, "cd": 100.0 / (q * 2.0), "cl": -300.0 / (q * 2.0), "cs": 2.0 / (q * 2.0),
                   "reference_area": 2.0, "dynamic_pressure": q},
        "parts": [], "yplus": {"Body": {"min": 30.0, "max": 200.0, "average": 110.0}},
        "convergence": {"level": "converged", "label": "工学的に十分収束", "window": 100, "message": "荷重・残差・質量収支のすべてが基準を満たしています。",
                        "checks": [{"name": "抗力の安定", "ok": True, "text": "合成データ"}]},
        "final_residuals": {"p": 5e-4}, "mass": {"imbalance": 1e-7, "inlet": -600.0, "outlet": 600.0}, "notes": [],
        "ranges": {"speed_max": U0 * 1.3, "pressure_min": -8.0 * RHO, "pressure_max": 4.0 * RHO, "cp_min": -8 / (0.5 * U0 ** 2), "cp_max": 4 / (0.5 * U0 ** 2),
                   "surface_pressure_min": -2.0 * RHO, "surface_pressure_max": 0.0},
        "files": {"fields": FIELDS_FILE, "surfaces": surfaces}, "fields": sorted(volume.point_data.keys()), "continuity": None,
    }
    (results_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    summary["results_dir"] = str(results_dir)
    return summary
