"""レポート用のスナップショット: モデル・風洞・メッシュ・解析結果の「いま」を 1 つの辞書にまとめる。

Qt には依存しない。GUI はシーンの中身を渡すだけで、レポートの整形 (``export.py``) は辞書だけを見る。
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import __version__
from ..solver.results import read_history


def build_snapshot(*, project_name: str | None, assembly, analysis_parts: list, bounds, tunnel, mesh_report: dict | None,
                   mesh_state: str, solver_settings, solver_summary: dict | None, solver_state: str) -> dict:
    """
    assembly:        ``cad.assembly.Assembly``
    analysis_parts:  解析対象のパーツ (表示中のパーツ)
    bounds:          解析対象のワールド境界 [m] または ``None``
    tunnel:          ``tunnel.spec.TunnelSpec``
    """
    analysed = {part.id for part in analysis_parts}
    parts = [{
        "name": part.name, "file": Path(part.source_path).name, "unit": part.unit, "analysed": part.id in analysed,
        "position": (part.pose.tx, part.pose.ty, part.pose.tz), "rotation": (part.pose.yaw, part.pose.pitch, part.pose.roll),
    } for part in assembly.parts]
    flow = tunnel.flow
    length = None if bounds is None else bounds[1] - bounds[0]
    history = None
    if solver_summary is not None and solver_summary.get("results_dir"):
        path = Path(solver_summary["results_dir"]) / ((solver_summary.get("files") or {}).get("history") or "history.csv")
        if path.is_file():
            history = read_history(path)
    return {
        "app_version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": project_name,
        "model": {"parts": parts, "bounds": None if bounds is None else [float(v) for v in bounds], "vehicle_pose": assembly.pose.to_dict()},
        "tunnel": tunnel.to_dict(),
        "flow": {
            "speed_ms": flow.speed_ms, "speed_kmh": flow.speed_kmh, "density": flow.density, "nu": flow.nu, "mach": flow.mach,
            "dynamic_pressure": flow.dynamic_pressure, "reynolds": None if not length else flow.reynolds(length),
        },
        "domain": None if bounds is None or not tunnel.generated else [float(v) for v in tunnel.domain_bounds(bounds)],
        "blockage": None if bounds is None or not tunnel.generated else tunnel.blockage_ratio(bounds),
        "reference_area": tunnel.effective_reference_area(),
        "mesh": {"state": mesh_state, "report": mesh_report},
        "solver": {"state": solver_state, "settings": solver_settings.to_dict(), "summary": solver_summary, "history": history},
    }
