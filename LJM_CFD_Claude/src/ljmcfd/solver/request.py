"""シーンの状態から解析ジョブ (``solver-run``) の引数を組み立てる。"""

from __future__ import annotations

from pathlib import Path

from ..mesh.settings import input_hash
from ..tunnel.spec import TunnelSpec
from .environment import FoamEnvironment
from .settings import SolverSettings


def flow_payload(tunnel: TunnelSpec) -> dict:
    flow = tunnel.flow
    return {"speed": flow.speed_ms, "nu": flow.nu, "rho": flow.density, "k": flow.k_inlet, "omega": flow.omega_inlet}


def solver_input_hash(mesh_report: dict, tunnel: TunnelSpec, settings: SolverSettings) -> str:
    """解析結果を決める入力のハッシュ。メッシュ・流れの条件・地面の扱い・ソルバー設定から作る。

    基準面積は含めない (係数の換算にしか使わず、流れ場は変わらないため)。
    並列数と「続きから」も結果の同一性には関係しないので含めない。
    """
    solver = settings.to_dict()
    for key in ("n_procs", "continue_run"):
        solver.pop(key)
    return input_hash({
        "mesh": mesh_report.get("input_hash"),
        "flow": flow_payload(tunnel),
        "moving_ground": bool(tunnel.moving_ground and tunnel.has_ground),
        "solver": solver,
    })


def build_solver_request(mesh_report: dict, tunnel: TunnelSpec, settings: SolverSettings, environment: FoamEnvironment,
                         work_dir: str | Path) -> dict:
    """解析ジョブの引数。``work_dir`` の下に ``case`` と ``results`` を作る。"""
    work_dir = Path(work_dir)
    quality = mesh_report["quality"]
    return {
        "case_dir": str(work_dir / "case"),
        "results_dir": str(work_dir / "results"),
        "mesh_dir": mesh_report["out_dir"],
        "mesh": {
            "patches": mesh_report["patches"],
            "n_cells": quality["n_cells"],
            "non_ortho_max": quality["non_ortho_max"],
            "boundary_layer": mesh_report["boundary_layer"],
            "input_hash": mesh_report.get("input_hash"),
        },
        "flow": flow_payload(tunnel),
        "moving_ground": bool(tunnel.moving_ground and tunnel.has_ground),
        "reference_area": tunnel.effective_reference_area(),
        "settings": settings.to_dict(),
        "environment": environment.to_dict(),
        "input_hash": solver_input_hash(mesh_report, tunnel, settings),
    }
