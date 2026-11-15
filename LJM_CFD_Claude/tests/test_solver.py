"""ソルバー連携: 設定・環境・ケース生成・出力解析・収束判定と、実物の OpenFOAM による解析。

OpenFOAM が入っていない環境では、実行を伴うテストだけをスキップする。
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from ljmcfd.cad.assembly import Assembly
from ljmcfd.cad.import_job import import_cad_job
from ljmcfd.core.jobs import JobError, RecordingContext
from ljmcfd.mesh.gmsh_job import mesh_generate_job
from ljmcfd.mesh.request import build_mesh_request
from ljmcfd.mesh.settings import MeshSettings
from ljmcfd.solver import case, convergence as cv, monitor
from ljmcfd.solver.environment import (
    KIND_NATIVE, KIND_WSL, FoamEnvironment, check_environment_job, default_environment, native_openfoam_available,
)
from ljmcfd.solver.request import build_solver_request, solver_input_hash
from ljmcfd.solver.results import force_summary, read_history, write_history
from ljmcfd.solver.run_job import (
    STATUS_COMPLETED, STATUS_FAILED, STATUS_RESIDUAL_STOP, STATUS_STOPPED, load_summary, solver_run_job,
)
from ljmcfd.solver.settings import SolverSettings, auto_process_count
from ljmcfd.tunnel import spec as ts

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")

PATCHES = [
    {"name": "inlet", "type": "patch", "kind": "tunnel"}, {"name": "outlet", "type": "patch", "kind": "tunnel"},
    {"name": "sideL", "type": "patch", "kind": "tunnel"}, {"name": "sideR", "type": "patch", "kind": "tunnel"},
    {"name": "ground", "type": "wall", "kind": "tunnel"}, {"name": "top", "type": "patch", "kind": "tunnel"},
    {"name": "Body", "type": "wall", "kind": "part", "part_id": "a", "part_name": "Body"},
    {"name": "RearWing", "type": "wall", "kind": "part", "part_id": "b", "part_name": "RearWing"},
]
FLOW = {"speed": 55.5556, "nu": 1.5e-5, "rho": 1.2, "k": 0.46, "omega": 3000.0}


# ----------------------------------------------------------------------------- 設定・環境

def test_solver_settings():
    settings = SolverSettings()
    assert settings.iterations == 500 and not settings.validate()
    assert SolverSettings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings
    # 非直交補正の自動決定: 四面体 (非直交性 60° 台) では 1 回、70° 以上で 2 回
    assert [settings.resolved_non_ortho_correctors(a) for a in (20.0, 65.0, 75.0)] == [0, 1, 2]
    settings.non_ortho_correctors = 3
    assert settings.resolved_non_ortho_correctors(20.0) == 3
    assert SolverSettings(iterations=0).validate() and SolverSettings(relax_p=1.5).validate()
    # 並列数: 1 プロセス 2.5 万セル以上、物理コア数まで
    assert auto_process_count(10_000, 8) == 1 and auto_process_count(100_000, 8) == 4 and auto_process_count(1_000_000, 8) == 8


def test_environment_commands_and_paths(tmp_path):
    wsl = FoamEnvironment(kind=KIND_WSL, distro="Ubuntu-24.04")
    assert wsl.path(r"C:\LJM_CFD_Claude\projects\car01\case") == "/mnt/c/LJM_CFD_Claude/projects/car01/case"
    assert wsl.script_argv(r"C:\LJM_CFD_Claude\projects\car01\case\.ljmcfd_step.sh") == [
        "wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "/mnt/c/LJM_CFD_Claude/projects/car01/case/.ljmcfd_step.sh"]
    wsl.check_case_path(r"C:\LJM_CFD_Claude\projects\car01\case")
    with pytest.raises(JobError, match="空白・日本語"):
        wsl.check_case_path(r"C:\Users\太郎\My Projects\case")
    native = FoamEnvironment(kind=KIND_NATIVE, bashrc="/opt/openfoam14/etc/bashrc")
    assert native.script_argv("/tmp/x/s.sh") == ["bash", "/tmp/x/s.sh"]
    script = native.write_script(tmp_path, "checkMesh > log 2>&1")
    raw = script.read_bytes()
    assert b"\r" not in raw                                   # Windows で書いても LF のみ (bash が壊れない)
    text = raw.decode()
    assert text.startswith("#!/bin/bash\n") and "source /opt/openfoam14/etc/bashrc" in text
    assert f"cd {tmp_path}" in text and text.rstrip().endswith("checkMesh > log 2>&1")
    assert FoamEnvironment.from_dict(wsl.to_dict()) == wsl
    assert default_environment().kind in (KIND_NATIVE, KIND_WSL)


def test_missing_openfoam_is_reported(tmp_path):
    env = FoamEnvironment(kind=KIND_NATIVE, bashrc="/nonexistent/openfoam/etc/bashrc")
    with pytest.raises(JobError, match="OpenFOAM が見つかりません"):
        env.probe(tmp_path)
    with pytest.raises(JobError, match="OpenFOAM が見つかりません"):
        check_environment_job({"environment": env.to_dict(), "work_dir": str(tmp_path)})


# ----------------------------------------------------------------------------- ケース生成

def _entry(text: str, patch: str) -> str:
    return re.search(rf"\n    {patch}\n    \{{(.*?)\n    \}}", text, re.S).group(1)


def test_boundary_conditions_per_patch():
    fields = case.field_files(PATCHES, 55.5556, 0.46, 3000.0, moving_ground=True)
    u = fields["U"]
    assert "dimensions      [velocity];" in u and "internalField   uniform (55.5556 0 0);" in u
    assert "fixedValue" in _entry(u, "inlet") and "(55.5556 0 0)" in _entry(u, "inlet")
    assert "inletOutlet" in _entry(u, "outlet")
    assert all("slip" in _entry(u, p) for p in ("sideL", "sideR", "top"))
    assert "noSlip" in _entry(u, "Body") and "noSlip" in _entry(u, "RearWing")
    assert "fixedValue" in _entry(u, "ground") and "(55.5556 0 0)" in _entry(u, "ground")    # Moving Ground
    still = case.field_files(PATCHES, 55.5556, 0.46, 3000.0, moving_ground=False)["U"]
    assert "noSlip" in _entry(still, "ground")
    p = fields["p"]
    assert "[kinematicPressure]" in p and "zeroGradient" in _entry(p, "inlet")
    assert "fixedValue" in _entry(p, "outlet") and "zeroGradient" in _entry(p, "Body")
    assert "kqRWallFunction" in _entry(fields["k"], "Body") and "fixedValue" in _entry(fields["k"], "inlet")
    assert "omegaWallFunction" in _entry(fields["omega"], "ground") and "3000" in _entry(fields["omega"], "inlet")
    assert "nutUSpaldingWallFunction" in _entry(fields["nut"], "RearWing") and "calculated" in _entry(fields["nut"], "outlet")
    # 地面なし (単体パーツ風洞) の底面はすべり壁
    bottom = [dict(p, name="bottom", type="patch") if p["name"] == "ground" else p for p in PATCHES]
    assert "slip" in _entry(case.field_files(bottom, 10.0, 0.1, 10.0, False)["U"], "bottom")
    with pytest.raises(JobError, match="境界条件を決められない"):
        case.field_files([{"name": "mystery", "type": "patch", "kind": "tunnel"}], 1, 1, 1, False)


def test_dictionaries():
    functions = case.functions_dict(PATCHES, 1.2041)
    assert "patches         (Body RearWing);" in functions and "rhoInf          1.2041;" in functions
    assert "forces_Body" in functions and "forces_RearWing" in functions          # パーツ別の荷重
    assert "residuals(name=residuals, fields=(p U k omega))" in functions
    assert "patchFlowRate(name=flowInlet, patch=inlet)" in functions and "stopAtFile" in functions
    assert "action          writeNow;" in functions and "#includeFunc yPlus" in functions
    single = case.functions_dict(PATCHES[:7], 1.2)
    assert "forces_Body" not in single                                            # 1 パーツなら全体と同じなので出さない
    with pytest.raises(JobError):
        case.functions_dict(PATCHES[:6], 1.2)
    control = case.control_dict(500)
    assert "solver          incompressibleFluid;" in control and "endTime         500;" in control
    assert "startFrom       latestTime;" in control
    solution = case.fv_solution(SolverSettings(relax_p=0.25, relax_u=0.6), 2)
    assert "nNonOrthogonalCorrectors 2;" in solution and "p               0.25;" in solution and "U               0.6;" in solution
    schemes = case.fv_schemes()
    assert "steadyState" in schemes and "bounded Gauss linearUpwind limited" in schemes
    assert "numberOfSubdomains 4;" in case.decompose_par_dict(4)


def test_write_case_fresh_and_continue(tmp_path):
    mesh_dir = tmp_path / "mesh"
    (mesh_dir / "polyMesh").mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        (mesh_dir / "polyMesh" / name).write_text(name)
    case_dir = tmp_path / "case"
    info = case.write_case(case_dir, mesh_dir, PATCHES, FLOW, True, SolverSettings(iterations=300), 65.0, 4)
    assert info == {"start_time": 0, "end_time": 300, "non_ortho_correctors": 1, "continued": False}
    for rel in ("system/controlDict", "system/fvSchemes", "system/fvSolution", "system/functions", "system/decomposeParDict",
                "constant/momentumTransport", "constant/physicalProperties", "constant/polyMesh/points", "0/U", "0/p", "0/k", "0/omega", "0/nut"):
        assert (case_dir / rel).is_file(), rel
    assert "kOmegaSST" in (case_dir / "constant/momentumTransport").read_text()
    assert "nu              1.5e-05;" in (case_dir / "constant/physicalProperties").read_text()
    assert b"\r" not in (case_dir / "system/controlDict").read_bytes()

    # 前回 300 反復まで計算済みの状態を模擬
    (case_dir / "300").mkdir()
    (case_dir / "300" / "U").write_text("solved")
    (case_dir / "postProcessing" / "forces" / "0").mkdir(parents=True)
    (case_dir / "0" / "U").write_text("keep-me")
    again = case.write_case(case_dir, mesh_dir, PATCHES, FLOW, True, SolverSettings(iterations=200, continue_run=True), 65.0, 4)
    assert again["start_time"] == 300 and again["end_time"] == 500 and again["continued"]
    assert (case_dir / "300" / "U").read_text() == "solved" and (case_dir / "0" / "U").read_text() == "keep-me"
    assert (case_dir / "postProcessing" / "forces" / "0").is_dir()                 # 履歴は残す
    assert "endTime         500;" in (case_dir / "system/controlDict").read_text()
    # 「続きから」を外すと最初からやり直す
    fresh = case.write_case(case_dir, mesh_dir, PATCHES, FLOW, True, SolverSettings(iterations=200), 65.0, 1)
    assert fresh["start_time"] == 0 and not (case_dir / "300").exists() and not (case_dir / "postProcessing").exists()
    with pytest.raises(JobError, match="polyMesh"):
        case.write_case(tmp_path / "c2", tmp_path / "nomesh", PATCHES, FLOW, True, SolverSettings(), 65.0, 1)


# ----------------------------------------------------------------------------- 出力の解析

FORCE_LINE = ("30              \t((8.79627367e+01 1.23370792e+01 3.46471294e+01) (7.51332652e+00 2.41153610e-02 4.10333399e-02)) "
              "((-1.96932175e+00 -2.30676839e+00 1.18917803e+00) (-5.24103939e-03 1.54022011e+00 -1.17539295e-02))")


def test_parse_real_openfoam_formats(tmp_path):
    row = monitor.parse_forces_line(FORCE_LINE)
    assert row[0] == 30 and row[1:4] == pytest.approx([87.9627367, 12.3370792, 34.6471294])
    assert row[4:7] == pytest.approx([7.51332652, 0.0241153610, 0.0410333399]) and len(row) == 13
    assert monitor.parse_forces_line("# Time          forces(pressure viscous)") is None

    names, rows = monitor.parse_residuals_text(
        "# Residuals     \n# Time          \tp               \tUx              \tUy\n"
        "0               \tN/A             \tN/A             \tN/A\n"
        "1               \t1.00000000e+00\t9.99996820e-01\t1.00000000e+00\n"
        "2               \t3.75673434e-01\t4.99261966e-01\t4.96194283e-01\n")
    assert names == ["p", "Ux", "Uy"] and rows.shape == (3, 4)
    assert np.isnan(rows[0, 1]) and rows[2, 1] == pytest.approx(0.375673434)

    # 続きから計算したときは開始時刻ごとのフォルダをつなぐ (重なる時刻は後の実行を採用)
    post = tmp_path / "postProcessing"
    for start, times, scale in (("0", (0, 1, 2, 3), 1.0), ("3", (3, 4, 5), 2.0)):
        folder = post / "forces" / start
        folder.mkdir(parents=True)
        lines = ["# Forces"] + [f"{t}\t(({scale * t} 0 {-scale * t}) (0.5 0 0)) ((0 0 0) (0 0 0))" for t in times]
        (folder / "forces.dat").write_text("\n".join(lines) + "\n")
    forces = monitor.read_forces(tmp_path)
    assert forces[:, 0].tolist() == [1, 2, 3, 4, 5]                    # 時刻 0 は除く
    assert forces[:, 1].tolist() == [1.0, 2.0, 6.0, 8.0, 10.0]         # 時刻 3 は 2 回目の実行の値
    assert monitor.read_forces(tmp_path, "nothing").shape == (0, 13)

    folder = post / "yPlus" / "0"
    folder.mkdir(parents=True)
    (folder / "yPlus.dat").write_text("# y+ ()\n# Time\tpatch\tmin\tmax\taverage\n0\tBody\t1\t2\t1.5\n500\tground\t0.5\t1382\t205\n500\tBody\t31\t1792\t735\n")
    assert monitor.read_yplus(tmp_path) == {"Body": {"min": 31.0, "max": 1792.0, "average": 735.0},
                                            "ground": {"min": 0.5, "max": 1382.0, "average": 205.0}}
    folder = post / "flowInlet" / "0"
    folder.mkdir(parents=True)
    (folder / "surfaceFieldValue.dat").write_text("# Time\tsum(phi)\n1\t-2.25e+02\n2\t-2.254e+02\n")
    assert monitor.read_flow_rate(tmp_path, "flowInlet")[-1].tolist() == [2.0, -225.4]


def test_log_analysis():
    banner = "sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE).\nStarting time loop\nTime = 1s\n"
    assert monitor.find_fatal_error(banner) is None                    # 起動時の通常行を誤検出しない
    fatal = banner + "\n--> FOAM FATAL ERROR: \n    cannot find patch ghost\n\nFOAM exiting\n"
    assert "cannot find patch ghost" in monitor.find_fatal_error(fatal)
    assert monitor.find_fatal_error(banner + "#3  Foam::sigFpe::sigHandler(int)\nFloating point exception\n")
    assert monitor.stopped_by_file("stopAtFile(timeIndex=79): stopping now after writing\nEnd\n")
    assert not monitor.stopped_by_file("    Selecting stopAtFile with name stopAtFile\n")
    assert monitor.reached_residual_control("SIMPLE solution converged in 212 iterations\n")
    assert not monitor.reached_residual_control("SIMPLE: Convergence criteria found\n")
    line = "time step continuity errors : sum local = 2.59e-05, global = -3.35e-07, cumulative = 0.00145\n"
    assert monitor.last_continuity_error("x\n" + line) == {"sum_local": 2.59e-05, "global": -3.35e-07, "cumulative": 0.00145}


def test_file_tail_reads_only_complete_new_lines(tmp_path):
    path = tmp_path / "forces.dat"
    tail = monitor.FileTail(path)
    assert tail.read_new_lines() == []                                 # まだファイルが無い
    path.write_text("# header\n1 a\n2 b")                              # 最後の行は書きかけ
    assert tail.read_new_lines() == ["# header", "1 a"]
    assert tail.read_new_lines() == []
    with open(path, "a") as fh:
        fh.write("b\n3 c\n")
    assert tail.read_new_lines() == ["2 bb", "3 c"]


# ----------------------------------------------------------------------------- 収束判定

def _series(n, final, amplitude=0.0, drift=0.0, noise=0.0):
    i = np.arange(n)
    rng = np.random.default_rng(1)
    return final * (1 + amplitude * np.exp(-i / 40.0) + drift * (i - n) / n) + noise * rng.standard_normal(n)


GOOD = {"p": 5e-4, "Ux": 5e-5, "k": 3e-4, "omega": 4e-4}


def test_convergence_levels():
    drag, lift = _series(500, 100.0, 0.5), _series(500, -300.0, 0.4)
    ok = cv.assess(drag, lift, GOOD, 1e-6)
    assert ok["level"] == cv.LEVEL_CONVERGED and ok["window"] == 100 and all(c["ok"] for c in ok["checks"])

    # 荷重は安定しているが残差が 1e-3 まで下がっていない → 「概ね収束」(完全な収束とは区別する)
    mostly = cv.assess(drag, lift, {**GOOD, "p": 3e-3}, 1e-6)
    assert mostly["level"] == cv.LEVEL_MOSTLY and "残差" in mostly["message"]

    # 指定反復数を回り切っていても、荷重がまだ一方向に動いていれば未収束
    drifting = cv.assess(_series(500, 100.0, drift=0.2), lift, GOOD, 1e-6)
    assert drifting["level"] == cv.LEVEL_NOT_CONVERGED
    assert next(c for c in drifting["checks"] if c["name"] == "抗力の安定")["ok"] is False

    # 荷重が振動している → 未収束
    oscillating = cv.assess(100.0 + 3.0 * np.sin(np.arange(500) / 5.0), lift, GOOD, 1e-6)
    assert oscillating["level"] == cv.LEVEL_NOT_CONVERGED

    # 残差が高い / 質量収支が悪い → 未収束
    assert cv.assess(drag, lift, {**GOOD, "p": 5e-2}, 1e-6)["level"] == cv.LEVEL_NOT_CONVERGED
    assert cv.assess(drag, lift, GOOD, 5e-3)["level"] == cv.LEVEL_NOT_CONVERGED

    # 反復数が少なすぎる → 判定不能 (収束とも未収束とも言わない)
    few = cv.assess(drag[:80], lift[:80], GOOD, 1e-6)
    assert few["level"] == cv.LEVEL_UNDETERMINED and few["checks"][0]["ok"] is None

    # 非数・発散
    bad = drag.copy()
    bad[-1] = np.nan
    assert cv.assess(bad, lift, GOOD, 1e-6)["level"] == cv.LEVEL_DIVERGED
    assert cv.assess(drag, lift, {**GOOD, "p": float("nan")}, 1e-6)["level"] == cv.LEVEL_DIVERGED
    assert cv.assess(drag, lift, GOOD, 1e-6, diverged=True)["level"] == cv.LEVEL_DIVERGED

    # 揚力がほぼ 0 の物体でも、相対変動の分母が 0 にならず判定できる (抗力の 10 % を基準にする)
    zero_lift = cv.assess(drag, 0.05 * np.sin(np.arange(500) / 7.0), GOOD, 1e-6)
    assert zero_lift["level"] == cv.LEVEL_CONVERGED
    # 流量が取得できないときは「十分収束」とは言わない
    assert cv.assess(drag, lift, GOOD, None)["level"] == cv.LEVEL_MOSTLY
    assert cv.window_size(200) == 50 and cv.window_size(1000) == 200 and cv.window_size(30) == 30


# ----------------------------------------------------------------------------- 荷重・係数

def test_force_summary_signs_and_coefficients(tmp_path):
    n = 200
    forces = np.zeros((n, 13))
    forces[:, 0] = np.arange(1, n + 1)
    forces[:, 1:4] = [80.0, 5.0, -400.0]         # 圧力: 抗力 80、横力 5、下向き 400 (= ダウンフォース)
    forces[:, 4:7] = [20.0, 0.0, -2.0]           # 粘性
    rho, speed, area = 1.2, 50.0, 2.0
    s = force_summary(forces, 50, rho, speed, area)
    q = 0.5 * rho * speed ** 2
    assert s["drag"] == pytest.approx(100.0) and s["drag_pressure"] == pytest.approx(80.0) and s["drag_viscous"] == pytest.approx(20.0)
    assert s["lift"] == pytest.approx(-402.0) and s["downforce"] == pytest.approx(402.0)      # ダウンフォース = −揚力
    assert s["side"] == pytest.approx(5.0)
    assert s["cd"] == pytest.approx(100.0 / (q * area)) and s["cl"] == pytest.approx(-402.0 / (q * area))
    assert s["averaged_over"] == 50 and s["drag_range"] == pytest.approx(0.0)
    # 基準面積が無ければ係数は出さない (推測値を入れない)
    none = force_summary(forces, 50, rho, speed, None)
    assert none["cd"] is None and none["cl"] is None and none["drag"] == pytest.approx(100.0)
    with pytest.raises(JobError):
        force_summary(np.zeros((0, 13)), 50, rho, speed, area)

    residuals = np.column_stack([np.arange(1, n + 1), np.geomspace(1, 1e-4, n), np.geomspace(1, 1e-5, n)])
    write_history(tmp_path / "history.csv", ["p", "Ux"], residuals, forces)
    history = read_history(tmp_path / "history.csv")
    assert history["iteration"][-1] == n and history["residuals"]["p"][-1] == pytest.approx(1e-4)
    assert history["drag"][0] == pytest.approx(100.0) and history["lift"][0] == pytest.approx(-402.0)


def test_solver_input_hash_tracks_what_changes_the_flow():
    report = {"input_hash": "meshhash"}
    tunnel, settings = ts.TunnelSpec(), SolverSettings()
    base = solver_input_hash(report, tunnel, settings)
    tunnel.reference_area = 2.0                       # 係数の換算だけ
    settings.n_procs, settings.continue_run = 8, True  # 実行方法だけ
    assert solver_input_hash(report, tunnel, settings) == base
    tunnel.flow.speed_ms = 40.0
    assert solver_input_hash(report, tunnel, settings) != base
    tunnel.flow.speed_ms = 200 / 3.6
    tunnel.moving_ground = False
    assert solver_input_hash(report, tunnel, settings) != base
    tunnel.moving_ground = True
    assert solver_input_hash({"input_hash": "other"}, tunnel, settings) != base
    settings.relax_u = 0.5
    assert solver_input_hash(report, tunnel, settings) != base


# ----------------------------------------------------------------------------- 実物の OpenFOAM による解析

@pytest.fixture(scope="module")
def solved(box_step, tmp_path_factory):
    """直方体の車両風洞を粗いメッシュで実際に解析した結果 (モジュール内で 1 回だけ実行)。"""
    if not native_openfoam_available():
        pytest.skip("OpenFOAM がこの環境にありません")
    root = tmp_path_factory.mktemp("solve")
    meta = import_cad_job({"path": str(box_step), "cache_root": str(root / "cache")})
    assembly = Assembly()
    assembly.add_from_meta(meta)
    assembly.center_xy()
    tunnel = ts.TunnelSpec()
    ts.apply_mode_defaults(tunnel, assembly.world_bounds())
    tunnel.frontal_area = 1.8 * 1.19
    mesh_settings = MeshSettings(preset="custom", surface_size=0.22, min_size=0.11, far_size=3.0, refine_distance=8.0, wake_length=6.0)
    report = mesh_generate_job(build_mesh_request(assembly, tunnel, mesh_settings, root / "mesh"))
    settings = SolverSettings(iterations=150, n_procs=2)
    ctx = RecordingContext()
    request = build_solver_request(report, tunnel, settings, default_environment(), root)
    summary = solver_run_job(request, ctx)
    return {"root": root, "report": report, "tunnel": tunnel, "settings": settings, "summary": summary, "ctx": ctx, "request": request}


@needs_openfoam
def test_environment_probe(tmp_path):
    info = check_environment_job({"environment": default_environment().to_dict(), "work_dir": str(tmp_path)})
    assert info["version"] == "14" and info["version_ok"] and info["has_mpirun"]
    assert info["logical_cores"] >= info["physical_cores"] >= 1
    assert not (tmp_path / ".ljmcfd_step.sh").exists()


@needs_openfoam
def test_openfoam_checkmesh_agrees_with_own_quality_metrics(solved):
    """Stage 2 の polyMesh を OpenFOAM が正常と判定し、自前の品質指標が checkMesh と一致する。"""
    log = (solved["root"] / "case" / "log.checkMesh").read_text()
    quality = solved["report"]["quality"]
    assert "Mesh OK." in log and solved["summary"]["mesh"]["checkmesh"]["ok"]
    assert "Upper triangular ordering OK." in log and "Boundary definition OK." in log
    number = lambda pattern: float(re.search(pattern, log).group(1).rstrip("."))   # 文末のピリオドは数値に含めない
    assert int(number(r"cells:\s+(\d+)")) == quality["n_cells"]
    assert int(number(r"internal faces:\s+(\d+)")) == quality["n_internal_faces"]
    assert number(r"non-orthogonality Max: ([\d.eE+-]+)") == pytest.approx(quality["non_ortho_max"], rel=1e-6)
    assert number(r"non-orthogonality Max: [\d.eE+-]+ average: ([\d.eE+-]+)") == pytest.approx(quality["non_ortho_mean"], rel=1e-6)
    assert number(r"Max skewness = ([\d.eE+-]+)") == pytest.approx(quality["skewness_max"], rel=1e-6)
    assert number(r"Total volume = ([\d.eE+-]+)") == pytest.approx(quality["volume_total"], rel=1e-6)
    assert number(r"Min volume = ([\d.eE+-]+)") == pytest.approx(quality["volume_min"], rel=1e-6)


@needs_openfoam
def test_real_analysis_results_are_physical(solved):
    summary, flow = solved["summary"], solved["request"]["flow"]
    assert summary["status"] == STATUS_COMPLETED and summary["error"] is None
    assert summary["iterations"] == {"start": 0, "requested_end": 150, "done": 150}
    assert summary["openfoam"] == {"version": "14", "kind": "native", "n_procs": 2}
    f = summary["forces"]
    q = 0.5 * flow["rho"] * flow["speed"] ** 2
    # 接地した直方体: 抗力は流れ方向に正、大半が圧力抗力。Cd は鈍頭物体として 0.5〜2 の範囲
    assert f["drag"] > 0 and f["drag_pressure"] > 5 * abs(f["drag_viscous"])
    assert f["drag"] == pytest.approx(f["drag_pressure"] + f["drag_viscous"])
    assert f["downforce"] == pytest.approx(-f["lift"])
    assert f["cd"] == pytest.approx(f["drag"] / (q * 1.8 * 1.19)) and 0.5 < f["cd"] < 2.0
    assert f["cl"] == pytest.approx(f["lift"] / (q * 1.8 * 1.19))
    assert f["dynamic_pressure"] == pytest.approx(q)
    # 質量保存: 流入 = U∞ × 流入面積、流出と釣り合う
    d = solved["report"]["domain"]
    inlet_area = (d[3] - d[2]) * (d[5] - d[4])
    assert summary["mass"]["inlet"] == pytest.approx(-flow["speed"] * inlet_area, rel=1e-6)
    assert summary["mass"]["imbalance"] < 1e-4
    # よどみ点の圧力係数は 1 前後、最大流速は主流より速い (側面の加速)。
    # このテストのメッシュは極端に粗く (表面 220 mm)、動く地面と静止した物体が接する前縁下端の角で
    # 圧力が行き過ぎるため、Cp の最大は 1 をやや超える。その場合は結果に注意書きが付く。
    r = summary["ranges"]
    assert 0.9 < r["cp_max"] < 2.0 and r["cp_min"] < 0
    assert (r["cp_max"] > 1.2) == any("Cp" in note for note in summary["notes"])
    assert flow["speed"] < r["speed_max"] < 3 * flow["speed"]
    assert r["pressure_max"] == pytest.approx(r["cp_max"] * q, rel=1e-6)
    # y+ は実測値を壁面ごとに取得できている (境界層なしの粗い四面体では非常に大きい)
    assert set(summary["yplus"]) == {"ground", "box_mm"} and summary["yplus"]["box_mm"]["average"] > 300
    # 収束判定は必ず付く。150 反復・粗いメッシュなので内容は問わないが、発散ではない
    assert summary["convergence"]["level"] in (cv.LEVEL_CONVERGED, cv.LEVEL_MOSTLY, cv.LEVEL_NOT_CONVERGED)
    assert {c["name"] for c in summary["convergence"]["checks"]} == {"抗力の安定", "揚力の安定", "残差", "質量収支"}
    assert set(summary["final_residuals"]) == {"p", "Ux", "Uy", "Uz", "k", "omega"}


@needs_openfoam
def test_real_analysis_outputs_and_live_events(solved):
    import pyvista as pv

    summary, ctx, root = solved["summary"], solved["ctx"], solved["root"]
    results = root / "results"
    assert load_summary(results)["input_hash"] == solved["request"]["input_hash"]
    assert not (root / "results.partial").exists() and not (root / "case" / ".ljmcfd_solver.sh").exists()
    history = read_history(results / "history.csv")
    assert history["iteration"].tolist() == list(range(1, 151))
    assert history["residuals"]["p"][-1] < history["residuals"]["p"][0]            # 残差は下がっている
    assert history["drag"][-1] == pytest.approx(summary["forces"]["drag"], rel=0.05)
    fields = pv.read(results / summary["files"]["fields"])
    assert fields.n_cells == solved["report"]["quality"]["n_cells"]
    assert {"U", "p", "k", "omega", "nut", "Umag", "Cp", "pressure", "vorticity", "vorticity_mag", "Q"} <= set(fields.cell_data.keys())
    assert set(summary["files"]["surfaces"]) == {"ground", "box_mm"}
    body = pv.read(results / summary["files"]["surfaces"]["box_mm"])
    assert body.n_cells > 0 and {"pressure", "Cp", "wallShearStress", "yPlus"} <= set(body.cell_data.keys())
    # 実行中に残差・荷重が逐次送られ、進捗が反復数で報告される (GUI のグラフ更新に使う)
    names = [name for name, _ in ctx.events]
    assert "residuals" in names and "forces" in names
    sent = sorted(i for name, data in ctx.events if name == "forces" for i in data["iteration"])
    assert sent == list(range(1, 151))
    residual_event = next(data for name, data in ctx.events if name == "residuals")
    assert set(residual_event["values"]) == {"p", "Ux", "Uy", "Uz", "k", "omega"}
    assert any("反復" in msg and "/ 150" in msg for _, msg in ctx.progress_events)
    assert ctx.progress_events[-1] == (1.0, "解析完了")
    assert any("mpirun" in line and "-np 2" in line for line in ctx.logs)


@needs_openfoam
def test_continue_stop_and_residual_stop(solved):
    """続きから計算・途中停止・残差による打ち切りを、同じケースで順に確かめる。"""
    root, report, tunnel = solved["root"], solved["report"], solved["tunnel"]
    env = default_environment()

    more = SolverSettings(iterations=60, n_procs=2, continue_run=True)
    summary = solver_run_job(build_solver_request(report, tunnel, more, env, root))
    assert summary["status"] == STATUS_COMPLETED
    assert summary["iterations"] == {"start": 150, "requested_end": 210, "done": 210}
    assert read_history(root / "results" / "history.csv")["iteration"].tolist() == list(range(1, 211))
    assert summary["mesh"]["checkmesh"] is None                 # 続きからの場合はメッシュ検査をやり直さない

    # 停止: ケースフォルダに stop ファイルを置くと、その時点の結果を書いて終了する
    long_run = SolverSettings(iterations=100_000, n_procs=1)
    request = build_solver_request(report, tunnel, long_run, env, root)
    threading.Timer(6.0, lambda: (Path(request["case_dir"]) / "stop").write_text("")).start()
    stopped = solver_run_job(request)
    assert stopped["status"] == STATUS_STOPPED and "停止" in stopped["status_label"]
    assert 0 < stopped["iterations"]["done"] < 100_000 and stopped["iterations"]["start"] == 0
    assert stopped["forces"] is not None and stopped["files"]["fields"] == "fields.vtu"     # 途中結果も読める
    assert not (Path(request["case_dir"]) / "stop").exists()

    # 残差の打ち切り基準を緩くすると、指定反復数より前に「残差到達」で終わる
    loose = SolverSettings(iterations=400, n_procs=1, residual_tolerance=0.05)
    early = solver_run_job(build_solver_request(report, tunnel, loose, env, root))
    assert early["status"] == STATUS_RESIDUAL_STOP and early["iterations"]["done"] < 400
    # 残差で終わっても、それだけで「収束」とはしない (荷重の安定を別に確認する)
    assert early["convergence"]["level"] != cv.LEVEL_CONVERGED


@needs_openfoam
def test_solver_failure_is_reported_without_fake_numbers(solved, tmp_path):
    """境界条件の欠けたケースでソルバーを異常終了させる → 計算失敗として記録し、荷重は出さない。"""
    request = build_solver_request(solved["report"], solved["tunnel"], SolverSettings(iterations=20, n_procs=1),
                                   default_environment(), tmp_path)
    # 「top」パッチの境界条件を書かせない (実際のメッシュには存在するので OpenFOAM が致命的エラーを出す)
    request["mesh"] = dict(request["mesh"], patches=[p for p in request["mesh"]["patches"] if p["name"] != "top"])
    ctx = RecordingContext()
    summary = solver_run_job(request, ctx)
    assert summary["status"] == STATUS_FAILED and summary["status_label"] == "計算失敗"
    assert summary["forces"] is None and summary["ranges"] is None and summary["parts"] == []
    assert summary["convergence"]["level"] == cv.LEVEL_DIVERGED
    assert "FOAM FATAL" in summary["error"] and "top" in summary["error"]
    assert load_summary(tmp_path / "results")["status"] == STATUS_FAILED
    assert any("異常終了" in line for line in ctx.logs)
    assert ctx.progress_events[-1] == (1.0, "解析失敗")
