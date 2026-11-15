"""CFD 解析ジョブ: ケース生成 → メッシュ検査 → 初期化 → ソルバー実行 (監視つき) → 変換 → 集計。

出力:

* ``case_dir``     OpenFOAM のケース (再計算・続きからの計算に使う)
* ``results_dir``  ``summary.json`` (荷重・係数・収束判定) / ``history.csv`` (残差・荷重の履歴) /
                   ``fields.vtu`` と ``surface/*.vtp`` (可視化用)

ソルバーが異常終了・発散した場合もジョブ自体は正常終了し、``summary.json`` に
``status = "failed"`` と原因を記録する (荷重の値は入れない)。OpenFOAM が見つからない等、
解析を始められない問題だけを ``JobError`` にする。
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..core.jobs import JobContext, JobError
from . import monitor
from .case import FORCES_ALL, STOP_FILE, forces_name, part_patches, write_case
from .convergence import LEVEL_DIVERGED, assess, window_size
from .environment import STEP_SCRIPT, FoamEnvironment, decode_output
from .results import HISTORY_FILE, SUMMARY_FILE, SUMMARY_VERSION, force_summary, package_fields, write_history
from .settings import SolverSettings, auto_process_count

STATUS_COMPLETED = "completed"            # 指定反復数を回り切った
STATUS_RESIDUAL_STOP = "residual_stop"    # 残差が打ち切り基準を下回って終了した
STATUS_STOPPED = "stopped"                # ユーザーが停止した (その時点の結果は保存済み)
STATUS_FAILED = "failed"                  # ソルバーが異常終了・発散した
STATUS_LABELS = {
    STATUS_COMPLETED: "計算完了 (指定反復数に到達)",
    STATUS_RESIDUAL_STOP: "計算完了 (残差が打ち切り基準に到達)",
    STATUS_STOPPED: "ユーザーにより停止 (途中結果)",
    STATUS_FAILED: "計算失敗",
}
POLL_SECONDS = 0.4


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _run_step(env: FoamEnvironment, case_dir: Path, ctx: JobContext, title: str, command: str, log: str,
              required: bool = True) -> tuple[bool, str]:
    """OpenFOAM のユーティリティを 1 つ実行する。ログはケースフォルダに残す。"""
    ctx.log(f"{title} を実行中…")
    code, _ = env.run(case_dir, f"{command} > {log} 2>&1")
    log_path = case_dir / log
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    if code == 0:
        return True, text
    message = f"{title} が失敗しました (終了コード {code})。\n{_tail(text)}"
    if required:
        raise JobError(message)
    ctx.log("注意: " + message)
    return False, text


def _mpirun(n_procs: int, info: dict) -> str:
    flags = []
    if info["is_root"]:
        flags.append("--allow-run-as-root")
    if n_procs > info["physical_cores"]:
        flags.append("--oversubscribe")
    return " ".join(["mpirun", *flags, "-np", str(n_procs)])


def _emit_new_rows(ctx: JobContext, residual_tail: monitor.FileTail, forces_tail: monitor.FileTail,
                   state: dict) -> None:
    """前回以降に増えた残差・荷重の行を GUI へ送る。"""
    lines = residual_tail.read_new_lines()
    if lines:
        header = [l for l in lines if l.startswith("#") and "Time" in l]
        if header:
            state["residual_names"] = header[-1][1:].split()[1:]
        names = state.get("residual_names")
        if names:
            _, rows = monitor.parse_residuals_text("# Time " + " ".join(names) + "\n" + "\n".join(lines))
            rows = rows[rows[:, 0] > 0] if len(rows) else rows
            if len(rows):
                ctx.event("residuals", {
                    "iteration": rows[:, 0].tolist(),
                    "values": {n: [None if np.isnan(v) else float(v) for v in rows[:, i + 1]] for i, n in enumerate(names)},
                })
    rows = [r for r in map(monitor.parse_forces_line, forces_tail.read_new_lines()) if r and r[0] > 0]
    if rows:
        data = np.array(rows)
        total = data[:, 1:4] + data[:, 4:7]
        ctx.event("forces", {
            "iteration": data[:, 0].tolist(), "drag": total[:, 0].tolist(),
            "side": total[:, 1].tolist(), "lift": total[:, 2].tolist(),
        })
        state["iteration"] = int(round(data[-1, 0]))


def _mass_imbalance(case_dir: Path) -> tuple[float | None, dict]:
    inlet = monitor.read_flow_rate(case_dir, "flowInlet")
    outlet = monitor.read_flow_rate(case_dir, "flowOutlet")
    if len(inlet) == 0 or len(outlet) == 0 or abs(inlet[-1, 1]) < 1e-30:
        return None, {}
    q_in, q_out = float(inlet[-1, 1]), float(outlet[-1, 1])
    return abs(q_in + q_out) / abs(q_in), {"inlet": q_in, "outlet": q_out}


def accuracy_notes(summary: dict) -> list[str]:
    """結果の精度に関する注意書き。数値をそのまま信用してよいかを判断する材料として結果に添える。"""
    notes = []
    boundary_layer = (summary.get("mesh") or {}).get("boundary_layer") or {}
    if not boundary_layer.get("generated"):
        notes.append("境界層メッシュがありません。粘性抗力・剥離位置・Cd の精度は限定的です (圧力による力の傾向比較向き)。")
    part_yplus = [v["average"] for name, v in (summary.get("yplus") or {}).items() if name != "ground"]
    if part_yplus:
        worst = max(part_yplus)
        if worst > 300:
            notes.append(f"車体表面の y+ の平均が {worst:.0f} で、壁関数の適正範囲 (30〜300) を超えています。壁面せん断の精度が下がります。")
        elif worst < 30 and worst > 5:
            notes.append(f"車体表面の y+ の平均が {worst:.0f} で、壁関数の適正範囲 (30〜300) と粘性底層 (5 以下) の中間にあります。")
    ranges = summary.get("ranges") or {}
    if ranges.get("cp_max") is not None and ranges["cp_max"] > 1.2:
        notes.append(f"圧力係数 Cp の最大値が {ranges['cp_max']:.2f} と 1 を大きく超えています。メッシュが粗い箇所で圧力が"
                     "数値的に行き過ぎている可能性があります (接地部の角など)。メッシュを細かくして確認してください。")
    checkmesh = (summary.get("mesh") or {}).get("checkmesh")
    if checkmesh and not checkmesh.get("ok"):
        notes.append("OpenFOAM の checkMesh がメッシュの問題を報告しています。")
    return notes


def solver_run_job(args: dict, ctx: JobContext | None = None) -> dict:
    """CFD 解析を実行する。戻り値は ``summary.json`` と同内容。

    args:
        case_dir, results_dir, mesh_dir
        mesh         メッシュレポートの抜粋 {"patches", "n_cells", "non_ortho_max", "boundary_layer", "input_hash"}
        flow         {"speed", "nu", "rho", "k", "omega"}  SI 単位
        moving_ground
        reference_area   Cd・Cl の基準面積 [m²] (None なら係数を計算しない)
        settings     SolverSettings.to_dict()
        environment  FoamEnvironment.to_dict()
        input_hash   入力のハッシュ (結果が現在の設定のものか判定するため)
    """
    ctx = ctx or JobContext()
    started = time.perf_counter()
    case_dir, results_dir, mesh_dir = Path(args["case_dir"]), Path(args["results_dir"]), Path(args["mesh_dir"])
    settings = SolverSettings.from_dict(args.get("settings"))
    errors = settings.validate()
    if errors:
        raise JobError(" ".join(errors))
    env = FoamEnvironment.from_dict(args.get("environment"))
    mesh = args["mesh"]
    patches = mesh["patches"]
    flow = {key: float(args["flow"][key]) for key in ("speed", "nu", "rho", "k", "omega")}
    area = args.get("reference_area")
    env.check_case_path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. 環境の確認
    ctx.progress(0.01, "OpenFOAM 環境を確認中…")
    info = env.probe(case_dir)
    if not info["version_ok"]:
        ctx.log(f"注意: OpenFOAM {info['version']} が検出されました。本ソフトは OpenFOAM 14 で検証しています。")
    n_procs = settings.n_procs or auto_process_count(int(mesh["n_cells"]), info["physical_cores"])
    if n_procs > 1 and not info["has_mpirun"]:
        ctx.log("注意: mpirun が見つからないため、並列ではなく 1 プロセスで実行します。")
        n_procs = 1
    parallel = n_procs > 1

    # ------------------------------------------------------------------ 2. ケース生成
    ctx.progress(0.03, "ケースを生成中…")
    case_info = write_case(case_dir, mesh_dir, patches, flow, bool(args.get("moving_ground")), settings,
                           float(mesh["non_ortho_max"]), n_procs)
    start, end = case_info["start_time"], case_info["end_time"]
    ctx.log(
        f"OpenFOAM {info['version']} / {'WSL ' + str(info['distro']) if info['kind'] == 'wsl' else 'ネイティブ'} / "
        f"{n_procs} プロセス / 反復 {start} → {end} / 非直交補正 {case_info['non_ortho_correctors']} 回"
        + (" / 前回の続きから" if case_info["continued"] else "")
    )

    checkmesh = None
    if not case_info["continued"]:
        # -------------------------------------------------------------- 3. メッシュ検査・初期化
        ctx.progress(0.05, "メッシュを検査中 (checkMesh)…")
        _, text = _run_step(env, case_dir, ctx, "checkMesh", "checkMesh", "log.checkMesh")
        failed = [line.strip() for line in text.splitlines() if line.strip().startswith("***")]
        checkmesh = {"ok": "Mesh OK." in text, "messages": failed[:10]}
        if not checkmesh["ok"]:
            ctx.log("注意: checkMesh がメッシュの問題を報告しました: " + " / ".join(failed[:3]))
        # Gmsh の並列メッシャーはセルの並びが空間的にばらばらで、行列の帯幅が非常に大きい。
        # 並べ替えるとキャッシュ効率が上がり、ソルバーが約 3 割速くなる (34 万セルで実測)。
        ctx.progress(0.07, "セル番号を最適化中 (renumberMesh)…")
        _run_step(env, case_dir, ctx, "renumberMesh", "renumberMesh", "log.renumberMesh", required=False)
        if settings.potential_init:
            ctx.progress(0.08, "ポテンシャル流れで初期化中…")
            _run_step(env, case_dir, ctx, "potentialFoam", "potentialFoam", "log.potentialFoam", required=False)
    if parallel:
        ctx.progress(0.10, "領域を分割中…")
        option = "-force -latestTime" if case_info["continued"] else "-force"
        _run_step(env, case_dir, ctx, "decomposePar", f"decomposePar {option}", "log.decomposePar")

    # ------------------------------------------------------------------ 4. ソルバー実行
    solver_command = f"{_mpirun(n_procs, info)} foamRun -parallel" if parallel else "foamRun"
    ctx.progress(0.12, "ソルバーを起動中…")
    ctx.log(f"ソルバーを開始します: {solver_command}")
    solver_started = time.perf_counter()
    process = env.popen(case_dir, f"{solver_command} > log.foamRun 2>&1", name=".ljmcfd_solver.sh")
    post = case_dir / "postProcessing"
    residual_tail = monitor.FileTail(post / "residuals" / str(start) / "residuals.dat")
    forces_tail = monitor.FileTail(post / FORCES_ALL / str(start) / "forces.dat")
    state: dict = {"iteration": start}
    last_reported = start
    try:
        while True:
            finished = process.poll() is not None
            _emit_new_rows(ctx, residual_tail, forces_tail, state)
            if state["iteration"] != last_reported:
                last_reported = state["iteration"]
                fraction = (last_reported - start) / max(end - start, 1)
                ctx.progress(0.12 + 0.70 * min(fraction, 1.0), f"反復 {last_reported} / {end}")
            if finished:
                break
            time.sleep(POLL_SECONDS)
    finally:
        if process.poll() is None:      # ジョブが強制終了される場合などに、ソルバーを残さない
            process.kill()
    solver_output = decode_output(process.stdout.read() if process.stdout else b"")
    solver_seconds = time.perf_counter() - solver_started
    log_path = case_dir / "log.foamRun"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else solver_output
    # OpenFOAM は停止を検知すると stop ファイルを自分で消すので、ログの記述でも判定する
    stopped = (case_dir / STOP_FILE).exists() or monitor.stopped_by_file(log_text)
    (case_dir / STOP_FILE).unlink(missing_ok=True)
    for name in (STEP_SCRIPT, ".ljmcfd_solver.sh"):
        (case_dir / name).unlink(missing_ok=True)

    # ------------------------------------------------------------------ 5. 履歴の読み込みと状態の判定
    forces = monitor.read_forces(case_dir, FORCES_ALL)
    residual_names, residuals = monitor.read_residuals(case_dir)
    fatal = monitor.find_fatal_error(log_text)
    solver_failed = process.returncode != 0 or fatal is not None or len(forces) == 0
    iterations_done = int(round(forces[-1, 0])) if len(forces) else start
    if solver_failed:
        status = STATUS_FAILED
    elif iterations_done >= end:
        status = STATUS_COMPLETED
    elif stopped:
        status = STATUS_STOPPED
    elif monitor.reached_residual_control(log_text):
        status = STATUS_RESIDUAL_STOP
    else:
        # 指定反復数に届かず、停止要求でも残差の打ち切りでもない終了は、原因不明の中断として失敗扱いにする
        status = STATUS_FAILED
        solver_failed = True

    summary: dict = {
        "summary_version": SUMMARY_VERSION,
        "status": status, "status_label": STATUS_LABELS[status],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_hash": args.get("input_hash"),
        "iterations": {"start": start, "requested_end": end, "done": iterations_done},
        "openfoam": {"version": info["version"], "kind": info["kind"], "n_procs": n_procs},
        "flow": flow, "moving_ground": bool(args.get("moving_ground")),
        "mesh": {"n_cells": int(mesh["n_cells"]), "boundary_layer": mesh.get("boundary_layer"),
                 "input_hash": mesh.get("input_hash"), "checkmesh": checkmesh},
        "non_ortho_correctors": case_info["non_ortho_correctors"],
        "seconds_solver": round(solver_seconds, 1),
        "error": None, "forces": None, "parts": [], "convergence": None, "mass": None, "yplus": {}, "notes": [],
        "ranges": None, "files": {}, "continuity": monitor.last_continuity_error(log_text),
    }

    tmp_dir = results_dir.with_name(results_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        if len(forces) or len(residuals):
            write_history(tmp_dir / HISTORY_FILE, residual_names, residuals, forces)
            summary["files"]["history"] = HISTORY_FILE

        if solver_failed:
            detail = fatal or _tail(log_text or solver_output, 15) or f"終了コード {process.returncode}"
            summary["error"] = detail
            summary["convergence"] = assess(np.zeros(0), np.zeros(0), {}, None, diverged=True)
            ctx.log("ソルバーが異常終了しました:\n" + detail)
        else:
            # ---------------------------------------------------------- 6. 集計
            ctx.progress(0.84, "結果を集計中…")
            total = forces[:, 1:4] + forces[:, 4:7]
            final_residuals = {}
            if len(residuals):
                final_residuals = {n: float(v) for n, v in zip(residual_names, residuals[-1, 1:]) if not np.isnan(v)}
            imbalance, flow_rates = _mass_imbalance(case_dir)
            convergence = assess(total[:, 0], total[:, 2], final_residuals, imbalance)
            window = convergence["window"] or window_size(len(forces))
            summary["convergence"] = convergence
            summary["final_residuals"] = final_residuals
            summary["mass"] = {"imbalance": imbalance, **flow_rates}
            summary["forces"] = force_summary(forces, window, flow["rho"], flow["speed"], area)
            for patch in part_patches(patches) if len(part_patches(patches)) > 1 else []:
                part_forces = monitor.read_forces(case_dir, forces_name(patch))
                if len(part_forces):
                    entry = force_summary(part_forces, window, flow["rho"], flow["speed"], area)
                    meta = next(p for p in patches if p["name"] == patch)
                    summary["parts"].append({"patch": patch, "part_id": meta.get("part_id"),
                                             "part_name": meta.get("part_name", patch), **entry})
            summary["yplus"] = monitor.read_yplus(case_dir)

            # ---------------------------------------------------------- 7. VTK への変換
            if parallel:
                ctx.progress(0.88, "並列の結果を統合中…")
                _run_step(env, case_dir, ctx, "reconstructPar", "reconstructPar -latestTime", "log.reconstructPar")
            ctx.progress(0.92, "VTK 形式へ変換中…")
            _run_step(env, case_dir, ctx, "foamToVTK", "foamToVTK -latestTime -noZero", "log.foamToVTK")
            packaged = package_fields(case_dir, tmp_dir, patches, flow["rho"], flow["speed"])
            summary["ranges"] = packaged["ranges"]
            summary["notes"] = accuracy_notes(summary)
            summary["files"].update(packaged["files"])
            summary["fields"] = packaged["fields"]
            if convergence["level"] == LEVEL_DIVERGED:
                summary["status"], summary["status_label"] = STATUS_FAILED, STATUS_LABELS[STATUS_FAILED]
                summary["error"] = "計算が発散しました。"
                summary["forces"], summary["parts"] = None, []

        summary["seconds_total"] = round(time.perf_counter() - started, 1)
        (tmp_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    shutil.rmtree(results_dir, ignore_errors=True)
    tmp_dir.rename(results_dir)
    for name in (STEP_SCRIPT,):
        (case_dir / name).unlink(missing_ok=True)

    summary["results_dir"] = str(results_dir)
    ctx.progress(1.0, "解析完了" if summary["status"] != STATUS_FAILED else "解析失敗")
    if summary["forces"]:
        f = summary["forces"]
        ctx.log(
            f"{summary['status_label']}: {iterations_done} 反復 / {solver_seconds:.0f} 秒 / "
            f"Drag {f['drag']:.2f} N、Downforce {f['downforce']:.2f} N / 収束判定: {summary['convergence']['label']}"
        )
    return summary


def load_summary(results_dir: str | Path) -> dict | None:
    """結果フォルダの ``summary.json`` を読む。無い・壊れている場合は ``None``。"""
    results_dir = Path(results_dir)
    path = results_dir / SUMMARY_FILE
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("summary_version") != SUMMARY_VERSION:
        return None
    summary["results_dir"] = str(results_dir)
    return summary
