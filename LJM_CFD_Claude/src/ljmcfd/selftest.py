"""動作確認 (セルフテスト)。

インストール直後や配布版の確認用に、小さな直方体を使って主要な経路を実際に動かす:

    ライブラリ → 作業フォルダ → STEP 読み込みジョブ → メッシュ生成ジョブ → OpenFOAM 環境 → 短い CFD 解析 → 結果の可視化 → GUI

ジョブは GUI と同じ方法 (子プロセス + 標準出力のプロトコル) で起動するので、配布版では
``ljmcfd-worker`` 実行ファイルの確認にもなる。

    python -m ljmcfd --self-test [--no-gui] [--no-solver]

結果は標準出力と ``workspace/logs/selftest.txt`` に書く。失敗が 1 つでもあれば終了コード 1。
OpenFOAM が無い場合、解析の項目は「スキップ」になる (失敗ではない)。
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

OK, FAILED, SKIPPED = "OK", "失敗", "スキップ"
SOLVER_ITERATIONS = 30


class Skip(Exception):
    """前提が無いので実行しない (失敗ではない)。"""


def run_child_job(name: str, args: dict, work_dir: Path, timeout: float = 900.0) -> dict:
    """ジョブを GUI と同じ方法で子プロセスとして実行し、結果の辞書を返す。"""
    from .core.jobs import job_command, job_environment, parse_protocol_line

    args_path = work_dir / f"selftest_{name}.json"
    args_path.write_text(json.dumps(args, ensure_ascii=False), encoding="utf-8")
    command = job_command(name, str(args_path))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=job_environment(),
                             timeout=timeout, creationflags=flags)
    events = [event for event in map(parse_protocol_line, process.stdout.splitlines()) if event]
    errors = [event for event in events if event["type"] == "error"]
    results = [event for event in events if event["type"] == "result"]
    if process.returncode != 0 or errors or not results:
        message = errors[0]["message"] if errors else (process.stderr.strip()[-800:] or process.stdout.strip()[-800:])
        raise RuntimeError(f"ジョブ {name} が失敗しました (終了コード {process.returncode}): {message}\nコマンド: {' '.join(command)}")
    return results[-1]["data"]


def write_box_step(path: Path, size_mm=(2000.0, 900.0, 500.0)) -> Path:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    from .cad.occ_io import silence_occ_messages

    silence_occ_messages()
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(*size_mm).Shape(), STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP ファイルを書けません: {path}")
    return path


def run_selftest(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    with_gui = "--no-gui" not in argv
    with_solver = "--no-solver" not in argv
    lines: list[str] = []
    rows: list[tuple[str, str, str]] = []
    state: dict = {}

    def say(text: str = "") -> None:
        lines.append(text)
        try:
            print(text, flush=True)
        except (OSError, ValueError, AttributeError):      # コンソールなしの実行ファイルでは標準出力が無い
            pass

    def check(name: str, func) -> bool:
        started = time.perf_counter()
        try:
            detail = func() or ""
            status = OK
        except Skip as exc:
            status, detail = SKIPPED, str(exc)
        except Exception as exc:  # noqa: BLE001 - 確認項目ごとに失敗を記録して先へ進む
            status, detail = FAILED, f"{type(exc).__name__}: {exc}"
            say(traceback.format_exc())
        rows.append((name, status, detail))
        say(f"[{status}] {name} ({time.perf_counter() - started:.1f} 秒) {detail}")
        return status == OK

    from . import __version__
    from .core import paths

    say(f"LJM CFD Claude Edition v{__version__} — 動作確認")
    say(f"{platform.platform()} / Python {platform.python_version()} / 配布版: {'はい' if getattr(sys, 'frozen', False) else 'いいえ'}")
    say(f"実行ファイル: {sys.executable}")
    say()

    def libraries() -> str:
        import gmsh
        import numpy
        import OCP
        import PySide6
        import pyvista
        import scipy
        import vtk

        return (f"PySide6 {PySide6.__version__}、VTK {vtk.vtkVersion.GetVTKVersion()}、PyVista {pyvista.__version__}、"
                f"OCP {getattr(OCP, '__version__', '?')}、Gmsh {gmsh.__version__}、NumPy {numpy.__version__}、SciPy {scipy.__version__}")

    def workspace() -> str:
        root = paths.workspace_root()
        work = paths.temp_dir() / "selftest"
        if work.exists():
            import shutil

            shutil.rmtree(work)
        work.mkdir(parents=True)
        (work / "write_test.txt").write_text("ok", encoding="utf-8")
        state["work"] = work
        return str(root)

    def cad_import() -> str:
        work = state["work"]
        step = write_box_step(work / "selftest_box.step")
        meta = run_child_job("cad-import", {"path": str(step), "cache_root": str(work / "cache"), "force": True}, work)
        state["meta"] = meta
        part = meta["parts"][0]
        if len(meta["parts"]) != 1 or part["n_triangles"] != 12:
            raise RuntimeError(f"直方体が正しく読めていません: {len(meta['parts'])} パーツ / {part['n_triangles']} 三角形")
        return f"直方体 2000 × 900 × 500 mm → {part['n_triangles']} 三角形"

    def mesh() -> str:
        from .cad.assembly import Assembly
        from .mesh.request import build_mesh_request
        from .mesh.settings import MeshSettings
        from .tunnel import spec as tunnel_spec

        if "meta" not in state:
            raise Skip("STEP の読み込みに失敗したため")
        work = state["work"]
        assembly = Assembly()
        assembly.add_from_meta(state["meta"])
        assembly.center_xy()
        assembly.drop_to_ground()
        bounds = assembly.world_bounds()
        tunnel = tunnel_spec.TunnelSpec()
        tunnel_spec.apply_mode_defaults(tunnel, bounds)
        tunnel.generated = True
        tunnel.frontal_area = (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
        settings = MeshSettings()
        settings.apply_preset("coarse", bounds)
        settings.preset = "custom"
        settings.surface_size, settings.min_size, settings.far_size = 0.12, 0.06, 1.6
        settings.refine_distance, settings.wake_length = 4.0, 3.0
        report = run_child_job("mesh-generate", build_mesh_request(assembly, tunnel, settings, work / "mesh"), work)
        report["out_dir"] = str(work / "mesh")
        quality = report["quality"]
        if quality["n_cells"] < 500 or quality["n_negative_volumes"] or not (work / "mesh" / "polyMesh" / "points").is_file():
            raise RuntimeError(f"メッシュが不正です: {quality['n_cells']} セル、負体積 {quality['n_negative_volumes']}")
        state.update(report=report, tunnel=tunnel)
        return f"{quality['n_cells']:,} セル、非直交性 最大 {quality['non_ortho_max']:.1f}°、polyMesh を書き出し"

    def openfoam() -> str:
        from .core.jobs import JobError
        from .solver.environment import default_environment

        environment = default_environment()
        try:
            info = environment.probe(state["work"])
        except JobError as exc:
            raise Skip(f"OpenFOAM を使えません — {exc}") from exc
        state["environment"] = environment
        where = f"WSL ({environment.distro})" if environment.kind == "wsl" else "この環境"
        return f"{where} の OpenFOAM {info.get('version')}、コア数 {info.get('physical_cores', info.get('logical_cores', '?'))}"

    def solver() -> str:
        from .solver.request import build_solver_request
        from .solver.settings import SolverSettings

        if "environment" not in state:
            raise Skip("OpenFOAM が無いため")
        if "report" not in state:
            raise Skip("メッシュ生成に失敗したため")
        if not with_solver:
            raise Skip("--no-solver が指定されたため")
        settings = SolverSettings()
        settings.iterations = SOLVER_ITERATIONS
        settings.n_procs = 2
        request = build_solver_request(state["report"], state["tunnel"], settings, state["environment"], state["work"])
        summary = run_child_job("solver-run", request, state["work"], timeout=1800.0)
        if summary.get("status") not in ("completed", "residual_stop") or summary.get("forces") is None:
            raise RuntimeError(f"解析が完了しませんでした: {summary.get('status_label')} {summary.get('error') or ''}")
        summary["results_dir"] = request["results_dir"]
        state["summary"] = summary
        forces = summary["forces"]
        return (f"{summary['iterations']['done']} 反復、Drag {forces['drag']:.1f} N、Lift {forces['lift']:.1f} N "
                f"(動作確認用の短い計算で、収束した値ではありません)")

    def visualisation() -> str:
        from .post.particles import SteadyPathlines
        from .post.result_set import ResultSet, default_seed

        if "summary" not in state:
            raise Skip("解析結果が無いため")
        summary = state["summary"]
        results = ResultSet(Path(summary["results_dir"]), summary)
        lo, hi = results.axis_range("y")
        cut = results.slice("y", (lo + hi) / 2)
        lines = results.streamlines(default_seed(state["report"]["model_bounds"], ground=True))
        animation = SteadyPathlines(lines, particles_per_line=4)
        n_cut, n_lines = cut.GetNumberOfPoints(), lines.GetNumberOfCells()
        if n_cut == 0 or n_lines == 0 or animation.duration <= 0:
            raise RuntimeError(f"結果から表示を作れません: 断面 {n_cut} 点、流線 {n_lines} 本")
        return f"断面 {n_cut:,} 点、流線 {n_lines} 本、粒子アニメーション {animation.duration:.3f} 秒"

    def smoke() -> str:
        from .post.result_set import ResultSet
        from .smoke.engine import SmokeEngine
        from .smoke.settings import SmokeSettings

        if "summary" not in state:
            raise Skip("解析結果が無いため")
        summary, report = state["summary"], state["report"]
        results = ResultSet(Path(summary["results_dir"]), summary)
        bounds = list(report["model_bounds"])
        bounds[4] -= report.get("sink", 0.0)
        bounds[5] -= report.get("sink", 0.0)
        engine = SmokeEngine()
        engine.set_flow(results.flow, results.free_stream_speed, results.domain)
        settings = SmokeSettings(quality="low")
        settings.emitters = engine.default_emitters(settings, bounds)
        built = engine.build(engine.plan(settings, bounds))
        smoke_field = built.smoke
        density = smoke_field.frame(built.duration)
        x, y, z = smoke_field.grid.centers()
        inside = ((x[None, None, :] > bounds[0] + 0.05) & (x[None, None, :] < bounds[1] - 0.05) & (y[None, :, None] > bounds[2] + 0.05)
                  & (y[None, :, None] < bounds[3] - 0.05) & (z[:, None, None] > max(bounds[4], 0.0) + 0.05) & (z[:, None, None] < bounds[5] - 0.05))
        if density.max() <= 0:
            raise RuntimeError("煙の密度場が空です")
        if inside.any() and density[inside].max() > 0:
            raise RuntimeError("煙が車体 (直方体) の内部に入っています")
        state["smoke_ok"] = True
        return (f"格子 {'×'.join(map(str, smoke_field.grid.dims))}、流跡線 {built.stats.get('tracks', 0)} 本、"
                f"車体の内部に煙なし、{built.stats.get('seconds_total', 0.0):.1f} 秒")

    def gui() -> str:
        if not with_gui:
            raise Skip("--no-gui が指定されたため")
        import numpy as np
        from PySide6.QtWidgets import QApplication

        from .gui import theme
        from .gui.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv[:1])
        app.setStyle("Fusion")
        app.setStyleSheet(theme.STYLESHEET)
        window = MainWindow()
        try:
            window.resize(1280, 800)
            window.show()
            for _ in range(20):
                app.processEvents()
                time.sleep(0.01)
            if "summary" in state:
                window.scene.set_mesh_report(state["report"])          # 車体の位置 (煙の放出源の自動配置に使う)
                window.scene.set_solver_summary(state["summary"])
                window.post_view.update(slice_on=True, particles_on=True)
                for _ in range(10):
                    app.processEvents()
            image = window.viewport.capture_frame()
            if image.shape[0] < 50 or float(np.std(image)) < 1.0:
                raise RuntimeError(f"3D ビューが描画されていません (画像 {image.shape}、濃淡の標準偏差 {np.std(image):.2f})")
            detail = f"ウィンドウと 3D ビュー ({image.shape[1]} × {image.shape[0]}) を描画"
            if "summary" in state:
                layers = window.post_view.layer_actors()
                if not (layers["slice"] and layers["particles"]):
                    raise RuntimeError(f"解析結果のレイヤーを表示できません: {layers}")
                detail += "、断面とスモークを表示"
                if state.get("smoke_ok"):
                    detail += "、" + gui_volume_smoke(app, window)
            return detail
        finally:
            window._modified = False
            window.close()
            app.processEvents()

    def gui_volume_smoke(app, window) -> str:
        """ボリューム煙を GUI で描画し、描画方式と 1 コマの描画時間を返す。"""
        from .gui.post_view import STYLE_VOLUME

        view = window.post_view
        view.update(particles_style=STYLE_VOLUME, slice_on=False)
        deadline = time.perf_counter() + 120.0
        while (view.particle_duration() <= 0 or view.smoke_layer.building) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.02)
        layer = view.smoke_layer
        if layer.error:
            raise RuntimeError(f"ボリューム煙を表示できません: {layer.error}")
        if view.particle_duration() <= 0 or not view.layer_actors()["particles"]:
            raise RuntimeError("ボリューム煙が描画されていません (時間切れ)")
        view.set_particle_time(0.8 * view.particle_duration(), block=True)
        plotter = window.viewport.plotter
        plotter.render()
        started = time.perf_counter()
        for step in range(6):
            view.set_particle_time((0.5 + 0.08 * step) * view.particle_duration(), block=True)
            plotter.render()
        per_frame = (time.perf_counter() - started) / 6 * 1000
        renderer = layer.renderer
        gl = renderer.gl.get("renderer", "不明")
        return f"ボリューム煙を描画 ({'GPU' if renderer.backend == 'gpu' else 'CPU'} レイキャスティング、OpenGL: {gl}、1 コマ {per_frame:.0f} ms)"

    check("ライブラリの読み込み", libraries)
    if check("作業フォルダへの書き込み", workspace):
        check("STEP 読み込みジョブ (子プロセス)", cad_import)
        check("メッシュ生成ジョブ (子プロセス)", mesh)
        check("OpenFOAM 環境", openfoam)
        check(f"CFD 解析 {SOLVER_ITERATIONS} 反復 (子プロセス)", solver)
        check("結果の可視化データ (断面・流線・粒子)", visualisation)
        check("ボリューム煙の密度場", smoke)
    check("GUI と 3D 描画", gui)

    failed = [name for name, status, _detail in rows if status == FAILED]
    skipped = [name for name, status, _detail in rows if status == SKIPPED]
    say()
    if failed:
        say(f"結果: 失敗 {len(failed)} 件 — {'、'.join(failed)}")
    elif skipped:
        say(f"結果: 実行した項目はすべて成功 (スキップ {len(skipped)} 件 — {'、'.join(skipped)})")
    else:
        say("結果: すべて成功")
    try:
        target = paths.log_dir() / "selftest.txt"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        say(f"この内容を保存しました: {target}")
    except OSError as exc:
        say(f"結果を保存できません: {exc}")
    return 1 if failed else 0
