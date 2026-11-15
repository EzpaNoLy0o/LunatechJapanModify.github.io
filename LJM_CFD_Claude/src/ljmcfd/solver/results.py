"""解析結果の集計: 荷重・係数の算出と、可視化用データ (VTK) の整理。

符号の約束 (風は +X、上は +Z):

* Drag (抗力)        = +X 方向の力
* Lift (揚力)        = +Z 方向の力
* Downforce          = −Lift (下向きを正とした値)
* Side Force (横力)  = +Y 方向の力

車両にヨー角を付けた場合も、力は風洞座標 (風向基準) で表す。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..core.jobs import JobError

SUMMARY_FILE = "summary.json"
HISTORY_FILE = "history.csv"
FIELDS_FILE = "fields.vtu"
SURFACE_DIR = "surface"
SUMMARY_VERSION = 1


def force_summary(forces: np.ndarray, window: int, rho: float, speed: float, area: float | None) -> dict:
    """荷重履歴 (N, 13) の直近 ``window`` 反復の平均から、荷重と係数を求める。

    係数は ``F / (0.5 ρ U² A)``。基準面積が無ければ係数は ``None`` (推測値を入れない)。
    """
    if len(forces) == 0:
        raise JobError("荷重の履歴がありません")
    tail = forces[-max(window, 1):]
    pressure, viscous = tail[:, 1:4], tail[:, 4:7]
    total = pressure + viscous
    mean_p, mean_v, mean_t = pressure.mean(axis=0), viscous.mean(axis=0), total.mean(axis=0)
    dynamic_pressure = 0.5 * rho * speed ** 2

    def coefficient(force: float) -> float | None:
        return None if not area else float(force / (dynamic_pressure * area))

    return {
        "averaged_over": int(len(tail)),
        "drag": float(mean_t[0]), "side": float(mean_t[1]), "lift": float(mean_t[2]), "downforce": float(-mean_t[2]),
        "drag_pressure": float(mean_p[0]), "drag_viscous": float(mean_v[0]),
        "lift_pressure": float(mean_p[2]), "lift_viscous": float(mean_v[2]),
        "drag_range": float(np.ptp(total[:, 0])), "lift_range": float(np.ptp(total[:, 2])),
        "cd": coefficient(mean_t[0]), "cl": coefficient(mean_t[2]), "cs": coefficient(mean_t[1]),
        "reference_area": float(area) if area else None,
        "dynamic_pressure": float(dynamic_pressure),
    }


def write_history(path: str | Path, residual_names: list[str], residuals: np.ndarray, forces: np.ndarray) -> None:
    """反復ごとの残差と荷重を CSV に書く (グラフの再表示・外部ツールでの確認用)。"""
    res = {int(round(row[0])): row[1:] for row in residuals}
    frc = {int(round(row[0])): row for row in forces}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", *[f"residual_{n}" for n in residual_names],
                         "Fx_pressure", "Fy_pressure", "Fz_pressure", "Fx_viscous", "Fy_viscous", "Fz_viscous"])
        for iteration in sorted(set(res) | set(frc)):
            r = res.get(iteration)
            f = frc.get(iteration)
            writer.writerow([
                iteration,
                *(["" for _ in residual_names] if r is None else [f"{v:.8e}" for v in r]),
                *(["" for _ in range(6)] if f is None else [f"{v:.8e}" for v in f[1:7]]),
            ])


def read_history(path: str | Path) -> dict:
    """``write_history`` の CSV を読む。{"iteration", "residuals": {名前: 配列}, "drag", "lift", "side"}"""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[float(v) if v != "" else np.nan for v in row] for row in reader]
    data = np.array(rows, dtype=float).reshape(-1, len(header))
    column = {name: data[:, i] for i, name in enumerate(header)}
    names = [h[len("residual_"):] for h in header if h.startswith("residual_")]
    return {
        "iteration": column["iteration"],
        "residuals": {n: column[f"residual_{n}"] for n in names},
        "drag": column["Fx_pressure"] + column["Fx_viscous"],
        "side": column["Fy_pressure"] + column["Fy_viscous"],
        "lift": column["Fz_pressure"] + column["Fz_viscous"],
    }


def _add_derived(dataset, rho: float, speed: float) -> None:
    """可視化で使う派生量を追加する: 流速の大きさ、圧力 [Pa]、圧力係数 Cp、渦度の大きさ。

    OpenFOAM の非圧縮ソルバーの ``p`` は密度で割った圧力 [m²/s²]。基準圧 (流出面) からのゲージ圧。
    """
    for data in (dataset.cell_data, dataset.point_data):
        if "U" in data:
            data["Umag"] = np.linalg.norm(np.asarray(data["U"]), axis=1)
        if "p" in data:
            p = np.asarray(data["p"], dtype=float)
            data["pressure"] = p * rho
            data["Cp"] = p / (0.5 * speed ** 2)
        if "vorticity" in data:
            data["vorticity_mag"] = np.linalg.norm(np.asarray(data["vorticity"]), axis=1)


def _save_uncompressed(dataset, path: Path) -> None:
    """VTK XML 形式 (バイナリ・無圧縮) で保存する。"""
    import vtk

    writer = vtk.vtkXMLUnstructuredGridWriter() if dataset.IsA("vtkUnstructuredGrid") else vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(dataset)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.SetCompressorTypeToNone()
    if not writer.Write():
        raise JobError(f"結果ファイルを書き込めません: {path}")


def package_fields(case_dir: str | Path, out_dir: str | Path, patches: list[dict], rho: float, speed: float) -> dict:
    """foamToVTK の出力を読み、派生量を加えて可視化用ファイルにまとめる。

    戻り値: {"files": {...}, "ranges": {"speed_max", "pressure_min", "pressure_max", …}}
    """
    import pyvista as pv

    case_dir, out_dir = Path(case_dir), Path(out_dir)
    vtk_dir = case_dir / "VTK"
    volumes = sorted(vtk_dir.glob("*.vtk"))
    if not volumes:
        raise JobError("VTK への変換結果が見つかりません (foamToVTK の出力がありません)")
    volume = pv.read(volumes[-1])
    _add_derived(volume, rho, speed)
    cells = {name: np.asarray(volume.cell_data[name]) for name in ("Umag", "pressure", "Cp") if name in volume.cell_data}
    field_names = sorted(volume.point_data.keys())
    # 可視化では節点値 (滑らかに補間できる) だけを使う。セル値を落として無圧縮で書くと、
    # ファイルの読み込みが速くなり GUI の待ち時間が短くなる。
    volume.cell_data.clear()
    _save_uncompressed(volume, out_dir / FIELDS_FILE)

    surface_dir = out_dir / SURFACE_DIR
    surface_dir.mkdir(parents=True, exist_ok=True)
    surfaces = {}
    wall_pressure = []
    for patch in patches:
        if patch.get("type") != "wall":
            continue
        files = sorted((vtk_dir / patch["name"]).glob("*.vtk"))
        if not files:
            continue
        surface = pv.read(files[-1])
        _add_derived(surface, rho, speed)
        _save_uncompressed(surface, surface_dir / f"{patch['name']}.vtp")
        surfaces[patch["name"]] = f"{SURFACE_DIR}/{patch['name']}.vtp"
        if patch.get("kind") == "part" and "pressure" in surface.cell_data and surface.n_cells:
            wall_pressure.append(np.asarray(surface.cell_data["pressure"]))

    ranges = {
        "speed_max": float(np.max(cells["Umag"])) if "Umag" in cells else None,
        "pressure_min": float(np.min(cells["pressure"])) if "pressure" in cells else None,
        "pressure_max": float(np.max(cells["pressure"])) if "pressure" in cells else None,
        "cp_min": float(np.min(cells["Cp"])) if "Cp" in cells else None,
        "cp_max": float(np.max(cells["Cp"])) if "Cp" in cells else None,
        "surface_pressure_min": float(min(a.min() for a in wall_pressure)) if wall_pressure else None,
        "surface_pressure_max": float(max(a.max() for a in wall_pressure)) if wall_pressure else None,
    }
    return {
        "files": {"fields": FIELDS_FILE, "surfaces": surfaces},
        "ranges": ranges,
        "fields": field_names,
    }
