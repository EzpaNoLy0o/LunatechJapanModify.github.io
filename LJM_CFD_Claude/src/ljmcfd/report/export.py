"""解析レポートの書き出し (HTML・CSV・JSON)。

GUI に依存しない。入力は「スナップショット」(モデル・風洞・メッシュ・解析結果を辞書にまとめたもの) で、
ここでは値を計算し直さず、解析ジョブが残した値をそのまま整形する。

約束:

* 計算していない値は「未計算」、解析が失敗した場合は「計算失敗」と書き、推測値や 0 を入れない。
* 「計算の状態」(指定反復数まで計算できたか) と「収束判定」(工学的に十分収束したか) を別の項目として書く。
* 境界層メッシュの有無など、精度に関する注意は必ず載せる。
* 表示中の結果が現在の設定より前の条件で計算したものなら、その旨を先頭に書く。
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
import re
import shutil
import time
from pathlib import Path

NOT_COMPUTED = "未計算"
FAILED = "計算失敗"
REPORT_VERSION = 1

REPORT_HTML = "report.html"
SUMMARY_CSV = "summary.csv"
PARTS_CSV = "parts.csv"
HISTORY_CSV = "history.csv"
REPORT_JSON = "report.json"
VIEW_PNG = "view.png"

STATE_TEXT = {
    "none": "未実行",
    "current": "現在の設定で計算した結果",
    "stale": "現在の設定より前の条件で計算した結果 (設定を変更した後、再計算していません)",
}
LEVEL_MARK = {"converged": "✔", "mostly_converged": "▲", "not_converged": "✖", "diverged": "✖", "undetermined": "—"}
MAX_CHART_POINTS = 600
_GROUPED_NUMBER = re.compile(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?")


# ----------------------------------------------------------------------------- 値の整形

def _number(value, digits: int = 2, signed: bool = False) -> str | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"


def _sci(value) -> str | None:
    if value is None:
        return None
    value = float(value)
    return f"{value:.3e}" if math.isfinite(value) else None


def _length(value) -> str | None:
    return _number(value, 3)


def _forces_failed(summary: dict | None) -> bool:
    return summary is not None and (summary.get("status") == "failed" or summary.get("forces") is None)


def missing_text(summary: dict | None) -> str:
    """解析結果の値が無いときに書く文言。"""
    return FAILED if _forces_failed(summary) else NOT_COMPUTED


def coefficients(summary: dict | None, reference_area: float | None) -> dict:
    """Cd・Cl・Cs。基準面積は「現在の」設定値を使う (画面の空力結果タブと同じ)。"""
    if summary is None or _forces_failed(summary) or not reference_area:
        return {"cd": None, "cl": None, "cs": None}
    forces = summary["forces"]
    scale = forces["dynamic_pressure"] * reference_area
    return {"cd": forces["drag"] / scale, "cl": forces["lift"] / scale, "cs": forces["side"] / scale}


# ----------------------------------------------------------------------------- 項目の一覧

def axle_rows(forces: dict | None, snapshot: dict) -> list[tuple]:
    """前後の荷重配分の行 (ホイールベースが設定されているとき)。"""
    from ..solver.results import axle_loads

    tunnel = snapshot.get("tunnel") or {}
    wheelbase = tunnel.get("reference_length")
    if forces is None:
        return []
    if not wheelbase:
        return [("前後の荷重配分", "ホイールベースが未設定", "")]
    if forces.get("moment") is None:
        return [("前後の荷重配分", "この結果には無し (v0.8 以降の解析で計算)", "")]
    axle = axle_loads(forces, float(tunnel.get("front_axle_x") or 0.0), float(wheelbase), forces.get("dynamic_pressure"),
                      snapshot.get("reference_area"))
    rows = [("前輪側ダウンフォース", _number(axle["front"], 2), "N"), ("後輪側ダウンフォース", _number(axle["rear"], 2), "N"),
            ("前後配分 (前輪側)", None if axle["front_share"] is None else f"{axle['front_share'] * 100:.1f}", "%"),
            ("前輪軸まわりのピッチングモーメント", _number(axle["pitch_moment"], 2), "N·m")]
    if "cl_front" in axle:
        rows += [("Cl 前 (上向き正)", _number(axle["cl_front"], 4), ""), ("Cl 後 (上向き正)", _number(axle["cl_rear"], 4), ""),
                 ("Cm (前輪軸まわり、代表長さ = ホイールベース)", _number(axle["cm"], 4), "")]
    rows.append(("前後配分の基準", f"前輪軸 X = {float(tunnel.get('front_axle_x') or 0.0):.4f} m、ホイールベース {float(wheelbase):.4f} m", ""))
    return rows


def build_sections(snapshot: dict) -> list[dict]:
    """レポートの表 (節ごとの「項目・値・単位」)。HTML と CSV の両方がこれを使う。"""
    solver = snapshot.get("solver") or {}
    summary = solver.get("summary")
    missing = missing_text(summary)
    area = snapshot.get("reference_area")
    sections: list[dict] = []

    def section(title: str, rows: list[tuple]) -> None:
        sections.append({"title": title, "rows": [(label, missing_or(value), unit) for label, value, unit in rows]})

    def missing_or(value, text: str | None = None) -> str:
        return (text or missing) if value is None else str(value)

    # --- 計算の状態と収束 ---------------------------------------------------
    if summary is None:
        section("計算の状態と収束", [("計算の状態", "CFD 解析はまだ実行されていません", ""), ("収束判定", None, "")])
    else:
        iterations = summary.get("iterations") or {}
        convergence = summary.get("convergence") or {}
        openfoam = summary.get("openfoam") or {}
        rows = [
            ("計算の状態", summary.get("status_label"), ""),
            ("収束判定", convergence.get("label"), ""),
            ("収束判定の説明", convergence.get("message"), ""),
            ("結果と現在の設定の関係", STATE_TEXT.get(solver.get("state"), solver.get("state")), ""),
            ("反復数 (実行 / 指定)", f"{iterations.get('done', '—')} / {iterations.get('requested_end', '—')}", "反復"),
            ("ソルバーの計算時間", _number(summary.get("seconds_solver"), 0), "秒"),
            ("OpenFOAM", f"{openfoam.get('version', '—')} ({openfoam.get('n_procs', '—')} プロセス)", ""),
            ("計算日時", summary.get("created"), ""),
        ]
        for check in convergence.get("checks") or []:
            mark = {True: "✔ 満たす", False: "✖ 満たさない", None: "— 判定不能"}[check.get("ok")]
            rows.append((f"収束の確認: {check.get('name')}", f"{mark} — {check.get('text')}", ""))
        if summary.get("error"):
            rows.append(("ソルバーのエラー", " ".join(str(summary["error"]).strip().splitlines()[:6]), ""))
        section("計算の状態と収束", rows)

    # --- 空力結果 -----------------------------------------------------------
    forces = None if _forces_failed(summary) or summary is None else summary["forces"]
    ranges = (summary or {}).get("ranges") or {}
    coefficient = coefficients(summary, area)
    no_area = None if forces is None else "基準面積が未設定"

    def force(key: str) -> str | None:
        return None if forces is None else _number(forces.get(key), 2)

    rows = [
        ("Drag (抗力)", force("drag"), "N"),
        ("Downforce (下向き正)", force("downforce"), "N"),
        ("Lift (揚力・上向き正)", force("lift"), "N"),
        ("Side Force (横力)", force("side"), "N"),
        ("Cd", missing_or(_number(coefficient["cd"], 4), no_area), ""),
        ("Cl (上向き正)", missing_or(_number(coefficient["cl"], 4), no_area), ""),
        ("Cs", missing_or(_number(coefficient["cs"], 4), no_area), ""),
        *axle_rows(forces, snapshot),
        ("抗力の圧力成分", force("drag_pressure"), "N"),
        ("抗力の粘性成分", force("drag_viscous"), "N"),
        ("揚力の圧力成分", force("lift_pressure"), "N"),
        ("揚力の粘性成分", force("lift_viscous"), "N"),
        ("荷重の平均区間 (直近)", None if forces is None else forces.get("averaged_over"), "反復"),
        ("平均区間内の Drag の変動幅", force("drag_range"), "N"),
        ("平均区間内の Lift の変動幅", force("lift_range"), "N"),
        ("動圧", None if forces is None else _number(forces.get("dynamic_pressure"), 2), "Pa"),
        ("基準面積 (Cd・Cl の換算に使用)", missing_or(_number(area, 4), "未設定"), "m²"),
        ("最大流速", _number(ranges.get("speed_max"), 2), "m/s"),
        ("最大圧力 (ゲージ)", _number(ranges.get("pressure_max"), 1), "Pa"),
        ("最小圧力 (ゲージ)", _number(ranges.get("pressure_min"), 1), "Pa"),
        ("Cp の最大 / 最小", None if ranges.get("cp_max") is None else f"{ranges['cp_max']:.3f} / {ranges['cp_min']:.3f}", ""),
    ]
    section("空力結果", rows)

    # --- 収束の数値 ---------------------------------------------------------
    if summary is not None:
        residuals = summary.get("final_residuals") or {}
        mass = summary.get("mass") or {}
        rows = [(f"最終残差 {name}", _sci(value), "") for name, value in residuals.items()]
        rows.append(("質量収支 (流入出の差)", None if mass.get("imbalance") is None else f"{mass['imbalance'] * 100:.5f}", "%"))
        section("残差と質量収支", rows)

    # --- 解析条件 -----------------------------------------------------------
    flow = snapshot.get("flow") or {}
    tunnel = snapshot.get("tunnel") or {}
    settings = solver.get("settings") or {}
    rows = [
        ("風速", missing_or(_number(flow.get("speed_kmh"), 1), "—"), "km/h"),
        ("風速 (m/s 換算)", missing_or(_number(flow.get("speed_ms"), 3), "—"), "m/s"),
        ("空気密度", missing_or(_number(flow.get("density"), 4), "—"), "kg/m³"),
        ("動粘性係数", missing_or(_sci(flow.get("nu")), "—"), "m²/s"),
        ("マッハ数", missing_or(_number(flow.get("mach"), 3), "—"), ""),
        ("レイノルズ数 (車両全長基準)", missing_or(_sci(flow.get("reynolds")), "—"), ""),
        ("動圧", missing_or(_number(flow.get("dynamic_pressure"), 2), "—"), "Pa"),
        ("地面", ("あり (移動地面)" if tunnel.get("moving_ground") else "あり (静止)") if tunnel.get("has_ground") else "なし", ""),
        ("解析手法", "定常 RANS (k-ω SST)、非圧縮 — OpenFOAM foamRun / incompressibleFluid", ""),
        ("指定反復数", settings.get("iterations", "—"), "反復"),
        ("緩和係数 (圧力 / 運動量 / 乱流)",
         f"{settings.get('relax_p', '—')} / {settings.get('relax_u', '—')} / {settings.get('relax_turbulence', '—')}", ""),
        ("残差の打ち切り基準", missing_or(_sci(settings.get("residual_tolerance")), "—"), ""),
    ]
    section("解析条件", rows)

    # --- 風洞 ---------------------------------------------------------------
    domain = snapshot.get("domain")
    blockage = snapshot.get("blockage")
    not_generated = "風洞が未生成"
    rows = [
        ("計算領域 X (流れ方向)", missing_or(None if domain is None else f"{domain[0]:.3f} … {domain[1]:.3f}", not_generated), "m"),
        ("計算領域 Y (左右)", missing_or(None if domain is None else f"{domain[2]:.3f} … {domain[3]:.3f}", not_generated), "m"),
        ("計算領域 Z (高さ)", missing_or(None if domain is None else f"{domain[4]:.3f} … {domain[5]:.3f}", not_generated), "m"),
        ("余白 前 / 後 / 側面 / 上", " / ".join(_length(tunnel.get(f"margin_{k}")) or "—" for k in ("front", "rear", "side", "top")), "m"),
        ("閉塞率 (前面投影面積 / 風洞断面積)", missing_or(None if blockage is None else f"{blockage * 100:.2f}", "未計算"), "%"),
        ("前面投影面積", missing_or(_number(tunnel.get("frontal_area"), 4), "未計算"), "m²"),
        ("接地時の沈み込み量", missing_or(_length(tunnel.get("contact_sink")), "—"), "m"),
    ]
    section("仮想風洞", rows)

    # --- メッシュ -----------------------------------------------------------
    mesh = snapshot.get("mesh") or {}
    report = mesh.get("report")
    if report is None:
        section("メッシュ", [("メッシュ", "未生成", "")])
    else:
        quality = report.get("quality") or {}
        layer = report.get("boundary_layer") or {}
        counts = quality.get("cell_counts") or {}
        if layer.get("generated"):
            layer_text = (f"あり ({layer.get('layers')} 層、第 1 層 {layer.get('first_height', 0) * 1000:.3f} mm、"
                          f"プリズム {layer.get('n_prisms', 0):,} 個)")
        else:
            layer_text = f"なし — {layer.get('reason') or ''}"
        verdict = (quality.get("summary") or {})
        rows = [
            ("メッシュと現在の設定の関係", STATE_TEXT.get(mesh.get("state"), mesh.get("state")), ""),
            ("セル数", f"{quality.get('n_cells', 0):,}", "個"),
            ("セルの内訳", "、".join(f"{name} {count:,}" for name, count in counts.items() if count) or "—", ""),
            ("節点数 / 面数", f"{quality.get('n_points', 0):,} / {quality.get('n_faces', 0):,}", ""),
            ("境界層プリズム", layer_text, ""),
            ("非直交性 最大 / 平均", f"{quality.get('non_ortho_max', 0):.1f} / {quality.get('non_ortho_mean', 0):.1f}", "°"),
            ("非直交性 70° 超の面", f"{quality.get('n_non_ortho_over_70', 0):,}", "面"),
            ("歪度 最大", f"{quality.get('skewness_max', 0):.2f}", ""),
            ("セル体積 最小 / 最大", f"{quality.get('volume_min', 0):.3e} / {quality.get('volume_max', 0):.3e}", "m³"),
            ("負体積のセル", f"{quality.get('n_negative_volumes', 0):,}", "個"),
            ("品質の判定", {"ok": "良好", "warning": "注意", "error": "不良"}.get(verdict.get("level"), "—"), ""),
            ("メッシュ生成時間", _number(report.get("seconds_total"), 1), "秒"),
        ]
        checkmesh = ((summary or {}).get("mesh") or {}).get("checkmesh")
        if checkmesh is not None:
            rows.append(("OpenFOAM checkMesh", "Mesh OK" if checkmesh.get("ok") else "問題あり: " + " ".join(checkmesh.get("messages") or []), ""))
        for message in verdict.get("messages") or []:
            rows.append(("品質の所見", message, ""))
        section("メッシュ", rows)
    return sections


def accuracy_notes(snapshot: dict) -> list[str]:
    """精度に関する注意。解析ジョブが残した注意に、結果が古い場合の警告を加える。"""
    solver = snapshot.get("solver") or {}
    summary = solver.get("summary")
    notes: list[str] = []
    if summary is None:
        return notes
    if solver.get("state") == "stale":
        notes.append("この結果は、現在のモデル・風洞・メッシュ・解析設定より前の条件で計算したものです。"
                     "下の「解析条件」などは現在の設定を示しており、結果を計算したときの条件と異なる可能性があります。")
    convergence = summary.get("convergence") or {}
    if convergence.get("level") not in ("converged", None):
        notes.append(f"収束判定は「{convergence.get('label')}」です。指定した反復数の計算が終わったことと、"
                     "工学的に十分収束したことは別です。荷重と係数は参考値として扱ってください。")
    notes.extend(str(note) for note in summary.get("notes") or [])
    return notes


# ----------------------------------------------------------------------------- CSV

def write_summary_csv(sections: list[dict], path: Path) -> None:
    # Excel がそのまま開けるよう BOM 付き UTF-8 で書く
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["節", "項目", "値", "単位"])
        for block in sections:
            for label, value, unit in block["rows"]:
                # 表計算で数値として扱えるよう、桁区切りのカンマは外す ("1,870.85" → "1870.85")
                plain = value.replace(",", "") if _GROUPED_NUMBER.fullmatch(value) else value
                writer.writerow([block["title"], label, plain, unit])


def part_rows(snapshot: dict) -> list[dict]:
    summary = (snapshot.get("solver") or {}).get("summary")
    if summary is None or _forces_failed(summary):
        return []
    yplus = summary.get("yplus") or {}
    rows = []
    for part in summary.get("parts") or []:
        wall = yplus.get(part.get("patch")) or {}
        rows.append({
            "name": part.get("part_name"), "patch": part.get("patch"),
            "drag": part.get("drag"), "downforce": part.get("downforce"), "lift": part.get("lift"), "side": part.get("side"),
            "drag_pressure": part.get("drag_pressure"), "drag_viscous": part.get("drag_viscous"),
            "yplus_average": wall.get("average"), "yplus_min": wall.get("min"), "yplus_max": wall.get("max"),
        })
    return rows


def write_parts_csv(rows: list[dict], path: Path) -> None:
    columns = [("name", "パーツ"), ("patch", "パッチ"), ("drag", "Drag [N]"), ("downforce", "Downforce [N]"), ("lift", "Lift [N]"),
               ("side", "Side Force [N]"), ("drag_pressure", "抗力の圧力成分 [N]"), ("drag_viscous", "抗力の粘性成分 [N]"),
               ("yplus_average", "y+ 平均"), ("yplus_min", "y+ 最小"), ("yplus_max", "y+ 最大")]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([title for _key, title in columns])
        for row in rows:
            writer.writerow(["" if row.get(key) is None else row[key] for key, _title in columns])


# ----------------------------------------------------------------------------- グラフ (SVG)

def _decimate(values: list[float], limit: int = MAX_CHART_POINTS) -> list[int]:
    """グラフ用に点を間引くときの添字。最初と最後は必ず残す。"""
    n = len(values)
    if n <= limit:
        return list(range(n))
    step = (n - 1) / (limit - 1)
    return sorted({int(round(i * step)) for i in range(limit)})


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw), default=10) * magnitude
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    value = first
    while value <= hi + step * 1e-9:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _tick_text(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < 0.01:
        return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return f"{value:,.6g}"


def line_chart(chart_id: str, title: str, x: list[float], series: list[dict], *, log_y: bool = False, y_unit: str = "",
               y_range: tuple[float, float] | None = None, end_labels: bool = False,
               width: int = 760, height: int = 280) -> str:
    """折れ線グラフ (SVG)。``series`` は ``{"name", "values", "slot"}`` の並び。

    1 つのグラフの縦軸は 1 本だけ。値は表 (最終残差・荷重) と CSV にも載るので、グラフでしか読めない値は無い。
    ホバー・キーボード用のデータは隣の JSON に置き、ページ末尾のスクリプトが十字線とツールチップを付ける。
    """
    left, right, top, bottom = 64, 96 if end_labels else 20, 14, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    keep = _decimate(x)
    xs = [float(x[i]) for i in keep]
    data = []
    for item in series:
        values = [float(item["values"][i]) for i in keep]
        data.append({"name": item["name"], "slot": int(item["slot"]), "values": [v if math.isfinite(v) else None for v in values]})
    finite = [v for item in data for v in item["values"] if v is not None and (not log_y or v > 0)]
    if not xs or not finite:
        return ""
    if y_range is None:
        lo, hi = min(finite), max(finite)
    else:
        lo, hi = y_range
    if log_y:
        lo_e, hi_e = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        if hi_e == lo_e:
            hi_e += 1
        ticks = [10.0 ** e for e in range(lo_e, hi_e + 1)]
        if len(ticks) > 7:
            ticks = ticks[::2] if (len(ticks) - 1) % 2 == 0 else ticks[::2] + [ticks[-1]]
        lo, hi = 10.0 ** lo_e, 10.0 ** hi_e

        def scale_y(value: float) -> float:
            return top + plot_h * (1 - (math.log10(value) - lo_e) / (hi_e - lo_e))
    else:
        if hi == lo:
            pad = abs(hi) * 0.05 or 1.0
            lo, hi = lo - pad, hi + pad
        ticks = _nice_ticks(lo, hi)
        lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

        def scale_y(value: float) -> float:
            return top + plot_h * (1 - (value - lo) / (hi - lo))

    x_lo, x_hi = xs[0], xs[-1] if xs[-1] > xs[0] else xs[0] + 1

    def scale_x(value: float) -> float:
        return left + plot_w * (value - x_lo) / (x_hi - x_lo)

    out = [f'<svg class="chart" id="{chart_id}" viewBox="0 0 {width} {height}" role="img" tabindex="0" '
           f'aria-label="{html.escape(title)}" data-left="{left}" data-right="{right}" data-top="{top}" data-bottom="{bottom}">']
    out.append(f'<clipPath id="{chart_id}-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath>')
    for tick in ticks:
        y = scale_y(tick)
        label = (f"1e{round(math.log10(tick))}" if tick != 1 else "1") if log_y else _tick_text(tick)
        out.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    for tick in _nice_ticks(x_lo, x_hi, 6):
        if x_lo <= tick <= x_hi:
            out.append(f'<text class="tick" x="{scale_x(tick):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{_tick_text(tick)}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    out.append(f'<text class="axis-title" x="{left + plot_w / 2:.0f}" y="{height - 2}" text-anchor="middle">反復</text>')
    if y_unit:
        out.append(f'<text class="axis-title" x="{left - 8}" y="{top - 2}" text-anchor="end">{html.escape(y_unit)}</text>')
    ends = []
    for item in data:
        points = []
        segments = []
        for px, value in zip(xs, item["values"]):
            if value is None or (log_y and value <= 0):
                if points:
                    segments.append(points)
                points = []
                continue
            points.append(f"{scale_x(px):.1f},{scale_y(value):.1f}")
        if points:
            segments.append(points)
        for points in segments:
            out.append(f'<polyline class="series s{item["slot"]}" clip-path="url(#{chart_id}-clip)" points="{" ".join(points)}"/>')
        last = next((v for v in reversed(item["values"]) if v is not None), None)
        if last is not None and lo <= last <= hi:
            ends.append((scale_y(last), item, last))
    if end_labels and ends:
        ends.sort(key=lambda entry: entry[0])
        # 終点のラベルが重なる場合は付けない (凡例・表・ツールチップで読める)
        if all(b[0] - a[0] >= 15 for a, b in zip(ends, ends[1:])):
            for y, item, last in ends:
                out.append(f'<circle class="dot s{item["slot"]}" cx="{left + plot_w:.1f}" cy="{y:.1f}" r="4"/>')
                out.append(f'<text class="end-label" x="{left + plot_w + 9:.1f}" y="{y + 4:.1f}">{_number(last, 1)}</text>')
    out.append(f'<line class="cross" x1="0" x2="0" y1="{top}" y2="{top + plot_h}" visibility="hidden"/>')
    out.append("</svg>")
    payload = {"x": xs, "series": data, "log": log_y, "unit": y_unit, "xlo": x_lo, "xhi": x_hi}
    legend = "".join(f'<span class="key"><span class="swatch s{item["slot"]}"></span>{html.escape(item["name"])}</span>' for item in data)
    legend_html = f'<div class="legend">{legend}</div>' if len(data) > 1 else ""
    return (f'<figure class="chart-box"><figcaption>{html.escape(title)}</figcaption>{legend_html}'
            f'<div class="chart-wrap">{"".join(out)}<div class="tip" hidden></div></div>'
            f'<script type="application/json" data-for="{chart_id}">{json.dumps(payload).replace("</", "<\\/")}</script></figure>')


def history_charts(history: dict | None) -> str:
    """残差と荷重の履歴グラフ。履歴が無ければ空文字。"""
    if not history or len(history.get("iteration", [])) < 2:
        return ""
    iteration = [float(v) for v in history["iteration"]]
    charts = []
    residuals = history.get("residuals") or {}
    if residuals:
        series = [{"name": name, "values": list(values), "slot": slot} for slot, (name, values) in enumerate(residuals.items(), start=1)][:6]
        charts.append(line_chart("chart-residuals", "残差の履歴 (初期残差、対数軸)", iteration, series, log_y=True))
    drag = [float(v) for v in history.get("drag", [])]
    lift = [float(v) for v in history.get("lift", [])]
    # Drag と Downforce は値が離れているので、同じ縦軸に描くと収束の様子 (細かい変動) が読めない。
    # それぞれ自分の縦軸を持つ別のグラフにする。
    for chart_id, name, values, slot in (("chart-drag", "Drag", drag, 1), ("chart-downforce", "Downforce", [-v for v in lift], 2)):
        if len(values) < 2:
            continue
        # 計算開始直後の過渡 (桁違いに大きい値) に縦軸を合わせると変動が潰れるので、
        # 後半 70 % の範囲に余裕を付けた縦軸にし、範囲外は枠で切る。
        tail = [v for v in values[int(len(values) * 0.3):] if math.isfinite(v)]
        y_range = None
        if tail:
            lo, hi = min(tail), max(tail)
            pad = max((hi - lo) * 0.5, abs(hi) * 0.01, 1e-9)
            y_range = (lo - pad, hi + pad)
        charts.append(line_chart(chart_id, f"{name} の履歴 [N] (縦軸は後半の変動に合わせています。計算開始直後の過渡は範囲外)",
                                 iteration, [{"name": name, "values": values, "slot": slot}], y_unit="N", y_range=y_range,
                                 end_labels=True, height=220))
    return "".join(charts)


# ----------------------------------------------------------------------------- HTML

_CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#0ca30c; } }
@media print { :root { color-scheme: light; --page:#ffffff; --surface:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --grid:#e1e0d9;
  --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.6 system-ui,-apple-system,"Segoe UI","Yu Gothic UI","Meiryo",sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 2px; } h2 { font-size: 16px; margin: 30px 0 8px; }
.sub { color: var(--ink2); margin: 0 0 18px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
.status { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
.status .label { color:var(--ink2); font-size:12px; } .status .value { font-size:17px; font-weight:600; }
.status .detail { color:var(--ink2); font-size:13px; margin-top:2px; }
.mark { display:inline-block; width:1.2em; font-weight:700; }
.mark.converged { color:var(--good); } .mark.mostly_converged, .mark.undetermined { color:var(--ink2); }
.mark.not_converged, .mark.diverged, .mark.failed { color:var(--critical); }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-top:12px; }
.tile .label { color:var(--ink2); font-size:12px; } .tile .value { font-size:26px; font-weight:600; line-height:1.3; }
.tile .unit { color:var(--ink2); font-size:13px; font-weight:400; margin-left:4px; }
.tile .value.missing { font-size:18px; color:var(--ink2); }
.notes { margin:0; padding-left:1.3em; } .notes li { margin:4px 0; }
.warn-box { border-left:4px solid var(--warning); }
table { width:100%; border-collapse:collapse; } th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }
th { color:var(--ink2); font-weight:500; font-size:12px; } td.num { text-align:right; font-variant-numeric: tabular-nums; white-space:nowrap; }
td.item { color:var(--ink2); width:34%; } td.unit { color:var(--muted); width:9%; white-space:nowrap; }
.table-scroll { overflow-x:auto; }
figure.chart-box { margin:12px 0 0; } figcaption { color:var(--ink2); font-size:13px; margin-bottom:4px; }
.legend { display:flex; flex-wrap:wrap; gap:4px 16px; font-size:12px; color:var(--ink2); margin-bottom:2px; }
.swatch { display:inline-block; width:16px; height:0; border-top:2px solid; vertical-align:middle; margin-right:6px; }
.chart-wrap { position:relative; } svg.chart { width:100%; height:auto; display:block; outline-offset:2px; }
.grid { stroke:var(--grid); stroke-width:1; } .axis { stroke:var(--axis); stroke-width:1; }
.tick, .axis-title { fill:var(--muted); font-size:11px; font-variant-numeric: tabular-nums; } .end-label { fill:var(--ink); font-size:12px; font-weight:600; }
.series { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; } .cross { stroke:var(--axis); stroke-width:1; }
.dot { stroke:var(--surface); stroke-width:2; }
.s1 { stroke:var(--s1); border-color:var(--s1); } .s2 { stroke:var(--s2); border-color:var(--s2); } .s3 { stroke:var(--s3); border-color:var(--s3); }
.s4 { stroke:var(--s4); border-color:var(--s4); } .s5 { stroke:var(--s5); border-color:var(--s5); } .s6 { stroke:var(--s6); border-color:var(--s6); }
circle.s1 { fill:var(--s1); } circle.s2 { fill:var(--s2); } circle.s3 { fill:var(--s3); } circle.s4 { fill:var(--s4); } circle.s5 { fill:var(--s5); } circle.s6 { fill:var(--s6); }
.tip { position:absolute; top:6px; pointer-events:none; background:var(--surface); border:1px solid var(--border); border-radius:6px;
  padding:6px 9px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,0.12); white-space:nowrap; z-index:2; }
.tip .head { color:var(--ink2); margin-bottom:2px; } .tip .row { display:flex; align-items:center; gap:6px; }
.tip .row b { font-variant-numeric: tabular-nums; } .tip .row .name { color:var(--ink2); }
img.view { width:100%; height:auto; border-radius:6px; display:block; }
footer { color:var(--muted); font-size:12px; margin-top:28px; }
@media print { main { max-width:none; padding:0; } .card { break-inside:avoid; } .tip { display:none; } }
"""

_SCRIPT = """
(function () {
  function format(v, log) { if (v === null) return '—'; return log ? v.toExponential(2) : v.toLocaleString(undefined, {maximumFractionDigits: 2}); }
  document.querySelectorAll('script[data-for]').forEach(function (node) {
    var svg = document.getElementById(node.getAttribute('data-for')); if (!svg) return;
    var data = JSON.parse(node.textContent), tip = svg.parentNode.querySelector('.tip'), cross = svg.querySelector('.cross');
    var box = svg.viewBox.baseVal, left = +svg.dataset.left, right = +svg.dataset.right, index = data.x.length - 1;
    function show(i) {
      index = Math.max(0, Math.min(data.x.length - 1, i));
      var px = left + (box.width - left - right) * (data.x[index] - data.xlo) / (data.xhi - data.xlo);
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      while (tip.firstChild) tip.removeChild(tip.firstChild);
      var head = document.createElement('div'); head.className = 'head'; head.textContent = '反復 ' + data.x[index]; tip.appendChild(head);
      data.series.forEach(function (s) {
        var row = document.createElement('div'); row.className = 'row';
        var key = document.createElement('span'); key.className = 'swatch s' + s.slot; row.appendChild(key);
        var value = document.createElement('b'); value.textContent = format(s.values[index], data.log) + (data.unit ? ' ' + data.unit : ''); row.appendChild(value);
        var name = document.createElement('span'); name.className = 'name'; name.textContent = s.name; row.appendChild(name);
        tip.appendChild(row);
      });
      tip.hidden = false;
      var ratio = svg.getBoundingClientRect().width / box.width, x = px * ratio;
      if (x > svg.getBoundingClientRect().width / 2) { tip.style.left = ''; tip.style.right = (svg.getBoundingClientRect().width - x + 12) + 'px'; }
      else { tip.style.right = ''; tip.style.left = (x + 12) + 'px'; }
    }
    function hide() { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); }
    svg.addEventListener('pointermove', function (event) {
      var rect = svg.getBoundingClientRect(), vx = (event.clientX - rect.left) / rect.width * box.width;
      var target = data.xlo + (vx - left) / (box.width - left - right) * (data.xhi - data.xlo), best = 0, distance = Infinity;
      for (var i = 0; i < data.x.length; i++) { var d = Math.abs(data.x[i] - target); if (d < distance) { distance = d; best = i; } }
      show(best);
    });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function () { show(index); });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowLeft') { show(index - 1); event.preventDefault(); }
      if (event.key === 'ArrowRight') { show(index + 1); event.preventDefault(); }
    });
  });
})();
"""


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _table(rows: list[tuple]) -> str:
    body = "".join(f'<tr><td class="item">{_esc(label)}</td><td>{_esc(value)}</td><td class="unit">{_esc(unit)}</td></tr>'
                   for label, value, unit in rows)
    return f'<div class="table-scroll"><table>{body}</table></div>'


def render_html(snapshot: dict, sections: list[dict], screenshot_png: bytes | None = None) -> str:
    solver = snapshot.get("solver") or {}
    summary = solver.get("summary")
    missing = missing_text(summary)
    area = snapshot.get("reference_area")
    coefficient = coefficients(summary, area)
    forces = None if summary is None or _forces_failed(summary) else summary["forces"]
    project = snapshot.get("project_name") or "(名前なし)"
    out = ['<!doctype html><html lang="ja"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f'<title>解析レポート — {_esc(project)}</title><style>{_CSS}</style></head><body><main>',
           f'<h1>空力解析レポート</h1><p class="sub">{_esc(project)} ・ 作成 {_esc(snapshot.get("created"))} ・ '
           f'LJM CFD Claude Edition v{_esc(snapshot.get("app_version"))}</p>']

    # 計算の状態と収束判定は別々に示す (計算が終わったことと、収束したことは別)
    if summary is None:
        status_value, status_detail, level, verdict, verdict_detail = "未実行", "CFD 解析はまだ実行されていません。", "undetermined", NOT_COMPUTED, ""
    else:
        iterations = summary.get("iterations") or {}
        convergence = summary.get("convergence") or {}
        status_value = summary.get("status_label") or "—"
        status_detail = f"{iterations.get('done', '—')} 反復 ・ {summary.get('created', '')}"
        level = convergence.get("level") or "undetermined"
        verdict, verdict_detail = convergence.get("label") or "—", convergence.get("message") or ""
    status_mark = "✖" if _forces_failed(summary) else ("—" if summary is None else "✔")
    status_class = "failed" if _forces_failed(summary) else ("undetermined" if summary is None else "converged")
    out.append('<div class="status">'
               f'<div class="card"><div class="label">計算の状態 (指定した反復まで計算できたか)</div>'
               f'<div class="value"><span class="mark {status_class}">{status_mark}</span>{_esc(status_value)}</div>'
               f'<div class="detail">{_esc(status_detail)}</div></div>'
               f'<div class="card"><div class="label">収束判定 (工学的に十分収束したか)</div>'
               f'<div class="value"><span class="mark {_esc(level)}">{LEVEL_MARK.get(level, "—")}</span>{_esc(verdict)}</div>'
               f'<div class="detail">{_esc(verdict_detail)}</div></div></div>')

    def tile(label: str, value: str | None, unit: str, fallback: str | None = None) -> str:
        if value is None:
            return f'<div class="card tile"><div class="label">{_esc(label)}</div><div class="value missing">{_esc(fallback or missing)}</div></div>'
        return (f'<div class="card tile"><div class="label">{_esc(label)}</div>'
                f'<div class="value">{_esc(value)}<span class="unit">{_esc(unit)}</span></div></div>')

    no_area = None if forces is None else "基準面積が未設定"
    out.append('<div class="tiles">'
               + tile("Drag (抗力)", None if forces is None else _number(forces["drag"], 1), "N")
               + tile("Downforce (下向き正)", None if forces is None else _number(forces["downforce"], 1), "N")
               + tile("Cd", _number(coefficient["cd"], 3), "", no_area)
               + tile("Cl (上向き正)", _number(coefficient["cl"], 3), "", no_area)
               + '</div>')

    notes = accuracy_notes(snapshot)
    if notes:
        out.append('<h2>精度に関する注意</h2><div class="card warn-box"><ul class="notes">'
                   + "".join(f"<li>{_esc(note)}</li>" for note in notes) + "</ul></div>")

    if screenshot_png:
        encoded = base64.b64encode(screenshot_png).decode("ascii")
        out.append('<h2>3D ビュー (書き出し時の表示)</h2><div class="card">'
                   f'<img class="view" alt="書き出し時の 3D ビュー" src="data:image/png;base64,{encoded}"></div>')

    charts = history_charts(solver.get("history"))
    by_title = {block["title"]: block for block in sections}
    for title in ("計算の状態と収束", "空力結果"):
        out.append(f'<h2>{_esc(title)}</h2><div class="card">{_table(by_title[title]["rows"])}</div>')
    if charts or "残差と質量収支" in by_title:
        out.append('<h2>残差・荷重の履歴</h2><div class="card">' + charts)
        if "残差と質量収支" in by_title:
            out.append(_table(by_title["残差と質量収支"]["rows"]))
        out.append("</div>")

    parts = part_rows(snapshot)
    if parts:
        head = "".join(f"<th>{title}</th>" for title in ("パーツ", "Drag [N]", "Downforce [N]", "Side Force [N]", "抗力の圧力成分 [N]",
                                                        "抗力の粘性成分 [N]", "y+ 平均"))
        body = "".join(
            f'<tr><td>{_esc(row["name"])}</td>' + "".join(
                f'<td class="num">{_esc(_number(row[key], digits) or "—")}</td>'
                for key, digits in (("drag", 2), ("downforce", 2), ("side", 2), ("drag_pressure", 2), ("drag_viscous", 2), ("yplus_average", 0)))
            + "</tr>" for row in parts)
        out.append(f'<h2>パーツ別の荷重</h2><div class="card"><div class="table-scroll"><table><tr>{head}</tr>{body}</table></div></div>')

    for title in ("解析条件", "仮想風洞", "メッシュ"):
        out.append(f'<h2>{_esc(title)}</h2><div class="card">{_table(by_title[title]["rows"])}</div>')

    model = snapshot.get("model") or {}
    model_parts = model.get("parts") or []
    if model_parts:
        head = "".join(f"<th>{title}</th>" for title in ("パーツ", "ファイル", "単位", "解析対象", "移動 X / Y / Z [m]", "Yaw / Pitch / Roll [°]"))
        body = "".join(
            f'<tr><td>{_esc(part.get("name"))}</td><td>{_esc(part.get("file"))}</td><td>{_esc(part.get("unit"))}</td>'
            f'<td>{"○" if part.get("analysed") else "— (非表示)"}</td>'
            f'<td class="num">{_esc(" / ".join(f"{v:.4f}" for v in part.get("position", (0, 0, 0))))}</td>'
            f'<td class="num">{_esc(" / ".join(f"{v:.2f}" for v in part.get("rotation", (0, 0, 0))))}</td></tr>'
            for part in model_parts)
        bounds = model.get("bounds")
        pose = model.get("vehicle_pose") or {}
        if any(pose.values()):
            body += (f'<tr><td colspan="4">車両全体の姿勢 (全パーツに共通)</td>'
                     f'<td class="num">{_esc(" / ".join(f"{pose.get(k, 0.0):.4f}" for k in ("tx", "ty", "tz")))}</td>'
                     f'<td class="num">{_esc(" / ".join(f"{pose.get(k, 0.0):.2f}" for k in ("yaw", "pitch", "roll")))}</td></tr>')
        size = "" if not bounds else (f'<p class="sub" style="margin:8px 0 0">解析対象の寸法 (全長 × 全幅 × 全高): '
                                      f'{bounds[1] - bounds[0]:.3f} × {bounds[3] - bounds[2]:.3f} × {bounds[5] - bounds[4]:.3f} m</p>')
        out.append(f'<h2>モデル</h2><div class="card"><div class="table-scroll"><table><tr>{head}</tr>{body}</table></div>{size}</div>')

    out.append('<footer>力は風洞座標 (風向 +X、上 +Z) で表します。Downforce = −Lift。係数 = 力 / (動圧 × 基準面積)。'
               '荷重は平均区間 (直近の反復) の平均値です。同じフォルダの summary.csv・parts.csv・history.csv に数値を、'
               'report.json に元データを保存しています。</footer>')
    out.append(f"</main><script>{_SCRIPT}</script></body></html>")
    return "".join(out)


# ----------------------------------------------------------------------------- 書き出し

def _jsonable(value):
    """NumPy の配列・数値を含む辞書を JSON にできる形へ変換する。"""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_report(snapshot: dict, out_dir: str | Path, screenshot_png: bytes | None = None) -> dict[str, Path]:
    """レポート一式を ``out_dir`` へ書き出し、書いたファイルを返す。

    * ``report.html``  … 表・グラフ・3D ビューの画像を 1 ファイルに収めたレポート (ブラウザで開く。印刷可)
    * ``summary.csv``  … 「節・項目・値・単位」の一覧 (Excel で開ける)
    * ``parts.csv``    … パーツ別の荷重 (解析結果があるときだけ)
    * ``history.csv``  … 反復ごとの残差・荷重 (解析結果フォルダからコピー)
    * ``report.json``  … 元データ
    * ``view.png``     … 3D ビューの画像 (渡されたときだけ)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(snapshot)
    snapshot.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    snapshot["report_version"] = REPORT_VERSION
    sections = build_sections(snapshot)
    files: dict[str, Path] = {}

    files["html"] = out_dir / REPORT_HTML
    files["html"].write_text(render_html(snapshot, sections, screenshot_png), encoding="utf-8")
    files["summary"] = out_dir / SUMMARY_CSV
    write_summary_csv(sections, files["summary"])
    parts = part_rows(snapshot)
    if parts:
        files["parts"] = out_dir / PARTS_CSV
        write_parts_csv(parts, files["parts"])
    solver = snapshot.get("solver") or {}
    summary = solver.get("summary")
    if summary is not None and summary.get("results_dir"):
        history = Path(summary["results_dir"]) / ((summary.get("files") or {}).get("history") or HISTORY_CSV)
        if history.is_file():
            files["history"] = out_dir / HISTORY_CSV
            shutil.copyfile(history, files["history"])
    if screenshot_png:
        files["view"] = out_dir / VIEW_PNG
        files["view"].write_bytes(screenshot_png)
    data = {key: value for key, value in snapshot.items() if key != "solver"}
    data["solver"] = {key: value for key, value in solver.items() if key != "history"}     # 履歴は history.csv にある
    data["sections"] = [{"title": block["title"], "rows": [list(row) for row in block["rows"]]} for block in sections]
    data["accuracy_notes"] = accuracy_notes(snapshot)
    files["json"] = out_dir / REPORT_JSON
    files["json"].write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
    return files
