"""解析レポート (HTML・CSV・JSON) の書き出し。GUI・OpenFOAM なしで実行できる。"""

from __future__ import annotations

import csv
import json
import re

import numpy as np
import pytest

from ljmcfd.report.export import (
    FAILED, NOT_COMPUTED, accuracy_notes, build_sections, coefficients, export_report, history_charts, line_chart,
)
from ljmcfd.solver.results import read_history, write_history
from tests.synthetic import RHO, U0, make_results


def snapshot(summary=None, state="none", area=2.0, history=None, parts=None) -> dict:
    return {
        "app_version": "9.9.9", "created": "2026-02-03 04:05:06", "project_name": "試験プロジェクト",
        "model": {"parts": parts or [{"name": "Body", "file": "car.step", "unit": "mm", "analysed": True,
                                      "position": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0)}],
                  "bounds": [0.0, 4.0, -0.9, 0.9, 0.0, 1.2], "vehicle_pose": {"tx": 0.0, "ty": 0.0, "tz": 0.0, "yaw": 3.0, "pitch": 0.0, "roll": 0.0}},
        "tunnel": {"has_ground": True, "moving_ground": True, "margin_front": 12.0, "margin_rear": 24.0, "margin_side": 5.4,
                   "margin_top": 4.8, "frontal_area": 1.9, "contact_sink": 0.01},
        "flow": {"speed_ms": 200 / 3.6, "speed_kmh": 200.0, "density": 1.204, "nu": 1.5e-5, "mach": 0.162, "dynamic_pressure": 1858.0,
                 "reynolds": 1.48e7},
        "domain": [-12.0, 28.0, -6.3, 6.3, 0.0, 6.0], "blockage": 0.025, "reference_area": area,
        "mesh": {"state": "none", "report": None},
        "solver": {"state": state, "settings": {"iterations": 500, "relax_p": 0.3, "relax_u": 0.7, "relax_turbulence": 0.7,
                                               "residual_tolerance": 1e-5}, "summary": summary, "history": history},
    }


def rows(sections) -> dict:
    return {(block["title"], label): value for block in sections for label, value, _unit in block["rows"]}


def fake_history(n=400) -> dict:
    iteration = np.arange(1, n + 1, dtype=float)
    decay = np.exp(-iteration / 60.0)
    return {"iteration": iteration, "residuals": {"p": 1e-3 + decay, "Ux": 1e-4 + 0.5 * decay, "k": 2e-4 + 0.8 * decay},
            "drag": 100.0 + 400.0 * decay * np.cos(iteration / 3.0), "lift": -300.0 + 900.0 * decay * np.sin(iteration / 4.0),
            "side": np.zeros(n)}


def test_report_without_results_says_not_computed(tmp_path):
    """解析していない値を、数値や 0 で埋めない。"""
    files = export_report(snapshot(), tmp_path / "report")
    assert set(files) == {"html", "summary", "json"}                       # パーツ別荷重・履歴・画像は書かない
    values = rows(build_sections(snapshot()))
    for label in ("Drag (抗力)", "Downforce (下向き正)", "Lift (揚力・上向き正)", "Side Force (横力)", "Cd", "Cl (上向き正)",
                  "抗力の圧力成分", "抗力の粘性成分", "最大流速", "最大圧力 (ゲージ)"):
        assert values[("空力結果", label)] == NOT_COMPUTED
    assert values[("計算の状態と収束", "収束判定")] == NOT_COMPUTED
    assert values[("メッシュ", "メッシュ")] == "未生成"
    # 解析条件・風洞は設定値として載る
    assert values[("解析条件", "指定反復数")] == "500" and values[("仮想風洞", "閉塞率 (前面投影面積 / 風洞断面積)")] == "2.50"
    html = files["html"].read_text(encoding="utf-8")
    assert html.count(NOT_COMPUTED) >= 5 and "未実行" in html and "<svg" not in html
    assert "試験プロジェクト" in html and "v9.9.9" in html
    with open(files["summary"], encoding="utf-8-sig", newline="") as fh:
        table = list(csv.reader(fh))
    assert table[0] == ["節", "項目", "値", "単位"] and ["空力結果", "Drag (抗力)", NOT_COMPUTED, "N"] in table
    assert files["summary"].read_bytes().startswith(b"\xef\xbb\xbf")        # Excel が文字化けしないよう BOM 付き


def test_report_with_results(tmp_path):
    summary = make_results(tmp_path / "results")
    summary["parts"] = [{"patch": "Body", "part_name": "Body", "drag": 90.0, "side": 1.0, "lift": -280.0, "downforce": 280.0,
                         "drag_pressure": 80.0, "drag_viscous": 10.0}]
    history = fake_history()
    forces = np.column_stack([history["iteration"], history["drag"], history["side"], history["lift"], np.zeros((400, 3))])
    residuals = np.column_stack([history["iteration"], *history["residuals"].values()])
    write_history(tmp_path / "results" / "history.csv", list(history["residuals"]), residuals, forces)
    summary["files"]["history"] = "history.csv"
    data = snapshot(summary, state="current", area=2.0, history=read_history(tmp_path / "results" / "history.csv"))
    files = export_report(data, tmp_path / "report", screenshot_png=b"\x89PNG\r\n\x1a\nfake")
    assert set(files) == {"html", "summary", "parts", "history", "view", "json"}

    q = 0.5 * RHO * U0 ** 2
    values = rows(build_sections(data))
    assert values[("空力結果", "Drag (抗力)")] == "100.00" and values[("空力結果", "Downforce (下向き正)")] == "300.00"
    assert values[("空力結果", "Lift (揚力・上向き正)")] == "-300.00" and values[("空力結果", "抗力の粘性成分")] == "15.00"
    assert values[("空力結果", "Cd")] == f"{100.0 / (q * 2.0):.4f}" and values[("空力結果", "Cl (上向き正)")] == f"{-300.0 / (q * 2.0):.4f}"
    # 「計算の状態」と「収束判定」は別の項目
    assert values[("計算の状態と収束", "計算の状態")] == "計算完了 (指定反復数に到達)"
    assert values[("計算の状態と収束", "収束判定")] == "工学的に十分収束"
    assert values[("計算の状態と収束", "収束の確認: 抗力の安定")].startswith("✔ 満たす")
    assert values[("残差と質量収支", "最終残差 p")] == "5.000e-04"

    html = files["html"].read_text(encoding="utf-8")
    assert "計算の状態 (指定した反復まで計算できたか)" in html and "収束判定 (工学的に十分収束したか)" in html
    for chart_id in ("chart-residuals", "chart-drag", "chart-downforce"):
        assert f'id="{chart_id}"' in html and f'data-for="{chart_id}"' in html
    assert "data:image/png;base64," in html and files["view"].read_bytes().startswith(b"\x89PNG")
    assert "パーツ別の荷重" in html and "http://" not in html and "https://" not in html      # 外部への参照なし (単体で開ける)
    # CSV: 桁区切りなしの数値、パーツ別の荷重、履歴はそのままコピー
    with open(files["parts"], encoding="utf-8-sig", newline="") as fh:
        parts = list(csv.reader(fh))
    assert parts[0][:3] == ["パーツ", "パッチ", "Drag [N]"] and parts[1][:4] == ["Body", "Body", "90.0", "280.0"]
    assert parts[1][8] == "110.0"                                             # y+ 平均
    assert files["history"].read_bytes() == (tmp_path / "results" / "history.csv").read_bytes()
    stored = json.loads(files["json"].read_text(encoding="utf-8"))
    assert stored["solver"]["summary"]["forces"]["drag"] == 100.0 and "history" not in stored["solver"]
    assert stored["report_version"] == 1 and stored["sections"][0]["title"] == "計算の状態と収束"


def test_item_names_are_unique_within_a_section(tmp_path):
    """CSV を「節 + 項目」で引けるよう、同じ節に同じ項目名を 2 つ作らない。"""
    summary = make_results(tmp_path / "results")
    for data in (snapshot(), snapshot(summary, "current")):
        keys = [(block["title"], label) for block in build_sections(data) for label, _value, _unit in block["rows"]]
        duplicated = {key for key in keys if keys.count(key) > 1 and not key[1].startswith("品質の所見")}
        assert not duplicated, duplicated


def test_coefficients_follow_the_current_reference_area(tmp_path):
    summary = make_results(tmp_path / "results")
    q = 0.5 * RHO * U0 ** 2
    assert coefficients(summary, 4.0)["cd"] == pytest.approx(100.0 / (q * 4.0))
    values = rows(build_sections(snapshot(summary, "current", area=None)))
    assert values[("空力結果", "Cd")] == "基準面積が未設定" and values[("空力結果", "Drag (抗力)")] == "100.00"
    assert coefficients(None, 2.0) == {"cd": None, "cl": None, "cs": None}


def test_failed_run_reports_failure_not_numbers(tmp_path):
    summary = make_results(tmp_path / "results", status="failed")
    summary.update(status_label="計算失敗", forces=None, ranges=None, error="FOAM FATAL ERROR\nfloating point exception",
                   convergence={"level": "diverged", "label": "発散", "message": "計算が発散しました。", "checks": []})
    data = snapshot(summary, "current")
    values = rows(build_sections(data))
    for label in ("Drag (抗力)", "Downforce (下向き正)", "Cd", "Cl (上向き正)", "最大流速"):
        assert values[("空力結果", label)] == FAILED
    assert "FOAM FATAL ERROR" in values[("計算の状態と収束", "ソルバーのエラー")]
    files = export_report(data, tmp_path / "report")
    html = files["html"].read_text(encoding="utf-8")
    assert "parts" not in files and html.count(FAILED) >= 5 and "発散" in html
    assert not re.search(r'class="value">[-\d]', html)                        # 見出しのタイルに数値を出していない


def test_stale_and_unconverged_results_carry_warnings(tmp_path):
    summary = make_results(tmp_path / "results")
    summary["convergence"] = {"level": "not_converged", "label": "未収束", "message": "基準を満たしていません。", "checks": []}
    summary["notes"] = ["境界層メッシュがありません。粘性抗力・剥離位置・Cd の精度は限定的です。"]
    notes = accuracy_notes(snapshot(summary, "stale"))
    assert any("前の条件で計算" in note for note in notes)
    assert any("未収束" in note and "工学的に十分収束したことは別" in note for note in notes)
    assert any("境界層メッシュがありません" in note for note in notes)
    assert accuracy_notes(snapshot(summary | {"convergence": {"level": "converged", "label": "工学的に十分収束"}, "notes": []}, "current")) == []
    html = export_report(snapshot(summary, "stale"), tmp_path / "report")["html"].read_text(encoding="utf-8")
    assert "精度に関する注意" in html and "境界層メッシュがありません" in html and "前の条件で計算" in html


def test_html_escapes_names_from_files(tmp_path):
    summary = make_results(tmp_path / "results")
    summary["parts"] = [{"patch": "p", "part_name": "<script>alert(1)</script>", "drag": 1.0, "side": 0.0, "lift": 0.0, "downforce": 0.0,
                         "drag_pressure": 1.0, "drag_viscous": 0.0}]
    parts = [{"name": "<img src=x onerror=alert(2)>", "file": "a&b.step", "unit": "mm", "analysed": False,
              "position": (0, 0, 0), "rotation": (0, 0, 0)}]
    data = snapshot(summary, "current", parts=parts)
    data["project_name"] = "</title><script>alert(3)</script>"
    html = export_report(data, tmp_path / "report")["html"].read_text(encoding="utf-8")
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html and "a&amp;b.step" in html and "— (非表示)" in html


def test_charts(tmp_path):
    history = fake_history(2000)
    charts = history_charts(history)
    assert charts.count("<svg") == 3 and charts.count('class="legend"') == 1      # 複数系列のグラフだけ凡例を付ける
    # 点は間引くが、最初と最後は残す
    payload = json.loads(re.search(r'data-for="chart-drag">(.*?)</script>', charts).group(1))
    assert len(payload["x"]) <= 600 and payload["x"][0] == 1.0 and payload["x"][-1] == 2000.0
    assert payload["series"][0]["name"] == "Drag" and payload["series"][0]["values"][-1] == pytest.approx(history["drag"][-1])
    # 荷重の縦軸は後半の変動に合わせる (最初の大きな過渡に合わせない)
    drag_svg = re.search(r'<svg class="chart" id="chart-drag".*?</svg>', charts, re.S).group(0)
    ticks = [float(t.replace(",", "")) for t in re.findall(r'class="tick"[^>]*text-anchor="end">([-\d.,]+)<', drag_svg)]
    assert max(ticks) < 150 and min(ticks) > 50
    # 残差は対数軸。0 以下や非数は線を切って描く
    chart = line_chart("c", "t", [1, 2, 3, 4], [{"name": "a", "values": [1.0, float("nan"), 0.0, 1e-3], "slot": 1}], log_y=True)
    assert chart.count("<polyline") == 2 and "1e-3" in chart and "nan" not in chart.lower()
    assert history_charts(None) == "" and history_charts({"iteration": [1.0]}) == ""
    assert line_chart("c", "t", [1, 2], [{"name": "a", "values": [float("nan")] * 2, "slot": 1}]) == ""
