"""STL 形状のメッシュ生成 (snappyHexMesh) と、空力荷重の前後配分。

単体テストは OpenFOAM なしで動く。実際に snappyHexMesh と解析を動かすテストは OpenFOAM がある環境だけで動く。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from ljmcfd.mesh.settings import MeshSettings
from ljmcfd.mesh.snappy import (
    SNAPPY_PRESETS, block_mesh_dict, parse_checkmesh, parse_layers, read_boundary, snappy_dict, snappy_plan, write_binary_stl,
)
from ljmcfd.solver.environment import native_openfoam_available
from ljmcfd.solver.results import axle_loads, force_summary

needs_openfoam = pytest.mark.skipif(not native_openfoam_available(), reason="OpenFOAM がこの環境にありません")

CAR = (-0.896, 3.774, -1.0, 1.0, 0.0, 1.43)          # 全長 4.67 m の車両 (前輪軸 x = 0)
DOMAIN = (-14.9, 31.8, -7.0, 7.0, 0.0, 7.15)

CHECKMESH = """
Mesh stats
    points:           367655
    faces:            1004616
    internal faces:   956369
    cells:            318840
    faces per cell:   6.150373228
Overall number of cells of each type:
    hexahedra:     296071
    prisms:        3108
    wedges:        0
    pyramids:      0
    tet wedges:    15
    tetrahedra:    0
    polyhedra:     19646
Checking geometry...
    Min volume = 4.160142491e-07. Max volume = 0.8293947287.  Total volume = 4668.164371.  Cell volumes OK.
    Mesh non-orthogonality Max: 62.69506924 average: 7.785060009
    Non-orthogonality check OK.
 ***Max skewness = 6.90802307, 87 highly skew faces detected which may impair the quality of the results

Failed 1 mesh checks.
"""

LAYERS = """
Doing final balancing
patch                faces    layers   overall thickness
                                       [m]       [%]
-----                -----    ------   ---       ---
isf_cfd_body_only_mm 33366    1.89     0.0185    90.5
isf_wheels_mm        4734     1.44     0.0105    51.6

Layer mesh : cells:318840  faces:1004616  points:367655
"""


def test_plan_follows_the_presets():
    sizes = {}
    for preset in SNAPPY_PRESETS:
        settings = MeshSettings()
        settings.apply_preset(preset, CAR)
        plan = snappy_plan(settings, CAR, DOMAIN)
        sizes[preset] = plan["surface_size"]
        assert plan["surface_size"] == pytest.approx(4.67 / SNAPPY_PRESETS[preset])
        # 背景の格子 = 表面サイズ × 2^レベル (表面との比が 2 のべき乗)。背景は全長の 1/6 前後
        assert plan["background"] == pytest.approx(plan["surface_size"] * 2 ** plan["level"])
        assert 4.67 / 12 < plan["background"] < 4.67 / 3
        assert plan["cells"] == [int(np.ceil((DOMAIN[2 * i + 1] - DOMAIN[2 * i]) / plan["background"] - 1e-6)) for i in range(3)]
        # 細分化の領域は計算領域の中で、外側ほど粗い
        levels = [lvl for _name, _box, lvl in plan["regions"]]
        assert levels == sorted(levels, reverse=True) and all(lvl >= 1 for lvl in levels)
        for _name, box, _lvl in plan["regions"]:
            assert all(DOMAIN[2 * i] <= box[2 * i] < box[2 * i + 1] <= DOMAIN[2 * i + 1] for i in range(3))
        assert plan["layers"] >= 2 and plan["estimated_cells"] > 0
    assert sizes["coarse"] == pytest.approx(2 * sizes["standard"])               # 粗い → 標準 で 1/2
    assert sizes["standard"] / sizes["fine"] == pytest.approx(2 ** 0.5, rel=0.01)  # 標準 → 細かい で 1/√2
    custom = MeshSettings(preset="custom", surface_size=0.02)
    assert snappy_plan(custom, CAR, DOMAIN)["surface_size"] == pytest.approx(0.02)


def test_locked_reference_keeps_the_domain_and_mesh_when_parts_are_added():
    """比較用の固定: フラップを足して外形が変わっても、計算領域とメッシュの細かさは基準ケースのまま。"""
    from ljmcfd.tunnel import spec as ts

    tunnel = ts.TunnelSpec()
    ts.apply_mode_defaults(tunnel, CAR)
    settings = MeshSettings()
    settings.apply_preset("standard", CAR)
    base_domain, base_plan = tunnel.domain_bounds(CAR), snappy_plan(settings, CAR, tunnel.domain_bounds(CAR))
    with_flap = (-0.896, 4.10, -1.0, 1.0, 0.0, 1.62)                   # 後方・上方へはみ出すフラップ
    unlocked = snappy_plan(settings, with_flap, tunnel.domain_bounds(with_flap))
    assert tunnel.domain_bounds(with_flap) != base_domain and unlocked["regions"] != base_plan["regions"]   # 固定しないと変わる
    tunnel.fixed_domain = list(base_domain)
    settings.reference_bounds = list(CAR)
    settings.apply_preset("standard", with_flap)                       # プリセットを選び直しても基準寸法で決まる
    locked = snappy_plan(settings, with_flap, tunnel.domain_bounds(with_flap))
    assert tunnel.domain_bounds(with_flap) == pytest.approx(base_domain)
    assert locked["surface_size"] == base_plan["surface_size"] and locked["cells"] == base_plan["cells"]
    assert locked["regions"] == base_plan["regions"]
    # 保存して読み直しても固定は残る
    assert ts.TunnelSpec.from_dict(tunnel.to_dict()).fixed_domain == pytest.approx(list(base_domain))
    assert MeshSettings.from_dict(settings.to_dict()).reference_bounds == list(CAR)


def test_part_size_factor_maps_to_refinement_levels():
    from ljmcfd.mesh.snappy import part_level

    assert part_level(6, 1.0) == 6 and part_level(6, 0.5) == 7 and part_level(6, 0.25) == 8 and part_level(6, 2.0) == 5
    assert part_level(6, 0.01) == 10 and part_level(1, 8.0) == 1                 # 上限 +4、下限 1
    plan = snappy_plan(MeshSettings(), CAR, DOMAIN)
    text = snappy_dict({"body": plan["level"], "flap": plan["level"] + 2}, plan, (-14.0, 0.01, 7.0))
    assert f"flap {{ level ({plan['level'] + 2} {plan['level'] + 2});" in text and f'file "flap.eMesh"; level {plan["level"] + 2};' in text
    assert f"body {{ level ({plan['level']} {plan['level']});" in text


def test_dictionaries():
    plan = snappy_plan(MeshSettings(), CAR, DOMAIN)
    block = block_mesh_dict(DOMAIN, plan["cells"], True)
    for name in ("inlet", "outlet", "sideL", "sideR", "top", "ground"):
        assert f"    {name}" in block
    assert "type wall" in block and f"({plan['cells'][0]} {plan['cells'][1]} {plan['cells'][2]})" in block
    assert "bottom" in block_mesh_dict(DOMAIN, plan["cells"], False)
    text = snappy_dict(["body", "wheels"], plan, (-14.0, 0.01, 7.0))
    assert 'file "body.stl"' in text and 'file "wheels.eMesh"' in text and "insidePoint (-14 0.01 7);" in text
    assert f"level ({plan['level']} {plan['level']})" in text and "nSurfaceLayers" in text and "addLayers       on;" in text
    for name, _box, lvl in plan["regions"]:
        assert f"{name} {{ mode inside; level {lvl}; }}" in text
    assert text.isascii()                                                       # OpenFOAM の設定ファイルは ASCII


def test_log_parsers(tmp_path):
    q = parse_checkmesh(CHECKMESH)
    assert q["n_cells"] == 318840 and q["n_points"] == 367655 and q["n_boundary_faces"] == 1004616 - 956369
    assert q["cell_counts"] == {"hex": 296071, "prism": 3108, "tetWedge": 15, "polyhedra": 19646}
    assert q["non_ortho_max"] == pytest.approx(62.695, abs=1e-3) and q["non_ortho_mean"] == pytest.approx(7.785, abs=1e-3)
    assert q["skewness_max"] == pytest.approx(6.908, abs=1e-3) and q["n_skewness_over_4"] == 87
    assert q["volume_min"] == pytest.approx(4.16e-7, rel=1e-3) and q["volume_total"] == pytest.approx(4668.164, rel=1e-6)
    assert not q["checkmesh_ok"] and q["checkmesh_failed"] == 1
    layers = parse_layers(LAYERS)
    assert layers == {"isf_cfd_body_only_mm": {"faces": 33366, "layers": 1.89, "thickness": 0.0185, "percent": 90.5},
                      "isf_wheels_mm": {"faces": 4734, "layers": 1.44, "thickness": 0.0105, "percent": 51.6}}
    boundary = tmp_path / "boundary"
    boundary.write_text("FoamFile\n{\n    format binary;\n}\n// * * //\n2\n(\n    inlet\n    {\n        type patch;\n"
                        "        nFaces 120;\n        startFace 956369;\n    }\n    body\n    {\n        type wall;\n"
                        "        inGroups List<word> 1(wall);\n        nFaces 33366;\n        startFace 956489;\n    }\n)\n")
    assert read_boundary(boundary) == [{"name": "inlet", "type": "patch", "size": 120, "start": 956369},
                                       {"name": "body", "type": "wall", "size": 33366, "start": 956489}]


def test_binary_stl_round_trip(tmp_path):
    from ljmcfd.cad.surface import read_stl

    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=float)
    write_binary_stl(tmp_path / "t.stl", triangles, "t")
    data = (tmp_path / "t.stl").read_bytes()
    assert len(data) == 84 + 50 * 2 and struct.unpack("<I", data[80:84])[0] == 2
    surface = read_stl(tmp_path / "t.stl")[0][1]
    assert surface.n_cells == 2 and {tuple(p) for p in np.asarray(surface.points)} == {tuple(p) for p in triangles.reshape(-1, 3)}


def test_axle_loads_split_downforce_between_axles():
    """前後の荷重配分: 接地点まわりのモーメントの釣り合い。解析的に分かる力で確かめる。"""
    wheelbase = 2.73

    def forces_for(points_forces):
        f = np.sum([force for _p, force in points_forces], axis=0)
        m = np.sum([np.cross(p, force) for p, force in points_forces], axis=0)          # 原点まわり
        return {"lift": float(f[2]), "drag": float(f[0]), "moment": m.tolist()}

    # 後輪軸の真上に 1000 N のダウンフォース → すべて後輪
    loads = axle_loads(forces_for([((wheelbase, 0, 0.5), (0, 0, -1000.0))]), 0.0, wheelbase)
    assert loads["rear"] == pytest.approx(1000.0) and loads["front"] == pytest.approx(0.0, abs=1e-9)
    # 前後の中間 → 半分ずつ。前輪軸の位置を原点からずらしても同じ (基準点を移して計算する)
    shifted = axle_loads(forces_for([((1.0 + wheelbase / 2, 0, 0.3), (0, 0, -800.0))]), 1.0, wheelbase)
    assert shifted["front"] == pytest.approx(400.0) and shifted["rear"] == pytest.approx(400.0) and shifted["front_share"] == pytest.approx(0.5)
    # 抗力は高さ h で働くと後輪側の荷重を増やし、前輪側を同じだけ減らす (合計は変わらない)
    drag = axle_loads(forces_for([((1.0, 0, 0.6), (300.0, 0, 0))]), 0.0, wheelbase)
    assert drag["rear"] == pytest.approx(300.0 * 0.6 / wheelbase) and drag["front"] == pytest.approx(-drag["rear"]) and drag["total"] == 0.0
    assert drag["front_share"] is None
    # 係数: Cl 前 + Cl 後 = Cl
    coeff = axle_loads(forces_for([((0.5, 0, 0.5), (0, 0, -600.0))]), 0.0, wheelbase, dynamic_pressure=500.0, area=2.0)
    assert coeff["cl_front"] + coeff["cl_rear"] == pytest.approx(-600.0 / 1000.0)
    # ホイールベースが無い・モーメントが無い (古い結果) と計算しない
    assert axle_loads(forces_for([((1, 0, 0), (0, 0, -1.0))]), 0.0, None) is None
    assert axle_loads({"lift": -1.0}, 0.0, wheelbase) is None


def test_force_summary_keeps_moments():
    rows = np.zeros((10, 13))
    rows[:, 0] = np.arange(1, 11)
    rows[:, 3] = -100.0         # 圧力による Fz
    rows[:, 8] = 50.0           # 圧力による My
    rows[:, 11] = 5.0           # 粘性による My
    summary = force_summary(rows, 5, 1.2, 30.0, 2.0)
    assert summary["moment"] == pytest.approx([0.0, 55.0, 0.0]) and summary["downforce"] == pytest.approx(100.0)


# ----------------------------------------------------------------------------- OpenFOAM を使う結合テスト

def _box_and_wheels(tmp_path: Path) -> tuple[Path, Path]:
    """mm 単位の STL: 車体 (直方体、地面から離れている) と、地面に線で接する円柱 4 本。"""
    import pyvista as pv

    def triangles(surface):
        surface = surface.triangulate().clean()
        return np.asarray(surface.points)[np.asarray(surface.faces).reshape(-1, 4)[:, 1:]]

    body = pv.Box(bounds=(-600, 2600, -600, 600, 250, 1100), level=4)
    wheels = [pv.Cylinder(center=(x, side * 650, 250), direction=(0, 1, 0), radius=250, height=200, resolution=48, capping=True)
              for x in (0, 2000) for side in (1, -1)]
    body_path, wheel_path = tmp_path / "body_mm.stl", tmp_path / "wheels_mm.stl"
    write_binary_stl(body_path, triangles(body), "body")
    write_binary_stl(wheel_path, np.vstack([triangles(w) for w in wheels]), "wheels")
    return body_path, wheel_path


@pytest.fixture()
def window(qtbot, tmp_path, monkeypatch):
    """メインウィンドウ (test_gui.py と同じ。ディスプレイが無い環境でも、このファイルの単体テストは動くようにここで定義)。"""
    import os
    import sys

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("ディスプレイが無いため GUI テストをスキップ")
    monkeypatch.setenv("LJMCFD_HOME", str(tmp_path / "home"))
    from ljmcfd.gui import theme
    from ljmcfd.gui.main_window import MainWindow

    win = MainWindow()
    win.setStyleSheet(theme.STYLESHEET)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win._modified = False
    win.close()


@needs_openfoam
@pytest.mark.gui
def test_stl_vehicle_from_import_to_axle_loads(qtbot, window, tmp_path):
    """STL (mm、車体とタイヤが別ファイル) を読み込み、snappyHexMesh でメッシュを作り、解析して前後配分まで出す。"""
    TIMEOUT = 120_000
    body, wheels = _box_and_wheels(tmp_path)
    window.import_files([str(body), str(wheels)])
    qtbot.waitUntil(lambda: len(window.scene.parts) == 2 and not window.jobs.is_busy(), timeout=TIMEOUT)
    assert [p.unit for p in window.scene.parts] == ["mm", "mm"]                  # mm 単位を自動で認識
    bounds = window.scene.analysis_bounds()
    assert bounds[0] == pytest.approx(-0.6) and bounds[1] == pytest.approx(2.6) and bounds[4] == pytest.approx(0.0, abs=1e-3)
    panel = window.mesh_panel
    assert panel.uses_snappy() and "snappyHexMesh" in panel.mesher_info.text() and not panel.bl_box.isEnabled()
    tunnel = window.scene.tunnel
    window.tunnel_panel.speed_kmh.setValue(100.0)
    assert window.generate_tunnel()
    qtbot.waitUntil(lambda: tunnel.frontal_area is not None and not window.jobs.is_busy(), timeout=TIMEOUT)
    settings = window.scene.mesh_settings
    settings.preset = "custom"
    settings.surface_size = 0.1
    window.scene.notify_mesh_settings_changed()
    window.scene.solver_settings.n_procs = 2
    window.scene.solver_settings.iterations = 40
    window.scene.notify_solver_settings_changed()
    assert window.generate_mesh()
    qtbot.waitUntil(lambda: window._mesh_job is None, timeout=TIMEOUT * 5)
    report = window.scene.mesh_report
    assert report is not None and report["mesher"] == "snappyHexMesh" and report["quality"]["n_cells"] > 10000
    kinds = {p["name"]: p["kind"] for p in report["patches"]}
    assert kinds == {"inlet": "tunnel", "outlet": "tunnel", "sideL": "tunnel", "sideR": "tunnel", "top": "tunnel",
                     "ground": "tunnel", "body_mm": "part", "wheels_mm": "part"}
    # 接地しているタイヤだけを沈め、車体 (地面から離れている) は動かさない
    assert report["part_sinks"] == {"body_mm": 0.0, "wheels_mm": pytest.approx(tunnel.contact_sink)}
    assert report["quality"]["cell_counts"]["hex"] > 0.5 * report["quality"]["n_cells"]
    assert report["boundary_layer"]["coverage_percent"] > 30.0
    out = Path(report["out_dir"])
    assert (out / "polyMesh" / "faces").is_file() and (out / "volume.vtk").is_file() and (out / "log.snappyHexMesh").is_file()
    assert "snappyHexMesh" in window.mesh_quality.toPlainText()                # 品質表示が snappy のレポートを表示できる
    window.mesh_display.section_check.setChecked(True)                         # 体積メッシュの断面も表示できる
    assert "mesh_section" in window.viewport.plotter.actors

    assert window.run_solver()
    qtbot.waitUntil(lambda: window._solver_job is None and not window.jobs.is_busy(), timeout=TIMEOUT * 5)
    summary = window.scene.solver_summary
    assert summary["status"] in ("completed", "residual_stop") and summary["forces"]["moment"] is not None
    labels = window.results_panel.result_labels
    assert labels["downforce_front"].text() == "ホイールベースが未設定"
    tunnel.reference_length = 2.0
    tunnel.front_axle_x = 0.0
    window.scene.notify_tunnel_changed()
    front, rear = float(labels["downforce_front"].text()), float(labels["downforce_rear"].text())
    assert front + rear == pytest.approx(float(labels["downforce"].text()), abs=0.02)
    assert labels["front_share"].text().endswith("%")
    stored = json.loads((Path(summary["results_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert len(stored["forces"]["moment"]) == 3
