"""テスト用 3D データ（真値が既知）を生成する.

    python samples/generate_samples.py [出力フォルダ]

出力（既定: samples/output/）:
  plane_100x100.stl               100×100mm の平面
  cylinder_d50.stl                直径50mm・長さ100mm の円筒（側面）
  plate_hole_d20.stl              穴径20mm のプレート (100×100×10)
  bracket_multi_holes.stl         穴が複数あるブラケット（ベース4穴 + フランジ1穴）
  multi_cylinder_part.stl         複数の円筒から構成された部品（段付きシャフト + 直交ボス）
  bracket_noisy_points.ply        ノイズを付加した点群（σ=0.05mm）
  partial_cylinder_120deg.stl     部分的に欠損した円筒（120°のみ）
  pair_source.ply / pair_target.ply  位置と回転を変えた点群のペア
  truth.json                      すべての真値（寸法・中心・軸・変換行列）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ljm_re.core import synthetic as S  # noqa: E402
from ljm_re.io.mesh_io import export_mesh  # noqa: E402


def main(out_dir=None):
    out = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)
    truth = {"units": "mm"}

    m = S.make_plane_mesh(100.0, 50)
    export_mesh(os.path.join(out, "plane_100x100.stl"), m.vertices, m.faces)
    truth["plane_100x100"] = {"origin": [0, 0, 0], "normal": [0, 0, 1], "size": [100, 100]}

    m = S.make_tube_mesh(50.0, 100.0, 128, 20)
    export_mesh(os.path.join(out, "cylinder_d50.stl"), m.vertices, m.faces)
    truth["cylinder_d50"] = {"diameter": 50, "length": 100, "axis_point": [0, 0, 0], "axis": [0, 0, 1],
                             "note": "128分割の多角形近似（頂点は真円上）"}

    m = S.make_plate_with_holes(100, 100, 10, [(0, 0, 20)])
    export_mesh(os.path.join(out, "plate_hole_d20.stl"), m.vertices, m.faces)
    truth["plate_hole_d20"] = {"size": [100, 100, 10], "hole_center_top": [0, 0, 10], "hole_diameter": 20,
                               "hole_axis": [0, 0, 1]}

    bracket, tb = S.make_bracket()
    export_mesh(os.path.join(out, "bracket_multi_holes.stl"), bracket.vertices, bracket.faces)
    truth["bracket_multi_holes"] = tb

    m, tm = S.make_multi_cylinder_part()
    export_mesh(os.path.join(out, "multi_cylinder_part.stl"), m.vertices, m.faces)
    truth["multi_cylinder_part"] = {"cylinders": tm}

    pts, nrm = S.sample_surface(bracket, 80000, 0.05, seed=1)
    export_mesh(os.path.join(out, "bracket_noisy_points.ply"), pts, None)
    truth["bracket_noisy_points"] = {"noise_sigma_mm": 0.05, "n_points": 80000, "source": "bracket_multi_holes"}

    m = S.make_tube_mesh(50.0, 60.0, 128, 12, arc_deg=120.0)
    export_mesh(os.path.join(out, "partial_cylinder_120deg.stl"), m.vertices, m.faces)
    truth["partial_cylinder_120deg"] = {"diameter": 50, "arc_deg": 120, "axis": [0, 0, 1], "axis_point": [0, 0, 0]}

    src, tgt, T = S.registration_pair(bracket, 30000, 0.03, seed=2)
    export_mesh(os.path.join(out, "pair_source.ply"), src, None)
    export_mesh(os.path.join(out, "pair_target.ply"), tgt, None)
    truth["registration_pair"] = {"T_target_from_source": T.tolist(), "noise_sigma_mm": 0.03,
                                  "rotation_xyz_deg": [10.0, -5.0, 20.0], "translation_mm": [15.0, -8.0, 4.0]}

    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    print(f"サンプルデータを出力しました: {out}")
    for n in sorted(os.listdir(out)):
        print("  ", n, f"{os.path.getsize(os.path.join(out, n)) / 1024:.0f} KB")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
