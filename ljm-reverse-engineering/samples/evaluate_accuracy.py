"""真値が既知のモデルでフィッティング精度を評価し、表を出力する.

    python samples/evaluate_accuracy.py

ノイズなし / ノイズあり（σ = 0.02, 0.05 mm）を分けて評価する。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ljm_re.core import synthetic as S  # noqa: E402
from ljm_re.core.fitting import fit_cylinder, fit_plane  # noqa: E402
from ljm_re.core.hole import detect_holes_on_plane  # noqa: E402
from ljm_re.core.registration import icp  # noqa: E402
from ljm_re.core.geometry import euler_xyz_to_matrix  # noqa: E402


def ang(a, b):
    a, b = np.asarray(a) / np.linalg.norm(a), np.asarray(b) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def main():
    rows = []
    n_true = np.array([0.1, -0.2, 1.0])
    for noise in (0.0, 0.02, 0.05):
        p = S.plane_points(20000, 100, noise, normal=n_true, seed=1)
        f = fit_plane(p)
        rows.append(("平面 100×100", noise, "法線誤差 [deg]", ang(f.normal, n_true), f"RMS {f.stats.rms:.4f}"))
    for arc in (360, 180, 120, 60):
        for noise in (0.0, 0.02, 0.05):
            P, N = S.partial_cylinder_points(50, 60, arc, 6000, noise, axis=(0.2, 0.1, 1), seed=3)
            c = fit_cylinder(P, N)
            rows.append((f"円筒 φ50 ({arc}°)", noise, "直径誤差 [mm]", c.diameter - 50,
                         f"軸誤差 {ang(c.axis_dir, (0.2, 0.1, 1)):.4f}° / 1σ(D) {2 * c.sigma_radius:.4f}"))
    mesh, truth = S.make_bracket()
    for noise in (0.0, 0.02, 0.05):
        m = mesh.copy()
        if noise:
            rng = np.random.default_rng(4)
            m.vertices = m.vertices + m.vertex_normals * rng.normal(0, noise, (len(m.vertices), 1))
        holes = detect_holes_on_plane(np.asarray(m.vertices), np.asarray(m.faces), [0, 0, 8], [0, 0, 1],
                                      depth=0.5, abs_tol=0.3)
        errs, derr = [], []
        for t in truth["holes"][:4]:
            c = np.array(t["center_top"])
            if not holes:
                break
            h = min(holes, key=lambda h: np.linalg.norm(h.center - c))
            errs.append(np.linalg.norm(h.center[:2] - c[:2]))
            derr.append(h.diameter - t["diameter"])
        rows.append(("ブラケット穴中心(断面)", noise, "中心位置誤差 最大 [mm]", max(errs) if errs else float("nan"),
                     f"検出 {len(holes)}/4, 径誤差最大 {max(np.abs(derr)) if derr else float('nan'):.4f} mm"))
    from ljm_re.core.mesh_ops import estimate_normals
    for noise in (0.0, 0.03):
        tgt, _ = S.sample_surface(mesh, 40000, noise, 2)
        R = euler_xyz_to_matrix(2.0, -1.5, 3.0)
        t = np.array([1.5, -1.0, 0.8])
        src_true, _ = S.sample_surface(mesh, 20000, noise, 5)
        src = (src_true - t) @ R
        r = icp(src, tgt, estimate_normals(tgt), max_iterations=100)
        rot_err = np.degrees(np.arccos(np.clip((np.trace(r.transform[:3, :3].T @ R) - 1) / 2, -1, 1)))
        rows.append(("ICP 位置合わせ", noise, "並進誤差 [mm]", float(np.linalg.norm(r.transform[:3, 3] - t)),
                     f"回転誤差 {rot_err:.4f}° / RMS {r.rms:.4f} / 重複率 {r.fitness * 100:.1f}%"))

    print("| モデル | ノイズσ [mm] | 評価項目 | 誤差 | 備考 |")
    print("|---|---|---|---|---|")
    for name, noise, item, err, note in rows:
        print(f"| {name} | {noise:g} | {item} | {err:+.5f} | {note} |")


if __name__ == "__main__":
    main()
