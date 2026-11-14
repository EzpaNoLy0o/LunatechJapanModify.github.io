"""メッシュ品質評価。OpenFOAM の checkMesh と同じ定義の指標を polyMesh から直接計算する。

* **非直交性** (non-orthogonality): 面法線と、両側のセル中心を結ぶ線のなす角。
  70° を超えると離散化誤差と発散の危険が増える。
* **歪度** (skewness): セル中心を結ぶ線が面を貫く点と、面中心とのずれ。目安は 4 以下。
* **負体積・開いたセル**: 1 つでもあれば解析できない。
"""

from __future__ import annotations

import numpy as np

from .polymesh import PolyMesh

NON_ORTHO_WARN = 70.0      # [deg] OpenFOAM の既定の警告しきい値
NON_ORTHO_FAIL = 85.0
SKEWNESS_WARN = 4.0
OPEN_CELL_TOL = 1.0e-6


def face_geometry(mesh: PolyMesh) -> tuple[np.ndarray, np.ndarray]:
    """面中心 (F, 3) と面積ベクトル (F, 3)。四角形は中心まわりの三角形へ分割して求める。"""
    pts = mesh.points
    faces = mesh.faces
    centres = np.empty((len(faces), 3))
    areas = np.empty((len(faces), 3))
    tri = faces[:, 3] < 0
    if tri.any():
        p = pts[faces[tri, :3]]
        centres[tri] = p.mean(axis=1)
        areas[tri] = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if (~tri).any():
        p = pts[faces[~tri]]                        # (n, 4, 3)
        estimate = p.mean(axis=1)
        normal_sum = np.zeros((len(p), 3))
        weight_sum = np.zeros(len(p))
        centre_sum = np.zeros((len(p), 3))
        for i in range(4):
            a, b = p[:, i], p[:, (i + 1) % 4]
            n = np.cross(b - a, estimate - a)      # 部分三角形の面積ベクトルの 2 倍
            w = np.linalg.norm(n, axis=1)
            normal_sum += n
            weight_sum += w
            centre_sum += w[:, None] * (a + b + estimate) / 3.0
        areas[~tri] = 0.5 * normal_sum
        safe = np.where(weight_sum > 0, weight_sum, 1.0)
        centres[~tri] = np.where(weight_sum[:, None] > 0, centre_sum / safe[:, None], estimate)
    return centres, areas


def cell_geometry(mesh: PolyMesh, face_centres: np.ndarray, face_areas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """セル中心 (C, 3)・体積 (C,)・開き具合 (C,) を、面を底面とするピラミッド分割で求める。

    開き具合 = |Σ 外向き面積ベクトル| / Σ |面積ベクトル| (閉じたセルでは 0)。
    """
    n_cells, n_int = mesh.n_cells, mesh.n_internal
    own, nei = mesh.owner, mesh.neighbour

    def scatter(values_owner, values_nei=None):
        out = np.zeros((n_cells,) + values_owner.shape[1:])
        np.add.at(out, own, values_owner)
        if values_nei is not None:
            np.add.at(out, nei, values_nei)
        return out

    counts = np.bincount(own, minlength=n_cells) + np.bincount(nei, minlength=n_cells)
    estimate = scatter(face_centres, face_centres[:n_int]) / np.maximum(counts, 1)[:, None]

    pyr_own = np.einsum("ij,ij->i", face_areas, face_centres - estimate[own])                  # 体積の 3 倍
    pyr_nei = np.einsum("ij,ij->i", face_areas[:n_int], estimate[nei] - face_centres[:n_int])
    centroid_own = 0.75 * face_centres + 0.25 * estimate[own]
    centroid_nei = 0.75 * face_centres[:n_int] + 0.25 * estimate[nei]
    vol3 = scatter(pyr_own, pyr_nei)
    weighted = scatter(pyr_own[:, None] * centroid_own, pyr_nei[:, None] * centroid_nei)
    safe = np.where(np.abs(vol3) > 0, vol3, 1.0)
    centres = np.where((np.abs(vol3) > 0)[:, None], weighted / safe[:, None], estimate)

    area_sum = scatter(face_areas, -face_areas[:n_int])
    magnitude = np.linalg.norm(face_areas, axis=1)
    magnitude_sum = scatter(magnitude, magnitude[:n_int])
    openness = np.linalg.norm(area_sum, axis=1) / np.maximum(magnitude_sum, 1e-300)
    return centres, vol3 / 3.0, openness


def _cell_slice(mesh: PolyMesh, kind: str) -> slice:
    """セル番号のうち ``kind`` のセルが占める範囲 (セルは種別ごとに連続して番号付けされる)。"""
    from .polymesh import CELL_ORDER

    start = 0
    for name in CELL_ORDER:
        count = int(mesh.cell_counts.get(name, 0))
        if name == kind:
            return slice(start, start + count)
        start += count
    return slice(0, 0)


def compute_quality(mesh: PolyMesh) -> dict:
    """品質指標を計算し、判定 (ok / warning / error) と日本語メッセージつきで返す。"""
    fc, fa = face_geometry(mesh)
    cc, volumes, openness = cell_geometry(mesh, fc, fa)
    n_int = mesh.n_internal
    own, nei = mesh.owner, mesh.neighbour
    magnitude = np.linalg.norm(fa, axis=1)

    # --- 内部面: 非直交性と歪度
    d = cc[nei] - cc[own[:n_int]]
    d_len = np.linalg.norm(d, axis=1)
    sf = fa[:n_int]
    dot = np.einsum("ij,ij->i", d, sf)
    cosine = dot / np.maximum(d_len * magnitude[:n_int], 1e-300)
    non_ortho = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def max_projection(face_slice, centres, direction):
        """面の各頂点 − 面中心 を direction 方向へ射影した長さの最大値。"""
        faces = mesh.faces[face_slice]
        valid = faces >= 0
        p = mesh.points[np.where(valid, faces, 0)] - centres[:, None, :]
        proj = np.abs(np.einsum("ijk,ik->ij", p, direction))
        return np.where(valid, proj, 0.0).max(axis=1)

    cpf = fc[:n_int] - cc[own[:n_int]]
    safe_dot = np.where(np.abs(dot) > 1e-300, dot, 1e-300)
    skew_vec = cpf - (np.einsum("ij,ij->i", sf, cpf) / safe_dot)[:, None] * d
    skew_len = np.linalg.norm(skew_vec, axis=1)
    direction = skew_vec / np.maximum(skew_len, 1e-300)[:, None]
    denom = np.maximum(0.2 * d_len, max_projection(slice(0, n_int), fc[:n_int], direction))
    skew_internal = skew_len / np.maximum(denom, 1e-300)

    # --- 境界面: 歪度 (セル中心から面へ下ろした垂線の足と面中心のずれ)
    bfa, bfc = fa[n_int:], fc[n_int:]
    if len(bfa):
        normal = bfa / np.maximum(magnitude[n_int:], 1e-300)[:, None]
        cpf_b = bfc - cc[own[n_int:]]
        height = np.einsum("ij,ij->i", normal, cpf_b)
        skew_vec_b = cpf_b - height[:, None] * normal
        skew_len_b = np.linalg.norm(skew_vec_b, axis=1)
        direction_b = skew_vec_b / np.maximum(skew_len_b, 1e-300)[:, None]
        denom_b = np.maximum(0.4 * np.abs(height), max_projection(slice(n_int, None), bfc, direction_b))
        skew_boundary = skew_len_b / np.maximum(denom_b, 1e-300)
    else:
        skew_boundary = np.zeros(0)

    # 隣接セルの体積比 (急なサイズ変化の指標)
    v_own, v_nei = np.abs(volumes[own[:n_int]]), np.abs(volumes[nei])
    ratio = np.maximum(v_own, v_nei) / np.maximum(np.minimum(v_own, v_nei), 1e-300)

    boundary_closure = float(np.linalg.norm(bfa.sum(axis=0)) / max(magnitude[n_int:].sum(), 1e-300)) if len(bfa) else 0.0
    patch_areas = {
        p["name"]: float(magnitude[p["start"]:p["start"] + p["size"]].sum()) for p in mesh.patches
    }
    skew_all_max = float(max(skew_internal.max(initial=0.0), skew_boundary.max(initial=0.0)))

    report = {
        "n_points": int(len(mesh.points)),
        "n_cells": int(mesh.n_cells),
        "n_faces": int(mesh.n_faces),
        "n_internal_faces": int(n_int),
        "n_boundary_faces": int(mesh.n_faces - n_int),
        "cell_counts": dict(mesh.cell_counts),
        "volume_total": float(volumes.sum()),
        "prism_volume": float(volumes[_cell_slice(mesh, "prism")].sum()),
        "volume_min": float(volumes.min()),
        "volume_max": float(volumes.max()),
        "n_negative_volumes": int((volumes <= 0).sum()),
        "n_open_cells": int((openness > OPEN_CELL_TOL).sum()),
        "max_openness": float(openness.max()),
        "boundary_closure": boundary_closure,
        "non_ortho_max": float(non_ortho.max(initial=0.0)),
        "non_ortho_mean": float(non_ortho.mean()) if len(non_ortho) else 0.0,
        "n_non_ortho_over_70": int((non_ortho > NON_ORTHO_WARN).sum()),
        "skewness_max": skew_all_max,
        "n_skewness_over_4": int((skew_internal > SKEWNESS_WARN).sum() + (skew_boundary > SKEWNESS_WARN).sum()),
        "volume_ratio_max": float(ratio.max(initial=1.0)),
        "patch_areas": patch_areas,
        "n_unmatched_faces": int(mesh.n_unmatched_faces),
        "non_ortho_histogram": np.histogram(non_ortho, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 181])[0].tolist(),
    }
    report["summary"] = summarize(report)
    return report


def summarize(report: dict) -> dict:
    """指標から判定と日本語メッセージを作る。"""
    level = "ok"
    messages: list[str] = []
    order = {"ok": 0, "warning": 1, "error": 2}

    def add(new_level: str, text: str) -> None:
        nonlocal level
        if order[new_level] > order[level]:
            level = new_level
        messages.append(text)

    if report["n_negative_volumes"]:
        add("error", f"体積が負またはゼロのセルが {report['n_negative_volumes']:,} 個あります。このメッシュでは解析できません。")
    if report["n_open_cells"]:
        add("error", f"閉じていないセルが {report['n_open_cells']:,} 個あります。")
    if report["n_unmatched_faces"]:
        add("error", f"どの境界パッチにも属さない境界面が {report['n_unmatched_faces']:,} 面あります。")
    if report["boundary_closure"] > 1e-6:
        add("error", "計算領域の外周が閉じていません。")
    if report["non_ortho_max"] > NON_ORTHO_FAIL:
        add("error", f"非直交性の最大値 {report['non_ortho_max']:.1f}° が {NON_ORTHO_FAIL:.0f}° を超えています。発散する可能性が高いメッシュです。")
    elif report["n_non_ortho_over_70"]:
        add("warning", f"非直交性が 70° を超える面が {report['n_non_ortho_over_70']:,} 面あります (最大 {report['non_ortho_max']:.1f}°)。"
                       "非直交補正を増やすか、メッシュを細かくしてください。")
    if report["skewness_max"] > SKEWNESS_WARN:
        add("warning", f"歪度の最大値 {report['skewness_max']:.2f} が目安の {SKEWNESS_WARN:.0f} を超えています ({report['n_skewness_over_4']:,} 面)。")
    if level == "ok":
        messages.insert(0, f"品質は良好です (非直交性 最大 {report['non_ortho_max']:.1f}° / 平均 {report['non_ortho_mean']:.1f}°、"
                           f"歪度 最大 {report['skewness_max']:.2f})。")
    return {"level": level, "messages": messages}
