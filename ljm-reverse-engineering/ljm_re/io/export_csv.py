"""CSV 出力（測定結果・穴中心・基準点などの座標一覧）.

Excel（日本語 Windows）で文字化けしないよう UTF-8 BOM 付きで出力する。
座標は指定した座標系（通常は作業座標系）で出力し、座標系名を各行に記録する。
"""
from __future__ import annotations

import csv
import datetime as _dt

import numpy as np

from ..core.coords import Frame
from ..model.objects import KIND_LABELS, MEASURE_LABELS

FEATURE_KINDS = ("point", "hole", "cylinder", "axis", "plane", "sphere", "cs")

HEADER = ["種類", "名前", "分類", "X", "Y", "Z", "方向X", "方向Y", "方向Z", "直径", "半径",
          "RMS残差", "最大残差", "使用点数", "インライア率", "円周カバー[deg]", "直径不確かさ1σ",
          "算出方法", "安全上重要", "実測確認", "実測値・確認メモ", "警告", "座標系", "元データ"]


def _r(v, d):
    if v is None:
        return ""
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def feature_rows(doc, frame: Frame, frame_name: str, decimals: int = 4, ids=None):
    rows = []
    objs = doc.objects(FEATURE_KINDS) if ids is None else [doc.get(i) for i in ids if doc.get(i)]
    for o in objs:
        if o.kind not in FEATURE_KINDS:
            continue
        p = d = None
        dia = rad = None
        cov = sig = None
        method = ""
        cat = getattr(o, "category", "")
        if o.kind == "point":
            p = o.position
        elif o.kind == "hole":
            p, d, dia = o.center, o.axis, o.diameter
            rad = dia / 2
            cov, sig, method = o.coverage_deg, o.sigma_diameter, o.method
        elif o.kind == "cylinder":
            a, b = o.end_points()
            p, d, dia, rad = 0.5 * (a + b), o.axis_dir, o.diameter, o.radius
            cov = o.coverage_deg
            sig = None if o.sigma_radius is None else 2 * o.sigma_radius
        elif o.kind == "axis":
            p, d = o.point, o.direction
        elif o.kind == "plane":
            p, d = o.center(), o.normal
        elif o.kind == "sphere":
            p, dia, rad = o.center, 2 * o.radius, o.radius
            sig = None if o.sigma_radius is None else 2 * o.sigma_radius
        elif o.kind == "cs":
            p, d = o.frame.origin, o.frame.z_axis
        pl = frame.to_local(p) if p is not None else [None] * 3
        dl = frame.dir_to_local(d) if d is not None else [None] * 3
        st = getattr(o, "stats", None) or {}
        rows.append([
            KIND_LABELS.get(o.kind, o.kind), o.name, cat,
            _r(pl[0], decimals), _r(pl[1], decimals), _r(pl[2], decimals),
            _r(dl[0], 6), _r(dl[1], 6), _r(dl[2], 6),
            _r(dia, decimals), _r(rad, decimals),
            _r(st.get("rms"), decimals), _r(st.get("max_abs"), decimals),
            st.get("n_used", ""), _r(st.get("inlier_ratio"), 3), _r(cov, 1), _r(sig, decimals),
            method or st.get("method", ""),
            "はい" if o.critical else "", "済" if o.verified else ("未確認" if o.critical else ""),
            o.verified_note, " / ".join(getattr(o, "warnings", []) or []),
            frame_name, o.meta.get("source_file", ""),
        ])
    return rows


def unverified_critical(doc) -> list:
    return [o for o in doc.objects() if o.critical and not o.verified]


def export_features_csv(doc, path, frame: Frame, frame_name: str, decimals: int = 4, ids=None):
    rows = feature_rows(doc, frame, frame_name, decimals, ids)
    crit = unverified_critical(doc)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# LJM Reverse Engineering 座標一覧", _dt.datetime.now().isoformat(timespec="seconds")])
        w.writerow(["# 単位", "mm / 方向は単位ベクトル", "座標系", frame_name])
        w.writerow(["# 注意", "表示桁数は測定精度を表しません。安全上重要な寸法は必ず実測で確認してください。"])
        if crit:
            w.writerow(["# 警告", f"実測未確認の安全上重要な要素が {len(crit)} 件あります: "
                        + ", ".join(o.name for o in crit)])
        w.writerow(HEADER)
        w.writerows(rows)
    return len(rows)


def export_measurements_csv(doc, path, fmt_decimals: int = 4):
    from ..model.formatting import NumberFormat
    fmt = NumberFormat(fmt_decimals, fmt_decimals)
    ms = doc.objects("measurement")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# LJM Reverse Engineering 測定結果", _dt.datetime.now().isoformat(timespec="seconds")])
        w.writerow(["名前", "種類", "主値", "単位", "結果", "距離[mm]", "角度[deg]", "同軸度[mm]",
                    "dX", "dY", "dZ", "測定時の座標系", "安全上重要", "実測確認", "実測値・確認メモ"])
        for m in ms:
            v, unit = m.main_value()
            r = m.result
            w.writerow([m.name, MEASURE_LABELS.get(m.mtype, m.mtype), _r(v, fmt_decimals), unit,
                        m.summary_text(fmt), _r(r.get("distance"), fmt_decimals),
                        _r(r.get("angle"), fmt_decimals), _r(r.get("coaxiality"), fmt_decimals),
                        _r(r.get("dx"), fmt_decimals), _r(r.get("dy"), fmt_decimals),
                        _r(r.get("dz"), fmt_decimals), r.get("frame_name", ""),
                        "はい" if m.critical else "", "済" if m.verified else "", m.verified_note])
    return len(ms)


def export_distance_matrix_csv(doc, path, ids, frame: Frame, decimals: int = 3):
    """取付点などの点間距離マトリクス."""
    objs = [doc.get(i) for i in ids if doc.get(i) is not None]
    pts, names = [], []
    for o in objs:
        kp = o.key_points()
        if kp:
            pts.append(kp[0][1])
            names.append(o.name)
    P = np.array(pts)
    D = np.linalg.norm(P[:, None] - P[None], axis=2) if len(P) else np.zeros((0, 0))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# 点間距離マトリクス [mm]"])
        w.writerow([""] + names)
        for n, row in zip(names, D):
            w.writerow([n] + [_r(x, decimals) for x in row])
    return len(names)
