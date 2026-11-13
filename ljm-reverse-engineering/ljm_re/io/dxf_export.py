"""DXF 出力（断面曲線・スケッチ要素）.

- mode='2d': 断面平面の (U, V) 座標で出力。スケッチ要素（直線/円弧/円/スプライン）は
  DXF のネイティブ要素で出力し、Fusion 等でスケッチとして編集できる。
  スケッチ化されていない曲線は LWPOLYLINE（閉/開）で出力する。
  複数断面は断面ごとにレイヤーを分け、重ならないよう横方向にずらして配置できる。
- mode='3d': ワールド座標の 3D ポリラインで出力（形状確認・3D スケッチ用）。
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.sketch_fit import SketchEntity, arc_angles


def _layer_name(name: str) -> str:
    s = re.sub(r'[<>/\\":;?*|=,]', "_", name)
    return s[:200] or "SECTION"


def _add_entity_2d(msp, e: SketchEntity, off, layer):
    ox, oy = off
    if e.kind == "line":
        msp.add_line((e.start[0] + ox, e.start[1] + oy), (e.end[0] + ox, e.end[1] + oy),
                     dxfattribs={"layer": layer})
    elif e.kind == "circle":
        msp.add_circle((e.center[0] + ox, e.center[1] + oy), e.radius, dxfattribs={"layer": layer})
    elif e.kind == "arc":
        a0, a1 = arc_angles(e.center, e.start, e.mid, e.end)
        if a1 < a0:  # DXF の ARC は反時計回りのみ → 始終点を入れ替える
            a0, a1 = a1, a0
        msp.add_arc((e.center[0] + ox, e.center[1] + oy), e.radius, math.degrees(a0),
                    math.degrees(a1), dxfattribs={"layer": layer})
    elif e.kind == "spline":
        ctrl = [(p[0] + ox, p[1] + oy, 0.0) for p in e.ctrl]
        if e.closed:
            # scipy の周期スプラインは制御点を周期的に拡張した非周期表現として出力
            fit = [(p[0] + ox, p[1] + oy, 0.0) for p in e.fit_points]
            msp.add_spline(fit, degree=e.degree, dxfattribs={"layer": layer}).closed = True
        else:
            knots = list(map(float, e.knots))
            n_ctrl = len(knots) - e.degree - 1
            msp.add_open_spline(ctrl[:n_ctrl], degree=e.degree, knots=knots,
                                dxfattribs={"layer": layer})


def export_sections_dxf(sections, path, mode: str = "2d", layout_gap: float = 20.0,
                        use_sketch: bool = True, include_raw: bool = False) -> dict:
    """sections: SectionObject のリスト."""
    import ezdxf

    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4  # mm
    doc.header["$MEASUREMENT"] = 1
    msp = doc.modelspace()
    counts = {"line": 0, "arc": 0, "circle": 0, "spline": 0, "polyline": 0}
    x_off = 0.0
    for si, s in enumerate(sections):
        layer = _layer_name(s.name)
        if layer not in doc.layers:
            doc.layers.add(layer, color=(si % 6) + 1)
        raw_layer = layer + "_RAW"
        if mode == "3d":
            for pl, closed in zip(s.polylines, s.closed):
                msp.add_polyline3d([tuple(map(float, p)) for p in pl], close=bool(closed),
                                   dxfattribs={"layer": layer})
                counts["polyline"] += 1
            continue
        # 2D: 断面ごとのバウンディングボックスで横に並べる
        all2d = [s.to_2d(pl) for pl in s.polylines]
        if all2d:
            pts = np.vstack(all2d)
            lo, hi = pts.min(0), pts.max(0)
        else:
            lo = hi = np.zeros(2)
        off = (0.0, 0.0) if len(sections) == 1 else (x_off - lo[0], -lo[1])
        x_off += (hi[0] - lo[0]) + layout_gap
        for i, (uv, closed) in enumerate(zip(all2d, s.closed)):
            ents = s.sketch_entities(i) if use_sketch else []
            if ents:
                for e in ents:
                    _add_entity_2d(msp, e, off, layer)
                    counts[e.kind] += 1
            if not ents or include_raw:
                if include_raw and raw_layer not in doc.layers:
                    doc.layers.add(raw_layer, color=8)
                msp.add_lwpolyline([(p[0] + off[0], p[1] + off[1]) for p in uv], close=bool(closed),
                                   dxfattribs={"layer": raw_layer if (ents and include_raw) else layer})
                counts["polyline"] += 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        doc.write(f)
    return counts


def export_points_dxf(points, names, path):
    """基準点・穴中心を 3D POINT + 文字で出力."""
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    doc.layers.add("POINTS", color=2)
    for p, n in zip(points, names):
        msp.add_point(tuple(map(float, p)), dxfattribs={"layer": "POINTS"})
        msp.add_text(n, height=2.0, dxfattribs={"layer": "POINTS"}).set_placement(tuple(map(float, p)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        doc.write(f)
