"""
logo.py - "Lunatech" raised lettering that follows the curved main bridge.

Glyph outlines come from the bundled Inter Display SemiBold font (OFL).
They are converted to a manifold3d CrossSection (exact 2-D polygons, with
optional offset for insert clearance), extruded, refined to ~0.35 mm edges
and finally *warped* onto the crowned, bowed outer surface of the main
bridge, so the letters are true solids that sit on the curved surface
(not a decal / texture).

Text coordinates: u along the text (reading direction), v = up (radially
outward on the caliper), w = height above the bridge surface.
"""
import math
import os
import numpy as np
from fontTools.ttLib import TTFont
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from manifold3d import CrossSection, FillRule, JoinType, Manifold

import geometry as G


def glyph_polygons(p):
    lg = p["logo"]
    font_path = os.path.join(G.ROOT, lg["font_file"])
    tt = TTFont(font_path)
    upm = tt["head"].unitsPerEm
    cap = tt["OS/2"].sCapHeight
    size = lg["cap_height"] * upm / cap
    cmap = tt.getBestCmap()
    hmtx = tt["hmtx"]
    fp = FontProperties(fname=font_path)
    polys = []
    x = 0.0
    for ch in lg["text"]:
        tp = TextPath((x, 0.0), ch, size=size, prop=fp)
        for poly in tp.to_polygons(closed_only=True):
            if len(poly) >= 3:
                polys.append(np.asarray(poly[:-1] if np.allclose(poly[0], poly[-1]) else poly, float))
        adv = hmtx[cmap[ord(ch)]][0] * size / upm
        x += adv + lg["tracking"]
    allp = np.vstack(polys)
    xmin, xmax = allp[:, 0].min(), allp[:, 0].max()
    width = xmax - xmin
    # centre: u in [-W/2, W/2], v centred on cap height
    for q in polys:
        q[:, 0] -= xmin + width / 2
        q[:, 1] -= lg["cap_height"] / 2
    return polys, width


def logo_cross_section(p, offset=0.0):
    polys, width = glyph_polygons(p)
    cs = CrossSection(polys, FillRule.EvenOdd)
    if abs(offset) > 1e-9:
        cs = cs.offset(offset, JoinType.Round)
    return cs, width


def surface_map(p, info):
    """Return warp function (u, v, w) -> xyz on the main bridge."""
    lg = p["logo"]
    mbi = info["main_bridge"]
    th_c = math.radians(lg["center_angle_deg"])
    u0 = (th_c - mbi["th0"]) / (mbi["th1"] - mbi["th0"])
    r_line = mbi["rc"](u0) + lg["radial_offset"]

    def warp(P):
        u, v, w = P[:, 0], P[:, 1], P[:, 2]
        r = r_line + v
        th = th_c - u / r_line            # reading direction = -X seen from outboard
        uu = (th - mbi["th0"]) / (mbi["th1"] - mbi["th0"])
        out = np.empty_like(P)
        for i in range(len(P)):
            rr_, ys = G.logo_surface(info, th[i], r[i] - mbi["rc"](uu[i]), p)
            out[i, 1] = ys + w[i]
        out[:, 0] = r * np.sin(th)
        out[:, 2] = r * np.cos(th)
        return out

    return warp, r_line


def logo_solid(p, info, w0, w1, offset=0.0, edge=0.35):
    """Letters as a Manifold between heights w0..w1 (mm, relative to the
    bridge surface; negative = sunk into the body)."""
    cs, width = logo_cross_section(p, offset)
    m = Manifold.extrude(cs, w1 - w0).translate((0, 0, w0))
    m = m.refine_to_length(edge)
    warp, r_line = surface_map(p, info)
    m = m.warp_batch(warp)
    return m, width, r_line
