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
from manifold3d import CrossSection, FillRule, JoinType, Manifold, Mesh

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


def _slab(p, info, w0, w1, th_lo, th_hi, r_lo, r_hi, step=0.25):
    """Closed height-field slab between (bridge surface + w0) and (+ w1)
    over the (theta, r) window -> exact offsets of the curved surface."""
    mbi = info["main_bridge"]
    nt = int(math.ceil((th_hi - th_lo) * (r_hi) / step)) + 1
    nr = int(math.ceil((r_hi - r_lo) / step)) + 1
    TH, RR = np.meshgrid(np.linspace(th_lo, th_hi, nt), np.linspace(r_lo, r_hi, nr), indexing="ij")
    YS = np.empty_like(TH)
    for i in range(nt):
        uu = (TH[i, 0] - mbi["th0"]) / (mbi["th1"] - mbi["th0"])
        rc = mbi["rc"](uu)
        for j in range(nr):
            YS[i, j] = G.logo_surface(info, TH[i, j], RR[i, j] - rc, p)[1]
    X, Z = RR * np.sin(TH), RR * np.cos(TH)
    top = np.stack([X, YS + w1, Z], -1).reshape(-1, 3)
    bot = np.stack([X, YS + w0, Z], -1).reshape(-1, 3)
    V = np.vstack([top, bot])
    off = nt * nr
    idx = np.arange(off).reshape(nt, nr)
    F = []
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F += [np.c_[a, b, c], np.c_[a, c, d]]                    # top
    F += [np.c_[a + off, c + off, b + off], np.c_[a + off, d + off, c + off]]   # bottom
    ring = list(idx[:, 0]) + list(idx[-1, 1:]) + list(idx[::-1, -1][1:]) + list(idx[0, ::-1][1:-1])
    ring = np.array(ring)
    r2 = np.roll(ring, -1)
    F += [np.c_[ring, ring + off, r2], np.c_[r2, ring + off, r2 + off]]
    F = np.vstack(F)
    m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
    if m.volume() < 0:
        m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F[:, ::-1].astype(np.uint32)))
    return m


def logo_solid(p, info, w0, w1, offset=0.0, edge=0.3):
    """Letters as a Manifold between heights w0..w1 (mm, relative to the
    bridge surface; negative = sunk into the body).

    Construction (robust, no folded walls): the 2-D glyphs are extruded as
    a prism that is only *bent* around the rotor axis (straight along Y),
    then intersected with a slab bounded by exact offsets of the curved
    bridge surface.  Letter flanks are therefore true ruled surfaces normal
    to the bridge, and the letter tops follow the bridge crown/bow."""
    lg = p["logo"]
    mbi = info["main_bridge"]
    cs, width = logo_cross_section(p, offset)
    th_c = math.radians(lg["center_angle_deg"])
    u0 = (th_c - mbi["th0"]) / (mbi["th1"] - mbi["th0"])
    r_line = mbi["rc"](u0) + lg["radial_offset"]
    pr = Manifold.extrude(cs, 1.0).refine_to_length(edge)

    def bend(P):
        u, v, w = P[:, 0], P[:, 1], P[:, 2]
        r = r_line + v
        th = th_c - u / r_line
        return np.stack([r * np.sin(th), 40.0 + 70.0 * w, r * np.cos(th)], 1)

    prism = pr.warp_batch(bend)
    bx0, by0, bx1, by1 = cs.bounds()
    th_lo = th_c - (max(bx0, bx1) + 2.0) / r_line
    th_hi = th_c - (min(bx0, bx1) - 2.0) / r_line
    slab = _slab(p, info, w0, w1, th_lo, th_hi, r_line + min(by0, by1) - 2.0, r_line + max(by0, by1) + 2.0)
    m = prism ^ slab
    return m, width, r_line
