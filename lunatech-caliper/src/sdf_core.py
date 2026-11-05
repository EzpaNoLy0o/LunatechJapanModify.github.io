"""
sdf_core.py - Implicit-surface (signed distance field) kernel for the
Lunatech caliper replica.

The organic ("generative design" looking) body is described as a list of
primitives that are combined with *polynomial smooth union / smooth
subtraction*.  Smooth blending gives the continuous, fillet-like transitions
(visually G1/G2) between ribs, bridges and piston housings that a classic
B-Rep fillet pipeline cannot produce robustly for this many branches.

Primitive table layout (one float64 row per primitive):
    col 0  type
    col 1  op        0 = smooth union, 1 = smooth subtraction
    col 2  k         blend radius (mm)  (0 = sharp)
    col 3-8 AABB     xmin ymin zmin xmax ymax zmax (already inflated)
    col 9+ params    (type dependent, see below)

Types
    0 ROUNDCONE  a(3) b(3) ra rb               variable-radius rib segment
    1 CYLINDER   c(3) u(3) half_h R round      rounded capped cylinder
    2 ELLIPSOID  c(3) radii(3) R(9 row-major)  oriented ellipsoid (node)
    3 BAND       th0 th1 n table_off sigma     variable-section swept beam
                                               along the rotor arc
    4 POLARBOX   half_y R_out th_c half_th R_in y_c   slot / pocket
    5 TRI        a(3) b(3) c(3) half_thickness  thin membrane web
    6 PCAP       sa ya sb yb ra rb r_mid r_half Rref   radially extruded
                                               teardrop (windows in skins)
    7 PPOLY      nv Rref r_mid r_half rho  s0 y0 .. s5 y5   rounded polygon
                                               window extruded radially
"""
import math
import numpy as np
from numba import njit, prange

T_ROUNDCONE, T_CYL, T_ELL, T_BAND, T_PBOX, T_TRI, T_PCAP, T_PPOLY = 0, 1, 2, 3, 4, 5, 6, 7
NCOL = 32
BAND_COLS = 6  # rc, yc, ht, hw, rr, crown


@njit(cache=True, fastmath=True)
def smin(a, b, k):
    if k <= 0.0:
        return a if a < b else b
    h = k - abs(a - b)
    if h < 0.0:
        h = 0.0
    h = h / k
    m = a if a < b else b
    return m - h * h * k * 0.25


@njit(cache=True, fastmath=True)
def smax(a, b, k):
    return -smin(-a, -b, k)


@njit(cache=True, fastmath=True)
def rc_core(px, py, pz, ax, ay, az, bx, by, bz, r1, r2):
    bax, bay, baz = bx - ax, by - ay, bz - az
    l2 = bax * bax + bay * bay + baz * baz
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pax, pay, paz = px - ax, py - ay, pz - az
    y = pax * bax + pay * bay + paz * baz
    z = y - l2
    qx, qy, qz = pax * l2 - bax * y, pay * l2 - bay * y, paz * l2 - baz * y
    x2 = qx * qx + qy * qy + qz * qz
    y2 = y * y * l2
    z2 = z * z * l2
    sgn = 1.0 if rr >= 0.0 else -1.0
    k = sgn * rr * rr * x2
    sz = 1.0 if z >= 0.0 else -1.0
    sy = 1.0 if y >= 0.0 else -1.0
    if sz * a2 * z2 > k:
        return math.sqrt(x2 + z2) * il2 - r2
    if sy * a2 * y2 < k:
        return math.sqrt(x2 + y2) * il2 - r1
    return (math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1


@njit(cache=True, fastmath=True)
def sd_roundcone(px, py, pz, p):
    return rc_core(px, py, pz, p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16])


@njit(cache=True, fastmath=True)
def sd_pcap(px, py, pz, p):
    """teardrop drawn in (arc-length, y) and extruded radially r in [rm-hr, rm+hr]"""
    r = math.sqrt(px * px + pz * pz)
    s = math.atan2(px, pz) * p[17]
    d2 = rc_core(s, py, 0.0, p[9], p[10], 0.0, p[11], p[12], 0.0, p[13], p[14])
    dr = abs(r - p[15]) - p[16]
    return d2 if d2 > dr else dr


@njit(cache=True, fastmath=True)
def sd_cyl(px, py, pz, p):
    cx, cy, cz = p[9], p[10], p[11]
    ux, uy, uz = p[12], p[13], p[14]
    hh, R, rr = p[15], p[16], p[17]
    dx, dy, dz = px - cx, py - cy, pz - cz
    t = dx * ux + dy * uy + dz * uz
    rx, ry, rz = dx - ux * t, dy - uy * t, dz - uz * t
    rad = math.sqrt(rx * rx + ry * ry + rz * rz)
    qa = rad - R + rr
    qb = abs(t) - hh + rr
    inside = qa if qa > qb else qb
    if inside > 0.0:
        inside = 0.0
    ma = qa if qa > 0.0 else 0.0
    mb = qb if qb > 0.0 else 0.0
    return inside + math.sqrt(ma * ma + mb * mb) - rr


@njit(cache=True, fastmath=True)
def sd_ell(px, py, pz, p):
    dx, dy, dz = px - p[9], py - p[10], pz - p[11]
    # local = R^T (p-c) with R rows = local axes
    lx = p[15] * dx + p[16] * dy + p[17] * dz
    ly = p[18] * dx + p[19] * dy + p[20] * dz
    lz = p[21] * dx + p[22] * dy + p[23] * dz
    ax, ay, az = p[12], p[13], p[14]
    k0 = math.sqrt((lx / ax) ** 2 + (ly / ay) ** 2 + (lz / az) ** 2)
    k1 = math.sqrt((lx / (ax * ax)) ** 2 + (ly / (ay * ay)) ** 2 + (lz / (az * az)) ** 2)
    if k1 < 1e-9:
        return -min(ax, min(ay, az))
    return k0 * (k0 - 1.0) / k1


@njit(cache=True, fastmath=True)
def band_frame(px, py, pz, p, tab):
    """Return interpolated (rc, yc, ht, hw, rr, crown, theta, r)."""
    th0, th1, n, off = p[9], p[10], int(p[11]), int(p[12])
    r = math.sqrt(px * px + pz * pz)
    th = math.atan2(px, pz)
    f = (th - th0) / (th1 - th0) * (n - 1)
    if f < 0.0:
        f = 0.0
    if f > n - 1.0:
        f = n - 1.0
    i0 = int(f)
    if i0 >= n - 1:
        i0 = n - 2
    w = f - i0
    a = off + i0 * BAND_COLS
    b = a + BAND_COLS
    rc = tab[a] * (1 - w) + tab[b] * w
    yc = tab[a + 1] * (1 - w) + tab[b + 1] * w
    ht = tab[a + 2] * (1 - w) + tab[b + 2] * w
    hw = tab[a + 3] * (1 - w) + tab[b + 3] * w
    rr = tab[a + 4] * (1 - w) + tab[b + 4] * w
    cr = tab[a + 5] * (1 - w) + tab[b + 5] * w
    return rc, yc, ht, hw, rr, cr, th, r


@njit(cache=True, fastmath=True)
def sd_band(px, py, pz, p, tab):
    th0, th1, sigma = p[9], p[10], p[13]
    rc, yc, ht, hw, rr, cr, th, r = band_frame(px, py, pz, p, tab)
    dr = r - rc
    u = dr / hw
    c = 1.0 - u * u
    if c < 0.0:
        c = 0.0
    hte = ht + cr * c
    ycen = yc + sigma * (hte - ht) * 0.5
    half = (ht + hte) * 0.5
    dl = th0 - th
    if th - th1 > dl:
        dl = th - th1
    dl = dl * r
    qa = abs(py - ycen) - half + rr
    qb = abs(dr) - hw + rr
    qc = dl + rr
    m = qa
    if qb > m:
        m = qb
    if qc > m:
        m = qc
    if m > 0.0:
        m = 0.0
    qa = qa if qa > 0.0 else 0.0
    qb = qb if qb > 0.0 else 0.0
    qc = qc if qc > 0.0 else 0.0
    return m + math.sqrt(qa * qa + qb * qb + qc * qc) - rr


@njit(cache=True, fastmath=True)
def sd_pbox(px, py, pz, p):
    hy, Ro, thc, hth, Ri, yc = p[9], p[10], p[11], p[12], p[13], p[14]
    r = math.sqrt(px * px + pz * pz)
    th = math.atan2(px, pz)
    d = abs(py - yc) - hy
    d2 = r - Ro
    if d2 > d:
        d = d2
    d3 = Ri - r
    if d3 > d:
        d = d3
    d4 = (abs(th - thc) - hth) * r
    if d4 > d:
        d = d4
    return d


@njit(cache=True, fastmath=True)
def _dot2(x, y, z):
    return x * x + y * y + z * z


@njit(cache=True, fastmath=True)
def sd_tri(px, py, pz, p):
    ax, ay, az, bx, by, bz, cx, cy, cz = p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17]
    bax, bay, baz = bx - ax, by - ay, bz - az
    pax, pay, paz = px - ax, py - ay, pz - az
    cbx, cby, cbz = cx - bx, cy - by, cz - bz
    pbx, pby, pbz = px - bx, py - by, pz - bz
    acx, acy, acz = ax - cx, ay - cy, az - cz
    pcx, pcy, pcz = px - cx, py - cy, pz - cz
    nx = bay * acz - baz * acy
    ny = baz * acx - bax * acz
    nz = bax * acy - bay * acx
    # cross(ba, nor) . pa  etc.
    s1 = (bay * nz - baz * ny) * pax + (baz * nx - bax * nz) * pay + (bax * ny - bay * nx) * paz
    s2 = (cby * nz - cbz * ny) * pbx + (cbz * nx - cbx * nz) * pby + (cbx * ny - cby * nx) * pbz
    s3 = (acy * nz - acz * ny) * pcx + (acz * nx - acx * nz) * pcy + (acx * ny - acy * nx) * pcz
    sg = (1.0 if s1 > 0 else -1.0) + (1.0 if s2 > 0 else -1.0) + (1.0 if s3 > 0 else -1.0)
    if sg < 2.0:
        t = min(max((bax * pax + bay * pay + baz * paz) / _dot2(bax, bay, baz), 0.0), 1.0)
        d1 = _dot2(bax * t - pax, bay * t - pay, baz * t - paz)
        t = min(max((cbx * pbx + cby * pby + cbz * pbz) / _dot2(cbx, cby, cbz), 0.0), 1.0)
        d2 = _dot2(cbx * t - pbx, cby * t - pby, cbz * t - pbz)
        t = min(max((acx * pcx + acy * pcy + acz * pcz) / _dot2(acx, acy, acz), 0.0), 1.0)
        d3 = _dot2(acx * t - pcx, acy * t - pcy, acz * t - pcz)
        d = min(d1, min(d2, d3))
    else:
        q = nx * pax + ny * pay + nz * paz
        d = q * q / _dot2(nx, ny, nz)
    return math.sqrt(d) - p[18]


@njit(cache=True, fastmath=True)
def sd_ppoly(px, py, pz, p):
    nv = int(p[9])
    Rref, rm, rh, rho = p[10], p[11], p[12], p[13]
    r = math.sqrt(px * px + pz * pz)
    sx = math.atan2(px, pz) * Rref
    sy = py
    vx0, vy0 = p[14], p[15]
    d = (sx - vx0) ** 2 + (sy - vy0) ** 2
    sgn = 1.0
    j = nv - 1
    for i in range(nv):
        vix, viy = p[14 + 2 * i], p[15 + 2 * i]
        vjx, vjy = p[14 + 2 * j], p[15 + 2 * j]
        ex, ey = vjx - vix, vjy - viy
        wx, wy = sx - vix, sy - viy
        t = (wx * ex + wy * ey) / (ex * ex + ey * ey)
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        bx, by = wx - ex * t, wy - ey * t
        dd = bx * bx + by * by
        if dd < d:
            d = dd
        c1 = sy >= viy
        c2 = sy < vjy
        c3 = ex * wy > ey * wx
        if (c1 and c2 and c3) or ((not c1) and (not c2) and (not c3)):
            sgn = -sgn
        j = i
    d2 = sgn * math.sqrt(d) - rho
    dr = abs(r - rm) - rh
    return d2 if d2 > dr else dr


@njit(cache=True, fastmath=True)
def sd_prim(px, py, pz, p, tab):
    t = int(p[0])
    if t == 0:
        return sd_roundcone(px, py, pz, p)
    elif t == 1:
        return sd_cyl(px, py, pz, p)
    elif t == 2:
        return sd_ell(px, py, pz, p)
    elif t == 3:
        return sd_band(px, py, pz, p, tab)
    elif t == 4:
        return sd_pbox(px, py, pz, p)
    elif t == 5:
        return sd_tri(px, py, pz, p)
    elif t == 6:
        return sd_pcap(px, py, pz, p)
    else:
        return sd_ppoly(px, py, pz, p)


@njit(cache=True, fastmath=True)
def eval_point(px, py, pz, prims, tab, idx, nidx):
    d = 1e3
    for j in range(nidx):
        p = prims[idx[j]]
        if p[1] != 0.0:
            continue
        if px < p[3] or py < p[4] or pz < p[5] or px > p[6] or py > p[7] or pz > p[8]:
            continue
        d = smin(d, sd_prim(px, py, pz, p, tab), p[2])
    for j in range(nidx):
        p = prims[idx[j]]
        if p[1] == 0.0:
            continue
        if px < p[3] or py < p[4] or pz < p[5] or px > p[6] or py > p[7] or pz > p[8]:
            continue
        d = smax(d, -sd_prim(px, py, pz, p, tab), p[2])
    return d


@njit(parallel=True, cache=True)
def eval_grid(origin, h, nx, ny, nz, prims, tab, B):
    out = np.empty((nx, ny, nz), dtype=np.float32)
    bx = (nx + B - 1) // B
    by = (ny + B - 1) // B
    bz = (nz + B - 1) // B
    nb = bx * by * bz
    npr = prims.shape[0]
    for bi in prange(nb):
        ix0 = (bi // (by * bz)) * B
        iy0 = ((bi // bz) % by) * B
        iz0 = (bi % bz) * B
        ix1 = min(ix0 + B, nx)
        iy1 = min(iy0 + B, ny)
        iz1 = min(iz0 + B, nz)
        x0 = origin[0] + ix0 * h
        y0 = origin[1] + iy0 * h
        z0 = origin[2] + iz0 * h
        x1 = origin[0] + (ix1 - 1) * h
        y1 = origin[1] + (iy1 - 1) * h
        z1 = origin[2] + (iz1 - 1) * h
        idx = np.empty(npr, dtype=np.int64)
        n = 0
        nun = 0
        for j in range(npr):
            p = prims[j]
            if x1 < p[3] or y1 < p[4] or z1 < p[5] or x0 > p[6] or y0 > p[7] or z0 > p[8]:
                continue
            idx[n] = j
            n += 1
            if p[1] == 0.0:
                nun += 1
        for ix in range(ix0, ix1):
            px = origin[0] + ix * h
            for iy in range(iy0, iy1):
                py = origin[1] + iy * h
                for iz in range(iz0, iz1):
                    if nun == 0:
                        out[ix, iy, iz] = 20.0
                    else:
                        pz = origin[2] + iz * h
                        out[ix, iy, iz] = eval_point(px, py, pz, prims, tab, idx, n)
    return out


@njit(parallel=True, cache=True)
def eval_points(P, prims, tab):
    n = P.shape[0]
    out = np.empty(n, dtype=np.float64)
    npr = prims.shape[0]
    idx = np.arange(npr)
    for i in prange(n):
        out[i] = eval_point(P[i, 0], P[i, 1], P[i, 2], prims, tab, idx, npr)
    return out


# --------------------------------------------------------------------------
#  Python-side builder
# --------------------------------------------------------------------------
def erode_convex(pts, d):
    """Offset a CCW convex polygon inward by d (returns None if it vanishes)."""
    n = len(pts)
    lines = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        t = b - a
        nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)   # inward for CCW
        lines.append((a + nrm * d, t))
    out = []
    for i in range(n):
        p1, t1 = lines[i - 1]
        p2, t2 = lines[i]
        m = np.array([t1, -t2]).T
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        s_ = np.linalg.solve(m, p2 - p1)
        out.append(p1 + t1 * s_[0])
    out = np.array(out)
    # validity: area must stay positive and same orientation
    x, y = out[:, 0], out[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area <= 1.0:
        return None
    return out


class SDFScene:
    """Collects primitives; computes conservative AABBs automatically."""

    def __init__(self):
        self.rows = []
        self.tab = []
        self.names = []

    def _add(self, row, aabb, op, k, name):
        r = np.zeros(NCOL)
        r[: len(row)] = row
        r[1] = op
        r[2] = k
        pad = k + 1.0
        r[3:6] = np.asarray(aabb[0]) - pad
        r[6:9] = np.asarray(aabb[1]) + pad
        self.rows.append(r)
        self.names.append(name)

    # ---- primitives -----------------------------------------------------
    def roundcone(self, a, b, ra, rb, k=3.0, op=0, name=""):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if np.linalg.norm(b - a) < 1e-6:
            b = a + 1e-3
        rm = max(ra, rb)
        lo = np.minimum(a, b) - rm
        hi = np.maximum(a, b) + rm
        self._add([T_ROUNDCONE, 0, 0, 0, 0, 0, 0, 0, 0, *a, *b, ra, rb], (lo, hi), op, k, name)

    def rib(self, pts, radii, k=3.0, op=0, samples=6, name=""):
        """Curved, tapered rib through control points (Catmull-Rom),
        radius interpolated along the curve.  Internally a chain of
        round-cone segments joined with a tiny blend -> smooth tube."""
        pts = np.asarray(pts, float)
        radii = np.asarray(radii, float)
        cp = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
        out_p, out_r = [], []
        for i in range(len(pts) - 1):
            p0, p1, p2, p3 = cp[i], cp[i + 1], cp[i + 2], cp[i + 3]
            for s in range(samples):
                t = s / samples
                t2, t3 = t * t, t * t * t
                q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                           + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
                # smoothstep radius between nodes -> continuous taper
                ts = t * t * (3 - 2 * t)
                out_p.append(q)
                out_r.append(radii[i] * (1 - ts) + radii[i + 1] * ts)
        out_p.append(pts[-1])
        out_r.append(radii[-1])
        # consecutive round cones share their end spheres -> a hard min gives a
        # tangent-continuous tube (a smooth-min here would add ridges at every
        # joint).  Only the first and last segment blend into the structure.
        nseg = len(out_p) - 1
        for i in range(nseg):
            kk = k if i in (0, nseg - 1) else 0.0
            self.roundcone(out_p[i], out_p[i + 1], out_r[i], out_r[i + 1], k=kk, op=op, name=name)

    def cylinder(self, c, u, half_h, R, rnd=0.0, k=3.0, op=0, name=""):
        c = np.asarray(c, float)
        u = np.asarray(u, float)
        u = u / np.linalg.norm(u)
        e = np.sqrt(np.clip(1 - u * u, 0, 1)) * R + np.abs(u) * half_h
        self._add([T_CYL, 0, 0, 0, 0, 0, 0, 0, 0, *c, *u, half_h, R, rnd], (c - e, c + e), op, k, name)

    def ellipsoid(self, c, radii, axes=None, k=3.0, op=0, name=""):
        c = np.asarray(c, float)
        if axes is None:
            axes = np.eye(3)
        axes = np.asarray(axes, float)
        rm = max(radii)
        self._add([T_ELL, 0, 0, 0, 0, 0, 0, 0, 0, *c, *radii, *axes.reshape(-1)],
                  (c - rm, c + rm), op, k, name)

    def band(self, th0, th1, rc, yc, ht, hw, rr, crown=None, sigma=1.0, n=48, k=4.0,
             op=0, name=""):
        """Variable-section beam swept along the rotor arc.
        rc, yc, ht, hw, rr, crown: callables f(u) with u in [0,1] (u=0 -> th0)."""
        off = len(self.tab)
        rmax, ymin, ymax = 0.0, 1e9, -1e9
        for i in range(n):
            u = i / (n - 1)
            row = [rc(u), yc(u), ht(u), hw(u), rr(u), crown(u) if crown else 0.0]
            self.tab.extend(row)
            rmax = max(rmax, row[0] + row[3])
            ymin = min(ymin, row[1] - row[2] - row[5])
            ymax = max(ymax, row[1] + row[2] + row[5])
        rmin = min(self.tab[off + i * BAND_COLS] - self.tab[off + i * BAND_COLS + 3] for i in range(n))
        ths = np.linspace(th0, th1, 64)
        xs = np.concatenate([rmax * np.sin(ths), rmin * np.sin(ths)])
        zs = np.concatenate([rmax * np.cos(ths), rmin * np.cos(ths)])
        lo = (xs.min(), ymin, zs.min())
        hi = (xs.max(), ymax, rmax)
        self._add([T_BAND, 0, 0, 0, 0, 0, 0, 0, 0, th0, th1, n, off, sigma], (lo, hi), op, k, name)

    def membrane(self, a, b, c, thickness, k=3.0, op=0, name=""):
        a, b, c = (np.asarray(v, float) for v in (a, b, c))
        lo = np.minimum(np.minimum(a, b), c) - thickness
        hi = np.maximum(np.maximum(a, b), c) + thickness
        self._add([T_TRI, 0, 0, 0, 0, 0, 0, 0, 0, *a, *b, *c, thickness / 2], (lo, hi), op, k, name)

    def window(self, tha, ya, thb, yb, ra, rb, r_mid, r_half, Rref=190.0, k=2.5, op=1, name=""):
        """Radially extruded teardrop opening (angles in degrees)."""
        sa, sb = math.radians(tha) * Rref, math.radians(thb) * Rref
        if abs(sa - sb) + abs(ya - yb) < 1e-6:
            sb += 1e-3
        ro = r_mid + r_half
        ths = np.radians([tha, thb])
        dth = (max(ra, rb) + 1.0) / max(r_mid - r_half, 1.0)
        th_lo, th_hi = ths.min() - dth, ths.max() + dth
        tt = np.linspace(th_lo, th_hi, 16)
        xs = np.concatenate([ro * np.sin(tt), (r_mid - r_half) * np.sin(tt)])
        zs = np.concatenate([ro * np.cos(tt), (r_mid - r_half) * np.cos(tt)])
        lo = (xs.min(), min(ya, yb) - max(ra, rb), zs.min())
        hi = (xs.max(), max(ya, yb) + max(ra, rb), ro)
        self._add([T_PCAP, 0, 0, 0, 0, 0, 0, 0, 0, sa, ya, sb, yb, ra, rb, r_mid, r_half, Rref],
                  (lo, hi), op, k, name)

    def poly_window(self, corners_deg_y, inset, rho, r_mid, r_half, Rref=190.0, k=2.5, op=1, name=""):
        """Rounded convex-polygon opening.  corners: [(theta_deg, y), ...] of
        the rib cell (rib centre lines); the window is that cell eroded by
        `inset` with corner radius `rho` -> uniform rim around the opening."""
        pts = np.array([(math.radians(t) * Rref, y) for t, y in corners_deg_y], float)
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(ang)]
        er = erode_convex(pts, inset + rho)
        if er is None:
            return
        row = [T_PPOLY, 0, 0, 0, 0, 0, 0, 0, 0, len(er), Rref, r_mid, r_half, rho]
        for q in er:
            row += [q[0], q[1]]
        ths = er[:, 0] / Rref
        dth = (rho + 1.0) / max(r_mid - r_half, 1.0)
        tt = np.linspace(ths.min() - dth, ths.max() + dth, 16)
        ro, ri = r_mid + r_half, r_mid - r_half
        xs = np.concatenate([ro * np.sin(tt), ri * np.sin(tt)])
        zs = np.concatenate([ro * np.cos(tt), ri * np.cos(tt)])
        lo = (xs.min(), er[:, 1].min() - rho, zs.min())
        hi = (xs.max(), er[:, 1].max() + rho, ro)
        self._add(row, (lo, hi), op, k, name)

    def polarbox(self, half_y, R_out, th_c=0.0, half_th=math.pi, R_in=0.0, y_c=0.0, k=0.0,
                 op=1, name=""):
        lo = (-R_out, y_c - half_y, -R_out)
        hi = (R_out, y_c + half_y, R_out)
        self._add([T_PBOX, 0, 0, 0, 0, 0, 0, 0, 0, half_y, R_out, th_c, half_th, R_in, y_c],
                  (lo, hi), op, k, name)

    # ---- evaluation -----------------------------------------------------
    def arrays(self):
        prims = np.array(self.rows, dtype=np.float64)
        tab = np.array(self.tab if self.tab else [0.0], dtype=np.float64)
        return prims, tab

    def grid(self, lo, hi, h, block=16):
        prims, tab = self.arrays()
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        n = np.ceil((hi - lo) / h).astype(int) + 1
        f = eval_grid(lo, float(h), int(n[0]), int(n[1]), int(n[2]), prims, tab, block)
        return f, lo

    def points(self, P):
        prims, tab = self.arrays()
        return eval_points(np.ascontiguousarray(P, dtype=np.float64), prims, tab)
