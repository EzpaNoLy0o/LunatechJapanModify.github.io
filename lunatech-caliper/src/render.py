"""
render.py - headless software renderer (numba z-buffer rasterizer).

No OpenGL is needed: triangles are rasterized on the CPU with per-pixel
interpolated normals, Blinn-Phong lighting (key/fill/rim), a depth-based
ambient-occlusion term (makes the through-openings read correctly) and
silhouette lines.  Used for the comparison views in renders/.
"""
import math
import numpy as np
from numba import njit, prange
from PIL import Image, ImageDraw, ImageFont


@njit(cache=True)
def _raster(sv, sn, F, mat, W, H, depth, nbuf, mbuf):
    for f in range(F.shape[0]):
        i0, i1, i2 = F[f, 0], F[f, 1], F[f, 2]
        x0, y0, z0 = sv[i0, 0], sv[i0, 1], sv[i0, 2]
        x1, y1, z1 = sv[i1, 0], sv[i1, 1], sv[i1, 2]
        x2, y2, z2 = sv[i2, 0], sv[i2, 1], sv[i2, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xmin = max(int(math.floor(min(x0, min(x1, x2)))), 0)
        xmax = min(int(math.ceil(max(x0, max(x1, x2)))), W - 1)
        ymin = max(int(math.floor(min(y0, min(y1, y2)))), 0)
        ymax = min(int(math.ceil(max(y0, max(y1, y2)))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        ia = 1.0 / area
        for py in range(ymin, ymax + 1):
            fy = py + 0.5
            for px in range(xmin, xmax + 1):
                fx = px + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * ia
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * ia
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z < depth[py, px]:
                    depth[py, px] = z
                    for c in range(3):
                        nbuf[py, px, c] = w0 * sn[i0, c] + w1 * sn[i1, c] + w2 * sn[i2, c]
                    mbuf[py, px] = mat[f]


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn


def look_at(eye, target, up):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


MATERIALS = {
    # name: (base rgb, specular strength, shininess)
    "titanium": ((0.60, 0.585, 0.56), 0.55, 38.0),
    "logo": ((0.86, 0.80, 0.62), 0.75, 60.0),
    "piston": ((0.80, 0.80, 0.82), 0.9, 90.0),
    "pad": ((0.20, 0.20, 0.21), 0.15, 10.0),
    "rotor": ((0.36, 0.36, 0.38), 0.35, 25.0),
    "wheel": ((0.72, 0.74, 0.78), 0.4, 30.0),
    "bolt": ((0.25, 0.25, 0.27), 0.7, 60.0),
}


def render(objects, eye, target, up=(0, 0, 1), size=(1600, 1100), ortho=None, fov=30.0,
           ss=2, bg=(0.965, 0.965, 0.97), title=None, subtitle=None, out=None, ao=True):
    """objects: list of (V, F, material_name)."""
    W, H = size[0] * ss, size[1] * ss
    r, u, f = look_at(eye, target, np.asarray(up, float))
    eye = np.asarray(eye, float)
    Vs, Ns, Fs, Ms = [], [], [], []
    mats = list(MATERIALS.keys())
    off = 0
    for V, F, m in objects:
        V = np.asarray(V, float)
        F = np.asarray(F, np.int64)
        Vs.append(V)
        Ns.append(vertex_normals(V, F))
        Fs.append(F + off)
        Ms.append(np.full(len(F), mats.index(m), np.int64))
        off += len(V)
    V = np.vstack(Vs)
    N = np.vstack(Ns)
    F = np.vstack(Fs)
    M = np.concatenate(Ms)
    d = V - eye
    cx, cy, cz = d @ r, d @ u, d @ f
    if ortho:
        s = min(W, H) / ortho
        sx = W / 2 + cx * s
        sy = H / 2 - cy * s
    else:
        s = (min(W, H) / 2) / math.tan(math.radians(fov / 2))
        sx = W / 2 + cx / cz * s
        sy = H / 2 - cy / cz * s
    sv = np.stack([sx, sy, cz], 1)
    sn = np.stack([N @ r, N @ u, N @ f], 1)
    depth = np.full((H, W), np.inf)
    nbuf = np.zeros((H, W, 3))
    mbuf = np.full((H, W), -1, np.int64)
    _raster(sv, sn, F, M, W, H, depth, nbuf, mbuf)

    hit = mbuf >= 0
    n = nbuf / (np.linalg.norm(nbuf, axis=2, keepdims=True) + 1e-12)
    # camera space: x right, y up, z forward -> view vector = -z
    n[..., 1] *= -1  # flip to image-up convention for lighting below
    n_cam = np.stack([n[..., 0], -n[..., 1], n[..., 2]], -1)  # back to (r,u,f)
    # make normals face the camera (handles any inverted part)
    flip = n_cam[..., 2] > 0
    n_cam[flip] *= -1
    lights = [((-0.45, 0.65, -0.62), 0.85), ((0.7, 0.25, -0.65), 0.35), ((0.1, -0.6, 0.8), 0.25)]
    base = np.zeros((H, W, 3))
    spec_k = np.zeros((H, W))
    shin = np.ones((H, W))
    for i, m in enumerate(mats):
        sel = mbuf == i
        if sel.any():
            base[sel] = MATERIALS[m][0]
            spec_k[sel] = MATERIALS[m][1]
            shin[sel] = MATERIALS[m][2]
    col = np.zeros((H, W, 3))
    vdir = np.array([0, 0, -1.0])
    amb = 0.22 + 0.12 * (n_cam[..., 1] * 0.5 + 0.5)
    for L, k in lights:
        L = np.asarray(L, float)
        L /= np.linalg.norm(L)
        ndl = np.clip((n_cam * L).sum(-1), 0, 1)
        hv = L + vdir
        hv /= np.linalg.norm(hv)
        ndh = np.clip((n_cam * hv).sum(-1), 0, 1)
        col += base * (k * ndl)[..., None]
        col += (k * spec_k * ndh ** shin)[..., None] * np.array([1.0, 0.98, 0.95])
    col += base * amb[..., None]
    # fresnel-ish rim for metal
    rim = (1 - np.abs(n_cam[..., 2])) ** 3
    col += 0.18 * rim[..., None]
    if ao:
        from scipy.ndimage import gaussian_filter, sobel
        dd = np.where(hit, depth, np.nan)
        fill = np.nanmax(dd) if np.isfinite(np.nanmax(dd)) else 0
        dz = np.where(hit, depth, fill + 50)
        aot = np.zeros((H, W))
        for sig, wgt in ((3 * ss, 0.5), (10 * ss, 0.35), (28 * ss, 0.25)):
            bl = gaussian_filter(dz, sig)
            aot += wgt * np.clip((dz - bl) / (sig * 0.9), 0, 1.5)
        occ = np.clip(1 - 0.55 * aot, 0.35, 1)
        col *= occ[..., None]
        # silhouette / crease lines
        gx = sobel(dz, 0)
        gy = sobel(dz, 1)
        g = np.hypot(gx, gy)
        edge = g > (2.5 * (ortho / min(W, H) if ortho else 0.8)) * 6
        col[edge & hit] *= 0.45
    img = np.empty((H, W, 3))
    img[:] = bg
    img[hit] = np.clip(col[hit], 0, 1)
    # soft ground shadow-less floor gradient
    im = Image.fromarray((img * 255).astype(np.uint8))
    if ss > 1:
        im = im.resize(size, Image.LANCZOS)
    if title or subtitle:
        dr = ImageDraw.Draw(im)
        try:
            fnt = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", 30)
            fns = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Regular.otf", 20)
        except Exception:
            fnt = fns = ImageFont.load_default()
        if title:
            dr.text((28, 22), title, fill=(30, 30, 34), font=fnt)
        if subtitle:
            dr.text((28, 62), subtitle, fill=(90, 90, 96), font=fns)
    if out:
        im.save(out)
    return im


VIEWS = {
    # name: (eye direction from target, up)
    "Front": ((0, 1, 0), (0, 0, 1)),            # looking at the outboard (logo) face
    "Rear": ((0, -1, 0), (0, 0, 1)),            # inboard / mounting face
    "Top": ((0, 0, 1), (0, -1, 0)),             # radial top view
    "Bottom": ((0, 0, -1), (0, 1, 0)),
    "Left": ((-1, 0, 0), (0, 0, 1)),
    "Right": ((1, 0, 0), (0, 0, 1)),
    "Isometric": ((-0.55, 0.75, 0.62), (0, 0, 1)),
    "Iso_Rear": ((0.6, -0.72, 0.55), (0, 0, 1)),
    "Iso_Top": ((0.25, 0.55, 1.0), (0, 0, 1)),
}


def view(objects, name, target, dist=900.0, ortho=None, **kw):
    d, up = VIEWS[name]
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)
    eye = np.asarray(target, float) + d * dist
    return render(objects, eye, target, up=up, ortho=ortho, **kw)
