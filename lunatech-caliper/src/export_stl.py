#!/usr/bin/env python3
"""
export_stl.py - print-ready STL export for the QIDI MAX3 (325^3 mm).

Reads the meshes produced by generate_caliper.py (build/*.npz) and writes
stl/*.stl.  Also evaluates candidate print orientations quantitatively
(support volume, support scar area, logo orientation, height / time) and
writes build/orientation.json, used by reports/Print_Settings.md.

Files
  Lunatech_Caliper_310mm.stl          one-piece body, logo integrated, assembly coords
  Lunatech_Caliper_FDM_Optimized.stl  one-piece body with logo pockets, in the chosen
                                      print orientation, centred on the bed
  Lunatech_Logo.stl                   flat letters to print in a 2nd colour (glue-in)
  Lunatech_Logo_Fitted.stl            the same letters, curved, in assembly coords
  Lunatech_Caliper_Split_Outboard.stl / _Inboard.stl  split at the rotor plane Y=0,
                                      flat split face on the bed, pin holes
  Lunatech_Locating_Pins.stl          4 locating pins for the split version
  Lunatech_Pistons.stl / Lunatech_Pads.stl / Lunatech_Hardware.stl   detail parts
  Rotor_355x32_Quarter_{1..4}.stl     optional printable rotor (each < 320 mm)
"""
import json
import math
import os
import sys

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, CrossSection, FillRule

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G   # noqa: E402

ROOT = G.ROOT
BUILD = os.path.join(ROOT, "build")
STL = os.path.join(ROOT, "stl")
BED = 325.0


def log(*a):
    print("[export]", *a, flush=True)


def load_m(name):
    d = np.load(os.path.join(BUILD, name + ".npz"))
    return Manifold(Mesh(vert_properties=d["V"].astype(np.float32), tri_verts=d["F"].astype(np.uint32)))


def mesh_of(m):
    mm = m.to_mesh()
    return np.asarray(mm.vert_properties)[:, :3].astype(float), np.asarray(mm.tri_verts).astype(np.int64)


def repair_float32(V, F):
    """Check self-intersections at the float32 precision the STL will store.
    If near-coplanar facets touch (numerical, < 1e-5 mm), delete them and
    re-close the tiny hole; returns (V, F, n_repaired)."""
    import pymeshlab as ml
    V32 = np.asarray(V, np.float32).astype(np.float64)
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V32, np.asarray(F, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    n = ms.current_mesh().selected_face_number()
    if n == 0:
        return V, F, 0
    sel = ms.current_mesh().face_selection_array()
    tm = trimesh.Trimesh(np.asarray(V, float), np.asarray(F)[~sel], process=False)
    tm.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(tm)
    ms2 = ml.MeshSet()
    ms2.add_mesh(ml.Mesh(np.asarray(tm.vertices, np.float32).astype(np.float64), np.asarray(tm.faces, np.int32)))
    ms2.compute_selection_by_self_intersections_per_face()
    left = ms2.current_mesh().selected_face_number()
    V2, F2 = np.asarray(tm.vertices), np.asarray(tm.faces, np.int64)
    if left or not tm.is_watertight:
        log(f"  float32 repair incomplete (left {left}, watertight {tm.is_watertight}) - keeping original")
        return V, F, -n
    return V2, F2, n


def write_stl(name, m_or_vf):
    if isinstance(m_or_vf, Manifold):
        V, F = mesh_of(m_or_vf)
    else:
        V, F = m_or_vf
    if len(F) > 50000:
        V, F, nrep = repair_float32(V, F)
        if nrep:
            log(f"  {name}: repaired {nrep} float32-touching facets")
    tm = trimesh.Trimesh(V, F, process=False)
    path = os.path.join(STL, name)
    tm.export(path)
    ext = V.max(0) - V.min(0)
    log(f"{name:42s} {len(F):>8,} tris  {ext[0]:6.1f} x {ext[1]:6.1f} x {ext[2]:6.1f} mm")
    return path


def rot(M, m):
    M = np.asarray(M, float)
    return m.transform(np.hstack([M, np.zeros((3, 1))]))


def on_bed(m, centre=True):
    V, _ = mesh_of(m)
    lo, hi = V.min(0), V.max(0)
    t = [-(lo[0] + hi[0]) / 2 + (BED / 2 if centre else 0), -(lo[1] + hi[1]) / 2 + (BED / 2 if centre else 0), -lo[2]]
    return m.translate(t)


# --------------------------------------------------------------------------
# orientation study
# --------------------------------------------------------------------------
def Rx(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


ORIENTATIONS = {
    # +Y (logo) -> +Z : outboard face up, inboard face towards the bed
    "O1_logo_up": Rx(90),
    # logo face down on the bed
    "O2_logo_down": Rx(-90),
    # as mounted: arch upright, standing on the end feet
    "O3_arch_upright": np.eye(3),
    # logo-up tilted 45 deg about the length axis
    "O4_logo_up_tilt45": Rx(45),
    # upside-down arch: roof towards the bed
    "O5_roof_down": Rx(180),
}


def support_metrics(m, logo_dir, overhang_deg=45.0, layer=0.14):
    V, F = mesh_of(m)
    V = V - [0, 0, V[:, 2].min()]
    tri = V[F]
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    n = n / (a2[:, None] + 1e-12)
    A = 0.5 * a2
    lim = -math.sin(math.radians(overhang_deg))
    over = (n[:, 2] < lim) & (c[:, 2] > 0.4)
    up = n[:, 2] > 0.05
    cell = 1.0
    ij = np.floor(c[:, :2] / cell).astype(np.int64)
    ij -= ij.min(0)
    cid = ij[:, 0] * 100000 + ij[:, 1]
    keys = np.sort(cid[up] * 1.0e4 + c[up, 2])
    q = cid[over] * 1.0e4 + c[over, 2] - 0.3
    idx = np.searchsorted(keys, q) - 1
    zb = np.zeros(len(q))
    ok = idx >= 0
    same = np.zeros(len(q), bool)
    same[ok] = np.floor(keys[idx[ok]] / 1.0e4) == cid[over][ok]
    zb[same] = keys[idx[same]] - cid[over][same] * 1.0e4
    proj = A[over] * np.abs(n[over, 2])
    h = c[over, 2] - zb
    sup_vol = float(np.sum(proj * h)) / 1000.0              # cm^3 (solid equivalent)
    scar = float(np.sum(proj[same])) / 100.0                # cm^2 of supports landing on the part
    bed_contact = float(np.sum(A[(n[:, 2] < -0.95) & (c[:, 2] < 0.4)])) / 100.0
    # logo orientation: outward direction of the letters (+Y in assembly coords)
    ln = np.asarray(logo_dir, float)
    ext = V.max(0) - V.min(0)
    vol = m.volume() / 1000.0
    area = m.surface_area() / 100.0
    # material: 3 perimeters (1.2 mm) + top/bottom skins + 15 % gyroid infill
    shell = min(vol, area * 0.12)
    material = shell + 0.15 * (vol - shell) + 0.15 * sup_vol
    layers = ext[2] / layer
    # rough time model: ~11 cm3/h extruded at 0.14 mm layers incl. travel,
    # + 3 s per layer change (QIDI MAX3, PLA, quality profile)
    t_h = material / 11.0 + layers * 3.0 / 3600.0
    return dict(extent=[round(float(e), 1) for e in ext], overhang_area_cm2=round(float(proj.sum()) / 100, 1),
                support_volume_cm3=round(sup_vol, 1), support_scar_area_cm2=round(scar, 1),
                bed_contact_cm2=round(bed_contact, 1), logo_normal_z=round(float(ln[2]), 3),
                height_mm=round(float(ext[2]), 1), est_time_h=round(t_h, 1), part_volume_cm3=round(vol, 1),
                est_material_cm3=round(material, 1), est_filament_g_PLA=round(material * 1.24))


def choose(results):
    best, bs = None, 1e18
    for k, r in results.items():
        fits = max(r["extent"][:2]) <= BED - 10 and r["extent"][2] <= BED - 5
        nz = r["logo_normal_z"]
        # raised letters print best facing up, acceptably on a vertical wall,
        # worst facing down (support scars on the letters)
        logo_pen = 0.0 if nz > 0.8 else (15.0 if nz > -0.3 else 120.0)
        stab_pen = 0.0 if r["bed_contact_cm2"] >= 4.0 else 30.0
        score = r["support_volume_cm3"] * 0.6 + r["support_scar_area_cm2"] * 0.6 + logo_pen + stab_pen + r["est_time_h"] * 2.0
        if not fits:
            score += 1e6
        r["score"] = round(score, 1)
        r["fits_bed_with_brim"] = bool(fits)
        if score < bs:
            best, bs = k, score
    return best


# --------------------------------------------------------------------------
def pin_sites(body_logo_up, n_pins, r_need):
    """locating-pin sites on the split plane (z = 0 after logo-up rotation)"""
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import polygon as rpoly
    cs = body_logo_up.slice(0.0)
    polys = cs.to_polygons()
    allp = np.vstack(polys)
    lo, hi = allp.min(0) - 2, allp.max(0) + 2
    px = 0.2
    shp = np.ceil((hi - lo) / px).astype(int) + 1
    img = np.zeros(shp, np.uint8)
    for pl in polys:
        rr, cc = rpoly((pl[:, 0] - lo[0]) / px, (pl[:, 1] - lo[1]) / px, shp)
        img[rr, cc] ^= 1
    dist = distance_transform_edt(img) * px
    cand = np.argwhere(dist >= r_need)
    if len(cand) == 0:
        return [], float(dist.max())
    pts = cand * px + lo
    dv = dist[cand[:, 0], cand[:, 1]]
    chosen = [pts[np.argmax(pts[:, 0])], pts[np.argmin(pts[:, 0])]]
    while len(chosen) < n_pins:
        d = np.min(np.linalg.norm(pts[:, None, :] - np.array(chosen)[None], axis=2), axis=1) + 0.5 * dv
        chosen.append(pts[np.argmax(d)])
    return [tuple(map(float, c)) for c in chosen], float(dist.max())


def layout(meshes, gap=8.0):
    """place a list of manifolds side by side on the bed (rows)"""
    out = []
    x = y = rowh = 0.0
    for m in meshes:
        V, _ = mesh_of(m)
        lo, hi = V.min(0), V.max(0)
        w, d = hi[0] - lo[0], hi[1] - lo[1]
        if x + w > BED - 20:
            x = 0.0
            y += rowh + gap
            rowh = 0.0
        out.append(m.translate([x - lo[0] + 10, y - lo[1] + 10, -lo[2]]))
        x += w + gap
        rowh = max(rowh, d)
    res = out[0]
    for m in out[1:]:
        res = res + m
    return res


def cq_to_manifold(shape, tol=0.03):
    vs, fs = shape.tessellate(tol, 0.2)
    V = np.array([[v.x, v.y, v.z] for v in vs])
    tm = trimesh.Trimesh(V, np.array(fs), process=True)
    tm.merge_vertices()
    m = Manifold(Mesh(vert_properties=tm.vertices.astype(np.float32), tri_verts=tm.faces.astype(np.uint32)))
    if m.status().name != "NoError" or m.is_empty():
        raise RuntimeError(f"tessellation not manifold: {m.status()}")
    return m


def main():
    os.makedirs(STL, exist_ok=True)
    p = G.load_params()
    meta = json.load(open(os.path.join(BUILD, "meta.json")))
    fdm = p["fdm"]
    body_logo = load_m("body_logo")
    body_pocket = load_m("body_pocket")
    insert = load_m("logo_insert")

    # 1. display / assembly-coordinate body
    write_stl("Lunatech_Caliper_310mm.stl", body_logo)

    # 2. orientation study on the one-piece body
    res = {}
    for k, M in ORIENTATIONS.items():
        res[k] = support_metrics(rot(M, body_pocket), M @ [0, 1, 0])
    best = choose(res)
    log("orientation study:")
    for k, r in res.items():
        log(f"  {k:20s} support {r['support_volume_cm3']:6.1f} cm3  scar {r['support_scar_area_cm2']:6.1f} cm2  "
            f"bed {r['bed_contact_cm2']:5.1f} cm2  logo n_z {r['logo_normal_z']:+.2f}  h {r['height_mm']:5.1f}  ~{r['est_time_h']:4.1f} h  score {r['score']}"
            + ("  <== chosen" if k == best else ""))
    # orient with a plain array transform (bit-stable topology; manifold's own
    # transform re-meshes and could create two touching facets)
    _d = np.load(os.path.join(BUILD, "body_pocket.npz"))   # raw arrays, no re-meshing
    Vb, Fb = _d["V"], _d["F"]
    Vo = Vb @ np.asarray(ORIENTATIONS[best]).T
    lo, hi = Vo.min(0), Vo.max(0)
    base = Vo + [BED / 2 - (lo[0] + hi[0]) / 2, BED / 2 - (lo[1] + hi[1]) / 2, -lo[2]]
    write_stl("Lunatech_Caliper_FDM_Optimized.stl", (base, Fb))

    # 3. logo inserts
    import logo as LG
    cs, width = LG.logo_cross_section(p)
    th = p["logo"]["insert_root_depth"] - 0.05 + p["logo"]["protrusion"]
    flat = Manifold.extrude(cs, th)
    write_stl("Lunatech_Logo.stl", on_bed(flat))
    write_stl("Lunatech_Logo_Fitted.stl", insert)

    # 4. split version (rotor mid-plane) with locating pins
    M1 = ORIENTATIONS["O1_logo_up"]
    up = rot(M1, body_pocket)
    r_pin = fdm["pin_diameter"] / 2 + fdm["pin_hole_clearance"]
    sites, maxin = pin_sites(up, fdm["pin_count"], r_pin + 1.6)
    hole_depth = fdm["pin_length"] / 2 + 0.5
    holes = None
    for (x, y) in sites:
        c = Manifold.cylinder(2 * hole_depth, r_pin, r_pin, 40).translate([x, y, -hole_depth])
        holes = c if holes is None else holes + c
    cut = up - holes if holes is not None else up
    top, bot = cut.split_by_plane((0, 0, 1), 0.0)
    ob = on_bed(top)
    ib = on_bed(rot(Rx(180), bot))
    write_stl("Lunatech_Caliper_Split_Outboard.stl", ob)
    write_stl("Lunatech_Caliper_Split_Inboard.stl", ib)
    pin = Manifold.cylinder(fdm["pin_length"], fdm["pin_diameter"] / 2, fdm["pin_diameter"] / 2, 40)
    cham = 0.4
    pin = pin ^ (Manifold.cylinder(fdm["pin_length"], fdm["pin_diameter"] / 2 - cham, fdm["pin_diameter"] / 2 + 6, 40)
                 ^ Manifold.cylinder(fdm["pin_length"], fdm["pin_diameter"] / 2 + 6, fdm["pin_diameter"] / 2 - cham, 40))
    write_stl("Lunatech_Locating_Pins.stl", layout([pin] * len(sites)))
    split_info = dict(pin_sites_xy_in_logo_up_frame=sites, max_inradius_on_split_plane=round(maxin, 2),
                      pin_hole_diameter=round(2 * r_pin, 2), hole_depth_each_side=hole_depth,
                      outboard=support_metrics(ob, [0, 0, 1]), inboard=support_metrics(ib, [0, 0, -1]))

    # 5. detail parts (exact B-Rep -> STL)
    import parts as PT
    sc, info = G.build_scene(p)
    pis = [cq_to_manifold(s) for _, s in PT.pistons(p)]
    pads = [cq_to_manifold(s) for _, s in PT.pads(p, info["pin_th"])]
    hw = [cq_to_manifold(s) for _, s in PT.pad_pins(p, info) + PT.mount_bolts(p) + PT.bleeders(p)]
    # pistons: closed end down; pads: backing plate down
    write_stl("Lunatech_Pistons.stl", layout([rot(Rx(90), m) for m in pis]))
    write_stl("Lunatech_Pads.stl", layout([rot(Rx(90 if i == 0 else -90), m) for i, m in enumerate(pads)]))
    write_stl("Lunatech_Hardware.stl", layout(hw))

    # 6. optional printable rotor quarters
    import cadquery as cq
    rotor = PT.rotor(p)
    R = p["rotor"]["outer_diameter"] / 2 + 5
    for k, (sx, sz) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        box = cq.Solid.makeBox(R, 200, R, cq.Vector(0 if sx > 0 else -R, -100, 0 if sz > 0 else -R))
        q = rotor.intersect(box)
        write_stl(f"Rotor_355x32_Quarter_{k + 1}.stl", on_bed(rot(Rx(90), cq_to_manifold(q)), centre=False))

    with open(os.path.join(BUILD, "orientation.json"), "w") as f:
        json.dump(dict(candidates=res, chosen=best, split=split_info), f, indent=1)


if __name__ == "__main__":
    main()
