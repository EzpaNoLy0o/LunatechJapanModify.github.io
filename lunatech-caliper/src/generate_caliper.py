#!/usr/bin/env python3
"""
generate_caliper.py - Lunatech Titanium Caliper Replica, master generator.

    python3 generate_caliper.py            # final quality (0.4 mm field)
    python3 generate_caliper.py --preview  # fast 1.0 mm field

Pipeline
  1. parametric layout (geometry.py) -> implicit-surface body (sdf_core.py)
  2. field sampled on a voxel grid, iso-surface extracted (marching cubes)
  3. automatic size gate: if any axis > limits.max_axis_mm the target
     length is reduced and the body regenerated (max 4 attempts)
  4. adaptive (quadric) decimation -> watertight manifold (manifold3d)
  5. exact boolean of the curved "Lunatech" letters (logo.py):
       body_logo   = body  U letters          (single-colour print)
       body_pocket = body  - letters(+clr)    (two-colour, glued insert)
       logo_insert = letters with 0.6 mm locating root
  6. exact B-Rep parts (parts.py) + faceted B-Rep body -> STEP files
Outputs: build/*.npz (meshes for export_stl / validate / render), cad/*.step
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from skimage import measure
import fast_simplification
from manifold3d import Manifold, Mesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G        # noqa: E402
import logo as LG            # noqa: E402

ROOT = G.ROOT
BUILD = os.path.join(ROOT, "build")
CAD = os.path.join(ROOT, "cad")


def log(*a):
    print("[generate]", *a, flush=True)


def field_mesh(p, h):
    sc, info = G.build_scene(p)
    lo, hi = G.domain(p, h)
    t = time.time()
    f, org = sc.grid(lo, hi, h)
    f = np.pad(f, 1, constant_values=20.0)
    org = org - h
    V, F, _, _ = measure.marching_cubes(f, 0.0, spacing=(h, h, h))
    V = V + org
    F = F.astype(np.int64)  # skimage winding is outward for this field sign
    log(f"field {f.shape} ({f.size / 1e6:.0f} M voxels, {len(sc.rows)} primitives), "
        f"iso-surface {len(F):,} tris in {time.time() - t:.1f}s")
    return V, F, sc, info


def to_manifold(V, F):
    m = Manifold(Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                      tri_verts=np.ascontiguousarray(F, np.uint32)))
    return m


def from_manifold(m):
    mm = m.to_mesh()
    return np.asarray(mm.vert_properties)[:, :3].astype(np.float64), np.asarray(mm.tri_verts).astype(np.int64)


def keep_main_components(m, min_frac=0.002):
    parts = m.decompose()
    vols = [q.volume() for q in parts]
    vmax = max(vols)
    keep = [q for q, v in zip(parts, vols) if v >= min_frac * vmax]
    dropped = [round(v, 3) for v in vols if v < min_frac * vmax]
    out = keep[0]
    for q in keep[1:]:
        out = out + q
    return out, len(parts), dropped


def decimate(V, F, target):
    if len(F) <= target:
        return V, F
    red = 1.0 - target / len(F)
    V2, F2 = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=red, agg=7)
    return V2.astype(np.float64), F2.astype(np.int64)


def robust_manifold(V, F, target, label):
    """decimate then convert; fall back to finer meshes if not manifold."""
    for tgt in (target, int(target * 1.3), int(target * 1.7)):
        V2, F2 = decimate(V, F, tgt)
        m = to_manifold(V2, F2)
        if m.status().name == "NoError" and not m.is_empty():
            log(f"{label}: decimated {len(F):,} -> {m.num_tri():,} tris, manifold OK")
            return m
        log(f"{label}: decimation to {tgt} not manifold ({m.status()}), retrying")
    m = to_manifold(V, F)
    if m.status().name != "NoError":
        raise RuntimeError(f"{label}: iso-surface is not manifold: {m.status()}")
    return m


def save_mesh(name, m):
    V, F = from_manifold(m)
    np.savez_compressed(os.path.join(BUILD, name + ".npz"), V=V, F=F)
    return V, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-step", action="store_true")
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(CAD, exist_ok=True)
    p = G.load_params()
    h = p["mesh"]["voxel_preview"] if args.preview else p["mesh"]["voxel_final"]
    lim = p["limits"]["max_axis_mm"]
    history = []

    # ---------------- 1-3: body + automatic size gate -------------------
    for attempt in range(4):
        V, F, sc, info = field_mesh(p, h)
        ext = V.max(0) - V.min(0)
        ext_logo = ext + np.array([0, p["logo"]["protrusion"], 0])
        history.append(dict(attempt=attempt + 1, target_length=p["limits"]["target_length_mm"],
                            extent=[round(float(e), 2) for e in ext_logo]))
        log(f"attempt {attempt + 1}: extent X {ext_logo[0]:.1f}  Y {ext_logo[1]:.1f}  Z {ext_logo[2]:.1f} mm")
        if ext_logo.max() <= lim - 0.5:
            break
        excess = ext_logo.max() - (lim - 0.5)
        p["limits"]["target_length_mm"] -= excess + 2.0
        log(f"  -> exceeds {lim} mm: auto-fix, target length -> {p['limits']['target_length_mm']:.1f}")
    else:
        raise SystemExit("size gate failed")

    # ---------------- 4: manifold body ---------------------------------
    raw = to_manifold(V, F)
    if raw.status().name != "NoError":
        raise SystemExit(f"iso-surface not manifold: {raw.status()}")
    raw, ncomp, dropped = keep_main_components(raw)
    log(f"components in iso-surface: {ncomp}, removed floating crumbs (mm3): {dropped}")
    Vr, Fr = from_manifold(raw)
    body = robust_manifold(Vr, Fr, p["mesh"]["target_triangles"], "body")

    # ---------------- 5: logo variants ---------------------------------
    lg = p["logo"]
    t = time.time()
    letters_full, width, r_line = LG.logo_solid(p, info, -lg["embed_depth"], lg["protrusion"])
    insert, _, _ = LG.logo_solid(p, info, -(lg["insert_root_depth"] - 0.05), lg["protrusion"])
    pocket, _, _ = LG.logo_solid(p, info, -lg["insert_root_depth"], lg["protrusion"] + 1.5,
                                 offset=lg["insert_clearance"])
    body_logo = body + letters_full
    body_pocket = body - pocket
    log(f"logo '{lg['text']}' width {width:.1f} mm, cap {lg['cap_height']} mm, booleans in {time.time() - t:.1f}s")

    Vb, Fb = save_mesh("body_logo", body_logo)
    save_mesh("body_pocket", body_pocket)
    save_mesh("logo_insert", insert)
    save_mesh("body_plain", body)
    meta = dict(voxel=h, history=history, logo_width=width, logo_r_line=r_line,
                pin_th=info["pin_th"], pad_th=info["pad_th"],
                derived={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info["derived"].items()},
                n_iso_components=ncomp, dropped_crumbs=dropped,
                params=p)
    with open(os.path.join(BUILD, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if args.no_step:
        return

    # ---------------- 6: STEP ------------------------------------------
    import cadquery as cq
    import parts as PT
    from step_faceted import merge_step
    t = time.time()
    # STEP body: quadric decimation -> manifold round trip (splits the few
    # non-manifold edges a collapse can create) -> keep the main solid
    Vs, Fs = decimate(Vb, Fb, 60000)
    ms, _, crumbs = keep_main_components(to_manifold(Vs, Fs))
    Vs, Fs = from_manifold(ms)
    import trimesh
    tchk = trimesh.Trimesh(Vs, Fs, process=False)
    assert tchk.is_watertight and tchk.is_winding_consistent and tchk.volume > 0, "STEP body mesh not closed"
    log(f"STEP body: {len(Fs):,} faceted faces, closed, volume {tchk.volume / 1000:.1f} cm3 (crumbs removed {crumbs})")
    named = []
    named += PT.pistons(p)
    named += PT.pads(p, info["pin_th"])
    named += PT.pad_pins(p, info)
    named += PT.mount_bolts(p)
    named += PT.bleeders(p)
    rot = PT.rotor(p)
    whl = PT.wheel(p)

    def brep_step(path, items, colors=None):
        assy = cq.Assembly(name="parts")
        for n, s in items:
            assy.add(s, name=n, color=cq.Color(*(colors or {}).get(n, (0.75, 0.75, 0.78))))
        assy.export(path)

    tmp_parts = os.path.join(BUILD, "_parts.step")
    tmp_rotor = os.path.join(BUILD, "_rotor.step")
    tmp_wheel = os.path.join(BUILD, "_wheel.step")
    brep_step(tmp_parts, named)
    brep_step(tmp_rotor, [("Rotor_355x32", rot)])
    brep_step(tmp_wheel, [("Wheel_18in_Reference", whl)])
    body_name = "Lunatech_Caliper_Body"
    merge_step(os.path.join(CAD, "Lunatech_Caliper_Master.step"),
               [("mesh", body_name, Vs, Fs), ("file", tmp_parts)], "Lunatech caliper master")
    merge_step(os.path.join(CAD, "Lunatech_Caliper_Assembly.step"),
               [("mesh", body_name, Vs, Fs), ("file", tmp_parts), ("file", tmp_rotor)],
               "Lunatech caliper + 355x32 rotor")
    merge_step(os.path.join(CAD, "Lunatech_Caliper_18inch.step"),
               [("mesh", body_name, Vs, Fs), ("file", tmp_parts), ("file", tmp_rotor), ("file", tmp_wheel)],
               "Lunatech caliper + rotor + 18in wheel reference")
    cq.exporters.export(rot, os.path.join(CAD, "Rotor_355x32_Reference.step"))
    cq.exporters.export(whl, os.path.join(CAD, "Wheel_18inch_Reference.step"))
    # tessellated parts for export_stl / render
    tess = {}
    for n, s in named + [("Rotor", rot), ("Wheel", whl)]:
        vs, fs = s.tessellate(0.05, 0.3)
        tess[n] = (np.array([[v.x, v.y, v.z] for v in vs]), np.array(fs, np.int64))
    np.savez_compressed(os.path.join(BUILD, "parts_tess.npz"),
                        **{f"{n}__V": v for n, (v, f) in tess.items()},
                        **{f"{n}__F": f for n, (v, f) in tess.items()})
    np.savez_compressed(os.path.join(BUILD, "body_step_mesh.npz"), V=Vs, F=Fs)
    for f in (tmp_parts, tmp_rotor, tmp_wheel):
        os.remove(f)
    log(f"STEP files written in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
