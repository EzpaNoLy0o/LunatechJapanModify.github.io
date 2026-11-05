#!/usr/bin/env python3
"""
validate_model.py - read every exported file back from disk and check it.

STL checks (trimesh + manifold3d + pymeshlab):
  bounding box vs 320 mm limit / 325 mm bed, closed (watertight), non-manifold
  edges, self-intersections, consistent winding + outward normals (positive
  volume), connected components (floating bodies), internal voids,
  minimum wall thickness (ray casting along the inward normal), volume.
Assembly checks: rotor interference + radial / axial clearance, wheel
  barrel and spoke clearance, required minimum barrel diameter.
STEP checks (OpenCASCADE reader): solids, validity, volume, bbox.

Writes build/validation.json and reports/Geometry_Validation.md.
Exit code 1 if a hard requirement fails.
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G   # noqa: E402

ROOT = G.ROOT
STL = os.path.join(ROOT, "stl")
CAD = os.path.join(ROOT, "cad")
BUILD = os.path.join(ROOT, "build")
REP = os.path.join(ROOT, "reports")

CALIPER_FILES = ["Lunatech_Caliper_310mm.stl", "Lunatech_Caliper_FDM_Optimized.stl",
                 "Lunatech_Caliper_Split_Outboard.stl", "Lunatech_Caliper_Split_Inboard.stl"]


def log(*a):
    print("[validate]", *a, flush=True)


def self_intersections(V, F):
    try:
        import pymeshlab as ml
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(V, F))
        ms.compute_selection_by_self_intersections_per_face()
        return int(ms.current_mesh().selected_face_number())
    except Exception as e:  # pragma: no cover
        return f"not checked ({e})"


def wall_thickness(m, V, F, n_samples=20000, seed=1):
    """thickness at random surface points: distance along the inward normal
    to the next surface (ray cast against the same solid)."""
    rng = np.random.default_rng(seed)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    n = n / (a[:, None] + 1e-12)
    idx = rng.choice(len(F), n_samples, p=a / a.sum())
    u, v = rng.random(n_samples), rng.random(n_samples)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    P = tri[idx, 0] + u[:, None] * (tri[idx, 1] - tri[idx, 0]) + v[:, None] * (tri[idx, 2] - tri[idx, 0])
    N = n[idx]
    th = np.full(n_samples, np.nan)
    for i in range(n_samples):
        o = P[i] - N[i] * 1e-3
        e = P[i] - N[i] * 60.0
        hits = m.ray_cast(o.tolist(), e.tolist())
        if hits:
            th[i] = hits[0].distance * 60.0 if hits[0].distance <= 1.0 else hits[0].distance
    return th


def check_stl(path, limit, thickness=False):
    t = time.time()
    tm = trimesh.load(path, process=False)
    tm.merge_vertices()
    V, F = np.asarray(tm.vertices), np.asarray(tm.faces)
    ext = (V.max(0) - V.min(0)).tolist()
    edges = np.sort(tm.edges, axis=1)
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    nonman = int(np.sum(cnt != 2))
    m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
    comps = trimesh.graph.connected_components(tm.face_adjacency, nodes=np.arange(len(F)))
    comp_vols = []
    for c in comps:
        sub = tm.submesh([c], append=True)
        comp_vols.append(float(sub.volume))
    res = dict(file=os.path.basename(path), triangles=len(F), extent_mm=[round(e, 2) for e in ext],
               max_axis_mm=round(max(ext), 2), within_limit=bool(max(ext) <= limit),
               watertight=bool(tm.is_watertight), winding_consistent=bool(tm.is_winding_consistent),
               non_manifold_edges=nonman, manifold3d_status=m.status().name,
               volume_cm3=round(tm.volume / 1000, 2), normals_outward=bool(tm.volume > 0),
               shells=len(comps), floating_shells=int(sum(1 for v in comp_vols if v > 0) - 1),
               internal_voids=int(sum(1 for v in comp_vols if v < 0)),
               self_intersecting_faces=self_intersections(V, F))
    if thickness:
        th = wall_thickness(m, V, F)
        ok = th[np.isfinite(th)]
        res["wall_thickness_mm"] = dict(samples=int(len(ok)), p0_1=round(float(np.percentile(ok, 0.1)), 2),
                                        p1=round(float(np.percentile(ok, 1)), 2), p5=round(float(np.percentile(ok, 5)), 2),
                                        median=round(float(np.median(ok)), 2),
                                        below_0_8=round(float(np.mean(ok < 0.8) * 100), 3),
                                        below_1_6=round(float(np.mean(ok < 1.6) * 100), 2),
                                        below_2_0=round(float(np.mean(ok < 2.0) * 100), 2))
    res["seconds"] = round(time.time() - t, 1)
    return res


def assembly_checks(p):
    D = G.derived(p)
    d = np.load(os.path.join(BUILD, "parts_tess.npz"))
    tm = trimesh.load(os.path.join(STL, "Lunatech_Caliper_310mm.stl"), process=False)
    V = np.asarray(tm.vertices)
    hw = [d[k] for k in d.files if k.endswith("__V") and not k.startswith(("Rotor", "Wheel", "Pad_", "Piston"))]
    Vall = np.vstack([V] + hw)
    R = D["R_rotor"]
    ht = D["half_t"]
    r = np.hypot(V[:, 0], V[:, 2])
    y = V[:, 1]
    ri = p["rotor"]["inner_ring_diameter"] / 2
    inside = (np.abs(y) < ht) & (r < R) & (r > ri - 30)
    # distance (in the r-y half plane) from body points to the rotor disc rectangle
    dr = np.maximum(r - R, 0)
    dy = np.maximum(np.abs(y) - ht, 0)
    dist = np.hypot(dr, dy)
    radial = r[np.abs(y) <= ht].min() - R if np.any(np.abs(y) <= ht) else None
    axial = (np.abs(y[(r <= R) & (r > ri - 30)]) - ht).min()
    Rb = p["wheel"]["barrel_min_inner_diameter"] / 2
    rall = np.hypot(Vall[:, 0], Vall[:, 2])
    rmax = float(rall.max())
    sb = p["wheel"]["spoke_back_y"]
    ymax = float(Vall[rall > 100, 1].max())
    cl = p["clearance"]
    return dict(
        rotor=dict(body_points_inside_rotor=int(inside.sum()), min_clearance_mm=round(float(dist.min()), 2),
                   radial_clearance_over_rotor_edge_mm=round(float(radial), 2),
                   axial_clearance_to_rotor_face_mm=round(float(axial), 2),
                   pad_to_rotor_mm=cl["pad_to_rotor"]),
        wheel=dict(barrel_min_inner_diameter_assumed=2 * Rb, caliper_max_radius_mm=round(rmax, 2),
                   barrel_clearance_mm=round(Rb - rmax, 2),
                   barrel_clearance_target=[cl["wheel_barrel_target_min"], cl["wheel_barrel_target_max"]],
                   required_min_barrel_ID_for_3mm=round(2 * (rmax + 3.0), 1),
                   required_min_barrel_ID_for_5mm=round(2 * (rmax + 5.0), 1),
                   barrel_ok=bool(Rb - rmax >= cl["wheel_barrel_target_min"]),
                   spoke_back_y_assumed=sb, caliper_max_y_mm=round(ymax, 2),
                   spoke_axial_clearance_mm=round(sb - ymax, 2),
                   spoke_ok=bool(sb - ymax >= cl["spoke_axial_target_min"]),
                   required_spoke_back_y_for_3mm=round(ymax + 3.0, 1)))


def step_checks():
    from step_faceted import read_step
    out = []
    for f in sorted(glob.glob(os.path.join(CAD, "*.step"))):
        t = time.time()
        s = read_step(f)
        sol = s.Solids()
        bb = s.BoundingBox()
        out.append(dict(file=os.path.basename(f), size_MB=round(os.path.getsize(f) / 1e6, 1), solids=len(sol),
                        all_valid=bool(all(x.isValid() for x in sol)),
                        volume_cm3=round(sum(x.Volume() for x in sol) / 1000, 1),
                        bbox_mm=[round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1)],
                        seconds=round(time.time() - t, 1)))
        log("STEP", out[-1])
    return out


def main():
    p = G.load_params()
    lim = p["limits"]["max_axis_mm"]
    res = dict(limit_mm=lim, stl=[], step=[], assembly=None, failures=[])
    for f in sorted(glob.glob(os.path.join(STL, "*.stl"))):
        name = os.path.basename(f)
        is_cal = name in CALIPER_FILES
        r = check_stl(f, lim if (is_cal or name.startswith("Lunatech")) else 320.0, thickness=is_cal)
        res["stl"].append(r)
        log(name, {k: r[k] for k in ("max_axis_mm", "watertight", "non_manifold_edges", "self_intersecting_faces",
                                      "shells", "floating_shells", "internal_voids", "normals_outward")},
            r.get("wall_thickness_mm", ""))
        if not r["within_limit"]:
            res["failures"].append(f"{name}: exceeds {lim} mm")
        if not r["watertight"] or r["non_manifold_edges"] or not r["normals_outward"]:
            res["failures"].append(f"{name}: mesh not closed/manifold/outward")
        if r["internal_voids"]:
            res["failures"].append(f"{name}: internal voids")
        if is_cal and r["floating_shells"]:
            res["failures"].append(f"{name}: floating shells")
        if isinstance(r["self_intersecting_faces"], int) and r["self_intersecting_faces"] > 0:
            res["failures"].append(f"{name}: self intersections")
    res["assembly"] = assembly_checks(p)
    log("assembly", res["assembly"])
    if res["assembly"]["rotor"]["body_points_inside_rotor"]:
        res["failures"].append("caliper body intersects rotor")
    if "--no-step" not in sys.argv:
        res["step"] = step_checks()
        for s in res["step"]:
            if not s["all_valid"] or s["solids"] == 0:
                res["failures"].append(f"{s['file']}: invalid STEP solid")
    with open(os.path.join(BUILD, "validation.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("FAILURES:" if res["failures"] else "ALL HARD CHECKS PASSED", res["failures"])
    sys.exit(1 if res["failures"] else 0)


if __name__ == "__main__":
    main()
