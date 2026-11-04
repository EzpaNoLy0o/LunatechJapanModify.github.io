"""Quick low-resolution preview of the body (design iteration helper)."""
import sys
import time
import os
import numpy as np
from skimage import measure
import geometry as G
import render as R

def body_mesh(p, h):
    sc, info = G.build_scene(p)
    lo, hi = G.domain(p, h)
    t = time.time()
    f, org = sc.grid(lo, hi, h)
    f = np.pad(f, 1, constant_values=20.0)
    org = org - h
    V, F, _, _ = measure.marching_cubes(f, 0.0, spacing=(h, h, h))
    V += org
    F = F.copy()
    print(f"grid {f.shape} prims {len(sc.rows)} eval+mc {time.time()-t:.1f}s tris {len(F)}")
    return V, F, info, sc

if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/prev"
    os.makedirs(outdir, exist_ok=True)
    p = G.load_params()
    V, F, info, sc = body_mesh(p, h)
    print("bbox", V.min(0).round(1), V.max(0).round(1), "ext", (V.max(0) - V.min(0)).round(1))
    tgt = 0.5 * (V.min(0) + V.max(0))
    views = sys.argv[3].split(",") if len(sys.argv) > 3 else ["Front", "Top", "Isometric", "Iso_Rear", "Rear", "Bottom", "Left"]
    for v in views:
        R.view([(V, F, "titanium")], v, tgt, dist=1000, ortho=340, size=(1200, 820), ss=1,
               out=f"{outdir}/{v}.png", title=v)
    np.savez_compressed(f"{outdir}/mesh.npz", V=V, F=F)
