#!/usr/bin/env python3
"""render_views.py - produce the comparison / documentation renders."""
import json
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render as R          # noqa: E402
import geometry as G        # noqa: E402

ROOT = G.ROOT
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "renders")


def load(name):
    d = np.load(os.path.join(BUILD, name + ".npz"))
    return d["V"], d["F"]


def parts():
    d = np.load(os.path.join(BUILD, "parts_tess.npz"))
    names = sorted({k.split("__")[0] for k in d.files})
    return {n: (d[n + "__V"], d[n + "__F"]) for n in names}


def mat_for(n):
    if n.startswith("Piston"):
        return "piston"
    if n.startswith("Pad_"):
        return "pad"
    if n.startswith(("MountBolt", "Bleeder", "PadPin")):
        return "bolt"
    if n == "Rotor":
        return "rotor"
    if n == "Wheel":
        return "wheel"
    return "titanium"


def main(only=None):
    os.makedirs(OUT, exist_ok=True)
    Vp, Fp = load("body_pocket")
    Vl, Fl = load("logo_insert")
    P = parts()
    cal = [(Vp, Fp, "titanium"), (Vl, Fl, "logo")]
    hw = [(v, f, mat_for(n)) for n, (v, f) in P.items() if n not in ("Rotor", "Wheel")]
    full = cal + hw
    tgt = 0.5 * (Vp.min(0) + Vp.max(0))
    meta = json.load(open(os.path.join(BUILD, "meta.json")))
    ext = Vp.max(0) - Vp.min(0)
    sub = f"Lunatech 8-piston titanium-style caliper replica  |  {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm"
    views = {
        "Front": ("Front (outboard / logo side)", "Front"),
        "Rear": ("Rear (inboard / mounting side)", "Rear"),
        "Top": ("Top (radial view)", "Top"),
        "Bottom": ("Bottom (rotor side)", "Bottom"),
        "Left": ("Left end (-X, leading)", "Left"),
        "Right": ("Right end (+X, trailing)", "Right"),
        "Isometric": ("Isometric front", "Isometric"),
        "Isometric_Rear": ("Isometric rear", "Iso_Rear"),
        "Isometric_Top": ("Isometric top", "Iso_Top"),
    }
    for fn, (title, v) in views.items():
        if only and fn not in only:
            continue
        ortho = 225 if v in ("Front", "Rear", "Top", "Bottom") else (190 if v in ("Left", "Right") else 300)
        R.view(full, v, tgt, dist=1200, ortho=ortho, size=(1600, 1000), ss=2,
               title=f"Lunatech Caliper - {title}", subtitle=sub, out=os.path.join(OUT, fn + ".png"))
        print("render", fn)
    if not only or "Side" in only:
        R.view(full, "Right", tgt, dist=1200, ortho=190, size=(1600, 1000), ss=2,
               title="Lunatech Caliper - Side (end view, +X)", subtitle=sub, out=os.path.join(OUT, "Side.png"))
    if not only or "Logo" in only:
        c = np.array([0.0, 80.0, 183.0])
        R.render(full, c + np.array([0, 420, 40]), c, up=(0, 0, 1), ortho=110, size=(1600, 900), ss=2,
                 title="Lunatech logo on the main bridge (raised 0.8 mm, curved)", out=os.path.join(OUT, "Logo_Detail.png"))
    if not only or "Rotor_Assembly" in only:
        Vr, Fr = P["Rotor"]
        objs = full + [(Vr, Fr, "rotor")]
        R.view(objs, "Isometric", np.array([0, 0, 40.0]), dist=1500, ortho=470, size=(1600, 1200), ss=2,
               title="Rotor assembly - 355 x 32 mm ventilated rotor (full scale reference)",
               subtitle="caliper on rotor centre plane Y=0, radial clearance %.1f mm" % meta["params"]["clearance"]["rotor_radial"],
               out=os.path.join(OUT, "Rotor_Assembly.png"))
        print("render Rotor_Assembly")
    if not only or "Wheel_Fitment" in only:
        wheel_fitment(full, P, meta)


def wheel_fitment(full, P, meta):
    """3D cut-away (wheel sectioned at X<0 removed) + 2D clearance section."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    Vw, Fw = P["Wheel"]
    Vr, Fr = P["Rotor"]
    # cut the wheel: keep triangles with all vertices x < 0 -> see caliper inside
    keep = (Vw[Fw][:, :, 0] < -5).all(1)
    objs = full + [(Vr, Fr, "rotor"), (Vw, Fw[keep], "wheel")]
    im = R.view(objs, "Isometric", np.array([0, 10, 40.0]), dist=1600, ortho=560, size=(1300, 1100), ss=2,
                title="18-inch wheel fitment (wheel cut away)", subtitle="barrel min. ID %.0f mm (assumed), spoke back Y=%.0f mm (assumed)" % (
                    meta["params"]["wheel"]["barrel_min_inner_diameter"], meta["params"]["wheel"]["spoke_back_y"]))
    val = json.load(open(os.path.join(ROOT, "build", "validation.json"))) if os.path.exists(
        os.path.join(ROOT, "build", "validation.json")) else None
    Vb, Fb = load("body_logo")
    r = np.hypot(Vb[:, 0], Vb[:, 2])
    fig, ax = plt.subplots(1, 2, figsize=(13, 11), dpi=100)
    # left: front projection (X-Z)
    a = ax[0]
    th = np.linspace(0, 2 * np.pi, 400)
    w = meta["params"]["wheel"]
    Rb = w["barrel_min_inner_diameter"] / 2
    Ro = meta["params"]["rotor"]["outer_diameter"] / 2
    a.fill(Rb * np.cos(th) * 1.12, Rb * np.sin(th) * 1.12, color="#e9eaee")
    a.fill(Rb * np.cos(th), Rb * np.sin(th), color="white")
    a.plot(Rb * np.cos(th), Rb * np.sin(th), color="#4a5263", lw=1.5, label=f"wheel barrel min. ID {2 * Rb:.0f} (assumed)")
    a.plot(Ro * np.cos(th), Ro * np.sin(th), color="#8a8f99", lw=1.2, ls="--", label=f"rotor OD {2 * Ro:.0f}")
    idx = np.random.default_rng(0).choice(len(Vb), min(120000, len(Vb)), replace=False)
    a.scatter(Vb[idx, 0], Vb[idx, 2], s=0.2, color="#2b6cb0", alpha=0.35)
    j = np.argmax(r)
    a.plot([0, Vb[j, 0] / r[j] * Rb], [0, Vb[j, 2] / r[j] * Rb], color="#c0392b", lw=0.8)
    a.annotate(f"max caliper radius {r[j]:.1f} mm\nclearance {Rb - r[j]:.1f} mm", (Vb[j, 0], Vb[j, 2]),
               xytext=(-235, 228), fontsize=10, arrowprops=dict(arrowstyle="->", color="#c0392b"))
    a.set_aspect("equal")
    a.set_xlim(-240, 240)
    a.set_ylim(-60, 250)
    a.set_title("Front projection (X-Z), rotor axis at origin")
    a.legend(loc="lower left", fontsize=9)
    a.grid(alpha=0.3)
    # right: (y, r) section envelope
    b = ax[1]
    b.scatter(Vb[idx, 1], r[idx], s=0.2, color="#2b6cb0", alpha=0.35, label="caliper (all points, revolved)")
    b.axhline(Rb, color="#4a5263", lw=1.5, label="barrel min. inner radius")
    b.axhline(Ro, color="#8a8f99", ls="--", lw=1.2, label="rotor outer radius")
    b.axvline(w["spoke_back_y"], color="#7d3c98", lw=1.5, label="spoke back face (assumed)")
    b.axvspan(-meta["params"]["rotor"]["thickness"] / 2, meta["params"]["rotor"]["thickness"] / 2, ymax=(Ro - 80) / 160,
              color="#bbb", alpha=0.4)
    ymax = Vb[r > 100, 1].max()
    b.annotate(f"axial spoke clearance {w['spoke_back_y'] - ymax:.1f} mm", (ymax, 150), xytext=(-40, 130),
               fontsize=11, arrowprops=dict(arrowstyle="->", color="#7d3c98"))
    b.set_xlim(-110, 110)
    b.set_ylim(80, 240)
    b.set_xlabel("Y (axial, + = outboard) [mm]")
    b.set_ylabel("radius from rotor axis [mm]")
    b.set_title("Revolved section (Y-R envelope)")
    b.legend(loc="upper left", fontsize=9)
    b.grid(alpha=0.3)
    fig.tight_layout()
    tmp = os.path.join(BUILD, "_fit.png")
    fig.savefig(tmp)
    plt.close(fig)
    sec = Image.open(tmp)
    sec = sec.resize((int(sec.width * 1100 / sec.height), 1100))
    W = Image.new("RGB", (im.width + sec.width, 1100), "white")
    W.paste(im, (0, 0))
    W.paste(sec, (im.width, 0))
    W.save(os.path.join(OUT, "Wheel_Fitment.png"))
    os.remove(tmp)
    print("render Wheel_Fitment")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
