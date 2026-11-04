"""
geometry.py - Parametric layout of the Lunatech 8-piston caliper replica.

Everything is derived from parameters.json.  The caliper is laid out on the
rotor arc: a point is addressed as (theta [deg], r [mm], y [mm]) where theta
is measured from +Z towards +X around the rotor axis (Y), r is the distance
from the rotor axis and y is the axial position (+Y = outboard).

Design intent (see reports/Reference_Comparison.md):
  * strongly arched "banana" plan like the 2018 titanium caliper
  * outboard MAIN BRIDGE: broad, crowned, variable-thickness swept beam that
    widens in the middle (logo) and forks into several ribs at each end
  * upper bridge: K/X shaped branching cross-members over the rotor
  * lobed piston housings (4 + 4) blended into rails
  * end trusses: lug nodes with branching tapered ribs -> triangular,
    teardrop and slit-like through openings of different sizes
  * lower rails with short struts -> row of small openings under the pistons
"""
import json
import math
import os
import numpy as np
from sdf_core import SDFScene

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_params(path=None):
    with open(path or os.path.join(HERE, "parameters.json")) as f:
        return json.load(f)


def P(th_deg, r, y):
    t = math.radians(th_deg)
    return np.array([r * math.sin(t), y, r * math.cos(t)])


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def derived(p):
    """Derived dimensions shared by geometry, parts and validation."""
    ro, cl, pad, pis, body = p["rotor"], p["clearance"], p["pad"], p["piston"], p["body"]
    d = {}
    d["R_rotor"] = ro["outer_diameter"] / 2
    d["half_t"] = ro["thickness"] / 2
    d["y_lining"] = d["half_t"] + cl["pad_to_rotor"]
    d["y_pad_back"] = d["y_lining"] + pad["lining_thickness"] + pad["backing_thickness"]
    d["y_backing_front"] = d["y_lining"] + pad["lining_thickness"]
    d["y_face"] = d["y_pad_back"] + pis["proud_of_housing"]
    d["y_bore_bottom"] = d["y_face"] + pis["bore_depth"]
    d["y_back"] = d["y_bore_bottom"] + pis["housing_back_wall"]
    d["R_p"] = 0.5 * (pad["radial_inner"] + pad["radial_outer"])
    d["r_slot"] = d["R_rotor"] + cl["rotor_radial"]
    d["R_cap"] = p["wheel"]["barrel_min_inner_diameter"] / 2 - cl["wheel_barrel_design"]
    d["y_slot"] = d["half_t"] + cl["rotor_axial"]
    dia = pis["diameters"]
    s = [0.0]
    for i in range(1, len(dia)):
        s.append(s[-1] + 0.5 * (dia[i - 1] + dia[i]) + pis["web_between_bores"])
    s = np.array(s)
    s_lo = s[0] - dia[0] / 2 - pad["end_margin"]
    s_hi = s[-1] + dia[-1] / 2 + pad["end_margin"]
    shift = 0.5 * (s_lo + s_hi)
    s -= shift
    d["piston_s"] = s
    d["piston_th"] = np.degrees(s / d["R_p"]) + pis["angle_offset_deg"]
    d["pad_th0"] = math.degrees((s_lo - shift) / d["R_p"]) + pis["angle_offset_deg"]
    d["pad_th1"] = math.degrees((s_hi - shift) / d["R_p"]) + pis["angle_offset_deg"]
    d["housing_R"] = [x / 2 + pis["housing_wall_radial"] for x in dia]
    # end geometry is driven by the target length: the outer tip arch is
    # placed so that its outermost point lands at L/2.
    L = p["limits"]["target_length_mm"]
    d["tip_r"] = 182.0
    d["tip_rib"] = 6.2
    d["tip_th"] = math.degrees(math.asin((L / 2 - d["tip_rib"] - 0.2) / (d["tip_r"] + 2.0)))
    return d


def build_scene(p, quality="final"):
    D = derived(p)
    b = p["body"]
    sc = SDFScene()
    rs = b["rib_scale"]
    KM, Km = b["blend_major"], b["blend_minor"]
    thp = D["piston_th"]
    hR = D["housing_R"]
    Rp = D["R_p"]
    th0, th1 = D["pad_th0"], D["pad_th1"]
    yf, yb, ybb = D["y_face"], D["y_back"], D["y_bore_bottom"]
    tip = D["tip_th"]
    tr = D["tip_r"]
    sides = (+1, -1)
    info = {"derived": D}
    ytr = b["top_rail_y"]
    lrc, lyy = b["lower_rail_radial_center"], b["lower_rail_y"]
    la = b["lower_rail_angle_deg"]
    lug_th, lug_r = b["end_lug_angle_deg"], b["end_lug_radius"]
    n = len(thp)

    # ------------------------------------------------------------------
    # 1. Piston housings: lobed, back-domed cylinders
    # ------------------------------------------------------------------
    for sg in sides:
        for i, th in enumerate(thp):
            y0 = yf - 4.0
            sc.cylinder(P(th, Rp, sg * 0.5 * (y0 + yb)), (0, 1, 0), 0.5 * (yb - y0), hR[i],
                        rnd=p["piston"]["housing_back_round"], k=KM, name="housing")
            # shallow dome on the bore bottom (reads as a closed bore cap)
            sc.ellipsoid(P(th, Rp, sg * (yb - 3.0)), (hR[i] - 5.0, 5.2, hR[i] - 5.0), k=4.0, name="dome")
        # radiating root-ribs: lobe -> top rail and lobe -> lower rail
        for i, th in enumerate(thp):
            lean = (-1) ** i * 2.0
            top_end = P(th + lean, 186.0, sg * (ytr + 9.0))
            sc.rib([P(th, Rp + hR[i] - 4.0, sg * (yb - 7.0)), P(th + 0.5 * lean, 178.5, sg * (yb - 12.0)), top_end],
                   [5.0 * rs, 3.6 * rs, 4.6 * rs], k=Km + 1.5, name="lobe_up")
            sc.rib([P(th, Rp - hR[i] + 4.0, sg * (yb - 8.0)), P(th - 0.4 * lean, 0.5 * (Rp - hR[i] + lrc) - 1.0, sg * (lyy + 6.0)),
                    P(th - lean, lrc + 2.0, sg * lyy)],
                   [4.6 * rs, 3.0 * rs, 4.2 * rs], k=Km + 1.0, name="lobe_down")
        # valley ribs between lobes (forked towards the bridges)
        for i in range(n - 1):
            tm = 0.5 * (thp[i] + thp[i + 1])
            rv = Rp + 0.5 * (hR[i] + hR[i + 1]) - 7.0
            sc.rib([P(tm, rv, sg * (yb - 4.0)), P(tm + 2.5, 181.0, sg * (yb - 2.0)), P(tm + 4.5, 187.0, sg * (yb + 1.0))],
                   [3.4 * rs, 2.6 * rs, 3.0 * rs], k=Km, name="valley_rib")

    # ------------------------------------------------------------------
    # 2. Top side rails
    # ------------------------------------------------------------------
    trc, thr, tyr = b["top_rail_radial_center"], b["top_rail_half_r"], b["top_rail_half_y"]
    for sg in sides:
        ang = b["top_rail_angle_deg"] + (1.5 if sg < 0 else 0.0)
        sc.band(math.radians(-ang), math.radians(ang),
                rc=lambda u: trc - 1.5 * (2 * u - 1) ** 2,
                yc=lambda u, sg=sg: sg * (ytr + 2.5 * (1 - (2 * u - 1) ** 2)
                                          + 1.8 * math.sin(3.0 * math.pi * (2 * u - 1) + (0.6 if sg > 0 else 2.1))),
                ht=lambda u: tyr * (0.68 + 0.32 * math.cos(2.5 * math.pi * (2 * u - 1)) ** 2),
                hw=lambda u: thr * (0.85 + 0.15 * (1 - (2 * u - 1) ** 2)),
                rr=lambda u: 5.0, sigma=sg, k=KM, name="top_rail")

    # ------------------------------------------------------------------
    # 3. Outboard MAIN BRIDGE (logo carrier) - broad crowned swept beam
    # ------------------------------------------------------------------
    mb = math.radians(tip - 3.5)
    mrc, mhc, mhe = b["main_bridge_radial_center"], b["main_bridge_half_width_center"], b["main_bridge_half_width_end"]
    myc, mye = b["main_bridge_y_center"], b["main_bridge_y_end"]
    mtc, mte = b["main_bridge_half_thickness_center"], b["main_bridge_half_thickness_end"]

    def mb_rc(u):
        v = 2 * u - 1
        drop = 6.0 if v < 0 else 8.0            # asymmetric ends (lead/trail)
        return mrc + 2.0 * (1 - v * v) - drop * v ** 4

    def mb_w(u):
        v = 2 * u - 1
        return mhe + (mhc - mhe) * (1 - v * v) ** 1.1

    def mb_y(u):
        v = 2 * u - 1
        return mye + (myc - mye) * (1 - v * v) ** 0.7   # bows inward at the ends

    def mb_t(u):
        v = 2 * u - 1
        return mte + (mtc - mte) * (1 - v * v) ** 1.2

    def mb_rr(u):
        return min(mb_t(u), mb_w(u)) * 0.85

    def mb_cr(u):
        v = 2 * u - 1
        return b["main_bridge_crown"] * (1 - v * v) ** 1.5

    sc.band(-mb, mb, rc=mb_rc, yc=mb_y, ht=mb_t, hw=mb_w, rr=mb_rr, crown=mb_cr, sigma=+1.0,
            n=120, k=KM, name="main_bridge")
    info["main_bridge"] = dict(th0=-mb, th1=mb, rc=mb_rc, yc=mb_y, ht=mb_t, hw=mb_w, cr=mb_cr)

    # inboard bridge: slimmer, lower and more strongly bowed (asymmetric)
    ib = math.radians(tip - 5.0)
    sc.band(-ib, ib, rc=lambda u: b["inboard_bridge_radial_center"] - 6.0 * (2 * u - 1) ** 4,
            yc=lambda u: b["inboard_bridge_y"] + 9.0 * (2 * u - 1) ** 2,
            ht=lambda u: 3.6 + 1.2 * (1 - (2 * u - 1) ** 2),
            hw=lambda u: 5.5 + 4.0 * (1 - (2 * u - 1) ** 2),
            crown=lambda u: 1.2 * (1 - (2 * u - 1) ** 2),
            rr=lambda u: 3.4, sigma=-1.0, k=KM, name="inboard_bridge")

    # ------------------------------------------------------------------
    # 4. Connectors top rail <-> side bridges (forked -> top windows)
    # ------------------------------------------------------------------
    mids = [0.5 * (thp[i] + thp[i + 1]) for i in range(n - 1)]
    cons = [thp[0] - 9.0] + mids + [thp[-1] + 9.0]
    for j, tm in enumerate(cons):
        root = P(tm, 190.5, 53.0)
        if j % 2 == 0:
            sc.rib([root, P(tm - 3.0, 188.5, 61.0), P(tm - 5.0, 186.0, 68.0)],
                   [4.4 * rs, 3.1 * rs, 3.6 * rs], k=Km + 2.0, name="conn_out")
            sc.rib([root, P(tm + 3.0, 189.0, 61.5), P(tm + 5.0, 187.0, 68.0)],
                   [4.4 * rs, 3.0 * rs, 3.5 * rs], k=Km + 2.0, name="conn_out")
        else:
            sc.rib([root, P(tm + 0.8, 189.0, 61.0), P(tm + 1.5, 187.0, 68.0)],
                   [4.6 * rs, 3.5 * rs, 4.2 * rs], k=Km + 2.0, name="conn_out")
        lean = 2.5 if j % 2 == 0 else -2.5
        sc.rib([P(tm, 190.0, -53.0), P(tm + lean, 189.0, -60.0), P(tm + 2 * lean, 187.0, -66.0)],
               [4.2 * rs, 3.2 * rs, 3.8 * rs], k=Km + 2.0, name="conn_in")

    # ------------------------------------------------------------------
    # 5. Upper bridge over the rotor: central X + K-shaped end crossings
    # ------------------------------------------------------------------
    cx = 10.0
    sc.ellipsoid(P(0.0, 193.5, 0.0), (9.0, 7.0, 4.8), k=KM, name="x_node")
    for a in (-cx, cx):
        for sg in sides:
            sc.rib([P(0.0, 193.5, 0.0), P(0.62 * a, 193.5, sg * ytr * 0.45), P(a, 190.0, sg * (ytr - 1))],
                   [5.0 * rs, 3.9 * rs, 5.6 * rs], k=Km + 1.0, name="x_arm")
    for e in (-1, 1):
        the = th1 if e > 0 else th0
        tc = the + e * 5.0
        node = P(tc, 193.0, 0.0)
        sc.ellipsoid(node, (9.0, 8.0, 5.2), k=KM, name="k_node")
        for sg in sides:
            sc.rib([node, P(tc - e * 2.5, 192.5, sg * 22.0), P(the - e * 0.5, 190.0, sg * ytr)],
                   [5.6 * rs, 4.6 * rs, 6.0 * rs], k=Km + 1.0, name="k_arm")
            sc.rib([node, P(tc + e * 3.5, 192.0, sg * 24.0), P(tc + e * 7.5, 188.0, sg * (ytr + 3))],
                   [5.4 * rs, 4.2 * rs, 5.6 * rs], k=Km + 1.0, name="k_arm")

    for e in (-1, 1):
        the = th1 if e > 0 else th0
        sc.rib([P(0.0, 193.5, 0.0), P(e * 0.35 * abs(the), 194.0, 1.5 * e), P(e * 0.7 * abs(the), 194.0, -1.0 * e),
                P(the + e * 5.0, 193.0, 0.0)], [4.6, 3.4, 3.6, 4.8], k=Km + 1.0, name="spine")

    # 5b. Roof skin over the rotor (thin, crowned) - pierced by windows below
    rf = math.radians(max(abs(th0), abs(th1)) + 9.5)
    sc.band(-rf, rf, rc=lambda u: 192.0 + 1.2 * (1 - (2 * u - 1) ** 2),
            yc=lambda u: 0.0,
            ht=lambda u: (ytr + 1.0) - 6.0 * (2 * u - 1) ** 8,
            hw=lambda u: 2.7,
            rr=lambda u: 2.5, sigma=1.0, n=64, k=KM, name="roof")
    info["roof_rc"] = 192.0

    # ------------------------------------------------------------------
    # 6. Lower rails (variable section) - closes the bottom of each half
    # ------------------------------------------------------------------
    lhr, lhy = b["lower_rail_half_r"], b["lower_rail_half_y"]
    for sg in sides:
        sc.band(math.radians(-la - (2.0 if sg < 0 else 0.0)), math.radians(la),
                rc=lambda u: lrc + 1.5 * (2 * u - 1) ** 2,
                yc=lambda u, sg=sg: sg * (lyy + 3.0 * (2 * u - 1) ** 2),
                ht=lambda u: lhy * (0.7 + 0.3 * (1 - (2 * u - 1) ** 2)),
                hw=lambda u: lhr * (0.85 + 0.15 * math.cos(math.pi * (2 * u - 1) * 2.0) ** 2),
                rr=lambda u: 3.8, sigma=sg, k=KM, name="lower_rail")

    # ------------------------------------------------------------------
    # 7. End trusses, membranes, abutments, outer tip arches
    # ------------------------------------------------------------------
    for e in (-1, 1):
        the = th1 if e > 0 else th0
        ilast = n - 1 if e > 0 else 0
        thl = thp[ilast]
        # outer tip arch crossing over the rotor edge (sets the overall length)
        sc.rib([P(e * (tip - 5.0), tr - 5.0, -64.0), P(e * (tip - 1.2), tr - 0.5, -36.0), P(e * tip, tr + 1.0, 0.0),
                P(e * (tip - 1.2), tr - 0.5, 36.0), P(e * (tip - 4.5), tr - 5.0, 64.0)],
               [5.6 * rs, 5.4 * rs, D["tip_rib"] * rs, 5.4 * rs, 5.6 * rs], k=KM, name="tip_arch")
        sc.ellipsoid(P(e * (tip - 1.0), tr + 0.5, 0.0), (6.5, 8.5, 6.0), k=Km, name="tip_node")
        # rib fan K-node -> tip arch (branching, slightly domed)
        kn = P(the + e * 5.0, 193.0, 0.0)
        for yy, rr0 in ((-46.0, 4.4), (-17.0, 3.6), (19.0, 3.8), (47.0, 4.6)):
            tpt = P(e * (tip - 1.2 - 3.0 * (abs(yy) / 64.0) ** 2), tr - 0.5 - 4.0 * (abs(yy) / 64.0) ** 2, yy)
            mid = P(0.5 * (the + e * 5.0 + e * tip) + e * 0.8, 194.5, 0.55 * yy)
            sc.rib([kn, mid, tpt], [4.8 * rs, rr0 * 0.8 * rs, rr0 * rs], k=Km + 1.0, name="end_fan")
        for sg in sides:
            lth = e * (lug_th - (1.5 if sg < 0 else 0.0))
            lr = lug_r - (3.0 if sg < 0 else 0.0)
            lug = P(lth, lr, sg * 50.0)
            sc.ellipsoid(lug, (9.0, 10.0, 9.0), k=KM, name="lug")
            kfoot = P(the + e * 12.5, 188.0, sg * (ytr + 3))
            tfoot = P(e * (tip - 4.8), tr - 5.0, sg * 64.0)
            rail_end = P(e * (la + (2.0 if (sg < 0 and e < 0) else 0.0)), lrc + 1.5, sg * (lyy + 3.0))
            hfl = P(thl + e * math.degrees(hR[ilast] / Rp) * 0.8, Rp + 3.0, sg * (yb - 8.0))
            hbot = P(thl + e * 4.0, Rp - hR[ilast] + 3.0, sg * (yb - 10.0))
            # main members
            sc.rib([kfoot, P(e * (lug_th - 7.0), 174.0, sg * 52.0), lug], [6.0 * rs, 4.6 * rs, 6.2 * rs], k=KM, name="lug_up")
            sc.rib([lug, P(e * (tip - 2.5), 170.0, sg * 58.0), tfoot], [6.0 * rs, 4.4 * rs, 5.6 * rs], k=KM, name="lug_tip")
            sc.rib([lug, P(e * (lug_th - 1.5), 136.0, sg * 54.0), P(e * (la + 4.5), 122.0, sg * 54.0), rail_end],
                   [6.2 * rs, 4.8 * rs, 4.4 * rs, 5.0 * rs], k=KM, name="lug_rail")
            sc.rib([lug, P(0.5 * (lth + thl) + e * 3.0, 154.0, sg * 56.0), hfl], [5.6 * rs, 4.0 * rs, 5.2 * rs], k=KM, name="lug_house")
            # bionic membranes with organic openings (rim thicker than web)
            mc1 = (kfoot + lug + hfl) / 3.0
            sc.membrane(kfoot, lug, hfl, 2.6, k=5.0, name="web")
            mc2 = (lug + rail_end + hbot) / 3.0
            sc.membrane(lug, rail_end, hbot, 2.6, k=5.0, name="web")
            info.setdefault("web_holes", []).append((mc1, kfoot, lug, hfl))
            info.setdefault("web_holes", []).append((mc2, lug, rail_end, hbot))
            sc.membrane(kfoot, lug, tfoot, 2.6, k=5.0, name="web")
            info["web_holes"].append(((kfoot + lug + tfoot) / 3.0, kfoot, lug, tfoot))
            # secondary ribs splitting the big end triangle (lug, kfoot, tfoot)
            m1 = 0.5 * (kfoot + lug) + np.array([0, sg * 2.0, 0])
            sc.rib([m1, P(e * (tip - 7.0), 183.0, sg * 58.0)], [3.4 * rs, 2.8 * rs], k=Km, name="sec_rib")
            # pad abutment pillar (seen through the windows)
            sc.rib([P(the + e * 2.8, 128.0, sg * 29.0), P(the + e * 2.6, 156.0, sg * 28.0),
                    P(the + e * 1.6, 186.0, sg * 36.0)], [5.2, 4.4, 5.6], k=Km, name="abutment")
            sc.rib([P(the + e * 2.8, 128.0, sg * 29.0), P(the + e * 3.5, 120.0, sg * 40.0),
                    P(e * (la - 3.0), lrc + 1.0, sg * lyy)], [4.8, 3.6, 4.4], k=Km, name="abut_foot")

        # outboard: main bridge end forks into 3 ribs
        mbe = math.degrees(mb) - 1.0
        root = P(e * mbe, mrc - 6.0, mye + 1.0)
        sc.ellipsoid(root, (6.0, 5.0, 7.0), k=Km, name="mb_fork")
        sc.rib([root, P(e * (mbe - 1.5), 168.0, 58.0), P(e * lug_th, lug_r + 3.0, 54.0)],
               [5.0 * rs, 3.8 * rs, 4.8 * rs], k=Km, name="mb_branch")
        sc.rib([root, P(e * (mbe - 7.0), 175.0, 65.0), P(thl + e * 5.0, 168.0, yb - 2.0)],
               [4.6 * rs, 3.0 * rs, 4.0 * rs], k=Km, name="mb_branch")
        sc.rib([root, P(e * (mbe + 1.8), 181.0, 63.0), P(e * (tip - 4.5), tr - 5.0, 64.0)],
               [4.4 * rs, 4.0 * rs, 5.0 * rs], k=Km, name="mb_branch")

        # inboard: radial mount boss + webs
        th_m = e * b["mount_angle_deg"]
        mc = P(th_m, b["mount_radial_center"], b["mount_y"])
        u = np.array([math.sin(math.radians(th_m)), 0.0, math.cos(math.radians(th_m))])
        sc.cylinder(mc, u, b["mount_half_length"], b["mount_boss_radius"], rnd=3.5, k=KM, name="mount")
        top = P(th_m, b["mount_radial_center"] + b["mount_half_length"] - 2.0, b["mount_y"] + 4.0)
        sc.rib([top, P(thl + e * 4.0, 150.0, -(yb - 6.0))], [6.0, 4.6], k=KM, name="mount_web")
        sc.rib([P(th_m + e * 2.0, b["mount_radial_center"] + 6, b["mount_y"] - 5.0),
                P(e * (lug_th - 1.5), lug_r - 3.0, -50.0)], [5.4, 5.6], k=KM, name="mount_web")
        sc.rib([P(th_m - e * 3.0, b["mount_radial_center"] - 9.0, b["mount_y"] + 3.0),
                P(e * (la + 1.0), lrc + 1.0, -(lyy + 1.0))], [5.0, 4.4], k=KM, name="mount_web")
        # bleeder boss (inboard, top of each end)
        sc.cylinder(P(the + e * 8.0, 182.0, -66.0), (0, 1, 0), 4.5, 6.4, rnd=1.8, k=Km, name="bleeder_boss")

    # pad-pin bosses on the inboard face
    info["pin_th"] = [-15.5, 15.5]
    for tpn in info["pin_th"]:
        sc.cylinder(P(tpn, 184.5, -66.5), (0, 1, 0), 3.5, 6.0, rnd=1.6, k=Km, name="pin_boss_in")

    # ------------------------------------------------------------------
    # 8. Cuts: rotor slot, pad pocket, bores, membrane openings, holes
    # ------------------------------------------------------------------
    rsl = D["r_slot"]
    sc.polarbox(D["y_slot"], rsl, k=1.5, op=1, name="rotor_slot")
    # wheel-barrel envelope: nothing may exceed R_cap (smoothly trimmed)
    sc.polarbox(300.0, 600.0, R_in=D["R_cap"], k=1.5, op=1, name="wheel_envelope")
    thc = math.radians(0.5 * (th0 + th1))
    hth = math.radians(0.5 * (th1 - th0)) + 0.8 / Rp
    sc.polarbox(yf, rsl, th_c=thc, half_th=hth, k=1.5, op=1, name="pad_pocket")
    ew = p["pad"]["ear_width"]
    for tpn in info["pin_th"]:
        sc.polarbox(D["y_pad_back"] + 0.8, p["pad"]["ear_top_radius"] + 1.5,
                    th_c=math.radians(tpn), half_th=(ew / 2 + 1.2) / 185.0, R_in=150.0, k=1.0, op=1,
                    name="ear_window")
    # organic openings in the membranes: teardrops pointing to the sharpest corner
    osc = b["opening_scale"]
    for (c, a1, a2, a3) in info.get("web_holes", []):
        v = [a1, a2, a3]
        far = max(v, key=lambda q: np.linalg.norm(q - c))
        near = min(v, key=lambda q: np.linalg.norm(q - c))
        dirv = (far - c) / np.linalg.norm(far - c)
        rad = 0.30 * np.linalg.norm(near - c) * osc
        sc.roundcone(c - dirv * rad * 0.4, c + dirv * rad * 1.6, rad, rad * 0.35, k=3.0, op=1, name="web_hole")
    # roof windows: rounded polygons inset from the rib cells -> uniform
    # rims, large openings in the middle, smaller ones towards the ends
    ow = b["opening_scale"]
    rmw, rhw = 192.5, 6.5
    ins, rho = 3.5 / ow, 8.0
    for sg in sides:
        sc.poly_window([(0.0, sg * 9.0), (-cx + 1.0, sg * 39.0), (cx - 1.0, sg * 39.0)], 2.5, 5.0,
                       rmw, rhw, name="roof_win")
    for e in (-1, 1):
        the = th1 if e > 0 else th0
        ae = abs(the)
        for sg in sides:
            sc.poly_window([(e * 1.0, sg * 1.0), (e * (cx - 0.5), sg * 39.0), (e * (ae - 1.0), sg * 39.0),
                            (e * (ae + 5.0), sg * 1.0)], ins, rho, rmw, rhw, name="roof_win")
            sc.poly_window([(e * (ae + 5.0), sg * 2.0), (e * (ae - 0.5), sg * 42.0), (e * (ae + 12.5), sg * 44.0)],
                           2.0, 3.6, rmw, rhw, name="roof_win_k")
    pc = p["piston"]["bore_clearance"]
    for sg in sides:
        for i, th in enumerate(thp):
            dia = p["piston"]["diameters"][i]
            y0, y1 = yf - 6.0, ybb
            sc.cylinder(P(th, Rp, sg * 0.5 * (y0 + y1)), (0, 1, 0), 0.5 * (y1 - y0), dia / 2 + pc,
                        rnd=1.2, k=0.8, op=1, name="bore")
    for tpn in info["pin_th"]:
        sc.cylinder(P(tpn, 184.5, 0.0), (0, 1, 0), 95.0, 2.75, rnd=0.0, k=0.6, op=1, name="pin_hole")
    for e in (-1, 1):
        th_m = e * b["mount_angle_deg"]
        u = np.array([math.sin(math.radians(th_m)), 0.0, math.cos(math.radians(th_m))])
        sc.cylinder(P(th_m, b["mount_radial_center"], b["mount_y"]), u, 40.0, b["mount_bolt_hole"] / 2,
                    k=0.8, op=1, name="mount_hole")
        sc.cylinder(P(th_m, b["mount_radial_center"] + b["mount_half_length"] + 6.0, b["mount_y"]), u, 6.5,
                    b["mount_bolt_hole"] / 2 + 5.0, k=1.0, op=1, name="spotface")
        the = th1 if e > 0 else th0
        sc.cylinder(P(the + e * 8.0, 182.0, -66.0), (0, 1, 0), 8.0, 2.6, k=0.5, op=1, name="bleeder_hole")

    info["pad_th"] = (th0, th1)
    return sc, info


def logo_surface(info, th, dr, p):
    """Outer surface of the main bridge at angle th (rad) and radial offset dr
    from the band centre line.  Returns (r, y_surface)."""
    mbi = info["main_bridge"]
    u = (th - mbi["th0"]) / (mbi["th1"] - mbi["th0"])
    rc, yc, ht, hw, cr = mbi["rc"](u), mbi["yc"](u), mbi["ht"](u), mbi["hw"](u), mbi["cr"](u)
    c = max(0.0, 1.0 - (dr / hw) ** 2)
    return rc + dr, yc + ht + cr * c


def domain(p, h):
    L = p["limits"]["target_length_mm"]
    lo = np.array([-L / 2 - 8, -98.0, 60.0])
    hi = np.array([L / 2 + 8, 98.0, 212.0])
    return lo, hi
