"""
parts.py - exact B-Rep (CadQuery / OpenCASCADE) parts:
pistons, brake pads, pad pins, radial-mount bolts, bleeder screws,
the 355 x 32 mm ventilated rotor and the 18-inch wheel reference.
All positions are derived from parameters.json via geometry.derived().
"""
import math
import numpy as np
import cadquery as cq
import geometry as G


def _place_axis(wp_obj, axis, origin):
    """Rotate a solid built along +Z so that +Z -> axis, then translate."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    shp = wp_obj.val() if isinstance(wp_obj, cq.Workplane) else wp_obj
    if s > 1e-9:
        ang = math.degrees(math.atan2(s, float(np.dot(z, axis))))
        shp = shp.rotate(cq.Vector(0, 0, 0), cq.Vector(*v), ang)
    elif np.dot(z, axis) < 0:
        shp = shp.rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180)
    return shp.translate(cq.Vector(*map(float, origin)))


def _plane_y(y0):
    # local x = global X, local y = global -Z, normal = +Y
    return cq.Plane(origin=(0, y0, 0), xDir=(1, 0, 0), normal=(0, 1, 0))


def _uv(th_deg, r):
    t = math.radians(th_deg)
    return (r * math.sin(t), -r * math.cos(t))


def sector(r_in, r_out, th0, th1, y0, y1):
    """Annular sector between angles th0..th1 (deg), extruded y0..y1."""
    tm = 0.5 * (th0 + th1)
    wp = (cq.Workplane(_plane_y(y0)).moveTo(*_uv(th0, r_in)).lineTo(*_uv(th0, r_out))
          .threePointArc(_uv(tm, r_out), _uv(th1, r_out)).lineTo(*_uv(th1, r_in))
          .threePointArc(_uv(tm, r_in), _uv(th0, r_in)).close().extrude(y1 - y0))
    return wp.val()


# --------------------------------------------------------------------------
def pistons(p):
    D = G.derived(p)
    out = []
    L = p["piston"]["piston_length"]
    for sg in (1, -1):
        for i, th in enumerate(D["piston_th"]):
            d = p["piston"]["diameters"][i]
            R = d / 2
            wp = (cq.Workplane("XY").circle(R).extrude(L)
                  .faces(">Z").edges().fillet(1.2)
                  .faces("<Z").workplane().circle(R - 3.2).cutBlind(-15.0)
                  .faces("<Z").edges().chamfer(0.6))
            # dust-seal groove
            groove = cq.Workplane("XY").workplane(offset=4.0).circle(R + 1).circle(R - 0.7).extrude(1.6)
            wp = wp.cut(groove)
            c = G.P(th, D["R_p"], sg * D["y_pad_back"])
            out.append((f"Piston_{'OB' if sg > 0 else 'IB'}_{i + 1}_D{int(d)}",
                        _place_axis(wp, (0, sg, 0), c)))
    return out


def pads(p, pin_th):
    D = G.derived(p)
    pd = p["pad"]
    out = []
    th0, th1 = D["pad_th0"], D["pad_th1"]
    for sg in (1, -1):
        yb0, yb1 = D["y_backing_front"], D["y_pad_back"]
        yl0, yl1 = D["y_lining"], D["y_backing_front"]
        back = sector(pd["radial_inner"], pd["radial_outer"], th0, th1, yb0, yb1)
        for tp in pin_th:
            half = math.degrees((pd["ear_width"] / 2) / 180.0)
            ear = sector(pd["radial_outer"] - 6, pd["ear_top_radius"], tp - half, tp + half, yb0, yb1)
            back = back.fuse(ear)
            hole = cq.Solid.makeCylinder(2.75, 20, cq.Vector(*G.P(tp, 184.5, -10.0 + yb0)), cq.Vector(0, 1, 0))
            back = back.cut(hole)
        lin = sector(pd["radial_inner"] + 2.0, pd["radial_outer"] - 1.5, th0 + 0.7, th1 - 0.7, yl0, yl1)
        tc = 0.5 * (th0 + th1)
        dslot = math.degrees(1.6 / 150.0)
        slot = sector(pd["radial_inner"], pd["radial_outer"], tc - dslot, tc + dslot, yl0 - 1, yl0 + 4.0)
        lin = lin.cut(slot)
        pad = back.fuse(lin).clean()
        if sg < 0:
            pad = pad.mirror("XZ")
        out.append((f"Pad_{'OB' if sg > 0 else 'IB'}", pad))
    return out


def pad_pins(p, info):
    out = []
    for k, tp in enumerate(info["pin_th"]):
        mbi = info["main_bridge"]
        th = math.radians(tp)
        u = (th - mbi["th0"]) / (mbi["th1"] - mbi["th0"])
        _, ys = G.logo_surface(info, th, 184.5 - mbi["rc"](u), p)
        y_in = -73.0
        c0 = G.P(tp, 184.5, y_in)
        shank = cq.Solid.makeCylinder(2.5, ys - y_in, cq.Vector(*c0), cq.Vector(0, 1, 0))
        head = cq.Workplane("XY").circle(4.6).extrude(2.4).faces(">Z").edges().fillet(1.0).val()
        head = _place_axis(head, (0, 1, 0), G.P(tp, 184.5, ys - 0.2))
        clip = cq.Solid.makeCylinder(3.8, 1.6, cq.Vector(*G.P(tp, 184.5, -71.6)), cq.Vector(0, 1, 0))
        out.append((f"PadPin_{k + 1}", shank.fuse(head).fuse(clip).clean()))
    return out


def mount_bolts(p):
    b = p["body"]
    out = []
    for e in (-1, 1):
        th = e * b["mount_angle_deg"]
        u = (math.sin(math.radians(th)), 0.0, math.cos(math.radians(th)))
        r_seat = b["mount_radial_center"] + b["mount_half_length"] - 0.5
        af = 22.0
        bolt = (cq.Workplane("XY").polygon(6, af / math.cos(math.pi / 6)).extrude(9.0)
                .edges(">Z").chamfer(1.0)
                .faces("<Z").workplane().circle(7.0).extrude(42.0))
        # washer face
        bolt = bolt.union(cq.Workplane("XY").circle(12.0).extrude(1.0))
        out.append((f"MountBolt_M14_{'L' if e < 0 else 'T'}",
                    _place_axis(bolt, u, np.array(G.P(th, r_seat, b["mount_y"])))))
    return out


def bleeders(p):
    D = G.derived(p)
    out = []
    for e in (-1, 1):
        the = D["pad_th1"] if e > 0 else D["pad_th0"]
        base = G.P(the + e * 8.0, 182.0, -70.5)
        bl = (cq.Workplane("XY").polygon(6, 8.0 / math.cos(math.pi / 6)).extrude(4.0)
              .faces(">Z").workplane().circle(3.0).extrude(5.5)
              .faces(">Z").workplane().circle(1.0).cutThruAll())
        cap = cq.Workplane("XY").workplane(offset=6.0).circle(3.9).extrude(4.2).faces(">Z").edges().fillet(1.2)
        bl = bl.union(cap)
        out.append((f"Bleeder_{'L' if e < 0 else 'T'}", _place_axis(bl, (0, -1, 0), base)))
    return out


# --------------------------------------------------------------------------
def rotor(p):
    ro = p["rotor"]
    R = ro["outer_diameter"] / 2
    t = ro["thickness"]
    fr = ro["friction_ring_thickness"]
    ri = ro["inner_ring_diameter"] / 2

    def ring(y0, y1, r_out, r_in):
        return cq.Solid.makeCylinder(r_out, y1 - y0, cq.Vector(0, y0, 0), cq.Vector(0, 1, 0)).cut(
            cq.Solid.makeCylinder(r_in, y1 - y0 + 2, cq.Vector(0, y0 - 1, 0), cq.Vector(0, 1, 0)))

    ob = ring(t / 2 - fr, t / 2, R, ri)
    ib = ring(-t / 2, -t / 2 + fr, R, ri)
    gap = t - 2 * fr
    vanes = []
    for k in range(ro["vane_count"]):
        a = 360.0 * k / ro["vane_count"]
        L = R - ri - 2.0
        v = cq.Solid.makeBox(L, gap + 0.2, 3.6, cq.Vector(ri + 1.0, -gap / 2 - 0.1, -1.8))
        v = v.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), a)
        vanes.append(v)
    disc = ob.fuse(ib, *vanes)
    # cross-drilling pattern (3 rows, curved)
    holes = []
    for k in range(24):
        for j, rr in enumerate((ri + 14, ri + 29, ri + 44)):
            a = math.radians(360.0 * k / 24 + j * 4.5)
            holes.append(cq.Solid.makeCylinder(3.0, t + 4, cq.Vector(rr * math.sin(a), -t / 2 - 2, rr * math.cos(a)),
                                               cq.Vector(0, 1, 0)))
    disc = disc.cut(cq.Compound.makeCompound(holes))
    # hat
    hh = ro["hat_height"]
    hr = ro["hat_outer_diameter"] / 2
    hat = ring(-t / 2, hh, hr, hr - 5.0)
    face = ring(hh - 6.0, hh, hr, ro["hat_bore_diameter"] / 2)
    flange = ring(-t / 2, -t / 2 + 4.0, ri + 3.0, hr - 4.0)
    hat = hat.fuse(face, flange)
    for k in range(ro["bolt_count"]):
        a = math.radians(360.0 * k / ro["bolt_count"])
        rr = ro["bolt_pcd"] / 2
        hat = hat.cut(cq.Solid.makeCylinder(7.25, 20, cq.Vector(rr * math.sin(a), hh - 10, rr * math.cos(a)),
                                            cq.Vector(0, 1, 0)))
    return disc.fuse(hat).clean()


def wheel(p):
    w = p["wheel"]
    Rb = w["barrel_min_inner_diameter"] / 2
    mount_y = p["rotor"]["hat_height"]
    yc = mount_y - w["offset_et"]
    half = w["rim_width_in"] * 25.4 / 2
    bead = w["nominal_size_in"] * 25.4 / 2
    y_in, y_out = yc - half, yc + half
    prof = [(Rb, y_in + 8), (Rb, y_out - 4), (Rb + 4, y_out + 6), (bead + 13, y_out + 12),
            (bead + 15, y_out + 8), (bead, y_out + 1), (bead, y_out - 12), (bead - 15, y_out - 30),
            (bead - 15, y_in + 30), (bead, y_in + 12), (bead, y_in), (bead + 15, y_in - 6),
            (bead + 13, y_in - 10), (Rb + 4, y_in - 2)]
    # revolve the (r, y) profile around the Y axis: sketch on XY plane with x=r, y=y
    wp = cq.Workplane("XY").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
    barrel = wp.val()
    # hub disc
    hub = cq.Solid.makeCylinder(82.0, 16.0, cq.Vector(0, mount_y, 0), cq.Vector(0, 1, 0)).cut(
        cq.Solid.makeCylinder(36.0, 20.0, cq.Vector(0, mount_y - 2, 0), cq.Vector(0, 1, 0)))
    for k in range(5):
        a = math.radians(72.0 * k)
        hub = hub.cut(cq.Solid.makeCylinder(7.0, 20.0, cq.Vector(65 * math.sin(a), mount_y - 2, 65 * math.cos(a)),
                                            cq.Vector(0, 1, 0)))
    sb = w["spoke_back_y"]
    spokes = []
    for k in range(w["spoke_count"]):
        # spoke side profile in the (r, y) plane, then thickened tangentially
        pr = [(70.0, mount_y), (110.0, sb), (Rb + 3.0, sb), (Rb + 3.0, y_out - 4.0), (110.0, y_out - 2.0),
              (78.0, mount_y + 16.0)]
        sp = cq.Workplane("XY").polyline(pr).close().extrude(13.0, both=True).val()
        sp = sp.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), 36.0 + 72.0 * k)
        spokes.append(sp)
    return barrel.fuse(hub, *spokes).clean()
