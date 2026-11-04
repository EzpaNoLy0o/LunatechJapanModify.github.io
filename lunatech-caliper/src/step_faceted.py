"""
step_faceted.py - compact STEP (ISO 10303-21, AP214 schema) writer for
closed triangle meshes as FACETED_BREP solids (POLY_LOOP faces).

Why: the organic caliper body is an implicit-surface model; its exact
B-Rep would be a faceted solid anyway.  OCCT's generic writer stores
~2.5 kB per triangle (planes, edges, p-curves); FACETED_BREP stores
~0.4 kB per triangle.

`merge_step` combines several STEP files / faceted bodies into one file
with several root products (one per part).
"""
import re
import numpy as np


def _fmt(v):
    s = repr(float(v))
    if "e" in s or "E" in s:
        m, e = s.lower().split("e")
        if "." not in m:
            m += "."
        return f"{m}E{int(e)}"
    if "." not in s:
        s += "."
    return s


class StepBuilder:
    def __init__(self):
        self.lines = []
        self.n = 0
        self.ctx = None
        self.app = None

    def add(self, txt):
        self.n += 1
        self.lines.append(f"#{self.n}={txt};")
        return self.n

    def contexts(self):
        if self.ctx:
            return
        self.app = self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
        self.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{self.app})")
        mm = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
        rad = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        sr = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        unc = self.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#{mm},'distance_accuracy_value','confusion accuracy')")
        self.ctx = self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#{rad},#{sr})) REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")

    def product(self, name, items, rep_type):
        self.contexts()
        pc = self.add(f"PRODUCT_CONTEXT('',#{self.app},'mechanical')")
        pr = self.add(f"PRODUCT('{name}','{name}','',(#{pc}))")
        self.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{pr}))")
        pf = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{pr})")
        pdc = self.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{self.app},'design')")
        pd = self.add(f"PRODUCT_DEFINITION('design','',#{pf},#{pdc})")
        pds = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
        o = self.add("CARTESIAN_POINT('',(0.,0.,0.))")
        z = self.add("DIRECTION('',(0.,0.,1.))")
        x = self.add("DIRECTION('',(1.,0.,0.))")
        ax = self.add(f"AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})")
        it = ",".join(f"#{i}" for i in [ax] + list(items))
        rep = self.add(f"{rep_type}('{name}',({it}),#{self.ctx})")
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
        return pr

    def faceted_body(self, name, V, F):
        V = np.asarray(V, float)
        F = np.asarray(F, np.int64)
        base = self.n + 1
        for v in V:
            self.add(f"CARTESIAN_POINT('',({_fmt(v[0])},{_fmt(v[1])},{_fmt(v[2])}))")
        faces = []
        nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
        for (a, b, c), n in zip(F, nrm):
            lp = self.add(f"POLY_LOOP('',(#{base + a},#{base + b},#{base + c}))")
            fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
            dn = self.add(f"DIRECTION('',({_fmt(n[0])},{_fmt(n[1])},{_fmt(n[2])}))")
            ax = self.add(f"AXIS2_PLACEMENT_3D('',#{base + a},#{dn},$)")
            pl = self.add(f"PLANE('',#{ax})")
            faces.append(self.add(f"FACE_SURFACE('',(#{fb}),#{pl},.T.)"))
        sh = self.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
        br = self.add(f"FACETED_BREP('{name}',#{sh})")
        return self.product(name, [br], "FACETED_BREP_SHAPE_REPRESENTATION")

    def text(self, description="Lunatech caliper replica"):
        hdr = ("ISO-10303-21;\nHEADER;\n"
               f"FILE_DESCRIPTION(('{description}'),'2;1');\n"
               "FILE_NAME('lunatech','2026-11-04T00:00:00',('Lunatech'),(''),'lunatech step_faceted.py','python','');\n"
               "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")
        return hdr + "\n".join(self.lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"


def write_faceted(path, bodies, description="Lunatech caliper replica"):
    sb = StepBuilder()
    for name, V, F in bodies:
        sb.faceted_body(name, V, F)
    with open(path, "w") as f:
        f.write(sb.text(description))


_ref = re.compile(r"#(\d+)")


def _data_section(txt):
    a = txt.index("DATA;") + len("DATA;")
    b = txt.rindex("ENDSEC;")
    body = txt[a:b]
    # join multi-line entities
    ents = [e.strip() for e in body.split(";") if e.strip()]
    return ents


def merge_step(path, parts, description="Lunatech caliper assembly"):
    """parts: list of either ('file', step_path) or ('mesh', name, V, F)."""
    out = []
    offset = 0
    for prt in parts:
        if prt[0] == "file":
            with open(prt[1]) as f:
                txt = f.read()
            ents = _data_section(txt)
        else:
            sb = StepBuilder()
            sb.faceted_body(prt[1], prt[2], prt[3])
            ents = [l.rstrip(";") for l in sb.lines]
        mx = 0
        for e in ents:
            e2 = _ref.sub(lambda m: f"#{int(m.group(1)) + offset}", e)
            out.append(e2 + ";")
            m = re.match(r"#(\d+)", e)
            if m:
                mx = max(mx, int(m.group(1)))
        offset += mx
    hdr = ("ISO-10303-21;\nHEADER;\n"
           f"FILE_DESCRIPTION(('{description}'),'2;1');\n"
           "FILE_NAME('lunatech','2026-11-04T00:00:00',('Lunatech'),(''),'lunatech step_faceted.py','python','');\n"
           "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")
    with open(path, "w") as f:
        f.write(hdr + "\n".join(out) + "\nENDSEC;\nEND-ISO-10303-21;\n")


def read_step(path):
    """Read back with OpenCASCADE -> cadquery Shape."""
    import cadquery as cq
    from OCP.STEPControl import STEPControl_Reader
    r = STEPControl_Reader()
    st = r.ReadFile(path)
    r.TransferRoots()
    return cq.Shape.cast(r.OneShape())
