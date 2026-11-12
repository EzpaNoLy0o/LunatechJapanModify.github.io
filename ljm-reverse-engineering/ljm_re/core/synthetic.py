"""真値が既知のテスト用3Dデータ生成.

実機スキャン (POP 4) のデータがなくても、開発・検証を進められるようにする。
すべて mm 単位。
"""
from __future__ import annotations

import numpy as np
import trimesh

from .geometry import normalize, orthonormal_basis, rotation_between, euler_xyz_to_matrix


def make_plane_mesh(size: float = 100.0, divisions: int = 50) -> trimesh.Trimesh:
    """100x100mm の平面 (z=0, 中心=原点)."""
    xs = np.linspace(-size / 2, size / 2, divisions + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    F = []
    n = divisions + 1
    for i in range(divisions):
        for j in range(divisions):
            a = i * n + j
            b = (i + 1) * n + j
            F.append([a, b, b + 1])
            F.append([a, b + 1, a + 1])
    return trimesh.Trimesh(V, np.array(F), process=False)


def make_tube_mesh(diameter: float = 50.0, length: float = 100.0, sections: int = 128,
                   length_divisions: int = 20, arc_deg: float = 360.0, center=(0, 0, 0),
                   axis=(0, 0, 1), inward: bool = False) -> trimesh.Trimesh:
    """円筒面（側面のみ）。arc_deg<360 で部分円筒。inward=True で法線を内向き（穴の内壁）."""
    r = diameter / 2
    full = arc_deg >= 360.0 - 1e-9
    na = sections if full else max(2, int(round(sections * arc_deg / 360.0)) + 1)
    t = np.linspace(0, np.radians(arc_deg), na, endpoint=not full)
    z = np.linspace(-length / 2, length / 2, length_divisions + 1)
    T, Z = np.meshgrid(t, z, indexing="ij")
    V = np.column_stack([r * np.cos(T.ravel()), r * np.sin(T.ravel()), Z.ravel()])
    nz = length_divisions + 1
    F = []
    seg = na if full else na - 1
    for i in range(seg):
        i2 = (i + 1) % na
        for j in range(length_divisions):
            a, b = i * nz + j, i2 * nz + j
            F.append([a, b, b + 1])
            F.append([a, b + 1, a + 1])
    F = np.array(F)
    if inward:
        F = F[:, ::-1]
    m = trimesh.Trimesh(V, F, process=False)
    R = rotation_between([0, 0, 1], axis)
    T4 = np.eye(4)
    T4[:3, :3] = R
    T4[:3, 3] = center
    m.apply_transform(T4)
    return m


def make_plate_with_holes(width=100.0, depth=100.0, thickness=10.0, holes=None,
                          resolution=128) -> trimesh.Trimesh:
    """穴付きプレート（押し出し）。holes = [(x, y, diameter), ...]. 上面 z=thickness, 下面 z=0."""
    from shapely.geometry import Point, Polygon
    if holes is None:
        holes = [(0.0, 0.0, 20.0)]
    outer = Polygon([(-width / 2, -depth / 2), (width / 2, -depth / 2),
                     (width / 2, depth / 2), (-width / 2, depth / 2)])
    rings = []
    for (x, y, d) in holes:
        c = Point(x, y).buffer(d / 2, quad_segs=resolution // 4)
        rings.append(list(c.exterior.coords)[::-1])
    poly = Polygon(outer.exterior.coords, rings)
    m = trimesh.creation.extrude_polygon(poly, thickness)
    return m


def make_bracket():
    """複数穴ブラケット: ベースプレート(4穴) + 立ち上がりフランジ(1穴).

    戻り値 (mesh, truth)。truth に穴の中心・径・軸を記録。
    """
    t = 8.0
    base_holes = [(-55.0, -25.0, 10.0), (55.0, -25.0, 10.0), (-55.0, 25.0, 12.0), (30.0, 25.0, 16.0)]
    base = make_plate_with_holes(150.0, 80.0, t, base_holes)
    # フランジ: ローカルで 80x60x8 の板（穴はローカル原点, 径14）を
    # X軸周り +90° 回転 [(x,y,z)→(x,-z,y)] し、y=40〜48, z=8〜68 に配置する。
    fl_d = 14.0
    fl = make_plate_with_holes(80.0, 60.0, t, [(0.0, 0.0, fl_d)])
    T = np.eye(4)
    T[:3, :3] = euler_xyz_to_matrix(90, 0, 0)
    T[:3, 3] = [0.0, 40.0 + t, 30.0 + t]
    fl.apply_transform(T)
    mesh = trimesh.util.concatenate([base, fl])
    truth = {"holes": []}
    for x, y, d in base_holes:
        truth["holes"].append({"center_top": [x, y, t], "center_bottom": [x, y, 0.0],
                               "diameter": d, "axis": [0, 0, 1]})
    # フランジ穴中心: ローカル (0,0,0) と (0,0,t) → ワールド
    c_world = trimesh.transform_points(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, t]]), T)
    truth["holes"].append({"center_a": c_world[0].tolist(), "center_b": c_world[1].tolist(),
                           "diameter": fl_d, "axis": [0, 1, 0]})
    truth["base_top_plane"] = {"origin": [0, 0, t], "normal": [0, 0, 1]}
    return mesh, truth


def make_multi_cylinder_part():
    """複数の円筒からなる部品（段付きシャフト + 直交ボス）. (mesh, truth)."""
    parts = []
    truth = []
    specs = [  # (diameter, length, center, axis)
        (40.0, 50.0, (0, 0, 25.0), (0, 0, 1)),
        (25.0, 40.0, (0, 0, 70.0), (0, 0, 1)),
        (60.0, 10.0, (0, 0, -5.0), (0, 0, 1)),
        (20.0, 30.0, (35.0, 0, 25.0), (1, 0, 0)),
    ]
    for d, L, c, a in specs:
        parts.append(make_tube_mesh(d, L, 96, 10, center=c, axis=a))
        truth.append({"diameter": d, "length": L, "center": list(c), "axis": list(a)})
    return trimesh.util.concatenate(parts), truth


def sample_surface(mesh: trimesh.Trimesh, n: int, noise_sigma: float = 0.0, seed: int = 0):
    """メッシュ表面から点をサンプリング（法線付き）。ノイズは法線方向のガウス."""
    rng = np.random.default_rng(seed)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
    nrm = mesh.face_normals[fid]
    if noise_sigma > 0:
        pts = pts + nrm * rng.normal(0, noise_sigma, size=(len(pts), 1))
    return np.asarray(pts), np.asarray(nrm)


def partial_cylinder_points(diameter=50.0, length=60.0, arc_deg=120.0, n=5000, noise_sigma=0.0,
                            center=(0, 0, 0), axis=(0, 0, 1), seed=0):
    rng = np.random.default_rng(seed)
    r = diameter / 2
    t = rng.uniform(0, np.radians(arc_deg), n)
    z = rng.uniform(-length / 2, length / 2, n)
    nl = np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])
    p = nl * r
    p[:, 2] = z
    if noise_sigma > 0:
        p = p + nl * rng.normal(0, noise_sigma, (n, 1))
    R = rotation_between([0, 0, 1], axis)
    return p @ R.T + np.asarray(center, float), nl @ R.T


def plane_points(n=5000, size=100.0, noise_sigma=0.0, origin=(0, 0, 0), normal=(0, 0, 1),
                 outlier_ratio=0.0, seed=0):
    rng = np.random.default_rng(seed)
    nrm = normalize(np.asarray(normal, float))
    u, v = orthonormal_basis(nrm)
    a = rng.uniform(-size / 2, size / 2, (n, 2))
    p = np.asarray(origin, float) + np.outer(a[:, 0], u) + np.outer(a[:, 1], v)
    if noise_sigma > 0:
        p = p + np.outer(rng.normal(0, noise_sigma, n), nrm)
    k = int(n * outlier_ratio)
    if k > 0:
        idx = rng.choice(n, k, replace=False)
        p[idx] += np.outer(rng.uniform(2, 20, k), nrm)
    return p


def registration_pair(mesh: trimesh.Trimesh, n=8000, noise_sigma=0.0, seed=0,
                      rot_deg=(10.0, -5.0, 20.0), trans=(15.0, -8.0, 4.0)):
    """位置と回転を変えた点群ペア. 戻り値 (source, target, T_true) で target ≈ T_true · source."""
    src, _ = sample_surface(mesh, n, 0.0, seed)
    R = euler_xyz_to_matrix(*rot_deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    tgt, _ = sample_surface(mesh, n, noise_sigma, seed + 1)
    tgt = tgt @ R.T + np.asarray(trans)
    return src, tgt, T
