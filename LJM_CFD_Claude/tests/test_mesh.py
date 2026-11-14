"""メッシュ: polyMesh の組み立て・品質指標・Gmsh による実際の風洞メッシュ生成。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from ljmcfd.cad.assembly import Assembly
from ljmcfd.cad.import_job import import_cad_job
from ljmcfd.core.jobs import JobError, RecordingContext
from ljmcfd.core.pose import Pose
from ljmcfd.mesh import polymesh as pm
from ljmcfd.mesh.gmsh_job import load_report, mesh_generate_job, patch_name_for, similarity_from_matrix
from ljmcfd.mesh.quality import cell_geometry, compute_quality, face_geometry
from ljmcfd.mesh.request import build_mesh_request
from ljmcfd.mesh.settings import BoundaryLayerSettings, MeshSettings, input_hash
from ljmcfd.tunnel import spec as ts

CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


def _all_boundary(mesh_points, cells):
    """境界パッチを指定しない (全面が defaultFaces になる) 簡易組み立て。"""
    return pm.build_polymesh(mesh_points, cells, [])


# ----------------------------------------------------------------------------- polyMesh

def test_single_cells_are_closed_with_outward_faces():
    shapes = {
        pm.TET: (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float), [[0, 1, 2, 3]], 1 / 6),
        pm.HEX: (CUBE, [list(range(8))], 1.0),
        pm.PRISM: (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=float), [list(range(6))], 1.0),
        pm.PYRAMID: (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 3]], dtype=float), [list(range(5))], 1.0),
    }
    for kind, (points, cells, volume) in shapes.items():
        for cell in (cells, [pm._FLIP[kind]]):          # 正の向き / 裏返しの節点順のどちらでも
            mesh = _all_boundary(points, {kind: np.array(cell)})
            fc, fa = face_geometry(mesh)
            cc, vol, openness = cell_geometry(mesh, fc, fa)
            assert vol[0] == pytest.approx(volume), kind
            assert openness[0] < 1e-12, kind
            assert (np.einsum("ij,ij->i", fa, fc - cc[0]) > 0).all(), kind     # 全面が外向き
            assert mesh.n_internal == 0 and mesh.n_unmatched_faces == mesh.n_faces


def test_two_tets_share_one_internal_face_and_patches_are_assigned():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    cells = {pm.TET: np.array([[0, 1, 2, 3], [0, 2, 1, 4]])}
    groups = [
        pm.BoundaryGroup("upper", "patch", tris=np.array([[0, 1, 3], [1, 2, 3], [0, 3, 2]])),
        pm.BoundaryGroup("lower", "wall", tris=np.array([[4, 1, 0], [2, 1, 4], [0, 2, 4]]), info={"kind": "part"}),
    ]
    mesh = pm.build_polymesh(points, cells, groups)
    assert mesh.n_cells == 2 and mesh.n_faces == 7 and mesh.n_internal == 1
    assert mesh.owner[0] == 0 and mesh.neighbour[0] == 1
    assert sorted(mesh.faces[0, :3]) == [0, 1, 2]
    # 内部面の法線は owner (上の四面体) → neighbour (下) の向き = -Z
    _, fa = face_geometry(mesh)
    assert fa[0] == pytest.approx([0, 0, -0.5])
    assert [(p["name"], p["type"], p["start"], p["size"]) for p in mesh.patches] == [("upper", "patch", 1, 3), ("lower", "wall", 4, 3)]
    assert mesh.n_unmatched_faces == 0 and mesh.n_orphan_boundary == 0
    assert (mesh.owner[1:4] == 0).all() and (mesh.owner[4:7] == 1).all()
    report = compute_quality(mesh)
    assert report["summary"]["level"] == "ok" and report["volume_total"] == pytest.approx(1 / 3)
    assert report["patch_areas"]["upper"] == pytest.approx(0.5 + 0.5 + np.sqrt(3) / 2)


def test_internal_faces_follow_openfoam_ordering():
    """六面体の柱と、プリズム 2 個に分けた柱を市松状に並べた混合メッシュで順序規約を確認する。"""
    nx, ny, nz = 5, 4, 3
    xs, ys, zs = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0), np.arange(nz + 1.0), indexing="ij")
    points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    node = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    prisms, hexes = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                     node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)]
                if (i + j) % 2:      # 柱ごとに種類を変える (上下の隣は同じ種類なので面が整合する)
                    hexes.append(c)
                else:
                    prisms.append([c[0], c[1], c[2], c[4], c[5], c[6]])
                    prisms.append([c[0], c[2], c[3], c[4], c[6], c[7]])
    mesh = _all_boundary(points, {pm.PRISM: np.array(prisms), pm.HEX: np.array(hexes)})
    assert mesh.n_cells == len(prisms) + len(hexes)
    own, nei = mesh.owner[:mesh.n_internal], mesh.neighbour
    assert (own < nei).all()                                             # 上三角
    keys = own * mesh.n_cells + nei
    assert (np.diff(keys) > 0).all()                                     # owner 昇順、同一 owner 内で neighbour 昇順
    assert (mesh.owner[mesh.n_internal:] >= 0).all()
    report = compute_quality(mesh)
    assert report["volume_total"] == pytest.approx(nx * ny * nz)
    assert report["n_open_cells"] == 0 and report["n_negative_volumes"] == 0
    assert report["boundary_closure"] < 1e-12
    assert report["patch_areas"][pm.PATCH_DEFAULT] == pytest.approx(2 * (nx * ny + ny * nz + nz * nx))
    # 内部面の法線は必ず owner → neighbour 向き
    fc, fa = face_geometry(mesh)
    cc, _, _ = cell_geometry(mesh, fc, fa)
    assert (np.einsum("ij,ij->i", fa[:mesh.n_internal], cc[nei] - cc[own]) > 0).all()


def test_quality_metrics_on_known_geometry():
    # 直交格子の六面体 2 個 → 非直交性 0°、歪度 0
    points = np.vstack([CUBE, CUBE[[1, 2, 5, 6]] + [1, 0, 0]])
    hexes = np.array([list(range(8)), [1, 8, 9, 2, 5, 10, 11, 6]])
    report = compute_quality(_all_boundary(points, {pm.HEX: hexes}))
    assert report["non_ortho_max"] == pytest.approx(0.0, abs=1e-6)
    assert report["skewness_max"] == pytest.approx(0.0, abs=1e-9)
    assert report["volume_ratio_max"] == pytest.approx(1.0)
    # 2 個目を X 方向へせん断すると、共有面の法線とセル中心を結ぶ線がずれる
    sheared = points.copy()
    sheared[[8, 9, 10, 11], 1] += 1.0
    report = compute_quality(_all_boundary(sheared, {pm.HEX: hexes}))
    assert report["non_ortho_max"] == pytest.approx(np.degrees(np.arctan(0.5)), abs=1e-6)
    assert report["skewness_max"] > 0.1
    # 同じ面の同じ側に 2 個のセルが重なっている不正なメッシュ → 片方が裏返って見え、エラーと判定される
    overlapping = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.5]], dtype=float)
    report = compute_quality(_all_boundary(overlapping, {pm.TET: np.array([[0, 1, 2, 3], [0, 1, 2, 4]])}))
    assert report["summary"]["level"] == "error"
    assert report["n_negative_volumes"] + report["n_open_cells"] > 0


def test_polymesh_write_and_read_roundtrip(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float) * 0.123456789012
    cells = {pm.TET: np.array([[0, 1, 2, 3], [0, 2, 1, 4]])}
    groups = [pm.BoundaryGroup("inlet", "patch", tris=np.array([[0, 1, 3], [1, 2, 3], [0, 3, 2]])),
              pm.BoundaryGroup("Body", "wall", tris=np.array([[4, 1, 0], [2, 1, 4], [0, 2, 4]]))]
    mesh = pm.build_polymesh(points, cells, groups)
    folder = pm.write_polymesh(mesh, tmp_path / "polyMesh")
    assert {p.name for p in folder.iterdir()} == {"points", "faces", "owner", "neighbour", "boundary"}
    text = (folder / "owner").read_text()
    assert 'note        "nPoints: 5 nCells: 2 nFaces: 7 nInternalFaces: 1";' in text and "class       labelList;" in text
    assert "inGroups        1(wall);" in (folder / "boundary").read_text()
    again = pm.read_polymesh(folder)
    assert np.allclose(again.points, mesh.points, rtol=1e-11)
    assert (again.faces == mesh.faces).all() and (again.owner == mesh.owner).all() and (again.neighbour == mesh.neighbour).all()
    assert [(p["name"], p["type"], p["start"], p["size"]) for p in again.patches] == \
           [(p["name"], p["type"], p["start"], p["size"]) for p in mesh.patches]
    # 四角形面を含む場合の書式
    hexmesh = _all_boundary(CUBE, {pm.HEX: np.array([list(range(8))])})
    pm.write_polymesh(hexmesh, tmp_path / "hex")
    assert "4(0 3 2 1)" in (tmp_path / "hex" / "faces").read_text()
    assert (pm.read_polymesh(tmp_path / "hex").faces == hexmesh.faces).all()


# ----------------------------------------------------------------------------- 補助関数

def test_patch_names_are_openfoam_safe_and_unique():
    used: set[str] = set()
    assert patch_name_for("Rear Wing (L)", 0, used) == "Rear_Wing__L"
    assert patch_name_for("Rear Wing (L)", 1, used) == "Rear_Wing__L_2"
    assert patch_name_for("ボディ", 2, used) == "part2"
    assert patch_name_for("3rd_element", 3, used) == "part3_3rd_element"
    assert patch_name_for("inlet", 4, used) == "inlet_2"            # 風洞のパッチ名と衝突させない


def test_similarity_decomposition():
    from ljmcfd.core.pose import pose_matrix

    matrix = pose_matrix(Pose(tx=1.0, ty=-2.0, tz=0.5, yaw=30.0, pitch=-5.0, roll=10.0), pivot=(0.4, 0.1, 0.2), scale=0.001)
    scale, rotvec, translation = similarity_from_matrix(matrix)
    from scipy.spatial.transform import Rotation

    rebuilt = np.eye(4)
    rebuilt[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() * scale
    rebuilt[:3, 3] = translation
    assert scale == pytest.approx(0.001) and np.allclose(rebuilt, matrix, atol=1e-12)
    with pytest.raises(JobError):
        similarity_from_matrix(np.diag([1.0, 2.0, 1.0, 1.0]))
    with pytest.raises(JobError):
        similarity_from_matrix(np.diag([1.0, 1.0, -1.0, 1.0]))


def test_settings_presets_validation_and_hash():
    settings = MeshSettings()
    settings.apply_preset("standard", (0.0, 4.5, -0.9, 0.9, 0.0, 1.2))
    assert settings.surface_size == pytest.approx(0.05) and settings.min_size == pytest.approx(0.01)
    assert settings.far_size == pytest.approx(1.5) and not settings.validate()
    coarse = MeshSettings()
    coarse.apply_preset("coarse", (0.0, 4.5, -0.9, 0.9, 0.0, 1.2))
    assert coarse.surface_size > settings.surface_size
    settings.part_size_factors = {"abc": 0.5}
    settings.boundary_layer = BoundaryLayerSettings(enabled=True, layers=3, first_height=0.001, growth_ratio=1.2)
    assert settings.boundary_layer.total_thickness() == pytest.approx(0.001 * (1 + 1.2 + 1.44))
    assert settings.boundary_layer.layer_heights() == pytest.approx([0.001, 0.0022, 0.00364])
    again = MeshSettings.from_dict(json.loads(json.dumps(settings.to_dict())))
    assert again == settings
    settings.min_size = 1.0
    assert settings.validate()
    assert input_hash({"a": 1.0, "b": [1, 2]}) == input_hash({"b": [1, 2], "a": 1.0000000000001})
    assert input_hash({"a": 1.0}) != input_hash({"a": 1.001})


# ----------------------------------------------------------------------------- Gmsh による実メッシュ

def _scene(path, cache_root, mode=ts.MODE_VEHICLE, lift=None):
    meta = import_cad_job({"path": str(path), "cache_root": str(cache_root)})
    assembly = Assembly()
    assembly.add_from_meta(meta)
    assembly.center_xy()
    if lift is not None:
        assembly.drop_to_ground(lift)
    spec = ts.TunnelSpec(mode=mode)
    ts.apply_mode_defaults(spec, assembly.world_bounds())
    settings = MeshSettings()
    settings.apply_preset("coarse", assembly.world_bounds())
    return assembly, spec, settings


def _run(assembly, spec, settings, out_dir):
    ctx = RecordingContext()
    report = mesh_generate_job(build_mesh_request(assembly, spec, settings, out_dir), ctx)
    return report, ctx


def test_vehicle_tunnel_mesh_of_grounded_box(box_step, cache_root, tmp_path):
    """4.0 x 1.8 x 1.2 m の直方体を接地させた車両風洞。体積・パッチ面積を解析解と照合する。"""
    assembly, spec, settings = _scene(box_step, cache_root)
    assert spec.model_sink(assembly.world_bounds()) == pytest.approx(0.010)
    out = tmp_path / "mesh"
    report, ctx = _run(assembly, spec, settings, out)

    quality = report["quality"]
    assert quality["summary"]["level"] in ("ok", "warning")
    assert quality["n_negative_volumes"] == 0 and quality["n_open_cells"] == 0 and quality["n_unmatched_faces"] == 0
    assert quality["cell_counts"] == {"tet": quality["n_cells"]} and 2_000 < quality["n_cells"] < 400_000
    # 流体体積 = 風洞 − 直方体 (10 mm 沈めた分だけ直方体が低い)
    d = report["domain"]
    tunnel_volume = (d[1] - d[0]) * (d[3] - d[2]) * (d[5] - d[4])
    box_volume = 4.0 * 1.8 * (1.2 - 0.010)
    assert quality["volume_total"] == pytest.approx(tunnel_volume - box_volume, rel=1e-9)
    assert report["volume_error"] < 1e-9 and report["sink"] == pytest.approx(0.010)

    names = [p["name"] for p in report["patches"]]
    assert names == ["inlet", "outlet", "sideL", "sideR", "ground", "top", "box_mm"]
    types = {p["name"]: p["type"] for p in report["patches"]}
    assert types["ground"] == "wall" and types["box_mm"] == "wall" and types["inlet"] == "patch"
    areas = quality["patch_areas"]
    cross = (d[3] - d[2]) * (d[5] - d[4])
    assert areas["inlet"] == pytest.approx(cross) and areas["outlet"] == pytest.approx(cross)
    assert areas["ground"] == pytest.approx((d[1] - d[0]) * (d[3] - d[2]) - 4.0 * 1.8)      # 接地面は地面から除かれる
    assert areas["box_mm"] == pytest.approx(4.0 * 1.8 + 2 * (4.0 + 1.8) * 1.19)            # 上面 + 側面 (底面なし)
    assert report["patches"][-1]["part_id"] == assembly.parts[0].id
    assert report["boundary_layer"] == {**report["boundary_layer"], "requested": False, "generated": False, "n_prisms": 0}

    # 出力ファイル: polyMesh は読み戻して同じ品質、VTK はセル数が一致
    assert not out.with_name("mesh.partial").exists()
    mesh = pm.read_polymesh(out / "polyMesh")
    assert mesh.n_cells == quality["n_cells"] and compute_quality(mesh)["non_ortho_max"] == pytest.approx(quality["non_ortho_max"])
    assert pv.read(out / "volume.vtu").n_cells == quality["n_cells"]
    boundary = pv.read(out / "boundary.vtp")
    assert boundary.n_cells == quality["n_boundary_faces"]
    assert sorted(np.unique(boundary.cell_data["patch"])) == list(range(7))
    assert load_report(out)["input_hash"] == report["input_hash"]
    assert any("メッシュ生成完了" in line for line in ctx.logs)
    # 表面近傍が細かく、遠方が粗い
    body = boundary.extract_cells(np.flatnonzero(boundary.cell_data["patch"] == 6))
    top = boundary.extract_cells(np.flatnonzero(boundary.cell_data["patch"] == 5))
    mean_edge = lambda m: float(np.sqrt(m.compute_cell_sizes()["Area"].mean()))
    assert mean_edge(top) > 3 * mean_edge(body)


def test_parts_get_their_own_patches_and_local_refinement(assembly_step, cache_root, tmp_path):
    assembly, spec, settings = _scene(assembly_step, cache_root)
    body, wing = assembly.parts
    report, _ = _run(assembly, spec, settings, tmp_path / "a")
    parts = {p["part_name"]: p for p in report["patches"] if p.get("kind") == "part"}
    assert set(parts) == {"Body", "RearWing"} and parts["RearWing"]["part_id"] == wing.id
    areas = report["quality"]["patch_areas"]
    assert areas["RearWing"] == pytest.approx(2 * (0.3 * 1.6 + 0.3 * 0.03 + 1.6 * 0.03), rel=1e-6)   # 浮いたウイングの全表面
    assert areas["Body"] == pytest.approx(4.0 * 1.8 + 2 * (4.0 + 1.8) * 1.19, rel=1e-6)
    # ウイングだけ表面サイズを 1/3 にすると、ウイングの面数が増える
    settings.part_size_factors = {wing.id: 1 / 3}
    refined, _ = _run(assembly, spec, settings, tmp_path / "b")
    faces = lambda rep, name: next(p["n_faces"] for p in rep["patches"] if p.get("part_name") == name)
    assert faces(refined, "RearWing") > 3 * faces(report, "RearWing")
    assert faces(refined, "Body") < 1.5 * faces(report, "Body")
    assert refined["input_hash"] != report["input_hash"]
    # 非表示パーツは解析から除外される
    wing.visible = False
    hidden, _ = _run(assembly, spec, settings, tmp_path / "c")
    assert [p["part_name"] for p in hidden["patches"] if p.get("kind") == "part"] == ["Body"]


def test_component_tunnel_with_attitude(box_step, cache_root, tmp_path):
    """単体パーツ風洞 (地面なし) に、ヨー・ピッチを付けた直方体を置く。体積は姿勢によらず保存される。"""
    assembly, spec, settings = _scene(box_step, cache_root, mode=ts.MODE_COMPONENT)
    assembly.pose = Pose(yaw=12.0, pitch=-6.0, tz=0.4)
    ts.apply_mode_defaults(spec, assembly.world_bounds())
    report, _ = _run(assembly, spec, settings, tmp_path / "mesh")
    d = report["domain"]
    assert [p["name"] for p in report["patches"]][4] == "bottom" and report["sink"] == 0.0
    assert {p["name"]: p["type"] for p in report["patches"]}["bottom"] == "patch"
    volume = (d[1] - d[0]) * (d[3] - d[2]) * (d[5] - d[4]) - 4.0 * 1.8 * 1.2
    assert report["quality"]["volume_total"] == pytest.approx(volume, rel=1e-9)
    assert report["quality"]["patch_areas"]["box_mm"] == pytest.approx(2 * (4 * 1.8 + 4 * 1.2 + 1.8 * 1.2), rel=1e-9)


def test_boundary_layer_prisms_on_floating_body(box_step, cache_root, tmp_path):
    assembly, spec, settings = _scene(box_step, cache_root, lift=0.3)
    settings.boundary_layer = BoundaryLayerSettings(enabled=True, layers=3, first_height=0.004, growth_ratio=1.3)
    report, _ = _run(assembly, spec, settings, tmp_path / "mesh")
    bl = report["boundary_layer"]
    wall_faces = next(p["n_faces"] for p in report["patches"] if p.get("kind") == "part")
    assert bl["generated"] and bl["layers"] == 3 and bl["n_prisms"] == 3 * wall_faces
    assert bl["total_thickness"] == pytest.approx(0.004 * (1 + 1.3 + 1.69))
    counts = report["quality"]["cell_counts"]
    assert counts["prism"] == bl["n_prisms"] and counts["tet"] > 0
    assert report["quality"]["n_negative_volumes"] == 0 and report["quality"]["n_open_cells"] == 0
    assert report["algorithm"] == "delaunay"
    assert report["volume_error"] < 1e-6
    # 壁面に接するセルはすべてプリズム (壁面の面はすべて四角形ではなく三角形 = プリズムの底面)
    mesh = pm.read_polymesh(tmp_path / "mesh" / "polyMesh")
    wall = mesh.patch("box_mm")
    assert (mesh.faces[wall["start"]:wall["start"] + wall["size"], 3] < 0).all()
    n_tet = counts["tet"]
    assert (mesh.owner[wall["start"]:wall["start"] + wall["size"]] >= n_tet).all()   # セル番号は四面体 → プリズムの順


def test_boundary_layer_is_refused_when_touching_ground(box_step, cache_root, tmp_path):
    assembly, spec, settings = _scene(box_step, cache_root)
    settings.boundary_layer.enabled = True
    with pytest.raises(JobError, match="接して"):
        _run(assembly, spec, settings, tmp_path / "mesh")
    assert not (tmp_path / "mesh").exists() and not (tmp_path / "mesh.partial").exists()


def test_unsupported_geometry_is_reported(cube_stl, open_box_step, cache_root, tmp_path):
    assembly, spec, settings = _scene(cube_stl, cache_root)
    with pytest.raises(JobError, match="STL"):
        _run(assembly, spec, settings, tmp_path / "stl")
    assembly, spec, settings = _scene(open_box_step, cache_root)
    with pytest.raises(JobError, match="閉じたソリッドではない"):
        _run(assembly, spec, settings, tmp_path / "open")


def test_failed_run_keeps_previous_mesh(box_step, cache_root, tmp_path):
    assembly, spec, settings = _scene(box_step, cache_root)
    out = tmp_path / "mesh"
    first, _ = _run(assembly, spec, settings, out)
    spec.margin_front = -100.0                       # 流入面がモデルの後ろへ回り込む不正な風洞
    with pytest.raises(JobError):
        _run(assembly, spec, settings, out)
    assert load_report(out)["input_hash"] == first["input_hash"]
    assert load_report(tmp_path / "nothing") is None


def test_inward_boundary_layer_is_detected(box_step, cache_root, tmp_path, monkeypatch):
    """押し出し方向の判定が誤っても、体積の検証で「内側へ押し出された」ことを検出して失敗させる。"""
    from ljmcfd.mesh import gmsh_job

    original = gmsh_job._normals_point_into_fluid
    monkeypatch.setattr(gmsh_job, "_normals_point_into_fluid",
                        lambda *a: [not d for d in original(*a)])
    assembly, spec, settings = _scene(box_step, cache_root, lift=0.3)
    settings.boundary_layer = BoundaryLayerSettings(enabled=True, layers=3, first_height=0.004, growth_ratio=1.3)
    with pytest.raises(JobError, match="内側へ押し出されました"):
        _run(assembly, spec, settings, tmp_path / "mesh")
    assert not (tmp_path / "mesh").exists()
