import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from ljmcfd.cad import unit_detect
from ljmcfd.cad.assembly import Assembly
from ljmcfd.cad.import_job import import_cad_job
from ljmcfd.cad.surface import FACE_ID
from ljmcfd.core.jobs import JobError, RecordingContext, job_command, job_environment, parse_protocol_line
from ljmcfd.core.pose import Pose


def _import(path, cache_root, **extra):
    ctx = RecordingContext()
    meta = import_cad_job({"path": str(path), "cache_root": str(cache_root), **extra}, ctx)
    return meta, ctx


def _extent(bounds):
    return (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])


def test_step_box_import(box_step, cache_root):
    meta, ctx = _import(box_step, cache_root)
    assert meta["format"] == "step" and meta["from_cache"] is False
    assert meta["file_unit"] == "mm" and meta["geometry_unit"] == "mm" and meta["unit_source"] == "file"
    assert len(meta["parts"]) == 1
    part = meta["parts"][0]
    assert part["name"] == "box_mm" and part["n_faces"] == 6 and part["n_triangles"] == 12
    assert _extent(part["bounds"]) == pytest.approx((4000, 1800, 1200), abs=1e-3)
    assert part["face_types"] == ["plane"] * 6
    cache_dir = Path(meta["cache_dir"])
    surface = pv.read(cache_dir / part["surface"])
    assert sorted(np.unique(surface.cell_data[FACE_ID])) == [0, 1, 2, 3, 4, 5]
    assert "Normals" in surface.point_data
    # 法線が外向き (面ごとの三角形の向きが B-Rep の向きに従っている)
    centers = surface.cell_centers().points - np.array([2000, 900, 600])
    normals = surface.compute_normals(cell_normals=True, point_normals=False).cell_data["Normals"]
    assert (np.einsum("ij,ij->i", centers, normals) > 0).all()
    edges = pv.read(cache_dir / part["edges"])
    assert edges.n_cells == 24  # 6 面 x 4 辺 (面ごとに独立)
    assert (cache_dir / part["brep"]).stat().st_size > 0
    assert ctx.progress_events and ctx.progress_events[-1][0] == 1.0


def test_step_in_metres_is_converted_to_mm(box_step_metre, cache_root):
    meta, _ = _import(box_step_metre, cache_root)
    assert meta["file_unit"] == "m"           # 元ファイルはメートル
    assert meta["geometry_unit"] == "mm"      # OpenCASCADE が mm へ換算
    assert _extent(meta["parts"][0]["bounds"]) == pytest.approx((4000, 1800, 1200), abs=1e-3)
    assembly = Assembly()
    part = assembly.add_from_meta(meta)[0]
    assert _extent(assembly.world_bounds()) == pytest.approx((4.0, 1.8, 1.2), abs=1e-6)
    assert part.scale == pytest.approx(1e-3)


def test_step_assembly_names_colors_locations(assembly_step, cache_root):
    meta, _ = _import(assembly_step, cache_root)
    parts = {p["name"]: p for p in meta["parts"]}
    assert set(parts) == {"Body", "RearWing"}
    assert parts["RearWing"]["color"] == pytest.approx([1.0, 0.0, 0.0])
    # インスタンスの配置 (3700, 100, 1300) が適用されている
    assert parts["RearWing"]["bounds"] == pytest.approx([3700, 4000, 100, 1700, 1300, 1330], abs=1e-3)
    assert _extent(parts["Body"]["bounds"]) == pytest.approx((4000, 1800, 1200), abs=1e-3)


def test_multi_solid_compound_is_split(multi_solid_step, cache_root):
    meta, _ = _import(multi_solid_step, cache_root)
    # 書き出しソフトの既定名は捨て、ファイル名から重複しない名前を付ける
    assert [p["name"] for p in meta["parts"]] == ["two_solids", "two_solids (2)"]
    meta_merged, ctx = _import(multi_solid_step, cache_root, max_parts=1, force=True)
    assert len(meta_merged["parts"]) == 1 and meta_merged["parts"][0]["n_faces"] == 12
    assert any("統合" in line for line in ctx.logs)


def test_iges_import(cylinder_iges, cache_root):
    meta, _ = _import(cylinder_iges, cache_root)
    assert meta["format"] == "iges" and meta["file_unit"] == "mm"
    part = meta["parts"][0]
    assert _extent(part["bounds"]) == pytest.approx((100, 100, 200), abs=0.5)
    assert "plane" in part["face_types"] and len(part["face_types"]) == 3


def test_stl_import_and_unit_heuristic(cube_stl, sphere_stl_mm, cache_root):
    meta, _ = _import(cube_stl, cache_root)
    assert meta["format"] == "stl" and meta["file_unit"] is None
    assert meta["geometry_unit"] == "m" and meta["unit_source"] == "heuristic"
    part = meta["parts"][0]
    assert part["name"] == "cube" and part["brep"] is None
    assert part["n_triangles"] == 12 and part["n_faces"] == 6  # 折れ角で 6 面に分割
    assert _extent(part["bounds"]) == pytest.approx((2.0, 1.0, 0.5))
    meta_mm, _ = _import(sphere_stl_mm, cache_root)
    assert meta_mm["geometry_unit"] == "mm"
    assert meta_mm["parts"][0]["n_faces"] == 1  # 滑らかな球は 1 面
    assembly = Assembly()
    assembly.add_from_meta(meta_mm)
    assert _extent(assembly.world_bounds()) == pytest.approx((1.0, 1.0, 1.0), abs=5e-3)


def test_ascii_multi_solid_stl(tmp_path, cache_root):
    def facet(a, b, c):
        return ("facet normal 0 0 1\n outer loop\n" + "".join(f"  vertex {x} {y} {z}\n" for x, y, z in (a, b, c))
                + " endloop\nendfacet\n")
    text = ("solid wing\n" + facet((0, 0, 0), (1, 0, 0), (0, 1, 0)) + "endsolid wing\n"
            "solid flap\n" + facet((5, 0, 0), (6, 0, 0), (5, 1, 0)) + "endsolid flap\n")
    path = tmp_path / "two.stl"
    path.write_text(text)
    meta, _ = _import(path, cache_root)
    assert [p["name"] for p in meta["parts"]] == ["wing", "flap"]
    assert meta["parts"][1]["bounds"][0] == pytest.approx(5.0)


def test_cache_hit_and_invalidation(box_step, cache_root, tmp_path):
    src = tmp_path / "copy.step"
    shutil.copyfile(box_step, src)
    first, _ = _import(src, cache_root)
    second, ctx = _import(src, cache_root)
    assert second["from_cache"] is True and second["key"] == first["key"]
    assert any("キャッシュ" in line for line in ctx.logs)
    # 元ファイルが更新されるとキーが変わり再変換される
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 5_000_000_000))
    third, _ = _import(src, cache_root)
    assert third["from_cache"] is False and third["key"] != first["key"]
    # キャッシュ内のファイルが欠けていれば再変換
    (Path(third["cache_dir"]) / third["parts"][0]["surface"]).unlink()
    fourth, _ = _import(src, cache_root)
    assert fourth["from_cache"] is False


def test_non_ascii_source_path(box_step, cache_root, tmp_path):
    folder = tmp_path / "車両データ"
    folder.mkdir()
    src = folder / "ボディ.step"
    shutil.copyfile(box_step, src)
    meta, _ = _import(src, cache_root)
    assert meta["parts"][0]["name"] == "ボディ"
    assert not list(Path(meta["cache_dir"]).glob("source.*"))  # 一時コピーは残さない


def test_import_errors(tmp_path, cache_root):
    with pytest.raises(JobError, match="見つかりません"):
        _import(tmp_path / "missing.step", cache_root)
    bad = tmp_path / "model.obj"
    bad.write_text("x")
    with pytest.raises(JobError, match="未対応"):
        _import(bad, cache_root)
    garbage = tmp_path / "garbage.step"
    garbage.write_text("this is not a STEP file")
    with pytest.raises(JobError):
        _import(garbage, cache_root)
    assert not list(cache_root.glob("*.partial"))  # 失敗時に中途半端なキャッシュを残さない


def test_unit_detection_variants(tmp_path, box_step, box_step_metre, cylinder_iges):
    assert unit_detect.detect_step_unit(box_step) == "mm"
    assert unit_detect.detect_step_unit(box_step_metre) == "m"
    inch = tmp_path / "inch.step"
    inch.write_text(
        "#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n"
        "#12=(CONVERSION_BASED_UNIT('INCH',#11)LENGTH_UNIT()NAMED_UNIT(#9));\n"
    )
    assert unit_detect.detect_step_unit(inch) == "inch"
    nothing = tmp_path / "none.step"
    nothing.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
    assert unit_detect.detect_step_unit(nothing) is None
    assert unit_detect.detect_iges_unit(cylinder_iges) == "mm"
    assert unit_detect.detect_iges_unit(nothing) is None


def test_assembly_transforms(assembly_step, cache_root):
    meta, _ = _import(assembly_step, cache_root)
    assembly = Assembly()
    body, wing = assembly.add_from_meta(meta)
    assert wing.color == pytest.approx((1.0, 0.0, 0.0))
    assert assembly.world_bounds() == pytest.approx((0, 4.0, 0, 1.8, 0, 1.33), abs=1e-6)

    # パーツ個別: ウイングを 0.1 m 上げ、自身の中心まわりに yaw 90°
    wing.pose = Pose(tz=0.1, yaw=90.0)
    wb = assembly.world_bounds([wing])
    assert (wb[1] - wb[0], wb[3] - wb[2]) == pytest.approx((1.6, 0.3), abs=1e-6)
    assert (wb[0] + wb[1]) / 2 == pytest.approx(3.85) and wb[4] == pytest.approx(1.4)
    wing.pose = Pose()

    # 車両全体: yaw 90° は全体境界の中心まわり → 長さと幅が入れ替わる
    assembly.pose = Pose(yaw=90.0)
    b = assembly.world_bounds()
    assert (b[1] - b[0], b[3] - b[2], b[5] - b[4]) == pytest.approx((1.8, 4.0, 1.33), abs=1e-6)
    assert ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2) == pytest.approx((2.0, 0.9), abs=1e-6)

    # ピッチを付けると最下点が地面から外れる → 接地で Z=0 に戻る
    assembly.pose = Pose(pitch=5.0)
    assert assembly.world_bounds()[4] < -0.01
    assembly.drop_to_ground()
    assert assembly.world_bounds()[4] == pytest.approx(0.0, abs=1e-9)
    assembly.drop_to_ground(clearance=0.05)
    assert assembly.world_bounds()[4] == pytest.approx(0.05, abs=1e-9)
    assembly.center_xy()
    b = assembly.world_bounds()
    assert (b[0] + b[1], b[2] + b[3]) == pytest.approx((0.0, 0.0), abs=1e-9)

    # 単位の上書き (例: 実は m で書かれていた) で寸法が 1000 倍になる
    assembly.pose = Pose()
    body.unit = "m"
    assert assembly.world_bounds([body])[1] - assembly.world_bounds([body])[0] == pytest.approx(4000.0)

    assembly.remove(wing.id)
    assert [p.name for p in assembly.parts] == ["Body"]
    with pytest.raises(KeyError):
        assembly.get(wing.id)


def test_job_subprocess_protocol(box_step, cache_root, tmp_path):
    """GUI と同じ経路 (python -m ljmcfd --job …) で子プロセスとして実行する。"""
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps({"path": str(box_step), "cache_root": str(cache_root)}))
    env = job_environment()
    proc = subprocess.run(
        job_command("cad-import", str(args_file)),
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    events = [e for e in map(parse_protocol_line, proc.stdout.splitlines()) if e]
    types = [e["type"] for e in events]
    assert "progress" in types and types[-1] == "result"
    assert events[-1]["data"]["parts"][0]["n_triangles"] == 12

    args_file.write_text(json.dumps({"path": str(tmp_path / "nope.step"), "cache_root": str(cache_root)}))
    proc = subprocess.run(
        job_command("cad-import", str(args_file)),
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=300,
    )
    assert proc.returncode == 2
    errors = [e for e in map(parse_protocol_line, proc.stdout.splitlines()) if e and e["type"] == "error"]
    assert errors and "見つかりません" in errors[0]["message"]


def test_split_multi_solid_helper():
    from tests.conftest import make_box
    from ljmcfd.cad.occ_io import OccPart, _split_multi_solid, make_compound

    compound = make_compound([make_box(1, 1, 1), make_box(2, 2, 2, origin=(5, 0, 0))])
    parts = _split_multi_solid(OccPart("pair", compound, (0.1, 0.2, 0.3)))
    assert [p.name for p in parts] == ["pair_solid1", "pair_solid2"]
    assert all(p.color == (0.1, 0.2, 0.3) for p in parts)
    single = OccPart("one", make_box(1, 1, 1))
    assert _split_multi_solid(single) == [single]


def test_instances_of_one_shape_keep_their_own_names(instanced_step, cache_root):
    meta, _ = _import(instanced_step, cache_root)
    parts = {p["name"]: p["bounds"] for p in meta["parts"]}
    assert set(parts) == {"Wheel_FL", "Wheel_RL"}
    assert parts["Wheel_RL"][0] - parts["Wheel_FL"][0] == pytest.approx(2600.0, abs=1e-3)


def test_flat_stl_and_cached_bounds(tmp_path, cache_root):
    """厚みのない平板 (凸包が退化する形状) でも読み込め、境界が正しく求まる。"""
    plane = pv.Plane(i_size=2000.0, j_size=1000.0, i_resolution=20, j_resolution=10).triangulate()
    path = tmp_path / "floor.stl"
    plane.save(path)
    meta, _ = _import(path, cache_root)
    assert meta["geometry_unit"] == "mm"
    assembly = Assembly()
    part = assembly.add_from_meta(meta)[0]
    assert len(part.hull_points) < plane.n_points          # 凸包へ縮約されている
    b = assembly.world_bounds()
    assert (b[1] - b[0], b[3] - b[2], b[5] - b[4]) == pytest.approx((2.0, 1.0, 0.0), abs=1e-6)
    # 姿勢を変えるたびに境界が更新される (古いキャッシュを返さない)
    part.pose.roll = 90.0
    b = assembly.world_bounds()
    assert (b[3] - b[2], b[5] - b[4]) == pytest.approx((0.0, 1.0), abs=1e-6)
    assembly.pose.yaw = 90.0
    b = assembly.world_bounds()
    assert (b[1] - b[0], b[3] - b[2], b[5] - b[4]) == pytest.approx((0.0, 2.0, 1.0), abs=1e-6)
    part.unit = "m"
    assert assembly.world_bounds()[5] - assembly.world_bounds()[4] == pytest.approx(1000.0)


def test_instance_location_is_baked_into_brep(instanced_step, cache_root):
    """キャッシュした B-Rep は配置を座標へ焼き込んである (後段のメッシャーが配置を落とさないように)。"""
    from ljmcfd.cad import occ_io

    meta, _ = _import(instanced_step, cache_root)
    rear = next(p for p in meta["parts"] if p["name"] == "Wheel_RL")
    shape = occ_io.read_brep(Path(meta["cache_dir"]) / rear["brep"])
    assert shape.Location().IsIdentity()
    assert occ_io.shape_bounds(shape)[0] == pytest.approx(2600.0 - 300.0, abs=1e-3)
