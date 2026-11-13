import numpy as np
import pytest

from ljmcfd.core import paths, units
from ljmcfd.core.jobs import PROTOCOL_PREFIX, parse_protocol_line, worker_python
from ljmcfd.core.pose import Pose, bounds_of_points, merge_bounds, pose_matrix, rotation_matrix, transform_points
from ljmcfd.core.project import PartRecord, Project, ProjectError, load_project, save_project


def test_unit_conversions():
    assert units.unit_scale("mm") == pytest.approx(1e-3)
    assert units.to_display(1.5, "mm") == pytest.approx(1500.0)
    assert units.from_display(1500.0, "mm") == pytest.approx(1.5)
    assert units.format_length(4.0, "mm") == "4000.0 mm"
    assert units.format_length(4.0, "m") == "4.0000 m"
    assert units.kmh_to_ms(200.0) == pytest.approx(55.5556, rel=1e-5)
    assert units.ms_to_kmh(units.kmh_to_ms(123.0)) == pytest.approx(123.0)
    with pytest.raises(ValueError):
        units.unit_scale("parsec")


def test_guess_unit_from_size():
    assert units.guess_unit_from_size(4500.0) == "mm"
    assert units.guess_unit_from_size(4.5) == "m"


def test_rotation_conventions():
    # yaw +90°: +X が +Y へ
    assert rotation_matrix(90, 0, 0) @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-12)
    # pitch +90°: +X が -Z へ (Y 軸まわり右手系)
    assert rotation_matrix(0, 90, 0) @ [1, 0, 0] == pytest.approx([0, 0, -1], abs=1e-12)
    # roll +90°: +Y が +Z へ
    assert rotation_matrix(0, 0, 90) @ [0, 1, 0] == pytest.approx([0, 0, 1], abs=1e-12)
    r = rotation_matrix(12, -7, 33)
    assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_pose_matrix_pivot_and_scale():
    pose = Pose(tx=1.0, ty=2.0, tz=3.0, yaw=90.0)
    pivot = np.array([0.5, 0.0, 0.0])
    m = pose_matrix(pose, pivot, scale=0.001)
    # 回転中心そのもの (ローカル 500 mm) は平行移動だけ受ける
    assert transform_points(m, [[500.0, 0.0, 0.0]])[0] == pytest.approx([1.5, 2.0, 3.0])
    # ローカル 1500 mm は中心から +X に 1 m → yaw 90° で +Y に 1 m
    assert transform_points(m, [[1500.0, 0.0, 0.0]])[0] == pytest.approx([1.5, 3.0, 3.0])


def test_bounds_helpers():
    b = bounds_of_points([[0, 0, 0], [1, 2, 3]])
    assert b == (0, 1, 0, 2, 0, 3)
    assert merge_bounds([b, (-1, 0.5, 0, 1, 0, 4), None]) == (-1, 1, 0, 2, 0, 4)
    assert merge_bounds([]) is None
    with pytest.raises(ValueError):
        bounds_of_points(np.empty((0, 3)))


@pytest.mark.parametrize("win,wsl", [
    (r"C:\LJM_CFD_Claude\workspace\run1", "/mnt/c/LJM_CFD_Claude/workspace/run1"),
    ("D:/data/case", "/mnt/d/data/case"),
    ("C:\\", "/mnt/c"),
    (r"\\wsl.localhost\Ubuntu-24.04\home\user\case", "/home/user/case"),
    (r"\\wsl$\Ubuntu-24.04\home\user\case", "/home/user/case"),
    ("/home/user/case", "/home/user/case"),
])
def test_windows_to_wsl(win, wsl):
    assert paths.windows_to_wsl(win) == wsl


def test_wsl_to_windows():
    assert paths.wsl_to_windows("/mnt/c/LJM_CFD_Claude/run") == r"C:\LJM_CFD_Claude\run"
    assert paths.wsl_to_windows("/mnt/c") == "C:\\"
    assert paths.wsl_to_windows("/home/u/case") == r"\\wsl.localhost\Ubuntu-24.04\home\u\case"
    with pytest.raises(ValueError):
        paths.windows_to_wsl("relative\\path")
    with pytest.raises(ValueError):
        paths.wsl_to_windows("relative/path")


def test_path_roundtrip_and_safety():
    original = r"C:\LJM_CFD_Claude\projects\car_01"
    assert paths.wsl_to_windows(paths.windows_to_wsl(original)) == original
    assert paths.is_safe_case_path(original)
    assert not paths.is_safe_case_path(r"C:\Users\太郎\My Case")
    assert not paths.is_ascii_path("C:/データ/car.step")


def test_protocol_line_parsing():
    assert parse_protocol_line("plain log line") is None
    assert parse_protocol_line(PROTOCOL_PREFIX + "{broken") is None
    assert parse_protocol_line(PROTOCOL_PREFIX + '{"type": "progress", "value": 0.5}') == {"type": "progress", "value": 0.5}


def test_project_roundtrip(tmp_path):
    cad = tmp_path / "cad" / "car.step"
    cad.parent.mkdir()
    cad.write_text("dummy")
    project = Project(name="テスト車両", display_unit="m")
    project.vehicle_pose = Pose(tz=0.05, yaw=3.0)
    project.parts.append(PartRecord(
        id="abc", name="Body", source_path=str(cad), source_rel=None, source_index=0, cache_key="k",
        unit="mm", unit_source="file", pose=Pose(tx=0.1, pitch=-2.0), visible=False, color=(1.0, 0.5, 0.25),
    ))
    project.view = {"background": "light", "opacity": 0.4}
    project.sections["tunnel"] = {"speed_kmh": 200.0}
    project.extra["future_key"] = {"keep": True}
    proj_dir = tmp_path / "proj"
    file = save_project(project, proj_dir)
    assert file.name == "project.ljmcfd.json"

    loaded, loaded_dir = load_project(proj_dir)
    assert loaded_dir == proj_dir
    assert loaded.name == "テスト車両" and loaded.display_unit == "m"
    assert loaded.vehicle_pose == Pose(tz=0.05, yaw=3.0)
    rec = loaded.parts[0]
    assert rec.pose == Pose(tx=0.1, pitch=-2.0) and rec.visible is False and rec.color == (1.0, 0.5, 0.25)
    assert rec.source_rel == "../cad/car.step"
    assert loaded.view == {"background": "light", "opacity": 0.4}
    assert loaded.sections["tunnel"] == {"speed_kmh": 200.0}
    assert loaded.extra == {"future_key": {"keep": True}}

    # 元ファイルが移動しても相対パスで見つかる
    rec.source_path = str(tmp_path / "missing" / "car.step")
    assert rec.resolve_source(proj_dir) == cad.resolve()
    cad.unlink()
    assert rec.resolve_source(proj_dir) is None


def test_project_rejects_foreign_and_newer_files(tmp_path):
    (tmp_path / "project.ljmcfd.json").write_text('{"format": "something-else"}')
    with pytest.raises(ProjectError):
        load_project(tmp_path)
    (tmp_path / "project.ljmcfd.json").write_text('{"format": "ljmcfd-project", "format_version": 999}')
    with pytest.raises(ProjectError):
        load_project(tmp_path)
    with pytest.raises(ProjectError):
        load_project(tmp_path / "nope")


def test_worker_python_avoids_pythonw(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "pythonw.exe").write_text("")
    # python.exe が無ければそのまま、あればコンソール版へ切り替える
    assert worker_python(str(scripts / "pythonw.exe")) == str(scripts / "pythonw.exe")
    (scripts / "python.exe").write_text("")
    assert worker_python(str(scripts / "pythonw.exe")) == str(scripts / "python.exe")
    assert worker_python(str(scripts / "python.exe")) == str(scripts / "python.exe")
