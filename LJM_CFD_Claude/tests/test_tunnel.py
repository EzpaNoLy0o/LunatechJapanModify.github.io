import math

import numpy as np
import pytest
import pyvista as pv

from ljmcfd.cad.assembly import Assembly
from ljmcfd.cad.import_job import import_cad_job
from ljmcfd.mesh.request import part_payloads
from ljmcfd.tunnel import spec as ts
from ljmcfd.tunnel.frontal_area import frontal_area_job, projected_area

CAR = (0.0, 4.0, -0.9, 0.9, 0.0, 1.2)   # 4.0 x 1.8 x 1.2 m、接地


def test_air_properties_at_standard_conditions():
    flow = ts.FlowConditions()
    assert flow.speed_kmh == pytest.approx(200.0)
    assert flow.speed_ms == pytest.approx(55.5556, rel=1e-5)
    assert flow.density == pytest.approx(1.2041, rel=1e-3)          # 20 °C, 1 atm
    assert flow.mu == pytest.approx(1.813e-5, rel=5e-3)
    assert flow.nu == pytest.approx(1.506e-5, rel=5e-3)
    assert flow.mach == pytest.approx(55.5556 / 343.2, rel=1e-3)
    assert flow.dynamic_pressure == pytest.approx(0.5 * flow.density * (200.0 / 3.6) ** 2)
    assert flow.reynolds(4.0) == pytest.approx(200.0 / 3.6 * 4.0 / flow.nu)
    # 乱流量: k = 1.5 (U I)^2、ω = k / (ν · 粘性比)
    assert flow.k_inlet == pytest.approx(1.5 * (200.0 / 3.6 * 0.01) ** 2)
    assert flow.omega_inlet == pytest.approx(flow.k_inlet / (flow.nu * 10.0))
    # 密度の手動指定
    flow.density_override = 1.1
    assert flow.density == 1.1 and flow.nu == pytest.approx(flow.mu / 1.1)
    # 温度が上がると密度は下がる
    assert ts.air_density(101325.0, 35.0) < ts.air_density(101325.0, 5.0)


def test_default_tunnel_and_blockage():
    spec = ts.TunnelSpec()
    ts.apply_mode_defaults(spec, CAR)
    assert spec.has_ground and spec.mode == ts.MODE_VEHICLE
    assert (spec.margin_front, spec.margin_rear, spec.margin_side, spec.margin_top) == pytest.approx((12.0, 24.0, 5.4, 4.8))
    assert spec.domain_bounds(CAR) == pytest.approx((-12.0, 28.0, -6.3, 6.3, 0.0, 6.0))
    assert spec.cross_section_area(CAR) == pytest.approx(12.6 * 6.0)
    assert spec.blockage_ratio(CAR) is None                       # 前面投影面積が未計算なら推測しない
    assert spec.effective_reference_area() is None
    spec.frontal_area = 2.0
    assert spec.blockage_ratio(CAR) == pytest.approx(2.0 / 75.6)
    assert spec.effective_reference_area() == 2.0
    spec.reference_area = 1.9                                     # ユーザー指定が優先
    assert spec.effective_reference_area() == 1.9
    assert spec.patch_names() == ["inlet", "outlet", "sideL", "sideR", "ground", "top"]
    assert not [i for i in ts.validate(spec, CAR) if i[0] == "error"]


def test_component_mode_has_no_ground():
    spec = ts.TunnelSpec(mode=ts.MODE_COMPONENT)
    wing = (0.0, 0.3, -0.8, 0.8, 1.0, 1.05)
    ts.apply_mode_defaults(spec, wing)
    assert not spec.has_ground and not spec.moving_ground
    d = spec.domain_bounds(wing)
    assert d[4] == pytest.approx(1.0 - 4.8) and d[5] == pytest.approx(1.05 + 4.8)   # 上下対称に余白
    assert spec.patch_names()[4] == "bottom"
    assert spec.model_sink(wing) == 0.0


def test_ground_contact_sink_and_validation():
    spec = ts.TunnelSpec()
    ts.apply_mode_defaults(spec, CAR)
    assert spec.model_sink(CAR) == pytest.approx(0.010)           # 接地 → 沈める
    lifted = (*CAR[:4], 0.05, 1.25)
    assert spec.model_sink(lifted) == 0.0
    levels = lambda bounds: {lvl for lvl, _ in ts.validate(spec, bounds)}
    assert "info" in levels(CAR)
    narrow = (*CAR[:4], 0.004, 1.204)
    assert any("隙間" in msg for _, msg in ts.validate(spec, narrow))
    buried = (*CAR[:4], -0.6, 0.6)
    assert "error" in levels(buried)
    spec.margin_front = 0.0
    assert any(lvl == "error" and "前方" in msg for lvl, msg in ts.validate(spec, CAR))
    spec.mode = ts.MODE_INTERNAL
    assert ts.validate(spec, CAR)[0][0] == "error" and "未実装" in ts.validate(spec, CAR)[0][1]


def test_validation_warns_on_blockage_and_mach():
    spec = ts.TunnelSpec()
    ts.apply_mode_defaults(spec, CAR)
    spec.margin_side, spec.margin_top = 0.5, 1.0
    spec.frontal_area = 2.0
    assert any("閉塞率" in msg for _, msg in ts.validate(spec, CAR))
    spec.flow.speed_ms = 150.0
    assert any("マッハ" in msg for _, msg in ts.validate(spec, CAR))


def test_spec_roundtrip():
    spec = ts.TunnelSpec(mode=ts.MODE_COMPONENT, has_ground=False, moving_ground=False, margin_front=1.5,
                         frontal_area=0.31, reference_area=None, generated=True)
    spec.flow = ts.FlowConditions(speed_ms=40.0, temperature_c=30.0, density_override=1.15)
    again = ts.TunnelSpec.from_dict(spec.to_dict())
    assert again == spec
    assert ts.TunnelSpec.from_dict({}) == ts.TunnelSpec()
    assert ts.TunnelSpec.from_dict({"flow": {"density_override": None}}).flow.density_override is None


def test_yplus_estimate_is_consistent():
    flow = ts.FlowConditions()
    h = ts.first_layer_height_for_yplus(50.0, flow, 4.0)
    assert ts.estimate_yplus(h, flow, 4.0) == pytest.approx(50.0)
    assert 0.0005 < h < 0.003                                     # 200 km/h・全長 4 m で 1 mm 前後
    assert ts.estimate_yplus(0.05, flow, 4.0) > 1000              # 境界層なしの粗い四面体は壁関数の範囲外


def test_projected_area_analytic_shapes():
    cube = pv.Cube(x_length=3.0, y_length=2.0, z_length=1.0).triangulate()
    tris = cube.faces.reshape(-1, 4)[:, 1:]
    assert projected_area([(cube.points, tris)])["area"] == pytest.approx(2.0, rel=2e-3)
    sphere = pv.Sphere(radius=0.5, theta_resolution=120, phi_resolution=120)
    s_tris = sphere.faces.reshape(-1, 4)[:, 1:]
    assert projected_area([(sphere.points, s_tris)])["area"] == pytest.approx(math.pi * 0.25, rel=3e-3)
    # 重なる 2 つの物体は二重に数えない / 離れていれば和になる
    behind = cube.points + [10.0, 0.0, 0.0]
    assert projected_area([(cube.points, tris), (behind, tris)])["area"] == pytest.approx(2.0, rel=2e-3)
    beside = cube.points + [0.0, 5.0, 0.0]
    assert projected_area([(cube.points, tris), (beside, tris)])["area"] == pytest.approx(4.0, rel=2e-3)
    # 地面より下は含めない
    assert projected_area([(cube.points, tris)], clip_below_z=0.0)["area"] == pytest.approx(1.0, rel=4e-3)


def test_frontal_area_job_uses_world_pose(assembly_step, cache_root):
    meta = import_cad_job({"path": str(assembly_step), "cache_root": str(cache_root)})
    assembly = Assembly()
    assembly.add_from_meta(meta)
    # ボディ 1.8 x 1.2 = 2.16 m²、ウイング (1.6 x 0.03) はボディの上に 0.1 m 浮いている
    result = frontal_area_job({"parts": part_payloads(assembly)})
    assert result["area"] == pytest.approx(2.16 + 1.6 * 0.03, rel=3e-3)
    # ヨー 90° にすると側面積 (4.0 x 1.2 + ウイング 0.3 x 0.03) になる
    assembly.pose.yaw = 90.0
    result = frontal_area_job({"parts": part_payloads(assembly)})
    assert result["area"] == pytest.approx(4.0 * 1.2 + 0.3 * 0.03, rel=3e-3)
    # 非表示パーツは解析対象外
    assembly.pose.yaw = 0.0
    assembly.parts[1].visible = False
    assert frontal_area_job({"parts": part_payloads(assembly)})["area"] == pytest.approx(2.16, rel=3e-3)
