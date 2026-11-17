"""煙エンジン (流跡線方式) の数値検証。解析解の分かる速度場を使う。GUI・OpenFOAM なしで実行できる。"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from ljmcfd.smoke.density import PathlineSmoke, fluid_mask, make_grid, smoke_roi
from ljmcfd.smoke.emitter import (
    KIND_GRID, KIND_LINE, KIND_PLANE, KIND_POINT, PRESET_LABELS, PRESET_WHEEL, Emitter, preset_emitters,
)
from ljmcfd.smoke.trajectory import build_tracks, nozzle_seeds

BOX = (-2.0, 6.0, -1.5, 1.5, 0.0, 2.0)


def make_flow(velocity, bounds=BOX, spacing=0.125, hole=None) -> pv.UnstructuredGrid:
    """速度が座標の関数 ``velocity(points)`` で与えられる四面体メッシュ。``hole`` の直方体の中にはセルを作らない。"""
    dims = [int(round((bounds[2 * i + 1] - bounds[2 * i]) / spacing)) + 1 for i in range(3)]
    grid = pv.ImageData(dimensions=dims, spacing=(spacing,) * 3, origin=(bounds[0], bounds[2], bounds[4])).cast_to_unstructured_grid()
    if hole is not None:
        c = grid.cell_centers().points
        inside = ((c[:, 0] > hole[0]) & (c[:, 0] < hole[1]) & (c[:, 1] > hole[2]) & (c[:, 1] < hole[3])
                  & (c[:, 2] > hole[4]) & (c[:, 2] < hole[5]))
        grid = grid.extract_cells(np.nonzero(~inside)[0])
    grid = grid.triangulate()
    grid.point_data.clear()
    grid.cell_data.clear()
    grid.point_data["U"] = np.asarray(velocity(np.asarray(grid.points, dtype=float)), dtype=float)
    return grid


def uniform(speed=10.0):
    return lambda p: np.column_stack([np.full(len(p), speed), np.zeros(len(p)), np.zeros(len(p))])


def build(flow, emitters, *, speed, dims=(96, 48, 48), dt=None, max_age=1.0, seeds=4, spread=0.0, roi=BOX, with_mask=True):
    grid = make_grid(roi, dims)
    dt = dt or 0.5 * grid.spacing / speed
    tracks = build_tracks(flow, emitters, dt=dt, max_age=max_age, seeds_per_nozzle=seeds)
    mask = fluid_mask(flow, grid) if with_mask else None
    return PathlineSmoke(tracks, emitters, grid, mask, free_stream_speed=speed, spread=spread)


def centroid(smoke, density) -> np.ndarray:
    x, y, z = smoke.grid.centers()
    total = density.sum(dtype=np.float64)
    return np.array([(density.sum(axis=(0, 1)) * x).sum(), (density.sum(axis=(0, 2)) * y).sum(), (density.sum(axis=(1, 2)) * z).sum()]) / total


# ----------------------------------------------------------------------------- 放出源

def test_emitter_shapes_and_time_window():
    point = Emitter(kind=KIND_POINT, x=1.0, y=0.2, z=0.5)
    assert point.nozzle_points().tolist() == [[1.0, 0.2, 0.5]]
    line = Emitter(kind=KIND_LINE, x=-1.0, y=0.0, z=0.5, width=1.6, height=0.0, n_y=9)
    points = line.nozzle_points()
    assert points.shape == (9, 3) and np.allclose(points[:, 1], np.linspace(-0.8, 0.8, 9)) and np.allclose(points[:, 2], 0.5)
    vertical = Emitter(kind=KIND_LINE, width=0.0, height=1.0, z=0.6, n_y=5).nozzle_points()
    assert np.allclose(vertical[:, 1], 0.0) and np.allclose(vertical[:, 2], np.linspace(0.1, 1.1, 5))
    assert Emitter(kind=KIND_GRID, width=1.0, height=0.5, n_y=4, n_z=3).nozzle_points().shape == (12, 3)
    plane = Emitter(kind=KIND_PLANE, width=1.0, height=0.5).nozzle_points()
    assert len(plane) == 21 * 11                                   # 面は約 5 cm 間隔で連続に埋める
    assert len(Emitter(kind=KIND_PLANE, width=20.0, height=20.0).nozzle_points()) <= 2000
    # 放出の時間: 開始・終了・間欠
    emitter = Emitter(start=0.1, stop=0.5)
    assert emitter.emission(np.array([0.0, 0.1, 0.3, 0.5, 0.6])).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert Emitter().emission(np.array([-0.1, 0.0, 99.0])).tolist() == [0.0, 1.0, 1.0]      # 既定は止めない
    pulsed = Emitter(pulse_period=0.2, pulse_duty=0.25)
    assert pulsed.emission(np.array([0.0, 0.04, 0.06, 0.19, 0.2, 0.24, 0.26])).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert Emitter.from_dict(emitter.to_dict()) == emitter and Emitter.from_dict({"kind": "bogus"}).kind == KIND_LINE
    # ノズル内の開始点: 決まった配置で、重みの合計はノズル 1 つにつき 1
    seeds, weights = nozzle_seeds(Emitter(kind=KIND_LINE, n_y=3, radius=0.05), 6)
    again, _ = nozzle_seeds(Emitter(kind=KIND_LINE, n_y=3, radius=0.05), 6)
    assert seeds.shape == (18, 3) and weights.sum() == pytest.approx(3.0) and np.array_equal(seeds, again)
    offsets = seeds.reshape(3, 6, 3) - Emitter(kind=KIND_LINE, n_y=3).nozzle_points()[:, None, :]
    assert np.allclose(offsets[..., 0], 0.0) and np.linalg.norm(offsets, axis=2).max() < 0.05


def test_presets_place_emitters_from_the_vehicle_size():
    bounds = (-2.0, 2.0, -0.9, 0.9, 0.0, 1.2)
    domain = (-14.0, 26.0, -6.0, 6.0, 0.0, 7.0)
    for preset in PRESET_LABELS:
        emitters = preset_emitters(preset, bounds, domain)
        assert emitters and all(e.enabled and len(e.nozzle_points()) >= 1 for e in emitters)
        for emitter in emitters:
            b = emitter.bounds()
            assert domain[0] < emitter.x < bounds[1] and b[4] >= 0.0 and b[2] > domain[2] and b[3] < domain[3]
    assert len(preset_emitters(PRESET_WHEEL, bounds, domain)) == 2                 # 左右のタイヤ
    roi = smoke_roi(bounds, [e.bounds() for e in preset_emitters("front", bounds, domain)], domain)
    assert roi[0] < -3.0 and roi[1] > 6.0 and roi[4] == 0.0 and roi[1] <= domain[1]


# ----------------------------------------------------------------------------- 一様流

def test_uniform_flow_moves_smoke_at_the_flow_speed():
    """U = (10, 0, 0) m/s。煙の位置は x(t) = x0 + 10 t に従う。"""
    flow = make_flow(uniform(10.0))
    emitter = Emitter(kind=KIND_POINT, x=-1.5, y=0.1, z=1.0, radius=0.02, stop=0.04)      # 0.04 秒だけ放出した煙のかたまり
    smoke = build(flow, [emitter], speed=10.0, seeds=1)
    tracks = smoke.tracks
    # 流跡線の積分誤差: 標本位置は x0 + U·a
    live = tracks.ages <= tracks.end_age[0]
    assert np.abs(tracks.positions[0, live, 0] - (-1.5 + 10.0 * tracks.ages[live])).max() < 1e-4
    assert np.abs(tracks.positions[0, live, 1] - 0.1).max() < 1e-5 and np.allclose(tracks.speeds[0, live], 10.0, atol=1e-4)
    # 密度場の輸送誤差: 煙のかたまりの重心は x0 + U·(t − 0.02)。誤差はボクセルの 1/4 以内
    for time in (0.1, 0.25, 0.4, 0.6):
        density = smoke.frame(time)
        center = centroid(smoke, density)
        assert center[0] == pytest.approx(-1.5 + 10.0 * (time - 0.02), abs=0.25 * smoke.grid.spacing), time
        assert center[1] == pytest.approx(0.1, abs=0.1 * smoke.grid.spacing) and center[2] == pytest.approx(1.0, abs=0.1 * smoke.grid.spacing)
        # かたまりの長さ = 放出時間 × 流速 = 0.4 m
        along = density.sum(axis=(0, 1))
        x = smoke.grid.centers()[0]
        occupied = x[along > 0.05 * along.max()]
        assert occupied.max() - occupied.min() == pytest.approx(0.4, abs=3 * smoke.grid.spacing)
    # 領域を出た煙は残らない (出口から流出する)
    assert smoke.frame(0.9).sum() == 0.0


def test_timeline_is_deterministic_and_reversible():
    flow = make_flow(uniform(10.0))
    smoke = build(flow, [Emitter(kind=KIND_LINE, x=-1.5, z=1.0, width=1.0, n_y=5)], speed=10.0, spread=0.01)
    assert 0.5 < smoke.duration <= 0.76                              # 入口側から出口 (7.5 m) まで約 0.75 秒
    forward = [smoke.frame(t).copy() for t in (0.1, 0.3, 0.5)]
    assert np.array_equal(smoke.frame(0.3), forward[1]) and np.array_equal(smoke.frame(0.1), forward[0])   # 戻しても同じ
    # 時間を進めると煙の先端が下流へ進む。放出を続けているので上流側は残る
    x = smoke.grid.centers()[0]

    def front(frame):
        """先端 = 流れ方向の線密度が、放出を続けている部分の半分になる位置 (ぼかしの裾に左右されない)。"""
        along = frame.sum(axis=(0, 1))
        plateau = np.median(along[(x > -1.2) & (x < -0.8)])
        return x[along > 0.5 * plateau].max()

    fronts = [front(frame) for frame in forward]
    assert fronts == pytest.approx([-0.5, 1.5, 3.5], abs=smoke.grid.spacing)
    assert all(x[frame.sum(axis=(0, 1)) > 1e-6].min() < -1.3 for frame in forward)
    # タイムラインを等分したコマはキャッシュされ、同じ配列が返る。上限を超えたら古いものから捨てる
    first = smoke.cached_frame(40, 120)
    assert smoke.cached_frame(40, 120) is first and smoke.frame_index(smoke.duration * 40 / 119, 120) == 40
    assert first.dtype == np.float16                                  # キャッシュは半精度で持つ (メモリを半分に)
    exact = smoke.frame(smoke.duration * 40 / 119)
    assert np.allclose(first.astype(np.float32), exact, rtol=2e-3, atol=1e-4 * exact.max())
    assert smoke.peek_frame(40, 120) is first and smoke.peek_frame(41, 120) is None
    smoke.cache_bytes = int(first.nbytes * 2.5)
    for index in range(41, 46):
        smoke.cached_frame(index, 120)
    info = smoke.cache_info()
    assert info["frames"] == 2 and info["bytes"] <= smoke.cache_bytes
    assert smoke.frame(-1.0).sum() == 0.0                            # 放出開始前は煙が無い
    # 寿命: 0.2 秒より古い煙は消える (流跡線は作り直さない)。キャッシュは捨てる
    tracks = smoke.tracks
    smoke.lifetime = 0.2
    assert smoke.tracks is tracks and smoke.cache_info()["frames"] == 0
    aged = smoke.frame(0.5)
    assert x[aged.sum(axis=(0, 1)) > 1e-6].max() < -1.5 + 10.0 * 0.2 + 0.3     # 先端は 0.2 秒ぶん (2 m) まで
    smoke.lifetime = 0.0
    assert np.array_equal(smoke.frame(0.5), forward[2])


def test_total_amount_does_not_depend_on_sampling_or_resolution():
    """標本の数・ノズル内の開始点数・格子の細かさを変えても、煙の総量は変わらない。"""
    flow = make_flow(uniform(10.0))
    emitter = Emitter(kind=KIND_LINE, x=-1.5, z=1.0, width=1.0, n_y=4, radius=0.03)
    time = 0.4
    # 連続放出: 0.4 秒ぶん × ノズル 4 本 × 基準濃度の単位 (U∞·π·r_ref²)
    expected = 4 * time * 10.0 * np.pi * 0.02 ** 2
    amounts = []
    for kwargs in (dict(), dict(seeds=1), dict(seeds=9), dict(dt=0.002), dict(dt=0.0007), dict(dims=(64, 32, 32)),
                   dict(dims=(128, 64, 64)), dict(spread=0.03)):
        smoke = build(flow, [emitter], speed=10.0, **kwargs)
        amounts.append(smoke.total_mass(smoke.frame(time)))
        assert smoke.expected_mass(time) == pytest.approx(expected, rel=0.02), kwargs
    assert np.allclose(amounts, expected, rtol=0.03), amounts
    # 放出量を 2 倍にすれば総量も 2 倍
    double = build(flow, [Emitter(kind=KIND_LINE, x=-1.5, z=1.0, width=1.0, n_y=4, rate=2.0)], speed=10.0)
    assert double.total_mass(double.frame(time)) == pytest.approx(2 * expected, rel=0.03)


def test_spread_follows_the_visual_diffusion_model():
    """σ(a)² = σ0² + 2·D·a。一様流では経過時間 a = (x − x0) / U なので、断面の広がりを直接比べられる。"""
    flow = make_flow(uniform(10.0))
    spread, radius = 0.02, 0.03
    smoke = build(flow, [Emitter(kind=KIND_POINT, x=-1.5, y=0.0, z=1.0, radius=radius)], speed=10.0, dims=(160, 60, 40),
                  spread=spread, seeds=1)
    density = smoke.frame(0.7)
    x, y, _z = smoke.grid.centers()
    widths = []
    for station in (-0.5, 1.0, 3.0, 5.0):
        i = int(np.argmin(np.abs(x - station)))
        profile = density[:, :, i].sum(axis=0).astype(float)
        mean = (profile * y).sum() / profile.sum()
        measured = np.sqrt((profile * (y - mean) ** 2).sum() / profile.sum())
        age = (x[i] + 1.5) / 10.0
        target = np.sqrt(radius ** 2 + 2 * spread * age)
        widths.append(measured)
        assert measured == pytest.approx(target, rel=0.1), (station, measured, target)   # 広がりを 12 段階に分けているぶんの誤差
        assert smoke.sigma(age) == pytest.approx(np.sqrt(0.02 ** 2 + 2 * spread * age))
    assert all(b > a for a, b in zip(widths, widths[1:])) and widths[-1] > 1.5 * widths[0]   # 下流ほど広い
    # 広がり係数 0 なら、太さはほぼ初期のまま (格子の解像度ぶんは広がる)
    thin = build(flow, [Emitter(kind=KIND_POINT, x=-1.5, y=0.0, z=1.0, radius=radius)], speed=10.0, dims=(160, 60, 40), seeds=1)
    profile = thin.frame(0.7)[:, :, int(np.argmin(np.abs(x - 5.0)))].sum(axis=0).astype(float)
    assert np.sqrt((profile * y ** 2).sum() / profile.sum()) < 0.06


# ----------------------------------------------------------------------------- 回転流・固体

def test_speed_field_for_colouring():
    """流速で色付けするための流速場は、煙のある所で速度場の流速に一致し、濃度とは別に返る。"""
    flow = make_flow(lambda p: np.column_stack([5.0 + 5.0 * p[:, 2], np.zeros(len(p)), np.zeros(len(p))]))   # 高さで速くなる
    smoke = build(flow, [Emitter(kind=KIND_LINE, x=-1.5, y=0.0, z=1.0, width=0.0, height=1.2, n_y=7)], speed=10.0, spread=0.005)
    density, speed = smoke.frame(0.4, with_speed=True)
    assert np.array_equal(density, smoke.frame(0.4))
    _x, _y, z = smoke.grid.centers()
    dense = density > 0.2 * density.max()
    expected = np.broadcast_to((5.0 + 5.0 * z)[:, None, None], density.shape)[dense]
    assert np.abs(speed[dense] - expected).max() < 0.6 and speed[density == 0].max() == 0.0


def test_rotating_flow_bends_the_smoke():
    """剛体回転 U = ω × r。煙は円弧を描き、時刻 t には角度 ωt まで届く。"""
    omega = 2.0
    bounds = (-2.0, 2.0, -2.0, 2.0, 0.0, 1.0)
    flow = make_flow(lambda p: np.column_stack([-omega * p[:, 1], omega * p[:, 0], np.zeros(len(p))]), bounds=bounds, spacing=0.1)
    emitter = Emitter(kind=KIND_POINT, x=1.0, y=0.0, z=0.5, radius=0.03, stop=0.05)
    smoke = build(flow, [emitter], speed=2.0, dims=(80, 80, 20), roi=bounds, max_age=1.2, seeds=1, dt=0.004)
    tracks = smoke.tracks
    radius = np.linalg.norm(tracks.positions[0, :, :2], axis=1)
    assert np.abs(radius - 1.0).max() < 5e-3                          # 半径 1 の円から外れない (積分誤差)
    for time in (0.3, 0.6, 0.9):
        center = centroid(smoke, smoke.frame(time))
        angle = omega * (time - 0.025)
        assert center[:2] == pytest.approx([np.cos(angle), np.sin(angle)], abs=0.03), time
    again = smoke.frame(0.3)
    smoke.frame(0.9)
    assert np.array_equal(smoke.frame(0.3), again)                    # 戻すと同じ位置


def test_smoke_does_not_enter_solids():
    hole = (1.0, 2.0, -0.5, 0.5, 0.0, 1.0)                            # 流体のセルが無い直方体 = 固体
    flow = make_flow(uniform(10.0), hole=hole)
    grid = make_grid(BOX, (96, 48, 48))
    mask = fluid_mask(flow, grid)
    x, y, z = grid.centers()
    inside = ((x[None, None, :] > 1.0) & (x[None, None, :] < 2.0) & (np.abs(y)[None, :, None] < 0.5) & (z[:, None, None] < 1.0))
    assert set(np.unique(mask)) == {0.0, 1.0} and not mask[inside].any() and mask[~inside].mean() > 0.97
    # 直方体に正面から当たる煙 (この速度場は回り込まないので、軌跡は壁で終わる) と、すぐ上を通る煙
    emitters = [Emitter(kind=KIND_GRID, x=-1.5, y=0.0, z=0.6, width=0.6, height=1.0, n_y=4, n_z=6, radius=0.03)]
    smoke = build(flow, emitters, speed=10.0, spread=0.05)            # 広がりを大きくして、ぼかしが壁を越えないことを見る
    density = smoke.frame(0.7)
    assert density[inside].max() == 0.0 and density.max() > 0.0
    above = (x[None, None, :] > 1.2) & (x[None, None, :] < 1.8) & (np.abs(y)[None, :, None] < 0.3) & (z[:, None, None] > 1.0) & (z[:, None, None] < 1.3)
    assert density[above].mean() > 0.0                                # 上を通る煙はある
    behind = (x[None, None, :] > 2.2) & (x[None, None, :] < 3.0) & (np.abs(y)[None, :, None] < 0.3) & (z[:, None, None] < 0.5)
    assert density[behind].max() < 0.2 * density[above].max()         # 壁で止まった軌跡は直方体を突き抜けない
    # 壁で終わった軌跡の標本は、終点より後を使わない
    stopped = smoke.tracks.end_age < 0.5
    assert stopped.any() and smoke.tracks.positions[stopped][:, :, 0].max() <= 1.0 + 1e-3
    # マスクなしだと固体の中へぼかしが入る (マスクが効いていることの対照)
    unmasked = build(flow, emitters, speed=10.0, spread=0.05, with_mask=False)
    assert unmasked.frame(0.7)[inside].max() > 0.0


def test_empty_and_disabled_emitters():
    flow = make_flow(uniform(10.0))
    smoke = build(flow, [Emitter(enabled=False), Emitter(kind=KIND_POINT, x=50.0)], speed=10.0)   # 無効 / 領域の外
    assert smoke.tracks.n_tracks == 0 and smoke.duration == 0.0 and smoke.frame(0.3).sum() == 0.0
    assert smoke.frame_index(0.3, 100) == 0 and smoke.expected_mass(0.3) == 0.0


# ----------------------------------------------------------------------------- エンジン (キャッシュの無効化)

def test_engine_rebuilds_only_what_a_setting_change_needs():
    from ljmcfd.smoke.engine import Cancelled, SmokeEngine
    from ljmcfd.smoke.settings import SmokeSettings

    flow = make_flow(uniform(10.0))
    engine = SmokeEngine()
    assert engine.plan(SmokeSettings(), (0.0, 1.0, -0.5, 0.5, 0.0, 1.0)) is None          # 速度場なし
    engine.set_flow(flow, 10.0, BOX)
    model = (0.0, 2.0, -0.5, 0.5, 0.0, 1.0)
    settings = SmokeSettings(quality="medium")
    assert engine.plan(settings, model) is None                                          # 放出源なし
    settings.emitters = engine.default_emitters(settings, model)
    assert settings.emitters and settings.emitters[0].name == "Front Flow"
    plan = engine.plan(settings, model)
    messages = []
    state = engine.build(plan, progress=lambda f, m: messages.append(m))
    assert state.smoke.tracks.n_tracks > 0 and state.duration > 0 and state.preview.grid.n_voxels <= state.smoke.grid.n_voxels
    assert messages[-1] == "完了" and {"seconds_tracks", "seconds_mask", "seconds_bands"} <= set(state.stats)
    assert state.preview.duration == state.smoke.duration
    # 低品質のときは本表示と低解像度表示が同じ格子なので、マスクは 1 回だけ計算する
    low = SmokeSettings(quality="low", emitters=settings.emitters)
    assert "seconds_mask" not in engine.build(engine.plan(low, model)).stats

    def rebuild(**changes):
        for key, value in changes.items():
            setattr(settings, key, value)
        new = engine.build(engine.plan(settings, model))
        return new, set(new.stats) & {"seconds_tracks", "seconds_mask", "seconds_preview_mask", "seconds_bands"}

    # 見た目だけの設定は計画に入らない → 同じ煙 (何も作り直さない)
    same, work = rebuild(color="dark_gray", density=2.0, opacity=0.5, lighting=False)
    assert same.smoke is state.smoke and work == set()
    # 寿命: 同じ流跡線・同じ密度場の準備で、コマのキャッシュだけ捨てる
    state.smoke.cached_frame(10, 120)
    aged, work = rebuild(lifetime=0.3)
    assert aged.smoke is state.smoke and work == set() and aged.smoke.lifetime == 0.3 and aged.smoke.cache_info()["frames"] == 0
    # 広がり: 流跡線とマスクは使い回し、密度場の準備だけ作り直す
    wider, work = rebuild(spread=0.05)
    assert wider.smoke is not state.smoke and wider.smoke.tracks is state.smoke.tracks and work == {"seconds_bands"}
    # 放出源を少し動かす: 流跡線は作り直すが、格子 (マスク) はそのまま
    settings.emitters[0].z += 0.1
    moved, work = rebuild()
    assert moved.smoke.tracks is not state.smoke.tracks and moved.smoke.grid == wider.smoke.grid
    assert work == {"seconds_tracks", "seconds_bands"}
    # 品質: 格子が変わる → マスクと (時間刻みが変わるので) 流跡線も
    finer, work = rebuild(quality="high")
    assert finer.smoke.grid.n_voxels > moved.smoke.grid.n_voxels and work == {"seconds_tracks", "seconds_mask", "seconds_bands"}
    # 元の品質に戻すと、キャッシュから使い回す
    back, work = rebuild(quality="medium")
    assert work == set() and back.smoke is moved.smoke
    # タイムラインの長さを指定できる (流跡線を追った時間まで)
    longer, _ = rebuild(timeline_length=0.5)
    assert longer.duration == 0.5 and longer.preview.duration == 0.5
    # 速度場 (解析結果) が変わったら全部捨てる。古い計画での作り直しは打ち切られる
    old_plan = engine.plan(settings, model)
    engine.set_flow(flow, 10.0, BOX)
    with pytest.raises(Cancelled):
        engine.build(old_plan)
    fresh = engine.build(engine.plan(settings, model))
    assert fresh.smoke is not longer.smoke and {"seconds_tracks", "seconds_mask"} <= set(fresh.stats)
    # 打ち切り
    import threading
    stop = threading.Event()
    stop.set()
    settings.emitters[0].z += 0.1
    with pytest.raises(Cancelled):
        engine.build(engine.plan(settings, model), cancel=stop)


def test_smoke_settings_round_trip_and_old_projects():
    from ljmcfd.smoke.settings import SmokeSettings

    settings = SmokeSettings(quality="high", color="custom", custom_color=(0.2, 0.4, 0.6), density=1.7, lifetime=0.4,
                             emitters=[Emitter(kind=KIND_GRID, x=-1.0, n_y=3, n_z=2), Emitter(name="B", enabled=False)])
    restored = SmokeSettings.from_dict(settings.to_dict())
    assert restored == settings and restored.rgb() == (0.2, 0.4, 0.6)
    assert SmokeSettings.from_dict(None) == SmokeSettings()                          # 煙の設定が無い古いプロジェクト
    broken = SmokeSettings.from_dict({"quality": "bogus", "color": 3, "density": "x", "emitters": [None, {"x": 1.5}], "backend": "?"})
    assert broken.quality == "medium" and broken.color == "white" and broken.density == 1.0 and broken.backend == "auto"
    assert len(broken.emitters) == 1 and broken.emitters[0].x == 1.5


def test_concurrent_builds_run_one_at_a_time():
    """設定を立て続けに変えて作り直しが重なっても、VTK のデータを同時に読まない (同時に読むと異常終了する)。"""
    import threading

    from ljmcfd.smoke.engine import Cancelled, SmokeEngine
    from ljmcfd.smoke.settings import SmokeSettings

    flow = make_flow(uniform(10.0))
    engine = SmokeEngine()
    engine.set_flow(flow, 10.0, BOX)
    model = (0.0, 2.0, -0.5, 0.5, 0.0, 1.0)
    active, peak, results = [0], [0], []
    original = engine._build

    def counting(*args, **kwargs):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        try:
            return original(*args, **kwargs)
        finally:
            active[0] -= 1

    engine._build = counting
    threads = []
    for index in range(6):
        settings = SmokeSettings(quality="low", emitters=[Emitter(kind=KIND_LINE, x=-1.5, z=0.3 + 0.2 * index, width=1.0, n_y=5)])
        plan = engine.plan(settings, model)
        cancel = threading.Event()
        if index % 2:
            cancel.set()                                            # 途中で取り消された作り直し

        def run(plan=plan, cancel=cancel):
            try:
                results.append(engine.build(plan, cancel=cancel))
            except Cancelled:
                results.append(None)

        threads.append(threading.Thread(target=run))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert peak[0] == 1 and len(results) == 6
    assert sum(r is None for r in results) == 3 and all(r.smoke.tracks.n_tracks > 0 for r in results if r is not None)


# ----------------------------------------------------------------------------- 受動スカラー方式

def scalar(flow, emitters, *, duration, roi=BOX, dims=(96, 48, 48), diffusion=0.0, lifetime=0.0, n_frames=60, seeds=1):
    from ljmcfd.smoke.scalar import ScalarSmoke, velocity_on_grid

    grid = make_grid(roi, dims)
    velocity, mask = velocity_on_grid(flow, grid)
    return ScalarSmoke(grid, velocity, mask, emitters, free_stream_speed=10.0, duration=duration, diffusion=diffusion,
                       lifetime=lifetime, n_frames=n_frames, seeds_per_nozzle=seeds)


def test_scalar_diffusion_without_flow_conserves_mass_and_spreads():
    """速度 0 では拡散だけ: 総量が保存され、分散が 2·D·t ずつ増える。"""
    still = make_flow(lambda p: np.zeros((len(p), 3)), bounds=(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0), spacing=0.1)
    D = 0.01
    smoke = scalar(still, [Emitter(kind=KIND_POINT, x=0.0, y=0.0, z=1.0, stop=0.02)], duration=1.0, n_frames=51,
                   roi=(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0), dims=(64, 64, 64), diffusion=D)
    assert smoke.cfl == 0.0 and smoke.substeps == 1
    emitted = smoke.emitted_mass(0.1)
    variances = []
    for time in (0.2, 0.6, 1.0):
        c = smoke.frame(time)
        assert smoke.total_mass(c) == pytest.approx(emitted, rel=0.01), time          # 許容誤差 1 %
        x, y, z = smoke.grid.centers()
        total = c.sum(dtype=np.float64)
        mean_x = (c.sum(axis=(0, 1)) * x).sum() / total
        variances.append((c.sum(axis=(0, 1)) * (x - mean_x) ** 2).sum() / total)
    # 分散の増え方 = 2·D·Δt (放出時の初期の広がりは差し引かれる)。許容誤差 10 %
    assert variances[1] - variances[0] == pytest.approx(2 * D * 0.4, rel=0.1)
    assert variances[2] - variances[1] == pytest.approx(2 * D * 0.4, rel=0.1)


def test_scalar_advection_speed_cfl_and_numerical_diffusion():
    flow = make_flow(uniform(10.0))
    smoke = scalar(flow, [Emitter(kind=KIND_POINT, x=-1.5, y=0.0, z=1.0, stop=0.04)], duration=1.0, n_frames=101)
    h = smoke.grid.spacing
    assert 0 < smoke.cfl <= 1.5 + 1e-9 and smoke.dt == pytest.approx(0.01 / smoke.substeps)   # コマの間隔 1.0 s / 100
    assert smoke.numerical_diffusion >= 0.0
    for time in (0.2, 0.4):
        c = smoke.frame(time)
        center = centroid(smoke, c)
        # 重心は x0 + U·(t − 0.02)。セミラグランジュ法の誤差を含めて、ボクセル 1 個ぶん以内
        assert center[0] == pytest.approx(-1.5 + 10.0 * (time - 0.02), abs=h), time
        assert center[1] == pytest.approx(0.0, abs=0.2 * h) and center[2] == pytest.approx(1.0, abs=0.2 * h)
    # 領域の中にある間は、移流で煙の量は失われない (拡散なし)。許容誤差 0.5 %
    assert smoke.total_mass(smoke.frame(0.5)) == pytest.approx(smoke.emitted_mass(0.5), rel=0.005)
    # 流出: 出口 (x = 6 m) を越えた煙は領域から出ていく (1.0 秒後の重心は x = 8.3 m)
    assert smoke.total_mass(smoke.frame(1.0)) < 0.05 * smoke.emitted_mass(1.0)


def test_scalar_timeline_is_deterministic_with_keyframes():
    flow = make_flow(uniform(10.0))
    emitters = [Emitter(kind=KIND_LINE, x=-1.5, z=1.0, width=1.0, n_y=5, pulse_period=0.1, pulse_duty=0.5)]
    sequential = scalar(flow, emitters, duration=0.6, n_frames=61, diffusion=0.005)
    frames = [sequential.frame(sequential.duration * k / 60).copy() for k in range(0, 61, 5)]
    assert sequential.cache_info()["keyframes"] >= 4
    jumped = scalar(flow, emitters, duration=0.6, n_frames=61, diffusion=0.005)
    for k in (55, 20, 45, 5, 60, 0):
        assert np.array_equal(jumped.frame(jumped.duration * k / 60), frames[k // 5]), k    # 飛んでも同じ結果
    # キャッシュは半精度で持つ
    item = jumped.cached_frame(30, 61)
    assert item.dtype == np.float16 and jumped.cached_frame(30, 61) is item
    assert np.allclose(item.astype(np.float32), frames[6], rtol=2e-3, atol=1e-3 * frames[6].max())
    # 寿命: 減衰を掛ける (キャッシュとキーフレームは捨てる)
    jumped.lifetime = 0.05
    assert jumped.cache_info()["keyframes"] == 0 and jumped.total_mass(jumped.frame(0.3)) < 0.5 * sequential.total_mass(frames[6])


def test_scalar_smoke_stays_out_of_solids():
    hole = (1.0, 2.0, -0.5, 0.5, 0.0, 1.0)
    flow = make_flow(uniform(10.0), hole=hole)
    smoke = scalar(flow, [Emitter(kind=KIND_GRID, x=-1.5, y=0.0, z=0.6, width=0.6, height=1.0, n_y=4, n_z=6)], duration=0.6,
                   diffusion=0.05)
    x, y, z = smoke.grid.centers()
    inside = ((x[None, None, :] > 1.0) & (x[None, None, :] < 2.0) & (np.abs(y)[None, :, None] < 0.5) & (z[:, None, None] < 1.0))
    for time in (0.2, 0.4, 0.6):
        c = smoke.frame(time)
        assert c.max() > 0 and c[inside].max() == 0.0
    assert smoke.velocity[inside].max() == 0.0 and smoke.mask[inside].max() == 0.0


def test_engine_builds_scalar_smoke():
    from ljmcfd.smoke.engine import SmokeEngine
    from ljmcfd.smoke.scalar import ScalarSmoke
    from ljmcfd.smoke.settings import TRANSPORT_SCALAR, SmokeSettings

    engine = SmokeEngine()
    engine.set_flow(make_flow(uniform(10.0)), 10.0, BOX)
    model = (0.0, 2.0, -0.5, 0.5, 0.0, 1.0)
    settings = SmokeSettings(quality="low", transport=TRANSPORT_SCALAR)
    settings.emitters = engine.default_emitters(settings, model)
    state = engine.build(engine.plan(settings, model))
    assert isinstance(state.smoke, ScalarSmoke) and isinstance(state.preview, ScalarSmoke) and state.duration > 0
    assert {"cfl", "substeps", "numerical_diffusion"} <= set(state.stats)
    c = state.smoke.cached_frame(120, 240)
    assert c.dtype == np.float16 and float(c.max()) > 0
    # 流跡線方式に戻すと、流跡線方式の煙になる (速度場は使い回す)
    settings.transport = "pathline"
    assert not isinstance(engine.build(engine.plan(settings, model)).smoke, ScalarSmoke)
