"""煙エンジン: キャッシュの階層と、設定の変更に応じた最小限の作り直し。Qt に依存しない。

    Level 1  速度場        解析結果の流れ場 (別スレッドで使うため複製を持つ)。結果が変わったら全部捨てる
    Level 2  流跡線        放出源・開始点の数・時間刻みが同じ間は使い回す
    Level 3  密度場        格子 (+ 流体マスク)・広がりが同じ間は使い回す。コマは LRU でキャッシュ
    Level 4  描画          ``gui/smoke_view.py`` (色・濃さ・不透明度・ライティング)

``build`` は重い処理をまとめて行い、別スレッドから呼んでよい (VTK と SciPy は計算中に GIL を手放すので、
GUI は止まらない)。途中で ``cancel`` が立てば ``Cancelled`` を送出して終わる。
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from .density import QUALITY_VOXELS, PathlineSmoke, SmokeGrid, fluid_mask, make_grid, smoke_roi
from .emitter import Emitter, preset_emitters
from .scalar import ScalarSmoke, velocity_on_grid
from .settings import TRANSPORT_SCALAR, SmokeSettings
from .trajectory import Tracks, build_tracks

PREVIEW_VOXELS = QUALITY_VOXELS["low"]      # 操作中の低解像度表示
MAX_SAMPLES = 4000                          # 流跡線 1 本あたりの標本数の上限
TIMELINE_FRAMES = 240                       # タイムラインを何コマに分けてキャッシュするか
VISUAL_NOISE = 0.5                          # 視覚効果のゆらぎの強さ (ON のとき)


class Cancelled(Exception):
    """新しい設定が来たので、途中の計算を打ち切った。"""


@dataclass(frozen=True)
class BuildPlan:
    """1 回の作り直しの内容。キーが同じものはキャッシュから使い回す。"""

    token: int                         # 速度場 (Level 1) の識別子
    emitters: tuple                    # 有効な放出源 (Emitter の複製)
    roi: tuple
    budget: int
    dt: float
    max_age: float
    seeds: int
    spread: float
    lifetime: float
    cache_bytes: int
    timeline_length: float
    noise: float = 0.0
    transport: str = "pathline"

    def _emitters_json(self) -> str:
        return json.dumps([e.to_dict() for e in self.emitters], sort_keys=True)

    @property
    def track_key(self) -> tuple:
        return (self.token, self._emitters_json(), self.seeds, round(self.dt, 12), round(self.max_age, 9))

    @property
    def grid_key(self) -> tuple:
        return (self.token, tuple(round(v, 9) for v in self.roi), self.budget)

    @property
    def preview_grid_key(self) -> tuple:
        return (self.token, tuple(round(v, 9) for v in self.roi), PREVIEW_VOXELS)

    @property
    def smoke_key(self) -> tuple:
        if self.transport == TRANSPORT_SCALAR:
            # 受動スカラー方式は流跡線を使わない。寿命は時間発展そのものに入るので、作り直しの単位に含める
            return ("scalar", self.token, self._emitters_json(), self.seeds, self.grid_key, round(self.spread, 12),
                    round(self.lifetime, 12), round(self.timeline_length, 12))
        return (self.track_key, self.grid_key, round(self.spread, 12))


@dataclass
class SmokeState:
    """作り終えた煙。``smoke`` が本表示、``preview`` が操作中の低解像度表示 (同じ流跡線・同じ時刻の意味)。"""

    plan: BuildPlan
    smoke: PathlineSmoke
    preview: PathlineSmoke
    stats: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.smoke.duration

    @property
    def grid(self) -> SmokeGrid:
        return self.smoke.grid


class _Lru:
    def __init__(self, size: int) -> None:
        self.size = size
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key, value) -> None:
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


class SmokeEngine:
    def __init__(self) -> None:
        self.flow = None
        self.free_stream_speed = 1.0
        self.domain = None
        self.token = 0
        self._tracks = _Lru(2)
        self._masks = _Lru(4)
        self._smokes = _Lru(2)
        self._fields = _Lru(2)        # 受動スカラー方式: 煙の格子へ写した速度場と流体マスク
        # VTK のデータセットは、同時に複数のスレッドから読むと壊れる (セルの取り出しに共有の作業領域を使う)。
        # 作り直しは 1 度に 1 つだけ行い、後から来た作り直しは前のものが終わる (中止される) のを待つ
        self._build_lock = threading.Lock()

    # ------------------------------------------------------------------ Level 1
    def set_flow(self, flow, free_stream_speed: float, domain) -> None:
        """解析結果の流れ場を設定する。別スレッドで使うので複製を持つ (表示側の処理と干渉しない)。"""
        self.token += 1
        self._tracks.clear()
        self._masks.clear()
        self._smokes.clear()
        self._fields.clear()
        if flow is None:
            self.flow = None
            return
        self.flow = flow.copy(deep=True)
        self.free_stream_speed = float(max(free_stream_speed, 1e-6))
        self.domain = tuple(float(v) for v in domain)

    def has_flow(self) -> bool:
        return self.flow is not None

    # ------------------------------------------------------------------ 計画
    def default_emitters(self, settings: SmokeSettings, model_bounds) -> list[Emitter]:
        return preset_emitters(settings.preset, model_bounds, self.domain)

    def plan(self, settings: SmokeSettings, model_bounds) -> BuildPlan | None:
        """設定から作り直しの内容を決める。速度場・有効な放出源が無ければ ``None``。"""
        if self.flow is None or model_bounds is None:
            return None
        emitters = tuple(replace(e) for e in settings.emitters if e.enabled and e.rate > 0)
        if not emitters:
            return None
        roi = smoke_roi(model_bounds, [e.bounds() for e in emitters], self.domain)
        budget = int(settings.voxel_budget) if settings.voxel_budget > 0 else QUALITY_VOXELS.get(settings.quality, QUALITY_VOXELS["medium"])
        grid = make_grid(roi, budget)
        speed = self.free_stream_speed
        # 煙が領域を 3 回通り抜けるだけの時間まで追う (後流で回り続ける軌跡も、ある程度は追える)
        max_age = 3.0 * (roi[1] - roi[0]) / speed
        dt = max(float(settings.sampling_step) * grid.spacing / speed, max_age / MAX_SAMPLES)
        nozzles = sum(len(e.nozzle_points()) for e in emitters)
        seeds = max(min(int(settings.seeds_per_nozzle), int(settings.max_tracks) // max(nozzles, 1)), 1)
        return BuildPlan(
            token=self.token, emitters=emitters, roi=roi, budget=budget, dt=dt, max_age=max_age, seeds=seeds,
            spread=float(settings.spread), lifetime=float(settings.lifetime),
            cache_bytes=int(settings.max_cache_mb) * 1024 * 1024, timeline_length=float(settings.timeline_length),
            noise=VISUAL_NOISE if settings.visual_noise else 0.0, transport=settings.transport,
        )

    # ------------------------------------------------------------------ 作り直し
    def build(self, plan: BuildPlan, progress=None, cancel: threading.Event | None = None) -> SmokeState:
        """計画に従って煙を作る。キャッシュにあるものは使い回す。同時に呼ばれたら順番に実行する。"""
        with self._build_lock:
            return self._build(plan, progress, cancel)

    def _build(self, plan: BuildPlan, progress=None, cancel: threading.Event | None = None) -> SmokeState:
        progress = progress or (lambda fraction, message: None)
        stats: dict = {}

        def check() -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled()

        check()
        if plan.token != self.token or self.flow is None:
            raise Cancelled()
        started = time.perf_counter()
        smoke = self._smokes.get(plan.smoke_key)
        if smoke is None and plan.transport == TRANSPORT_SCALAR:
            smoke = self._build_scalar(plan, stats, progress, check)
            self._smokes.put(plan.smoke_key, smoke)
        if smoke is None:
            tracks = self._tracks.get(plan.track_key)
            if tracks is None:
                progress(0.05, "煙の通り道 (流跡線) を計算中…")
                t = time.perf_counter()
                tracks = build_tracks(self.flow, list(plan.emitters), dt=plan.dt, max_age=plan.max_age, seeds_per_nozzle=plan.seeds)
                stats["seconds_tracks"] = time.perf_counter() - t
                self._tracks.put(plan.track_key, tracks)
            check()
            progress(0.2, "低解像度の表示を準備中…")
            preview_grid = make_grid(plan.roi, PREVIEW_VOXELS)
            preview_mask = self._mask(plan.preview_grid_key, preview_grid, stats, "seconds_preview_mask")
            check()
            progress(0.35, "車体の内部を判定中 (流体マスク)…")
            grid = make_grid(plan.roi, plan.budget)
            mask = self._mask(plan.grid_key, grid, stats, "seconds_mask")
            check()
            progress(0.8, "煙の密度場を準備中…")
            t = time.perf_counter()
            smoke = PathlineSmoke(tracks, list(plan.emitters), grid, mask, free_stream_speed=self.free_stream_speed,
                                  spread=plan.spread, cache_bytes=plan.cache_bytes)
            stats["seconds_bands"] = time.perf_counter() - t
            preview = PathlineSmoke(tracks, list(plan.emitters), preview_grid, preview_mask, free_stream_speed=self.free_stream_speed,
                                    spread=plan.spread, bands=8, cache_bytes=64 * 1024 * 1024)
            smoke.preview = preview
            self._smokes.put(plan.smoke_key, smoke)
        check()
        preview = smoke.preview
        smoke.cache_bytes = plan.cache_bytes
        if isinstance(smoke, ScalarSmoke):
            stats.update(cfl=smoke.cfl, substeps=smoke.substeps, numerical_diffusion=smoke.numerical_diffusion, dt=smoke.dt,
                         tracks=0, samples=0)
        else:
            for item in (smoke, preview):
                item.lifetime = plan.lifetime
                item.noise = plan.noise
            duration = plan.timeline_length if plan.timeline_length > 0 else smoke.natural_duration
            duration = float(min(max(duration, 0.0), smoke.tracks.ages[-1] if smoke.tracks.n_tracks else 0.0))
            for item in (smoke, preview):
                if item.duration != duration:
                    item.duration = duration
                    item.clear_cache()
            stats["tracks"] = smoke.tracks.n_tracks
            stats["samples"] = smoke.tracks.n_samples
        stats["seconds_total"] = time.perf_counter() - started
        progress(1.0, "完了")
        return SmokeState(plan, smoke, preview, stats)

    def _build_scalar(self, plan: BuildPlan, stats: dict, progress, check) -> ScalarSmoke:
        """受動スカラー方式: 速度場を煙の格子へ写し、移流拡散の準備をする (時間発展はコマを求めるときに行う)。"""
        emitters = list(plan.emitters)
        x_start = min(e.bounds()[0] for e in emitters)
        natural = 1.1 * (plan.roi[1] - x_start) / self.free_stream_speed
        duration = plan.timeline_length if plan.timeline_length > 0 else natural
        built = []
        for fraction, budget, key, label in ((0.15, PREVIEW_VOXELS, plan.preview_grid_key, "低解像度の表示を準備中…"),
                                             (0.45, plan.budget, plan.grid_key, "速度場を煙の格子へ写しています…")):
            progress(fraction, label)
            grid = make_grid(plan.roi, budget)
            fields = self._fields.get(key)
            if fields is None:
                t = time.perf_counter()
                fields = velocity_on_grid(self.flow, grid)
                stats["seconds_velocity" if budget == plan.budget else "seconds_preview_velocity"] = time.perf_counter() - t
                self._fields.put(key, fields)
                self._masks.put(key, fields[1])
            check()
            built.append(ScalarSmoke(grid, fields[0], fields[1], emitters, free_stream_speed=self.free_stream_speed, duration=duration,
                                     diffusion=plan.spread, lifetime=plan.lifetime, n_frames=TIMELINE_FRAMES,
                                     cache_bytes=plan.cache_bytes if budget == plan.budget else 64 * 1024 * 1024,
                                     seeds_per_nozzle=plan.seeds))
        preview, smoke = built
        smoke.preview = preview
        return smoke

    def _mask(self, key, grid: SmokeGrid, stats: dict, name: str) -> np.ndarray:
        mask = self._masks.get(key)
        if mask is None:
            t = time.perf_counter()
            mask = fluid_mask(self.flow, grid)
            stats[name] = time.perf_counter() - t
            self._masks.put(key, mask)
        return mask
