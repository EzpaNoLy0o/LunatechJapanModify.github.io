"""煙の密度場 (Level 3): 流跡線の標本を、空間に連続した濃度分布 (ボクセル格子) へ変換する。

標本 1 個を、中心が標本位置・広がりが σ のガウス分布として格子へ足し込む:

    ρ(x, t) = Σ_i  m_i(t) · G(x − x_i ; σ(a_i))

* m_i(t) = (軌跡の重み) × dt × S(t − a_i)   … 放出時刻 t − a_i に放出していた量。標本は経過時間で等間隔なので、
  流れの遅い所には標本が詰まり、濃度が上がる (実際の煙と同じ)。標本の数・格子の細かさを変えても総量は変わらない。
* σ(a)² = σ0² + 2·D_visual·a            … 放出直後は細く、流れるにつれて広がる。**表示用の近似** であり、
  OpenFOAM が計算した乱流拡散ではない。

ガウス分布の足し込みは、標本を経過時間の帯に分け、帯ごとに「格子へ配分 → ガウスぼかし」で行う
(標本 1 個ずつ分布を描くより桁違いに速い)。固体 (車体) の中へは煙を置かない。ぼかしで固体側へ
はみ出した分は、流体側へ戻す (正規化畳み込み)。

濃度の単位は無次元: 「標準の放出量のノズル 1 本から出た直後の煙を、主流の速さで運んだときの濃度」を 1 とする。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk
from scipy import ndimage

from .trajectory import Tracks

#: 品質ごとのボクセル数の予算。仕様の格子寸法 (Low 64³、Medium 128×96×96、High 192×128×128) と同じ総数を、
#: 煙の領域 (ROI) の形に合わせて立方体のボクセルで割り当てる (前後に長い領域でも、どの向きも同じ解像度になる)
QUALITY_VOXELS = {"low": 64 * 64 * 64, "medium": 128 * 96 * 96, "high": 192 * 128 * 128, "ultra": 288 * 192 * 192}
QUALITY_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "ultra": "Ultra"}
MAX_AXIS_VOXELS = 640
REFERENCE_RADIUS = 0.02      # 濃度 1 の基準にするノズル半径 [m]
BLUR_TRUNCATE = 2.5


@dataclass
class SmokeGrid:
    """煙を置く領域 (ROI) のボクセル格子。ボクセルは立方体で、実寸に対応する。"""

    origin: tuple[float, float, float]    # ボクセル (0, 0, 0) の中心 [m]
    spacing: float                        # ボクセルの一辺 [m]
    dims: tuple[int, int, int]            # (nx, ny, nz)

    @property
    def shape(self) -> tuple[int, int, int]:
        """NumPy 配列の形 (nz, ny, nx)。"""
        return (self.dims[2], self.dims[1], self.dims[0])

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.dims))

    @property
    def bounds(self) -> tuple[float, ...]:
        h = self.spacing
        lo = [o - h / 2 for o in self.origin]
        return (lo[0], lo[0] + h * self.dims[0], lo[1], lo[1] + h * self.dims[1], lo[2], lo[2] + h * self.dims[2])

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(self.origin[i] + self.spacing * np.arange(self.dims[i]) for i in range(3))

    def image(self) -> pv.ImageData:
        """ボクセル中心を節点とする VTK の画像データ (描画・標本化用)。"""
        return pv.ImageData(dimensions=self.dims, spacing=(self.spacing,) * 3, origin=self.origin)


def make_grid(roi, budget) -> SmokeGrid:
    """領域 ``roi`` (xmin, xmax, …) を立方体ボクセルで覆う格子。

    budget: 総ボクセル数 (int)、または軸ごとの上限 (nx, ny, nz)
    """
    lengths = [max(roi[2 * i + 1] - roi[2 * i], 1e-6) for i in range(3)]
    if isinstance(budget, (tuple, list)):
        h = max(lengths[i] / budget[i] for i in range(3))
    else:
        h = (lengths[0] * lengths[1] * lengths[2] / max(int(budget), 8)) ** (1.0 / 3.0)
        h = max(h, max(lengths) / MAX_AXIS_VOXELS)
    dims = tuple(max(int(np.ceil(lengths[i] / h - 1e-9)), 2) for i in range(3))
    return SmokeGrid(origin=tuple(roi[2 * i] + h / 2 for i in range(3)), spacing=float(h), dims=dims)


def smoke_roi(model_bounds, emitter_bounds: list, domain, wake: float = 1.2) -> tuple[float, ...]:
    """煙を追う領域 (ROI): 車両の少し前から後方 (車長の ``wake`` 倍) まで。計算領域の外へは出さない。

    放出源を少し動かしただけで格子 (と、その流体マスク) を作り直さずに済むよう、既定の領域は車両の前方に
    余裕を持たせてあり、放出源がはみ出したときも車長の 1/4 刻みで広げる。
    """
    x0, x1, y0, y1, z0, z1 = model_bounds
    length, width, height = x1 - x0, y1 - y0, z1 - z0
    roi = [x0 - 0.35 * length, x1 + wake * length, y0 - 0.45 * width, y1 + 0.45 * width, z0, z1 + 0.6 * height]
    step = 0.25 * length

    def lower(edge, value):
        return edge if value >= edge else edge - step * np.ceil((edge - value) / step)

    def upper(edge, value):
        return edge if value <= edge else edge + step * np.ceil((value - edge) / step)

    for b in emitter_bounds:
        margin = 0.05 * length
        roi[0] = lower(roi[0], b[0] - margin)
        roi[1] = upper(roi[1], b[1] + margin)
        roi[2], roi[3] = lower(roi[2], b[2] - margin), upper(roi[3], b[3] + margin)
        roi[4], roi[5] = lower(roi[4], b[4] - margin), upper(roi[5], b[5] + margin)
    for i in range(3):
        roi[2 * i], roi[2 * i + 1] = max(roi[2 * i], domain[2 * i]), min(roi[2 * i + 1], domain[2 * i + 1])
    return tuple(float(v) for v in roi)


def fluid_mask(flow, grid: SmokeGrid) -> np.ndarray:
    """ボクセル中心が流体のセルの中にあれば 1、固体 (車体の内部) や領域の外なら 0。形は (nz, ny, nx)。"""
    probe = vtk.vtkResampleWithDataSet()
    probe.SetInputData(grid.image())
    probe.SetSourceData(flow)
    probe.SetPassPointArrays(False)
    probe.Update()
    output = pv.wrap(probe.GetOutput())
    return np.asarray(output.point_data["vtkValidPointMask"]).reshape(grid.shape).astype(np.float32)


@dataclass
class _Band:
    sigma: float                 # この帯の広がり [m]
    sigma_voxels: float
    sample: np.ndarray           # (M,) 標本の通し番号 (track * n + step)
    flat: np.ndarray             # (M, 8) 配分先のボクセル (平坦化した添字)
    tri: np.ndarray              # (M, 8) 三線形の配分重み
    box: tuple[slice, slice, slice]    # 配分先 + ぼかしの届く範囲
    norm: np.ndarray | None      # box 内の、ぼかしたマスク (正規化畳み込み用)


class PathlineSmoke:
    """流跡線方式の煙。``frame(t)`` は時刻 t の濃度場を返す、t だけで決まる関数 (巻き戻しても同じ結果)。"""

    is_steady = True
    sequential = False          # どの時刻のコマも直接求まる

    def __init__(self, tracks: Tracks, emitters: list, grid: SmokeGrid, mask: np.ndarray | None, *, free_stream_speed: float,
                 spread: float = 0.02, bands: int = 12, cache_bytes: int = 256 * 1024 * 1024, lifetime: float = 0.0) -> None:
        """
        spread:      表示用の広がり係数 D_visual [m²/s]
        bands:       広がりの段階数 (12 で、広がりのモデルとの差が 5 % 程度。少ないと幅が段々に変わって見える)
        cache_bytes: 濃度場のキャッシュの上限
        lifetime:    煙の寿命 [s]。これより古い煙は消える (最後の 2 割で薄くなる)。0 以下なら寿命なし
        """
        self.tracks = tracks
        self.emitters = emitters
        self.grid = grid
        self.mask = None if mask is None else mask.astype(np.float32)
        self.spread = float(spread)
        self.free_stream_speed = float(max(free_stream_speed, 1e-9))
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._lifetime = 0.0
        self._fade = None
        self._noise = 0.0
        self._noise_factor = None
        n = tracks.n_samples
        ages = tracks.ages
        # 濃度 1 の基準: 標準のノズル 1 本 (放出量 1) の煙を主流の速さで運んだときの濃度
        self._unit = 1.0 / (self.free_stream_speed * np.pi * REFERENCE_RADIUS ** 2)
        self._ages_flat = np.broadcast_to(ages[None, :], (tracks.n_tracks, n)).ravel()
        valid = ages[None, :] <= tracks.end_age[:, None]
        self._mass = (tracks.weights[:, None] * tracks.dt * valid).astype(np.float32).ravel() / (grid.spacing ** 3) / self._unit
        self._emitter_flat = np.repeat(tracks.emitter, n)
        self._speed_flat = tracks.speeds.ravel()
        # タイムラインの長さ: 8 割の軌跡が領域を出る (または終わる) までの時間
        self._exit_age = self._exit_ages()
        natural = float(np.percentile(self._exit_age, 80)) if tracks.n_tracks else 0.0
        #: 煙が領域を通り抜ける時間 (自動のタイムラインの長さ)。``duration`` は変更できる
        self.natural_duration = float(min(max(natural, tracks.dt), ages[-1])) if tracks.n_tracks else 0.0
        self.duration = self.natural_duration
        self.preview: "PathlineSmoke | None" = None
        self._bands = self._build_bands(max(int(bands), 1))
        self.lifetime = lifetime

    # ------------------------------------------------------------------ 準備
    def _voxel_coordinates(self) -> np.ndarray:
        return (self.tracks.positions.reshape(-1, 3) - np.asarray(self.grid.origin, dtype=np.float32)) / np.float32(self.grid.spacing)

    def _exit_ages(self) -> np.ndarray:
        """軌跡ごとの、ROI を出る (または軌跡が終わる) 経過時間。"""
        tracks = self.tracks
        if tracks.n_tracks == 0:
            return np.zeros(0)
        b = self.grid.bounds
        p = tracks.positions
        inside = ((p[..., 0] >= b[0]) & (p[..., 0] <= b[1]) & (p[..., 1] >= b[2]) & (p[..., 1] <= b[3])
                  & (p[..., 2] >= b[4]) & (p[..., 2] <= b[5]))
        outside = ~inside
        first_out = np.where(outside.any(axis=1), outside.argmax(axis=1), tracks.n_samples - 1)
        return np.minimum(first_out * tracks.dt, tracks.end_age)

    def sigma(self, age) -> np.ndarray:
        """経過時間 ``age`` [s] の煙の広がり σ [m] (放出源ごとの初期の太さ + 表示用の広がり)。"""
        return np.sqrt(REFERENCE_RADIUS ** 2 + 2.0 * self.spread * np.asarray(age, dtype=float))

    def _build_bands(self, count: int) -> list[_Band]:
        tracks, grid = self.tracks, self.grid
        if tracks.n_tracks == 0:
            return []
        nx, ny, nz = grid.dims
        coordinates = self._voxel_coordinates()
        base = np.floor(coordinates).astype(np.int64)
        frac = (coordinates - base).astype(np.float32)
        inside = ((base[:, 0] >= 0) & (base[:, 0] < nx - 1) & (base[:, 1] >= 0) & (base[:, 1] < ny - 1)
                  & (base[:, 2] >= 0) & (base[:, 2] < nz - 1) & (self._mass > 0))
        # 初期の太さは放出源ごとに違う: σ0² = radius² を足す
        radius = np.array([e.radius for e in self.emitters] or [REFERENCE_RADIUS], dtype=float)
        sigma_all = np.sqrt(radius[self._emitter_flat] ** 2 + 2.0 * self.spread * self._ages_flat)
        lo, hi = float(sigma_all[inside].min()) if inside.any() else REFERENCE_RADIUS, float(sigma_all[inside].max()) if inside.any() else 1.0
        edges = np.geomspace(max(lo, 1e-6), max(hi, lo * (1 + 1e-6)), count + 1) if hi > lo * 1.02 else np.array([lo, hi + 1e-9])
        bands = []
        for k in range(len(edges) - 1):
            last = k == len(edges) - 2
            select = inside & (sigma_all >= edges[k]) & ((sigma_all <= edges[k + 1]) if last else (sigma_all < edges[k + 1]))
            sample = np.nonzero(select)[0]
            if len(sample) == 0:
                continue
            b, f = base[sample], frac[sample]
            flat = np.empty((len(sample), 8), dtype=np.int64)
            tri = np.empty((len(sample), 8), dtype=np.float32)
            corner = 0
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                for dy in (0, 1):
                    wy = f[:, 1] if dy else 1.0 - f[:, 1]
                    for dx in (0, 1):
                        wx = f[:, 0] if dx else 1.0 - f[:, 0]
                        flat[:, corner] = ((b[:, 2] + dz) * ny + (b[:, 1] + dy)) * nx + (b[:, 0] + dx)
                        tri[:, corner] = wx * wy * wz
                        corner += 1
            # 帯の代表の広がり: σ² の質量加重平均 (総量と 2 次モーメントを保つ)
            mass = self._mass[sample].astype(float)
            sigma = float(np.sqrt(np.average(sigma_all[sample] ** 2, weights=mass if mass.sum() > 0 else None)))
            # 三線形の配分だけで約 0.5 ボクセルの広がりが付くので、その分を差し引く
            sigma_voxels = float(np.sqrt(max((sigma / grid.spacing) ** 2 - 0.25, 0.0)))
            margin = int(np.ceil(BLUR_TRUNCATE * sigma_voxels)) + 1
            box = tuple(slice(max(int(b[:, axis].min()) - margin, 0), min(int(b[:, axis].max()) + 2 + margin, grid.dims[axis]))
                        for axis in (2, 1, 0))
            norm = None
            if self.mask is not None and sigma_voxels >= 0.3:
                norm = ndimage.gaussian_filter(self.mask[box], sigma_voxels, mode="nearest", truncate=BLUR_TRUNCATE)
            bands.append(_Band(sigma, sigma_voxels, sample, flat, tri, box, norm))
        return bands

    # ------------------------------------------------------------------ 寿命
    @property
    def lifetime(self) -> float:
        return self._lifetime

    @lifetime.setter
    def lifetime(self, value: float) -> None:
        """寿命の変更は濃度場の重みだけに効く (流跡線も格子も作り直さない)。"""
        value = float(value) if value and value > 0 else 0.0
        if value == self._lifetime and (self._fade is not None or value == 0.0):
            return
        self._lifetime = value
        if value > 0:
            self._fade = np.clip((value - self._ages_flat) / (0.2 * value), 0.0, 1.0).astype(np.float32)
        else:
            self._fade = None
        self.clear_cache()

    # ------------------------------------------------------------------ 視覚効果
    @property
    def noise(self) -> float:
        return self._noise

    @noise.setter
    def noise(self, amplitude: float) -> None:
        """視覚効果のゆらぎ (0 = なし)。CFD の結果ではない、表示だけの効果。

        煙の筋ごとに、流れに沿って濃淡を付ける (固定の乱数で決まるので、同じ時刻には同じ見え方)。
        濃淡は煙と一緒に流れ、流れの経路・速さは変えない。平均すると煙の総量は変わらない。
        """
        amplitude = min(max(float(amplitude), 0.0), 0.9)
        if amplitude == self._noise:
            return
        self._noise = amplitude
        if amplitude > 0 and self.tracks.n_tracks:
            rng = np.random.default_rng(20261117)
            phase = rng.uniform(0.0, 2.0 * np.pi, (self.tracks.n_tracks, 2))
            period = max(0.04 * self.natural_duration, 4.0 * self.tracks.dt)
            age = self.tracks.ages[None, :]
            wave = 0.6 * np.sin(2 * np.pi * age / period + phase[:, :1]) + 0.4 * np.sin(2 * np.pi * age / (0.37 * period) + phase[:, 1:])
            self._noise_factor = (1.0 + amplitude * wave).astype(np.float32).ravel()
        else:
            self._noise_factor = None
        self.clear_cache()

    # ------------------------------------------------------------------ 濃度場
    def emission_weights(self, time: float) -> np.ndarray:
        """時刻 ``time`` に各標本が持つ煙の量 (放出していなかった時刻の標本・寿命を過ぎた標本は 0)。"""
        emission_time = float(time) - self._ages_flat
        weights = np.zeros(len(self._mass), dtype=np.float32)
        for index, emitter in enumerate(self.emitters):
            owned = self._emitter_flat == index
            if owned.any():
                weights[owned] = self._mass[owned] * emitter.emission(emission_time[owned])
        if self._fade is not None:
            weights *= self._fade
        if self._noise_factor is not None:
            weights *= self._noise_factor
        return weights

    def _splat(self, weights: np.ndarray) -> np.ndarray:
        grid = self.grid
        out = np.zeros(grid.shape, dtype=np.float32)
        for band in self._bands:
            w = weights[band.sample]
            if not w.any():
                continue
            deposit = np.bincount(band.flat.ravel(), weights=(w[:, None] * band.tri).ravel(), minlength=grid.n_voxels)
            part = deposit.astype(np.float32).reshape(grid.shape)[band.box]
            if self.mask is not None:
                part = part * self.mask[band.box]
            if band.sigma_voxels >= 0.3:
                part = ndimage.gaussian_filter(part, band.sigma_voxels, mode="constant", truncate=BLUR_TRUNCATE)
                if band.norm is not None:
                    # 固体側へはみ出した分を流体側へ戻す (壁の近くで煙が薄くならないようにする)
                    part = part * self.mask[band.box] / np.maximum(band.norm, 0.05)
            out[band.box] += part
        return out

    def frame(self, time: float, with_speed: bool = False):
        """時刻 ``time`` [s] の濃度場 (nz, ny, nx)、float32。同じ時刻には必ず同じ結果を返す。

        ``with_speed`` なら (濃度, 流速) を返す。流速は、そのボクセルにある煙の流速の濃度加重平均 [m/s]
        (煙の無い所は 0)。色付け用で、濃度 (不透明度) とは別に扱う。
        """
        if not self._bands:
            empty = np.zeros(self.grid.shape, dtype=np.float32)
            return (empty, empty.copy()) if with_speed else empty
        weights = self.emission_weights(time)
        density = self._splat(weights)
        if not with_speed:
            return density
        flux = self._splat(weights * self._speed_flat)
        speed = np.divide(flux, density, out=np.zeros_like(density), where=density > 1e-6 * max(float(density.max()), 1e-30))
        return density, speed

    def frame_index(self, time: float, n_frames: int) -> int:
        if self.duration <= 0 or n_frames < 2:
            return 0
        return int(round(min(max(time / self.duration, 0.0), 1.0) * (n_frames - 1)))

    def frame_time(self, index: int, n_frames: int) -> float:
        return self.duration * int(index) / max(int(n_frames) - 1, 1)

    def peek_frame(self, index: int, n_frames: int, with_speed: bool = False):
        """キャッシュにあればそのコマ、無ければ ``None`` (計算しない)。"""
        key = (int(index), int(n_frames), bool(with_speed))
        with self._lock:
            item = self._cache.get(key)
            if item is not None:
                self._cache.move_to_end(key)
            return item

    def cached_frame(self, index: int, n_frames: int, with_speed: bool = False):
        """タイムラインを ``n_frames`` 等分したうちの ``index`` 番目の濃度場 (float16 で保持)。

        上限 (``cache_bytes``) までキャッシュし、超えたら最も長く使っていないものから捨てる。
        別スレッドからの先読みと同時に呼んでも安全。``with_speed`` なら (濃度, 流速) の組。
        """
        item = self.peek_frame(index, n_frames, with_speed)
        if item is not None:
            return item
        result = self.frame(self.frame_time(index, n_frames), with_speed)
        item = tuple(a.astype(np.float16) for a in result) if with_speed else result.astype(np.float16)
        key = (int(index), int(n_frames), bool(with_speed))
        with self._lock:
            self._cache[key] = item
            self._cache.move_to_end(key)
            used = sum(_nbytes(value) for value in self._cache.values())
            while used > self.cache_bytes and len(self._cache) > 1:
                _key, dropped = self._cache.popitem(last=False)
                used -= _nbytes(dropped)
        return item

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_info(self) -> dict:
        with self._lock:
            return {"frames": len(self._cache), "bytes": int(sum(_nbytes(v) for v in self._cache.values())), "limit": self.cache_bytes}

    def total_mass(self, density: np.ndarray) -> float:
        """濃度場に含まれる煙の総量 [m³ 相当 × 基準濃度]。検証用。"""
        return float(density.sum(dtype=np.float64) * self.grid.spacing ** 3)

    def expected_mass(self, time: float) -> float:
        """時刻 ``time`` に格子の中にあるはずの煙の総量 (格子へ配分する前の標本の合計)。検証用。"""
        weights = self.emission_weights(time)
        samples = np.concatenate([band.sample for band in self._bands]) if self._bands else np.zeros(0, dtype=np.int64)
        return float(weights[samples].sum(dtype=np.float64) * self.grid.spacing ** 3)


def _nbytes(item) -> int:
    return sum(a.nbytes for a in item) if isinstance(item, tuple) else int(item.nbytes)
