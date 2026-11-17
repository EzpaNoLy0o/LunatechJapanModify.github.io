"""受動スカラー方式の煙 (Phase D): 煙を濃度 c として、計算済みの速度場で移流・拡散させる。

    ∂c/∂t + ∇·(U c) = ∇·(D_eff ∇c) + S − λ c

定常 RANS の速度場 U を使う **可視化用** の移流拡散モデルで、OpenFOAM の解析ではない。
U は時間変化しないので、渦の放出などの非定常な現象は現れない。

離散化 (煙の格子 = ``density.SmokeGrid`` の上):

* 移流  … セミラグランジュ法。格子点から速度場を RK2 で Δt だけ遡り、その位置の濃度を三線形補間で取る。
           U が時間変化しないので、遡った位置は最初に 1 回だけ計算して使い回す。無条件に安定だが、
           補間による数値拡散がある (``numerical_diffusion`` に目安を記録する)。
* 拡散  … 1 ステップごとに、分散 2·D_eff·Δt のガウスぼかし (自由空間の拡散方程式の厳密解)。
           固体の近くでは正規化畳み込みで、固体へ入った分を流体側へ戻す。
* 生成  … 放出源のノズルに、ステップごとに S·Δt を足す。放出の強さの単位は流跡線方式と同じ。
* 減衰  … exp(−λ Δt) を掛ける (寿命を指定したとき)。
* 境界  … 固体 (流体マスク 0) では濃度 0。格子の外から遡った値は 0 (流入する空気はきれい、出口からは流出する)。

時刻は決まった刻みで先頭から順に進めるので、同じ時刻には必ず同じ結果になる。途中の時刻へ飛ぶときは、
保存しておいた一番近いキーフレームから決まった手順で計算し直す (逆向きに解くことはしない)。
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import pyvista as pv
import vtk
from scipy import ndimage

from .density import REFERENCE_RADIUS, SmokeGrid, _nbytes
from .trajectory import nozzle_seeds

KEYFRAME_EVERY = 12          # 何コマごとに状態を保存するか (途中の時刻へ飛ぶときの出発点)
TARGET_CFL = 1.5             # 1 ステップで進む距離 (ボクセル幅に対する比) の目安


def velocity_on_grid(flow, grid: SmokeGrid) -> tuple[np.ndarray, np.ndarray]:
    """格子点での速度 (nz, ny, nx, 3) と流体マスク (nz, ny, nx)。流体のセルの外 (固体・領域外) の速度は 0。"""
    probe = vtk.vtkResampleWithDataSet()
    probe.SetInputData(grid.image())
    probe.SetSourceData(flow)
    probe.Update()
    output = pv.wrap(probe.GetOutput())
    mask = np.asarray(output.point_data["vtkValidPointMask"]).reshape(grid.shape).astype(np.float32)
    velocity = np.asarray(output.point_data["U"], dtype=np.float32).reshape(grid.shape + (3,))
    velocity *= mask[..., None]
    return velocity, mask


class ScalarSmoke:
    """受動スカラー方式の煙。``PathlineSmoke`` と同じ使い方 (``frame``・``cached_frame`` など) ができる。"""

    is_steady = True
    #: コマは先頭から順にしか求まらない (途中の時刻は、保存した状態からの計算が必要)
    sequential = True

    def __init__(self, grid: SmokeGrid, velocity: np.ndarray, mask: np.ndarray, emitters: list, *, free_stream_speed: float,
                 duration: float, diffusion: float = 0.02, lifetime: float = 0.0, n_frames: int = 240,
                 cache_bytes: int = 256 * 1024 * 1024, seeds_per_nozzle: int = 6) -> None:
        self.grid = grid
        self.velocity = velocity.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.emitters = emitters
        self.free_stream_speed = float(max(free_stream_speed, 1e-9))
        self.diffusion = float(max(diffusion, 0.0))
        self.n_frames = int(n_frames)
        self.cache_bytes = int(cache_bytes)
        self.natural_duration = float(duration)
        self.duration = float(duration)
        self.preview = None
        self.noise = 0.0                 # 受動スカラー方式では使わない (流跡線方式だけの視覚効果)
        self._lifetime = float(lifetime)
        self._lock = threading.RLock()          # キャッシュの出し入れ (短時間)
        self._sim_lock = threading.Lock()       # 時間発展 (長時間。キャッシュの参照を待たせない)
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._keyframes: dict[int, np.ndarray] = {}
        self._last: tuple[int, np.ndarray] | None = None
        self._speed = np.linalg.norm(self.velocity, axis=-1).astype(np.float32)
        self._seeds, self._seed_weights, self._seed_owner = self._sources(seeds_per_nozzle)
        self._setup_steps()

    # ------------------------------------------------------------------ 準備
    def _sources(self, seeds_per_nozzle: int):
        points, weights, owner = [], [], []
        for index, emitter in enumerate(self.emitters):
            if not emitter.enabled or emitter.rate <= 0:
                continue
            seeds, w = nozzle_seeds(emitter, seeds_per_nozzle)
            points.append(seeds)
            weights.append(w * emitter.rate)
            owner.append(np.full(len(seeds), index))
        if not points:
            return np.zeros((0, 3)), np.zeros(0, np.float32), np.zeros(0, np.int64)
        return np.vstack(points), np.concatenate(weights).astype(np.float32), np.concatenate(owner)

    def _setup_steps(self) -> None:
        """時間刻み (CFL の目安から決める) と、遡った位置 (使い回す) を用意する。"""
        h = self.grid.spacing
        max_speed = float(self._speed.max()) if self._speed.size else 0.0
        frame_dt = self.duration / max(self.n_frames - 1, 1)
        if max_speed > 0 and frame_dt > 0:
            self.substeps = max(int(np.ceil(frame_dt * max_speed / (TARGET_CFL * h))), 1)
        else:
            self.substeps = 1
        self.dt = frame_dt / self.substeps if frame_dt > 0 else 0.0
        self.cfl = max_speed * self.dt / h                    # 1 ステップでボクセルを何個ぶん進むか
        # 移流: 格子点から RK2 で遡った位置 (ボクセル座標)
        nz, ny, nx = self.grid.shape
        index = np.indices((nz, ny, nx), dtype=np.float32)    # (3, nz, ny, nx) の順は z, y, x
        u = self.velocity
        step = np.float32(self.dt / h)
        vel = np.stack([u[..., 2], u[..., 1], u[..., 0]])     # z, y, x の順にそろえる
        mid = index - 0.5 * step * vel
        vel_mid = np.stack([ndimage.map_coordinates(vel[k], mid, order=1, mode="nearest") for k in range(3)])
        self._back = (index - step * vel_mid).astype(np.float32)
        # 拡散: 1 ステップぶんのガウスぼかし
        self._sigma = float(np.sqrt(2.0 * self.diffusion * self.dt) / h) if self.dt > 0 else 0.0
        self._mask_blur = ndimage.gaussian_filter(self.mask, self._sigma, mode="nearest") if self._sigma >= 0.3 else None
        # 生成: ノズルの開始点を三線形で格子へ配分する (単位: 流跡線方式と同じ「基準濃度 × 体積 / 秒」)
        unit = self.free_stream_speed * np.pi * REFERENCE_RADIUS ** 2
        self._source_index, self._source_weight = self._trilinear(self._seeds)
        self._source_rate = (self._seed_weights[:, None] * self._source_weight) * np.float32(unit / h ** 3)
        # 数値拡散の目安: 線形補間のセミラグランジュ法で 1 ステップに付く分散 h²·α(1−α) (α は CFL の端数) を拡散係数に換算
        alpha = self.cfl - np.floor(self.cfl)
        self.numerical_diffusion = float(h * h * alpha * (1 - alpha) / (2 * self.dt)) if self.dt > 0 else 0.0
        self._decay = float(np.exp(-3.0 * self.dt / self._lifetime)) if self._lifetime > 0 and self.dt > 0 else 1.0

    def _trilinear(self, points: np.ndarray):
        h = self.grid.spacing
        nx, ny, nz = self.grid.dims
        coordinates = (np.asarray(points, dtype=float) - np.asarray(self.grid.origin)) / h
        base = np.floor(coordinates).astype(np.int64)
        frac = coordinates - base
        flat = np.zeros((len(points), 8), dtype=np.int64)
        weight = np.zeros((len(points), 8), dtype=np.float32)
        corner = 0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    i, j, k = base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz
                    w = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1]) * (frac[:, 2] if dz else 1 - frac[:, 2])
                    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
                    flat[:, corner] = np.where(inside, (k * ny + j) * nx + i, 0)
                    weight[:, corner] = np.where(inside, w, 0.0)
                    corner += 1
        return flat, weight

    # ------------------------------------------------------------------ 寿命
    @property
    def lifetime(self) -> float:
        return self._lifetime

    @lifetime.setter
    def lifetime(self, value: float) -> None:
        value = float(value) if value and value > 0 else 0.0
        if value != self._lifetime:
            self._lifetime = value
            self._decay = float(np.exp(-3.0 * self.dt / value)) if value > 0 and self.dt > 0 else 1.0
            self.clear_cache()

    # ------------------------------------------------------------------ 時間発展
    def _source(self, time: float) -> np.ndarray:
        strength = np.zeros(len(self._seeds), dtype=np.float32)
        for index, emitter in enumerate(self.emitters):
            owned = self._seed_owner == index
            if owned.any():
                strength[owned] = emitter.emission(np.array([time]))[0]
        rates = (strength[:, None] * self._source_rate).ravel()
        return np.bincount(self._source_index.ravel(), weights=rates, minlength=self.grid.n_voxels).astype(np.float32).reshape(self.grid.shape)

    def step(self, c: np.ndarray, time: float) -> np.ndarray:
        """1 ステップ (時刻 ``time`` → ``time + dt``) 進めた濃度場。"""
        c = ndimage.map_coordinates(c, self._back, order=1, mode="constant", cval=0.0).astype(np.float32)
        if self._sigma >= 0.3:
            c = ndimage.gaussian_filter(c * self.mask, self._sigma, mode="constant") * self.mask / np.maximum(self._mask_blur, 0.05)
        c *= self.mask
        if self._decay != 1.0:
            c *= np.float32(self._decay)
        c += self._source(time + 0.5 * self.dt) * np.float32(self.dt) * self.mask
        return c

    def _simulate_to(self, index: int) -> np.ndarray:
        """コマ ``index`` の濃度場。保存した状態のうち一番近い手前のものから、決まった手順で計算する。"""
        with self._sim_lock:
            start, state = 0, np.zeros(self.grid.shape, dtype=np.float32)
            candidates = [k for k in self._keyframes if k <= index]
            if candidates:
                start = max(candidates)
                state = self._keyframes[start].copy()
            if self._last is not None and start <= self._last[0] <= index:
                start, state = self._last[0], self._last[1].copy()
            for frame in range(start, index):
                for sub in range(self.substeps):
                    state = self.step(state, (frame * self.substeps + sub) * self.dt)
                if (frame + 1) % KEYFRAME_EVERY == 0 and (frame + 1) not in self._keyframes:
                    self._keyframes[frame + 1] = state.copy()      # 単精度で持つ (再開しても最初から計算したのと同じ値)
            self._last = (index, state.copy())
            return state

    # ------------------------------------------------------------------ PathlineSmoke と同じ使い方
    def frame_index(self, time: float, n_frames: int) -> int:
        if self.duration <= 0 or n_frames < 2:
            return 0
        return int(round(min(max(time / self.duration, 0.0), 1.0) * (n_frames - 1)))

    def frame(self, time: float, with_speed: bool = False):
        """時刻 ``time`` に一番近いコマの濃度場 (受動スカラー方式は決まった刻みで進めるため)。"""
        density = self._simulate_to(self.frame_index(time, self.n_frames))
        if with_speed:
            return density, np.where(density > 0, self._speed, 0.0).astype(np.float32)
        return density

    def peek_frame(self, index: int, n_frames: int, with_speed: bool = False):
        with self._lock:
            item = self._cache.get((int(index), bool(with_speed)))
            if item is not None:
                self._cache.move_to_end((int(index), bool(with_speed)))
            return item

    def cached_frame(self, index: int, n_frames: int, with_speed: bool = False):
        index = int(round(index * (self.n_frames - 1) / max(n_frames - 1, 1)))
        item = self.peek_frame(index, self.n_frames, with_speed)
        if item is not None:
            return item
        density = self._simulate_to(index)
        if with_speed:
            item = (density.astype(np.float16), np.where(density > 0, self._speed, 0.0).astype(np.float16))
        else:
            item = density.astype(np.float16)
        with self._lock:
            self._cache[(index, bool(with_speed))] = item
            used = sum(_nbytes(v) for v in self._cache.values())
            while used > self.cache_bytes and len(self._cache) > 1:
                _key, dropped = self._cache.popitem(last=False)
                used -= _nbytes(dropped)
        return item

    def clear_cache(self) -> None:
        with self._sim_lock, self._lock:
            self._cache.clear()
            self._keyframes.clear()
            self._last = None

    def cache_info(self) -> dict:
        with self._lock:
            used = sum(_nbytes(v) for v in self._cache.values()) + sum(v.nbytes for v in self._keyframes.values())
            return {"frames": len(self._cache), "bytes": int(used), "limit": self.cache_bytes, "keyframes": len(self._keyframes)}

    def total_mass(self, density: np.ndarray) -> float:
        return float(density.sum(dtype=np.float64) * self.grid.spacing ** 3)

    def emitted_mass(self, time: float) -> float:
        """時刻 ``time`` までに放出した煙の総量 (検証用。格子の外へ出た分・減衰した分も含む)。"""
        unit = self.free_stream_speed * np.pi * REFERENCE_RADIUS ** 2
        total = 0.0
        frames = self.frame_index(time, self.n_frames)
        for frame in range(frames):
            for sub in range(self.substeps):
                t = (frame * self.substeps + sub + 0.5) * self.dt
                for index, emitter in enumerate(self.emitters):
                    owned = self._seed_owner == index
                    if owned.any():
                        total += float(emitter.emission(np.array([t]))[0] * self._seed_weights[owned].sum()) * unit * self.dt
        return total
