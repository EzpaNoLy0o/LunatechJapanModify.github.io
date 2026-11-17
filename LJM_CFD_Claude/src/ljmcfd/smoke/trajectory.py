"""流跡線のキャッシュ (Level 2): 放出源のノズルから、計算済みの速度場に沿って煙の通り道を求める。

定常の速度場では、ある点から連続して放出した煙の列 (流脈線) は、その点を通る流跡線・流線と一致する。
そこで「ノズルごとの流跡線を経過時間で等間隔に標本化した配列」を持てば、時刻 t の煙は
「経過時間 a ≤ t の標本のうち、放出時刻 t − a に放出していたもの」として、積分し直さずに求まる。
同じ時刻には必ず同じ結果になるので、タイムラインを戻せば正確に元へ戻る。

速度の補間は VTK の流線積分 (非構造メッシュのセル内補間、RK45 の適応刻み) を使う。
流体のセルが無い場所 (車体の内部・領域の外) へは進まず、そこで軌跡が終わる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk

from .emitter import Emitter


def trace_pathlines(flow, points: np.ndarray, max_length: float, terminal_speed: float = 1e-4) -> pv.PolyData:
    """``points`` から速度場 ``flow`` (節点データ ``U`` を持つ) に沿って前向きに積分した流跡線。"""
    tracer = vtk.vtkStreamTracer()
    tracer.SetInputData(flow)
    tracer.SetSourceData(pv.PolyData(np.asarray(points, dtype=float)))
    tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U")
    tracer.SetIntegratorTypeToRungeKutta45()
    tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.CELL_LENGTH_UNIT)
    tracer.SetInitialIntegrationStep(0.2)
    tracer.SetMinimumIntegrationStep(0.01)
    tracer.SetMaximumIntegrationStep(0.5)
    tracer.SetMaximumNumberOfSteps(8000)
    tracer.SetMaximumPropagation(float(max_length))
    tracer.SetTerminalSpeed(float(terminal_speed))
    tracer.SetComputeVorticity(False)
    tracer.SetInterpolatorTypeToCellLocator()
    tracer.SetIntegrationDirectionToForward()
    tracer.Update()
    return pv.wrap(tracer.GetOutput())


def nozzle_seeds(emitter: Emitter, seeds_per_nozzle: int) -> tuple[np.ndarray, np.ndarray]:
    """ノズルごとに、半径内へ決まった配置で開始点を置く。戻り値は (座標 (N, 3), 1 点あたりの重み (N,))。

    1 つのノズルを複数の開始点で表すと、煙の筋に太さができ、流れが分かれる所では筋も分かれる。
    配置は乱数ではなく固定の渦巻き配置 (同じ設定なら必ず同じ結果)。重みの合計はノズル 1 つにつき 1。
    """
    nozzles = emitter.nozzle_points()
    count = max(int(seeds_per_nozzle), 1)
    if count == 1 or emitter.radius <= 0:
        return nozzles, np.ones(len(nozzles), dtype=np.float32)
    index = np.arange(count)
    radius = emitter.radius * np.sqrt((index + 0.5) / count)        # 円板を面積が等しくなるように埋める
    angle = index * 2.399963229728653                                # 黄金角
    offsets = np.column_stack([np.zeros(count), radius * np.cos(angle), radius * np.sin(angle)])
    points = (nozzles[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return points, np.full(len(points), 1.0 / count, dtype=np.float32)


@dataclass
class Tracks:
    """流跡線を経過時間で等間隔に標本化したもの。

    positions: (L, n, 3)  標本の座標。軌跡が終わった後は終点の値が入る
    speeds:    (L, n)     標本位置の流速 [m/s]
    end_age:   (L,)       軌跡が終わる経過時間 [s] (これより後の標本は使わない)
    weights:   (L,)       軌跡 1 本が運ぶ煙の量の比 (放出量 × ノズル内の重み)
    emitter:   (L,)       どの放出源のものか (``emitters`` の添字)
    dt:        標本の時間間隔 [s]
    """

    positions: np.ndarray
    speeds: np.ndarray
    end_age: np.ndarray
    weights: np.ndarray
    emitter: np.ndarray
    dt: float

    @property
    def n_tracks(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.positions.shape[1])

    @property
    def ages(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt

    def nbytes(self) -> int:
        return int(self.positions.nbytes + self.speeds.nbytes)


def build_tracks(flow, emitters: list[Emitter], *, dt: float, max_age: float, seeds_per_nozzle: int = 4,
                 max_length: float | None = None, terminal_speed: float = 1e-4) -> Tracks:
    """有効な放出源すべての流跡線を求め、時間間隔 ``dt`` で ``max_age`` まで標本化する。"""
    n = max(int(np.ceil(max_age / dt)) + 1, 2)
    ages = np.arange(n) * dt
    if max_length is None:
        bounds = flow.bounds
        max_length = 2.0 * float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
    positions, speeds, end_age, weights, owner = [], [], [], [], []
    for emitter_index, emitter in enumerate(emitters):
        if not emitter.enabled or emitter.rate <= 0:
            continue
        seeds, seed_weights = nozzle_seeds(emitter, seeds_per_nozzle)
        lines = trace_pathlines(flow, seeds, max_length, terminal_speed)
        if lines.n_points == 0 or "IntegrationTime" not in lines.point_data:
            continue
        time_all = np.asarray(lines.point_data["IntegrationTime"], dtype=float)
        points_all = np.asarray(lines.points, dtype=float)
        velocity = np.asarray(lines.point_data["U"], dtype=float)
        speed_all = np.linalg.norm(velocity, axis=1)
        seed_ids = np.asarray(lines.cell_data["SeedIds"]) if "SeedIds" in lines.cell_data else np.arange(lines.n_cells)
        connectivity = np.asarray(lines.lines)
        offset = 0
        for cell in range(lines.n_cells):
            count = int(connectivity[offset])
            ids = connectivity[offset + 1:offset + 1 + count]
            offset += count + 1
            time = time_all[ids] - time_all[ids[0]]
            keep = np.concatenate([[True], np.diff(time) > 0])           # np.interp は単調増加が前提
            if keep.sum() < 2:
                continue
            time, points, speed = time[keep], points_all[ids][keep], speed_all[ids][keep]
            track = np.empty((n, 3))
            for axis in range(3):
                track[:, axis] = np.interp(ages, time, points[:, axis])  # 終点より後は終点の値
            positions.append(track)
            speeds.append(np.interp(ages, time, speed))
            end_age.append(min(time[-1], ages[-1]))
            weights.append(emitter.rate * seed_weights[int(seed_ids[cell])])
            owner.append(emitter_index)
    if not positions:
        return Tracks(np.zeros((0, n, 3), np.float32), np.zeros((0, n), np.float32), np.zeros(0), np.zeros(0, np.float32),
                      np.zeros(0, np.int64), float(dt))
    return Tracks(np.asarray(positions, dtype=np.float32), np.asarray(speeds, dtype=np.float32), np.asarray(end_age),
                  np.asarray(weights, dtype=np.float32), np.asarray(owner, dtype=np.int64), float(dt))
