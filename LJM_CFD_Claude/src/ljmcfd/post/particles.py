"""粒子・スモークのアニメーション用データ。

流跡線をあらかじめ等間隔の「経過時間」で標本化して配列に持つ。ある時刻の粒子位置は
配列の添字計算だけで求まるので、タイムラインを前へも後ろへも即座に動かせる
(スライダー操作のたびに流線を積分し直したり、OpenFOAM を実行したりしない)。

現在の結果は定常 RANS なので、粒子は **定常の速度場に沿った軌跡** を動く
(``SteadyPathlines``)。これは非定常解析の時刻歴ではない。将来、非定常解析に対応する場合は、
時刻ごとの速度場から粒子を進める別の ``AnimationSource`` を追加すれば、GUI 側は変更なしで再生できる。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pyvista as pv


@dataclass
class ParticleFrame:
    """ある時刻の粒子群。配列の長さは時刻によらず一定 (表示側は中身の書き換えだけで済む)。"""

    points: np.ndarray      # (N, 3)  見えない粒子にも有限の座標が入る (軌跡の始点・終点に留まる)
    speed: np.ndarray       # (N,)  粒子位置の流速 [m/s]
    fade: np.ndarray        # (N,)  1 = 粒子の先頭、0 に近いほど尾 (スモークの薄れ)
    visible: np.ndarray     # (N,)  bool。まだ放出されていない・軌跡が終わった粒子は False


class AnimationSource(ABC):
    """粒子アニメーションのデータ源。定常・非定常の違いをここで吸収する。"""

    #: True = 定常の速度場に沿った軌跡 (非定常解析の結果ではない)。GUI はこの旨を必ず表示する。
    is_steady: bool = True

    @property
    @abstractmethod
    def duration(self) -> float:
        """タイムラインの長さ [s] (流れの時間)。"""

    @abstractmethod
    def frame(self, time: float, loop: bool) -> ParticleFrame:
        """時刻 ``time`` の粒子群。同じ時刻には必ず同じ結果を返す (巻き戻しが正確に戻る)。"""


class SteadyPathlines(AnimationSource):
    """定常の速度場の流線に沿って粒子を流す。

    各流線から一定間隔で粒子を放出する。粒子 k の放出時刻は ``k · duration / particles_per_line``。
    ループ再生では経過時間を ``duration`` で折り返すので、切れ目なく流れ続ける。
    """

    is_steady = True

    def __init__(self, lines: pv.PolyData, particles_per_line: int = 8, trail_points: int = 1,
                 trail_fraction: float = 0.04, n_steps: int = 300) -> None:
        """
        lines:              流線 (``IntegrationTime`` と ``Umag`` を節点に持つ。前向きに積分したもの)
        particles_per_line: 流線 1 本あたりの粒子数 (粒子密度)
        trail_points:       粒子 1 個を何点で描くか。1 = 点、2 以上 = 尾を引くスモーク
        trail_fraction:     尾の長さ (タイムライン全体に対する比)
        n_steps:            経過時間の標本数
        """
        self.particles_per_line = max(int(particles_per_line), 1)
        self.trail_points = max(int(trail_points), 1)
        self.n_steps = max(int(n_steps), 2)
        tracks = self._tracks(lines)
        self.n_lines = len(tracks)
        if self.n_lines == 0:
            self._duration = 0.0
            self._positions = np.zeros((0, self.n_steps, 3))
            self._speeds = np.zeros((0, self.n_steps))
            self._end_age = np.zeros(0)
        else:
            ends = np.array([t[0][-1] for t in tracks])
            # 後流に捕まって極端に長い時間を回り続ける流線にタイムライン全体を引き延ばされないよう、
            # 80 % の流線が終点まで到達する時間をタイムラインの長さにする
            self._duration = float(max(np.percentile(ends, 80), 1e-9))
            ages = np.linspace(0.0, self._duration, self.n_steps)
            self._positions = np.empty((self.n_lines, self.n_steps, 3))
            self._speeds = np.empty((self.n_lines, self.n_steps))
            for i, (time, points, speed) in enumerate(tracks):
                # 軌跡の終点より先の経過時間には終点の値が入る (np.interp の既定)。
                # 終点を過ぎた粒子は「見えない」扱いにするが、座標は終点に留めておく
                # → 尾を線で描くときに、消えた先頭へ向かって不自然な線が伸びない
                for axis in range(3):
                    self._positions[i, :, axis] = np.interp(ages, time, points[:, axis])
                self._speeds[i] = np.interp(ages, time, speed)
            self._end_age = np.minimum(ends, self._duration)
        self._release = np.arange(self.particles_per_line) * (self._duration / self.particles_per_line)
        self._trail_dt = trail_fraction * self._duration / max(self.trail_points - 1, 1)
        self._fade = np.linspace(1.0, 1.0 / self.trail_points, self.trail_points)

    @staticmethod
    def _tracks(lines: pv.PolyData) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """流線ごとの (経過時間, 座標, 流速)。経過時間が単調増加でない点や 1 点だけの線は除く。"""
        if lines is None or lines.n_points == 0 or "IntegrationTime" not in lines.point_data:
            return []
        time_all = np.asarray(lines.point_data["IntegrationTime"], dtype=float)
        points_all = np.asarray(lines.points, dtype=float)
        if "Umag" in lines.point_data:
            speed_all = np.asarray(lines.point_data["Umag"], dtype=float)
        else:
            speed_all = np.zeros(len(points_all))
        connectivity = np.asarray(lines.lines)
        tracks = []
        offset = 0
        while offset < len(connectivity):
            count = int(connectivity[offset])
            ids = connectivity[offset + 1:offset + 1 + count]
            offset += count + 1
            time = time_all[ids] - time_all[ids[0]]
            keep = np.concatenate([[True], np.diff(time) > 0])       # np.interp は単調増加が前提
            if keep.sum() < 2:
                continue
            tracks.append((time[keep], points_all[ids][keep], speed_all[ids][keep]))
        return tracks

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def n_particles(self) -> int:
        """描画する点の総数 (流線数 × 1 本あたりの粒子数 × 尾の点数)。"""
        return self.n_lines * self.particles_per_line * self.trail_points

    def trail_connectivity(self) -> np.ndarray:
        """粒子ごとの尾を 1 本の折れ線として結ぶ接続情報 (VTK の lines 形式)。``frame().points`` の並びに対応する。"""
        m = self.trail_points
        count = self.n_lines * self.particles_per_line
        if m < 2 or count == 0:
            return np.zeros(0, dtype=np.int64)
        cells = np.empty((count, m + 1), dtype=np.int64)
        cells[:, 0] = m
        cells[:, 1:] = np.arange(count * m, dtype=np.int64).reshape(count, m)
        return cells.ravel()

    def frame(self, time: float, loop: bool) -> ParticleFrame:
        n = self.n_particles
        if n == 0 or self._duration <= 0:
            empty = np.zeros(0)
            return ParticleFrame(np.zeros((0, 3)), empty, empty, np.zeros(0, dtype=bool))
        # 粒子ごと・尾の点ごとの経過時間 (K, M)
        head = float(time) - self._release                    # 粒子の先頭の経過時間 (K,)
        if loop:
            head = np.mod(head, self._duration)               # 終点まで行った粒子は開始点から流れ直す
        age = head[:, None] - self._trail_dt * np.arange(self.trail_points)[None, :]
        valid_age = (age >= 0.0) & (age <= self._duration)
        position = np.clip(age, 0.0, self._duration) / self._duration * (self.n_steps - 1)
        lower = np.minimum(np.floor(position).astype(np.int64), self.n_steps - 2)
        weight = position - lower

        # 標本間を線形補間 (ゆっくり再生しても粒子が滑らかに動く)
        p0 = self._positions[:, lower, :]               # (L, K, M, 3)
        p1 = self._positions[:, lower + 1, :]
        points = p0 + (p1 - p0) * weight[None, :, :, None]
        s0 = self._speeds[:, lower]
        s1 = self._speeds[:, lower + 1]
        speed = s0 + (s1 - s0) * weight[None, :, :]

        # まだ放出されていない粒子 (経過時間 < 0) は始点に、軌跡の終点を過ぎた粒子は終点に留まる
        visible = valid_age[None, :, :] & (age[None, :, :] <= self._end_age[:, None, None])
        speed = np.where(visible, speed, 0.0)
        fade = np.broadcast_to(self._fade[None, None, :], visible.shape)
        return ParticleFrame(points.reshape(n, 3), speed.reshape(n), fade.reshape(n).copy(), visible.reshape(n))
