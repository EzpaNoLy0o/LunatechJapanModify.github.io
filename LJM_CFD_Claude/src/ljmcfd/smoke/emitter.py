"""煙の放出源 (Smoke Emitter)。

放出源は「どこから・いつ・どれだけ煙を出すか」だけを決める。煙の動きは CFD の速度場に従い、
放出源が気流を変えることはない。座標はメートル、時刻は秒 (流れの時間)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

KIND_POINT, KIND_LINE, KIND_PLANE, KIND_GRID = "point", "line", "plane", "grid"
KINDS = (KIND_POINT, KIND_LINE, KIND_PLANE, KIND_GRID)
KIND_LABELS = {KIND_POINT: "点", KIND_LINE: "線 (ノズルの列)", KIND_PLANE: "面 (連続)", KIND_GRID: "格子 (ノズルの格子)"}
#: 面の放出源で、隣り合う開始点の間隔の目安 [m] (煙が 1 枚の連続した面に見える細かさ)
PLANE_SPACING = 0.05
MAX_NOZZLES = 2000


@dataclass
class Emitter:
    """X 軸に垂直な面内に置く放出源。``y``・``z`` が中心、``width`` が Y 方向、``height`` が Z 方向の広がり。"""

    name: str = "Emitter"
    kind: str = KIND_LINE
    enabled: bool = True
    x: float = 0.0
    y: float = 0.0
    z: float = 0.5
    width: float = 1.6          # Y 方向の長さ [m]
    height: float = 0.0         # Z 方向の長さ [m]
    n_y: int = 9                # Y 方向のノズル数 (線・格子)
    n_z: int = 1                # Z 方向のノズル数 (格子)。線では width・height の対角に n_y 個を並べる
    radius: float = 0.02        # ノズルの半径 = 煙の初期の太さ [m]
    rate: float = 1.0           # 放出量 (相対値。1 = 標準)
    start: float = 0.0          # 放出開始 [s]
    stop: float = -1.0          # 放出終了 [s]。負なら止めない
    pulse_period: float = 0.0   # 間欠放出の周期 [s]。0 なら連続放出
    pulse_duty: float = 0.5     # 間欠放出で、1 周期のうち放出している割合

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Emitter":
        emitter = cls()
        for name, default in asdict(emitter).items():
            if data and name in data:
                value = data[name]
                value = bool(value) if isinstance(default, bool) else int(value) if isinstance(default, int) else \
                    float(value) if isinstance(default, float) else str(value)
                setattr(emitter, name, value)
        if emitter.kind not in KINDS:
            emitter.kind = KIND_LINE
        return emitter

    # ------------------------------------------------------------------ 空間
    def nozzle_points(self) -> np.ndarray:
        """ノズル (煙の筋 1 本ずつの中心) の座標 (N, 3)。"""
        if self.kind == KIND_POINT:
            yz = np.array([[self.y, self.z]])
        elif self.kind == KIND_LINE:
            n = max(int(self.n_y), 1)
            s = np.linspace(-0.5, 0.5, n) if n > 1 else np.zeros(1)
            yz = np.column_stack([self.y + s * self.width, self.z + s * self.height])
        else:
            if self.kind == KIND_PLANE:
                n_y = max(int(round(self.width / PLANE_SPACING)) + 1, 1)
                n_z = max(int(round(self.height / PLANE_SPACING)) + 1, 1)
                scale = max((n_y * n_z / MAX_NOZZLES) ** 0.5, 1.0)       # 多すぎる場合は均等に間引く
                n_y, n_z = max(int(n_y / scale), 1), max(int(n_z / scale), 1)
            else:
                n_y, n_z = max(int(self.n_y), 1), max(int(self.n_z), 1)
            ys = self.y + (np.linspace(-0.5, 0.5, n_y) if n_y > 1 else np.zeros(1)) * self.width
            zs = self.z + (np.linspace(-0.5, 0.5, n_z) if n_z > 1 else np.zeros(1)) * self.height
            grid_y, grid_z = np.meshgrid(ys, zs)
            yz = np.column_stack([grid_y.ravel(), grid_z.ravel()])
        return np.column_stack([np.full(len(yz), self.x), yz])

    def bounds(self) -> tuple[float, ...]:
        points = self.nozzle_points()
        r = self.radius
        return (self.x, self.x, float(points[:, 1].min() - r), float(points[:, 1].max() + r),
                float(points[:, 2].min() - r), float(points[:, 2].max() + r))

    # ------------------------------------------------------------------ 時間
    def emission(self, emission_time: np.ndarray) -> np.ndarray:
        """放出時刻 ``emission_time`` [s] における放出の強さ (0〜1)。開始前・終了後・間欠の休止中は 0。"""
        tau = np.asarray(emission_time, dtype=float)
        on = tau >= self.start
        if self.stop >= 0.0:
            on &= tau <= self.stop
        if self.pulse_period > 0.0:
            phase = np.mod(tau - self.start, self.pulse_period) / self.pulse_period
            on &= phase < min(max(self.pulse_duty, 0.0), 1.0)
        return on.astype(np.float32)


# ----------------------------------------------------------------------------- 実車向けの配置

PRESET_FRONT, PRESET_UNDERFLOOR, PRESET_DIFFUSER, PRESET_WING, PRESET_WHEEL, PRESET_CUSTOM = (
    "front", "underfloor", "diffuser", "rear_wing", "wheel_wake", "custom")
PRESET_LABELS = {
    PRESET_FRONT: "Front Flow (車両前方)", PRESET_UNDERFLOOR: "Underfloor Flow (床下)", PRESET_DIFFUSER: "Diffuser Flow (床下後部)",
    PRESET_WING: "Rear Wing Flow (リアウイング前方)", PRESET_WHEEL: "Wheel Wake (タイヤまわり)", PRESET_CUSTOM: "Custom (自由配置)",
}


def preset_emitters(preset: str, model_bounds, domain=None) -> list[Emitter]:
    """車両の寸法から放出源を配置する。配置を決めるだけで、気流は変えない。

    車両の細部 (ウイングやディフューザーの正確な位置) は形状から分からないので、車体の外形寸法に対する
    比率で置く。位置は GUI で調整できる。
    """
    x0, x1, y0, y1, z0, z1 = model_bounds
    length, width, height = x1 - x0, y1 - y0, z1 - z0
    yc = 0.5 * (y0 + y1)
    upstream = x0 - 0.25 * length
    if domain is not None:
        upstream = max(upstream, domain[0] + 0.02 * (domain[1] - domain[0]))

    def make(name, **values) -> Emitter:
        return Emitter(name=name, **values)

    if preset == PRESET_FRONT:
        return [make("Front Flow", kind=KIND_LINE, x=upstream, y=yc, z=z0 + 0.45 * height, width=0.9 * width, height=0.0, n_y=11)]
    if preset == PRESET_UNDERFLOOR:
        return [make("Underfloor Flow", kind=KIND_LINE, x=upstream, y=yc, z=z0 + 0.06 * height, width=0.8 * width, height=0.0, n_y=9)]
    if preset == PRESET_DIFFUSER:
        return [make("Diffuser Flow", kind=KIND_LINE, x=x0 + 0.55 * length, y=yc, z=z0 + 0.05 * height, width=0.7 * width, height=0.0,
                     n_y=9, radius=0.015)]
    if preset == PRESET_WING:
        return [make("Rear Wing Flow", kind=KIND_GRID, x=x0 + 0.55 * length, y=yc, z=z0 + 1.08 * height, width=0.7 * width,
                     height=0.25 * height, n_y=7, n_z=3)]
    if preset == PRESET_WHEEL:
        return [make(f"Wheel Wake {side}", kind=KIND_GRID, x=upstream, y=yc + sign * 0.45 * width, z=z0 + 0.2 * height,
                     width=0.2 * width, height=0.3 * height, n_y=3, n_z=4) for side, sign in (("L", 1.0), ("R", -1.0))]
    return [make("Custom", kind=KIND_LINE, x=upstream, y=yc, z=z0 + 0.5 * height, width=0.6 * width, height=0.0, n_y=7)]
