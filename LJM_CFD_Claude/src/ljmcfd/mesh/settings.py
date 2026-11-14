"""メッシュ設定とプリセット。寸法はすべてメートル。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

PRESETS = {
    # 名前: (表面サイズ = L / n, 表示名)
    "coarse": (50.0, "粗い (動作確認用)"),
    "standard": (90.0, "標準"),
    "fine": (140.0, "細かい"),
}
PRESET_CUSTOM = "custom"

ALGORITHM_HXT = "hxt"            # 並列 Delaunay (高速)
ALGORITHM_DELAUNAY = "delaunay"  # 逐次 Delaunay (境界層プリズム使用時は必須)


@dataclass
class BoundaryLayerSettings:
    """境界層プリズム層 (実験的機能)。"""

    enabled: bool = False
    layers: int = 5
    first_height: float = 0.001   # 第 1 層の厚さ [m]
    growth_ratio: float = 1.3

    def total_thickness(self) -> float:
        return sum(self.first_height * self.growth_ratio ** i for i in range(self.layers))

    def layer_heights(self) -> list[float]:
        """壁面からの累積高さ [m]。"""
        heights, total = [], 0.0
        for i in range(self.layers):
            total += self.first_height * self.growth_ratio ** i
            heights.append(total)
        return heights


@dataclass
class MeshSettings:
    preset: str = "standard"
    surface_size: float = 0.045        # 車体表面の要素サイズ [m]
    min_size: float = 0.009            # 曲率細分化で許す最小サイズ [m]
    far_size: float = 1.5              # 遠方の最大サイズ [m]
    refine_distance: float = 6.0       # 表面サイズ → 遠方サイズへ広げる距離 [m]
    curvature_elements: int = 24       # 曲率細分化: 円 1 周あたりの要素数 (0 で無効)
    wake_enabled: bool = True
    wake_size_factor: float = 2.5      # 後流領域の要素サイズ = 表面サイズ × この値
    wake_length: float = 6.0           # 後流細分化領域の長さ (モデル後端から) [m]
    part_size_factors: dict[str, float] = field(default_factory=dict)   # パーツ id → 表面サイズ倍率
    algorithm: str = ALGORITHM_HXT
    optimize: bool = True
    threads: int = 0                   # 0 = 自動 (CPU コア数)
    boundary_layer: BoundaryLayerSettings = field(default_factory=BoundaryLayerSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "MeshSettings":
        data = dict(data or {})
        settings = cls()
        bl = data.pop("boundary_layer", None) or {}
        for name, default in asdict(settings).items():
            if name == "boundary_layer" or name not in data:
                continue
            value = data[name]
            if name == "part_size_factors":
                value = {str(k): float(v) for k, v in (value or {}).items()}
            elif isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            setattr(settings, name, value)
        base = BoundaryLayerSettings()
        settings.boundary_layer = BoundaryLayerSettings(
            enabled=bool(bl.get("enabled", base.enabled)), layers=int(bl.get("layers", base.layers)),
            first_height=float(bl.get("first_height", base.first_height)),
            growth_ratio=float(bl.get("growth_ratio", base.growth_ratio)),
        )
        return settings

    def apply_preset(self, preset: str, model_bounds) -> None:
        """プリセットとモデル寸法から各サイズを決める。``custom`` では何も変えない。"""
        self.preset = preset
        if preset not in PRESETS:
            return
        length = max(model_bounds[1] - model_bounds[0], 1e-6)
        width = model_bounds[3] - model_bounds[2]
        height = model_bounds[5] - model_bounds[4]
        reference = max(length, width, height)
        self.surface_size = reference / PRESETS[preset][0]
        self.min_size = self.surface_size / 5.0
        self.far_size = reference / 3.0
        self.refine_distance = 1.5 * reference
        self.wake_length = 1.5 * length

    def validate(self) -> list[str]:
        """設定の矛盾を日本語メッセージで返す (空なら問題なし)。"""
        errors = []
        if not (0 < self.min_size <= self.surface_size <= self.far_size):
            errors.append("メッシュサイズは 最小 ≤ 表面 ≤ 遠方 かつ正の値にしてください。")
        if self.refine_distance <= 0:
            errors.append("細分化距離は正の値にしてください。")
        if self.curvature_elements < 0:
            errors.append("曲率細分化の要素数は 0 以上にしてください。")
        if self.wake_enabled and (self.wake_size_factor < 1.0 or self.wake_length <= 0):
            errors.append("後流領域のサイズ倍率は 1 以上、長さは正の値にしてください。")
        if any(f <= 0 for f in self.part_size_factors.values()):
            errors.append("パーツ別のサイズ倍率は正の値にしてください。")
        bl = self.boundary_layer
        if bl.enabled and (bl.layers < 1 or bl.first_height <= 0 or bl.growth_ratio < 1.0):
            errors.append("境界層は 層数 ≥ 1、第 1 層厚さ > 0、成長率 ≥ 1 にしてください。")
        return errors


def input_hash(payload: dict) -> str:
    """メッシュの入力 (形状・姿勢・風洞・設定) を表すハッシュ。メッシュが最新かどうかの判定に使う。"""
    def normalize(value):
        if isinstance(value, float):
            return round(value, 9)
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    text = json.dumps(normalize(payload), sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(text.encode("ascii")).hexdigest()[:16]
