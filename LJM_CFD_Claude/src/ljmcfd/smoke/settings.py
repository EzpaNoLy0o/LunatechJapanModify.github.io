"""煙の設定 (プロジェクトに保存する)。

どの設定がどのキャッシュに効くか (``engine.py`` の無効化の規則):

* 放出源・開始点の数・品質 (格子)・時間刻み     → Level 2 (流跡線) から作り直す
* 広がり                                         → Level 3 (密度場の準備) から作り直す
* 寿命                                           → Level 3 のコマだけ捨てる
* 色・濃さ・不透明度・ライティング・描画方式     → Level 4 (描画) だけ
* カメラの回転・ズーム                           → 何も作り直さない
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .emitter import Emitter

COLOR_WHITE, COLOR_LIGHT, COLOR_DARK, COLOR_VELOCITY, COLOR_CUSTOM = "white", "light_gray", "dark_gray", "velocity", "custom"
COLOR_LABELS = {
    COLOR_WHITE: "White Smoke (白)", COLOR_LIGHT: "Light Gray Smoke (明るい灰色)", COLOR_DARK: "Dark Gray Smoke (濃い灰色)",
    COLOR_VELOCITY: "Velocity Colored (流速で色付け)", COLOR_CUSTOM: "Custom Color (任意の色)",
}
COLOR_RGB = {COLOR_WHITE: (0.97, 0.97, 0.97), COLOR_LIGHT: (0.80, 0.81, 0.82), COLOR_DARK: (0.42, 0.43, 0.45)}
BACKEND_AUTO, BACKEND_GPU, BACKEND_CPU = "auto", "gpu", "cpu"
BACKEND_LABELS = {BACKEND_AUTO: "自動", BACKEND_GPU: "GPU (レイキャスティング)", BACKEND_CPU: "CPU (レイキャスティング)"}
TRANSPORT_PATHLINE, TRANSPORT_SCALAR = "pathline", "scalar"
TRANSPORT_LABELS = {TRANSPORT_PATHLINE: "流跡線方式 (Pathline-Based)", TRANSPORT_SCALAR: "受動スカラー方式 (Passive Scalar)"}
QUALITIES = ("low", "medium", "high", "ultra")


@dataclass
class SmokeSettings:
    emitters: list[Emitter] = field(default_factory=list)
    emitters_initialized: bool = False      # 車両の寸法から放出源を自動配置済みか
    preset: str = "front"
    quality: str = "medium"
    transport: str = TRANSPORT_PATHLINE
    # --- 見た目 (Level 4) -------------------------------------------------
    color: str = COLOR_WHITE
    custom_color: tuple[float, float, float] = (0.85, 0.92, 1.0)
    density: float = 1.0          # 光学的な濃さ (吸収係数 κ の倍率)
    opacity: float = 1.0          # 不透明度 (0〜1、全体の掛け率)
    lighting: bool = False        # 陰影 (密度の勾配による柔らかい立体感)。高品質ボリューム煙では常に付ける
    light_intensity: float = 1.0
    # --- 煙の振る舞い (Level 2・3) ----------------------------------------
    spread: float = 0.02          # 表示用の広がり係数 D_visual [m²/s] (OpenFOAM の乱流拡散ではない)
    lifetime: float = 0.0         # 寿命 [s]。0 = 寿命なし (領域から出るまで)
    seeds_per_nozzle: int = 6     # ノズル 1 本を何本の流跡線で表すか (煙の筋の太さを表す)
    # --- 詳細設定 ---------------------------------------------------------
    voxel_budget: int = 0         # 0 = 品質から決める
    sampling_step: float = 0.5    # 流跡線の標本間隔 (ボクセル幅に対する比)
    sample_distance: float = 0.0  # レイキャスティングのサンプル間隔 [m]。0 = 自動 (ボクセル幅の半分)
    max_cache_mb: int = 512       # 濃度場のキャッシュの上限
    max_tracks: int = 20000       # 流跡線の本数の上限 (Maximum Active Particles)
    backend: str = BACKEND_AUTO
    visual_noise: bool = False    # 視覚効果としての微細なゆらぎ (CFD の結果ではない)。既定は OFF
    timeline_length: float = 0.0  # タイムラインの長さ [s]。0 = 自動 (煙が領域を通り抜ける時間)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["emitters"] = [e.to_dict() for e in self.emitters]
        data["custom_color"] = list(self.custom_color)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "SmokeSettings":
        """保存データから復元する。無い項目 (古いプロジェクト) は既定値。"""
        settings = cls()
        data = dict(data or {})
        for name, default in asdict(settings).items():
            if name in ("emitters", "custom_color") or name not in data:
                continue
            value = data[name]
            try:
                value = bool(value) if isinstance(default, bool) else int(value) if isinstance(default, int) else \
                    float(value) if isinstance(default, float) else str(value)
            except (TypeError, ValueError):
                continue
            setattr(settings, name, value)
        settings.emitters = [Emitter.from_dict(item) for item in data.get("emitters") or [] if isinstance(item, dict)]
        color = data.get("custom_color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            settings.custom_color = tuple(min(max(float(c), 0.0), 1.0) for c in color)
        if settings.quality not in QUALITIES:
            settings.quality = "medium"
        if settings.color not in COLOR_LABELS:
            settings.color = COLOR_WHITE
        if settings.backend not in BACKEND_LABELS:
            settings.backend = BACKEND_AUTO
        if settings.transport not in TRANSPORT_LABELS:
            settings.transport = TRANSPORT_PATHLINE
        return settings

    def rgb(self) -> tuple[float, float, float]:
        if self.color == COLOR_CUSTOM:
            return tuple(self.custom_color)
        return COLOR_RGB.get(self.color, COLOR_RGB[COLOR_WHITE])
