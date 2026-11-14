"""仮想風洞の仕様: 流れの条件と計算領域の寸法。

座標系はアプリ共通 (風向 +X、上 +Z、地面 Z = 0)。モデルの姿勢は ``Assembly`` 側が持ち、
ここでは「モデルのワールド境界」を受け取って計算領域を決める。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

R_AIR = 287.05          # 乾燥空気の気体定数 [J/(kg K)]
GAMMA_AIR = 1.4
KELVIN = 273.15

MODE_VEHICLE = "vehicle"
MODE_COMPONENT = "component"
MODE_INTERNAL = "internal"
MODE_LABELS = {
    MODE_VEHICLE: "Vehicle Wind Tunnel (車両全体)",
    MODE_COMPONENT: "Component Wind Tunnel (単体パーツ)",
    MODE_INTERNAL: "Internal Flow (内部流れ・未実装)",
}
#: 実際に解析できるモード。Internal Flow は拡張点として定義のみ。
IMPLEMENTED_MODES = (MODE_VEHICLE, MODE_COMPONENT)

#: これより接地面に近い最下点は「接地している」とみなす [m]
CONTACT_TOLERANCE = 1.0e-3


def air_density(pressure_pa: float, temperature_c: float) -> float:
    """理想気体としての乾燥空気の密度 [kg/m³]。"""
    return pressure_pa / (R_AIR * (temperature_c + KELVIN))


def dynamic_viscosity(temperature_c: float) -> float:
    """サザーランドの式による空気の粘性係数 [Pa s]。"""
    t = temperature_c + KELVIN
    return 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)


@dataclass
class FlowConditions:
    speed_ms: float = 200.0 / 3.6            # 主流速度 [m/s] (既定 200 km/h)
    temperature_c: float = 20.0
    pressure_pa: float = 101325.0
    density_override: float | None = None    # None なら温度・気圧から算出
    turbulence_intensity_pct: float = 1.0    # 乱流強度 [%]
    viscosity_ratio: float = 10.0            # 流入の渦粘性比 ν_t/ν (ω の流入値に使用)

    @property
    def speed_kmh(self) -> float:
        return self.speed_ms * 3.6

    @property
    def density(self) -> float:
        if self.density_override is not None:
            return self.density_override
        return air_density(self.pressure_pa, self.temperature_c)

    @property
    def mu(self) -> float:
        return dynamic_viscosity(self.temperature_c)

    @property
    def nu(self) -> float:
        return self.mu / self.density

    @property
    def mach(self) -> float:
        return self.speed_ms / math.sqrt(GAMMA_AIR * R_AIR * (self.temperature_c + KELVIN))

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.density * self.speed_ms ** 2

    def reynolds(self, length: float) -> float:
        return self.speed_ms * length / self.nu

    @property
    def k_inlet(self) -> float:
        """流入の乱流エネルギー k = 1.5 (U·I)² [m²/s²]。"""
        return 1.5 * (self.speed_ms * self.turbulence_intensity_pct / 100.0) ** 2

    @property
    def omega_inlet(self) -> float:
        """流入の比散逸率 ω = k / ν_t [1/s]。"""
        return self.k_inlet / (self.nu * self.viscosity_ratio)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "FlowConditions":
        data = data or {}
        base = cls()
        kwargs = {}
        for name in cls.__dataclass_fields__:
            value = data.get(name, getattr(base, name))
            kwargs[name] = None if value is None else float(value)
        return cls(**kwargs)


@dataclass
class TunnelSpec:
    mode: str = MODE_VEHICLE
    has_ground: bool = True
    moving_ground: bool = True
    ground_clearance: float = 0.0     # モデル最下点の地上高の目標値 [m]
    contact_sink: float = 0.010       # 接地時にモデルを沈める量 [m] (接地面を線ではなく面にする)
    margin_front: float = 0.0         # モデル前端から流入面まで [m]
    margin_rear: float = 0.0          # モデル後端から流出面まで [m]
    margin_side: float = 0.0          # モデル側面から側壁まで (左右同じ) [m]
    margin_top: float = 0.0           # モデル上端から天井まで [m]
    margin_bottom: float = 0.0        # 地面なしのときモデル下端から底面まで [m]
    flow: FlowConditions = field(default_factory=FlowConditions)
    frontal_area: float | None = None       # 前面投影面積の計算値 [m²] (未計算なら None)
    reference_area: float | None = None     # Cd・Cl の基準面積 [m²]。None なら前面投影面積を使う
    generated: bool = False                 # ユーザーが風洞を生成 (確定) したか

    def to_dict(self) -> dict:
        data = asdict(self)
        data["flow"] = self.flow.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "TunnelSpec":
        data = data or {}
        spec = cls()
        for name, default in asdict(spec).items():
            if name == "flow" or name not in data:
                continue
            value = data[name]
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, float) or name in ("frontal_area", "reference_area"):
                value = None if value is None else float(value)
            setattr(spec, name, value)
        spec.flow = FlowConditions.from_dict(data.get("flow"))
        return spec

    # ------------------------------------------------------------------ 寸法
    def effective_reference_area(self) -> float | None:
        """Cd・Cl に使う基準面積。ユーザー指定 → 前面投影面積 の順。どちらも無ければ ``None``。"""
        return self.reference_area if self.reference_area is not None else self.frontal_area

    def domain_bounds(self, model_bounds) -> tuple[float, float, float, float, float, float]:
        """計算領域 (xmin, xmax, ymin, ymax, zmin, zmax) [m]。"""
        x0, x1, y0, y1, z0, z1 = model_bounds
        bottom = 0.0 if self.has_ground else z0 - self.margin_bottom
        return (
            x0 - self.margin_front, x1 + self.margin_rear,
            y0 - self.margin_side, y1 + self.margin_side,
            bottom, z1 + self.margin_top,
        )

    def cross_section_area(self, model_bounds) -> float:
        d = self.domain_bounds(model_bounds)
        return (d[3] - d[2]) * (d[5] - d[4])

    def blockage_ratio(self, model_bounds) -> float | None:
        """閉塞率 = 前面投影面積 / 風洞断面積。前面投影面積が未計算なら ``None``。"""
        if self.frontal_area is None:
            return None
        return self.frontal_area / self.cross_section_area(model_bounds)

    def model_sink(self, model_bounds) -> float:
        """メッシュ生成時にモデルを下げる量 [m]。

        タイヤのように地面へ「線」で接する形状は、接地部に厚みゼロのくさび形セルを生む。
        最下点がほぼ Z = 0 にあるときは ``contact_sink`` だけ沈め、有限の接地面を作る。
        半径 330 mm のタイヤでは、5 mm だと接地端の角度が約 10° と鋭く品質警告が残り、
        10 mm (約 14°、接地長 約 160 mm = 荷重のかかった実タイヤ相当) で良好になる。
        沈めた分だけ車高が下がる点に注意 (GUI でも明示する)。
        """
        if self.has_ground and abs(model_bounds[4]) <= CONTACT_TOLERANCE:
            return self.contact_sink
        return 0.0

    def patch_names(self) -> list[str]:
        """計算領域外周のパッチ名 (順序固定)。"""
        return ["inlet", "outlet", "sideL", "sideR", "ground" if self.has_ground else "bottom", "top"]


def default_margins(model_bounds, mode: str = MODE_VEHICLE) -> dict[str, float]:
    """モデル寸法から風洞の余白を決める。

    車両: 前 3L / 後 6L / 左右 各 3W / 上 4H (閉塞率は概ね 3 % 以下になる)。
    単体パーツ: 代表寸法 D = max(L, W, H) に対して 前 3D / 後 6D / 上下左右 各 3D。
    """
    length = model_bounds[1] - model_bounds[0]
    width = model_bounds[3] - model_bounds[2]
    height = model_bounds[5] - model_bounds[4]
    if mode == MODE_COMPONENT:
        d = max(length, width, height)
        return {"margin_front": 3 * d, "margin_rear": 6 * d, "margin_side": 3 * d, "margin_top": 3 * d, "margin_bottom": 3 * d}
    return {
        "margin_front": 3 * length, "margin_rear": 6 * length, "margin_side": 3 * width,
        "margin_top": 4 * height, "margin_bottom": 3 * height,
    }


def apply_mode_defaults(spec: TunnelSpec, model_bounds) -> None:
    """解析モードに応じた既定値 (余白・地面) を ``spec`` へ設定する。"""
    for name, value in default_margins(model_bounds, spec.mode).items():
        setattr(spec, name, value)
    if spec.mode == MODE_VEHICLE:
        spec.has_ground = True
    elif spec.mode == MODE_COMPONENT:
        spec.has_ground = False
        spec.moving_ground = False


def validate(spec: TunnelSpec, model_bounds) -> list[tuple[str, str]]:
    """設定の妥当性を調べ、(レベル, メッセージ) のリストを返す。レベルは "error" / "warning" / "info"。"""
    issues: list[tuple[str, str]] = []
    if spec.mode not in IMPLEMENTED_MODES:
        issues.append(("error", "Internal Flow (内部流れ) モードは未実装です。Vehicle または Component を選択してください。"))
        return issues
    flow = spec.flow
    if flow.speed_ms <= 0:
        issues.append(("error", "風速は正の値にしてください。"))
    if flow.density <= 0 or flow.pressure_pa <= 0:
        issues.append(("error", "気圧・空気密度は正の値にしてください。"))
    margins = {"前方": spec.margin_front, "後方": spec.margin_rear, "左右": spec.margin_side, "上方": spec.margin_top}
    if not spec.has_ground:
        margins["下方"] = spec.margin_bottom
    for label, value in margins.items():
        if value <= 0:
            issues.append(("error", f"{label}の余白は正の値にしてください。"))
    length = model_bounds[1] - model_bounds[0]
    height = model_bounds[5] - model_bounds[4]
    if spec.margin_front < 1.5 * length or spec.margin_rear < 3.0 * length:
        issues.append(("warning", "前後の余白が小さいため、流入・流出境界が結果へ影響する可能性があります (目安: 前 3L / 後 6L 以上)。"))
    if spec.has_ground:
        zmin = model_bounds[4]
        if zmin < -0.25 * height:
            issues.append(("error", f"モデルが地面 (Z=0) より {-zmin * 1000:.0f} mm 下にあります。「地面との距離」を適用してください。"))
        elif zmin < -CONTACT_TOLERANCE:
            issues.append(("info", f"モデルの最下部 {-zmin * 1000:.1f} mm が地面より下にあり、その部分は切り取られます。"))
        elif zmin <= CONTACT_TOLERANCE:
            issues.append(("info", f"モデルが接地しています。接地面を作るためメッシュ生成時に {spec.contact_sink * 1000:.1f} mm 沈めます "
                                   f"(車高もその分下がります。「接地部の沈み込み」で変更できます)。"))
        elif zmin < 0.02:
            issues.append(("warning", f"地面との隙間が {zmin * 1000:.1f} mm しかありません。隙間のセルが極端に薄くなるため、接地 (0 mm) にするか 20 mm 以上を推奨します。"))
    if flow.mach > 0.3:
        issues.append(("warning", f"マッハ数 {flow.mach:.2f} は 0.3 を超えています。非圧縮性の仮定による誤差が大きくなります。"))
    blockage = spec.blockage_ratio(model_bounds)
    if blockage is not None:
        if blockage > 0.10:
            issues.append(("warning", f"閉塞率 {blockage * 100:.1f} % は大きすぎます。風洞壁の影響で抗力が過大になります (目安 5 % 以下)。"))
        elif blockage > 0.05:
            issues.append(("warning", f"閉塞率 {blockage * 100:.1f} % はやや大きめです (目安 5 % 以下)。"))
    return issues


def estimate_yplus(first_layer_height: float, flow: FlowConditions, length: float) -> float:
    """壁面第 1 セル中心の y+ の概算 (平板乱流境界層の摩擦係数 Cf = 0.058 Re^-0.2 による目安)。"""
    reynolds = max(flow.reynolds(length), 1.0)
    cf = 0.058 * reynolds ** -0.2
    u_tau = flow.speed_ms * math.sqrt(cf / 2.0)
    return 0.5 * first_layer_height * u_tau / flow.nu


def first_layer_height_for_yplus(yplus: float, flow: FlowConditions, length: float) -> float:
    """目標 y+ (セル中心) を与える第 1 層の厚さ [m] の概算。"""
    reynolds = max(flow.reynolds(length), 1.0)
    cf = 0.058 * reynolds ** -0.2
    u_tau = flow.speed_ms * math.sqrt(cf / 2.0)
    return 2.0 * yplus * flow.nu / u_tau
