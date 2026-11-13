"""単位変換。

アプリ内部の長さは常に SI (メートル) で保持する。CAD ファイル固有の単位は
``unit_scale`` (ファイル単位 → m) としてパーツごとに持ち、表示単位 (mm / m) は
GUI の書式設定だけに影響する。
"""

from __future__ import annotations

#: 長さ単位 → メートル換算係数
UNIT_TO_M: dict[str, float] = {
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "m": 1.0,
    "inch": 0.0254,
    "ft": 0.3048,
}

#: GUI で選択できる表示単位
DISPLAY_UNITS = ("mm", "m")


def unit_scale(unit: str) -> float:
    """``unit`` で書かれた長さをメートルへ変換する係数を返す。"""
    try:
        return UNIT_TO_M[unit]
    except KeyError as exc:
        raise ValueError(f"未対応の長さ単位です: {unit!r}") from exc


def to_display(value_m: float, display_unit: str) -> float:
    """メートル値を表示単位の数値へ変換する。"""
    return value_m / unit_scale(display_unit)


def from_display(value: float, display_unit: str) -> float:
    """表示単位の数値をメートルへ変換する。"""
    return value * unit_scale(display_unit)


def format_length(value_m: float, display_unit: str, decimals: int | None = None) -> str:
    """メートル値を ``"1234.5 mm"`` のような表示文字列にする。"""
    if decimals is None:
        decimals = 1 if display_unit == "mm" else 4
    return f"{to_display(value_m, display_unit):.{decimals}f} {display_unit}"


def guess_unit_from_size(max_dimension: float) -> str:
    """単位情報を持たないファイル (STL 等) の単位を寸法から推定する。

    自動車および空力パーツを対象とした経験則であり、確定情報ではない。
    最大寸法が 50 以上なら mm (50 m を超える車両部品は想定しない)、
    それ未満なら m とみなす。呼び出し側は必ず「推定」であることを
    ユーザーへ示し、変更手段を提供すること。
    """
    return "mm" if max_dimension >= 50.0 else "m"


def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh / 3.6


def ms_to_kmh(v_ms: float) -> float:
    return v_ms * 3.6
