"""数値表示の書式.

表示桁数はユーザー設定。表示桁数は「測定精度」を意味しないことに注意
（精度はフィッティング残差・不確かさ・実測検証で判断する）。
"""
from __future__ import annotations

import math


class NumberFormat:
    def __init__(self, length_decimals: int = 3, angle_decimals: int = 3):
        self.length_decimals = int(length_decimals)
        self.angle_decimals = int(angle_decimals)

    def _f(self, v, d):
        if v is None:
            return "—"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        if math.isnan(fv):
            return "NaN"
        if abs(fv) < 0.5 * 10 ** (-d):
            fv = 0.0  # "-0.000" を避ける
        return f"{fv:.{d}f}"

    def length(self, v) -> str:
        return self._f(v, self.length_decimals)

    def angle(self, v) -> str:
        return self._f(v, self.angle_decimals)

    def value(self, v, unit: str = "") -> str:
        if unit in ("deg", "°"):
            return self.angle(v)
        if unit in ("mm", "mm²", "mm³"):
            return self.length(v)
        if isinstance(v, float):
            return f"{v:.6g}"
        return "" if v is None else str(v)


DEFAULT_FORMAT = NumberFormat()
