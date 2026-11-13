"""CAD ファイルに記録された長さ単位の検出。

STEP / IGES は OpenCASCADE が読み込み時に mm へ換算するため、ここでの検出結果は
「元ファイルが何の単位で書かれていたか」をユーザーへ示すための情報である
(形状の換算そのものには使わない)。検出できなかった場合は ``None`` を返し、
推測値は返さない。
"""

from __future__ import annotations

import re
from pathlib import Path

_CHUNK = 8 * 1024 * 1024
_OVERLAP = 4096

# ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )
_SI_LENGTH_RE = re.compile(
    rb"LENGTH_UNIT\s*\(\s*\)\s*NAMED_UNIT\s*\([^)]*\)\s*SI_UNIT\s*\(\s*(\.[A-Z]+\.|\$)\s*,\s*\.METRE\.\s*\)"
)
# ( CONVERSION_BASED_UNIT('INCH',#12) LENGTH_UNIT() NAMED_UNIT(#9) )
_CONV_LENGTH_RE = re.compile(rb"CONVERSION_BASED_UNIT\s*\(\s*'([^']*)'\s*,[^)]*\)\s*LENGTH_UNIT")

_SI_PREFIX = {b"$": "m", b".MILLI.": "mm", b".CENTI.": "cm", b".DECI.": "dm", b".MICRO.": "um", b".KILO.": "km"}
_CONV_NAMES = {
    "INCH": "inch", "INCHES": "inch", "IN": "inch",
    "FOOT": "ft", "FEET": "ft", "FT": "ft",
    "MILLIMETRE": "mm", "MILLIMETER": "mm", "MM": "mm",
    "CENTIMETRE": "cm", "CENTIMETER": "cm", "CM": "cm",
    "METRE": "m", "METER": "m", "M": "m",
}


def detect_step_unit(path: str | Path) -> str | None:
    """STEP ファイルの長さ単位名 (``"mm"``, ``"m"``, ``"inch"`` …) を返す。

    換算単位 (CONVERSION_BASED_UNIT) が定義されていればそれを優先する
    (インチ系ファイルは定義用の SI 単位も併せて持つため)。
    """
    si_unit: str | None = None
    conv_unit: str | None = None
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            data = tail + chunk
            m = _CONV_LENGTH_RE.search(data)
            if m and conv_unit is None:
                name = m.group(1).decode("ascii", "replace").strip().upper()
                conv_unit = _CONV_NAMES.get(name, name.lower())
            if si_unit is None:
                m = _SI_LENGTH_RE.search(data)
                if m:
                    si_unit = _SI_PREFIX.get(m.group(1), m.group(1).decode("ascii").strip(".").lower() + "m")
            tail = data[-_OVERLAP:]
    return conv_unit or si_unit


_IGES_UNIT_FLAGS = {
    1: "inch", 2: "mm", 4: "ft", 5: "mile", 6: "m", 7: "km", 8: "mil", 9: "um", 10: "cm", 11: "uin",
}


def _parse_iges_global(text: str) -> list[str]:
    """IGES グローバルセクションをパラメータ列へ分解する (Hollerith 文字列対応)。"""
    param_delim, record_delim = ",", ";"
    params: list[str] = []
    i, n = 0, len(text)
    # 先頭 2 パラメータは区切り文字自身の定義 ("1H," と "1H;") または省略
    for idx in range(2):
        if text.startswith("1H", i) and i + 2 < n:
            ch = text[i + 2]
            if idx == 0:
                param_delim = ch
            else:
                record_delim = ch
            params.append(ch)
            i += 3
        else:
            params.append("")
        if i < n and text[i] == param_delim:
            i += 1
    current = ""
    while i < n:
        m = re.match(r"\s*(\d+)H", text[i:])
        if m and not current.strip():
            length = int(m.group(1))
            start = i + m.end()
            current = text[start:start + length]
            i = start + length
            continue
        ch = text[i]
        if ch == param_delim or ch == record_delim:
            params.append(current.strip())
            current = ""
            i += 1
            if ch == record_delim:
                break
            continue
        current += ch
        i += 1
    return params


def detect_iges_unit(path: str | Path) -> str | None:
    """IGES グローバルセクション (パラメータ 14/15) から長さ単位を返す。"""
    global_text = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            if len(line) < 73:
                continue
            section = line[72]
            if section == "G":
                global_text.append(line[:72])
            elif section in "DPT":
                break
    if not global_text:
        return None
    params = _parse_iges_global("".join(global_text))
    if len(params) < 15:
        return None
    try:
        flag = int(params[13])
    except ValueError:
        return None
    if flag == 3:  # 単位名はパラメータ 15 を参照
        name = params[14].strip().upper()
        return _CONV_NAMES.get(name, name.lower() or None)
    return _IGES_UNIT_FLAGS.get(flag)
