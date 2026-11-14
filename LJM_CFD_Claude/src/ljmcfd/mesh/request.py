"""シーンの状態 (アセンブリ・風洞・メッシュ設定) からメッシュ生成ジョブの引数を組み立てる。"""

from __future__ import annotations

from pathlib import Path

from ..cad.assembly import Assembly
from ..tunnel.spec import TunnelSpec
from .settings import MeshSettings, input_hash


def analysis_parts(assembly: Assembly) -> list:
    """解析対象のパーツ。非表示のパーツは解析から除外する。"""
    return [p for p in assembly.parts if p.visible]


def part_payloads(assembly: Assembly) -> list[dict]:
    """ジョブへ渡すパーツ情報 (キャッシュファイルの絶対パスとワールド変換行列)。"""
    attitude = assembly.attitude_matrix()
    payloads = []
    for part in analysis_parts(assembly):
        cache = Path(part.cache_dir)
        payloads.append({
            "id": part.id,
            "name": part.name,
            "brep": str(cache / part.entry["brep"]) if part.entry.get("brep") else None,
            "surface": str(cache / part.entry["surface"]),
            "matrix": assembly.world_matrix(part, attitude).tolist(),
            "cache_key": part.cache_key,
            "source_index": part.source_index,
        })
    return payloads


def build_mesh_request(assembly: Assembly, spec: TunnelSpec, settings: MeshSettings, out_dir: str | Path) -> dict:
    """メッシュ生成ジョブ (``mesh-generate``) の引数。``input_hash`` は入力が同じなら同じ値になる。"""
    parts = analysis_parts(assembly)
    bounds = assembly.world_bounds(parts)
    if bounds is None:
        raise ValueError("解析対象のパーツがありません")
    payload = {
        "parts": part_payloads(assembly),
        "domain": list(spec.domain_bounds(bounds)),
        "model_bounds": list(bounds),
        "has_ground": spec.has_ground,
        "sink": spec.model_sink(bounds),
        "settings": settings.to_dict(),
    }
    hashed = dict(payload)
    # ハッシュにはファイルの置き場所ではなく内容の識別子を使う (プロジェクトを移動しても変わらないように)
    hashed["parts"] = [
        {"key": p["cache_key"], "index": p["source_index"], "matrix": p["matrix"], "id": p["id"]} for p in payload["parts"]
    ]
    payload["input_hash"] = input_hash(hashed)
    payload["out_dir"] = str(out_dir)
    return payload
