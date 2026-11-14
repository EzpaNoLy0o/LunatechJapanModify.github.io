"""CAD インポートジョブ: ファイル → キャッシュ (表示メッシュ + B-Rep + メタ情報)。

出力先 ``<cache_root>/<key>/``:

* ``meta.json``            パーツ一覧・単位・統計
* ``part_NNN.vtp``         表示用サーフェス (面番号つき、座標は ``geometry_unit``)
* ``part_NNN_edges.vtp``   面輪郭線
* ``part_NNN_hull.npy``    凸包頂点 (姿勢を変えたときの境界ボックス計算用)
* ``part_NNN.brep``        B-Rep (STEP / IGES のみ。Stage 2 のメッシュ生成で使用)

キーは「元ファイルの絶対パス・サイズ・更新時刻・品質・インポータ版」から決まるため、
同じファイルの再読み込みは即座に完了し、ファイルが更新されれば自動的に再変換される。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..core import paths
from ..core.jobs import JobContext, JobError
from ..core.units import guess_unit_from_size

#: キャッシュ形式を変えたら上げる (古いキャッシュは無視される)
IMPORTER_VERSION = 3

STEP_SUFFIXES = (".step", ".stp")
IGES_SUFFIXES = (".iges", ".igs")
STL_SUFFIXES = (".stl",)
SUPPORTED_SUFFIXES = STEP_SUFFIXES + IGES_SUFFIXES + STL_SUFFIXES


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in STEP_SUFFIXES:
        return "step"
    if suffix in IGES_SUFFIXES:
        return "iges"
    if suffix in STL_SUFFIXES:
        return "stl"
    raise JobError(f"未対応のファイル形式です: {suffix or '(拡張子なし)'}  対応形式: STEP / STP / IGES / IGS / STL")


def cache_key(path: str | Path, quality: str) -> str:
    p = Path(path).resolve()
    stat = p.stat()
    text = f"{p}|{stat.st_size}|{stat.st_mtime_ns}|{quality}|v{IMPORTER_VERSION}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def load_cached_meta(cache_dir: str | Path) -> dict | None:
    """有効なキャッシュがあればメタ情報を返す。"""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("importer_version") != IMPORTER_VERSION:
        return None
    for part in meta.get("parts", []):
        for key in ("surface", "edges", "hull", "brep"):
            name = part.get(key)
            if name and not (cache_dir / name).is_file():
                return None
    return meta


def _bounds_list(bounds) -> list[float]:
    return [float(v) for v in bounds]


def _import_occ(path: Path, name: str, fmt: str, quality: str, max_parts: int, out_dir: Path, ctx: JobContext) -> tuple[list[dict], str | None]:
    # OCP の import は重いので、必要になってから読み込む
    from . import occ_io, unit_detect
    from .surface import build_surface, convex_hull_points, face_boundary_edges

    occ_io.silence_occ_messages()
    ctx.progress(0.05, f"{fmt.upper()} ファイルを解析中…")
    if fmt == "step":
        file_unit = unit_detect.detect_step_unit(path)
        occ_parts = occ_io.read_step(path, ctx, max_parts, name)
    else:
        file_unit = unit_detect.detect_iges_unit(path)
        occ_parts = occ_io.read_iges(path, ctx, max_parts, name)

    entries: list[dict] = []
    total = len(occ_parts)
    for index, part in enumerate(occ_parts):
        ctx.progress(0.35 + 0.6 * index / total, f"表示メッシュを生成中… {index + 1}/{total}  {part.name}")
        tess = occ_io.tessellate(part.shape, quality)
        if tess.skipped_faces:
            ctx.log(f"警告: {part.name} の {tess.skipped_faces} 面は三角形化できず、表示から欠落しています")
        surface = build_surface(tess.points, tess.triangles, tess.face_ids)
        stem = f"part_{index:03d}"
        surface.save(out_dir / f"{stem}.vtp", binary=True)
        face_boundary_edges(surface).save(out_dir / f"{stem}_edges.vtp", binary=True)
        np.save(out_dir / f"{stem}_hull.npy", convex_hull_points(surface.points))
        occ_io.write_brep(part.shape, out_dir / f"{stem}.brep")
        entries.append({
            "index": index,
            "name": part.name,
            "color": list(part.color) if part.color else None,
            "surface": f"{stem}.vtp",
            "edges": f"{stem}_edges.vtp",
            "hull": f"{stem}_hull.npy",
            "brep": f"{stem}.brep",
            "n_points": int(surface.n_points),
            "n_triangles": int(surface.n_cells),
            "n_faces": len(tess.face_types),
            "skipped_faces": int(tess.skipped_faces),
            "bounds": _bounds_list(surface.bounds),
            "face_types": tess.face_types,
        })
    return entries, file_unit


def _import_stl(path: Path, name: str, out_dir: Path, ctx: JobContext) -> list[dict]:
    from .surface import FACE_ID, convex_hull_points, face_boundary_edges, read_stl

    ctx.progress(0.1, "STL ファイルを読み込み中…")
    groups = read_stl(path, name)
    entries: list[dict] = []
    for index, (name, surface) in enumerate(groups):
        ctx.progress(0.3 + 0.6 * index / len(groups), f"表示メッシュを生成中… {name}")
        stem = f"part_{index:03d}"
        surface.save(out_dir / f"{stem}.vtp", binary=True)
        face_boundary_edges(surface).save(out_dir / f"{stem}_edges.vtp", binary=True)
        np.save(out_dir / f"{stem}_hull.npy", convex_hull_points(surface.points))
        entries.append({
            "index": index,
            "name": name,
            "color": None,
            "surface": f"{stem}.vtp",
            "edges": f"{stem}_edges.vtp",
            "hull": f"{stem}_hull.npy",
            "brep": None,
            "n_points": int(surface.n_points),
            "n_triangles": int(surface.n_cells),
            "n_faces": int(surface.cell_data[FACE_ID].max()) + 1,
            "skipped_faces": 0,
            "bounds": _bounds_list(surface.bounds),
            "face_types": None,
        })
    return entries


def import_cad_job(args: dict, ctx: JobContext | None = None) -> dict:
    """CAD ファイルを読み込みキャッシュへ変換する。戻り値はメタ情報 (``meta.json`` と同内容)。

    args:
        path        元ファイル
        cache_root  キャッシュのルート (省略時はワークスペース)
        quality     "coarse" | "standard" | "fine"
        max_parts   これを超えるパーツ数は 1 パーツへ統合
        force       True ならキャッシュを無視して再変換
    """
    ctx = ctx or JobContext()
    source = Path(args["path"]).expanduser()
    if not source.is_file():
        raise JobError(f"ファイルが見つかりません: {source}")
    fmt = detect_format(source)
    quality = args.get("quality", "standard")
    max_parts = int(args.get("max_parts", 300))
    cache_root = Path(args.get("cache_root") or paths.cad_cache_root())
    key = cache_key(source, quality)
    out_dir = cache_root / key

    if not args.get("force"):
        cached = load_cached_meta(out_dir)
        if cached is not None:
            ctx.log(f"キャッシュを使用: {source.name}")
            cached["cache_dir"] = str(out_dir)
            cached["from_cache"] = True
            return cached

    started = time.perf_counter()
    tmp_dir = cache_root / f"{key}.partial"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    # OpenCASCADE / VTK は Windows で非 ASCII パスの扱いが不安定なため、ASCII の一時コピーを読む
    read_path = source
    temp_copy: Path | None = None
    if not paths.is_ascii_path(source):
        temp_copy = tmp_dir / f"source{source.suffix.lower()}"
        shutil.copyfile(source, temp_copy)
        read_path = temp_copy

    try:
        if fmt == "stl":
            entries = _import_stl(read_path, source.stem, tmp_dir, ctx)
            file_unit = None
            max_dim = max(
                max(e["bounds"][1] - e["bounds"][0], e["bounds"][3] - e["bounds"][2], e["bounds"][5] - e["bounds"][4])
                for e in entries
            )
            geometry_unit = guess_unit_from_size(max_dim)
            unit_source = "heuristic"
        else:
            entries, file_unit = _import_occ(read_path, source.stem, fmt, quality, max_parts, tmp_dir, ctx)
            geometry_unit = "mm"  # OpenCASCADE が mm へ換算済み
            unit_source = "file" if file_unit else "occt"
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if temp_copy is not None:
        temp_copy.unlink(missing_ok=True)

    stat = source.stat()
    meta = {
        "importer_version": IMPORTER_VERSION,
        "key": key,
        "format": fmt,
        "quality": quality,
        "source": {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "file_unit": file_unit,
        "geometry_unit": geometry_unit,
        "unit_source": unit_source,
        "parts": entries,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)

    n_tris = sum(e["n_triangles"] for e in entries)
    ctx.progress(1.0, "読み込み完了")
    ctx.log(f"{source.name}: {len(entries)} パーツ / {n_tris:,} 三角形 / {meta['elapsed_s']:.2f} 秒")
    meta["cache_dir"] = str(out_dir)
    meta["from_cache"] = False
    return meta
