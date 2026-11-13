"""プロジェクトファイル (.ljmre) の保存と読込.

形式: ZIP アーカイブ
  project.json          … メタデータ・全オブジェクトの属性・座標系・設定
  arrays/<id>.npz       … メッシュ/点群/断面などの配列（圧縮）
  cad/<id>.brep         … CAD形状（OpenCASCADE BRep 形式）

元のスキャンファイルはパスと SHA-256 を記録し、メッシュ自体も（元データ・編集後とも）
アーカイブ内に保存するため、元ファイルが移動しても作業を継続できる。
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import os
import zipfile

import numpy as np

from .. import __app_name__, __project_format_version__, __version__
from .document import Document
from .objects import OBJECT_CLASSES

log = logging.getLogger(__name__)

PROJECT_EXT = ".ljmre"


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, tuple):
            return list(o)
        return super().default(o)


def project_dict(doc: Document, settings: dict | None = None) -> dict:
    return {
        "app": __app_name__,
        "app_version": __version__,
        "format_version": __project_format_version__,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "units": doc.units,
        "active_cs_id": doc.active_cs_id,
        "view_settings": doc.view_settings,
        "processing_params": doc.processing_params,
        "settings": settings or {},
        "objects": [o.to_dict() for o in doc.objects()],
    }


def save_project(doc: Document, path: str, settings: dict | None = None):
    path = os.fspath(path)
    tmp = path + ".tmp"
    data = project_dict(doc, settings)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("project.json", json.dumps(data, ensure_ascii=False, indent=1, cls=_Encoder))
        for o in doc.objects():
            arrs = o.arrays()
            if arrs:
                buf = io.BytesIO()
                np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrs.items()})
                z.writestr(f"arrays/{o.id}.npz", buf.getvalue())
            if o.kind == "cad" and getattr(o, "shape", None) is not None:
                from ..cad import occ_backend as occ
                z.writestr(f"cad/{o.id}.brep", occ.shape_to_brep_bytes(o.shape))
    os.replace(tmp, path)  # 書込み途中で失敗しても既存ファイルを壊さない
    doc.path = path
    doc.undo_stack.set_clean()


def load_project(path: str) -> tuple[Document, dict]:
    """戻り値 (Document, 保存されていた設定 dict)."""
    path = os.fspath(path)
    doc = Document()
    with zipfile.ZipFile(path, "r") as z:
        data = json.loads(z.read("project.json").decode("utf-8"))
        fv = data.get("format_version", 0)
        if fv > __project_format_version__:
            raise ValueError(f"このプロジェクトは新しいバージョン (形式 {fv}) で保存されています")
        names = set(z.namelist())
        for od in data.get("objects", []):
            cls = OBJECT_CLASSES.get(od.get("kind"))
            if cls is None:
                log.warning("未知のオブジェクト種類をスキップ: %s", od.get("kind"))
                continue
            arrays = None
            key = f"arrays/{od['id']}.npz"
            if key in names:
                with np.load(io.BytesIO(z.read(key)), allow_pickle=False) as npz:
                    arrays = {k: npz[k] for k in npz.files}
            o = cls.from_dict(od, arrays)
            if o.kind == "cad":
                bkey = f"cad/{o.id}.brep"
                if bkey in names:
                    try:
                        from ..cad import occ_backend as occ
                        o.shape = occ.shape_from_brep_bytes(z.read(bkey))
                    except Exception:
                        log.exception("CAD形状の読込に失敗: %s", o.name)
            doc._objects[o.id] = o
    doc.active_cs_id = data.get("active_cs_id")
    if doc.active_cs_id not in doc:
        doc.active_cs_id = None
    doc.units = data.get("units", "mm")
    doc.view_settings = data.get("view_settings", {})
    doc.processing_params = data.get("processing_params", {})
    doc.path = path
    doc.undo_stack.clear()
    return doc, data.get("settings", {})


def export_project_json(doc: Document, path: str, settings: dict | None = None):
    """プロジェクト情報（配列を除く）を JSON で出力."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_dict(doc, settings), f, ensure_ascii=False, indent=2, cls=_Encoder)
