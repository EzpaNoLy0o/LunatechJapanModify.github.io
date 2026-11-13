"""ドキュメント: シーンオブジェクトの集合・作業座標系・Undo/Redo・変更通知."""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional

import numpy as np

from ..core.coords import Frame, WORLD
from .objects import SceneObject, CoordinateSystemObject, KIND_LABELS
from .undo import Command, UndoStack

log = logging.getLogger(__name__)


class _AddCmd(Command):
    def __init__(self, doc, objs, label):
        self.doc, self.objs, self.label = doc, list(objs), label
        self.positions = None

    def do(self):
        for o in self.objs:
            self.doc._objects[o.id] = o
        self.doc._emit("added", [o.id for o in self.objs])

    def undo(self):
        ids = [o.id for o in self.objs]
        for i in ids:
            self.doc._objects.pop(i, None)
        if self.doc.active_cs_id in ids:
            self.doc.active_cs_id = None
            self.doc._emit("active_cs", [])
        self.doc._emit("removed", ids)


class _RemoveCmd(Command):
    def __init__(self, doc, ids, label):
        self.doc, self.ids, self.label = doc, list(ids), label
        self.saved = []
        self.was_active = None

    def do(self):
        keys = list(self.doc._objects.keys())
        self.saved = [(keys.index(i), self.doc._objects[i]) for i in self.ids if i in self.doc._objects]
        for i in self.ids:
            self.doc._objects.pop(i, None)
        if self.doc.active_cs_id in self.ids:
            self.was_active = self.doc.active_cs_id
            self.doc.active_cs_id = None
            self.doc._emit("active_cs", [])
        self.doc._emit("removed", self.ids)

    def undo(self):
        items = list(self.doc._objects.items())
        for pos, o in sorted(self.saved, key=lambda t: t[0]):
            items.insert(min(pos, len(items)), (o.id, o))
        self.doc._objects = OrderedDict(items)
        if self.was_active:
            self.doc.active_cs_id = self.was_active
            self.doc._emit("active_cs", [self.was_active])
        self.doc._emit("added", [o.id for _, o in self.saved])


class _StateCmd(Command):
    def __init__(self, doc, ids, before, after, label):
        self.doc, self.ids, self.before, self.after, self.label = doc, ids, before, after, label

    def _apply(self, states):
        for i, st in zip(self.ids, states):
            o = self.doc._objects.get(i)
            if o is not None:
                o.set_state(st)
        self.doc._emit("changed", self.ids)

    def do(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class Document:
    def __init__(self):
        self._objects: "OrderedDict[str, SceneObject]" = OrderedDict()
        self.undo_stack = UndoStack()
        self.listeners: List[Callable[[str, list], None]] = []
        self.active_cs_id: Optional[str] = None
        self.path: Optional[str] = None
        self.units = "mm"
        self.view_settings: dict = {}
        self.processing_params: dict = {}

    # ---- 通知 --------------------------------------------------------------
    def _emit(self, event: str, ids: list):
        for f in list(self.listeners):
            try:
                f(event, ids)
            except Exception:  # 通知先の不具合でモデルを壊さない
                log.exception("ドキュメント通知の処理中にエラー")

    # ---- 参照 --------------------------------------------------------------
    def __contains__(self, oid):
        return oid in self._objects

    def get(self, oid) -> Optional[SceneObject]:
        return self._objects.get(oid)

    def objects(self, kind: Optional[str] = None) -> List[SceneObject]:
        if kind is None:
            return list(self._objects.values())
        kinds = (kind,) if isinstance(kind, str) else tuple(kind)
        return [o for o in self._objects.values() if o.kind in kinds]

    def ids(self):
        return list(self._objects.keys())

    def __len__(self):
        return len(self._objects)

    def unique_name(self, base: str) -> str:
        names = {o.name for o in self._objects.values()}
        if base not in names:
            return base
        i = 2
        while f"{base} {i}" in names:
            i += 1
        return f"{base} {i}"

    def auto_name(self, kind: str) -> str:
        base = KIND_LABELS.get(kind, kind)
        n = len([o for o in self._objects.values() if o.kind == kind]) + 1
        name = f"{base}{n}"
        while any(o.name == name for o in self._objects.values()):
            n += 1
            name = f"{base}{n}"
        return name

    # ---- 作業座標系 ----------------------------------------------------------
    @property
    def active_frame(self) -> Frame:
        o = self.get(self.active_cs_id) if self.active_cs_id else None
        if isinstance(o, CoordinateSystemObject):
            return o.frame
        return WORLD

    @property
    def active_frame_name(self) -> str:
        o = self.get(self.active_cs_id) if self.active_cs_id else None
        return o.name if o is not None else "ワールド"

    def set_active_cs(self, oid: Optional[str]):
        if oid is not None and not isinstance(self.get(oid), CoordinateSystemObject):
            raise ValueError("座標系ではありません")
        self.active_cs_id = oid
        self._emit("active_cs", [oid] if oid else [])

    # ---- 変更（Undo対応） ----------------------------------------------------
    def add(self, objs, label: Optional[str] = None):
        if isinstance(objs, SceneObject):
            objs = [objs]
        objs = list(objs)
        if not objs:
            return
        self.undo_stack.push(_AddCmd(self, objs, label or f"{objs[0].name} を追加"))

    def remove(self, ids: Iterable[str], label: Optional[str] = None):
        ids = [i for i in ids if i in self._objects]
        if not ids:
            return
        self.undo_stack.push(_RemoveCmd(self, ids, label or f"{len(ids)} 個のオブジェクトを削除"))

    def modify(self, ids, fn: Callable[[SceneObject], None], label: str = "変更"):
        """fn を各オブジェクトに適用し、前後の状態を Undo に記録する."""
        if isinstance(ids, str):
            ids = [ids]
        ids = [i for i in ids if i in self._objects]
        if not ids:
            return
        before = [self._objects[i].get_state() for i in ids]
        try:
            for i in ids:
                fn(self._objects[i])
        except Exception:
            for i, st in zip(ids, before):
                self._objects[i].set_state(st)
            raise
        after = [self._objects[i].get_state() for i in ids]
        self.undo_stack.push(_StateCmd(self, ids, before, after, label), execute=False)
        self._emit("changed", ids)

    def set_property(self, oid: str, key: str, value, label: Optional[str] = None):
        frame = self.active_frame
        self.modify(oid, lambda o: o.set_property(key, value, frame), label or "プロパティ変更")
        if key == "name":
            self._emit("renamed", [oid])

    def rename(self, oid: str, name: str):
        self.set_property(oid, "name", name, "名前の変更")

    def transform(self, ids, T: np.ndarray, label: str = "移動"):
        T = np.asarray(T, float)
        self.modify(ids, lambda o: o.apply_transform(T), label)

    def scale(self, ids, s: float, center=(0, 0, 0), label: str = "スケール補正"):
        self.modify(ids, lambda o: o.apply_scale(s, center), label)

    def set_visible(self, oid: str, visible: bool):
        o = self.get(oid)
        if o is not None and o.visible != visible:
            o.visible = visible
            self._emit("visibility", [oid])

    def clear(self):
        ids = self.ids()
        self._objects.clear()
        self.active_cs_id = None
        self.undo_stack.clear()
        self.path = None
        self._emit("cleared", ids)

    def undo(self):
        return self.undo_stack.undo()

    def redo(self):
        return self.undo_stack.redo()

    @property
    def modified(self) -> bool:
        return not self.undo_stack.is_clean

    def all_points_for_bounds(self) -> Optional[np.ndarray]:
        pts = []
        for o in self._objects.values():
            if not o.visible:
                continue
            if o.kind == "mesh" and len(o.V):
                pts.append(np.vstack([o.V.min(0), o.V.max(0)]))
            elif o.kind == "points" and len(o.P):
                pts.append(np.vstack([o.P.min(0), o.P.max(0)]))
            else:
                b = o.bounds_points()
                if b is not None and len(b):
                    pts.append(np.atleast_2d(b))
        return np.vstack(pts) if pts else None
