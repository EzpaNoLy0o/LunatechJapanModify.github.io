"""モデルツリー（左パネル）: 表示/非表示・名前変更・削除・選択."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..model.objects import KIND_LABELS

GROUP_ORDER = ["mesh", "points", "cs", "point", "axis", "plane", "cylinder", "hole", "sphere",
               "section", "cad", "measurement"]
GROUP_TITLES = {
    "mesh": "インポート済みメッシュ", "points": "点群", "cs": "基準座標系", "point": "基準点",
    "axis": "基準軸", "plane": "抽出した平面", "cylinder": "抽出した円筒", "hole": "穴中心 / 円",
    "sphere": "球", "section": "断面曲線", "cad": "CAD形状", "measurement": "測定結果",
}
ROLE_ID = QtCore.Qt.UserRole + 1


class ModelTree(QtWidgets.QTreeWidget):
    selectionIdsChanged = QtCore.Signal(list)
    requestDelete = QtCore.Signal(list)
    requestActivateCS = QtCore.Signal(str)
    requestZoom = QtCore.Signal(str)
    requestColor = QtCore.Signal(str)

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setHeaderLabels(["名前", "情報"])
        self.setColumnWidth(0, 210)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditKeyPressed |
                             QtWidgets.QAbstractItemView.SelectedClicked)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.setAlternatingRowColors(True)
        self.groups = {}
        self.items = {}
        self._updating = False
        for k in GROUP_ORDER:
            g = QtWidgets.QTreeWidgetItem([GROUP_TITLES[k], ""])
            g.setFlags(QtCore.Qt.ItemIsEnabled)
            f = g.font(0)
            f.setBold(True)
            g.setFont(0, f)
            g.setForeground(0, QtGui.QBrush(QtGui.QColor("#9fb7d9")))
            self.addTopLevelItem(g)
            g.setExpanded(True)
            g.setHidden(True)
            self.groups[k] = g
        self.itemChanged.connect(self._item_changed)
        self.itemSelectionChanged.connect(self._sel_changed)
        self.itemDoubleClicked.connect(self._dbl)
        doc.listeners.append(self.on_doc_event)
        self.rebuild()

    def set_document(self, doc):
        try:
            self.doc.listeners.remove(self.on_doc_event)
        except ValueError:
            pass
        self.doc = doc
        doc.listeners.append(self.on_doc_event)
        self.rebuild()

    # ------------------------------------------------------------------
    def _info(self, o):
        k = o.kind
        try:
            if k == "mesh":
                return f"{len(o.F):,} 面"
            if k == "points":
                return f"{len(o.P):,} 点"
            if k == "cylinder":
                return f"φ{o.diameter:.3f}"
            if k == "hole":
                return f"φ{o.diameter:.3f}"
            if k == "sphere":
                return f"φ{2 * o.radius:.3f}"
            if k == "plane" and o.stats:
                return f"RMS {o.stats['rms']:.3f}"
            if k == "cs":
                return "作業座標系" if self.doc.active_cs_id == o.id else ""
            if k == "section":
                return f"{len(o.polylines)} 曲線"
            if k == "cad":
                return o.summary.get("face_types_text", "")
            if k == "measurement":
                v, u = o.main_value()
                return "" if v is None else f"{v:.3f} {u}"
            if k in ("point", "axis") and getattr(o, "category", ""):
                return o.category
        except Exception:
            return ""
        return ""

    def _decorate(self, it, o):
        name = o.name
        if o.critical:
            name = ("✔ " if o.verified else "⚠ ") + name
        it.setText(0, name)
        it.setText(1, self._info(o))
        it.setCheckState(0, QtCore.Qt.Checked if o.visible else QtCore.Qt.Unchecked)
        if getattr(o, "warnings", None):
            it.setForeground(1, QtGui.QBrush(QtGui.QColor("#f0b35a")))
            it.setToolTip(0, "\n".join(o.warnings))
        else:
            it.setToolTip(0, KIND_LABELS.get(o.kind, o.kind))
        if o.kind == "cs" and self.doc.active_cs_id == o.id:
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)
        else:
            f = it.font(0)
            f.setBold(False)
            it.setFont(0, f)

    def rebuild(self):
        self._updating = True
        sel = set(self.selected_ids())
        for g in self.groups.values():
            g.takeChildren()
        self.items.clear()
        for o in self.doc.objects():
            self._add_item(o)
        for oid in sel:
            if oid in self.items:
                self.items[oid].setSelected(True)
        self._update_groups()
        self._updating = False

    def _add_item(self, o):
        g = self.groups.get(o.kind)
        if g is None:
            return
        it = QtWidgets.QTreeWidgetItem()
        it.setData(0, ROLE_ID, o.id)
        it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable |
                    QtCore.Qt.ItemIsUserCheckable)
        self._decorate(it, o)
        g.addChild(it)
        self.items[o.id] = it

    def _update_groups(self):
        for k, g in self.groups.items():
            n = g.childCount()
            g.setHidden(n == 0)
            g.setText(1, f"{n}")

    def on_doc_event(self, event, ids):
        self._updating = True
        try:
            if event == "cleared":
                self.rebuild()
            elif event == "added":
                for oid in ids:
                    o = self.doc.get(oid)
                    if o is not None and oid not in self.items:
                        self._add_item(o)
            elif event == "removed":
                for oid in ids:
                    it = self.items.pop(oid, None)
                    if it is not None and it.parent() is not None:
                        it.parent().removeChild(it)
            elif event in ("changed", "renamed", "visibility"):
                for oid in ids:
                    o = self.doc.get(oid)
                    it = self.items.get(oid)
                    if o is not None and it is not None:
                        self._decorate(it, o)
            elif event == "active_cs":
                for o in self.doc.objects("cs"):
                    if o.id in self.items:
                        self._decorate(self.items[o.id], o)
            self._update_groups()
        finally:
            self._updating = False
        if event == "removed":
            self._sel_changed()

    def selected_ids(self):
        return [it.data(0, ROLE_ID) for it in self.selectedItems() if it.data(0, ROLE_ID)]

    def select_ids(self, ids, clear=True):
        self.blockSignals(True)
        if clear:
            self.clearSelection()
        for oid in ids:
            it = self.items.get(oid)
            if it is not None:
                it.setSelected(True)
                self.scrollToItem(it)
        self.blockSignals(False)
        self._sel_changed()

    # ------------------------------------------------------------------
    def _item_changed(self, it, col):
        if self._updating:
            return
        oid = it.data(0, ROLE_ID)
        o = self.doc.get(oid) if oid else None
        if o is None:
            return
        vis = it.checkState(0) == QtCore.Qt.Checked
        if vis != o.visible:
            self.doc.set_visible(oid, vis)
        text = it.text(0).lstrip("⚠✔ ").strip()
        if col == 0 and text and text != o.name:
            self.doc.rename(oid, text)

    def _sel_changed(self):
        self.selectionIdsChanged.emit(self.selected_ids())

    def _dbl(self, it, col):
        oid = it.data(0, ROLE_ID)
        if oid:
            self.requestZoom.emit(oid)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Delete:
            ids = self.selected_ids()
            if ids:
                self.requestDelete.emit(ids)
                return
        super().keyPressEvent(e)

    def _menu(self, pos):
        ids = self.selected_ids()
        if not ids:
            return
        m = QtWidgets.QMenu(self)
        a_show = m.addAction("表示")
        a_hide = m.addAction("非表示")
        a_only = m.addAction("これだけ表示")
        m.addSeparator()
        a_ren = m.addAction("名前の変更 (F2)")
        a_col = m.addAction("色の変更…")
        a_zoom = m.addAction("ズーム")
        a_cs = None
        if len(ids) == 1 and self.doc.get(ids[0]).kind == "cs":
            m.addSeparator()
            a_cs = m.addAction("作業座標系に設定")
        m.addSeparator()
        a_del = m.addAction("削除 (Delete)")
        act = m.exec(self.viewport().mapToGlobal(pos))
        if act is None:
            return
        if act == a_show:
            for i in ids:
                self.doc.set_visible(i, True)
        elif act == a_hide:
            for i in ids:
                self.doc.set_visible(i, False)
        elif act == a_only:
            for o in self.doc.objects():
                self.doc.set_visible(o.id, o.id in ids)
        elif act == a_ren:
            self.editItem(self.items[ids[0]], 0)
        elif act == a_col:
            self.requestColor.emit(ids[0])
        elif act == a_zoom:
            self.requestZoom.emit(ids[0])
        elif act == a_del:
            self.requestDelete.emit(ids)
        elif a_cs is not None and act == a_cs:
            self.requestActivateCS.emit(ids[0])
