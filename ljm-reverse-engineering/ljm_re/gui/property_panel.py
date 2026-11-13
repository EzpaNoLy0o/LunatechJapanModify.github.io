"""プロパティパネル（右パネル）: 寸法・座標・フィット結果の表示と数値編集."""
from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)

ROLE_KEY = QtCore.Qt.UserRole + 1
ROLE_KIND = QtCore.Qt.UserRole + 2


def _parse_vec(text):
    parts = [p for p in text.replace("(", " ").replace(")", " ").replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError("3つの数値を カンマ区切り で入力してください（例: 10, 0, 5）")
    return [float(p) for p in parts]


class PropertyPanel(QtWidgets.QWidget):
    def __init__(self, doc, fmt_getter, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.fmt_getter = fmt_getter
        self.oid = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.title = QtWidgets.QLabel("オブジェクトを選択してください")
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        lay.addWidget(self.title)
        self.frame_label = QtWidgets.QLabel("")
        self.frame_label.setObjectName("hint")
        lay.addWidget(self.frame_label)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["項目", "値", "単位"])
        self.tree.setColumnWidth(0, 170)
        self.tree.setColumnWidth(1, 190)
        self.tree.setAlternatingRowColors(True)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked |
                                  QtWidgets.QAbstractItemView.EditKeyPressed)
        self.tree.itemChanged.connect(self._changed)
        lay.addWidget(self.tree, 1)
        note = QtWidgets.QLabel("※ 表示桁数は測定精度を表しません。精度は残差・不確かさ・実測で判断してください。\n"
                                "※ 編集可能な値（白字）はダブルクリックで変更できます。ベクトルは「x, y, z」。")
        note.setWordWrap(True)
        note.setObjectName("hint")
        lay.addWidget(note)
        self._updating = False
        doc.listeners.append(self.on_doc_event)

    def set_document(self, doc):
        try:
            self.doc.listeners.remove(self.on_doc_event)
        except ValueError:
            pass
        self.doc = doc
        doc.listeners.append(self.on_doc_event)
        self.show_object(None)

    def on_doc_event(self, event, ids):
        if event in ("changed", "renamed", "active_cs") and (self.oid in ids or event == "active_cs"):
            self.show_object(self.oid)
        elif event in ("removed", "cleared") and (self.oid in ids or event == "cleared"):
            self.show_object(None)

    def show_object(self, oid):
        self._updating = True
        self.tree.clear()
        self.oid = oid
        o = self.doc.get(oid) if oid else None
        if o is None:
            self.title.setText("オブジェクトを選択してください")
            self.frame_label.setText(f"作業座標系: {self.doc.active_frame_name}")
            self._updating = False
            return
        self.title.setText(o.name)
        self.frame_label.setText(f"表示座標系: {self.doc.active_frame_name}（座標・方向はこの座標系で表示）")
        fmt = self.fmt_getter()
        try:
            props = o.properties(self.doc.active_frame, fmt)
        except Exception:
            log.exception("プロパティ取得エラー")
            props = []
        groups = {}
        for p in props:
            g = groups.get(p.group)
            if g is None:
                g = QtWidgets.QTreeWidgetItem([p.group or "基本", "", ""])
                g.setFlags(QtCore.Qt.ItemIsEnabled)
                gf = g.font(0)
                gf.setBold(True)
                g.setFont(0, gf)
                g.setForeground(0, QtGui.QBrush(QtGui.QColor("#9fb7d9")))
                g.setFirstColumnSpanned(False)
                self.tree.addTopLevelItem(g)
                g.setExpanded(True)
                groups[p.group] = g
            it = QtWidgets.QTreeWidgetItem([p.label, "", p.unit])
            it.setData(0, ROLE_KEY, p.key)
            it.setData(0, ROLE_KIND, p.kind)
            flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
            if p.kind == "bool":
                it.setCheckState(1, QtCore.Qt.Checked if p.value else QtCore.Qt.Unchecked)
                it.setText(1, "はい" if p.value else "いいえ")
                if p.editable:
                    flags |= QtCore.Qt.ItemIsUserCheckable
            else:
                it.setText(1, self._fmt(p, fmt))
                if p.editable and p.kind != "choice":
                    flags |= QtCore.Qt.ItemIsEditable
            if not p.editable:
                it.setForeground(1, QtGui.QBrush(QtGui.QColor("#a9b0b9")))
            it.setFlags(flags)
            it.setToolTip(1, str(p.value))
            g.addChild(it)
            if p.kind == "choice" and p.editable:
                cb = QtWidgets.QComboBox()
                cb.setEditable(True)
                cb.addItems([c for c in (p.choices or [])])
                cb.setCurrentText(str(p.value))
                cb.currentTextChanged.connect(lambda txt, key=p.key: self._set(key, txt))
                self.tree.setItemWidget(it, 1, cb)
        self._updating = False

    def _fmt(self, p, fmt):
        v = p.value
        if v is None:
            return "—"
        if p.kind == "vec3":
            if p.unit == "deg":
                return ", ".join(fmt.angle(x) for x in v)
            if p.unit == "mm":
                return ", ".join(fmt.length(x) for x in v)
            return ", ".join(f"{float(x):.6f}" for x in v)
        if p.kind == "float":
            if p.unit in ("deg",):
                return fmt.angle(v)
            if p.unit in ("mm", "mm²", "mm³"):
                return fmt.length(v)
            return f"{float(v):.6g}"
        return str(v)

    def _set(self, key, value):
        if self._updating or self.oid is None:
            return
        try:
            self.doc.set_property(self.oid, key, value)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "入力エラー", str(e))
            self.show_object(self.oid)

    def _changed(self, it, col):
        if self._updating or col != 1:
            return
        key = it.data(0, ROLE_KEY)
        kind = it.data(0, ROLE_KIND)
        if not key:
            return
        try:
            if kind == "bool":
                value = it.checkState(1) == QtCore.Qt.Checked
            elif kind == "vec3":
                value = _parse_vec(it.text(1))
            elif kind == "float":
                value = float(it.text(1).replace(",", ""))
            elif kind == "int":
                value = int(it.text(1))
            else:
                value = it.text(1)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "入力エラー", f"数値を正しく入力してください。\n{e}")
            QtCore.QTimer.singleShot(0, lambda: self.show_object(self.oid))
            return
        oid = self.oid
        QtCore.QTimer.singleShot(0, lambda: self._set_deferred(oid, key, value))

    def _set_deferred(self, oid, key, value):
        if oid != self.oid:
            return
        self._set(key, value)
