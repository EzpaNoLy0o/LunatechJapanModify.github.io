"""ダイアログ群."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..io.mesh_io import UNIT_LABELS, UNIT_TO_MM


class ParamDialog(QtWidgets.QDialog):
    """汎用パラメータ入力ダイアログ.

    fields: [(key, label, kind, default, opts), ...]
      kind: 'float' | 'int' | 'bool' | 'choice' | 'text' | 'info'
      opts: float/int → dict(min, max, decimals, step, suffix) / choice → list[(value, label)]
    """

    def __init__(self, title, fields, parent=None, description: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(self)
        if description:
            d = QtWidgets.QLabel(description)
            d.setWordWrap(True)
            d.setObjectName("hint")
            lay.addWidget(d)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)
        self.widgets = {}
        for key, label, kind, default, opts in fields:
            opts = opts or {}
            if kind == "float":
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(opts.get("decimals", 3))
                w.setRange(opts.get("min", -1e9), opts.get("max", 1e9))
                w.setSingleStep(opts.get("step", 0.1))
                if opts.get("suffix"):
                    w.setSuffix(" " + opts["suffix"])
                w.setValue(float(default))
            elif kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(int(opts.get("min", -10 ** 9)), int(opts.get("max", 10 ** 9)))
                w.setValue(int(default))
                if opts.get("suffix"):
                    w.setSuffix(" " + opts["suffix"])
            elif kind == "bool":
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(default))
            elif kind == "choice":
                w = QtWidgets.QComboBox()
                for val, lab in opts:
                    w.addItem(lab, val)
                idx = max(0, w.findData(default))
                w.setCurrentIndex(idx)
            elif kind == "info":
                w = QtWidgets.QLabel(str(default))
                w.setWordWrap(True)
            else:
                w = QtWidgets.QLineEdit(str(default))
            self.widgets[key] = (kind, w)
            form.addRow(label, w)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("実行")
        bb.button(QtWidgets.QDialogButtonBox.Cancel).setText("キャンセル")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self) -> dict:
        out = {}
        for key, (kind, w) in self.widgets.items():
            if kind in ("float", "int"):
                out[key] = w.value()
            elif kind == "bool":
                out[key] = w.isChecked()
            elif kind == "choice":
                out[key] = w.currentData()
            elif kind == "info":
                continue
            else:
                out[key] = w.text()
        return out

    @staticmethod
    def ask(parent, title, fields, description=""):
        d = ParamDialog(title, fields, parent, description)
        if d.exec() == QtWidgets.QDialog.Accepted:
            return d.values()
        return None


class ImportUnitDialog(QtWidgets.QDialog):
    """読込データの単位確認（STL等は単位情報を持たない）."""

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("単位の確認")
        lay = QtWidgets.QVBoxLayout(self)
        size = data.bbox_size()
        info = QtWidgets.QLabel(
            f"<b>{data.source_path}</b><br>形式: {data.file_format} / "
            + (f"頂点 {data.n_vertices:,} / 面 {data.n_faces:,}" if data.kind == "mesh" else f"点 {data.n_vertices:,}")
            + f"<br>外形寸法（ファイルの数値）: {size[0]:.4g} × {size[1]:.4g} × {size[2]:.4g}")
        info.setWordWrap(True)
        lay.addWidget(info)
        msg = QtWidgets.QLabel("このファイル形式には単位情報が含まれていません。数値の単位を選択してください。\n"
                               + data.unit_message)
        msg.setWordWrap(True)
        msg.setObjectName("warn" if data.unit_guess != "mm" else "hint")
        lay.addWidget(msg)
        for w in data.warnings:
            wl = QtWidgets.QLabel("⚠ " + w)
            wl.setObjectName("warn")
            wl.setWordWrap(True)
            lay.addWidget(wl)
        form = QtWidgets.QFormLayout()
        self.combo = QtWidgets.QComboBox()
        for k, lab in UNIT_LABELS.items():
            self.combo.addItem(lab, k)
        self.combo.setCurrentIndex(self.combo.findData(data.unit_guess))
        form.addRow("ファイルの単位", self.combo)
        self.result_label = QtWidgets.QLabel()
        form.addRow("mm 換算後の寸法", self.result_label)
        lay.addLayout(form)
        self.size = size
        self.combo.currentIndexChanged.connect(self._upd)
        self._upd()
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("読み込む")
        bb.button(QtWidgets.QDialogButtonBox.Cancel).setText("キャンセル")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _upd(self):
        s = self.size * UNIT_TO_MM[self.combo.currentData()]
        self.result_label.setText(f"{s[0]:.3f} × {s[1]:.3f} × {s[2]:.3f} mm")

    def unit(self):
        return self.combo.currentData()


class ReportDialog(QtWidgets.QDialog):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 460)
        lay = QtWidgets.QVBoxLayout(self)
        te = QtWidgets.QPlainTextEdit(text)
        te.setReadOnly(True)
        lay.addWidget(te)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.Close).setText("閉じる")
        lay.addWidget(bb)


class ClipDialog(QtWidgets.QDialog):
    """断面表示（クリッピング）: 作業座標系の X/Y/Z 平面でモデルを切って内部を見る."""

    def __init__(self, viewer, doc, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.doc = doc
        self.setWindowTitle("断面表示")
        self.setModal(False)
        lay = QtWidgets.QFormLayout(self)
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(["X (作業座標系)", "Y (作業座標系)", "Z (作業座標系)"])
        self.flip = QtWidgets.QCheckBox("反対側を表示")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(-1e6, 1e6)
        self.spin.setDecimals(3)
        self.spin.setSuffix(" mm")
        lay.addRow("断面の方向", self.axis)
        lay.addRow("位置", self.slider)
        lay.addRow("位置 (数値)", self.spin)
        lay.addRow("", self.flip)
        btn = QtWidgets.QPushButton("断面表示を解除して閉じる")
        btn.clicked.connect(self.close)
        lay.addRow(btn)
        self.axis.currentIndexChanged.connect(self._range)
        self.slider.valueChanged.connect(self._slider)
        self.spin.valueChanged.connect(self._apply)
        self.flip.toggled.connect(self._apply)
        self._range()

    def _range(self):
        pts = self.doc.all_points_for_bounds()
        f = self.doc.active_frame
        k = self.axis.currentIndex()
        if pts is None:
            self.lo, self.hi = -100.0, 100.0
        else:
            L = f.to_local(pts)
            self.lo, self.hi = float(L[:, k].min()), float(L[:, k].max())
        self._slider(self.slider.value())

    def _slider(self, v):
        pos = self.lo + (self.hi - self.lo) * v / 1000.0
        self.spin.blockSignals(True)
        self.spin.setValue(pos)
        self.spin.blockSignals(False)
        self._apply()

    def _apply(self, *_):
        f = self.doc.active_frame
        k = self.axis.currentIndex()
        n = f.R[:, k] * (1 if self.flip.isChecked() else -1)
        q = np.zeros(3)
        q[k] = self.spin.value()
        self.viewer.set_clip(f.to_world(q), n)

    def closeEvent(self, e):
        self.viewer.set_clip(None, None)
        super().closeEvent(e)


class MountPointsDialog(QtWidgets.QDialog):
    """取付点一覧（サブフレーム等）: 基準点・穴中心・軸の座標を作業座標系で一覧表示."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        doc = win.doc
        self.setWindowTitle("取付点一覧（作業座標系: " + doc.active_frame_name + "）")
        self.resize(980, 520)
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("基準点・穴中心・基準軸・円筒軸を一覧表示します。分類は各要素のプロパティで設定できます。"
                                "⚠ は安全上重要で実測未確認の要素です。必ず実測で確認してください。")
        note.setWordWrap(True)
        note.setObjectName("hint")
        lay.addWidget(note)
        self.table = QtWidgets.QTableWidget()
        cols = ["名前", "種類", "分類", "X", "Y", "Z", "方向X", "方向Y", "方向Z", "直径", "安全上重要", "実測確認"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.ids = []
        f = doc.active_frame
        fmt = win.fmt()
        rows = []
        for o in doc.objects(("point", "hole", "axis", "cylinder")):
            if o.kind == "point":
                p, d, dia = o.position, None, None
            elif o.kind == "hole":
                p, d, dia = o.center, o.axis, o.diameter
            elif o.kind == "axis":
                p, d, dia = o.point, o.direction, None
            else:
                a, b = o.end_points()
                p, d, dia = 0.5 * (a + b), o.axis_dir, o.diameter
            pl = f.to_local(p)
            dl = f.dir_to_local(d) if d is not None else None
            rows.append((o, pl, dl, dia))
        self.table.setRowCount(len(rows))
        from ..model.objects import KIND_LABELS
        for r, (o, pl, dl, dia) in enumerate(rows):
            vals = [("⚠ " if o.critical and not o.verified else "") + o.name, KIND_LABELS[o.kind],
                    getattr(o, "category", ""), fmt.length(pl[0]), fmt.length(pl[1]), fmt.length(pl[2])]
            vals += [f"{x:.5f}" for x in dl] if dl is not None else ["", "", ""]
            vals += [fmt.length(dia) if dia else "", "はい" if o.critical else "",
                     "済" if o.verified else ("未確認" if o.critical else "")]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if c >= 3:
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(r, c, it)
            self.ids.append(o.id)
        self.table.resizeColumnsToContents()
        bl = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("座標一覧をCSV出力…")
        b2 = QtWidgets.QPushButton("点間距離マトリクスをCSV出力…")
        b3 = QtWidgets.QPushButton("選択行を安全上重要に設定")
        b4 = QtWidgets.QPushButton("閉じる")
        b1.clicked.connect(lambda: win.cmd_export_csv_features(self.ids))
        b2.clicked.connect(lambda: win.cmd_export_distance_matrix(self.ids))
        b3.clicked.connect(self._mark_critical)
        b4.clicked.connect(self.accept)
        for b in (b1, b2, b3):
            bl.addWidget(b)
        bl.addStretch(1)
        bl.addWidget(b4)
        lay.addLayout(bl)

    def _mark_critical(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        ids = [self.ids[r] for r in rows]
        if ids:
            self.win.doc.modify(ids, lambda o: setattr(o, "critical", True), "安全上重要に設定")
            for r in rows:
                self.table.item(r, 10).setText("はい")
