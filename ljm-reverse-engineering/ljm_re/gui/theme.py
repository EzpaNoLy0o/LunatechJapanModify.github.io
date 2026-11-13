"""ダークテーマ（落ち着いたブルー系アクセント）."""
from __future__ import annotations

from PySide6 import QtGui, QtWidgets

ACCENT = "#3d7bd9"
ACCENT_HOVER = "#4f8be6"

DARK_QSS = f"""
QWidget {{ background-color: #23262b; color: #dde1e6; }}
QMainWindow::separator {{ background: #1a1c20; width: 4px; height: 4px; }}
QMenuBar {{ background-color: #1d2024; }}
QMenuBar::item:selected {{ background: #2f3540; }}
QMenu {{ background-color: #262a30; border: 1px solid #3a3f47; }}
QMenu::item:selected {{ background-color: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: #6b717a; }}
QMenu::separator {{ height: 1px; background: #3a3f47; margin: 4px 8px; }}
QToolBar {{ background: #2a2e34; border: none; spacing: 2px; padding: 2px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px 8px; color: #dde1e6; }}
QToolButton:hover {{ background: #343a43; border-color: #46505c; }}
QToolButton:pressed, QToolButton:checked {{ background: {ACCENT}; color: white; }}
QToolButton:disabled {{ color: #5d636b; }}
QTabWidget::pane {{ border: 1px solid #353a41; top: -1px; }}
QTabBar::tab {{ background: #1f2226; color: #b9c0c8; padding: 5px 14px; border: 1px solid #353a41; border-bottom: none; }}
QTabBar::tab:selected {{ background: #2a2e34; color: white; border-top: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: white; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: #1d2024; padding: 4px 6px; border-bottom: 1px solid #353a41; }}
QTreeView, QTableView, QListView, QPlainTextEdit, QTextEdit {{ background-color: #1c1f23; alternate-background-color: #212429;
    border: 1px solid #353a41; selection-background-color: {ACCENT}; selection-color: white; }}
QHeaderView::section {{ background-color: #2a2e34; color: #c7ccd3; padding: 4px; border: none; border-right: 1px solid #353a41; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #1a1d21; border: 1px solid #3f4650; border-radius: 3px; padding: 3px 5px; color: #e8ebef; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: #262a30; selection-background-color: {ACCENT}; }}
QPushButton {{ background: #353b44; border: 1px solid #47505b; border-radius: 4px; padding: 5px 14px; color: #e8ebef; }}
QPushButton:hover {{ background: #3e4550; }}
QPushButton:pressed {{ background: {ACCENT}; }}
QPushButton:default {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #6b717a; }}
QGroupBox {{ border: 1px solid #3a3f47; border-radius: 4px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9fb7d9; }}
QStatusBar {{ background: #1d2024; color: #c7ccd3; }}
QProgressBar {{ background: #1a1d21; border: 1px solid #3f4650; border-radius: 3px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background-color: {ACCENT}; }}
QScrollBar:vertical {{ background: #1f2226; width: 12px; }}
QScrollBar::handle:vertical {{ background: #3e4550; min-height: 24px; border-radius: 4px; }}
QScrollBar:horizontal {{ background: #1f2226; height: 12px; }}
QScrollBar::handle:horizontal {{ background: #3e4550; min-width: 24px; border-radius: 4px; }}
QCheckBox::indicator, QTreeView::indicator {{ width: 14px; height: 14px; }}
QToolTip {{ background: #2f3540; color: #e8ebef; border: 1px solid #4a5260; }}
QLabel#hint {{ color: #9fb7d9; }}
QLabel#warn {{ color: #f0b35a; }}
"""


def apply_theme(app: QtWidgets.QApplication, theme: str = "dark"):
    app.setStyle("Fusion")
    if theme == "dark":
        pal = QtGui.QPalette()
        c = QtGui.QColor
        pal.setColor(QtGui.QPalette.Window, c("#23262b"))
        pal.setColor(QtGui.QPalette.WindowText, c("#dde1e6"))
        pal.setColor(QtGui.QPalette.Base, c("#1c1f23"))
        pal.setColor(QtGui.QPalette.AlternateBase, c("#212429"))
        pal.setColor(QtGui.QPalette.Text, c("#dde1e6"))
        pal.setColor(QtGui.QPalette.Button, c("#353b44"))
        pal.setColor(QtGui.QPalette.ButtonText, c("#e8ebef"))
        pal.setColor(QtGui.QPalette.Highlight, c(ACCENT))
        pal.setColor(QtGui.QPalette.HighlightedText, c("white"))
        pal.setColor(QtGui.QPalette.ToolTipBase, c("#2f3540"))
        pal.setColor(QtGui.QPalette.ToolTipText, c("#e8ebef"))
        pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, c("#6b717a"))
        pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c("#6b717a"))
        app.setPalette(pal)
        app.setStyleSheet(DARK_QSS)
    else:
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")


def japanese_font() -> QtGui.QFont:
    families = set(QtGui.QFontDatabase.families())
    for name in ("Yu Gothic UI", "Meiryo UI", "Meiryo", "Noto Sans CJK JP", "Noto Sans JP",
                 "IPAexGothic", "MS UI Gothic"):
        if name in families:
            f = QtGui.QFont(name)
            f.setPointSizeF(9.5)
            return f
    f = QtGui.QFont()
    f.setPointSizeF(9.5)
    return f
