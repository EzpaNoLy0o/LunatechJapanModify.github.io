"""ダークテーマ (Qt スタイルシート) と 3D ビューの配色。"""

from __future__ import annotations

# --- UI 配色 -----------------------------------------------------------------
BG = "#12151a"          # ウィンドウ背景
PANEL = "#1a1e25"       # パネル
PANEL_ALT = "#20252e"   # 入力欄・ヘッダ
BORDER = "#2c333f"
TEXT = "#d7dce4"
TEXT_DIM = "#8a93a3"
ACCENT = "#22b8e6"      # 操作可能な要素の強調 (シアン)
ACCENT_DARK = "#157a9a"
SELECT = "#ff9f1c"      # 3D 上の選択ハイライト (オレンジ)
OK = "#4cc38a"
WARN = "#e5b93c"
ERROR = "#ef5f5f"

# --- 3D ビュー配色 (RGB 0-1 または 16 進) -------------------------------------
VIEW_THEMES = {
    "dark": {
        "bottom": "#0d0f13", "top": "#262c38",
        "edge": "#0b0d10", "mesh_edge": "#2a313c", "grid": "#3a4352", "grid_major": "#56627a",
        "text": "#d7dce4", "bbox": "#22b8e6", "measure": "#ffd166",
    },
    "light": {
        "bottom": "#dfe5ec", "top": "#ffffff",
        "edge": "#2b2f36", "mesh_edge": "#5a6475", "grid": "#c3cad4", "grid_major": "#98a2b3",
        "text": "#1b1f26", "bbox": "#0b7fa8", "measure": "#c2410c",
    },
}
SELECT_RGB = (1.0, 0.62, 0.11)
FACE_RGB = (0.13, 0.85, 0.95)
OPEN_EDGE_RGB = (1.0, 0.25, 0.25)
NON_MANIFOLD_RGB = (1.0, 0.2, 0.9)

FONT_FAMILY = '"Yu Gothic UI", "Meiryo UI", "Segoe UI", "Noto Sans CJK JP", sans-serif'

STYLESHEET = f"""
* {{ font-family: {FONT_FAMILY}; font-size: 9.5pt; }}
QMainWindow, QDialog, QWidget {{ background: {BG}; color: {TEXT}; }}
QToolTip {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}

QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {PANEL_ALT}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 22px; border-radius: 3px; }}
QMenu::item:selected {{ background: {ACCENT_DARK}; }}
QMenu::item:disabled {{ color: {TEXT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER}; spacing: 2px; padding: 3px 6px; }}
QToolBar::separator {{ width: 1px; background: {BORDER}; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 5px 9px; }}
QToolButton:hover {{ background: {PANEL_ALT}; border-color: {BORDER}; }}
QToolButton:checked {{ background: {ACCENT_DARK}; border-color: {ACCENT}; color: white; }}
QToolButton:pressed {{ background: {ACCENT_DARK}; }}

QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 5px; }}
QSplitter::handle:vertical {{ height: 5px; }}
QSplitter::handle:hover {{ background: {ACCENT_DARK}; }}

QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; margin-top: 18px; padding: 8px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: 2px; color: {ACCENT}; font-weight: bold; }}

QLabel {{ background: transparent; }}
QLabel[role="dim"] {{ color: {TEXT_DIM}; }}
QLabel[role="value"] {{ color: {TEXT}; font-family: "Consolas", "Cascadia Mono", monospace; }}
QLabel[role="headline"] {{ color: {ACCENT}; font-weight: bold; font-size: 10.5pt; }}
QLabel[role="notice"] {{ color: {WARN}; }}

QPushButton {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_DARK}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton[role="primary"] {{ background: {ACCENT_DARK}; border-color: {ACCENT}; color: white; font-weight: bold; }}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px; min-height: 20px;
    selection-background-color: {ACCENT_DARK};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{ color: {TEXT_DIM}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DARK}; }}

QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator, QTreeView::indicator {{ width: 14px; height: 14px; border: 1px solid {TEXT_DIM}; border-radius: 3px; background: {PANEL_ALT}; }}
QCheckBox::indicator:checked, QTreeView::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: {TEXT}; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}

QTreeWidget, QTableWidget, QPlainTextEdit, QTextEdit {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; alternate-background-color: {PANEL_ALT};
    selection-background-color: {ACCENT_DARK}; selection-color: white; gridline-color: {BORDER};
}}
QTreeWidget::item, QTableWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {ACCENT_DARK}; color: white; }}
QHeaderView::section {{ background: {PANEL_ALT}; color: {TEXT_DIM}; border: none; border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; padding: 4px 6px; }}
QPlainTextEdit {{ font-family: "Consolas", "Cascadia Mono", "Noto Sans Mono CJK JP", monospace; font-size: 9pt; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; background: {PANEL}; }}
QTabBar::tab {{ background: transparent; color: {TEXT_DIM}; padding: 6px 14px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_DIM}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""
