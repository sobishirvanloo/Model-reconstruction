"""Application themes (light / dark).

Every colour of the UI is defined once here, as a palette of tokens, and
turned into an application-wide Qt style sheet + QPalette.  Widgets carry
*roles* instead of their own style sheets:

    QPushButton[variant="primary"]   main action of a panel (filled)
    QPushButton[variant="link"]      secondary, text-only action
    QPushButton[variant="danger"]    destructive action (red outline)
    QLabel[role="section"]           section header in a side panel
    QLabel[role="hint"]              small explanatory text
    QFrame[role="panel"]             side panel background
    QPlainTextEdit#console           log console
    QFrame#mouseGuide                mouse help strip under a 3D view

The 3D viewports read their colours from ``view_colors``.
"""

import os
import tempfile

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPalette, QPolygonF

LIGHT = "light"
DARK = "dark"

PALETTES = {
    LIGHT: {
        "window": "#f3f5f8", "surface": "#ffffff", "surface_alt": "#eef2f6",
        "border": "#d3dbe4", "border_strong": "#b7c3d0",
        "text": "#1b2430", "text_muted": "#5d6b7b", "text_disabled": "#9aa6b2",
        "primary": "#0a78c2", "primary_hover": "#0868a8", "primary_pressed": "#06548a",
        "on_primary": "#ffffff", "selection": "#d4e9f8", "selection_text": "#0b3b5c",
        "success": "#1d8f4e", "warning": "#b86e00", "error": "#c93535",
        "console_bg": "#f8fafc", "console_text": "#24303d",
        "tooltip_bg": "#1b2430", "tooltip_text": "#ffffff",
    },
    DARK: {
        "window": "#13171d", "surface": "#1b2129", "surface_alt": "#242c36",
        "border": "#313c49", "border_strong": "#445263",
        "text": "#e5eaf0", "text_muted": "#9ba8b6", "text_disabled": "#5d6976",
        "primary": "#2e9be0", "primary_hover": "#4aaeea", "primary_pressed": "#1f80bd",
        "on_primary": "#ffffff", "selection": "#1f4462", "selection_text": "#e5eaf0",
        "success": "#3cc47c", "warning": "#e5a33a", "error": "#ef6363",
        "console_bg": "#0f1318", "console_text": "#cfd8e2",
        "tooltip_bg": "#e5eaf0", "tooltip_text": "#13171d",
    },
}

# VTK colours are 0..1 RGB tuples
VIEW_COLORS = {
    LIGHT: {
        "background_top": (1.0, 1.0, 1.0),
        "background_bottom": (0.85, 0.89, 0.93),
        "mesh": (0.27, 0.56, 0.84),
        "points": (0.18, 0.27, 0.37),
        "text": (0.25, 0.30, 0.37),
    },
    DARK: {
        "background_top": (0.17, 0.21, 0.26),
        "background_bottom": (0.07, 0.09, 0.11),
        "mesh": (0.36, 0.66, 0.95),
        "points": (0.86, 0.89, 0.93),
        "text": (0.80, 0.85, 0.90),
    },
}


def view_colors(name):
    return VIEW_COLORS[name]


def status_color(name, level):
    """Text colour for a status message level (info/success/warning/error)."""
    palette = PALETTES[name]
    return {"success": palette["success"], "warning": palette["warning"],
            "error": palette["error"]}.get(level, palette["text"])


def build_palette(p):
    palette = QPalette()
    roles = {
        QPalette.Window: p["window"], QPalette.WindowText: p["text"],
        QPalette.Base: p["surface"], QPalette.AlternateBase: p["surface_alt"],
        QPalette.Text: p["text"], QPalette.Button: p["surface_alt"],
        QPalette.ButtonText: p["text"], QPalette.Highlight: p["primary"],
        QPalette.HighlightedText: p["on_primary"],
        QPalette.ToolTipBase: p["tooltip_bg"], QPalette.ToolTipText: p["tooltip_text"],
        QPalette.PlaceholderText: p["text_muted"], QPalette.Link: p["primary"],
        QPalette.Mid: p["border"], QPalette.Dark: p["border_strong"],
        QPalette.Light: p["surface"], QPalette.Midlight: p["surface_alt"],
        QPalette.Shadow: p["border_strong"], QPalette.BrightText: p["error"],
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(p["text_disabled"]))
    return palette


def _arrow_images(p):
    """Small arrow PNGs in the palette colours (Qt style sheets cannot draw
    triangles, and need file paths for images)."""
    folder = os.path.join(tempfile.gettempdir(), "model-reconstruction-ui")
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for name, color in (("", p["text_muted"]), ("_disabled", p["border_strong"])):
        for direction in ("up", "down"):
            path = os.path.join(
                folder, f"{direction}{name}_{color.lstrip('#')}.png")
            if not os.path.exists(path):
                image = QImage(18, 12, QImage.Format_ARGB32)
                image.fill(Qt.transparent)
                painter = QPainter(image)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(color))
                tip, base = (1, 11) if direction == "up" else (11, 1)
                painter.drawPolygon(QPolygonF(
                    [QPointF(1, base), QPointF(17, base), QPointF(9, tip)]))
                painter.end()
                image.save(path)
            paths[f"{direction}{name}"] = path.replace("\\", "/")
    return paths


def build_qss(p):
    arrows = _arrow_images(p)
    return f"""
* {{ font-size: 13px; }}
QWidget {{ color: {p['text']}; }}
QMainWindow, QDialog {{ background-color: {p['window']}; }}
QWidget#centralwidget, QFrame#frame {{ background-color: {p['window']}; }}

/* ---------- side panels ---------- */
QFrame[role="panel"] {{
    background-color: {p['surface']}; border: 1px solid {p['border']};
    border-radius: 10px; }}
QFrame[role="panel"] QLabel {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QLabel[role="section"] {{
    color: {p['text_muted']}; font-size: 11px; font-weight: bold;
    letter-spacing: 1px; padding: 10px 0 2px 0;
    border-bottom: 1px solid {p['border']}; }}
QLabel[role="hint"] {{ color: {p['text_muted']}; font-size: 12px; }}
QLabel[role="file"] {{ color: {p['text']}; font-weight: bold; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{ border: none; background: {p['window']}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {p['text_muted']};
    padding: 9px 22px; margin-right: 4px; border: none;
    border-bottom: 3px solid transparent; }}
QTabBar::tab:hover {{ color: {p['text']}; }}
QTabBar::tab:selected {{ color: {p['primary']}; border-bottom: 3px solid {p['primary']}; }}

/* ---------- buttons ---------- */
QPushButton {{
    background-color: {p['surface_alt']}; color: {p['text']};
    border: 1px solid {p['border']}; border-radius: 7px;
    padding: 7px 14px; font-weight: bold; }}
QPushButton:hover {{ border-color: {p['primary']}; color: {p['primary']}; }}
QPushButton:pressed {{ background-color: {p['selection']}; }}
QPushButton:disabled {{ color: {p['text_disabled']}; border-color: {p['border']};
                        background-color: {p['surface_alt']}; }}
QPushButton[variant="primary"] {{
    background-color: {p['primary']}; color: {p['on_primary']}; border: none; }}
QPushButton[variant="primary"]:hover {{ background-color: {p['primary_hover']}; color: {p['on_primary']}; }}
QPushButton[variant="primary"]:pressed {{ background-color: {p['primary_pressed']}; }}
QPushButton[variant="primary"]:disabled {{ background-color: {p['border']}; color: {p['text_disabled']}; }}
QPushButton[variant="link"] {{
    background: transparent; border: none; color: {p['primary']};
    font-weight: normal; padding: 2px 4px; text-decoration: underline; }}
QPushButton[variant="link"]:disabled {{ color: {p['text_disabled']}; }}
QPushButton[variant="danger"] {{ color: {p['error']}; border-color: {p['error']}; }}
QPushButton[variant="danger"]:hover {{ background-color: {p['error']}; color: {p['on_primary']}; }}
QPushButton[variant="danger"]:disabled {{ color: {p['text_disabled']}; border-color: {p['border']}; }}

/* ---------- mouse guide under the 3D views ---------- */
QFrame#mouseGuide {{
    background-color: {p['surface']}; border: none;
    border-top: 1px solid {p['border']}; }}
QLabel[role="mouseguide"] {{
    color: {p['text_muted']}; font-size: 8pt; background: transparent; border: none; }}
QFrame#mouseGuide QToolButton {{
    color: {p['text_muted']}; border: none; padding: 0 4px; font-size: 8pt; }}
QFrame#mouseGuide QToolButton:hover {{ color: {p['error']}; }}

/* ---------- inputs ---------- */
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {{
    background-color: {p['surface']}; color: {p['text']};
    border: 1px solid {p['border_strong']}; border-radius: 6px;
    padding: 4px 8px; min-height: 22px;
    selection-background-color: {p['primary']}; selection-color: {p['on_primary']}; }}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {p['primary']}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled, QComboBox:disabled {{
    color: {p['text_disabled']}; background-color: {p['surface_alt']}; }}
QAbstractSpinBox {{ padding-right: 20px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 20px; border: none; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {p['surface_alt']}; }}
QAbstractSpinBox::up-arrow {{ image: url({arrows['up']}); width: 9px; height: 6px; }}
QAbstractSpinBox::down-arrow {{ image: url({arrows['down']}); width: 9px; height: 6px; }}
QAbstractSpinBox::up-arrow:disabled {{ image: url({arrows['up_disabled']}); }}
QAbstractSpinBox::down-arrow:disabled {{ image: url({arrows['down_disabled']}); }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
                        width: 24px; border: none; }}
QComboBox::down-arrow {{ image: url({arrows['down']}); width: 9px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url({arrows['down_disabled']}); }}
QComboBox QAbstractItemView {{
    background-color: {p['surface']}; color: {p['text']};
    border: 1px solid {p['border_strong']};
    selection-background-color: {p['selection']}; selection-color: {p['selection_text']}; }}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox:disabled {{ color: {p['text_disabled']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {p['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p['primary']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {p['surface']}; border: 2px solid {p['primary']};
    width: 14px; height: 14px; margin: -7px 0; border-radius: 9px; }}
QSlider::handle:horizontal:disabled {{ border-color: {p['border_strong']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {p['border_strong']}; }}

/* ---------- console / tables / lists ---------- */
QPlainTextEdit#console {{
    background-color: {p['console_bg']}; color: {p['console_text']};
    border: 1px solid {p['border']}; border-radius: 8px;
    font-family: "DejaVu Sans Mono", Consolas, Menlo, monospace; font-size: 12px;
    padding: 4px; }}
QTableWidget, QTableView, QTreeView, QListView {{
    background-color: {p['surface']}; alternate-background-color: {p['surface_alt']};
    color: {p['text']}; gridline-color: {p['border']};
    border: 1px solid {p['border']}; border-radius: 6px;
    selection-background-color: {p['selection']}; selection-color: {p['selection_text']}; }}
QHeaderView::section {{
    background-color: {p['surface_alt']}; color: {p['text_muted']};
    border: none; border-bottom: 1px solid {p['border']};
    padding: 5px 8px; font-weight: bold; }}
QGroupBox {{
    border: 1px solid {p['border']}; border-radius: 8px;
    margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {p['text_muted']}; }}

/* ---------- menus / bars ---------- */
QMenuBar {{ background-color: {p['surface']}; border-bottom: 1px solid {p['border']}; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
QMenuBar::item:selected {{ background-color: {p['selection']}; color: {p['selection_text']};
                           border-radius: 4px; }}
QMenu {{ background-color: {p['surface']}; border: 1px solid {p['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 28px 6px 24px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {p['selection']}; color: {p['selection_text']}; }}
QMenu::item:disabled {{ color: {p['text_disabled']}; }}
QMenu::separator {{ height: 1px; background: {p['border']}; margin: 4px 8px; }}
QStatusBar {{ background-color: {p['surface']}; border-top: 1px solid {p['border']};
              color: {p['text_muted']}; }}
QStatusBar QLabel {{ color: {p['text_muted']}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}
QToolBar {{ background: {p['surface']}; border: 1px solid {p['border']};
            border-bottom: none; border-top-left-radius: 8px;
            border-top-right-radius: 8px; padding: 2px 6px; spacing: 2px; }}
QToolButton {{ background: transparent; border: 1px solid transparent;
               border-radius: 5px; padding: 4px 8px; color: {p['text']}; }}
QToolButton:hover {{ border-color: {p['border_strong']}; background: {p['surface_alt']}; }}
QToolButton:pressed {{ background: {p['selection']}; }}
QProgressBar {{ background: {p['surface_alt']}; border: 1px solid {p['border']};
                border-radius: 5px; text-align: center; max-height: 14px;
                min-width: 140px; color: {p['text_muted']}; font-size: 11px; }}
QProgressBar::chunk {{ background-color: {p['primary']}; border-radius: 4px; }}
QToolTip {{ background-color: {p['tooltip_bg']}; color: {p['tooltip_text']};
            border: none; padding: 6px 8px; border-radius: 4px; }}

/* ---------- scroll bars ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p['border_strong']}; border-radius: 4px;
                               min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p['border_strong']}; border-radius: 4px;
                                 min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


def apply_theme(app, name):
    """Apply a theme to the whole application; returns its palette dict."""
    palette = PALETTES[name]
    app.setStyle("Fusion")
    app.setPalette(build_palette(palette))
    app.setStyleSheet(build_qss(palette))
    return palette
