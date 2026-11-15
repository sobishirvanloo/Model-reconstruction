# MeshBench - Copyright (c) 2026 Sobhan Shirvanloo. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""One consistent line-icon set that follows the theme.

Qt's built-in icons (``QStyle.SP_*``) differ per platform and do not
change with the light / dark theme.  Here every icon is a small SVG
drawn with the current theme colour when it is painted, so a theme
switch recolours all of them; the program keeps asking for the usual
``style().standardIcon(QStyle.SP_...)`` - ``IconStyle`` (a proxy style
installed by ``theme.apply_theme``) answers with these icons.

Roles pick the colour: ``text`` (default), ``on_primary`` (white, on the
blue main buttons), ``danger`` (red), ``success``; disabled buttons get
the theme's disabled colour automatically.
"""

from PyQt5.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt5.QtGui import QIcon, QIconEngine, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QProxyStyle, QStyle

# 24 x 24 line drawings ({c} = colour); simple geometric shapes
SHAPES = {
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "save": '<path d="M12 4v11"/><path d="M7 10l5 5 5-5"/><path d="M4 19h16"/>',
    "arrow_right": '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    "arrow_left": '<path d="M19 12H5"/><path d="M11 6l-6 6 6 6"/>',
    "refresh": '<path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v5h-5"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 2.34-5.66"/><path d="M4 4v5h5"/>',
    "home": '<path d="M4 11l8-7 8 7"/><path d="M6 10v10h12V10"/>',
    "play": '<path d="M7 5l12 7-12 7z" fill="{c}"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="{c}"/>',
    "trash": '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6"/>',
    "report": '<path d="M4 20h16"/><path d="M7 16v-5M12 16V7M17 16v-8"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7"/>',
    "warning": '<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17.5v.5"/>',
    "discard": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="{c}"/>',
    "align": '<path d="M4 12h16"/><path d="M8 8l-4 4 4 4"/><path d="M16 8l4 4-4 4"/>',
    "pose": '<path d="M12 3v18M3 12h18"/><path d="M12 3l-2.5 2.5M12 3l2.5 2.5M21 12l-2.5-2.5M21 12l-2.5 2.5"/><circle cx="12" cy="12" r="2.5"/>',
}

# Qt standard icons replaced everywhere (message-box icons are left alone)
STANDARD = {
    QStyle.SP_DialogOpenButton: "open",
    QStyle.SP_DialogSaveButton: "save",
    QStyle.SP_ArrowForward: "arrow_right",
    QStyle.SP_ArrowBack: "arrow_left",
    QStyle.SP_BrowserReload: "refresh",
    QStyle.SP_MediaPlay: "play",
    QStyle.SP_MediaStop: "stop",
    QStyle.SP_TrashIcon: "trash",
    QStyle.SP_FileDialogDetailedView: "report",
    QStyle.SP_FileDialogContentsView: "copy",
    QStyle.SP_DialogResetButton: "reset",
    QStyle.SP_DialogApplyButton: "check",
    QStyle.SP_DialogDiscardButton: "discard",
    QStyle.SP_DialogCloseButton: "close",
}
# also for buttons that ask for them explicitly (``decorate``)
BUTTON_ONLY = {QStyle.SP_MessageBoxWarning: "warning"}

_colors = {
    "text": "#1b2430", "on_primary": "#ffffff", "danger": "#c93535",
    "success": "#1d8f4e", "disabled": "#9aa6b2", "primary": "#0a78c2",
}
_cache = {}


def set_palette(palette):
    """Theme colours (``theme.PALETTES[name]``); icons repaint with them."""
    _colors.update(text=palette["text"], on_primary=palette["on_primary"],
                   danger=palette["error"], success=palette["success"],
                   disabled=palette["text_disabled"], primary=palette["primary"])
    _cache.clear()


def _svg(name, color):
    body = SHAPES[name].replace("{c}", color)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


class _Engine(QIconEngine):
    """Draws the icon in the colour of its role at paint time."""

    def __init__(self, name, role):
        super().__init__()
        self.name, self.role = name, role

    def _color(self, mode):
        if mode == QIcon.Disabled:
            return _colors["disabled"]
        return _colors.get(self.role, _colors["text"])

    def pixmap(self, size, mode, state):
        color = self._color(mode)
        key = (self.name, color, size.width(), size.height())
        pixmap = _cache.get(key)
        if pixmap is None:
            pixmap = QPixmap(size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            QSvgRenderer(QByteArray(_svg(self.name, color).encode())).render(
                painter, QRectF(0, 0, size.width(), size.height()))
            painter.end()
            _cache[key] = pixmap
        return pixmap

    def paint(self, painter, rect, mode, state):
        ratio = painter.device().devicePixelRatioF() if painter.device() else 1.0
        size = QSize(round(rect.width() * ratio), round(rect.height() * ratio))
        painter.drawPixmap(rect, self.pixmap(size, mode, state))

    def clone(self):
        return _Engine(self.name, self.role)


def icon(name, role="text"):
    """A themed icon (``SHAPES`` name)."""
    return QIcon(_Engine(name, role))


ROLE_OF_VARIANT = {"primary": "on_primary", "danger": "danger"}


def for_standard(standard, variant=None):
    """Themed icon for a ``QStyle.SP_*`` value (``None`` if unknown);
    ``variant`` = the button's style variant ("primary" / "danger")."""
    name = STANDARD.get(standard) or BUTTON_ONLY.get(standard)
    if name is None:
        return None
    return icon(name, ROLE_OF_VARIANT.get(variant, "text"))


class IconStyle(QProxyStyle):
    """The application style, with the themed icons."""

    def standardIcon(self, standard, option=None, widget=None):
        name = STANDARD.get(standard)
        if name is not None:
            return icon(name)
        return super().standardIcon(standard, option, widget)
