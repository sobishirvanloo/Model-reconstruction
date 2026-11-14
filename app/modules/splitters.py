"""Resizable tab layout: drag the dividers between the 3D view, the side
panel and the console.

``make_resizable`` takes the three widgets out of the tab's grid layout
and puts them in two ``QSplitter``s::

    +----------------------+---------+
    |                      |         |
    |       3D view        |  panel  |      <- horizontal splitter
    |                      |         |
    +======================+=========+      <- vertical splitter
    |            console             |
    +--------------------------------+

The view gets every extra pixel when the window grows (stretch), the
panel and the console keep the size the user dragged them to.  Sizes are
saved per tab (``save_state`` / ``restore_state``).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QSplitter

from modules import settings

PANEL_WIDTH = 330          # default side panel width
PANEL_MIN, PANEL_MAX = 240, 720
CONSOLE_HEIGHT = 200       # default height of the log + object info row
INFO_WIDTH = 460           # default width of the object info panel
CONSOLE_MIN = 50
VIEW_MIN_WIDTH, VIEW_MIN_HEIGHT = 260, 180
HANDLE = 7                 # divider thickness (px)


def _grid_and_position(layout, widget):
    """(grid layout holding ``widget``, its index) - searches nested
    layouts."""
    index = layout.indexOf(widget)
    if index >= 0:
        return layout, index
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        if child is not None:
            found = _grid_and_position(child, widget)
            if found:
                return found
    return None


def _widget_at(grid, row, column):
    """Widget in a grid cell (``None`` when empty)."""
    item = grid.itemAtPosition(row, column)
    return item.widget() if item is not None else None


class TabSplitter:
    """The two splitters of one tab."""

    def __init__(self, key, view, panel, console, info=None):
        self.key = key
        grid, index = _grid_and_position(view.parentWidget().layout(), view)
        parent = view.parentWidget()
        for widget in (view, panel, console):
            grid.removeWidget(widget)

        # Let the dragged sizes win over the .ui fixed sizes
        panel.setMinimumWidth(PANEL_MIN)
        panel.setMaximumWidth(PANEL_MAX)
        inner = getattr(panel, "widget", lambda: None)()
        if inner is not None:                 # scroll area: the panel inside
            inner.setMaximumWidth(16777215)
            inner.setMinimumWidth(0)
        console.setMinimumHeight(CONSOLE_MIN)
        console.setMaximumHeight(16777215)
        view.setMinimumSize(VIEW_MIN_WIDTH, VIEW_MIN_HEIGHT)

        self.horizontal = QSplitter(Qt.Horizontal, parent)
        self.horizontal.addWidget(view)
        self.horizontal.addWidget(panel)
        self.horizontal.setStretchFactor(0, 1)
        self.horizontal.setStretchFactor(1, 0)

        # Bottom row: log | object info
        self.bottom = QSplitter(Qt.Horizontal, parent)
        self.bottom.addWidget(console)
        if info is not None:
            info.setMinimumWidth(220)
            self.bottom.addWidget(info)
        self.bottom.setStretchFactor(0, 1)
        self.bottom.setStretchFactor(1, 0)

        self.vertical = QSplitter(Qt.Vertical, parent)
        self.vertical.addWidget(self.horizontal)
        self.vertical.addWidget(self.bottom)
        self.vertical.setStretchFactor(0, 1)
        self.vertical.setStretchFactor(1, 0)

        splitters = [(self.horizontal, "panel"), (self.vertical, "console")]
        if self.bottom.count() > 1:
            splitters.append((self.bottom, "info"))
        for splitter, name in splitters:
            splitter.setObjectName(f"{key}_{name}_splitter")
            splitter.setHandleWidth(HANDLE)
            splitter.setChildrenCollapsible(False)
            splitter.setOpaqueResize(True)
            handle = splitter.handle(1)
            handle.setToolTip("Drag to resize - double-click for the "
                              "default size")
            handle.mouseDoubleClickEvent = (
                lambda event, s=splitter: self.reset(s))

        if isinstance(grid, QGridLayout):
            grid.addWidget(self.vertical, 0, 0, 1, -1)
        else:
            grid.insertWidget(index, self.vertical)
        self.reset()

    def reset(self, only=None):
        """Default sizes (the view takes what is left)."""
        if only in (None, self.horizontal):
            total = max(self.horizontal.width(), 1200)
            self.horizontal.setSizes([total - PANEL_WIDTH, PANEL_WIDTH])
        if only in (None, self.vertical):
            total = max(self.vertical.height(), 800)
            self.vertical.setSizes([total - CONSOLE_HEIGHT, CONSOLE_HEIGHT])
        if only in (None, self.bottom) and self.bottom.count() > 1:
            total = max(self.bottom.width(), 1200)
            self.bottom.setSizes([total - INFO_WIDTH, INFO_WIDTH])

    def save_state(self):
        settings.set_layout_state(self.key, {
            "panel": bytes(self.horizontal.saveState()),
            "console": bytes(self.vertical.saveState()),
            "info": bytes(self.bottom.saveState()),
        })

    def restore_state(self):
        state = settings.layout_state(self.key)
        if state.get("panel"):
            self.horizontal.restoreState(state["panel"])
        if state.get("console"):
            self.vertical.restoreState(state["console"])
        if state.get("info"):
            self.bottom.restoreState(state["info"])


def make_resizable(key, view, console, info=None):
    """Splitters for a tab whose grid holds ``view`` with the side panel
    to its right (the panel itself or the scroll area wrapping it) and
    ``console`` below (as in ``main.ui``)."""
    grid, index = _grid_and_position(view.parentWidget().layout(), view)
    row, column, _, _ = grid.getItemPosition(index)
    panel = _widget_at(grid, row, column + 1)
    splitter = TabSplitter(key, view, panel, console, info)
    splitter.restore_state()
    return splitter
