# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Collapsible sections in the side panels.

Every panel is a vertical list: section headers (``QLabel`` with
``role="section"``, made by ``point_data.add_section``) each followed by
their controls.  ``make_collapsible`` moves the items of each section into
a container right below its header; a click on the header hides / shows
that container.  The widgets keep their own visibility (controls of
another algorithm stay hidden when a section opens again), because only
the container is hidden.  Right-click a header: expand / collapse all.

The open / closed state is saved per tab and section.
"""

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QAction, QMenu, QVBoxLayout, QWidget

from modules import settings

OPEN, CLOSED = "▾", "▸"


def _is_header(widget):
    return widget is not None and widget.property("role") == "section"


class Section:
    """One header + the container of its items."""

    def __init__(self, panel, header, body, title):
        self.panel = panel
        self.header = header
        self.body = body
        self.title = title

    @property
    def collapsed(self):
        return self.body.isHidden()

    def set_collapsed(self, collapsed, save=True):
        self.body.setVisible(not collapsed)
        self.header.setText(f"{CLOSED if collapsed else OPEN}  {self.title}")
        if save:
            settings.set_panel_collapsed(self.panel.key, self.title, collapsed)

    def toggle(self):
        self.set_collapsed(not self.collapsed)


class CollapsiblePanel(QObject):
    """The sections of one panel (kept alive by the panel widget)."""

    def __init__(self, panel_widget, key):
        super().__init__(panel_widget)
        self.key = key
        self.sections = []
        layout = panel_widget.layout()
        items = [layout.itemAt(i) for i in range(layout.count())]
        groups, current = [], None
        for item in items:
            widget = item.widget()
            if _is_header(widget):
                current = (widget, [])
                # a hidden header (its controls moved elsewhere, e.g. the
                # pose bar) is not made foldable; its items stay in place
                if not widget.isHidden():
                    groups.append(current)
            elif current is not None:
                current[1].append(item)
                if current[0].isHidden() and item.spacerItem() is not None:
                    # the controls of a hidden section moved away: no gap
                    item.spacerItem().changeSize(0, 0)
        for header, members in groups:
            self._wrap(layout, header, members)

    def _wrap(self, layout, header, members):
        body = QWidget(header.parentWidget())
        body.setObjectName("sectionBody")
        box = QVBoxLayout(body)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(layout.spacing())
        for item in members:
            layout.removeItem(item)
            if item.widget() is not None:
                box.addWidget(item.widget())
            elif item.layout() is not None:
                box.addLayout(item.layout())
            else:
                box.addItem(item)              # spacer
        layout.insertWidget(layout.indexOf(header) + 1, body)
        title = header.text().strip()
        section = Section(self, header, body, title)
        header.setCursor(QCursor(Qt.PointingHandCursor))
        header.setToolTip("Click to collapse / expand this section\n"
                          "Right-click: expand or collapse all")
        header.installEventFilter(self)
        self.sections.append(section)
        section.set_collapsed(settings.panel_collapsed(self.key, title), save=False)

    def _section(self, header):
        return next((s for s in self.sections if s.header is header), None)

    def set_all(self, collapsed):
        for section in self.sections:
            section.set_collapsed(collapsed)

    def eventFilter(self, obj, event):
        section = self._section(obj)
        if section is None:
            return False
        if event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton and obj.rect().contains(event.pos()):
                section.toggle()
                return True
        elif event.type() == QEvent.ContextMenu:
            menu = QMenu(obj)
            expand = QAction("Expand all sections", menu)
            collapse = QAction("Collapse all sections", menu)
            expand.triggered.connect(lambda: self.set_all(False))
            collapse.triggered.connect(lambda: self.set_all(True))
            menu.addAction(expand)
            menu.addAction(collapse)
            menu.exec_(event.globalPos())
            return True
        return False


def make_collapsible(panel_widget, key):
    """Turn the sections of ``panel_widget`` (a vertical panel layout)
    into collapsible ones; ``key`` names the tab for the saved state."""
    panel = CollapsiblePanel(panel_widget, key)
    panel_widget.collapsible = panel
    return panel
