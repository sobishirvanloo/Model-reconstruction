"""Point cloud container shared by the tabs, plus single-file I/O.

``PointCloudData`` always carries points and their normals together, so
they can be handed from tab to tab in memory (no intermediate files) and
saved to / loaded from ONE file:

    .ply / .pcd   point cloud with normals (binary, via Open3D)
    .txt / .csv   6 columns  x y z nx ny nz
    .npy          (N, 6) array

The old "two TXT files" layout (points.txt + normals.txt) is still
supported as a legacy option.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
)

from modules import settings
from modules.geometry_utils import (
    apply_transform,
    estimate_oriented_normals,
    load_array,
    normalize,
)


OPEN_FILTER = "Point clouds (*.ply *.pcd *.txt *.csv *.npy);;All files (*)"
LEGACY_NORMALS_FILTER = "Normal files (*.txt *.csv *.npy)"

SAVE_FILTER_PLY = "PLY point cloud with normals (*.ply)"
SAVE_FILTER_TXT = "TXT - 6 columns x y z nx ny nz (*.txt)"
SAVE_FILTER_LEGACY = "Two TXT files - points + normals (legacy) (*.txt)"
SAVE_FILTERS = ";;".join((SAVE_FILTER_PLY, SAVE_FILTER_TXT, SAVE_FILTER_LEGACY))

CLOUD_EXTENSIONS = (".ply", ".pcd")


@dataclass
class PointCloudData:
    """Points + normals, plus the mesh they came from (``reference`` as
    ``(vertices, triangles)``) when known - the ground truth used by the
    quality metrics.  The reference is transformed together with the
    points, so it always stays aligned with them."""

    points: np.ndarray
    normals: Optional[np.ndarray] = None
    source: str = field(default="")
    reference: Optional[tuple] = None
    note: str = ""          # how the data was degraded (e.g. "noise σ=0.5%")

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)[:, :3]
        if self.normals is not None:
            normals = np.asarray(self.normals, dtype=float)[:, :3]
            if len(normals) != len(self.points):
                raise ValueError(
                    f"Mismatch: {len(self.points)} points vs "
                    f"{len(normals)} normals"
                )
            self.normals = normalize(normals)

    def __len__(self):
        return len(self.points)

    @property
    def has_normals(self):
        return self.normals is not None

    def copy(self):
        return PointCloudData(
            self.points.copy(),
            None if self.normals is None else self.normals.copy(),
            self.source,
            self.reference,
            self.note,
        )

    def transformed(self, matrix):
        """Copy with a 4x4 transform applied (e.g. the viewport rotation)."""
        points, normals = apply_transform(matrix, self.points, self.normals)
        reference = None
        if self.reference is not None:
            vertices, triangles = self.reference
            vertices, _ = apply_transform(matrix, vertices)
            reference = (vertices, triangles)
        return PointCloudData(points, normals, self.source, reference, self.note)

    def with_normals(self, voxel_size=0.0):
        """Copy that is guaranteed to have (oriented) normals."""
        if self.has_normals:
            return self.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        estimate_oriented_normals(pcd, voxel_size=voxel_size)
        return PointCloudData(
            self.points.copy(), np.asarray(pcd.normals).copy(), self.source,
            self.reference, self.note,
        )

    # =========================
    # File I/O
    # =========================
    @classmethod
    def load(cls, path):
        """Load a single file; normals are read when the file has them."""
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        if ext in CLOUD_EXTENSIONS:
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) == 0:
                raise ValueError(f"{name}: no points found")
            normals = np.asarray(pcd.normals) if pcd.has_normals() else None
            return cls(np.asarray(pcd.points), normals, name)

        data = load_array(path)
        normals = data[:, 3:6] if data.shape[1] >= 6 else None
        return cls(data[:, :3], normals, name)

    def save(self, path):
        """Save points + normals to ONE file (format from the extension)."""
        ext = os.path.splitext(path)[1].lower()
        if ext in CLOUD_EXTENSIONS:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            if self.has_normals:
                pcd.normals = o3d.utility.Vector3dVector(self.normals)
            if not o3d.io.write_point_cloud(path, pcd, write_ascii=False):
                raise IOError(f"could not write {path}")
            return
        table = (
            np.hstack([self.points, self.normals])
            if self.has_normals else self.points
        )
        if ext == ".npy":
            np.save(path, table)
        else:
            delimiter = "," if ext == ".csv" else " "
            np.savetxt(path, table, fmt="%.6f", delimiter=delimiter)

    def save_legacy(self, points_path, normals_path):
        """Old layout: points and normals in two row-aligned TXT files."""
        np.savetxt(points_path, self.points, fmt="%.6f")
        np.savetxt(normals_path, self.normals, fmt="%.6f")


# =========================
# Dialog helpers
# =========================
def ask_open_point_cloud(parent, title, normals_required, path=None):
    """Ask for a point cloud file (skipped when ``path`` is given, e.g. a
    dropped file).  When it has no normals, ask for a matching legacy
    normals file (optional unless ``normals_required``).

    Returns ``PointCloudData`` or ``None`` when cancelled.
    Raises on invalid files.
    """
    if not path:
        path, _ = QFileDialog.getOpenFileName(
            parent, title, settings.last_dir(), OPEN_FILTER)
        if not path:
            return None
    data = PointCloudData.load(path)
    settings.add_recent(path)
    if data.has_normals:
        return data

    hint = "" if normals_required else " (optional - Cancel to skip)"
    normals_path, _ = QFileDialog.getOpenFileName(
        parent, f"Import matching Normals file{hint}",
        os.path.dirname(path), LEGACY_NORMALS_FILTER
    )
    if not normals_path:
        return None if normals_required else data
    normals = load_array(normals_path)[:, :3]
    return PointCloudData(data.points, normals, data.source)


def ask_save_point_cloud(parent, data, default_name):
    """Ask where to save ``data`` and write it.

    Returns the list of written paths (empty when cancelled).
    """
    path, selected = QFileDialog.getSaveFileName(
        parent, "Save Point Cloud",
        os.path.join(settings.last_dir(), default_name), SAVE_FILTERS
    )
    if not path:
        return []
    settings.remember_path(path)

    root, ext = os.path.splitext(path)
    if selected == SAVE_FILTER_LEGACY:
        points_path = path if ext else root + ".txt"
        normals_path, _ = QFileDialog.getSaveFileName(
            parent, "Save Normals", os.path.join(
                os.path.dirname(points_path), "normals.txt"
            ), "Text Files (*.txt)"
        )
        if not normals_path:
            return []
        data.save_legacy(points_path, normals_path)
        return [points_path, normals_path]

    if ext.lower() not in (".ply", ".pcd", ".txt", ".csv", ".npy"):
        path = root + (".ply" if selected == SAVE_FILTER_PLY else ".txt")
    data.save(path)
    return [path]


def add_button_after(anchor, text):
    """Create a button right below ``anchor`` (styled by the theme)."""
    button = QPushButton(text, anchor.parentWidget())
    insert_widget(anchor, button, after=True)
    return button


def make_scrollable(widget):
    """Put ``widget`` in a vertical scroll area at the same place in its
    parent layout, so a tall panel no longer forces a tall window."""
    parent_layout = _layout_containing(widget.parentWidget().layout(), widget)
    scroll = QScrollArea(widget.parentWidget())
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    # Keep the panel's own width limits (plus room for the scrollbar) so
    # the 3D view still gets the remaining space
    bar = scroll.verticalScrollBar().sizeHint().width()
    scroll.setMinimumWidth(
        max(widget.minimumWidth(), widget.minimumSizeHint().width()) + bar
    )
    scroll.setMaximumWidth(min(widget.maximumWidth() + bar, 16777215))
    scroll.setSizePolicy(widget.sizePolicy())

    index = parent_layout.indexOf(widget)
    if isinstance(parent_layout, QGridLayout):
        row, col, row_span, col_span = parent_layout.getItemPosition(index)
        parent_layout.removeWidget(widget)
        scroll.setWidget(widget)
        parent_layout.addWidget(scroll, row, col, row_span, col_span)
    else:
        parent_layout.removeWidget(widget)
        scroll.setWidget(widget)
        parent_layout.insertWidget(index, scroll)
    return scroll


def add_section(anchor, title):
    """Section header (themed ``QLabel[role="section"]``) above ``anchor``."""
    label = QLabel(title.upper(), anchor.parentWidget())
    label.setProperty("role", "section")
    return insert_widget(anchor, label, after=False)


def move_widget_after(widget, anchor):
    """Move ``widget`` (already in a layout) to just below ``anchor``."""
    old = _layout_containing(widget.parentWidget().layout(), widget)
    if old is not None:
        old.removeWidget(widget)
    return insert_widget(anchor, widget, after=True)


def insert_widget(anchor, widget, after=True):
    """Insert ``widget`` right before/after ``anchor`` in its layout."""
    layout = _layout_containing(anchor.parentWidget().layout(), anchor)
    layout.insertWidget(layout.indexOf(anchor) + (1 if after else 0), widget)
    return widget


def _layout_containing(layout, widget):
    """The (possibly nested) layout that directly holds ``widget``."""
    if layout.indexOf(widget) >= 0:
        return layout
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        if child is not None:
            found = _layout_containing(child, widget)
            if found is not None:
                return found
    return None
