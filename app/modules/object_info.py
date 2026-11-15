# MeshBench - Copyright (c) 2026 Sobhan Shirvanloo. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Numbers about the displayed object, shown next to the log console.

``ObjectInfoPanel`` is a two-column tree (property / value) grouped in
sections:

    Pose          rotation, position
    Object box    the model's own box (turns with it): size, diagonal,
                  volume, surface, centre and the 8 corners
    World box     axis-aligned box in world coordinates (changes with the
                  rotation): min, max, size, volume
    Points        count, centroid, mean / min spacing, density
    Normals       count, unit length, outward share, coherence
    Mesh          triangles, edges, surface area, volume, closed /
                  manifold, edge lengths, fill of the box

Values that only depend on the data (spacing, mesh checks, ...) are
computed once per data set (``DataStats``); pose-dependent ones are cheap
and refreshed live (throttled) while the model is rotated or moved.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from modules.compute import KNN
from modules.i18n import tr

REFRESH_MS = 120          # live refresh while dragging
BACKGROUND_SIZE = 50_000  # points + triangles above which stats run off the UI thread
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="object-info")
SPACING_SAMPLE = 20000    # points used to estimate the spacing


def _fmt(value, decimals):
    return f"{value:,.{decimals}f}"


def _vec(values, decimals):
    return "(" + ", ".join(_fmt(v, decimals) for v in values) + ")"


def _decimals(scale):
    """Decimals suited to the model size (2 for a ~100-unit model)."""
    return min(8, max(0, 4 - math.floor(math.log10(max(scale, 1e-12)))))


class DataStats:
    """Pose-independent numbers of one data set (computed once)."""

    def __init__(self, points, normals=None, triangles=None):
        points = np.asarray(points, dtype=float)
        self.count = len(points)
        self.low = points.min(axis=0)
        self.high = points.max(axis=0)
        self.size = self.high - self.low
        self.diagonal = float(np.linalg.norm(self.size))
        self.centroid = points.mean(axis=0)
        self.box_center = (self.low + self.high) / 2.0
        self.spacing = self._spacing(points)
        self.normals = self._normals(points, normals)
        self.mesh = (self._mesh(points, np.asarray(triangles))
                     if triangles is not None and len(triangles) else None)

    @staticmethod
    def _spacing(points):
        """Mean and min distance to the nearest neighbour (sampled)."""
        if len(points) < 2:
            return None
        sample = points
        if len(points) > SPACING_SAMPLE:
            rng = np.random.default_rng(0)
            sample = points[rng.choice(len(points), SPACING_SAMPLE, replace=False)]
        _, distances = KNN(points).knn(sample, 2, distances=True)
        nearest = distances[:, 1]
        positive = nearest[nearest > 0]
        return {
            "mean": float(nearest.mean()),
            "min": float(positive.min()) if len(positive) else 0.0,
            "duplicates": int((nearest == 0).sum()),
            "sampled": len(sample) < len(points),
        }

    @staticmethod
    def _normals(points, normals):
        if normals is None or len(normals) != len(points):
            return None
        normals = np.asarray(normals, dtype=float)
        length = np.linalg.norm(normals, axis=1)
        unit = normals / np.maximum(length, 1e-12)[:, None]
        outward = np.einsum("ij,ij->i", unit, points - points.mean(axis=0))
        mean = unit.mean(axis=0)
        return {
            "count": len(normals),
            "length_mean": float(length.mean()),
            "unit_share": float((np.abs(length - 1.0) < 1e-3).mean()),
            "outward_share": float((outward > 0).mean()),
            "mean_direction": mean / max(np.linalg.norm(mean), 1e-12),
            "coherence": float(np.linalg.norm(mean)),
        }

    @staticmethod
    def _mesh(vertices, triangles):
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        cross = np.cross(b - a, c - a)
        double_area = np.linalg.norm(cross, axis=1)
        # signed volume (divergence theorem) - meaningful when closed
        volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        edges = np.sort(np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
            axis=1)
        # one int64 per edge: much faster than np.unique(axis=0)
        n = np.int64(len(vertices))
        codes, uses = np.unique(edges[:, 0].astype(np.int64) * n + edges[:, 1],
                                return_counts=True)
        unique = np.column_stack([codes // n, codes % n])
        lengths = np.linalg.norm(vertices[unique[:, 0]] - vertices[unique[:, 1]],
                                 axis=1)
        return {
            "triangles": len(triangles),
            "vertices": int(len(np.unique(triangles))),
            "edges": len(unique),
            "area": float(double_area.sum() / 2.0),
            "volume": abs(volume),
            "closed": bool((uses == 2).all()),
            "boundary_edges": int((uses == 1).sum()),
            "non_manifold_edges": int((uses > 2).sum()),
            "degenerate": int((double_area < 1e-12).sum()),
            "edge_mean": float(lengths.mean()),
            "edge_min": float(lengths.min()),
            "edge_max": float(lengths.max()),
            "euler": int(len(np.unique(triangles)) - len(unique) + len(triangles)),
        }


def info_rows(stats, matrix, rotation, position, points=None, extra=()):
    """Sections [(title, [(name, value, tooltip), ...]), ...]."""
    d = _decimals(stats.diagonal)
    rot, move = matrix[:3, :3], matrix[:3, 3]
    world = lambda p: rot @ np.asarray(p) + move
    sections = []

    sections.append(("Pose", [
        ("Rotation X, Y, Z", _vec(rotation, 1) + "°",
         "Angles of the model relative to the pose it was loaded in"),
        ("Position X, Y, Z", _vec(position, d),
         "Offset from the starting position (model units)"),
    ]))

    corners = np.array([(x, y, z) for x in (stats.low[0], stats.high[0])
                        for y in (stats.low[1], stats.high[1])
                        for z in (stats.low[2], stats.high[2])])
    corners_world = corners @ rot.T + move
    size = stats.size
    box_rows = [
        ("Size (W × H × D)", " × ".join(_fmt(v, d) for v in size),
         "Edge lengths of the model's own box (they do not change when it turns)"),
        ("Diagonal", _fmt(stats.diagonal, d), "Length of the box diagonal"),
        ("Volume", _fmt(float(np.prod(size)), d) + " u³",
         "Volume of the box around the model"),
        ("Surface", _fmt(2 * float(size[0] * size[1] + size[1] * size[2]
                                   + size[0] * size[2]), d) + " u²",
         "Surface area of the box"),
        ("Centre", _vec(world(stats.box_center), d),
         "Centre of the box in world coordinates (moves with the model)"),
    ]
    for number, corner in enumerate(corners_world, 1):
        box_rows.append((f"Corner {number}", _vec(corner, d),
                         "Box corner in world coordinates"))
    sections.append(("Object box", box_rows))

    if points is not None:
        moved = np.asarray(points) @ rot.T + move
        low, high = moved.min(axis=0), moved.max(axis=0)
    else:
        low, high = corners_world.min(axis=0), corners_world.max(axis=0)
    sections.append(("World box (axis-aligned)", [
        ("Min X, Y, Z", _vec(low, d), "Smallest coordinates in the world"),
        ("Max X, Y, Z", _vec(high, d), "Largest coordinates in the world"),
        ("Size", " × ".join(_fmt(v, d) for v in high - low),
         "Extent along the world axes (changes when the model turns)"),
        ("Volume", _fmt(float(np.prod(high - low)), d) + " u³",
         "Volume of the axis-aligned box"),
    ]))

    point_rows = [
        ("Count", f"{stats.count:,}", "Number of points / vertices"),
        ("Centroid", _vec(world(stats.centroid), d),
         "Mean of all points, in world coordinates"),
    ]
    if stats.spacing:
        sampled = " (sampled)" if stats.spacing["sampled"] else ""
        point_rows += [
            ("Mean spacing", _fmt(stats.spacing["mean"], d) + sampled,
             "Average distance to the nearest neighbour"),
            ("Min spacing", _fmt(stats.spacing["min"], d) + sampled,
             "Smallest non-zero distance between neighbours"),
        ]
        if stats.spacing["duplicates"]:
            point_rows.append(("Duplicates", f"{stats.spacing['duplicates']:,}"
                               + sampled, "Points at exactly the same place"))
    box_volume = float(np.prod(size))
    if box_volume > 0:
        point_rows.append(("Density", _fmt(stats.count / box_volume, 4) + " /u³",
                           "Points per unit volume of the object box"))
    sections.append(("Points", point_rows))

    n = stats.normals
    if n is None:
        sections.append(("Normals", [("Normals", "none",
                                      "This data has no normals")]))
    else:
        sections.append(("Normals", [
            ("Count", f"{n['count']:,}", "One normal per point"),
            ("Mean length", _fmt(n["length_mean"], 4),
             "1.0000 = normalised"),
            ("Unit length", f"{n['unit_share']:.1%}",
             "Share of normals with length 1"),
            ("Pointing outward", f"{n['outward_share']:.1%}",
             "Share pointing away from the centroid - close to 100% for a "
             "consistently oriented closed shape"),
            ("Mean direction", _vec(rot @ n["mean_direction"], 3),
             "Average normal direction (world coordinates)"),
            ("Coherence", _fmt(n["coherence"], 3),
             "Length of the mean normal: ~0 for a closed shape, 1 when all "
             "normals point the same way"),
        ]))

    m = stats.mesh
    if m is not None:
        fill = m["volume"] / box_volume if box_volume > 0 and m["closed"] else None
        sections.append(("Mesh", [
            ("Triangles", f"{m['triangles']:,}", "Number of faces"),
            ("Vertices used", f"{m['vertices']:,}", "Vertices referenced by faces"),
            ("Edges", f"{m['edges']:,}", "Unique edges"),
            ("Surface area", _fmt(m["area"], d) + " u²", "Sum of triangle areas"),
            ("Volume", (_fmt(m["volume"], d) + " u³") if m["closed"]
             else f"≈ {_fmt(m['volume'], d)} u³ (not closed)",
             "Enclosed volume (exact only for a closed mesh)"),
            ("Fills box", f"{fill:.1%}" if fill is not None else "-",
             "Mesh volume / object box volume"),
            ("Closed (no holes)", "yes" if m["closed"] else "no",
             "Every edge is shared by exactly two triangles (self-"
             "intersections are not checked)"),
            ("Boundary edges", f"{m['boundary_edges']:,}",
             "Edges of holes / open borders"),
            ("Non-manifold edges", f"{m['non_manifold_edges']:,}",
             "Edges shared by more than two triangles"),
            ("Degenerate triangles", f"{m['degenerate']:,}", "Zero-area faces"),
            ("Edge length min / mean / max",
             " / ".join(_fmt(m[k], d) for k in ("edge_min", "edge_mean", "edge_max")),
             "Lengths of the mesh edges"),
            ("Euler characteristic", f"{m['euler']:,}",
             "V − E + F: 2 for one closed shape without holes"),
        ]))

    if extra:
        sections.append(extra)
    return sections


class ObjectInfoPanel(QWidget):
    """Property / value tree with a title and a *Copy* button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("objectInfo")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        title = QLabel("Object info")
        title.setProperty("role", "section")
        header.addWidget(title)
        header.addStretch(1)
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy")
        self.copy_button.setToolTip("Copy all values as text (tab separated)")
        self.copy_button.setAutoRaise(True)
        self.copy_button.clicked.connect(self.copy)
        header.addWidget(self.copy_button)
        layout.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Property", "Value"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(True)
        layout.addWidget(self.tree)
        self._collapsed = {tr("Object box"): False}
        self.set_sections(None)

    def set_sections(self, sections):
        """Show ``info_rows`` output (``None`` = no data)."""
        # keep which sections the user folded
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            self._collapsed[item.text(0)] = not item.isExpanded()
        scroll = self.tree.verticalScrollBar().value()
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()
        if not sections:
            QTreeWidgetItem(self.tree, [tr("No data loaded"), ""])
        else:
            for title, rows in sections:
                parent = QTreeWidgetItem(self.tree, [tr(title), ""])
                font = parent.font(0)
                font.setBold(True)
                parent.setFont(0, font)
                parent.setFirstColumnSpanned(True)
                for name, value, tip in rows:
                    item = QTreeWidgetItem(parent, [tr(name), tr(value)])
                    item.setToolTip(0, tr(tip))
                    item.setToolTip(1, tr(tip))
                parent.setExpanded(not self._collapsed.get(tr(title), False))
        self.tree.setUpdatesEnabled(True)
        self.tree.verticalScrollBar().setValue(scroll)

    def text(self):
        lines = []
        for i in range(self.tree.topLevelItemCount()):
            section = self.tree.topLevelItem(i)
            lines.append(section.text(0))
            for j in range(section.childCount()):
                row = section.child(j)
                lines.append(f"  {row.text(0)}\t{row.text(1)}")
        return "\n".join(lines)

    def copy(self):
        QApplication.clipboard().setText(self.text())


class ObjectInfo:
    """Keeps a tab's info panel up to date.

    ``owner.info_source()`` returns ``(points, normals, triangles, extra)``
    of what the view shows (``None`` when empty).
    """

    def __init__(self, owner, panel):
        self.owner = owner
        self.panel = panel
        self._key = None
        self._stats = None
        self._pending = None          # (key, future) of a background job
        self._timer = QTimer(panel)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)

    def schedule(self):
        """Refresh soon (many pose changes in a row cost one refresh)."""
        if not self._timer.isActive():
            self._timer.start()

    def refresh(self):
        self._timer.stop()
        source = getattr(self.owner, "info_source", lambda: None)()
        if source is None or getattr(self.owner, "actor", None) is None:
            self._key, self._stats = None, None
            self.panel.set_sections(None)
            return
        points, normals, triangles, extra = source
        key = (id(points), id(normals), id(triangles))
        if key != self._key:
            size = len(points) + (0 if triangles is None else len(triangles))
            if size < BACKGROUND_SIZE:
                self._stats, self._key = DataStats(points, normals, triangles), key
            else:
                # Large data: compute off the UI thread, show it when ready
                if self._pending is None or self._pending[0] != key:
                    self._pending = (key, _executor.submit(
                        DataStats, points, normals, triangles))
                    self.panel.set_sections([("Object info", [
                        ("Computing…", f"{size:,} elements",
                         "Statistics are computed in the background")])])
                future = self._pending[1]
                if not future.done():
                    self._timer.start()          # poll again shortly
                    return
                self._pending = None
                try:
                    self._stats, self._key = future.result(), key
                except Exception as e:
                    self.panel.set_sections([("Object info", [
                        ("Error", str(e), "The statistics could not be computed")])])
                    return
        owner = self.owner
        matrix = owner.actor_matrix()
        rotation = (owner.rotation_x, owner.rotation_y, owner.rotation_z)
        self.panel.set_sections(info_rows(
            self._stats, matrix, rotation, owner.position, points, extra))
