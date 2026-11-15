# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Tab 4 - Compare a reconstruction with the reference (original) mesh.

Same layout as the other tabs (3D view + side panel + log), with the
results beside the log: an **Errors** table (mean e, SD, MAE, RMSE, R²,
... per method), the error **Distribution** (all methods overlaid or one
at a time) and the **Object info**.

The view colours the reconstruction by the error of one method, or shows
**all methods at once** in a grid of linked views (same camera, same
colour scale); the reference can be overlaid, translucent.  The
computations (``comparison_metrics``) run on a background thread.
"""

import csv
import math
import os

import numpy as np
import vtk
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from vtk.util import numpy_support

from modules import settings
from modules.alignment import align, describe, is_aligned, misalignment
from modules.comparison_metrics import (
    DEFAULT_PARAMS,
    MESH_METHODS,
    METHODS,
    ORDER,
    STAT_NAMES,
    Model,
    run_comparison_isolated,
)
from modules.geometry_utils import apply_transform, polydata_from_numpy
from modules.import_mesh import Geometry
from modules.point_data import add_section, insert_widget, make_scrollable
from modules.poisson_reconstruction import ReconstructionWorker
from modules.theme import PALETTES
from modules.view_settings import ColorButton
from modules.vtk_controls import TabMixin, VTKViewport

OPEN_FILTER = "3D files (*.ply *.obj *.stl *.pcd *.off *.gltf *.glb);;All files (*)"

# One colour per method (charts / grid titles)
METHOD_COLORS = {
    "p2p": "#4e79a7", "p2pl": "#f28e2b", "p2m": "#e15759", "c2c": "#76b7b2",
    "ac2c": "#59a14f", "c2m": "#b07aa1", "m3c2": "#9c755f",
}

TABLE_COLUMNS = (
    ("Method", None, ""),
    ("N", "n", "Number of compared points (finite values)"),
    ("Mean e", "mean", "Mean error (signed methods: + outside / − inside "
                       "the reference)"),
    ("SD", "sd", "Standard deviation of the error"),
    ("MAE", "mae", "Mean absolute error"),
    ("RMSE", "rmse", "Root mean square error"),
    ("R²", "r2", "1 − Σe² / Σ‖q − q̄‖²: share of the reference shape's "
                 "spatial variance reproduced (1 = perfect)"),
    ("Median |e|", "median", "Median absolute error"),
    ("P95 |e|", "p95", "95th percentile of |e|"),
    ("Max |e|", "max", "Largest |e|"),
    ("Time", "time", "Computation time in seconds"),
)


def load_model(path):
    """A mesh or point cloud file as a ``Model``."""
    ext = os.path.splitext(path)[1].lower()
    result = None
    if ext != ".pcd":
        result = Geometry._load_mesh(path)
        if result is None and ext != ".ply":
            raise ValueError("the file contains no triangles")
    if result is None:
        result = Geometry._load_point_cloud(path)
    vertices, normals, triangles, _ = result
    return Model(vertices, triangles, normals, os.path.basename(path))


def _fmt(value, decimals=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{decimals}g}"


# =========================
# Error distribution chart
# =========================
class DistributionChart(QWidget):
    """Histogram curves (% of points per bin) of several methods, drawn
    with QPainter (no plotting library needed)."""

    BINS = 60

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 150)
        self.curves = []          # [(label, QColor, x (bins), y (%))]
        self.signed = False
        self.limit = 1.0

    def set_curves(self, series, signed, limit):
        """``series``: [(label, colour, values)]; x range ±limit (signed)
        or [0, limit]."""
        self.signed, self.limit = signed, max(limit, 1e-12)
        low = -self.limit if signed else 0.0
        edges = np.linspace(low, self.limit, self.BINS + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        self.curves = []
        for label, color, values in series:
            values = values[np.isfinite(values)]
            if not signed:
                values = np.abs(values)
            if len(values) == 0:
                continue
            counts, _ = np.histogram(np.clip(values, low, self.limit), edges)
            self.curves.append((label, QColor(color), centres,
                                100.0 * counts / len(values)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.Text)
        grid = QColor(text)
        grid.setAlpha(40)
        painter.fillRect(self.rect(), palette.color(palette.Base))
        font = QFont(self.font())
        font.setPixelSize(10)
        painter.setFont(font)
        if not self.curves:
            painter.setPen(text)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Run a comparison to see the error distribution")
            return

        left, right, top, bottom = 44, 12, 10, 30
        plot = QRectF(left, top, max(self.width() - left - right, 10),
                      max(self.height() - top - bottom, 10))
        y_max = max(float(y.max()) for *_, y in self.curves) * 1.08 or 1.0
        low = -self.limit if self.signed else 0.0
        to_x = lambda v: plot.left() + (v - low) / (self.limit - low) * plot.width()
        to_y = lambda v: plot.bottom() - v / y_max * plot.height()

        # grid + axis labels
        painter.setPen(QPen(grid, 1))
        for i in range(5):
            y_value = y_max * i / 4
            y = to_y(y_value)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(text)
            painter.drawText(QRectF(0, y - 7, left - 5, 14),
                             Qt.AlignRight | Qt.AlignVCenter, f"{y_value:.1f}%")
            painter.setPen(QPen(grid, 1))
        for i in range(5):
            x_value = low + (self.limit - low) * i / 4
            x = to_x(x_value)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(text)
            painter.drawText(QRectF(x - 40, plot.bottom() + 3, 80, 14),
                             Qt.AlignHCenter | Qt.AlignTop, f"{x_value:.3g}")
            painter.setPen(QPen(grid, 1))
        painter.setPen(text)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 15, plot.width(), 14),
                         Qt.AlignHCenter, "error e" if self.signed else "|error|")

        # curves
        for label, color, x, y in self.curves:
            path = QPainterPath(QPointF(to_x(x[0]), to_y(y[0])))
            for xv, yv in zip(x[1:], y[1:]):
                path.lineTo(QPointF(to_x(xv), to_y(yv)))
            painter.setPen(QPen(color, 2))
            painter.drawPath(path)

        # legend
        line_height = 14
        box_width = max(painter.fontMetrics().horizontalAdvance(c[0])
                        for c in self.curves) + 30
        x0 = plot.right() - box_width
        background = QColor(palette.color(palette.Base))
        background.setAlpha(210)
        painter.fillRect(QRectF(x0 - 4, plot.top(), box_width + 4,
                                line_height * len(self.curves) + 4), background)
        for i, (label, color, *_) in enumerate(self.curves):
            y = plot.top() + 2 + i * line_height
            painter.setPen(QPen(color, 3))
            painter.drawLine(QPointF(x0, y + 7), QPointF(x0 + 16, y + 7))
            painter.setPen(text)
            painter.drawText(QRectF(x0 + 22, y, box_width, line_height),
                             Qt.AlignVCenter, label)


# =========================
# Tab 4
# =========================
MODEL_COLOUR_DEFAULTS = {
    "compared": None,            # None = theme colour / Object settings
    "reference": None,
    "reference_opacity": 28,     # %
    "error_colours": True,       # heatmap on the reconstruction
}


def load_model_colours():
    """Saved colours of the two Compare models (unknown keys dropped)."""
    saved = settings.view_settings("compare")
    colours = dict(MODEL_COLOUR_DEFAULTS)
    for key in colours:
        if key in saved:
            colours[key] = saved[key]
    for key in ("compared", "reference"):
        if colours[key] is not None and not QColor(str(colours[key])).isValid():
            colours[key] = None
    try:
        colours["reference_opacity"] = min(100, max(5, int(colours["reference_opacity"])))
    except (TypeError, ValueError):
        colours["reference_opacity"] = MODEL_COLOUR_DEFAULTS["reference_opacity"]
    colours["error_colours"] = bool(colours["error_colours"])
    return colours


class Compare(TabMixin):
    """Compare a *compared* model (reconstruction) with a *reference*."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.compared = None          # comparison_metrics.Model
        self.reference = None
        self.source_name = ""
        self.reference_name = ""
        self.comparison = None        # run_comparison() output
        self.alignment = np.eye(4)    # applied to the compared model
        self._worker = None
        self._generation = 0
        self.ref_actor = None         # reference drawn over the compared one
        self._grid = []               # extra renderers of the grid view
        self._grid_labels = []
        self._scalar_bar = None
        self._legend_renderer = None
        self.model_colours = load_model_colours()
        self.init_rotation_state()

        self._build_page()
        VTKViewport(
            self, self.frame_console, self.frame_view,
            empty_hint="Open a reconstruction and the reference mesh\n"
                       "(or take them from the Reconstruct / Import tabs)\n\n"
                       "then press Run comparison",
        )
        self._build_results()
        self.setup_rotation_controls()
        self._build_sections()
        self._refresh_colour_buttons()
        make_scrollable(self.btn_export.parentWidget())
        self._set_controls_enabled()
        self.console.appendPlainText(
            "Ready. Choose the reconstruction (compared model) and the "
            "reference, then Run comparison.")

    # =========================
    # Page / panel
    # =========================
    def _build_page(self):
        self.page = QWidget()
        grid = QGridLayout(self.page)
        self.frame_view = QFrame(self.page)
        self.frame_console = QFrame(self.page)
        panel = QFrame(self.page)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(320)
        grid.addWidget(self.frame_view, 0, 0)
        grid.addWidget(panel, 0, 1)
        grid.addWidget(self.frame_console, 1, 0, 1, 2)
        layout = QVBoxLayout(panel)

        def button(text, icon=None, tip="", variant=None):
            b = QPushButton(text, panel)
            if icon is not None:
                self.decorate(b, icon, tip, variant)
            layout.addWidget(b)
            return b

        def hint(text):
            label = QLabel(text, panel)
            label.setProperty("role", "hint")
            label.setWordWrap(True)
            layout.addWidget(label)
            return label

        # Models
        self.label_compared = QLabel("<b>Reconstruction</b> (compared)", panel)
        layout.addWidget(self.label_compared)
        self.file_label = hint("No model loaded")
        self.btn_import = button("Open reconstruction…", QStyle.SP_DialogOpenButton,
                                 "Open the reconstructed mesh (Ctrl+O)", "primary")
        self.btn_from_reconstruct = button(
            "From Reconstruct tab", QStyle.SP_ArrowBack,
            "Take the mesh of tab 3 (and its ground truth, if it has one)")
        layout.addSpacing(6)
        layout.addWidget(QLabel("<b>Reference</b> (original)", panel))
        self.ref_label = hint("No reference loaded")
        self.btn_open_reference = button(
            "Open reference…", QStyle.SP_DialogOpenButton,
            "Open the original mesh (same coordinates as the reconstruction)")
        self.btn_from_import = button(
            "From Import mesh tab", QStyle.SP_ArrowBack,
            "Use the mesh shown in tab 1 (its rotation applied)")
        self.check_auto_align = QCheckBox("Align automatically when opening", panel)
        self.check_auto_align.setChecked(True)
        self.check_auto_align.setToolTip(
            "If the two models are not in the same place / orientation "
            "(e.g. the reconstruction was saved rotated), move the "
            "reconstruction onto the reference: feature matching (FPFH + "
            "RANSAC) then ICP. Models that already match are left as they are.")
        layout.addWidget(self.check_auto_align)
        self.btn_align = button(
            "Align now", QStyle.SP_BrowserReload,
            "Register the reconstruction onto the reference now (also when "
            "they look aligned: a final ICP fit)")

        # Rotation (same widgets as the .ui tabs)
        for axis in "XYZ":
            label = QLabel(f"{axis}-Axis (angle)", panel)
            spin = QSpinBox(panel)
            spin.setRange(-360, 360)
            slider = QSlider(Qt.Horizontal, panel)
            slider.setRange(-360, 360)
            for widget in (label, spin, slider):
                layout.addWidget(widget)
            setattr(self, f"label_{axis.lower()}", label)
            setattr(self, f"spinBox_{axis}", spin)
            setattr(self, f"slider_{axis}", slider)
        self.pushButton = button("Reset rotation")

        # Methods + parameters
        self.method_checks = {}
        for key in ORDER:
            label, signed, tip = METHODS[key]
            check = QCheckBox(label + ("  (signed)" if signed else ""), panel)
            check.setChecked(True)
            check.setToolTip(tip + (" - needs reference triangles"
                                    if key in MESH_METHODS else ""))
            layout.addWidget(check)
            self.method_checks[key] = check
        form = QFormLayout()
        self.spin_samples = QSpinBox(panel)
        self.spin_samples.setRange(5_000, 2_000_000)
        self.spin_samples.setSingleStep(10_000)
        self.spin_samples.setValue(DEFAULT_PARAMS["samples"])
        self.spin_samples.setToolTip(
            "Points sampled uniformly on each surface for the cloud methods")
        self.spin_cores = QSpinBox(panel)
        self.spin_cores.setRange(1_000, 500_000)
        self.spin_cores.setSingleStep(5_000)
        self.spin_cores.setValue(DEFAULT_PARAMS["core_points"])
        self.spin_cores.setToolTip("Points where M3C2 measures a distance")
        self.spin_diameter = QDoubleSpinBox(panel)
        self.spin_diameter.setRange(1.0, 50.0)
        self.spin_diameter.setValue(DEFAULT_PARAMS["m3c2_diameter"])
        self.spin_diameter.setToolTip(
            "M3C2 cylinder diameter as a multiple of the mean sample spacing")
        self.spin_depth = QDoubleSpinBox(panel)
        self.spin_depth.setRange(0.1, 20.0)
        self.spin_depth.setSingleStep(0.5)
        self.spin_depth.setSuffix(" %")
        self.spin_depth.setValue(DEFAULT_PARAMS["m3c2_depth_pct"])
        self.spin_depth.setToolTip(
            "M3C2 maximum depth (cylinder half-length) in % of the diagonal: "
            "just above the expected error - too long and it reaches the "
            "other side of thin parts")
        form.addRow("Samples", self.spin_samples)
        form.addRow("M3C2 cores", self.spin_cores)
        form.addRow("M3C2 diameter ×", self.spin_diameter)
        form.addRow("M3C2 depth", self.spin_depth)
        layout.addLayout(form)
        self.btn_run = button("Run comparison", QStyle.SP_MediaPlay,
                              "Compute the checked methods", "primary")

        # Display
        self.label_display = QLabel("Show", panel)
        layout.addWidget(self.label_display)
        self.combo_mode = QComboBox(panel)
        self.combo_mode.addItem("One method", "single")
        self.combo_mode.addItem("All methods (grid)", "grid")
        self.combo_mode.setToolTip(
            "One coloured model, or every method side by side with a "
            "common colour scale (the views turn together)")
        layout.addWidget(self.combo_mode)
        self.combo_method = QComboBox(panel)
        for key in ORDER:
            self.combo_method.addItem(METHODS[key][0], key)
        layout.addWidget(self.combo_method)
        self.check_overlay = QCheckBox("Overlay the reference (translucent)", panel)
        self.check_overlay.setChecked(True)
        layout.addWidget(self.check_overlay)
        self.check_signed = QCheckBox("Signed colours (blue − / red +)", panel)
        self.check_signed.setToolTip(
            "Signed methods: blue = inside the reference, red = outside")
        layout.addWidget(self.check_signed)
        row = QHBoxLayout()
        row.addWidget(QLabel("Colour max", panel))
        self.spin_colour_max = QDoubleSpinBox(panel)
        self.spin_colour_max.setDecimals(4)
        self.spin_colour_max.setRange(0.0, 1e6)
        self.spin_colour_max.setSpecialValueText("auto (P95)")
        self.spin_colour_max.setToolTip(
            "Error shown in full red; 0 = 95th percentile of |e|")
        row.addWidget(self.spin_colour_max, 1)
        layout.addLayout(row)

        # Colours of the two models
        colours = self.model_colours
        self.label_colours = QLabel("Reconstruction (compared)", panel)
        layout.addWidget(self.label_colours)
        self.btn_color_compared = ColorButton(
            colours["compared"] or "#ffffff", panel)
        self.btn_color_compared.setToolTip(
            "Colour of the reconstruction (shown when it is not coloured by "
            "its error)")
        layout.addWidget(self.btn_color_compared)
        layout.addWidget(QLabel("Reference (original)", panel))
        self.btn_color_reference = ColorButton(
            colours["reference"] or "#ffffff", panel)
        self.btn_color_reference.setToolTip("Colour of the reference mesh")
        layout.addWidget(self.btn_color_reference)
        row = QHBoxLayout()
        row.addWidget(QLabel("Reference opacity", panel))
        self.spin_ref_opacity = QSpinBox(panel)
        self.spin_ref_opacity.setRange(5, 100)
        self.spin_ref_opacity.setSuffix(" %")
        self.spin_ref_opacity.setValue(colours["reference_opacity"])
        self.spin_ref_opacity.setToolTip(
            "Opacity of the reference drawn over the reconstruction")
        row.addWidget(self.spin_ref_opacity, 1)
        layout.addLayout(row)
        self.check_error_colours = QCheckBox(
            "Colour the reconstruction by its error", panel)
        self.check_error_colours.setChecked(colours["error_colours"])
        self.check_error_colours.setToolTip(
            "Off: both models keep their own colours after a comparison")
        layout.addWidget(self.check_error_colours)
        self.btn_default_colours = button(
            "Default colours", QStyle.SP_BrowserReload,
            "Theme colours, reference 28 % opaque, error colours on")

        # Output
        self.btn_export = button("Export results (CSV)…", QStyle.SP_DialogSaveButton,
                                 "Save the error table (Ctrl+S)")
        self.btn_copy = button("Copy table as Markdown")
        layout.addStretch(1)
        self.panel = panel

        self.btn_import.clicked.connect(lambda: self.import_file())
        self.btn_from_reconstruct.clicked.connect(self.from_reconstruction)
        self.btn_open_reference.clicked.connect(lambda: self.open_reference())
        self.btn_from_import.clicked.connect(self.from_import_tab)
        self.btn_align.clicked.connect(lambda: self.align_models(force=True))
        self.btn_run.clicked.connect(self.run)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_copy.clicked.connect(self.copy_markdown)
        self.combo_mode.currentIndexChanged.connect(self._render)
        self.combo_method.currentIndexChanged.connect(self._render)
        self.check_overlay.toggled.connect(self._render)
        self.check_signed.toggled.connect(self._display_changed)
        self.spin_colour_max.valueChanged.connect(self._display_changed)
        self.btn_color_compared.changed.connect(
            lambda c: self._set_model_colour("compared", c))
        self.btn_color_reference.changed.connect(
            lambda c: self._set_model_colour("reference", c))
        self.spin_ref_opacity.valueChanged.connect(
            lambda v: self._set_model_colour("reference_opacity", v))
        self.check_error_colours.toggled.connect(
            lambda on: self._set_model_colour("error_colours", on))
        self.btn_default_colours.clicked.connect(self.default_model_colours)

    def _build_sections(self):
        add_section(self.label_compared, "Models")
        add_section(self.label_x, "Rotation")
        add_section(self.method_checks[ORDER[0]], "Comparison methods")
        add_section(self.label_display, "Display")
        add_section(self.label_colours, "Colours")
        add_section(self.btn_export, "Output")
        self.label_display.hide()
        self.add_remove_button(self.btn_from_import)

    def _build_results(self):
        """Errors / Distribution / Object info, beside the log."""
        self.results = QTabWidget()
        self.results.setDocumentMode(True)

        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLUMNS])
        for column, (_, _, tip) in enumerate(TABLE_COLUMNS):
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.results.addTab(self.table, "Errors")

        chart_page = QWidget()
        chart_layout = QVBoxLayout(chart_page)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.radio_all = QRadioButton("All methods overlaid")
        self.radio_one = QRadioButton("Selected method only")
        self.radio_all.setChecked(True)
        group = QButtonGroup(chart_page)
        group.addButton(self.radio_all)
        group.addButton(self.radio_one)
        row.addWidget(self.radio_all)
        row.addWidget(self.radio_one)
        row.addStretch(1)
        chart_layout.addLayout(row)
        self.chart = DistributionChart()
        chart_layout.addWidget(self.chart, 1)
        self.radio_all.toggled.connect(self._update_chart)
        self.results.addTab(chart_page, "Distribution")
        self.results.addTab(self.info_panel, "Object info")

    def _set_controls_enabled(self):
        loaded = self.compared is not None or self.reference is not None
        for widget in self.rotation_widgets():
            widget.setEnabled(loaded)
        both = self.compared is not None and self.reference is not None
        self.btn_run.setEnabled(both and self._worker is None)
        self.btn_align.setEnabled(both and self._worker is None)
        done = self.comparison is not None
        for widget in (self.btn_export, self.btn_copy):
            widget.setEnabled(done)
        self.btn_remove.setEnabled(loaded)

    # =========================
    # Loading models
    # =========================
    def import_file(self, path=None):
        """Open the compared model (the reconstruction)."""
        self._open(path, compared=True)

    def open_reference(self, path=None):
        self._open(path, compared=False)

    def _open(self, path, compared):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Open Reconstruction" if compared else "Open Reference Mesh",
                settings.last_dir(), OPEN_FILTER)
            if not path:
                return
        try:
            model = load_model(path)
        except Exception as e:
            self.notify(f"❌ Could not open {os.path.basename(path)}: {e}",
                        "error", dialog=True)
            return
        settings.add_recent(path)
        if compared:
            self.set_models(model, self.reference, model.name, self.reference_name)
        else:
            self.set_models(self.compared, model, self.source_name, model.name)

    def from_reconstruction(self):
        """The mesh of tab 3 and its ground truth, in their own coordinates;
        the view takes tab 3's rotation / position."""
        tab = getattr(self.main_window, "poisson", None)
        if tab is None or tab._mesh_arrays is None:
            self.notify("❌ There is no reconstructed mesh in the Reconstruct tab",
                        "warning")
            return
        vertices, normals, triangles = tab._mesh_arrays
        build = tab._last_build or {}
        name = (f"{tab.source_name or 'reconstruction'} · "
                f"{build.get('setting', '')}").strip(" ·")
        compared = Model(vertices, triangles, normals, name)
        reference, reference_name = self.reference, self.reference_name
        if tab.reference is not None:
            ref_vertices, ref_triangles = tab.reference
            reference = Model(ref_vertices, ref_triangles, None,
                              tab.reference_name or "original mesh")
            reference_name = reference.name
        self.set_models(compared, reference, name, reference_name,
                        pose=tab.pose_state())
        if tab.reference is None and reference is None:
            self.notify("Mesh received - open the reference mesh to compare",
                        "warning")

    def from_import_tab(self):
        tab = getattr(self.main_window, "geometry", None)
        if tab is None or tab.points is None:
            self.notify("❌ The Import mesh tab has no model", "warning")
            return
        data = tab.current_data(apply_pose=False)   # same coordinates as tab 3
        if data.reference is not None:
            vertices, triangles = data.reference
            reference = Model(vertices, triangles, None, tab.source_name)
        else:
            reference = Model(data.points, None, data.normals, tab.source_name)
        self.set_models(self.compared, reference, self.source_name, reference.name)

    def set_models(self, compared, reference, compared_name, reference_name,
                   pose=None):
        """New models: old results no longer apply.

        The rotation / position the user gave the view is kept, so a model
        opened second appears in the same pose as the first one; when the
        two are not in the same coordinates the reconstruction is aligned
        onto the reference (``check_auto_align``)."""
        self._generation += 1
        if compared is not self.compared:
            self.alignment = np.eye(4)
        self.compared, self.reference = compared, reference
        self.source_name, self.reference_name = compared_name, reference_name
        self.comparison = None
        first = self.actor is None
        if first:
            self.reset_transform()
        if pose is not None:               # the sending tab's rotation / position
            self.restore_pose(pose)
        if (compared is not None and reference is not None
                and self.check_auto_align.isChecked()):
            self.align_models(force=False, show=False)
        self._show_models(reset_camera=first)
        self._fill_table()
        self._update_chart()
        names = " vs ".join(n for n in (compared_name, reference_name) if n)
        self.notify(f"Loaded {names}", "success")
        self.record_checkpoint(f"Models: {names}")

    def align_models(self, force=True, show=True):
        """Move the compared model onto the reference (rigid registration).

        ``force=False`` only aligns models that do not already match, so a
        reconstruction sent from tab 3 keeps its true error."""
        if self.compared is None or self.reference is None or self._worker:
            return False
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            show_message = getattr(self.main_window, "show_message", None)
            if show_message:
                show_message("⏳ Checking the alignment of the models…")
            QApplication.processEvents()
            if not force and is_aligned(self.compared, self.reference):
                self.console.appendPlainText(
                    "📐 The models are in the same coordinates - no alignment needed")
                return False
            before = misalignment(self.compared, self.reference)
            matrix, info = align(self.compared, self.reference)
            model = self.compared
            vertices, normals = apply_transform(matrix, model.vertices, model.normals)
            self.compared = Model(vertices, model.triangles, normals, model.name)
            self.alignment = matrix @ self.alignment
            after = misalignment(self.compared, self.reference)
        except Exception as e:
            self.notify(f"❌ Alignment failed: {e}", "error", dialog=True)
            return False
        finally:
            QApplication.restoreOverrideCursor()
        (rx, ry, rz), move = describe(matrix)
        self.console.appendPlainText(
            f"📐 Aligned the reconstruction to the reference ({info['method']}): "
            f"rotation {rx:.2f}°, {ry:.2f}°, {rz:.2f}° · move "
            f"{move[0]:.4g}, {move[1]:.4g}, {move[2]:.4g} · median gap "
            f"{before:.4g} → {after:.4g} · overlap {info['fitness']:.0%}")
        self.notify(f"📐 Models aligned (median gap {before:.3g} → {after:.3g})",
                    "success")
        if show:
            self.comparison = None
            self._generation += 1
            self._show_models()
            self._fill_table()
            self._update_chart()
            self.record_checkpoint("Aligned")
        return True

    def _show_models(self, reset_camera=False):
        def describe(model):
            if model is None:
                return None
            kind = "mesh" if model.has_triangles else "point cloud"
            text = f"{model.name}\n{kind} · {len(model.vertices):,} points"
            if model.has_triangles:
                text += f" · {len(model.triangles):,} triangles"
            return text
        for label, model, empty in ((self.file_label, self.compared, "No model loaded"),
                                    (self.ref_label, self.reference,
                                     "No reference loaded")):
            text = describe(model)
            label.setText(text or empty)
            label.setProperty("role", "file" if text else "hint")
            label.style().polish(label)
        main = self.compared or self.reference
        self.set_tab_info(
            file=self.source_name or self.reference_name,
            points=None if main is None else len(main.vertices),
            triangles=None if main is None or not main.has_triangles
            else len(main.triangles))
        self._render(reset_camera=reset_camera)
        self._set_controls_enabled()

    # =========================
    # Running the comparison
    # =========================
    def selected_methods(self):
        return [key for key in ORDER if self.method_checks[key].isChecked()]

    def params(self):
        return {
            "samples": self.spin_samples.value(),
            "core_points": self.spin_cores.value(),
            "m3c2_diameter": self.spin_diameter.value(),
            "m3c2_depth_pct": self.spin_depth.value(),
        }

    def run(self):
        if self.compared is None or self.reference is None or self._worker:
            return
        methods = self.selected_methods()
        if not methods:
            self.notify("❌ Check at least one method", "warning")
            return
        compared, reference, params = self.compared, self.reference, self.params()
        generation = self._generation
        self.console.appendPlainText(
            f"⏳ Comparing with {len(methods)} methods "
            f"({params['samples']:,} samples per surface)…")

        def job(progress):
            result = run_comparison_isolated(compared, reference, methods,
                                             params, progress)
            result["generation"] = generation
            return result

        busy = getattr(self.main_window, "set_busy", None)
        if busy:
            busy(True, "Comparing models…")
        self._worker = ReconstructionWorker(job, self.vtk_widget)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_done)
        self._worker.failed.connect(
            lambda message: self.notify(f"❌ Comparison failed: {message}",
                                        "error", dialog=True))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._set_controls_enabled()

    def _on_progress(self, step):
        done, total, key = step
        progress = getattr(self.main_window, "set_progress", None)
        if progress:
            progress(done, total, f"⏳ {METHODS[key][0]} done")

    def _on_finished(self):
        self._worker.deleteLater()
        self._worker = None
        busy = getattr(self.main_window, "set_busy", None)
        if busy:
            busy(False)
        self._set_controls_enabled()

    def _on_done(self, result):
        if result.get("generation") != self._generation:
            return                     # models changed meanwhile
        self.comparison = result
        for key, item in result["results"].items():
            name = METHODS[key][0]
            if "skipped" in item:
                self.console.appendPlainText(f"   {name}: skipped ({item['skipped']})")
                continue
            s = item["stats"]
            self.console.appendPlainText(
                f"   {name:24s} RMSE {s['rmse']:.4g} · MAE {s['mae']:.4g} · "
                f"mean {s['mean']:+.4g} · R² {s['r2']:.6f} ({item['time']:.2f} s)")
        for warning in result["warnings"]:
            self.console.appendPlainText(f"⚠ {warning}")
        best = min((k for k, v in result["results"].items() if "stats" in v),
                   key=lambda k: result["results"][k]["stats"]["rmse"], default=None)
        self._fill_table()
        self._update_chart()
        self._render()
        self._set_controls_enabled()
        self.notify(
            f"✅ Compared with {len(result['results'])} methods"
            + (f" - lowest RMSE: {METHODS[best][0]}" if best else ""),
            "warning" if result["warnings"] else "success")
        self.record_checkpoint(f"Comparison ({len(result['results'])} methods)")

    def computed(self):
        """Keys of the methods with results, in display order."""
        if self.comparison is None:
            return []
        results = self.comparison["results"]
        return [k for k in ORDER if k in results and "stats" in results[k]]

    # =========================
    # Errors table / chart
    # =========================
    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if self.comparison is not None:
            results = self.comparison["results"]
            keys = [k for k in ORDER if k in results]
            computed = [k for k in keys if "stats" in results[k]]
            best = {}
            for stat in ("mae", "rmse", "sd"):
                if computed:
                    best[stat] = min(computed, key=lambda k: results[k]["stats"][stat])
            if computed:
                best["r2"] = max(computed, key=lambda k: results[k]["stats"]["r2"])
            self.table.setRowCount(len(keys))
            for row, key in enumerate(keys):
                item = results[key]
                name = QTableWidgetItem(METHODS[key][0])
                name.setData(Qt.UserRole, key)
                name.setForeground(QColor(METHOD_COLORS[key]))
                bold = name.font()
                bold.setBold(True)
                name.setFont(bold)
                name.setToolTip(METHODS[key][2])
                self.table.setItem(row, 0, name)
                for column, (_, stat, _) in enumerate(TABLE_COLUMNS[1:], 1):
                    if "skipped" in item:
                        text = "skipped" if column == 1 else ""
                    elif stat == "time":
                        text = f"{item['time']:.2f} s"
                    elif stat == "r2":
                        text = f"{item['stats']['r2']:.6f}"
                    else:
                        text = _fmt(item["stats"][stat])
                    cell = QTableWidgetItem(text)
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if best.get(stat) == key:
                        font = cell.font()
                        font.setBold(True)
                        cell.setFont(font)
                        cell.setToolTip("Best of the methods")
                    self.table.setItem(row, column, cell)
        self.table.blockSignals(False)

    def _table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        key = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        if key in self.computed():
            self.combo_method.setCurrentIndex(self.combo_method.findData(key))

    def _update_chart(self, *_):
        keys = self.computed()
        if not self.radio_all.isChecked():
            current = self.combo_method.currentData()
            keys = [current] if current in keys else []
        results = self.comparison["results"] if self.comparison else {}
        series = [(METHODS[k][0], METHOD_COLORS[k], results[k]["values"])
                  for k in keys]
        signed = self.check_signed.isChecked()
        self.chart.set_curves(series, signed, self.colour_limit(keys))

    def colour_limit(self, keys=None):
        """Colour-scale maximum: the user's value, else the largest 95th
        percentile of |e| of ``keys``."""
        if self.spin_colour_max.value() > 0:
            return self.spin_colour_max.value()
        keys = self.computed() if keys is None else keys
        results = self.comparison["results"] if self.comparison else {}
        limits = [results[k]["stats"]["p95"] for k in keys
                  if k in results and "stats" in results[k]]
        limits = [v for v in limits if math.isfinite(v) and v > 0]
        return max(limits) if limits else 1.0

    def _display_changed(self, *_):
        self._update_chart()
        self._render()

    # =========================
    # 3D view
    # =========================
    def _lookup_table(self, limit, signed):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        if signed:
            # diverging blue - white - red
            low, mid, high = (np.array([0.23, 0.30, 0.75]),
                              np.array([0.95, 0.95, 0.95]),
                              np.array([0.71, 0.02, 0.15]))
            for i in range(256):
                t = i / 255 * 2 - 1
                color = mid + (high - mid) * t if t >= 0 else mid + (low - mid) * -t
                lut.SetTableValue(i, *color, 1.0)
            lut.SetTableRange(-limit, limit)
        else:
            lut.SetHueRange(0.667, 0.0)
            lut.SetTableRange(0.0, limit)
            lut.Build()
        lut.SetNanColor(0.55, 0.55, 0.55, 1.0)
        lut.SetUseBelowRangeColor(False)
        return lut

    def _model_actor(self, model, values=None, lut=None):
        polydata = polydata_from_numpy(model.vertices, model.normals,
                                       model.triangles)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        if values is not None:
            array = numpy_support.numpy_to_vtk(
                np.ascontiguousarray(values, dtype=np.float32), deep=True)
            array.SetName("Error")
            polydata.GetPointData().SetScalars(array)
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(lut.GetTableRange())
            mapper.SetScalarModeToUsePointData()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetPointSize(2)
        prop.SetInterpolationToPhong()
        return actor

    def _values(self, key, signed):
        values = self.comparison["results"][key]["vertex_values"]
        return values if signed and METHODS[key][1] else np.abs(values)

    def _reference_overlay(self):
        actor = self._model_actor(self.reference)
        # Drawn just behind coincident surfaces: no z-fighting speckle on
        # the coloured model where the two surfaces agree
        mapper = actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(4, 40)
        prop = actor.GetProperty()
        prop.SetColor(*self._model_rgb("reference"))
        prop.SetOpacity(self.model_colours["reference_opacity"] / 100.0)
        actor.SetPickable(False)
        return actor

    # ---------- colours of the two models ----------
    def _model_rgb(self, which):
        """RGB (0..1) of ``which`` model: the chosen colour, else the
        theme colour (reference = the overlay colour it always had)."""
        colour = self.model_colours[which]
        if colour:
            return QColor(colour).getRgbF()[:3]
        return self.viewport.color("mesh" if which == "compared" else "points")

    def _apply_main_colour(self, model, coloured):
        """Solid colour of the main actor (not when it shows the error);
        ``None`` keeps the Object settings / file colours."""
        if self.actor is None or coloured:
            return
        which = "compared" if model is self.compared else "reference"
        if self.model_colours[which]:
            self.actor.GetMapper().ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(*self._model_rgb(which))

    def _refresh_colour_buttons(self):
        for which, button in (("compared", self.btn_color_compared),
                              ("reference", self.btn_color_reference)):
            button.blockSignals(True)
            button.set_color(QColor.fromRgbF(*self._model_rgb(which)).name())
            button.setText(button.color() if self.model_colours[which]
                           else f"{button.color()} (theme)")
            button.blockSignals(False)

    def _set_model_colour(self, key, value):
        self.model_colours[key] = value
        settings.set_view_settings("compare", self.model_colours)
        self._refresh_colour_buttons()
        if self.compared is not None or self.reference is not None:
            self._render()

    def default_model_colours(self):
        self.model_colours = dict(MODEL_COLOUR_DEFAULTS)
        settings.set_view_settings("compare", self.model_colours)
        for widget, value in ((self.spin_ref_opacity,
                               self.model_colours["reference_opacity"]),
                              (self.check_error_colours,
                               self.model_colours["error_colours"])):
            widget.blockSignals(True)
            (widget.setValue if widget is self.spin_ref_opacity
             else widget.setChecked)(value)
            widget.blockSignals(False)
        self._refresh_colour_buttons()
        if self.compared is not None or self.reference is not None:
            self._render()

    def _render(self, *_, reset_camera=False):
        """Rebuild the view from the models, results and display options."""
        self._clear_grid()
        self._set_scalar_bar(None)            # bar of the single view
        if self.ref_actor is not None:
            self.renderer.RemoveActor(self.ref_actor)
            self.ref_actor = None
        main_model = self.compared or self.reference
        if main_model is None:
            if self.actor is not None:
                self.renderer.RemoveActor(self.actor)
                self.actor = None
            self._set_scalar_bar(None)
            self.viewport.set_empty(True)
            self.viewport.render()
            self.object_info.schedule()
            return

        computed = self.computed() if self.compared is not None else []
        if not self.model_colours["error_colours"]:
            computed = []                 # both models in their own colours
        signed = self.check_signed.isChecked()
        grid = self.combo_mode.currentData() == "grid" and len(computed) > 1
        key = self.combo_method.currentData()
        if computed and key not in computed:
            key = computed[0]
        keys = computed if grid else ([key] if computed else [])
        limit = self.colour_limit(keys)
        lut = self._lookup_table(limit, signed) if keys else None

        values = self._values(keys[0], signed) if keys else None
        actor = self._model_actor(main_model, values, lut)
        self.show_actor(actor, reset_camera=reset_camera or self.actor is None,
                        kind="mesh" if main_model.has_triangles else "points")
        self._apply_main_colour(main_model, values is not None)
        if self.compared is not None and self.reference is not None \
                and self.check_overlay.isChecked():
            self.ref_actor = self._reference_overlay()
            self.renderer.AddActor(self.ref_actor)
        if grid:
            self._build_grid(keys, signed, lut)
        else:
            self._set_scalar_bar(
                lut, f"{'e' if signed else '|e|'} · {METHODS[key][0]}"
                if keys else None)
        self._sync_secondary()
        self._update_chart()
        self.viewport.render()

    # ---------- grid of linked views ----------
    def _cells(self, count):
        columns = min(4, count)
        rows = math.ceil(count / columns)
        for i in range(count):
            row, column = divmod(i, columns)
            yield (column / columns, 1 - (row + 1) / rows,
                   (column + 1) / columns, 1 - row / rows)

    def _build_grid(self, keys, signed, lut):
        """One view per method (first = the main view) + a legend cell;
        all share the camera, so they turn and zoom together."""
        window = self.vtk_widget.GetRenderWindow()
        camera = self.renderer.GetActiveCamera()
        cells = list(self._cells(len(keys) + 1))
        self.renderer.SetViewport(*cells[0])
        results = self.comparison["results"]
        for i, key in enumerate(keys):
            if i == 0:
                renderer = self.renderer
            else:
                renderer = vtk.vtkRenderer()
                renderer.SetViewport(*cells[i])
                renderer.SetActiveCamera(camera)
                window.AddRenderer(renderer)
                actor = self._model_actor(self.compared, self._values(key, signed), lut)
                actor.GetProperty().SetOpacity(self.actor.GetProperty().GetOpacity())
                renderer.AddActor(actor)
                overlay = None
                if self.reference is not None and self.check_overlay.isChecked():
                    overlay = self._reference_overlay()
                    renderer.AddActor(overlay)
                self._grid.append((renderer, actor, overlay))
            stats = results[key]["stats"]
            label = vtk.vtkTextActor()
            label.SetInput(f"{METHODS[key][0]}\nRMSE {stats['rmse']:.4g} · "
                           f"MAE {stats['mae']:.4g}")
            text = label.GetTextProperty()
            text.SetFontSize(13)
            text.SetBold(True)
            text.SetVerticalJustificationToTop()
            text.SetColor(*QColor(METHOD_COLORS[key]).getRgbF()[:3])
            text.ShadowOff()
            label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            label.SetPosition(0.03, 0.95)
            renderer.AddViewProp(label)
            self._grid_labels.append((renderer, label))
        legend = vtk.vtkRenderer()
        legend.SetViewport(*cells[-1])
        window.AddRenderer(legend)
        self._legend_renderer = legend
        self._set_scalar_bar(lut, "e" if signed else "|e|", legend, wide=True)
        self._sync_backgrounds()

    def screenshot_renderers(self):
        """Renderers a screenshot copies: the main view, plus every grid
        cell and the legend in the grid mode."""
        return ([self.renderer] + [r for r, *_ in self._grid]
                + ([self._legend_renderer] if self._legend_renderer else []))

    def _clear_grid(self):
        window = self.vtk_widget.GetRenderWindow()
        for renderer, *_ in self._grid:
            window.RemoveRenderer(renderer)
        for renderer, label in self._grid_labels:
            renderer.RemoveViewProp(label)
        if self._legend_renderer is not None:
            window.RemoveRenderer(self._legend_renderer)
            self._legend_renderer = None
            self._scalar_bar = None
        self._grid, self._grid_labels = [], []
        self.renderer.SetViewport(0, 0, 1, 1)

    def _sync_backgrounds(self):
        source = self.renderer
        for renderer in [r for r, *_ in self._grid] + (
                [self._legend_renderer] if self._legend_renderer else []):
            renderer.SetGradientBackground(source.GetGradientBackground())
            renderer.SetBackground(source.GetBackground())
            renderer.SetBackground2(source.GetBackground2())

    def _set_scalar_bar(self, lut, title=None, renderer=None, wide=False):
        renderer = renderer or self.renderer
        if self._scalar_bar is not None and self._legend_renderer is None:
            self.renderer.RemoveViewProp(self._scalar_bar)
            self._scalar_bar = None
        if lut is None:
            return
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle(title or "")
        bar.SetNumberOfLabels(5)
        bar.SetLabelFormat("%.3g")
        if wide:
            bar.SetPosition(0.3, 0.1)
            bar.SetWidth(0.3)
            bar.SetHeight(0.8)
        else:
            bar.SetPosition(0.88, 0.1)
            bar.SetWidth(0.1)
            bar.SetHeight(0.8)
        color = self.viewport.color("text")
        bar.UnconstrainedFontSizeOn()            # fixed sizes, any cell size
        for prop, size in ((bar.GetTitleTextProperty(), 14),
                           (bar.GetLabelTextProperty(), 12)):
            prop.SetColor(*color)
            prop.SetFontSize(size)
            prop.ShadowOff()
        bar.SetPickable(False)
        renderer.AddViewProp(bar)
        self._scalar_bar = bar

    def _sync_secondary(self):
        """Overlay / grid actors take the main actor's rotation + position."""
        if self.actor is None:
            return
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(self.actor.GetMatrix())
        actors = [self.ref_actor] + [a for _, actor, overlay in self._grid
                                     for a in (actor, overlay)]
        for actor in actors:
            if actor is not None:
                actor.SetUserMatrix(matrix)

    def _pose_changed(self):
        super()._pose_changed()
        if hasattr(self, "_grid"):
            self._sync_secondary()

    def refresh_theme(self):
        """Theme switched: grid backgrounds, legend and chart follow."""
        self._refresh_colour_buttons()
        if self.compared is not None or self.reference is not None:
            self._render()
        self.chart.update()

    # =========================
    # Slice
    # =========================
    def slice_source(self):
        model = self.compared or self.reference
        if model is None:
            return None
        return {"kind": "mesh" if model.has_triangles else "points",
                "vertices": model.vertices, "triangles": model.triangles,
                "normals": model.normals, "colors": None}

    def slice_actors(self):
        actors = [self.actor, self.ref_actor]
        actors += [a for _, actor, overlay in self._grid for a in (actor, overlay)]
        return [a for a in actors if a is not None]

    def apply_slice(self, planes, cap):
        """Slice both models with the same planes: the comparison then
        covers that region only (run it again)."""
        from modules.slice_panel import slice_source_data

        def cut(model):
            if model is None:
                return None
            source = {"kind": "mesh" if model.has_triangles else "points",
                      "vertices": model.vertices, "triangles": model.triangles,
                      "normals": model.normals, "colors": None}
            kind, vertices, triangles, normals, _ = slice_source_data(source, planes, cap)
            if len(vertices) == 0:
                raise ValueError(f"nothing of {model.name or 'a model'} is "
                                 "left inside the planes")
            return Model(vertices, triangles, normals, f"{model.name} ✂")
        compared, reference = cut(self.compared), cut(self.reference)
        self.set_models(compared, reference,
                        compared.name if compared else self.source_name,
                        reference.name if reference else self.reference_name,
                        pose=self.pose_state())
        self.console.appendPlainText(
            "✂ Both models sliced - run the comparison again for this region")
        parts = [f"{len(m.vertices):,}" for m in (compared, reference) if m is not None]
        return "Sliced " + " / ".join(parts) + " vertices - run the comparison again"

    def view_settings_changed(self, kind):
        if self.actor is not None:          # Object settings restyled it
            main_model = self.compared or self.reference
            scalars = self.actor.GetMapper().GetInput().GetPointData().GetScalars()
            self._apply_main_colour(
                main_model, scalars is not None and scalars.GetName() == "Error")
        if self._grid or self._legend_renderer:
            self._sync_backgrounds()
            self.viewport.render()

    # =========================
    # Output
    # =========================
    def table_rows(self):
        rows = []
        for key in self.computed():
            item = self.comparison["results"][key]
            s = item["stats"]
            rows.append([METHODS[key][0]] + [
                s[k] for k in ("n", "mean", "sd", "mae", "rmse", "r2",
                               "median", "p95", "max")] + [item["time"]])
        return rows

    def export_data(self):
        if self.comparison is None:
            self.notify("❌ Run a comparison first", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Comparison",
            os.path.join(settings.last_dir(), "comparison.csv"),
            "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        settings.remember_path(path)
        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["compared", self.source_name])
                writer.writerow(["reference", self.reference_name])
                writer.writerow(["alignment (4x4, applied to compared)"]
                                + [f"{v:.9g}" for v in self.alignment.ravel()])
                for name, value in self.comparison["params"].items():
                    writer.writerow([name, value])
                writer.writerow([])
                writer.writerow(["method", "n", "mean_e", "sd", "mae", "rmse",
                                 "r2", "median_abs", "p95_abs", "max_abs",
                                 "time_s"])
                writer.writerows(self.table_rows())
            self.notify(f"✅ Saved {os.path.basename(path)}", "success")
        except OSError as e:
            self.notify(f"❌ Export failed: {e}", "error", dialog=True)

    def markdown(self):
        header = ["Method", "N", "Mean e", "SD", "MAE", "RMSE", "R²",
                  "Median |e|", "P95 |e|", "Max |e|", "Time (s)"]
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "---|" * len(header)]
        for row in self.table_rows():
            cells = [row[0], f"{row[1]:,}"] + [_fmt(v) for v in row[2:6]] + [
                f"{row[6]:.6f}"] + [_fmt(v) for v in row[7:10]] + [f"{row[10]:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def copy_markdown(self):
        if self.comparison is None:
            return
        QApplication.clipboard().setText(self.markdown())
        self.notify("Table copied as Markdown", "success")

    # =========================
    # Object info / checkpoints / remove
    # =========================
    def info_source(self):
        model = self.compared or self.reference
        if model is None:
            return None
        rows = [("Compared", self.source_name or "-", "Model coloured by the error"),
                ("Reference", self.reference_name or "-", "Model compared against")]
        if self.reference is not None:
            rows.append(("Reference points", f"{len(self.reference.vertices):,}",
                         "Vertices of the reference"))
        if not np.allclose(self.alignment, np.eye(4)):
            (rx, ry, rz), move = describe(self.alignment)
            rows.append(("Alignment rotation", f"({rx:.2f}, {ry:.2f}, {rz:.2f})°",
                         "Rotation applied to the reconstruction to match"))
            rows.append(("Alignment move",
                         "(" + ", ".join(f"{v:.4g}" for v in move) + ")",
                         "Translation applied to the reconstruction to match"))
        return model.vertices, model.normals, model.triangles, ("Comparison", rows)

    def snapshot(self):
        return {
            "compared": self.compared, "reference": self.reference,
            "source": self.source_name, "reference_name": self.reference_name,
            "comparison": self.comparison, "alignment": self.alignment,
            **self.pose_state(),
        }

    def restore(self, state):
        self._generation += 1
        self.compared, self.reference = state["compared"], state["reference"]
        self.source_name = state["source"]
        self.reference_name = state["reference_name"]
        self.comparison = state["comparison"]
        self.alignment = state["alignment"]
        self.restore_pose(state)
        self._show_models()
        self._fill_table()
        self._update_chart()

    def has_data(self):
        return self.compared is not None or self.reference is not None

    def clear_data(self):
        self._generation += 1
        self.compared = self.reference = None
        self.comparison = None
        self.alignment = np.eye(4)
        self.source_name = self.reference_name = ""
        self._clear_grid()
        if self.ref_actor is not None:
            self.renderer.RemoveActor(self.ref_actor)
            self.ref_actor = None
        self._set_scalar_bar(None)
        self._fill_table()
        self._update_chart()
        self.clear_view()
        self.ref_label.setText("No reference loaded")
        self.ref_label.setProperty("role", "hint")
        self.ref_label.style().polish(self.ref_label)
        self._set_controls_enabled()
        self.console.appendPlainText(
            "Ready. Choose the reconstruction (compared model) and the "
            "reference, then Run comparison.")
