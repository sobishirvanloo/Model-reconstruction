"""Object / background appearance of the 3D views.

* ``ObjectSettingsDialog``      colour (the file's own colours or a custom
                                one), display style, opacity, point size
* ``BackgroundSettingsDialog``  background colour (theme / solid /
                                gradient) and visual guides: floor grid,
                                ruler, start-pose axes and outline, a
                                rotation / position read-out
* ``SceneOverlays``             draws those guides in a renderer

Settings are stored with QSettings (they survive a restart) and apply to
every tab; each dialog has a *Restore Defaults* button.
"""

import math

import numpy as np
import vtk
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from modules import settings

OBJECT = "object"
BACKGROUND = "background"

DEFAULTS = {
    OBJECT: {
        "color_mode": "original",     # "original" / "custom"
        "color": "#e0a040",
        "style": "surface",           # see STYLES
        "opacity": 100,               # %
        "point_size": 2,
        "edge_color": "#1b2430",
    },
    BACKGROUND: {
        "mode": "theme",              # "theme" / "solid" / "gradient"
        "color": "#ffffff",
        "color_top": "#ffffff",
        "color_bottom": "#c9d3de",
        "grid": False,
        "ruler": False,
        "start_axes": False,
        "start_outline": False,
        "readout": False,
    },
}

STYLES = {
    "surface": "Surface",
    "surface_edges": "Surface with edges (mesh lines)",
    "wireframe": "Wireframe",
    "points": "Points (vertices only)",
}


def load(kind):
    """Saved settings of ``kind`` merged over the defaults."""
    values = dict(DEFAULTS[kind])
    values.update(settings.view_settings(kind))
    return values


def save(kind, values):
    settings.set_view_settings(kind, values)


def rgb(hex_color):
    """'#rrggbb' -> (r, g, b) in 0..1 for VTK."""
    color = QColor(hex_color)
    return color.redF(), color.greenF(), color.blueF()


# =========================
# Dialogs
# =========================
class ColorButton(QPushButton):
    """Button that shows a colour swatch and opens a colour picker."""

    changed = pyqtSignal(str)

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setMinimumWidth(110)
        self.clicked.connect(self._pick)
        self.set_color(color)

    def color(self):
        return self._color

    def set_color(self, color):
        self._color = QColor(color).name()
        text = "#000000" if QColor(self._color).lightness() > 140 else "#ffffff"
        self.setText(self._color)
        self.setStyleSheet(
            f"background-color: {self._color}; color: {text};"
            "border: 1px solid #8892a0; border-radius: 6px; padding: 5px 10px;")

    def _pick(self):
        color = QColorDialog.getColor(QColor(self._color), self, "Choose a colour")
        if color.isValid():
            self.set_color(color.name())
            self.changed.emit(self._color)


class _SettingsDialog(QDialog):
    """Common frame: Restore Defaults / Cancel / Apply / Save buttons.
    ``saved(kind, values)`` is emitted by Apply and Save."""

    saved = pyqtSignal(str, dict)
    KIND = None
    TITLE = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.layout_ = QVBoxLayout(self)
        self._build()
        buttons = QDialogButtonBox(
            QDialogButtonBox.RestoreDefaults | QDialogButtonBox.Cancel
            | QDialogButtonBox.Apply | QDialogButtonBox.Save)
        buttons.button(QDialogButtonBox.RestoreDefaults).setText("Default")
        buttons.button(QDialogButtonBox.RestoreDefaults).setToolTip(
            "Fill in the original settings (Save to keep them)")
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: self.set_values(DEFAULTS[self.KIND]))
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._emit)
        buttons.button(QDialogButtonBox.Save).setProperty("variant", "primary")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self.buttons = buttons
        self.layout_.addWidget(buttons)

    def open_with(self, values):
        self.set_values(values)
        self.show()
        self.raise_()
        self.activateWindow()

    def _emit(self):
        values = self.values()
        save(self.KIND, values)
        self.saved.emit(self.KIND, values)

    def _save(self):
        self._emit()
        self.accept()


class ObjectSettingsDialog(_SettingsDialog):
    KIND = OBJECT
    TITLE = "Object appearance"

    def _build(self):
        colour = QGroupBox("Colour")
        box = QVBoxLayout(colour)
        self.radio_original = QRadioButton(
            "Original colours (from the file; theme colour when it has none)")
        self.radio_custom = QRadioButton("Custom colour")
        self.color_button = ColorButton()
        self.color_button.changed.connect(lambda _: self.radio_custom.setChecked(True))
        group = QButtonGroup(self)
        group.addButton(self.radio_original)
        group.addButton(self.radio_custom)
        row = QHBoxLayout()
        row.addWidget(self.radio_custom)
        row.addWidget(self.color_button)
        row.addStretch(1)
        box.addWidget(self.radio_original)
        box.addLayout(row)
        self.layout_.addWidget(colour)

        display = QGroupBox("Display")
        form = QFormLayout(display)
        self.combo_style = QComboBox()
        for key, label in STYLES.items():
            self.combo_style.addItem(label, key)
        self.combo_style.setToolTip(
            "How a mesh is drawn (point clouds are always drawn as points)")
        self.edge_button = ColorButton()
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.label_opacity = QLabel()
        self.slider_opacity.valueChanged.connect(
            lambda v: self.label_opacity.setText(f"{v}%"))
        opacity = QHBoxLayout()
        opacity.addWidget(self.slider_opacity, 1)
        opacity.addWidget(self.label_opacity)
        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 15)
        self.spin_point_size.setSuffix(" px")
        form.addRow("Style", self.combo_style)
        form.addRow("Edge colour", self.edge_button)
        form.addRow("Opacity", opacity)
        form.addRow("Point size", self.spin_point_size)
        self.layout_.addWidget(display)

    def values(self):
        return {
            "color_mode": "custom" if self.radio_custom.isChecked() else "original",
            "color": self.color_button.color(),
            "style": self.combo_style.currentData(),
            "opacity": self.slider_opacity.value(),
            "point_size": self.spin_point_size.value(),
            "edge_color": self.edge_button.color(),
        }

    def set_values(self, values):
        custom = values["color_mode"] == "custom"
        self.radio_custom.setChecked(custom)
        self.radio_original.setChecked(not custom)
        self.color_button.set_color(values["color"])
        self.combo_style.setCurrentIndex(
            max(0, self.combo_style.findData(values["style"])))
        self.slider_opacity.setValue(int(values["opacity"]))
        self.label_opacity.setText(f"{int(values['opacity'])}%")
        self.spin_point_size.setValue(int(values["point_size"]))
        self.edge_button.set_color(values["edge_color"])


class BackgroundSettingsDialog(_SettingsDialog):
    KIND = BACKGROUND
    TITLE = "Background and guides"

    GUIDES = (
        ("grid", "Floor grid",
         "A grid under the model at its starting height; the size of one "
         "square is shown in the read-out"),
        ("ruler", "Ruler (scale on the model's box)",
         "Axes with tick labels in model units around the model"),
        ("start_axes", "Start-pose axes",
         "X / Y / Z axes at the model's starting centre (position 0, 0, 0, "
         "rotation 0°)"),
        ("start_outline", "Start-pose outline",
         "The box the model filled before it was rotated or moved"),
        ("readout", "Rotation / position read-out",
         "Angles and offsets relative to the starting pose, in a corner"),
    )

    def _build(self):
        colour = QGroupBox("Colour")
        grid = QVBoxLayout(colour)
        self.radio_theme = QRadioButton("Theme (light / dark gradient)")
        self.radio_solid = QRadioButton("Solid colour")
        self.radio_gradient = QRadioButton("Gradient")
        group = QButtonGroup(self)
        for radio in (self.radio_theme, self.radio_solid, self.radio_gradient):
            group.addButton(radio)
        self.color_button = ColorButton()
        self.top_button = ColorButton()
        self.bottom_button = ColorButton()
        self.color_button.changed.connect(lambda _: self.radio_solid.setChecked(True))
        for button in (self.top_button, self.bottom_button):
            button.changed.connect(lambda _: self.radio_gradient.setChecked(True))
        grid.addWidget(self.radio_theme)
        row = QHBoxLayout()
        row.addWidget(self.radio_solid)
        row.addWidget(self.color_button)
        row.addStretch(1)
        grid.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.radio_gradient)
        row.addWidget(QLabel("top"))
        row.addWidget(self.top_button)
        row.addWidget(QLabel("bottom"))
        row.addWidget(self.bottom_button)
        row.addStretch(1)
        grid.addLayout(row)
        self.layout_.addWidget(colour)

        guides = QGroupBox("Grid, scale and reference")
        box = QVBoxLayout(guides)
        self.checks = {}
        for key, text, tip in self.GUIDES:
            check = QCheckBox(text)
            check.setToolTip(tip)
            box.addWidget(check)
            self.checks[key] = check
        hint = QLabel("Rotation and position are measured from the pose the "
                      "model had when it was loaded.")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        box.addWidget(hint)
        self.layout_.addWidget(guides)

    def values(self):
        mode = ("solid" if self.radio_solid.isChecked()
                else "gradient" if self.radio_gradient.isChecked() else "theme")
        values = {
            "mode": mode,
            "color": self.color_button.color(),
            "color_top": self.top_button.color(),
            "color_bottom": self.bottom_button.color(),
        }
        values.update({key: check.isChecked() for key, check in self.checks.items()})
        return values

    def set_values(self, values):
        {"solid": self.radio_solid, "gradient": self.radio_gradient}.get(
            values["mode"], self.radio_theme).setChecked(True)
        self.color_button.set_color(values["color"])
        self.top_button.set_color(values["color_top"])
        self.bottom_button.set_color(values["color_bottom"])
        for key, check in self.checks.items():
            check.setChecked(bool(values[key]))


# =========================
# Guides drawn in the 3D view
# =========================
def nice_step(length, divisions=10):
    """A round grid step (1, 2 or 5 x 10^n) close to length / divisions."""
    raw = max(length / divisions, 1e-12)
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if raw <= factor * power:
            return factor * power
    return 10 * power


def format_length(value, digits=6):
    return f"{value:.{digits}g}"


class SceneOverlays:
    """Grid, ruler, start-pose axes / outline and the pose read-out of one
    renderer.  ``update`` is cheap; the grid is only rebuilt when the
    model's starting box or the colour changes."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.props = []
        self._grid_key = None

        self.grid = self._line_actor()
        self.outline = self._line_actor()
        self.start_axes = vtk.vtkAxesActor()
        self.start_axes.SetAxisLabels(0)
        self.start_axes.SetShaftTypeToLine()
        self.ruler = vtk.vtkCubeAxesActor()
        self.ruler.SetCamera(renderer.GetActiveCamera())
        self.ruler.SetFlyModeToOuterEdges()
        self.ruler.SetGridLineLocation(self.ruler.VTK_GRID_LINES_FURTHEST)
        self.ruler.SetScreenSize(7.0)          # small labels
        self.ruler.SetLabelOffset(12)
        for axis in "XYZ":
            getattr(self.ruler, f"{axis}AxisMinorTickVisibilityOff")()
        self.ruler.SetXTitle("X")
        self.ruler.SetYTitle("Y")
        self.ruler.SetZTitle("Z")
        self.readout = vtk.vtkTextActor()
        self.readout.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.readout.SetPosition(0.01, 0.97)
        text = self.readout.GetTextProperty()
        text.SetFontSize(13)
        text.SetVerticalJustificationToTop()
        text.SetFontFamilyToCourier()
        text.ShadowOff()

        for prop in (self.grid, self.outline, self.start_axes, self.ruler,
                     self.readout):
            prop.SetPickable(False)
            prop.SetVisibility(False)
            renderer.AddViewProp(prop)
            self.props.append(prop)

    @staticmethod
    def _line_actor():
        actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vtk.vtkPolyData())
        actor.SetMapper(mapper)
        actor.GetProperty().LightingOff()
        return actor

    def owns(self, prop):
        return any(prop is p for p in self.props)

    def update(self, options, start_bounds, actor, rotation, position, color):
        """Show the enabled guides for a model whose starting box is
        ``start_bounds`` (``None`` hides everything)."""
        if start_bounds is None or actor is None:
            for prop in self.props:
                prop.SetVisibility(False)
            return
        bounds = np.asarray(start_bounds, dtype=float)
        low, high = bounds[0::2], bounds[1::2]
        center = (low + high) / 2.0
        diagonal = float(np.linalg.norm(high - low)) or 1.0
        step = nice_step(diagonal)

        if options["grid"]:
            self._update_grid(low, high, center, diagonal, step, color)
        self.grid.SetVisibility(bool(options["grid"]))

        if options["start_outline"]:
            source = vtk.vtkOutlineSource()
            source.SetBounds(*bounds)
            source.Update()
            self.outline.GetMapper().SetInputData(source.GetOutput())
            prop = self.outline.GetProperty()
            prop.SetColor(*color)
            prop.SetOpacity(0.7)
            prop.SetLineWidth(1.5)
        self.outline.SetVisibility(bool(options["start_outline"]))

        if options["start_axes"]:
            length = diagonal * 0.4
            self.start_axes.SetTotalLength(length, length, length)
            transform = vtk.vtkTransform()
            transform.Translate(*center)
            self.start_axes.SetUserTransform(transform)
        self.start_axes.SetVisibility(bool(options["start_axes"]))

        if options["ruler"]:
            self.ruler.SetBounds(actor.GetBounds())
            self.ruler.SetCamera(self.renderer.GetActiveCamera())
            for axis in range(3):
                self.ruler.GetTitleTextProperty(axis).SetColor(*color)
                self.ruler.GetLabelTextProperty(axis).SetColor(*color)
            for prop in (self.ruler.GetXAxesLinesProperty(),
                         self.ruler.GetYAxesLinesProperty(),
                         self.ruler.GetZAxesLinesProperty()):
                prop.SetColor(*color)
        self.ruler.SetVisibility(bool(options["ruler"]))

        if options["readout"]:
            lines = [
                "Rotation  X {:7.1f}°  Y {:7.1f}°  Z {:7.1f}°".format(*rotation),
                "Position  X {:>8}  Y {:>8}  Z {:>8}".format(
                    *(format_length(v) for v in position)),
                "Size      {} × {} × {}".format(
                    *(format_length(v, 4) for v in high - low)),
            ]
            if options["grid"]:
                lines.append(f"Grid      1 square = {format_length(step)} units")
            self.readout.SetInput("\n".join(lines))
            self.readout.GetTextProperty().SetColor(*color)
        self.readout.SetVisibility(bool(options["readout"]))

    def _update_grid(self, low, high, center, diagonal, step, color):
        key = (tuple(low), tuple(high), tuple(color))
        if key == self._grid_key:
            return
        self._grid_key = key
        # Floor: the XZ plane at the bottom (lowest Y) of the starting box
        half = math.ceil(diagonal * 0.75 / step) * step
        x0 = math.floor((center[0] - half) / step) * step
        z0 = math.floor((center[2] - half) / step) * step
        count = int(round(2 * half / step)) + 1
        x1 = x0 + (count - 1) * step
        z1 = z0 + (count - 1) * step
        y = low[1]
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for i in range(count):
            for start, end in (((x0 + i * step, y, z0), (x0 + i * step, y, z1)),
                               ((x0, y, z0 + i * step), (x1, y, z0 + i * step))):
                a = points.InsertNextPoint(*start)
                b = points.InsertNextPoint(*end)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(a)
                lines.InsertCellPoint(b)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        self.grid.GetMapper().SetInputData(polydata)
        prop = self.grid.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(0.35)
