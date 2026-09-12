"""Shared building blocks for the three VTK tabs.

This module centralises code that used to be copy-pasted into every tab
module (import_mesh / point_processing / poisson_reconstruction):

* ``CustomInteractorStyle`` — mouse-wheel (dolly) zoom.
* ``VTKViewport``        — console + VTK render window + orientation axes.
* ``RotationMixin``      — slider/spinbox <-> actor rotation logic.
"""

import math

import vtk
from PyQt5.QtWidgets import QPlainTextEdit, QVBoxLayout
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


CONSOLE_STYLESHEET = (
    "background-color: #007ea7; color: white; "
    "font-family: Consolas; font-size: 16px;"
)


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballActor):
    """Default trackball interaction style plus mouse-wheel zooming."""

    ZOOM_IN_FACTOR = 1.1
    ZOOM_OUT_FACTOR = 0.9

    def __init__(self, renderer, parent=None):
        self.AddObserver("MouseWheelForwardEvent", self.zoom_in)
        self.AddObserver("MouseWheelBackwardEvent", self.zoom_out)
        self.renderer = renderer

    def _dolly(self, factor):
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.GetInteractor().GetRenderWindow().Render()

    def zoom_in(self, obj, event):
        self._dolly(self.ZOOM_IN_FACTOR)

    def zoom_out(self, obj, event):
        self._dolly(self.ZOOM_OUT_FACTOR)


class VTKViewport:
    """Creates the console + VTK render window for one tab.

    The widgets/objects are stored directly on ``owner`` using the same
    attribute names the rest of the application expects:

        console, vtk_widget, renderer, interactor, marker
    """

    def __init__(self, owner, frame_console, frame_view):
        owner.frame_console = frame_console
        owner.frame_view = frame_view

        owner.console = QPlainTextEdit(frame_console)
        owner.console.setReadOnly(True)
        owner.console.setGeometry(frame_console.rect())
        owner.console.setStyleSheet(CONSOLE_STYLESHEET)
        console_layout = QVBoxLayout(frame_console)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.addWidget(owner.console)

        owner.vtk_widget = QVTKRenderWindowInteractor(frame_view)
        layout = QVBoxLayout(frame_view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(owner.vtk_widget)

        owner.renderer = vtk.vtkRenderer()
        owner.renderer.SetBackground(0.95, 0.95, 0.95)
        owner.vtk_widget.GetRenderWindow().AddRenderer(owner.renderer)
        owner.vtk_widget.GetRenderWindow().SetMultiSamples(0)
        owner.interactor = owner.vtk_widget.GetRenderWindow().GetInteractor()

        # ---------- VTK Initialization ----------
        owner.interactor.Initialize()
        style = CustomInteractorStyle(owner.renderer)
        owner.interactor.SetInteractorStyle(style)

        self.setup_axes(owner)

        # Initial render
        owner.vtk_widget.GetRenderWindow().Render()

    @staticmethod
    def setup_axes(owner):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1, 1, 1)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.03)
        axes.SetAxisLabels(1)

        axes.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axes.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axes.GetZAxisShaftProperty().SetColor(0, 0, 1)

        owner.marker = vtk.vtkOrientationMarkerWidget()
        owner.marker.SetOrientationMarker(axes)
        owner.marker.SetInteractor(owner.interactor)
        owner.marker.SetViewport(0.0, 0.0, 0.2, 0.2)
        owner.marker.SetEnabled(1)
        owner.marker.InteractiveOff()

    @staticmethod
    def reset_camera(owner):
        owner.renderer.ResetCamera()
        owner.vtk_widget.GetRenderWindow().Render()

class RotationMixin:
    """Slider/spinbox <-> actor rotation, shared by every tab.

    Expected instance attributes:

        spinBox_X/Y/Z, slider_X/Y/Z     Qt widgets
        rotation_*, prev_rotation_*     float angles (degrees)
        mesh_actor or actor             the VTK actor being rotated
        vtk_widget, console
    """

    NO_ACTOR_MESSAGE = "❌ Please import a file first"

    # ---------- public API ----------

    def setup_rotation_controls(self):
        """Wire the X/Y/Z slider <-> spinbox pairs together."""
        self.slider_X.valueChanged.connect(
            lambda v: self.sync_rotation(self.spinBox_X, "X", v))
        self.spinBox_X.valueChanged.connect(
            lambda v: self.sync_rotation(self.slider_X, "X", v))
        self.slider_Y.valueChanged.connect(
            lambda v: self.sync_rotation(self.spinBox_Y, "Y", v))
        self.spinBox_Y.valueChanged.connect(
            lambda v: self.sync_rotation(self.slider_Y, "Y", v))
        self.slider_Z.valueChanged.connect(
            lambda v: self.sync_rotation(self.spinBox_Z, "Z", v))
        self.spinBox_Z.valueChanged.connect(
            lambda v: self.sync_rotation(self.slider_Z, "Z", v))

    def sync_rotation(self, widget_to_update, axis, value):
        setattr(self, f"rotation_{axis.lower()}", float(value))
        widget_to_update.blockSignals(True)
        widget_to_update.setValue(int(value))
        widget_to_update.blockSignals(False)

    def update_rotation(self, axis, value):
        setattr(self, f"rotation_{axis.lower()}", float(value))
        for widget in (getattr(self, f"spinBox_{axis}"),
                       getattr(self, f"slider_{axis}")):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)

    def apply_rotation(self):
        actor = self._rotation_actor()
        if actor is None:
            self.console.appendPlainText(self.NO_ACTOR_MESSAGE)
            return
        try:
            delta_x = self.rotation_x - self.prev_rotation_x
            delta_y = self.rotation_y - self.prev_rotation_y
            delta_z = self.rotation_z - self.prev_rotation_z

            actor.RotateX(delta_x)
            actor.RotateY(delta_y)
            actor.RotateZ(delta_z)

            self.prev_rotation_x = self.rotation_x
            self.prev_rotation_y = self.rotation_y
            self.prev_rotation_z = self.rotation_z

            self.vtk_widget.GetRenderWindow().Render()

            self.console.appendPlainText(
                f"✅ Rotation applied | X: {self.rotation_x:.1f}° | "
                f"Y: {self.rotation_y:.1f}° | Z: {self.rotation_z:.1f}°"
            )
        except Exception as e:
            self.console.appendPlainText(f"❌ Error: {str(e)}")

    # ---------- internals ----------

    def _rotation_actor(self):
        return getattr(self, "mesh_actor", None) or getattr(self, "actor", None)

    def update_spinbox_from_actor(self, caller=None, event=None):
        """Sync the UI after the user rotates the actor with the mouse.

        Uses ``vtkProp3D.GetOrientation()`` - suitable for actors rotated
        with RotateX/RotateY/RotateZ (used by the mesh / point tabs).
        """
        actor = self._rotation_actor()
        if actor is None:
            return
        ox, oy, oz = actor.GetOrientation()
        self.rotation_x, self.rotation_y, self.rotation_z = ox, oy, oz
        for axis, value in zip("XYZ", (ox, oy, oz)):
            for widget in (getattr(self, f"spinBox_{axis}"),
                           getattr(self, f"slider_{axis}")):
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)
        self.prev_rotation_x, self.prev_rotation_y, self.prev_rotation_z = ox, oy, oz

    def update_spinbox_from_matrix(self, caller=None, event=None):
        """Sync the UI after mouse rotation using the actor's 3x3 matrix.

        More accurate than ``GetOrientation`` for composed rotations
        (used by the reconstruction tab).
        """
        actor = self._rotation_actor()
        if actor is None:
            return
        try:
            matrix = actor.GetMatrix()
            sy = math.sqrt(matrix.GetElement(0, 0) ** 2
                           + matrix.GetElement(1, 0) ** 2)
            singular = sy < 1e-6

            if not singular:
                rx = math.degrees(math.atan2(matrix.GetElement(2, 1),
                                             matrix.GetElement(2, 2)))
                ry = math.degrees(math.atan2(-matrix.GetElement(2, 0), sy))
                rz = math.degrees(math.atan2(matrix.GetElement(1, 0),
                                             matrix.GetElement(0, 0)))
            else:
                rx = math.degrees(math.atan2(-matrix.GetElement(1, 2),
                                             matrix.GetElement(1, 1)))
                ry = math.degrees(math.atan2(-matrix.GetElement(2, 0), sy))
                rz = 0.0

            self._update_rotation_ui("X", rx)
            self._update_rotation_ui("Y", ry)
            self._update_rotation_ui("Z", rz)
        except Exception:
            pass

    def _update_rotation_ui(self, axis, value):
        value = int(round(value))
        setattr(self, f"rotation_{axis.lower()}", float(value))
        for widget in (getattr(self, f"spinBox_{axis}"),
                       getattr(self, f"slider_{axis}")):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)