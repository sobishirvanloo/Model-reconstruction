"""Shared building blocks for the three VTK tabs.

This module centralises code that used to be copy-pasted into every tab
module (import_mesh / point_processing / poisson_reconstruction):

* ``CustomInteractorStyle`` — mouse-wheel (dolly) zoom.
* ``VTKViewport``        — console + VTK render window + orientation axes.
* ``RotationMixin``      — slider/spinbox <-> actor rotation logic.
"""

import numpy as np
import vtk
from PyQt5.QtWidgets import QPlainTextEdit, QVBoxLayout
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from modules.geometry_utils import vtk_matrix_to_numpy


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


class RotationMixin:
    """Slider/spinbox <-> actor rotation, shared by every tab.

    The spinbox/slider values are *absolute* angles and map 1:1 onto
    ``vtkProp3D.SetOrientation`` / ``GetOrientation``, so the UI and the
    actor can never drift apart (incremental RotateX/Y/Z deltas do not
    commute and did).  The actor origin is placed at the centre of its
    bounds so it rotates in place - the same pivot the mouse uses.

    Expected instance attributes:

        spinBox_X/Y/Z, slider_X/Y/Z     Qt widgets
        actor                           the VTK actor being rotated
        renderer, vtk_widget, console
    """

    NO_ACTOR_MESSAGE = "❌ Please import a file first"

    # ---------- public API ----------

    def init_rotation_state(self):
        self.actor = None
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0

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

        # Update spinboxes/sliders when the actor is rotated with the mouse
        self.interactor.AddObserver(
            "EndInteractionEvent", self.update_spinbox_from_actor
        )

    def rotation_widgets(self):
        return (
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton,
        )

    def sync_rotation(self, widget_to_update, axis, value):
        setattr(self, f"rotation_{axis.lower()}", float(value))
        widget_to_update.blockSignals(True)
        widget_to_update.setValue(int(value))
        widget_to_update.blockSignals(False)

    def reset_rotation(self):
        """Zero the angles and the widgets (e.g. after a new import)."""
        for axis in "XYZ":
            self._update_rotation_ui(axis, 0.0)

    def apply_rotation(self):
        actor = self.actor
        if actor is None:
            self.console.appendPlainText(self.NO_ACTOR_MESSAGE)
            return
        try:
            actor.SetOrientation(
                self.rotation_x, self.rotation_y, self.rotation_z
            )
            self.vtk_widget.GetRenderWindow().Render()

            self.console.appendPlainText(
                f"✅ Rotation applied | X: {self.rotation_x:.1f}° | "
                f"Y: {self.rotation_y:.1f}° | Z: {self.rotation_z:.1f}°"
            )
        except Exception as e:
            self.console.appendPlainText(f"❌ Error: {str(e)}")

    def show_actor(self, actor, reset_camera=True):
        """Replace the displayed actor, keeping the current rotation."""
        if self.actor is not None:
            self.renderer.RemoveActor(self.actor)

        bounds = actor.GetBounds()
        actor.SetOrigin(
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        actor.SetOrientation(self.rotation_x, self.rotation_y, self.rotation_z)

        self.actor = actor
        self.renderer.AddActor(actor)
        if reset_camera:
            self.renderer.ResetCamera()
        else:
            self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
        return actor

    def actor_matrix(self):
        """4x4 numpy transform of the displayed actor (rotation, mouse pan
        and scale) - applied on export so the file matches the view."""
        if self.actor is None:
            return np.eye(4)
        return vtk_matrix_to_numpy(self.actor.GetMatrix())

    # ---------- internals ----------

    def update_spinbox_from_actor(self, caller=None, event=None):
        """Sync the UI after the user rotates the actor with the mouse."""
        if self.actor is None:
            return
        for axis, value in zip("XYZ", self.actor.GetOrientation()):
            self._update_rotation_ui(axis, value)

    def _update_rotation_ui(self, axis, value):
        setattr(self, f"rotation_{axis.lower()}", float(value))
        for widget in (getattr(self, f"spinBox_{axis}"),
                       getattr(self, f"slider_{axis}")):
            widget.blockSignals(True)
            widget.setValue(int(round(value)))
            widget.blockSignals(False)
