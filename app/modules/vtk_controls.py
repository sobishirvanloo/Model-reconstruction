"""Shared building blocks for the three VTK tabs.

This module centralises code that used to be copy-pasted into every tab
module (import_mesh / point_processing / poisson_reconstruction):

* ``CustomInteractorStyle`` — mouse-wheel (dolly) zoom.
* ``VTKViewport``        — console + camera toolbar + VTK render window
                           (themed gradient, empty-state hint, axes).
* ``RotationMixin``      — slider/spinbox <-> actor rotation logic (live).
* ``TabMixin``           — rotation + messages / status-bar helpers.
"""

import numpy as np
import vtk
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QStyle,
    QToolBar,
    QVBoxLayout,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from modules import settings
from modules.geometry_utils import vtk_matrix_to_numpy
from modules.point_data import move_widget_after
from modules.theme import LIGHT, view_colors


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


# name -> (direction from the focal point to the camera, view-up)
CAMERA_VIEWS = {
    "Front": ((0, 0, 1), (0, 1, 0)),
    "Top": ((0, 1, 0), (0, 0, -1)),
    "Side": ((1, 0, 0), (0, 1, 0)),
    "Iso": ((1, 1, 1), (0, 1, 0)),
}


class VTKViewport:
    """Creates the console, camera toolbar and VTK render window of a tab.

    The widgets/objects are stored directly on ``owner`` using the names
    the rest of the application expects:

        console, vtk_widget, renderer, interactor, marker, viewport
    """

    def __init__(self, owner, frame_console, frame_view, empty_hint=""):
        self.owner = owner
        owner.viewport = self
        owner.frame_console = frame_console
        owner.frame_view = frame_view
        self.theme = LIGHT

        owner.console = QPlainTextEdit(frame_console)
        owner.console.setObjectName("console")
        owner.console.setReadOnly(True)
        console_layout = QVBoxLayout(frame_console)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.addWidget(owner.console)

        layout = QVBoxLayout(frame_view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = self._build_toolbar(frame_view)
        layout.addWidget(self.toolbar)
        owner.vtk_widget = QVTKRenderWindowInteractor(frame_view)
        layout.addWidget(owner.vtk_widget)

        owner.renderer = vtk.vtkRenderer()
        owner.renderer.GradientBackgroundOn()
        owner.vtk_widget.GetRenderWindow().AddRenderer(owner.renderer)
        owner.vtk_widget.GetRenderWindow().SetMultiSamples(0)
        owner.interactor = owner.vtk_widget.GetRenderWindow().GetInteractor()

        # ---------- VTK Initialization ----------
        owner.interactor.Initialize()
        style = CustomInteractorStyle(owner.renderer)
        owner.interactor.SetInteractorStyle(style)

        self.setup_axes(owner)

        # Empty-state hint in the middle of the view
        self.hint = vtk.vtkTextActor()
        self.hint.SetInput(empty_hint)
        prop = self.hint.GetTextProperty()
        prop.SetFontSize(17)
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToCentered()
        prop.SetLineSpacing(1.5)
        self.hint.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.hint.SetPosition(0.5, 0.5)
        owner.renderer.AddActor2D(self.hint)

        self.apply_theme(LIGHT)

    # =========================
    # Toolbar / camera
    # =========================
    def _build_toolbar(self, parent):
        toolbar = QToolBar(parent)
        style = parent.style()
        reset = toolbar.addAction(
            style.standardIcon(QStyle.SP_BrowserReload), "Reset view")
        reset.setToolTip("Fit the model in the view (Ctrl+0)")
        reset.triggered.connect(self.reset_camera)
        toolbar.addSeparator()
        for name, (direction, up) in CAMERA_VIEWS.items():
            action = toolbar.addAction(name)
            action.setToolTip(f"Look at the model from the {name.lower()}")
            action.triggered.connect(
                lambda _=False, d=direction, u=up: self.set_view(d, u))
        toolbar.addSeparator()
        shot = toolbar.addAction(
            style.standardIcon(QStyle.SP_DialogSaveButton), "Screenshot")
        shot.setToolTip("Save the view as a PNG image")
        shot.triggered.connect(self.save_screenshot)
        return toolbar

    def render(self):
        self.owner.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self):
        self.owner.renderer.ResetCamera()
        self.render()

    def set_view(self, direction, up):
        renderer = self.owner.renderer
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        focal = np.array(camera.GetFocalPoint())
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        camera.SetPosition(*(focal + direction * camera.GetDistance()))
        camera.SetViewUp(*up)
        renderer.ResetCamera()
        self.render()

    def save_screenshot(self, path=None):
        """Save the current view as PNG (asks for a path when not given)."""
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self.owner.main_window, "Save Screenshot",
                f"{settings.last_dir()}/screenshot.png", "PNG image (*.png)")
            if not path:
                return None
        if not path.lower().endswith(".png"):
            path += ".png"
        self._render_to_png(path)
        settings.remember_path(path)
        self.owner.notify(f"Screenshot saved: {path}", "success")
        return path

    def _render_to_png(self, path):
        """Render a copy of the scene in a separate off-screen window.

        Reading pixels back from the on-screen Qt window with
        vtkWindowToImageFilter crashed the next render of another tab's
        window (OpenGL context switch), so the live window is never read.
        """
        source = self.owner.renderer
        width, height = self.owner.vtk_widget.GetRenderWindow().GetSize()

        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.SetMultiSamples(0)
        window.SetSize(max(width, 400), max(height, 300))
        renderer = vtk.vtkRenderer()
        window.AddRenderer(renderer)
        renderer.SetGradientBackground(source.GetGradientBackground())
        renderer.SetBackground(source.GetBackground())
        renderer.SetBackground2(source.GetBackground2())

        props = source.GetViewProps()
        props.InitTraversal()
        for _ in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if not prop.GetVisibility() or prop is self.hint:
                continue
            if isinstance(prop, vtk.vtkActor):
                renderer.AddActor(self._copy_actor(prop))
            elif isinstance(prop, vtk.vtkScalarBarActor):
                bar = vtk.vtkScalarBarActor()
                bar.ShallowCopy(prop)
                renderer.AddActor2D(bar)

        camera = vtk.vtkCamera()
        camera.DeepCopy(source.GetActiveCamera())
        renderer.SetActiveCamera(camera)
        renderer.ResetCameraClippingRange()
        window.Render()

        grabber = vtk.vtkWindowToImageFilter()
        grabber.SetInput(window)
        grabber.SetInputBufferTypeToRGB()
        grabber.ReadFrontBufferOff()
        grabber.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(grabber.GetOutput())
        writer.Write()
        window.Finalize()

    @staticmethod
    def _copy_actor(actor):
        """New actor + mapper (own GPU resources) showing the same data."""
        source_mapper = actor.GetMapper()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(source_mapper.GetInput())
        mapper.SetScalarVisibility(source_mapper.GetScalarVisibility())
        mapper.SetScalarMode(source_mapper.GetScalarMode())
        mapper.SetLookupTable(source_mapper.GetLookupTable())
        mapper.SetScalarRange(source_mapper.GetScalarRange())
        copy = vtk.vtkActor()
        copy.SetMapper(mapper)
        copy.GetProperty().DeepCopy(actor.GetProperty())
        copy.SetOrigin(actor.GetOrigin())
        copy.SetPosition(actor.GetPosition())
        copy.SetOrientation(actor.GetOrientation())
        copy.SetScale(actor.GetScale())
        return copy

    # =========================
    # Theme / empty state
    # =========================
    def set_empty(self, empty):
        self.hint.SetVisibility(bool(empty))

    def color(self, kind):
        """Theme colour for an actor kind ("mesh" / "points")."""
        return view_colors(self.theme)[kind]

    def apply_theme(self, name):
        self.theme = name
        colors = view_colors(name)
        renderer = self.owner.renderer
        renderer.SetBackground(*colors["background_bottom"])
        renderer.SetBackground2(*colors["background_top"])
        self.hint.GetTextProperty().SetColor(*colors["text"])
        axes = self.owner.marker.GetOrientationMarker()
        for caption in (axes.GetXAxisCaptionActor2D(),
                        axes.GetYAxisCaptionActor2D(),
                        axes.GetZAxisCaptionActor2D()):
            caption.GetCaptionTextProperty().SetColor(*colors["text"])
            caption.GetCaptionTextProperty().ShadowOff()
        actor = getattr(self.owner, "actor", None)
        kind = getattr(self.owner, "actor_kind", None)
        if actor is not None and kind:
            actor.GetProperty().SetColor(*colors[kind])
        self.render()

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
    Moving a slider rotates the model immediately.

    Expected instance attributes:

        spinBox_X/Y/Z, slider_X/Y/Z     Qt widgets
        pushButton                      "Reset rotation" button
        actor                           the VTK actor being rotated
        renderer, vtk_widget, console
    """

    NO_ACTOR_MESSAGE = "❌ Please import a file first"

    # ---------- public API ----------

    def init_rotation_state(self):
        self.actor = None
        self.actor_kind = None
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

        self.pushButton.setText("Reset rotation")
        self.pushButton.setIcon(
            self.pushButton.style().standardIcon(QStyle.SP_ArrowBack))
        self.pushButton.setToolTip("Set all rotation angles back to 0°")
        self.pushButton.clicked.connect(self.reset_rotation_view)
        # Sliders rotate live, so the button belongs right under them
        move_widget_after(self.pushButton, self.slider_Z)

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
        self._apply_orientation()        # live rotation

    def reset_rotation(self):
        """Zero the angles and the widgets (e.g. after a new import)."""
        for axis in "XYZ":
            self._update_rotation_ui(axis, 0.0)

    def reset_rotation_view(self):
        self.reset_rotation()
        if self._apply_orientation():
            self.console.appendPlainText("↺ Rotation reset to 0°")

    def apply_rotation(self):
        if self.actor is None:
            self.console.appendPlainText(self.NO_ACTOR_MESSAGE)
            return
        self._apply_orientation()
        self.console.appendPlainText(
            f"✅ Rotation applied | X: {self.rotation_x:.1f}° | "
            f"Y: {self.rotation_y:.1f}° | Z: {self.rotation_z:.1f}°"
        )

    def _apply_orientation(self):
        if self.actor is None:
            return False
        self.actor.SetOrientation(
            self.rotation_x, self.rotation_y, self.rotation_z
        )
        self.vtk_widget.GetRenderWindow().Render()
        return True

    def show_actor(self, actor, reset_camera=True, kind=None):
        """Replace the displayed actor, keeping the current rotation.

        ``kind`` ("mesh" / "points") gives the actor the theme colour.
        """
        if self.actor is not None:
            self.renderer.RemoveActor(self.actor)

        if kind:
            actor.GetProperty().SetColor(*self.viewport.color(kind))
        self.actor_kind = kind

        bounds = actor.GetBounds()
        actor.SetOrigin(
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        actor.SetOrientation(self.rotation_x, self.rotation_y, self.rotation_z)

        self.actor = actor
        self.renderer.AddActor(actor)
        self.viewport.set_empty(False)
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


class TabMixin(RotationMixin):
    """Rotation plus the helpers every tab uses to talk to the user."""

    def notify(self, text, level="info", dialog=False):
        """Log ``text`` to the console and the status bar; serious errors
        (``dialog=True``) also open a message box."""
        self.console.appendPlainText(text)
        show = getattr(self.main_window, "show_message", None)
        if show:
            show(text, level)
        if dialog:
            box = QMessageBox.warning if level == "error" else QMessageBox.information
            box(self.main_window, "Model Reconstruction", text.lstrip("❌✅⚠ "))

    def set_tab_info(self, **info):
        """Update what the status bar shows for this tab (file, points,
        triangles); ``None`` clears a field."""
        update = getattr(self.main_window, "update_tab_info", None)
        if update:
            update(self, **info)

    def decorate(self, button, icon, tooltip="", variant=None):
        """Standard icon / tooltip / style variant for a panel button."""
        button.setIcon(button.style().standardIcon(icon))
        if tooltip:
            button.setToolTip(tooltip)
        if variant:
            button.setProperty("variant", variant)
        return button
