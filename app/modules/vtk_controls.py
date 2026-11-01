"""Shared building blocks for the three VTK tabs.

This module centralises code that used to be copy-pasted into every tab
module (import_mesh / point_processing / poisson_reconstruction):

* ``CustomInteractorStyle`` — mouse: rotate / move the model anywhere in
                           the view, wheel (dolly) zoom.
* ``MouseGuide``         — small closable strip explaining the mouse.
* ``VTKViewport``        — console + camera toolbar + VTK render window
                           (themed gradient, empty-state hint, axes).
* ``RotationMixin``      — slider/spinbox <-> actor rotation and position
                           (live).
* ``TabMixin``           — rotation + messages / status-bar helpers.
"""

import ctypes
import gc
import sys

import math

import numpy as np
import vtk
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QStyle,
    QToolBar,
    QToolButton,
    QVBoxLayout,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from modules import settings
from modules.geometry_utils import vtk_matrix_to_numpy
from modules.point_data import add_section, insert_widget, move_widget_after
from modules import view_settings
from modules.theme import LIGHT, PALETTES, view_colors
from modules.view_settings import (
    BACKGROUND,
    OBJECT,
    BackgroundSettingsDialog,
    ObjectSettingsDialog,
    SceneOverlays,
    rgb,
)


def orientation_matrix(orientation):
    """vtkMatrix4x4 of a pure rotation given as vtkProp3D angles."""
    pose = vtk.vtkActor()
    pose.SetOrientation(*orientation)
    return pose.GetMatrix()


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballActor):
    """Mouse controls of every view (the model does not have to be under
    the cursor):

    * left drag   - rotate the model about its centre
    * right drag  - move the model with the mouse (screen plane);
      the middle button does the same
    * wheel       - zoom (camera dolly)

    ``owner.actor`` is changed and ``owner.on_actor_moved()`` keeps the
    side panel in sync.  An observer on a button / move event replaces
    VTK's own handler, which would only act on the prop under the cursor.
    """

    ZOOM_IN_FACTOR = 1.1
    ZOOM_OUT_FACTOR = 0.9
    DEGREES_PER_PIXEL = 0.4

    def __init__(self, renderer, owner=None):
        self.renderer = renderer
        self.owner = owner
        self._mode = None             # "rotate" / "move" while dragging
        self._last = None
        for event, handler in (
            ("MouseWheelForwardEvent", self.zoom_in),
            ("MouseWheelBackwardEvent", self.zoom_out),
            ("LeftButtonPressEvent", lambda *_: self._begin("rotate")),
            ("RightButtonPressEvent", lambda *_: self._begin("move")),
            ("MiddleButtonPressEvent", lambda *_: self._begin("move")),
            ("LeftButtonReleaseEvent", self._end),
            ("RightButtonReleaseEvent", self._end),
            ("MiddleButtonReleaseEvent", self._end),
            ("MouseMoveEvent", self._drag),
        ):
            self.AddObserver(event, handler)

    def _actor(self):
        return getattr(self.owner, "actor", None)

    def _begin(self, mode):
        if self._actor() is None:
            return
        self._mode = mode
        self._last = self.GetInteractor().GetEventPosition()

    def _end(self, *_):
        self._mode = None

    def _drag(self, *_):
        actor = self._actor()
        if self._mode is None or actor is None:
            return
        x, y = self.GetInteractor().GetEventPosition()
        last_x, last_y = self._last
        self._last = (x, y)
        if self._mode == "rotate":
            self._rotate(actor, x - last_x, y - last_y)
        else:
            self._move(actor, (last_x, last_y), (x, y))
        self.renderer.ResetCameraClippingRange()
        if self.owner is not None:
            self.owner.on_actor_moved()
        self.GetInteractor().GetRenderWindow().Render()

    def _rotate(self, actor, dx, dy):
        """Turn about the screen axes: the side under the mouse follows it."""
        camera = self.renderer.GetActiveCamera()
        up = np.array(camera.GetViewUp())
        right = np.cross(camera.GetDirectionOfProjection(), up)
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.SetMatrix(orientation_matrix(actor.GetOrientation()))
        transform.RotateWXYZ(dx * self.DEGREES_PER_PIXEL, *up)
        transform.RotateWXYZ(-dy * self.DEGREES_PER_PIXEL, *right)
        actor.SetOrientation(transform.GetOrientation())

    def _move(self, actor, start, end):
        """Shift by the mouse motion, measured at the model's depth."""
        renderer = self.renderer
        renderer.SetWorldPoint(*actor.GetCenter(), 1.0)
        renderer.WorldToDisplay()
        depth = renderer.GetDisplayPoint()[2]

        def to_world(point):
            renderer.SetDisplayPoint(point[0], point[1], depth)
            renderer.DisplayToWorld()
            world = np.array(renderer.GetWorldPoint())
            return world[:3] / world[3]

        actor.AddPosition(*(to_world(end) - to_world(start)))

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


def mouse_icon(button, color, accent, size=20):
    """Small drawn mouse with ``button`` ("left" / "right" / "wheel")
    highlighted."""
    scale = 2                       # drawn at 2x, crisp on HiDPI
    pixmap = QPixmap(int(size * 0.75) * scale, size * scale)
    pixmap.fill(Qt.transparent)
    pixmap.setDevicePixelRatio(scale)
    w, h = size * 0.75, size
    body = QRectF(1, 1, w - 2, h - 2)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(body, w * 0.45, w * 0.45)
    split = body.top() + body.height() * 0.42
    if button in ("left", "right"):
        half = QPainterPath()
        x = body.left() if button == "left" else body.center().x()
        half.addRect(QRectF(x, body.top(), body.width() / 2, split - body.top()))
        painter.fillPath(path.intersected(half), QColor(accent))
    pen = QPen(QColor(color), 1.2)
    painter.setPen(pen)
    painter.drawPath(path)
    painter.drawLine(int(body.left()), int(split), int(body.right()), int(split))
    painter.drawLine(int(body.center().x()), int(body.top()),
                     int(body.center().x()), int(split))
    wheel = QRectF(body.center().x() - 1.6, body.top() + 3, 3.2, 5)
    if button == "wheel":
        painter.setBrush(QColor(accent))
        painter.setPen(QPen(QColor(accent), 1))
    else:
        painter.setBrush(QColor(color))
    painter.drawRoundedRect(wheel, 1.5, 1.5)
    painter.end()
    return pixmap


class MouseGuide(QFrame):
    """Tiny strip under a view showing the mouse controls; the ✕ hides it
    until the application is started again (or *View ▸ Mouse Guide*)."""

    ITEMS = (
        ("left", "Drag: rotate"),
        ("right", "Drag: move"),
        ("wheel", "Scroll: zoom"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mouseGuide")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 4, 2)
        layout.setSpacing(4)
        self.icons = []
        for button, text in self.ITEMS:
            icon = QLabel(self)
            label = QLabel(text, self)
            label.setProperty("role", "mouseguide")
            self.icons.append((icon, button))
            layout.addWidget(icon)
            layout.addWidget(label)
            layout.addSpacing(12)
        layout.addStretch(1)
        self.close_button = QToolButton(self)
        self.close_button.setText("✕")
        self.close_button.setAutoRaise(True)
        self.close_button.setToolTip("Hide the mouse guide "
                                     "(View > Mouse Guide shows it again)")
        self.close_button.clicked.connect(self.hide)
        layout.addWidget(self.close_button)
        self.setToolTip("Left drag anywhere: rotate · right drag: move the "
                        "model · wheel: zoom")

    def apply_theme(self, name):
        palette = PALETTES[name]
        for icon, button in self.icons:
            icon.setPixmap(mouse_icon(button, palette["text_muted"],
                                      palette["primary"]))


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
        self.mouse_guide = MouseGuide(frame_view)
        layout.addWidget(self.mouse_guide)

        owner.renderer = vtk.vtkRenderer()
        owner.renderer.GradientBackgroundOn()
        owner.vtk_widget.GetRenderWindow().AddRenderer(owner.renderer)
        owner.vtk_widget.GetRenderWindow().SetMultiSamples(0)
        owner.interactor = owner.vtk_widget.GetRenderWindow().GetInteractor()

        # ---------- VTK Initialization ----------
        owner.interactor.Initialize()
        # Keep a Python reference: otherwise the wrapper is collected and
        # VTK hands back a bare one without the attributes set in __init__
        owner.interactor_style = CustomInteractorStyle(owner.renderer, owner)
        owner.interactor.SetInteractorStyle(owner.interactor_style)

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
        owner.renderer.AddViewProp(self.hint)

        self.object_settings = view_settings.load(OBJECT)
        self.background_settings = view_settings.load(BACKGROUND)
        self.overlays = SceneOverlays(owner.renderer)

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

        view_menu = self._dropdown(toolbar, "View", "Look at the model from "
                                   "the front, top, side or at an angle")
        for number, (name, (direction, up)) in enumerate(CAMERA_VIEWS.items(), 1):
            action = view_menu.addAction(f"{name}\tCtrl+{number}")
            action.triggered.connect(
                lambda _=False, d=direction, u=up: self.set_view(d, u))

        object_menu = self._dropdown(toolbar, "Object",
                                     "Colour and display style of the model")
        object_menu.addAction("Colour && display…").triggered.connect(
            lambda: self.open_settings(OBJECT))
        object_menu.addSeparator()
        object_menu.addAction("Restore default").triggered.connect(
            lambda: self.restore_default(OBJECT))

        background_menu = self._dropdown(
            toolbar, "Background",
            "Background colour, grid, ruler and start-pose guides")
        background_menu.addAction("Colour && guides…").triggered.connect(
            lambda: self.open_settings(BACKGROUND))
        background_menu.addSeparator()
        background_menu.addAction("Restore default").triggered.connect(
            lambda: self.restore_default(BACKGROUND))

        toolbar.addSeparator()
        shot = toolbar.addAction(
            style.standardIcon(QStyle.SP_DialogSaveButton), "Screenshot")
        shot.setToolTip("Save the view as a PNG image")
        shot.triggered.connect(self.save_screenshot)
        return toolbar

    @staticmethod
    def _dropdown(toolbar, text, tooltip):
        """Toolbar button that opens a menu (returned)."""
        button = QToolButton(toolbar)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        menu = QMenu(button)
        button.setMenu(menu)
        toolbar.addWidget(button)
        return menu

    def render(self):
        self.owner.vtk_widget.GetRenderWindow().Render()

    def fit_camera(self):
        """Frame the model only (the guides, e.g. the large floor grid,
        do not count)."""
        renderer = self.owner.renderer
        actor = getattr(self.owner, "actor", None)
        if actor is not None:
            renderer.ResetCamera(actor.GetBounds())
        else:
            renderer.ResetCamera()
        renderer.ResetCameraClippingRange()

    def reset_camera(self):
        self.fit_camera()
        self.render()

    def set_view(self, direction, up):
        renderer = self.owner.renderer
        self.fit_camera()
        camera = renderer.GetActiveCamera()
        focal = np.array(camera.GetFocalPoint())
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        camera.SetPosition(*(focal + direction * camera.GetDistance()))
        camera.SetViewUp(*up)
        self.fit_camera()
        self.render()

    # =========================
    # Object / background settings
    # =========================
    def _dialog(self, kind):
        """The settings window of ``kind``, shared by all tabs."""
        main = self.owner.main_window
        dialogs = main.__dict__.setdefault("_view_dialogs", {})
        if kind not in dialogs:
            cls = ObjectSettingsDialog if kind == OBJECT else BackgroundSettingsDialog
            dialogs[kind] = cls(main)
            dialogs[kind].saved.connect(self._broadcast)
        return dialogs[kind]

    def open_settings(self, kind):
        current = self.object_settings if kind == OBJECT else self.background_settings
        self._dialog(kind).open_with(dict(current))

    def restore_default(self, kind):
        values = dict(view_settings.DEFAULTS[kind])
        view_settings.save(kind, values)
        self._broadcast(kind, values)

    def _broadcast(self, kind, values):
        """New settings apply to every tab (main window), else to this one."""
        apply_all = getattr(self.owner.main_window, "apply_view_settings", None)
        if apply_all:
            apply_all(kind, values)
        else:
            self.apply_view_settings(kind, values)

    def apply_view_settings(self, kind, values):
        if kind == OBJECT:
            self.object_settings = dict(values)
            actor = getattr(self.owner, "actor", None)
            if actor is not None:
                self.style_actor(actor, self.owner.actor_kind)
        else:
            self.background_settings = dict(values)
            self.apply_background()
            self.update_overlays()
        self.render()

    def style_actor(self, actor, kind):
        """Colour / style of the model from the object settings.

        *Original* keeps the file's colours (a ``Colors`` array) or uses
        the theme colour; an error heatmap is never overridden.
        """
        options = self.object_settings
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        scalars = (polydata.GetPointData().GetScalars()
                   if polydata is not None else None)
        name = scalars.GetName() if scalars is not None else None
        prop = actor.GetProperty()
        if name == "Error":
            pass                                  # heatmap stays
        elif options["color_mode"] == "original" and name == "Colors":
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
        else:
            if mapper is not None:
                mapper.ScalarVisibilityOff()
            if options["color_mode"] == "custom":
                prop.SetColor(*rgb(options["color"]))
            elif kind:
                prop.SetColor(*self.color(kind))

        style = options["style"]
        prop.SetEdgeVisibility(style == "surface_edges")
        prop.SetEdgeColor(*rgb(options["edge_color"]))
        if style == "wireframe":
            prop.SetRepresentationToWireframe()
        elif style == "points":
            prop.SetRepresentationToPoints()
        else:
            prop.SetRepresentationToSurface()
        prop.SetOpacity(options["opacity"] / 100.0)
        prop.SetPointSize(options["point_size"])

    def apply_background(self):
        options = self.background_settings
        renderer = self.owner.renderer
        mode = options["mode"]
        if mode == "solid":
            renderer.GradientBackgroundOff()
            renderer.SetBackground(*rgb(options["color"]))
        elif mode == "gradient":
            renderer.GradientBackgroundOn()
            renderer.SetBackground(*rgb(options["color_bottom"]))
            renderer.SetBackground2(*rgb(options["color_top"]))
        else:
            colors = view_colors(self.theme)
            renderer.GradientBackgroundOn()
            renderer.SetBackground(*colors["background_bottom"])
            renderer.SetBackground2(*colors["background_top"])

    def guide_color(self):
        """Readable colour for guides on the current background."""
        if self.background_settings["mode"] == "theme":
            return view_colors(self.theme)["text"]
        options = self.background_settings
        background = (options["color"] if options["mode"] == "solid"
                      else options["color_bottom"])
        dark = QColor(background).lightness() < 128
        return (0.85, 0.88, 0.92) if dark else (0.22, 0.26, 0.32)

    def update_overlays(self, overlays=None):
        owner = self.owner
        rotation = (getattr(owner, "rotation_x", 0.0),
                    getattr(owner, "rotation_y", 0.0),
                    getattr(owner, "rotation_z", 0.0))
        (overlays or self.overlays).update(
            self.background_settings, getattr(owner, "start_bounds", None),
            getattr(owner, "actor", None), rotation,
            getattr(owner, "position", (0.0, 0.0, 0.0)), self.guide_color())

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
            if (not prop.GetVisibility() or prop is self.hint
                    or self.overlays.owns(prop)):
                continue
            if isinstance(prop, vtk.vtkActor):
                renderer.AddActor(self._copy_actor(prop))
            elif isinstance(prop, vtk.vtkScalarBarActor):
                bar = vtk.vtkScalarBarActor()
                bar.ShallowCopy(prop)
                renderer.AddViewProp(bar)

        camera = vtk.vtkCamera()
        camera.DeepCopy(source.GetActiveCamera())
        renderer.SetActiveCamera(camera)
        # Guides are rebuilt for the copy (cube axes / text need their own)
        self.update_overlays(SceneOverlays(renderer))
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
        mapper.SetColorMode(source_mapper.GetColorMode())
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
        self.mouse_guide.apply_theme(name)
        colors = view_colors(name)
        self.apply_background()
        self.hint.GetTextProperty().SetColor(*colors["text"])
        axes = self.owner.marker.GetOrientationMarker()
        for caption in (axes.GetXAxisCaptionActor2D(),
                        axes.GetYAxisCaptionActor2D(),
                        axes.GetZAxisCaptionActor2D()):
            caption.GetCaptionTextProperty().SetColor(*colors["text"])
            caption.GetCaptionTextProperty().ShadowOff()
        actor = getattr(self.owner, "actor", None)
        if actor is not None:
            self.style_actor(actor, getattr(self.owner, "actor_kind", None))
        self.update_overlays()
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


POSITION_SLIDER_STEPS = 1000


class RotationMixin:
    """Slider/spinbox <-> actor rotation and position, shared by every tab.

    The spinbox/slider values are *absolute* angles and map 1:1 onto
    ``vtkProp3D.SetOrientation`` / ``GetOrientation``, so the UI and the
    actor can never drift apart (incremental RotateX/Y/Z deltas do not
    commute and did).  The actor origin is placed at the centre of its
    bounds so it rotates in place - the same pivot the mouse uses.
    Moving a slider rotates the model immediately.

    The position X/Y/Z (``vtkProp3D.SetPosition``, an offset in model
    units) works the same way; its range follows the model size.

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
        self.position = [0.0, 0.0, 0.0]
        self.position_range = 1.0
        self.start_bounds = None      # model box at rotation 0 / position 0

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

        self.setup_position_controls()

    def setup_position_controls(self):
        """X/Y/Z position spinbox + slider pairs (built like the rotation
        ones) and a *Reset position* button, below *Reset rotation*."""
        parent = self.pushButton.parentWidget()
        anchor = self.pushButton
        self.position_boxes = {}
        self.position_sliders = {}
        labels = []
        for axis in "XYZ":
            label = QLabel(f"{axis}-Axis (position)", parent)
            box = QDoubleSpinBox(parent)
            slider = QSlider(Qt.Horizontal, parent)
            slider.setRange(-POSITION_SLIDER_STEPS, POSITION_SLIDER_STEPS)
            tip = (f"Move the model along {axis} (model units; "
                   "you can also drag it with the right mouse button)")
            box.setToolTip(tip)
            slider.setToolTip(tip)
            box.valueChanged.connect(
                lambda v, a=axis: self.set_position(a, v))
            slider.valueChanged.connect(
                lambda v, a=axis: self.set_position(
                    a, v * self.position_range / POSITION_SLIDER_STEPS))
            for widget in (label, box, slider):
                insert_widget(anchor, widget, after=True)
                anchor = widget
            labels.append(label)
            self.position_boxes[axis] = box
            self.position_sliders[axis] = slider

        self.btn_reset_position = QPushButton("Reset position", parent)
        self.btn_reset_position.setIcon(
            parent.style().standardIcon(QStyle.SP_ArrowBack))
        self.btn_reset_position.setToolTip("Move the model back to 0, 0, 0")
        self.btn_reset_position.clicked.connect(self.reset_position_view)
        insert_widget(anchor, self.btn_reset_position, after=True)
        add_section(labels[0], "Position")
        self._set_position_range(1.0)

    def rotation_widgets(self):
        """Every rotation and position control (enabled with the data)."""
        return (
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton,
            *self.position_boxes.values(), *self.position_sliders.values(),
            self.btn_reset_position,
        )

    def sync_rotation(self, widget_to_update, axis, value):
        setattr(self, f"rotation_{axis.lower()}", float(value))
        widget_to_update.blockSignals(True)
        widget_to_update.setValue(int(value))
        widget_to_update.blockSignals(False)
        self._apply_orientation()        # live rotation

    def reset_rotation(self):
        """Zero the angles and the widgets."""
        for axis in "XYZ":
            self._update_rotation_ui(axis, 0.0)

    def reset_transform(self):
        """Zero rotation and position (e.g. after a new import)."""
        self.reset_rotation()
        self.reset_position()

    def reset_position(self):
        self.position = [0.0, 0.0, 0.0]
        for axis in "XYZ":
            self._show_position(axis, 0.0)

    def reset_position_view(self):
        self.reset_position()
        if self.actor is not None:
            self.actor.SetPosition(*self.position)
            self.renderer.ResetCameraClippingRange()
            self._pose_changed()
            self.vtk_widget.GetRenderWindow().Render()
            self.console.appendPlainText("↺ Position reset to 0, 0, 0")

    def set_position(self, axis, value):
        """Spinbox / slider changed: move the model live."""
        self.position["XYZ".index(axis)] = float(value)
        self._show_position(axis, value)
        if self.actor is not None:
            self.actor.SetPosition(*self.position)
            self.renderer.ResetCameraClippingRange()
            self._pose_changed()
            self.vtk_widget.GetRenderWindow().Render()

    def on_actor_moved(self):
        """The mouse rotated / moved the model: update the panel."""
        if self.actor is None:
            return
        for axis, value in zip("XYZ", self.actor.GetOrientation()):
            self._update_rotation_ui(axis, value)
        self.position = [float(v) for v in self.actor.GetPosition()]
        for axis, value in zip("XYZ", self.position):
            self._show_position(axis, value)
        self._pose_changed()

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

    def _pose_changed(self):
        """Rotation / position changed: corner axes and guides follow."""
        self._sync_axes()
        self.viewport.update_overlays()

    def _sync_axes(self):
        """Give the corner axes the model's rotation, so they show the
        model's own X/Y/Z (the camera views still turn them too)."""
        orientation = (
            self.actor.GetOrientation() if self.actor is not None
            else (0.0, 0.0, 0.0)
        )
        # vtkAxesActor draws its parts with its *user transform* only
        # (SetOrientation is ignored), so pass the rotation that way.
        transform = vtk.vtkTransform()
        transform.SetMatrix(orientation_matrix(orientation))
        self.marker.GetOrientationMarker().SetUserTransform(transform)

    def _apply_orientation(self):
        if self.actor is None:
            return False
        self.actor.SetOrientation(
            self.rotation_x, self.rotation_y, self.rotation_z
        )
        self._pose_changed()
        self.vtk_widget.GetRenderWindow().Render()
        return True

    def show_actor(self, actor, reset_camera=True, kind=None):
        """Replace the displayed actor, keeping the current rotation and
        position.

        ``kind`` ("mesh" / "points") selects the theme colour used when
        the object settings do not ask for another one.
        """
        if self.actor is not None:
            self.renderer.RemoveActor(self.actor)

        self.actor_kind = kind
        self.viewport.style_actor(actor, kind)
        self.start_bounds = actor.GetMapper().GetBounds()

        bounds = actor.GetBounds()
        actor.SetOrigin(
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        actor.SetOrientation(self.rotation_x, self.rotation_y, self.rotation_z)
        actor.SetPosition(*self.position)
        diagonal = math.dist(bounds[0::2], bounds[1::2])
        self._set_position_range(
            max(diagonal, *(abs(v) for v in self.position), 1e-6))

        self.actor = actor
        self.renderer.AddActor(actor)
        self._pose_changed()
        self.viewport.set_empty(False)
        if reset_camera:
            self.viewport.fit_camera()
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

    def _set_position_range(self, extent):
        """Spinboxes / sliders cover +-``extent`` (the model diagonal)."""
        self.position_range = extent
        decimals = min(6, max(1, 3 - math.floor(math.log10(extent))))
        for axis, value in zip("XYZ", self.position):
            box = self.position_boxes[axis]
            box.blockSignals(True)
            box.setDecimals(decimals)
            box.setRange(-extent, extent)
            box.setSingleStep(extent / 100)
            box.blockSignals(False)
            self._show_position(axis, value)

    def _show_position(self, axis, value):
        """Show ``value`` in the spinbox and slider without feedback."""
        if abs(value) > self.position_range:     # dragged further away
            self._set_position_range(abs(value) * 1.5)
            return
        box = self.position_boxes[axis]
        slider = self.position_sliders[axis]
        for widget, shown in (
            (box, value),
            (slider, round(value / self.position_range * POSITION_SLIDER_STEPS)),
        ):
            widget.blockSignals(True)
            widget.setValue(shown)
            widget.blockSignals(False)

    def _update_rotation_ui(self, axis, value):
        setattr(self, f"rotation_{axis.lower()}", float(value))
        for widget in (getattr(self, f"spinBox_{axis}"),
                       getattr(self, f"slider_{axis}")):
            widget.blockSignals(True)
            widget.setValue(int(round(value)))
            widget.blockSignals(False)


def release_memory():
    """Collect garbage and hand freed heap pages back to the OS, so the
    RAM used by removed data shows up as free again."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


class TabMixin(RotationMixin):
    """Rotation plus the helpers every tab uses to talk to the user.

    Tabs that hold data implement ``has_data()`` and ``clear_data()``; the
    *Remove data* button (``add_remove_button``) asks, calls
    ``clear_data`` and frees the memory.
    """

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

    # =========================
    # Remove data (free memory)
    # =========================
    def add_remove_button(self, anchor):
        """*Remove data* button right below ``anchor`` (disabled while the
        tab is empty)."""
        self.btn_remove = QPushButton("Remove data", anchor.parentWidget())
        self.decorate(self.btn_remove, QStyle.SP_TrashIcon,
                      "Remove the data of this tab and free its memory "
                      "(Ctrl+W)", "danger")
        self.btn_remove.clicked.connect(lambda: self.remove_data())
        self.btn_remove.setEnabled(False)
        insert_widget(anchor, self.btn_remove, after=True)
        return self.btn_remove

    def remove_data(self, confirm=True):
        """Drop this tab's data (asks first) and free the memory."""
        if not self.has_data():
            return False
        if confirm:
            answer = QMessageBox.question(
                self.main_window, "Remove data",
                f"Remove {self.source_name or 'the data'} from this tab?\n\n"
                "Unsaved results are lost; the file on disk is not touched.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False
        name = self.source_name
        self.clear_data()
        release_memory()
        self.notify(f"🗑 Removed {name or 'data'} - memory freed", "success")
        return True

    def clear_view(self):
        """Remove the displayed actor and show the empty-state hint."""
        if self.actor is not None:
            self.renderer.RemoveActor(self.actor)
        self.actor = None
        self.actor_kind = None
        self.start_bounds = None
        self.reset_transform()
        self._pose_changed()
        for widget in self.rotation_widgets():
            widget.setEnabled(False)
        self.file_label.setText("No data loaded")
        self.file_label.setProperty("role", "hint")
        self.file_label.style().polish(self.file_label)
        self.btn_remove.setEnabled(False)
        self.set_tab_info(file=None, points=None, triangles=None)
        self.console.clear()
        self.viewport.set_empty(True)
        self.viewport.render()
