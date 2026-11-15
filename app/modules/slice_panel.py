# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""The *Slice* section of every tab's side panel.

Up to six planes, each with a position (X, Y, Z) and angles (X, Y, Z) in
the model's own coordinates, so they turn and move with the model.

Live preview, without touching the data:

* every enabled plane is drawn as a translucent square;
* the kept part is shown (the view is clipped by the planes) and the
  removed part stays visible as a faint ghost;
* the cut line of each plane is drawn bold;
* *Drag with the mouse* puts a handle on the selected plane
  (``vtkImplicitPlaneWidget2``): move it along its normal, turn it by its
  arrow; the numbers follow.

*Apply slice* replaces the tab's data with the kept part (the tab decides
what that means - ``slice_source`` / ``apply_slice``), a checkpoint makes
it undoable; *Save slice…* writes the kept part without applying it.

A tab provides:

    slice_source()            -> {"kind": "mesh"|"points", "vertices",
                                  "triangles", "normals", "colors"} or None
    apply_slice(planes, cap)  -> short description of what changed
    slice_actors()            -> actors the preview clips (optional)
"""

import numpy as np
import vtk
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from modules import icons, settings, slicing
from modules.slicing import MAX_PLANES, Plane

PLANE_COLORS = ("#38d6ff", "#ff9f43", "#a55eea", "#26de81", "#fc5c65", "#fed330")
SLIDER_STEPS = 1000


class SlicePanel(QWidget):
    def __init__(self, tab, parent, cap_default=True):
        super().__init__(parent)
        self.tab = tab
        self.planes = []
        self.index = -1
        self.low = np.zeros(3)
        self.high = np.ones(3)
        self._preview = {"quads": [], "sections": [], "ghost": None, "clipped": []}
        self._widget = None
        self._updating = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.refresh)
        self._build(cap_default)
        self._show_plane()

    # =========================
    # Widgets
    # =========================
    def _build(self, cap_default):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        row = QHBoxLayout()
        self.combo = QComboBox(self)
        self.combo.setToolTip("The plane being edited")
        self.combo.currentIndexChanged.connect(self._select)
        self.btn_add = QPushButton("+", self)
        self.btn_add.setToolTip(f"Add a plane (up to {MAX_PLANES})")
        self.btn_add.clicked.connect(self.add_plane)
        self.btn_delete = QPushButton("−", self)
        self.btn_delete.setToolTip("Remove the selected plane")
        self.btn_delete.clicked.connect(self.remove_plane)
        for button in (self.btn_add, self.btn_delete):
            button.setFixedWidth(34)
        row.addWidget(self.combo, 1)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_delete)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.check_enabled = QCheckBox("Enabled", self)
        self.check_enabled.setToolTip("Use this plane for the slice")
        self.check_enabled.toggled.connect(lambda on: self._edit(enabled=on))
        self.btn_flip = QPushButton("Flip side", self)
        self.btn_flip.setToolTip("Keep the other side of this plane")
        self.btn_flip.clicked.connect(self._flip)
        row.addWidget(self.check_enabled, 1)
        row.addWidget(self.btn_flip)
        layout.addLayout(row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        self.position_boxes, self.position_sliders = [], []
        self.angle_boxes, self.angle_sliders = [], []
        for row_index, axis in enumerate("XYZ"):
            name = QLabel(f"Position {axis}", self)
            box = QDoubleSpinBox(self)
            box.setDecimals(4)
            box.setToolTip(f"A point of the plane: {axis} (model units)")
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(0, SLIDER_STEPS)
            box.valueChanged.connect(lambda v, i=row_index: self._position(i, v))
            slider.valueChanged.connect(lambda v, i=row_index: self._position_slider(i, v))
            grid.addWidget(name, row_index, 0)
            grid.addWidget(box, row_index, 1)
            grid.addWidget(slider, row_index, 2)
            self.position_boxes.append(box)
            self.position_sliders.append(slider)
        for row_index, axis in enumerate("XYZ", start=3):
            name = QLabel(f"Angle {axis}", self)
            box = QSpinBox(self)
            box.setRange(-180, 180)
            box.setSuffix("°")
            box.setToolTip(f"Turn of the plane about {axis} (degrees, like the model rotation)")
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(-180, 180)
            i = row_index - 3
            box.valueChanged.connect(lambda v, i=i: self._angle(i, v))
            slider.valueChanged.connect(lambda v, i=i: self._angle(i, v))
            grid.addWidget(name, row_index, 0)
            grid.addWidget(box, row_index, 1)
            grid.addWidget(slider, row_index, 2)
            self.angle_boxes.append(box)
            self.angle_sliders.append(slider)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)

        presets = QHBoxLayout()
        presets.addWidget(QLabel("Preset", self))
        self.preset_buttons = {}
        for name, tip in (("X", "One plane across X (through the centre)"),
                          ("Y", "One plane across Y (through the centre)"),
                          ("Z", "One plane across Z (through the centre)"),
                          ("Box", "Six planes: keep a box around the centre"),
                          ("View", "One plane facing you, through the centre")):
            button = QPushButton(name, self)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, n=name: self.preset(n))
            presets.addWidget(button)
            self.preset_buttons[name] = button
        layout.addLayout(presets)

        self.check_preview = QCheckBox("Preview", self)
        self.check_preview.setChecked(True)
        self.check_preview.setToolTip("Show the planes, the kept part and the cut line")
        self.check_preview.toggled.connect(lambda _: self.schedule())
        self.check_mouse = QCheckBox("Drag the plane with the mouse", self)
        self.check_mouse.setToolTip("A handle on the selected plane: drag it to move, "
                                    "drag its arrow to turn it")
        self.check_mouse.toggled.connect(lambda _: self._update_widget())
        self.check_cap = QCheckBox("Close the cut (cap)", self)
        self.check_cap.setChecked(cap_default)
        self.check_cap.setToolTip("Fill the opening of a sliced mesh with the cross "
                                  "section, so a closed mesh stays closed")
        for check in (self.check_preview, self.check_mouse, self.check_cap):
            layout.addWidget(check)

        self.btn_apply = QPushButton("Apply slice", self)
        self.btn_apply.setProperty("variant", "primary")
        self.btn_apply.setIcon(icons.icon("check", "on_primary"))
        self.btn_apply.setToolTip("Keep only the part inside the planes "
                                  "(undo with the checkpoint arrow)")
        self.btn_apply.clicked.connect(self.apply)
        row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear planes", self)
        self.btn_clear.setIcon(icons.icon("discard"))
        self.btn_clear.clicked.connect(self.clear)
        self.btn_save = QPushButton("Save slice…", self)
        self.btn_save.setIcon(icons.icon("save"))
        self.btn_save.setToolTip("Save the kept part to a file without changing the data")
        self.btn_save.clicked.connect(lambda: self.save())
        row.addWidget(self.btn_clear)
        row.addWidget(self.btn_save)
        layout.addWidget(self.btn_apply)
        layout.addLayout(row)
        self.stats = QLabel("Add a plane or pick a preset", self)
        self.stats.setProperty("role", "hint")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)

    # =========================
    # Planes
    # =========================
    @property
    def current(self):
        return self.planes[self.index] if 0 <= self.index < len(self.planes) else None

    def _bounds(self):
        bounds = getattr(self.tab, "start_bounds", None)
        if bounds is None:
            return np.zeros(3), np.ones(3)
        low = np.array(bounds[0::2], dtype=float)
        high = np.array(bounds[1::2], dtype=float)
        return low, high

    def add_plane(self, plane=None):
        if len(self.planes) >= MAX_PLANES:
            self.tab.notify(f"⚠ At most {MAX_PLANES} planes", "warning")
            return
        if plane is None:
            low, high = self._bounds()
            plane = self.current or slicing.axis_plane("Z", low, high)
        self.planes.append(plane)
        self.index = len(self.planes) - 1
        self._fill_combo()

    def remove_plane(self):
        if self.current is None:
            return
        del self.planes[self.index]
        self.index = min(self.index, len(self.planes) - 1)
        self._fill_combo()

    def clear(self):
        self.planes, self.index = [], -1
        self._fill_combo()

    def preset(self, name):
        low, high = self._bounds()
        if name == "Box":
            self.planes = slicing.box_planes(low, high, 0.6)
        elif name == "View":
            self.planes = [self._view_plane(low, high)]
        else:
            self.planes = [slicing.axis_plane(name, low, high)]
        self.index = 0
        self._fill_combo()

    def _view_plane(self, low, high):
        """Through the centre, facing the camera (the near half is kept)."""
        camera = self.tab.renderer.GetActiveCamera()
        towards_camera = -np.array(camera.GetDirectionOfProjection())
        matrix = self.tab.actor_matrix()
        normal = matrix[:3, :3].T @ towards_camera         # world -> model
        return Plane(tuple((low + high) / 2), slicing.angles_for_normal(normal))

    def _fill_combo(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        for i, plane in enumerate(self.planes):
            self.combo.addItem(f"Plane {i + 1}" + ("" if plane.enabled else " (off)"))
        self.combo.setCurrentIndex(self.index)
        self.combo.blockSignals(False)
        self._show_plane()
        self.schedule()

    def _select(self, index):
        self.index = index
        self._show_plane()
        self.schedule()

    def _set_current(self, plane):
        self.planes[self.index] = plane
        self.schedule()

    def _edit(self, **changes):
        if self._updating or self.current is None:
            return
        self._set_current(slicing.with_changes(self.current, **changes))
        if "enabled" in changes:
            self.combo.setItemText(self.index, f"Plane {self.index + 1}"
                                   + ("" if changes["enabled"] else " (off)"))

    def _flip(self):
        if self.current is not None:
            self._edit(flip=not self.current.flip)

    def _position(self, axis, value):
        if self._updating or self.current is None:
            return
        position = list(self.current.position)
        position[axis] = float(value)
        self._sync_slider(axis, value)
        self._edit(position=tuple(position))

    def _position_slider(self, axis, step):
        if self._updating:
            return
        low, high = self._ranges()
        value = low[axis] + (high[axis] - low[axis]) * step / SLIDER_STEPS
        self.position_boxes[axis].setValue(value)

    def _angle(self, axis, value):
        if self._updating or self.current is None:
            return
        rotation = list(self.current.rotation)
        rotation[axis] = float(value)
        self._updating = True
        self.angle_boxes[axis].setValue(int(value))
        self.angle_sliders[axis].setValue(int(value))
        self._updating = False
        self._edit(rotation=tuple(rotation))

    def _ranges(self):
        low, high = self._bounds()
        pad = 0.1 * np.maximum(high - low, 1e-9)
        return low - pad, high + pad

    def _sync_slider(self, axis, value):
        low, high = self._ranges()
        span = max(high[axis] - low[axis], 1e-12)
        self._updating = True
        self.position_sliders[axis].setValue(
            int(round((value - low[axis]) / span * SLIDER_STEPS)))
        self._updating = False

    def _show_plane(self):
        """Controls show the selected plane."""
        plane = self.current
        has_data = getattr(self.tab, "actor", None) is not None
        for widget in (self.check_enabled, self.btn_flip, self.btn_delete,
                       *self.position_boxes, *self.position_sliders,
                       *self.angle_boxes, *self.angle_sliders):
            widget.setEnabled(plane is not None and has_data)
        for widget in (self.btn_add, *self.preset_buttons.values(),
                       self.check_preview, self.check_mouse, self.check_cap):
            widget.setEnabled(has_data)
        ready = has_data and bool(slicing.active(self.planes))
        self.btn_apply.setEnabled(ready)
        self.btn_save.setEnabled(ready)
        self.btn_clear.setEnabled(bool(self.planes))
        self.btn_add.setEnabled(has_data and len(self.planes) < MAX_PLANES)
        low, high = self._ranges()
        size = float(np.max(high - low))
        decimals = int(min(6, max(1, 3 - np.floor(np.log10(max(size, 1e-9))))))
        self._updating = True
        for axis in range(3):
            box = self.position_boxes[axis]
            box.setDecimals(decimals)
            box.setRange(float(low[axis]) - size, float(high[axis]) + size)
            box.setSingleStep(size / 200)
        if plane is not None:
            self.check_enabled.setChecked(plane.enabled)
            for axis in range(3):
                self.position_boxes[axis].setValue(plane.position[axis])
                self.angle_boxes[axis].setValue(int(round(plane.rotation[axis])))
                self.angle_sliders[axis].setValue(int(round(plane.rotation[axis])))
        self._updating = False
        if plane is not None:
            for axis in range(3):
                self._sync_slider(axis, plane.position[axis])

    # =========================
    # Preview
    # =========================
    def schedule(self):
        self._timer.start()

    def on_data_changed(self):
        """The tab shows new data (or none): ranges and preview follow."""
        if getattr(self.tab, "actor", None) is None:
            self._remove_widget()
        self._show_plane()
        self.schedule()

    def sync_pose(self):
        """The model turned / moved: planes follow (they are in model
        coordinates)."""
        if self._preview["quads"] or self._preview["clipped"] or self._widget:
            self.refresh()

    def _clear_preview(self):
        renderer = self.tab.renderer
        for actor in self._preview["quads"] + self._preview["sections"]:
            renderer.RemoveActor(actor)
        if self._preview["ghost"] is not None:
            renderer.RemoveActor(self._preview["ghost"])
        for actor in self._preview["clipped"]:
            mapper = actor.GetMapper()
            if mapper is not None:
                mapper.RemoveAllClippingPlanes()
        self._preview = {"quads": [], "sections": [], "ghost": None, "clipped": []}

    def _actors(self):
        get = getattr(self.tab, "slice_actors", None)
        actors = get() if get else [getattr(self.tab, "actor", None)]
        return [a for a in actors if a is not None]

    def _user_matrix(self):
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(self.tab.actor.GetMatrix())
        return matrix

    def refresh(self):
        """Redraw the preview and the numbers."""
        self._timer.stop()
        self._clear_preview()
        actor = getattr(self.tab, "actor", None)
        planes = slicing.active(self.planes)
        self._show_plane()
        if actor is None:
            self.stats.setText("Load data to slice it")
            self._render()
            return
        if not self.planes:
            self.stats.setText("Add a plane or pick a preset")
        self._update_widget()
        if self.check_preview.isChecked() and planes:
            matrix = self.tab.actor_matrix()
            world = [p.vtk_plane(matrix) for p in planes]
            for target in self._actors():
                mapper = target.GetMapper()
                for plane in world:
                    mapper.AddClippingPlane(plane)
                self._preview["clipped"].append(target)
            self._add_ghost(actor)
            self._add_planes(actor)
        self._update_stats()
        self._render()

    def _render(self):
        viewport = getattr(self.tab, "viewport", None)
        if viewport is not None:
            viewport.render()

    def _add_ghost(self, actor):
        """The removed part, faint (same geometry, not clipped)."""
        data = actor.GetMapper().GetInput()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        mapper.ScalarVisibilityOff()
        ghost = vtk.vtkActor()
        ghost.SetMapper(mapper)
        ghost.SetUserMatrix(self._user_matrix())
        prop = ghost.GetProperty()
        prop.SetColor(0.55, 0.6, 0.66)
        prop.SetOpacity(0.12)
        prop.SetPointSize(actor.GetProperty().GetPointSize())
        if actor.GetProperty().GetRepresentation() == vtk.VTK_POINTS:
            prop.SetRepresentationToPoints()
        ghost.PickableOff()
        self.tab.renderer.AddActor(ghost)
        self._preview["ghost"] = ghost

    def _add_planes(self, actor):
        low, high = self._bounds()
        size = float(np.linalg.norm(high - low)) * 0.6
        data = actor.GetMapper().GetInput()
        has_cells = data is not None and data.GetNumberOfPolys() > 0
        for i, plane in enumerate(self.planes):
            if not plane.enabled:
                continue
            color = QColor(PLANE_COLORS[i % len(PLANE_COLORS)]).getRgbF()[:3]
            dragging = self._widget is not None and i == self.index
            if not dragging:
                self._preview["quads"].append(self._quad(plane, size, color, i == self.index))
            if has_cells:
                self._preview["sections"].append(self._section_line(data, plane, color))

    def _quad(self, plane, size, color, selected):
        rotation = slicing.rotation_matrix(plane.rotation)
        u, v = rotation[:, 0] * size / 2, rotation[:, 1] * size / 2
        centre = plane.origin
        source = vtk.vtkPlaneSource()
        source.SetOrigin(*(centre - u - v))
        source.SetPoint1(*(centre + u - v))
        source.SetPoint2(*(centre - u + v))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        quad = vtk.vtkActor()
        quad.SetMapper(mapper)
        quad.SetUserMatrix(self._user_matrix())
        prop = quad.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(0.30 if selected else 0.16)
        prop.SetEdgeVisibility(True)
        prop.SetEdgeColor(*color)
        prop.SetLineWidth(2.5 if selected else 1.2)
        prop.LightingOff()
        quad.PickableOff()
        self.tab.renderer.AddActor(quad)
        return quad

    def _section_line(self, data, plane, color):
        cutter = vtk.vtkCutter()
        cutter.SetInputData(data)
        cutter.SetCutFunction(plane.vtk_plane())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
        mapper.ScalarVisibilityOff()
        line = vtk.vtkActor()
        line.SetMapper(mapper)
        line.SetUserMatrix(self._user_matrix())
        prop = line.GetProperty()
        prop.SetColor(*color)
        prop.SetLineWidth(3.0)
        prop.LightingOff()
        line.PickableOff()
        self.tab.renderer.AddActor(line)
        return line

    def _update_stats(self):
        source = self._source()
        planes = slicing.active(self.planes)
        if source is None or not planes:
            if source is not None and self.planes:
                self.stats.setText("All planes are off")
            return
        mask = slicing.point_mask(source["vertices"], planes)
        kept, total = int(mask.sum()), len(mask)
        unit = "points" if source["kind"] == "points" else "vertices"
        text = f"Keeps {kept:,} of {total:,} {unit} ({kept / max(total, 1):.0%})"
        plane = self.current
        if source["kind"] == "mesh" and plane is not None and plane.enabled \
                and len(source["triangles"]) <= 2_000_000:
            cut = slicing.section(source["vertices"], source["triangles"], plane)
            text += (f"\nPlane {self.index + 1} cut: length {cut['length']:.4g}"
                     + (f", area {cut['area']:.4g}" if cut["area"] > 0 else ""))
        self.stats.setText(text)

    def _source(self):
        get = getattr(self.tab, "slice_source", None)
        return get() if get else None

    # =========================
    # Mouse handle
    # =========================
    def _remove_widget(self):
        if self._widget is not None:
            self._widget.Off()
            self._widget.RemoveAllObservers()
            self._widget = None

    def _update_widget(self):
        plane = self.current
        actor = getattr(self.tab, "actor", None)
        want = (self.check_mouse.isChecked() and plane is not None
                and plane.enabled and actor is not None)
        if not want:
            self._remove_widget()
            return
        matrix = self.tab.actor_matrix()
        base_normal = -plane.normal if plane.flip else plane.normal
        origin = matrix[:3, :3] @ plane.origin + matrix[:3, 3]
        normal = matrix[:3, :3] @ base_normal
        if self._widget is None:
            representation = vtk.vtkImplicitPlaneRepresentation()
            representation.SetPlaceFactor(1.15)
            representation.OutlineTranslationOff()
            representation.ScaleEnabledOff()
            if hasattr(representation, "SetDrawOutline"):
                representation.SetDrawOutline(False)
            representation.SetTubing(False)
            representation.GetPlaneProperty().SetOpacity(0.25)
            representation.PlaceWidget(actor.GetBounds())
            widget = vtk.vtkImplicitPlaneWidget2()
            widget.SetInteractor(self.tab.interactor)
            widget.SetRepresentation(representation)
            widget.AddObserver("InteractionEvent", self._widget_moved)
            widget.AddObserver("EndInteractionEvent", lambda *_: self.refresh())
            widget.On()
            self._widget = widget
        representation = self._widget.GetRepresentation()
        representation.SetOrigin(*origin)
        representation.SetNormal(*normal)

    def _widget_moved(self, *_):
        """The handle moved: world plane -> model coordinates."""
        if self.current is None:
            return
        representation = self._widget.GetRepresentation()
        matrix = self.tab.actor_matrix()
        inverse = np.linalg.inv(matrix)
        origin = inverse[:3, :3] @ np.array(representation.GetOrigin()) + inverse[:3, 3]
        normal = matrix[:3, :3].T @ np.array(representation.GetNormal())
        self.planes[self.index] = slicing.with_changes(
            self.current, position=tuple(origin),
            rotation=slicing.angles_for_normal(normal))
        self._show_plane()
        self.schedule()

    # =========================
    # Apply / save
    # =========================
    def apply(self):
        planes = slicing.active(self.planes)
        if not planes or getattr(self.tab, "actor", None) is None:
            return
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            message = self.tab.apply_slice(list(planes), self.check_cap.isChecked())
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.tab.notify(f"❌ Slice failed: {e}", "error", dialog=True)
            return
        QApplication.restoreOverrideCursor()
        self.tab.console.appendPlainText(
            "✂ Slice planes: " + "; ".join(
                f"P{i + 1} at ({', '.join(f'{c:.4g}' for c in p.position)}) "
                f"angles ({', '.join(f'{a:.0f}' for a in p.rotation)})"
                + (" flipped" if p.flip else "") for i, p in enumerate(planes)))
        self.clear()
        self.tab.notify(f"✂ {message}", "success")

    def sliced_arrays(self):
        """The kept part of the shown data (not applied) ->
        (kind, vertices, triangles, normals, colors)."""
        source = self._source()
        planes = slicing.active(self.planes)
        if source is None or not planes:
            return None
        return slice_source_data(source, planes, self.check_cap.isChecked())

    def save(self, path=None):
        result = self.sliced_arrays()
        if result is None:
            return None
        kind, vertices, triangles, normals, colors = result
        if len(vertices) == 0:
            self.tab.notify("⚠ The slice is empty - nothing to save", "warning")
            return None
        if not path:
            filters = ("STL mesh (*.stl);;OBJ mesh (*.obj);;PLY mesh (*.ply)"
                       if kind == "mesh" else "PLY point cloud (*.ply)")
            path, _ = QFileDialog.getSaveFileName(
                self.tab.main_window, "Save slice",
                f"{settings.last_dir()}/slice.{'stl' if kind == 'mesh' else 'ply'}",
                filters)
            if not path:
                return None
        from modules.geometry_utils import apply_transform
        vertices, normals = apply_transform(self.tab.actor_matrix(), vertices, normals)
        import open3d as o3d
        if kind == "mesh":
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
            mesh.compute_vertex_normals()
            if colors is not None and not path.lower().endswith(".stl"):
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            ok = o3d.io.write_triangle_mesh(path, mesh)
        else:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
            if normals is not None:
                cloud.normals = o3d.utility.Vector3dVector(normals)
            if colors is not None:
                cloud.colors = o3d.utility.Vector3dVector(colors)
            ok = o3d.io.write_point_cloud(path, cloud)
        if not ok:
            self.tab.notify(f"❌ Could not write {path}", "error", dialog=True)
            return None
        settings.remember_path(path)
        self.tab.notify(f"💾 Slice saved: {path}", "success")
        return path


def mesh_normals(vertices, triangles):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertex_normals).copy()


def slice_source_data(source, planes, cap):
    """Slice a ``slice_source`` dict -> (kind, vertices, triangles,
    normals, colors)."""
    from modules.geometry_utils import transfer_colors
    kind = source["kind"]
    vertices, triangles, mask = slicing.slice_data(
        kind, source["vertices"], source.get("triangles"), planes, cap)
    if kind == "points":
        normals = None if source.get("normals") is None else np.asarray(source["normals"])[mask]
        colors = None if source.get("colors") is None else np.asarray(source["colors"])[mask]
        return kind, vertices, None, normals, colors
    normals = mesh_normals(vertices, triangles) if len(triangles) else np.zeros((0, 3))
    colors = (transfer_colors(source["vertices"], source.get("colors"), vertices)
              if len(vertices) else None)
    return kind, vertices, triangles, normals, colors


def install(tab, cap_default=True):
    """Put a *Slice* section above the tab's *Output* section."""
    from modules.point_data import add_section, insert_widget
    anchor = tab.btn_export
    layout = anchor.parentWidget().layout()
    index = layout.indexOf(anchor)
    header = layout.itemAt(index - 1).widget() if index > 0 else None
    panel = SlicePanel(tab, anchor.parentWidget(), cap_default)
    target = header if header is not None and header.property("role") == "section" else anchor
    insert_widget(target, panel, after=False)
    add_section(panel, "Slice")
    tab.slice_panel = panel
    return panel
