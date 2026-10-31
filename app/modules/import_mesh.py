"""Tab 1 - Import mesh / point cloud, rotate it and export points+normals."""

import os

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QStyle

from modules import settings
from modules.geometry_utils import (
    estimate_oriented_normals,
    normalize,
    polydata_from_numpy,
)
from modules.point_data import (
    PointCloudData,
    add_button_after,
    add_section,
    ask_save_point_cloud,
    insert_widget,
)
from modules.vtk_controls import TabMixin, VTKViewport

SAMPLE_MESH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models",
    "Stanford_Bunny.stl",
))
MESH_EXTENSIONS = (".ply", ".obj", ".stl", ".pcd", ".off", ".gltf", ".glb")
OPEN_FILTER = "3D files (*.ply *.obj *.stl *.pcd *.off *.gltf *.glb);;All files (*)"


class Geometry(TabMixin):
    """Import a 3D geometry (mesh or point cloud), display it in VTK,
    rotate it with sliders/spinboxes or the mouse, then export
    points + normals or send them straight to another tab (with the
    displayed rotation applied).
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.btn_import = main_window.btn_import
        self.btn_export = main_window.btn_export
        self.pushButton = main_window.pushButton

        self.spinBox_X = main_window.spinBox_X
        self.spinBox_Y = main_window.spinBox_Y
        self.spinBox_Z = main_window.spinBox_Z
        self.slider_X = main_window.slider_X
        self.slider_Y = main_window.slider_Y
        self.slider_Z = main_window.slider_Z

        # Geometry data
        self.points = None
        self.normals = None
        self.triangles = None   # None for a point cloud
        self.source_name = ""
        self.init_rotation_state()

        # Console + toolbar + VTK render window + axes
        VTKViewport(
            self, main_window.frame_console, main_window.frame_view,
            empty_hint="Drop a mesh or point cloud here\n"
                       "or press Ctrl+O to open one\n\n"
                       "(File > Open Sample to try the Stanford bunny)",
        )

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(lambda: self.import_file())
        self.btn_export.clicked.connect(self.export_data)
        self.setup_rotation_controls()

        # Send the data to another tab in memory (no intermediate files)
        self.btn_send_reconstruction = add_button_after(
            self.btn_export, "Send to Reconstruction"
        )
        self.btn_send_processing = add_button_after(
            self.btn_export, "Send to Processing"
        )
        self.btn_send_processing.clicked.connect(
            lambda: self.send_data("processing")
        )
        self.btn_send_reconstruction.clicked.connect(
            lambda: self.send_data("reconstruction")
        )
        self._build_panel()

        # Disable controls until a file is imported
        for widget in self.rotation_widgets() + self._data_buttons():
            widget.setEnabled(False)

        self.console.appendPlainText(
            "Ready. Open a mesh or point cloud (*.ply, *.obj, *.stl, *.pcd)."
        )

    def _build_panel(self):
        """Section headers, icons, tooltips and the file label."""
        self.btn_import.setText("Open file…")
        self.decorate(self.btn_import, QStyle.SP_DialogOpenButton,
                      "Open a mesh or point cloud (Ctrl+O)", "primary")
        self.file_label = QLabel("No file loaded")
        self.file_label.setProperty("role", "hint")
        self.file_label.setWordWrap(True)
        insert_widget(self.btn_import, self.file_label, after=True)
        self.btn_sample = QPushButton("or open the sample bunny")
        self.btn_sample.setProperty("variant", "link")
        self.btn_sample.clicked.connect(self.open_sample)
        insert_widget(self.file_label, self.btn_sample, after=True)

        self.btn_export.setText("Export points…")
        self.decorate(self.btn_export, QStyle.SP_DialogSaveButton,
                      "Save points + normals (PLY / TXT / CSV / NPY), "
                      "rotation applied (Ctrl+S)")
        self.decorate(self.btn_send_processing, QStyle.SP_ArrowForward,
                      "Continue in the Processing tab with this data")
        self.decorate(self.btn_send_reconstruction, QStyle.SP_ArrowForward,
                      "Reconstruct a surface from this data", "primary")

        add_section(self.btn_import, "File")
        add_section(self.main_window.label_x, "Rotation")
        add_section(self.btn_export, "Output")
        for axis in "XYZ":
            for widget in (getattr(self, f"spinBox_{axis}"),
                           getattr(self, f"slider_{axis}")):
                widget.setToolTip(
                    f"Rotation about the {axis} axis in degrees "
                    "(you can also drag the model with the left mouse button)")

    # =========================
    # Import Geometry
    # =========================
    @staticmethod
    def _load_point_cloud(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            raise ValueError("File contains no points!")
        pcd = pcd.remove_duplicated_points()
        if pcd.has_normals():
            normals = normalize(np.asarray(pcd.normals))
        else:
            normals = np.asarray(estimate_oriented_normals(pcd).normals)
        return np.asarray(pcd.points), normals, None

    @staticmethod
    def _load_mesh(file_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if len(mesh.triangles) == 0:
            return None
        # STL stores every triangle corner separately; merge them so each
        # surface point is exported once with an averaged vertex normal.
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return (
            np.asarray(mesh.vertices).copy(),
            normalize(np.asarray(mesh.vertex_normals)),
            np.asarray(mesh.triangles).copy(),
        )

    def import_geometry(self, file_path):
        """Load the file; the current state is only replaced on success."""
        ext = os.path.splitext(file_path)[1].lower()
        result = None
        if ext != ".pcd":
            result = self._load_mesh(file_path)
            if result is None and ext != ".ply":
                raise ValueError("the file contains no triangles")
        if result is None:
            # .pcd, or a .ply that stores a point cloud instead of a mesh
            result = self._load_point_cloud(file_path)
        self.points, self.normals, self.triangles = result
        self.source_name = os.path.basename(file_path)

    def _data_buttons(self):
        return (
            self.btn_export,
            self.btn_send_processing,
            self.btn_send_reconstruction,
        )

    # =========================
    # Current data (as displayed)
    # =========================
    def current_data(self):
        """Points + normals with the displayed rotation applied."""
        reference = (
            (self.points, self.triangles) if self.triangles is not None else None
        )
        data = PointCloudData(
            self.points, self.normals, self.source_name, reference
        )
        return data.transformed(self.actor_matrix())

    def send_data(self, target):
        if self.points is None:
            self.notify(self.NO_ACTOR_MESSAGE, "warning")
            return
        self.main_window.send_point_cloud(self.current_data(), target)
        self.console.appendPlainText(
            f"➡ Sent {len(self.points)} points + normals to {target}"
        )

    # =========================
    # Render Geometry
    # =========================
    def render_geometry(self):
        """Render geometry and return the actor."""
        polydata = polydata_from_numpy(self.points, self.normals, self.triangles)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)
        actor.GetProperty().SetInterpolationToPhong()

        kind = "points" if self.triangles is None else "mesh"
        return self.show_actor(actor, kind=kind)

    # =========================
    # Import File (UI)
    # =========================
    def open_sample(self):
        if not os.path.isfile(SAMPLE_MESH):
            self.notify(f"❌ Sample not found: {SAMPLE_MESH}", "error", dialog=True)
            return
        self.import_file(SAMPLE_MESH)

    def import_file(self, file_path=None):
        """Open ``file_path`` (asks with a dialog when not given)."""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "Open Mesh or Point Cloud",
                settings.last_dir(), OPEN_FILTER
            )
            if not file_path:
                return
        try:
            self.import_geometry(file_path)
        except Exception as e:
            self.notify(
                f"❌ Could not open {os.path.basename(file_path)}: {e}",
                "error", dialog=True)
            return

        settings.add_recent(file_path)
        kind = "Point cloud" if self.triangles is None else "Mesh"
        self.console.clear()
        self.console.appendPlainText(f"✅ Opened {file_path}")
        self.console.appendPlainText(
            f"   {kind} · {len(self.points):,} points"
            + (f" · {len(self.triangles):,} triangles"
               if self.triangles is not None else "")
        )
        self.file_label.setText(
            f"{self.source_name}\n{kind} · {len(self.points):,} points")
        self.file_label.setProperty("role", "file")
        self.file_label.style().polish(self.file_label)

        for widget in self.rotation_widgets() + self._data_buttons():
            widget.setEnabled(True)

        self.reset_rotation()
        self.render_geometry()
        self.set_tab_info(
            file=self.source_name, points=len(self.points),
            triangles=None if self.triangles is None else len(self.triangles))
        self.notify(f"Opened {self.source_name}", "success")

    # =========================
    # Export Data (UI)
    # =========================
    def export_data(self):
        if self.points is None:
            return
        try:
            paths = ask_save_point_cloud(
                self.main_window, self.current_data(), "point_cloud.ply"
            )
            if not paths:
                return
            for path in paths:
                self.console.appendPlainText(path)
            self.notify("✅ Exported (rotation applied): "
                        + ", ".join(os.path.basename(p) for p in paths), "success")
        except Exception as e:
            self.notify(f"❌ Export failed: {e}", "error", dialog=True)
