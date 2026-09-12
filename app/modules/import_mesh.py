"""Tab 1 - Import mesh / point cloud, rotate it and export points+normals."""

import os

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog

from modules.vtk_controls import VTKViewport, RotationMixin


class Geometry(RotationMixin):
    """Import a 3D geometry (mesh or point cloud), display it in VTK,
    rotate it with sliders/spinboxes or the mouse and export
    points + normals.
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
        self.geometry = None
        self.points = None
        self.normals = None
        self.is_pointcloud = False
        self.actor = None
        self.mesh_actor = None
        self.vtk_initialized = False

        # Console + VTK render window + axes
        VTKViewport(self, main_window.frame_console, main_window.frame_view)

        # ---------- Rotation ----------
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prev_rotation_x = 0.0
        self.prev_rotation_y = 0.0
        self.prev_rotation_z = 0.0

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_file)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)
        self.btn_export.setEnabled(False)

        self.setup_rotation_controls()

        # Disable controls until a file is imported
        for widget in [
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton
        ]:
            widget.setEnabled(False)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("The program ready to use ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(
            "📁 Please import a file: (*.ply, *.obj, *.stl, *.pcd)"
        )

        # Update spinboxes/sliders when the actor is rotated with the mouse
        self.interactor.AddObserver(
            "EndInteractionEvent", self.update_spinbox_from_actor
        )

    # =========================
    # Import Geometry
    # =========================
    def import_geometry(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pcd":
            self.is_pointcloud = True
            self.geometry = o3d.io.read_point_cloud(file_path)
            if not self.geometry.has_normals():
                self.geometry.estimate_normals()
            self.points = np.asarray(self.geometry.points)
            self.normals = np.asarray(self.geometry.normals)
        else:
            self.is_pointcloud = False
            self.geometry = o3d.io.read_triangle_mesh(file_path)
            self.geometry.compute_vertex_normals()
            if len(self.geometry.triangles) == 0:
                raise ValueError("Mesh has no triangles!")
            self.points = np.asarray(self.geometry.vertices)
            self.normals = np.asarray(self.geometry.vertex_normals)

    # =========================
    # Export Geometry
    # =========================
    def export_geometry(self, file_path_points, file_path_normals):
        if self.geometry is None:
            return
        np.savetxt(file_path_points, self.points, fmt="%.6f")
        np.savetxt(file_path_normals, self.normals, fmt="%.6f")
# =========================
    # Render Geometry
    # =========================
    def render_geometry(self):
        """Render geometry and return the actor."""
        if self.geometry is None:
            print("[ERROR] Geometry is None in render_geometry()")
            return None

        # Remove previous actor
        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        # ========== Build VTK Points ==========
        vtk_points = vtk.vtkPoints()
        vtk_normals = vtk.vtkFloatArray()
        vtk_normals.SetNumberOfComponents(3)
        vtk_normals.SetName("Normals")

        for p, n in zip(self.points, self.normals):
            vtk_points.InsertNextPoint(p[0], p[1], p[2])
            vtk_normals.InsertNextTuple((n[0], n[1], n[2]))

        # ========== Build PolyData ==========
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetNormals(vtk_normals)

        # Add cells only for a mesh (not a point cloud)
        if not self.is_pointcloud:
            vtk_cells = vtk.vtkCellArray()
            for tri in np.asarray(self.geometry.triangles):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, tri[0])
                triangle.GetPointIds().SetId(1, tri[1])
                triangle.GetPointIds().SetId(2, tri[2])
                vtk_cells.InsertNextCell(triangle)
            polydata.SetPolys(vtk_cells)

        # Point cloud -> glyphs
        if self.is_pointcloud:
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            polydata = glyph.GetOutput()

        # ========== Mapper & Actor ==========
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        self.actor.GetProperty().SetPointSize(2)
        self.actor.GetProperty().SetInterpolationToPhong()

        self.renderer.AddActor(self.actor)
        self.renderer.ResetCamera()

        if not self.vtk_initialized:
            self.interactor.Initialize()
            self.vtk_initialized = True

        self.interactor.GetRenderWindow().Render()

        return self.actor

    # =========================
    # Import File (UI)
    # =========================
    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Geometry", "",
            "3D Files (*.ply *.obj *.stl *.pcd)"
        )
        if not file_path:
            return
        try:
            self.import_geometry(file_path)
            self.console.clear()
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"File imported ✅: {file_path}")
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"📊 Points: {len(self.points)}")
            self.console.appendPlainText(f"📊 Normals: {len(self.normals)}\n")

            # Enable rotation controls
            for widget in [
                self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
                self.slider_X, self.slider_Y, self.slider_Z,
                self.pushButton, self.btn_export
            ]:
                widget.setEnabled(True)

            # Render mesh
            self.mesh_actor = self.render_geometry()

            # Reset rotation and UI
            self.rotation_x = self.rotation_y = self.rotation_z = 0.0
            self.prev_rotation_x = self.prev_rotation_y = self.prev_rotation_z = 0.0

            for spin in [self.spinBox_X, self.spinBox_Y, self.spinBox_Z]:
                spin.blockSignals(True)
                spin.setValue(0)
                spin.blockSignals(False)

            for slider in [self.slider_X, self.slider_Y, self.slider_Z]:
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)

            # Reset camera
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            self.console.appendPlainText(f"❌ Error: {str(e)}")

    # =========================
    # Export Data (UI)
    # =========================
    def export_data(self):
        file_path_points, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Points", "points.txt", "Text Files (*.txt)"
        )
        file_path_normals, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Normals", "normals.txt", "Text Files (*.txt)"
        )
        if not file_path_points or not file_path_normals:
            return
        try:
            self.export_geometry(file_path_points, file_path_normals)
            self.console.appendPlainText("File Exported ✅")
            self.console.appendPlainText(f"{file_path_points}")
            self.console.appendPlainText(f"{file_path_normals}")
        except Exception as e:
            self.console.appendPlainText(f"❌ Error in export: {str(e)}")