"""Tab 2 - Point cloud processing (import TXT, shuffle, voxel density)."""

import os

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog

from modules.vtk_controls import VTKViewport, RotationMixin


class Processing(RotationMixin):
    """Import a point cloud (TXT), shuffle the points, reduce the density
    with voxel downsampling and export the modified points (+ normals).
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.btn_import = main_window.btn_import_2
        self.btn_export = main_window.btn_export_2
        self.pushButton = main_window.pushButton_2
        self.random_point_btn = main_window.random_point_btn
        self.Point_density_spinbox = main_window.Point_density_spinbox

        self.spinBox_X = main_window.spinBox_X_2
        self.spinBox_Y = main_window.spinBox_Y_2
        self.spinBox_Z = main_window.spinBox_Z_2
        self.slider_X = main_window.slider_X_2
        self.slider_Y = main_window.slider_Y_2
        self.slider_Z = main_window.slider_Z_2

        # ---------- State ----------
        self.points = None
        self.normals = None
        self.points_changed = None
        self.normals_changed = None
        self.is_pointcloud = False
        self.actor = None
        self.mesh_actor = None
        self.vtk_initialized = False

        # Console + VTK render window + axes
        VTKViewport(self, main_window.frame_console_2, main_window.frame_view_2)

        # ---------- Rotation ----------
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prev_rotation_x = 0.0
        self.prev_rotation_y = 0.0
        self.prev_rotation_z = 0.0

        # ---------- Voxel density ----------
        self.current_voxel_size = 0.05   # default voxel size
        self.Point_density_spinbox.setDecimals(3)
        self.Point_density_spinbox.setSingleStep(0.005)
        self.Point_density_spinbox.setMinimum(0.001)
        self.Point_density_spinbox.setMaximum(1.0)
        self.Point_density_spinbox.setValue(self.current_voxel_size)
        self.Point_density_spinbox.setEnabled(False)
        self.Point_density_spinbox.valueChanged.connect(self.update_voxel_points)

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_file)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)
        self.random_point_btn.clicked.connect(self.randomize_points)
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
        self.console.appendPlainText("Program ready ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("📁 Please import a file: (*.txt)")

        # Update spinboxes/sliders when the actor is rotated with the mouse
        self.interactor.AddObserver(
            "EndInteractionEvent", self.update_spinbox_from_actor
        )

    # =========================
    # Randomize Points
    # =========================
    def randomize_points(self):
        source = self.points_changed if self.points_changed is not None else self.points
        if source is None:
            self.console.appendPlainText("❌ No points to shuffle — please import a file")
            return
        perm = np.random.permutation(len(source))
        self.points_changed = source[perm]
        if self.normals_changed is not None:
            self.normals_changed = self.normals_changed[perm]
        self.console.appendPlainText(f"🔀 Points shuffled ✅ ({len(self.points_changed)})")
        self.mesh_actor = self.render_processing_with_points(
            self.points_changed, reset_camera=False
        )

    # =========================
    # Voxel Density
    # =========================
    def update_voxel_points(self):
        if self.points is None:
            return
        voxel_size = self.Point_density_spinbox.value()
        if voxel_size <= 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        self.points_changed = np.asarray(pcd.points)
        self.normals_changed = None

        # Estimate fresh normals for the downsampled cloud (best effort)
        if len(self.points_changed) >= 3:
            try:
                den = o3d.geometry.PointCloud()
                den.points = o3d.utility.Vector3dVector(self.points_changed)
                den.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=max(voxel_size * 4.0, 1e-4), max_nn=30
                    )
                )
                self.normals_changed = np.asarray(den.normals)
            except Exception:
                self.normals_changed = None

        self.console.appendPlainText(
            f"✅ Voxel density {voxel_size:.3f} → {len(self.points_changed)} points"
        )
        self.mesh_actor = self.render_processing_with_points(
            self.points_changed, reset_camera=False
        )
# =========================
    # Render Point Cloud
    # =========================
    def render_processing_with_points(self, points_array, reset_camera=True):
        """Render an array of points as VTK glyphs and return the actor.

        ``reset_camera`` may be disabled when only the density changes so
        that the current camera position is kept.
        """
        if points_array is None:
            return None

        # Remove old actor
        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        vtk_points = vtk.vtkPoints()
        for p in points_array:
            vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # Point normals when available and matching in size
        if self.normals_changed is not None and len(self.normals_changed) == len(points_array):
            vtk_normals = vtk.vtkFloatArray()
            vtk_normals.SetNumberOfComponents(3)
            vtk_normals.SetName("Normals")
            for n in self.normals_changed:
                vtk_normals.InsertNextTuple((float(n[0]), float(n[1]), float(n[2])))
            polydata.GetPointData().SetNormals(vtk_normals)

        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(polydata)
        glyph.Update()
        polydata = glyph.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        self.actor.GetProperty().SetPointSize(2)
        self.actor.GetProperty().SetInterpolationToPhong()

        self.renderer.AddActor(self.actor)
        if reset_camera:
            self.renderer.ResetCamera()

        if not self.vtk_initialized:
            self.interactor.Initialize()
            self.vtk_initialized = True

        self.interactor.GetRenderWindow().Render()
        return self.actor

    # =========================
    # Import Point (data)
    # =========================
    def import_point(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".txt":
            self.console.appendPlainText("❌ Please import a txt file for point cloud")
            return False
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 3:
                self.console.appendPlainText(
                    "❌ TXT file must have at least 3 columns (x, y, z)"
                )
                return False

            self.points = data[:, :3]
            if data.shape[1] >= 6:
                self.normals = data[:, 3:6]
            else:
                self.normals = None

            self.points_changed = None
            self.normals_changed = (
                self.normals.copy() if self.normals is not None else None
            )
            self.is_pointcloud = True
            return True
        except Exception as e:
            self.console.appendPlainText(f"❌ Error reading txt: {str(e)}")
            return False

    # =========================
    # Import Point Cloud (UI)
    # =========================
    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Point Cloud", "", "Text Files (*.txt)"
        )
        if not file_path:
            return
        try:
            if not self.import_point(file_path):
                return

            self.console.clear()
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"File imported ✅: {file_path}")
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"📊 Points: {len(self.points)}")

            # Enable rotation and density controls
            for widget in [
                self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
                self.slider_X, self.slider_Y, self.slider_Z,
                self.pushButton, self.btn_export,
                self.Point_density_spinbox
            ]:
                widget.setEnabled(True)
            self.random_point_btn.setEnabled(True)

            # Render points initially
            self.mesh_actor = self.render_processing_with_points(self.points)

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
    # Export Point Cloud
    # =========================
    def export_data(self):
        data = self.points_changed if self.points_changed is not None else self.points
        if data is None:
            self.console.appendPlainText("❌ No data to export")
            return

        file_path_points, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Points", "points_changed.txt", "Text Files (*.txt)"
        )
        if not file_path_points:
            return

        try:
            np.savetxt(file_path_points, data, fmt="%.6f")
            self.console.appendPlainText("File Exported ✅")
            self.console.appendPlainText(f"{file_path_points}")

            # Also export normals when available
            if self.normals_changed is not None:
                file_path_normals, _ = QFileDialog.getSaveFileName(
                    self.main_window, "Save Normals", "normals_changed.txt",
                    "Text Files (*.txt)"
                )
                if file_path_normals:
                    np.savetxt(file_path_normals, self.normals_changed, fmt="%.6f")
                    self.console.appendPlainText(f"{file_path_normals}")
        except Exception as e:
            self.console.appendPlainText(f"❌ Export error: {str(e)}")