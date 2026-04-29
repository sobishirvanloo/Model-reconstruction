import os
import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit, QVBoxLayout
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# ---------- Customize class for VTK zooming ----------
class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballActor):

    def __init__(self, renderer, parent=None):
        self.AddObserver("MouseWheelForwardEvent", self.zoom_in)
        self.AddObserver("MouseWheelBackwardEvent", self.zoom_out)
        self.renderer = renderer

    def zoom_in(self, obj, event):
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(1.1)
        self.renderer.ResetCameraClippingRange()
        self.GetInteractor().GetRenderWindow().Render()

    def zoom_out(self, obj, event):
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(0.9)
        self.renderer.ResetCameraClippingRange()
        self.GetInteractor().GetRenderWindow().Render()


class PoissonReconstruction:
    def __init__(self, main_window):
        self.main_window = main_window

        # ---------- UI Elements ----------
        self.frame_console = main_window.frame_console_6
        self.frame_view    = main_window.frame_view_6
        self.btn_import    = main_window.btn_import_pointsandnormals
        self.btn_export    = main_window.btn_export_6
        self.pushButton    = main_window.pushButton_6

        # Rotation controls
        self.spinBox_X = main_window.spinBox_X_6
        self.spinBox_Y = main_window.spinBox_Y_6
        self.spinBox_Z = main_window.spinBox_Z_6
        self.slider_X  = main_window.slider_X_6
        self.slider_Y  = main_window.slider_Y_6
        self.slider_Z  = main_window.slider_Z_6

        # Poisson parameters
        self.spinBox_depth      = main_window.spinBox        # QSpinBox  — Depth (max 12)
        self.doubleSpinBox_width  = main_window.doubleSpinBox   # QDoubleSpinBox — Width
        self.doubleSpinBox_scale  = main_window.doubleSpinBox_2  # QDoubleSpinBox — Scale
        self.checkBox_linear_fit  = main_window.Linear_fit      # QCheckBox — linear_fit

        # ---------- State ----------
        self.points  = None
        self.normals = None
        self.mesh_o3d   = None   # open3d mesh after reconstruction
        self.actor      = None
        self.mesh_actor = None

        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prev_rotation_x = 0.0
        self.prev_rotation_y = 0.0
        self.prev_rotation_z = 0.0

        # ---------- Console ----------
        self.console = QPlainTextEdit(self.frame_console)
        self.console.setReadOnly(True)
        self.console.setGeometry(self.frame_console.rect())
        self.console.setStyleSheet(
            "background-color: #007ea7; color: white; font-family: Consolas; font-size: 16px;"
        )

        # ---------- VTK Widget ----------
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame_view)
        layout = QVBoxLayout(self.frame_view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.95)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.GetRenderWindow().SetMultiSamples(0)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.interactor.Initialize()
        style = CustomInteractorStyle(self.renderer)
        self.interactor.SetInteractorStyle(style)

        self.setup_axes()
        self.vtk_widget.GetRenderWindow().Render()

        # ---------- Poisson parameter defaults ----------
        self.spinBox_depth.setMinimum(1)
        self.spinBox_depth.setMaximum(12)
        self.spinBox_depth.setValue(8)

        self.doubleSpinBox_width.setMinimum(0.0)
        self.doubleSpinBox_width.setMaximum(9999.0)
        self.doubleSpinBox_width.setSingleStep(0.1)
        self.doubleSpinBox_width.setValue(0.0)

        self.doubleSpinBox_scale.setMinimum(0.01)
        self.doubleSpinBox_scale.setMaximum(100.0)
        self.doubleSpinBox_scale.setSingleStep(0.1)
        self.doubleSpinBox_scale.setValue(1.1)

        self.checkBox_linear_fit.setChecked(False)

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_files)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)
        self.btn_export.setEnabled(False)

        # Live update on parameter change
        self.spinBox_depth.valueChanged.connect(self.live_reconstruct)
        self.doubleSpinBox_width.valueChanged.connect(self.live_reconstruct)
        self.doubleSpinBox_scale.valueChanged.connect(self.live_reconstruct)
        self.checkBox_linear_fit.stateChanged.connect(self.live_reconstruct)

        # ---------- Rotation controls ----------
        self.setup_rotation_controls()

        # ---------- Disable controls until import ----------
        self._set_controls_enabled(False)

        self.interactor.AddObserver("EndInteractionEvent", self.update_spinbox_from_actor)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("Poisson Reconstruction ready ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("📁 Please import Points and Normals files (.txt / .npy)")

    # =========================
    # Helper
    # =========================
    def _set_controls_enabled(self, enabled: bool):
        for widget in [
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton, self.btn_export,
            self.spinBox_depth, self.doubleSpinBox_width,
            self.doubleSpinBox_scale, self.checkBox_linear_fit,
        ]:
            widget.setEnabled(enabled)

    # =========================
    # Axes
    # =========================
    def setup_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1, 1, 1)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.03)
        axes.SetAxisLabels(1)
        axes.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axes.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axes.GetZAxisShaftProperty().SetColor(0, 0, 1)

        self.marker = vtk.vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(axes)
        self.marker.SetInteractor(self.interactor)
        self.marker.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.marker.SetEnabled(1)
        self.marker.InteractiveOff()

    # =========================
    # Rotation Controls
    # =========================
    def setup_rotation_controls(self):
        self.slider_X.valueChanged.connect(lambda v: self.sync_rotation(self.spinBox_X, 'X', v))
        self.spinBox_X.valueChanged.connect(lambda v: self.sync_rotation(self.slider_X,  'X', v))
        self.slider_Y.valueChanged.connect(lambda v: self.sync_rotation(self.spinBox_Y, 'Y', v))
        self.spinBox_Y.valueChanged.connect(lambda v: self.sync_rotation(self.slider_Y,  'Y', v))
        self.slider_Z.valueChanged.connect(lambda v: self.sync_rotation(self.spinBox_Z, 'Z', v))
        self.spinBox_Z.valueChanged.connect(lambda v: self.sync_rotation(self.slider_Z,  'Z', v))

    def sync_rotation(self, widget_to_update, axis, value):
        if axis == 'X':
            self.rotation_x = float(value)
        elif axis == 'Y':
            self.rotation_y = float(value)
        elif axis == 'Z':
            self.rotation_z = float(value)

        widget_to_update.blockSignals(True)
        widget_to_update.setValue(int(value))
        widget_to_update.blockSignals(False)

    def update_spinbox_from_actor(self, obj, event):
        if self.actor is None:
            return
        try:
            matrix = self.actor.GetMatrix()

            import math
            sy = math.sqrt(matrix.GetElement(0, 0) ** 2 + matrix.GetElement(1, 0) ** 2)
            singular = sy < 1e-6

            if not singular:
                rx = math.degrees(math.atan2(matrix.GetElement(2, 1), matrix.GetElement(2, 2)))
                ry = math.degrees(math.atan2(-matrix.GetElement(2, 0), sy))
                rz = math.degrees(math.atan2(matrix.GetElement(1, 0), matrix.GetElement(0, 0)))
            else:
                rx = math.degrees(math.atan2(-matrix.GetElement(1, 2), matrix.GetElement(1, 1)))
                ry = math.degrees(math.atan2(-matrix.GetElement(2, 0), sy))
                rz = 0.0

            self._update_rotation_ui('X', rx)
            self._update_rotation_ui('Y', ry)
            self._update_rotation_ui('Z', rz)
        except Exception:
            pass

    def _update_rotation_ui(self, axis, value):
        value = int(round(value))
        if axis == 'X':
            self.rotation_x = float(value)
            for w in [self.spinBox_X, self.slider_X]:
                w.blockSignals(True); w.setValue(value); w.blockSignals(False)
        elif axis == 'Y':
            self.rotation_y = float(value)
            for w in [self.spinBox_Y, self.slider_Y]:
                w.blockSignals(True); w.setValue(value); w.blockSignals(False)
        elif axis == 'Z':
            self.rotation_z = float(value)
            for w in [self.spinBox_Z, self.slider_Z]:
                w.blockSignals(True); w.setValue(value); w.blockSignals(False)

    def apply_rotation(self):
        if self.actor is None:
            self.console.appendPlainText("❌ No mesh loaded to rotate")
            return
        try:
            delta_x = self.rotation_x - self.prev_rotation_x
            delta_y = self.rotation_y - self.prev_rotation_y
            delta_z = self.rotation_z - self.prev_rotation_z

            self.actor.RotateX(delta_x)
            self.actor.RotateY(delta_y)
            self.actor.RotateZ(delta_z)

            self.prev_rotation_x = self.rotation_x
            self.prev_rotation_y = self.rotation_y
            self.prev_rotation_z = self.rotation_z

            self.vtk_widget.GetRenderWindow().Render()
            self.console.appendPlainText(
                f"✅ Rotation applied | X: {self.rotation_x:.1f}° | Y: {self.rotation_y:.1f}° | Z: {self.rotation_z:.1f}°"
            )
        except Exception as e:
            self.console.appendPlainText(f"❌ Error: {str(e)}")

    # =========================
    # Import Files
    # =========================
    def import_files(self):
        # --- Import Points ---
        points_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import Points File",
            "",
            "Point Files (*.txt *.npy *.csv)"
        )
        if not points_path:
            self.console.appendPlainText("❌ Import cancelled (points file not selected)")
            return

        # --- Import Normals ---
        normals_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import Normals File",
            "",
            "Normal Files (*.txt *.npy *.csv)"
        )
        if not normals_path:
            self.console.appendPlainText("❌ Import cancelled (normals file not selected)")
            return

        try:
            self.points  = self._load_array(points_path)
            self.normals = self._load_array(normals_path)

            if self.points.shape[1] < 3:
                self.console.appendPlainText("❌ Points file must have at least 3 columns (x, y, z)")
                return
            if self.normals.shape[1] < 3:
                self.console.appendPlainText("❌ Normals file must have at least 3 columns (nx, ny, nz)")
                return

            self.points  = self.points[:, :3]
            self.normals = self.normals[:, :3]

            if self.points.shape[0] != self.normals.shape[0]:
                self.console.appendPlainText(
                    f"❌ Row count mismatch: points={self.points.shape[0]}, normals={self.normals.shape[0]}"
                )
                return

            self.console.clear()
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"✅ Points loaded : {points_path}")
            self.console.appendPlainText(f"✅ Normals loaded: {normals_path}")
            self.console.appendPlainText(f"📊 Point count   : {len(self.points)}")
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText("⏳ Running Poisson reconstruction…")

            # Enable controls
            self._set_controls_enabled(True)

            # Reset rotation UI
            self.rotation_x = self.rotation_y = self.rotation_z = 0.0
            self.prev_rotation_x = self.prev_rotation_y = self.prev_rotation_z = 0.0
            for spin in [self.spinBox_X, self.spinBox_Y, self.spinBox_Z]:
                spin.blockSignals(True); spin.setValue(0); spin.blockSignals(False)
            for slider in [self.slider_X, self.slider_Y, self.slider_Z]:
                slider.blockSignals(True); slider.setValue(0); slider.blockSignals(False)

            # Run reconstruction
            self._run_reconstruction()

        except Exception as e:
            self.console.appendPlainText(f"❌ Import error: {str(e)}")

    def _load_array(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
        elif ext in (".txt", ".csv"):
            arr = np.loadtxt(path)
        else:
            arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    # =========================
    # Poisson Reconstruction
    # =========================
    def _run_reconstruction(self):
        if self.points is None or self.normals is None:
            return
        try:
            depth       = self.spinBox_depth.value()
            width       = self.doubleSpinBox_width.value()
            scale       = self.doubleSpinBox_scale.value()
            linear_fit  = self.checkBox_linear_fit.isChecked()

            pcd = o3d.geometry.PointCloud()
            pcd.points  = o3d.utility.Vector3dVector(self.points)
            pcd.normals = o3d.utility.Vector3dVector(self.normals)

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=depth,
                width=width,
                scale=scale,
                linear_fit=linear_fit,
            )

            self.mesh_o3d = mesh

            vertices  = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            self.console.appendPlainText(
                f"✅ Reconstruction done | vertices: {len(vertices)} | triangles: {len(triangles)}"
            )
            self.console.appendPlainText(
                f"   depth={depth} | width={width} | scale={scale} | linear_fit={linear_fit}"
            )

            self._render_mesh(vertices, triangles)

        except Exception as e:
            self.console.appendPlainText(f"❌ Reconstruction error: {str(e)}")

    def live_reconstruct(self, *_):
        """Called whenever a Poisson parameter changes — re-runs reconstruction live."""
        if self.points is not None and self.normals is not None:
            self._run_reconstruction()

    # =========================
    # Render VTK Mesh
    # =========================
    def _render_mesh(self, vertices, triangles):
        # Remove old actor
        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        try:
            vtk_points = vtk.vtkPoints()
            for v in vertices:
                vtk_points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

            vtk_cells = vtk.vtkCellArray()
            for tri in triangles:
                vtk_cells.InsertNextCell(3)
                vtk_cells.InsertCellPoint(int(tri[0]))
                vtk_cells.InsertCellPoint(int(tri[1]))
                vtk_cells.InsertCellPoint(int(tri[2]))

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)
            polydata.SetPolys(vtk_cells)

            # Compute normals for smooth shading
            normals_filter = vtk.vtkPolyDataNormals()
            normals_filter.SetInputData(polydata)
            normals_filter.ComputePointNormalsOn()
            normals_filter.SplittingOff()
            normals_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals_filter.GetOutputPort())

            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetColor(0.2, 0.6, 0.9)
            self.actor.GetProperty().SetAmbient(0.1)
            self.actor.GetProperty().SetDiffuse(0.8)
            self.actor.GetProperty().SetSpecular(0.3)
            self.actor.GetProperty().SetSpecularPower(20)

            self.renderer.AddActor(self.actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            self.btn_export.setEnabled(True)

        except Exception as e:
            self.console.appendPlainText(f"❌ Render error: {str(e)}")

    # =========================
    # Export (STL / OBJ)
    # =========================
    def export_data(self):
        if self.mesh_o3d is None:
            self.console.appendPlainText("❌ No mesh to export")
            return

        mesh = self.mesh_o3d

        # --- clean mesh ---
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.main_window,
            "Export Mesh",
            "reconstructed_mesh",
            "STL Files (*.stl);;OBJ Files (*.obj)"
        )

        if not file_path:
            return

        try:
            if "stl" in selected_filter.lower():
                if not file_path.endswith(".stl"):
                    file_path += ".stl"

                success = o3d.io.write_triangle_mesh(file_path, mesh, write_ascii=False)
                self.console.appendPlainText(f"STL export success: {success}")

            else:
                if not file_path.endswith(".obj"):
                    file_path += ".obj"

                success = o3d.io.write_triangle_mesh(file_path, mesh)
                self.console.appendPlainText(f"OBJ export success: {success}")

        except Exception as e:
            self.console.appendPlainText(f"❌ Export error: {str(e)}")