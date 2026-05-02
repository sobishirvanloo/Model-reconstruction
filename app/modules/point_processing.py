import os
import sys
import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit, QVBoxLayout
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from modules.import_mesh import Geometry


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


class Processing:
    def __init__(self, main_window):
        self.main_window = main_window
        self.frame_console = main_window.frame_console_2
        self.frame_view = main_window.frame_view_2
        self.btn_import = main_window.btn_import_2
        self.btn_export = main_window.btn_export_2
        self.pushButton = main_window.pushButton_2

        self.spinBox_X = main_window.spinBox_X_2
        self.spinBox_Y = main_window.spinBox_Y_2
        self.spinBox_Z = main_window.spinBox_Z_2
        self.slider_X = main_window.slider_X_2
        self.slider_Y = main_window.slider_Y_2
        self.slider_Z = main_window.slider_Z_2
        self.random_point_btn = main_window.random_point_btn

        self.random_point_btn.clicked.connect(self.randomize_points)
        self.Point_density_spinbox = main_window.Point_density_spinbox
        self.current_voxel_size = 0.05  # default voxel size

        self.processing = None
        self.points = None
        self.points_changed = None
        self.points_down = None
        self.is_pointcloud = False
        self.actor = None
        self.vtk_initialized = False

        # ---------- Console ----------
        self.console = QPlainTextEdit(self.frame_console)
        self.console.setReadOnly(True)
        self.console.setGeometry(self.frame_console.rect())
        self.console.setStyleSheet(
            "background-color: #007ea7; color: white; font-family: Consolas; font-size: 16px;"
        )
        console_layout = QVBoxLayout(self.frame_console)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.addWidget(self.console)
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

        # ---------- VTK Initialization ----------
        self.interactor.Initialize()
        style = CustomInteractorStyle(self.renderer)
        self.interactor.SetInteractorStyle(style)

        # Setup axes
        self.setup_axes()

        # Initial render
        self.vtk_widget.GetRenderWindow().Render()

        # ---------- Rotation ----------
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prev_rotation_x = 0.0
        self.prev_rotation_y = 0.0
        self.prev_rotation_z = 0.0

        # Mesh actor
        self.mesh_actor = None

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_file)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)
        self.btn_export.setEnabled(False)

        # ---------- Setup rotation controls ----------
        self.setup_rotation_controls()

        # ---------- Setup point density controls ----------
        self.setup_density_controls()

        # Disable controls until file is imported
        for widget in [
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton, self.btn_export, self.Point_density_spinbox
        ]:
            widget.setEnabled(False)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("Program ready ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("📁 Please import a file: (*.txt)")

        # Update SpinBox/Slider when actor is rotated by mouse
        self.interactor.AddObserver("EndInteractionEvent", self.update_spinbox_from_actor)

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
        # ---------- X ----------
        self.slider_X.valueChanged.connect(
            lambda val: self.sync_rotation(self.spinBox_X, 'X', val)
        )
        self.spinBox_X.valueChanged.connect(
            lambda val: self.sync_rotation(self.slider_X, 'X', val)
        )
        # ---------- Y ----------
        self.slider_Y.valueChanged.connect(
            lambda val: self.sync_rotation(self.spinBox_Y, 'Y', val)
        )
        self.spinBox_Y.valueChanged.connect(
            lambda val: self.sync_rotation(self.slider_Y, 'Y', val)
        )
        # ---------- Z ----------
        self.slider_Z.valueChanged.connect(
            lambda val: self.sync_rotation(self.spinBox_Z, 'Z', val)
        )
        self.spinBox_Z.valueChanged.connect(
            lambda val: self.sync_rotation(self.slider_Z, 'Z', val)
        )

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

    def update_rotation(self, axis, value):
        if axis == 'X':
            self.rotation_x = float(value)
            self.spinBox_X.blockSignals(True)
            self.slider_X.blockSignals(True)
            self.spinBox_X.setValue(int(value))
            self.slider_X.setValue(int(value))
            self.spinBox_X.blockSignals(False)
            self.slider_X.blockSignals(False)
        elif axis == 'Y':
            self.rotation_y = float(value)
            self.spinBox_Y.blockSignals(True)
            self.slider_Y.blockSignals(True)
            self.spinBox_Y.setValue(int(value))
            self.slider_Y.setValue(int(value))
            self.spinBox_Y.blockSignals(False)
            self.slider_Y.blockSignals(False)
        elif axis == 'Z':
            self.rotation_z = float(value)
            self.spinBox_Z.blockSignals(True)
            self.slider_Z.blockSignals(True)
            self.spinBox_Z.setValue(int(value))
            self.slider_Z.setValue(int(value))
            self.spinBox_Z.blockSignals(False)
            self.slider_Z.blockSignals(False)

    def update_spinbox_from_actor(self, caller=None, event=None):
        if self.mesh_actor is None:
            return
        ox, oy, oz = self.mesh_actor.GetOrientation()
        self.rotation_x, self.rotation_y, self.rotation_z = ox, oy, oz

        for spin, val in zip([self.spinBox_X, self.spinBox_Y, self.spinBox_Z], [ox, oy, oz]):
            spin.blockSignals(True)
            spin.setValue(int(val))
            spin.blockSignals(False)

        for slider, val in zip([self.slider_X, self.slider_Y, self.slider_Z], [ox, oy, oz]):
            slider.blockSignals(True)
            slider.setValue(int(val))
            slider.blockSignals(False)

        self.prev_rotation_x, self.prev_rotation_y, self.prev_rotation_z = ox, oy, oz

    def apply_rotation(self):
        if self.mesh_actor is None:
            self.console.appendPlainText("❌ Please import a file first")
            return
        try:
            delta_x = self.rotation_x - self.prev_rotation_x
            delta_y = self.rotation_y - self.prev_rotation_y
            delta_z = self.rotation_z - self.prev_rotation_z

            self.mesh_actor.RotateX(delta_x)
            self.mesh_actor.RotateY(delta_y)
            self.mesh_actor.RotateZ(delta_z)

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
    # Shuffle Points
    # =========================
    def randomize_points(self):
        if self.points_changed is None:
            self.points_changed = self.points.copy()
    
        # Shuffle the rows of points_changed
        np.random.shuffle(self.points_changed)
    
        # Re-render updated points
        self.render_processing_with_points(self.points_changed)
    
        self.console.appendPlainText(f"🎲 Points shuffled | total points: {len(self.points_changed)}")
    # =========================
    # Point Density Controls
    # =========================
    # ---------- Setup density controls ----------
    def setup_density_controls(self):
        # Only SpinBox
        self.Point_density_spinbox.valueChanged.connect(self.sync_density_from_spinbox)

    def sync_density_from_spinbox(self, val):
        self.current_voxel_size = float(val)  # voxel size directly
        # Live update
        if self.points is not None:
            self.update_voxel_points()

    def update_voxel_points(self):
        try:
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(self.points)
            pc_down = pc_o3d.voxel_down_sample(self.current_voxel_size)
            self.points_down = np.asarray(pc_down.points)
            self.render_processing_with_points(self.points_down)
            self.points_changed = np.asarray(pc_down.points) 
            self.console.appendPlainText(
                f"✅ Point cloud updated | voxel_size: {self.current_voxel_size:.3f} | points: {len(self.points_down)}"
            )
        except Exception as e:
            self.console.appendPlainText(f"❌ Error updating density: {str(e)}")

    # =========================
    # Render VTK with points
    # =========================
    def render_processing_with_points(self, points_array):
        if points_array is None:
            return None

        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        try:
            vtk_points = vtk.vtkPoints()
            for p in points_array:
                vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)

            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(0.02)
            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(polydata)
            glyph.SetSourceConnection(sphere.GetOutputPort())
            glyph.Update()
            polydata = glyph.GetOutput()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetColor(0.0, 0.7, 1.0)
            self.actor.GetProperty().SetPointSize(5)
            self.actor.GetProperty().EdgeVisibilityOff()

            self.renderer.AddActor(self.actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            return self.actor
        except Exception as e:
            self.console.appendPlainText(f"❌ Rendering error: {str(e)}")
            return None

    # =========================
    # Import Point Cloud
    # =========================
    def import_point(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] >= 3:
                    self.points = data[:, :3]
                    if data.shape[1] >= 6:
                        self.normals = data[:, 3:6]
                    else:
                        self.normals = np.zeros_like(self.points)
                    
                    class SimpleGeometry:
                        def __init__(self):
                            self.points = None
                            self.normals = None
                            self.triangles = []

                    self.processing = SimpleGeometry()
                    self.processing.points = self.points
                    self.processing.normals = self.normals
                    self.processing.triangles = []
                    self.is_pointcloud = True
                    
                    return True
                else:
                    self.console.appendPlainText("❌ TXT file must have at least 3 columns (x, y, z)")
                    return False
            except Exception as e:
                self.console.appendPlainText(f"❌ Error reading txt: {str(e)}")
                return False
        else:
            self.console.appendPlainText("❌ Please import a txt file for point cloud")
            return False

    # =========================
    # Import Point Cloud (UI)
    # =========================
    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Point Cloud", "", "3D Files (*.ply *.obj *.stl *.pcd *.txt)"
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

            # Render mesh initially
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
        if self.points_changed  is None:
            self.console.appendPlainText("❌ No data to export")
            return

        file_path_points, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Points", "points_changed.txt", "Text Files (*.txt)"
        )
        if not file_path_points:
            return


        try:
            np.savetxt(file_path_points, self.points_changed , fmt="%.6f")
            self.console.appendPlainText(f"File Exported ✅")
            self.console.appendPlainText(f"{file_path_points}")
        except Exception as e:
            self.console.appendPlainText(f"❌ Export error: {str(e)}")