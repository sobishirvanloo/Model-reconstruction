import os
import sys
import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit, QVBoxLayout
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from modules.import_mesh import Geometry

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

        self.processing = None
        self.points = None
        self.normals = None
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

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("Import the Point Cloud File")
        self.console.appendPlainText("=" * 50)

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
        style = vtk.vtkInteractorStyleTrackballActor()  # Mouse rotates actor, not camera
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

        # Connect rotation controls
        self.setup_rotation_controls()


        # Disable controls until file is imported
        for widget in [
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton
        ]:
            widget.setEnabled(False)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("The program ready to use ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("📁 Please import a file: (*.ply, *.obj, *.stl, *.pcd, *.txt)")

        # Update SpinBox/Slider when actor is rotated by mouse
        self.interactor.AddObserver("EndInteractionEvent", self.update_spinbox_from_actor)

    # =========================
    # Axes
    # =========================
    def setup_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1, 1, 1)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
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

    def setup_rotation_controls(self):
        # SpinBoxes
        self.spinBox_X.valueChanged.connect(lambda val: self.update_rotation('X', val))
        self.spinBox_Y.valueChanged.connect(lambda val: self.update_rotation('Y', val))
        self.spinBox_Z.valueChanged.connect(lambda val: self.update_rotation('Z', val))
        # Sliders
        self.slider_X.valueChanged.connect(lambda val: self.update_rotation('X', val))
        self.slider_Y.valueChanged.connect(lambda val: self.update_rotation('Y', val))
        self.slider_Z.valueChanged.connect(lambda val: self.update_rotation('Z', val))

    def update_rotation(self, axis, value):
        if axis == 'X':
            self.rotation_x = float(value)
        elif axis == 'Y':
            self.rotation_y = float(value)
        elif axis == 'Z':
            self.rotation_z = float(value)

    def update_spinbox_from_actor(self, caller=None, event=None):
        """وقتی با موس shape رو تکون بدی - همین مثل Geometry"""
        if self.mesh_actor is None:
            return
        ox, oy, oz = self.mesh_actor.GetOrientation()
        self.rotation_x, self.rotation_y, self.rotation_z = ox, oy, oz

        # Update SpinBoxes and Sliders without triggering signals
        for spin, val in zip([self.spinBox_X, self.spinBox_Y, self.spinBox_Z], [ox, oy, oz]):
            spin.blockSignals(True)
            spin.setValue(int(val))
            spin.blockSignals(False)

        for slider, val in zip([self.slider_X, self.slider_Y, self.slider_Z], [ox, oy, oz]):
            slider.blockSignals(True)
            slider.setValue(int(val))
            slider.blockSignals(False)

        # Update previous rotation
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
                    
                    # ✅ ذخیره‌سازی داده‌ها (بدون نیاز به Geometry)
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
            self.console.appendPlainText(f"📊 Normals: {len(self.normals)}\n")

            # Enable rotation controls
            for widget in [
                self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
                self.slider_X, self.slider_Y, self.slider_Z,
                self.pushButton, self.btn_export
            ]:
                widget.setEnabled(True)

            # Render mesh
            self.mesh_actor = self.render_processing()

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
    # Export Point Cloud with Normals
    # =========================
    def export_data(self):
        """Export point cloud points and normals to separate files"""
        if self.points is None:
            self.console.appendPlainText("❌ No data to export")
            return
        
        # Save points
        file_path_points, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Points", "points.txt", "Text Files (*.txt)"
        )
        if not file_path_points:
            return
        
        # Save normals
        file_path_normals, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Normals", "normals.txt", "Text Files (*.txt)"
        )
        if not file_path_normals:
            return
        
        try:
            np.savetxt(file_path_points, self.points, fmt="%.6f")
            np.savetxt(file_path_normals, self.normals, fmt="%.6f")
            
            self.console.appendPlainText(f"File Exported ✅")
            self.console.appendPlainText(f"{file_path_points}")
            self.console.appendPlainText(f"{file_path_normals}")
            
        except Exception as e:
            self.console.appendPlainText(f"❌ Export error: {str(e)}")

            
    # =========================
    # Render Geometry
    # =========================
    def render_processing(self):
        """Render point cloud and return the actor"""
        if self.processing is None:
            self.console.appendPlainText("[ERROR] Geometry is None")
            return None

        # Remove previous actor
        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        try:
            # ایجاد VTK Points
            vtk_points = vtk.vtkPoints()
            vtk_normals = vtk.vtkFloatArray()
            vtk_normals.SetNumberOfComponents(3)
            vtk_normals.SetName("Normals")

            for p, n in zip(self.points, self.normals):
                vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
                vtk_normals.InsertNextTuple((float(n[0]), float(n[1]), float(n[2])))

            # ایجاد PolyData
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)
            polydata.GetPointData().SetNormals(vtk_normals)

            # برای Point Cloud از Sphere Glyph استفاده کن
            if self.is_pointcloud:
                sphere = vtk.vtkSphereSource()
                sphere.SetRadius(0.02)  # اندازه کره‌ها
                
                glyph = vtk.vtkGlyph3D()
                glyph.SetInputData(polydata)
                glyph.SetSourceConnection(sphere.GetOutputPort())
                glyph.Update()
                polydata = glyph.GetOutput()
            else:
                # برای Mesh، فقط VertexGlyph استفاده کن
                glyph = vtk.vtkVertexGlyphFilter()
                glyph.SetInputData(polydata)
                glyph.Update()
                polydata = glyph.GetOutput()

            # ایجاد Mapper و Actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetColor(0.0, 0.7, 1.0)  # آبی روشن
            self.actor.GetProperty().SetPointSize(5)
            self.actor.GetProperty().EdgeVisibilityOff()

            # Add to renderer
            self.renderer.AddActor(self.actor)
            self.renderer.ResetCamera()

            # Initialize VTK
            if not self.vtk_initialized:
                self.interactor.Initialize()
                self.vtk_initialized = True

            self.interactor.GetRenderWindow().Render()
            self.console.appendPlainText(f"✅ Point cloud rendered: {len(self.points)} points")

            return self.actor
            
        except Exception as e:
            self.console.appendPlainText(f"❌ Rendering error: {str(e)}")
            return None