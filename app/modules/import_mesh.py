import os
import sys
import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit, QVBoxLayout
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

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



class Geometry:
    def __init__(self, main_window):
        self.main_window = main_window
        self.frame_console = main_window.frame_console
        self.frame_view = main_window.frame_view
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
        self.vtk_initialized = False

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
        self.console.appendPlainText("📁 Please import a file: (*.ply, *.obj, *.stl, *.pcd)")

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
        """Render geometry and return the actor"""
        if self.geometry is None:
            print("[ERROR] Geometry is None in render_geometry()")
            return None

        # Remove previous actor
        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        # ========== ساخت VTK Points ==========
        vtk_points = vtk.vtkPoints()
        vtk_normals = vtk.vtkFloatArray()
        vtk_normals.SetNumberOfComponents(3)
        vtk_normals.SetName("Normals")

        for p, n in zip(self.points, self.normals):
            vtk_points.InsertNextPoint(p[0], p[1], p[2])
            vtk_normals.InsertNextTuple((n[0], n[1], n[2]))

        # ========== ساخت PolyData ==========
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetNormals(vtk_normals)

        # اضافه کردن cells (فقط برای Mesh، نه PointCloud)
        if not self.is_pointcloud:
            vtk_cells = vtk.vtkCellArray()
            for tri in np.asarray(self.geometry.triangles):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, tri[0])
                triangle.GetPointIds().SetId(1, tri[1])
                triangle.GetPointIds().SetId(2, tri[2])
                vtk_cells.InsertNextCell(triangle)
            polydata.SetPolys(vtk_cells)

        # PointCloud رو به Glyphs تبدیل کن
        if self.is_pointcloud:
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            polydata = glyph.GetOutput()

        # ========== ساخت Mapper و Actor ==========
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        self.actor.GetProperty().SetPointSize(2)
        self.actor.GetProperty().SetInterpolationToPhong()

        # اضافه کردن به Renderer
        self.renderer.AddActor(self.actor)
        self.renderer.ResetCamera()

        # Initialize VTK
        if not self.vtk_initialized:
            self.interactor.Initialize()
            self.vtk_initialized = True

        self.interactor.GetRenderWindow().Render()

        print(f"[DEBUG] Actor created and returned: {self.actor}")
        return self.actor

    # =========================
    # Import File (UI)
    # =========================
    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Geometry", "", "3D Files (*.ply *.obj *.stl *.pcd)"
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
            self.console.appendPlainText(f"File Exported ✅")
            self.console.appendPlainText(f"{file_path_points}")
            self.console.appendPlainText(f"{file_path_normals}")
        except Exception as e:
            self.console.appendPlainText(f"❌ Error in export: {str(e)}")