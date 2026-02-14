import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPlainTextEdit, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from modules.geometry_handler import GeometryHandler


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/main.ui", self)

        # ---------- Console ----------
        self.console = QPlainTextEdit(self.frame_console)
        self.console.setReadOnly(True)
        self.console.setGeometry(self.frame_console.rect())
        self.console.setStyleSheet(
            "background-color: #007ea7; color: white; font-family: Consolas; font-size: 12px;"
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
        style = vtk.vtkInteractorStyleTrackballActor()  # Mouse rotates actor, not camera
        self.interactor.SetInteractorStyle(style)

        # Setup axes
        self.setup_axes()

        # Initial render
        self.vtk_widget.GetRenderWindow().Render()

        # Geometry handler
        self.handler = GeometryHandler()

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

    # ---------- Axes ----------
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

    # ---------- Rotation Controls ----------
    def setup_rotation_controls(self):
        # SpinBoxes
        self.spinBox_X.valueChanged.connect(lambda val: self.update_rotation('X', val))
        self.spinBox_Y.valueChanged.connect(lambda val: self.update_rotation('Y', val))
        self.spinBox_Z.valueChanged.connect(lambda val: self.update_rotation('Z', val))
        # Sliders
        self.slider_X.valueChanged.connect(lambda val: self.update_rotation('X', val))
        self.slider_Y.valueChanged.connect(lambda val: self.update_rotation('Y', val))
        self.slider_Z.valueChanged.connect(lambda val: self.update_rotation('Z', val))

    # ---------- Update rotation ----------
    def update_rotation(self, axis, value):
        if axis == 'X':
            self.rotation_x = float(value)
        elif axis == 'Y':
            self.rotation_y = float(value)
        elif axis == 'Z':
            self.rotation_z = float(value)

    # ---------- Update SpinBox/Slider from actor ----------
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

    # ---------- Apply rotation ----------
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

    # ---------- Import file ----------
    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Geometry", "", "3D Files (*.ply *.obj *.stl *.pcd)"
        )
        if not file_path:
            return
        try:
            self.handler.import_geometry(file_path)
            self.console.clear()
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"File imported ✅: {file_path}")
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"📊 Points: {len(self.handler.points)}")
            self.console.appendPlainText(f"📊 Normals: {len(self.handler.normals)}\n")

            # Enable rotation controls
            for widget in [
                self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
                self.slider_X, self.slider_Y, self.slider_Z,
                self.pushButton, self.btn_export
            ]:
                widget.setEnabled(True)

            # Render mesh
            self.mesh_actor = self.handler.render(self.renderer, self.interactor)

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

    # ---------- Export Data ----------
    def export_data(self):
        file_path_points, _ = QFileDialog.getSaveFileName(
            self, "Save Points", "points.txt", "Text Files (*.txt)"
        )
        file_path_normals, _ = QFileDialog.getSaveFileName(
            self, "Save Normals", "normals.txt", "Text Files (*.txt)"
        )
        if not file_path_points or not file_path_normals:
            return
        try:
            self.handler.export_geometry(file_path_points, file_path_normals)
            self.console.appendPlainText(f"File Exported ✅")
            self.console.appendPlainText(f"{file_path_points}")
            self.console.appendPlainText(f"{file_path_normals}")
        except Exception as e:
            self.console.appendPlainText(f"❌ Error in export: {str(e)}")


# ---------- Main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
