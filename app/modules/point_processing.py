"""Tab 2 - Point cloud processing (import, shuffle, voxel density)."""

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtCore import QTimer

from modules.degradation import Degradation
from modules.geometry_utils import normalize, polydata_from_numpy
from modules.point_data import (
    PointCloudData,
    add_button_after,
    ask_open_point_cloud,
    ask_save_point_cloud,
    make_scrollable,
)
from modules.robustness_dialogs import DegradeDialog
from modules.vtk_controls import VTKViewport, RotationMixin


class Processing(RotationMixin):
    """Import a point cloud (or receive one from the Import tab), reduce
    the density with voxel downsampling, degrade it on purpose (noise,
    holes, ... for robustness tests), shuffle it, then export it or send it
    to the Reconstruction tab.

    Pipeline: original -> voxel -> degradation -> shuffle.  Points and
    normals are always kept as matching pairs.
    """

    VOXEL_DEBOUNCE_MS = 300

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
        self.points = None            # as imported
        self.normals = None           # as imported (None if not provided)
        self.points_changed = None    # after voxel / shuffle
        self.normals_changed = None
        self.shuffled = False
        self.source_name = ""
        self.reference = None
        self.degradation = Degradation()
        self.init_rotation_state()

        # Console + VTK render window + axes
        VTKViewport(self, main_window.frame_console_2, main_window.frame_view_2)

        # ---------- Voxel density (0 = original density) ----------
        self.Point_density_spinbox.setDecimals(3)
        self.Point_density_spinbox.setSingleStep(0.005)
        self.Point_density_spinbox.setMinimum(0.0)
        self.Point_density_spinbox.setMaximum(10.0)
        self.Point_density_spinbox.setValue(0.0)

        # Recomputing on every arrow click is expensive: debounce it.
        self._voxel_timer = QTimer(main_window)
        self._voxel_timer.setSingleShot(True)
        self._voxel_timer.setInterval(self.VOXEL_DEBOUNCE_MS)
        self._voxel_timer.timeout.connect(self.update_voxel_points)
        self.Point_density_spinbox.valueChanged.connect(self._voxel_timer.start)

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_file)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)
        self.random_point_btn.clicked.connect(self.randomize_points)
        self.setup_rotation_controls()

        self.btn_send_reconstruction = add_button_after(
            self.btn_export, "➡ Send to Reconstruction"
        )
        self.btn_send_reconstruction.clicked.connect(self.send_data)

        self.btn_degrade = add_button_after(
            self.random_point_btn, "🧪 Degrade data…"
        )
        self.degrade_dialog = DegradeDialog(main_window)
        self.degrade_dialog.applied.connect(self.set_degradation)
        self.btn_degrade.clicked.connect(self.show_degrade_dialog)

        # The panel grew: let it scroll on small screens
        make_scrollable(self.btn_export.parentWidget())

        # Disable controls until a file is imported
        self._set_controls_enabled(False)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("Program ready ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(
            "📁 Import a point cloud (*.ply, *.pcd, *.txt, *.csv, *.npy) "
            "or send one from the Import mesh tab"
        )

    def _set_controls_enabled(self, enabled):
        for widget in self.rotation_widgets() + (
            self.btn_export, self.btn_send_reconstruction,
            self.Point_density_spinbox, self.random_point_btn,
            self.btn_degrade,
        ):
            widget.setEnabled(enabled)

    # =========================
    # Degradation (robustness tests)
    # =========================
    def show_degrade_dialog(self):
        self.degrade_dialog.set_settings(self.degradation)
        self.degrade_dialog.show()
        self.degrade_dialog.raise_()

    def set_degradation(self, degradation):
        self.degradation = degradation
        self.degrade_dialog.set_settings(degradation)
        self.console.appendPlainText(f"🧪 Data: {degradation.describe()}")
        self.update_voxel_points()

    # =========================
    # Randomize Points
    # =========================
    def randomize_points(self):
        if self.points_changed is None:
            self.console.appendPlainText("❌ No points to shuffle — please import a file")
            return
        self._shuffle()
        self.shuffled = True
        self.console.appendPlainText(f"🔀 Points shuffled ✅ ({len(self.points_changed)})")
        self.render_points(reset_camera=False)

    def _shuffle(self):
        perm = np.random.permutation(len(self.points_changed))
        self.points_changed = self.points_changed[perm]
        if self.normals_changed is not None:
            self.normals_changed = self.normals_changed[perm]

    # =========================
    # Voxel Density
    # =========================
    def update_voxel_points(self):
        if self.points is None:
            return
        voxel_size = self.Point_density_spinbox.value()

        if voxel_size <= 0:
            self.points_changed = self.points.copy()
            self.normals_changed = (
                self.normals.copy() if self.normals is not None else None
            )
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            if self.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(self.normals)
            # Averages the normals per voxel too, keeping them paired
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            self.points_changed = np.asarray(pcd.points).copy()
            self.normals_changed = (
                normalize(np.asarray(pcd.normals)) if pcd.has_normals() else None
            )

        # Deliberate degradation (seeded, so identical on every rebuild)
        self.points_changed, self.normals_changed = self.degradation.apply(
            self.points_changed, self.normals_changed
        )

        # Keep the shuffled state when the density changes
        if self.shuffled:
            self._shuffle()

        label = "original" if voxel_size <= 0 else f"{voxel_size:.3f}"
        extra = (
            "" if self.degradation.is_identity
            else f" | {self.degradation.describe()}"
        )
        self.console.appendPlainText(
            f"✅ Voxel density {label}{extra} → {len(self.points_changed)} points"
        )
        self.render_points(reset_camera=False)

    # =========================
    # Render Point Cloud
    # =========================
    def render_points(self, reset_camera=True):
        """Render ``points_changed`` as VTK vertices and return the actor.

        ``reset_camera`` may be disabled when only the density changes so
        that the current camera position is kept.
        """
        polydata = polydata_from_numpy(self.points_changed, self.normals_changed)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        actor.GetProperty().SetPointSize(2)
        actor.GetProperty().SetInterpolationToPhong()

        return self.show_actor(actor, reset_camera=reset_camera)

    # =========================
    # Load data (file or another tab)
    # =========================
    def receive_data(self, data):
        """Called by the main window when another tab sends a cloud."""
        self.load_data(data, f"Received from Import mesh ✅: {data.source}")

    def load_data(self, data, title):
        self.points = data.points.copy()
        self.normals = data.normals.copy() if data.has_normals else None
        self.points_changed = self.points.copy()
        self.normals_changed = (
            self.normals.copy() if self.normals is not None else None
        )
        self.shuffled = False
        self.source_name = data.source
        self.reference = data.reference   # ground-truth mesh, if known
        self.degradation = Degradation(seed=self.degradation.seed)
        self.degrade_dialog.set_settings(self.degradation)

        self.console.clear()
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(title)
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(f"📊 Points: {len(self.points)}")
        self.console.appendPlainText(
            "📊 Normals: "
            + (str(len(self.normals)) if self.normals is not None
               else "none (will be estimated on export / send)")
        )

        self._set_controls_enabled(True)

        # Reset density without triggering a recompute
        self._voxel_timer.stop()
        self.Point_density_spinbox.blockSignals(True)
        self.Point_density_spinbox.setValue(0.0)
        self.Point_density_spinbox.blockSignals(False)

        self.reset_rotation()
        self.render_points()

    def import_file(self):
        try:
            data = ask_open_point_cloud(
                self.main_window, "Import Point Cloud", normals_required=False
            )
            if data is None:
                return
            self.load_data(data, f"File imported ✅: {data.source}")
        except Exception as e:
            self.console.appendPlainText(f"❌ Error: {str(e)}")

    # =========================
    # Current data (as displayed)
    # =========================
    def current_data(self):
        """Modified points + normals (estimated if missing) with the
        displayed rotation applied."""
        note = "" if self.degradation.is_identity else self.degradation.describe()
        data = PointCloudData(
            self.points_changed, self.normals_changed, self.source_name,
            self.reference, note,
        )
        if not data.has_normals:
            self.console.appendPlainText("⏳ Estimating normals…")
            data = data.with_normals(self.Point_density_spinbox.value())
            self.normals_changed = data.normals.copy()
        return data.transformed(self.actor_matrix())

    # =========================
    # Send / Export
    # =========================
    def send_data(self):
        if self.points_changed is None:
            self.console.appendPlainText("❌ No data to send")
            return
        try:
            data = self.current_data()
            self.main_window.send_point_cloud(data, "reconstruction")
            self.console.appendPlainText(
                f"➡ Sent {len(data)} points + normals to reconstruction"
            )
        except Exception as e:
            self.console.appendPlainText(f"❌ Send error: {str(e)}")

    def export_data(self):
        if self.points_changed is None:
            self.console.appendPlainText("❌ No data to export")
            return
        try:
            paths = ask_save_point_cloud(
                self.main_window, self.current_data(), "point_cloud_changed.ply"
            )
            if not paths:
                return
            self.console.appendPlainText("File Exported ✅ (rotation applied)")
            for path in paths:
                self.console.appendPlainText(path)
        except Exception as e:
            self.console.appendPlainText(f"❌ Export error: {str(e)}")
