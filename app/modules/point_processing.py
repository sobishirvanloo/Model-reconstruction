"""Tab 2 - Point cloud processing (import, shuffle, voxel density)."""

import os

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QStyle

from modules.degradation import Degradation
from modules.geometry_utils import normalize, polydata_from_numpy
from modules.point_data import (
    PointCloudData,
    add_button_after,
    add_section,
    ask_open_point_cloud,
    ask_save_point_cloud,
    insert_widget,
    make_scrollable,
)
from modules.robustness_dialogs import DegradeDialog
from modules.vtk_controls import TabMixin, VTKViewport

POINT_EXTENSIONS = (".ply", ".pcd", ".txt", ".csv", ".npy")


class Processing(TabMixin):
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

        # Console + toolbar + VTK render window + axes
        VTKViewport(
            self, main_window.frame_console_2, main_window.frame_view_2,
            empty_hint="Drop a point cloud here (PLY, PCD, TXT, CSV, NPY)\n"
                       "or press Ctrl+O to open one\n\n"
                       "(or send one from the Import mesh tab)",
        )

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
        self.btn_import.clicked.connect(lambda: self.import_file())
        self.btn_export.clicked.connect(self.export_data)
        self.random_point_btn.clicked.connect(self.randomize_points)
        self.setup_rotation_controls()

        self.btn_send_reconstruction = add_button_after(
            self.btn_export, "Send to Reconstruction"
        )
        self.btn_send_reconstruction.clicked.connect(self.send_data)

        self.btn_degrade = add_button_after(
            self.random_point_btn, "Degrade data…"
        )
        self.degrade_dialog = DegradeDialog(main_window)
        self.degrade_dialog.applied.connect(self.set_degradation)
        self.btn_degrade.clicked.connect(self.show_degrade_dialog)
        self._build_panel()

        # The panel grew: let it scroll on small screens
        make_scrollable(self.btn_export.parentWidget())

        # Disable controls until a file is imported
        self._set_controls_enabled(False)

        self.console.appendPlainText(
            "Ready. Open a point cloud (*.ply, *.pcd, *.txt, *.csv, *.npy) "
            "or send one from the Import mesh tab."
        )

    def _build_panel(self):
        """Section headers, icons, tooltips and the file label."""
        mw = self.main_window
        self.btn_import.setText("Open point cloud…")
        self.decorate(self.btn_import, QStyle.SP_DialogOpenButton,
                      "Open a point cloud: PLY, PCD, TXT, CSV or NPY (Ctrl+O)",
                      "primary")
        self.file_label = QLabel("No data loaded")
        self.file_label.setProperty("role", "hint")
        self.file_label.setWordWrap(True)
        insert_widget(self.btn_import, self.file_label, after=True)
        self.add_remove_button(self.file_label)

        mw.label.setText("Voxel size (0 = original density)")
        self.Point_density_spinbox.setToolTip(
            "Merge all points inside each cube of this size into one point "
            "(their normals are averaged). 0 keeps every point.")
        self.random_point_btn.setText("Shuffle point order")
        self.decorate(self.random_point_btn, QStyle.SP_BrowserReload,
                      "Randomise the order of the points (normals move with them)")
        self.decorate(self.btn_degrade, QStyle.SP_MessageBoxWarning,
                      "Add noise, holes or uneven sampling to test how robust "
                      "the reconstruction is")
        self.btn_export.setText("Export points…")
        self.decorate(self.btn_export, QStyle.SP_DialogSaveButton,
                      "Save points + normals (rotation applied) (Ctrl+S)")
        self.decorate(self.btn_send_reconstruction, QStyle.SP_ArrowForward,
                      "Reconstruct a surface from this data", "primary")

        add_section(self.btn_import, "File")
        add_section(mw.label_x_2, "Rotation")
        add_section(mw.label, "Density & sampling")
        add_section(self.btn_export, "Output")

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
        self._update_info()

    def _update_info(self):
        self.set_tab_info(file=self.source_name, points=len(self.points_changed))
        state = ("" if self.degradation.is_identity
                 else f"\n{self.degradation.describe()}")
        self.file_label.setText(
            f"{self.source_name or 'Received data'}\n"
            f"{len(self.points_changed):,} points"
            + ("" if self.normals_changed is not None else " · no normals")
            + state)

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
        actor.GetProperty().SetPointSize(2)
        actor.GetProperty().SetInterpolationToPhong()

        return self.show_actor(actor, reset_camera=reset_camera, kind="points")

    # =========================
    # Load data (file or another tab)
    # =========================
    def receive_data(self, data):
        """Called by the main window when another tab sends a cloud."""
        self.load_data(data, f"✅ Received from Import mesh: {data.source}")

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
        self.console.appendPlainText(title)
        self.console.appendPlainText(
            f"   {len(self.points):,} points · normals: "
            + ("yes" if self.normals is not None
               else "none (estimated on export / send)")
        )

        self._set_controls_enabled(True)
        self.btn_remove.setEnabled(True)

        # Reset density without triggering a recompute
        self._voxel_timer.stop()
        self.Point_density_spinbox.blockSignals(True)
        self.Point_density_spinbox.setValue(0.0)
        self.Point_density_spinbox.blockSignals(False)

        self.reset_rotation()
        self.render_points()
        self.file_label.setProperty("role", "file")
        self.file_label.style().polish(self.file_label)
        self._update_info()

    def import_file(self, path=None):
        """Open a point cloud (asks with a dialog when ``path`` is None)."""
        try:
            data = ask_open_point_cloud(
                self.main_window, "Open Point Cloud", normals_required=False,
                path=path,
            )
        except Exception as e:
            name = os.path.basename(path) if path else "the file"
            self.notify(f"❌ Could not open {name}: {e}", "error", dialog=True)
            return
        if data is None:
            return
        self.load_data(data, f"✅ Opened {data.source}")
        self.notify(f"Opened {data.source}", "success")

    # =========================
    # Remove data
    # =========================
    def has_data(self):
        return self.points is not None

    def clear_data(self):
        self._voxel_timer.stop()
        self.points = None
        self.normals = None
        self.points_changed = None
        self.normals_changed = None
        self.shuffled = False
        self.source_name = ""
        self.reference = None
        self._set_controls_enabled(False)
        self.clear_view()
        self.console.appendPlainText(
            "Ready. Open a point cloud (*.ply, *.pcd, *.txt, *.csv, *.npy) "
            "or send one from the Import mesh tab."
        )

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
            self.notify(f"❌ Send failed: {e}", "error", dialog=True)

    def export_data(self):
        if self.points_changed is None:
            self.notify("❌ No data to export", "warning")
            return
        try:
            paths = ask_save_point_cloud(
                self.main_window, self.current_data(), "point_cloud_changed.ply"
            )
            if not paths:
                return
            for path in paths:
                self.console.appendPlainText(path)
            self.notify("✅ Exported (rotation applied): "
                        + ", ".join(os.path.basename(p) for p in paths), "success")
        except Exception as e:
            self.notify(f"❌ Export failed: {e}", "error", dialog=True)
