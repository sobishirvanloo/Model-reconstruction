"""Tab 3 - Poisson surface reconstruction (Open3D) with VTK rendering.

The reconstruction pipeline follows ``models/test.py``:

    optional voxel downsampling
    -> statistical outlier removal
    -> consistent normal orientation
    -> Poisson reconstruction
    -> density filtering
    -> optional crop to the point-cloud bounding box
    -> optional Taubin smoothing

The heavy work runs on a background ``QThread`` so the UI stays responsive
while parameters are being tuned (live reconstruction).
"""

import os

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from modules.vtk_controls import VTKViewport, RotationMixin


class ReconstructionWorker(QThread):
    """Runs the Open3D reconstruction off the UI thread.

    Emits ``finished(vertices, triangles)`` on success or
    ``failed(error_message)`` on failure.
    """

    finished = pyqtSignal(object, object, object)  # (vertices, triangles, info)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job = None

    def set_job(self, job):
        self._job = job

    def run(self):
        try:
            result = self._job()
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished.emit(result[0], result[1], result[2])


class PoissonReconstruction(RotationMixin):
    """Import points + normals, reconstruct a mesh with Poisson and export it."""

    NO_ACTOR_MESSAGE = "❌ No mesh loaded to rotate"

    def __init__(self, main_window):
        self.main_window = main_window

        # ---------- UI Elements ----------
        self.btn_import = main_window.btn_import_pointsandnormals
        self.btn_export = main_window.btn_export_6
        self.pushButton = main_window.pushButton_6

        # Rotation controls
        self.spinBox_X = main_window.spinBox_X_6
        self.spinBox_Y = main_window.spinBox_Y_6
        self.spinBox_Z = main_window.spinBox_Z_6
        self.slider_X = main_window.slider_X_6
        self.slider_Y = main_window.slider_Y_6
        self.slider_Z = main_window.slider_Z_6

        # Poisson parameters
        self.spinBox_depth = main_window.spinBox          # depth (max 12)
        self.doubleSpinBox_width = main_window.doubleSpinBox   # width
        self.doubleSpinBox_scale = main_window.doubleSpinBox_2  # scale
        self.checkBox_linear_fit = main_window.Linear_fit

        # Pipeline parameters
        self.checkBox_preprocess = main_window.checkBox_preprocess
        self.doubleSpinBox_voxel = main_window.doubleSpinBox_voxel
        self.doubleSpinBox_density = main_window.doubleSpinBox_density
        self.checkBox_crop = main_window.checkBox_crop
        self.spinBox_smooth = main_window.spinBox_smooth

        # ---------- State ----------
        self.points = None
        self.normals = None
        self.mesh_o3d = None
        self.actor = None
        self.mesh_actor = None
        self.vtk_initialized = False
        self._worker = None
        self._pending_run = False

        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prev_rotation_x = 0.0
        self.prev_rotation_y = 0.0
        self.prev_rotation_z = 0.0

        # Console + VTK render window + axes
        VTKViewport(self, main_window.frame_console_6, main_window.frame_view_6)

        # ---------- Configure parameter widgets ----------
        self.spinBox_depth.setMinimum(1)
        self.spinBox_depth.setMaximum(12)
        self.spinBox_depth.setValue(10)

        self.doubleSpinBox_width.setMinimum(0.0)
        self.doubleSpinBox_width.setMaximum(100.0)
        self.doubleSpinBox_width.setSingleStep(0.5)
        self.doubleSpinBox_width.setValue(0.0)          # 0 = auto

        self.doubleSpinBox_scale.setDecimals(2)
        self.doubleSpinBox_scale.setMinimum(0.1)
        self.doubleSpinBox_scale.setMaximum(10.0)
        self.doubleSpinBox_scale.setSingleStep(0.1)
        self.doubleSpinBox_scale.setValue(1.1)
        self.checkBox_linear_fit.setChecked(False)

        # Preprocessing / cleaning
        self.checkBox_preprocess.setChecked(True)
        self.doubleSpinBox_voxel.setDecimals(4)
        self.doubleSpinBox_voxel.setMinimum(0.0)
        self.doubleSpinBox_voxel.setMaximum(10.0)
        self.doubleSpinBox_voxel.setSingleStep(0.01)
        self.doubleSpinBox_voxel.setValue(0.0)          # 0 = off

        self.doubleSpinBox_density.setDecimals(1)
        self.doubleSpinBox_density.setMinimum(0.0)
        self.doubleSpinBox_density.setMaximum(50.0)
        self.doubleSpinBox_density.setSingleStep(0.5)
        self.doubleSpinBox_density.setValue(2.0)        # quantile %

        self.checkBox_crop.setChecked(True)

        self.spinBox_smooth.setMinimum(0)
        self.spinBox_smooth.setMaximum(100)
        self.spinBox_smooth.setValue(5)                 # 0 = off

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_files)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)
        self.btn_export.setEnabled(False)

        self.setup_rotation_controls()
        self._connect_live_controls()

        self._set_controls_enabled(False)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("Poisson Reconstruction ready ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(
            "📁 Please import Points and Normals files (.txt / .npy / .csv)"
        )

        # Sync the UI after a mouse rotation (matrix based - more accurate)
        self.interactor.AddObserver(
            "EndInteractionEvent", self.update_spinbox_from_matrix
        )

    # =========================
    # Signal wiring / enable state
    # =========================
    def _connect_live_controls(self):
        """Re-run the reconstruction whenever a parameter changes."""
        for widget in (
            self.spinBox_depth,
            self.doubleSpinBox_width,
            self.doubleSpinBox_scale,
            self.doubleSpinBox_voxel,
            self.doubleSpinBox_density,
            self.spinBox_smooth,
        ):
            widget.valueChanged.connect(self.live_reconstruct)

        for widget in (
            self.checkBox_linear_fit,
            self.checkBox_preprocess,
            self.checkBox_crop,
        ):
            widget.toggled.connect(self.live_reconstruct)

    def _set_controls_enabled(self, enabled):
        for widget in (
            self.spinBox_X, self.spinBox_Y, self.spinBox_Z,
            self.slider_X, self.slider_Y, self.slider_Z,
            self.pushButton,
        ):
            widget.setEnabled(enabled)

    # =========================
    # Import Files
    # =========================
    def import_files(self):
        """Prompt for a points file and a normals file, then reconstruct."""
        # --- Import Points ---
        points_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Points File", "",
            "Point Files (*.txt *.npy *.csv)"
        )
        if not points_path:
            self.console.appendPlainText(
                "❌ Import cancelled (points file not selected)"
            )
            return

        # --- Import Normals ---
        normals_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Normals File", "",
            "Normal Files (*.txt *.npy *.csv)"
        )
        if not normals_path:
            self.console.appendPlainText(
                "❌ Import cancelled (normals file not selected)"
            )
            return

        try:
            self.points = self._load_array(points_path)
            self.normals = self._load_array(normals_path)

            if self.points.shape[0] != self.normals.shape[0]:
                self.console.appendPlainText(
                    f"❌ Mismatch: {self.points.shape[0]} points vs "
                    f"{self.normals.shape[0]} normals"
                )
                return

            self.console.clear()
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText("Files imported ✅")
            self.console.appendPlainText("=" * 50)
            self.console.appendPlainText(f"📊 Points:  {self.points.shape[0]}")
            self.console.appendPlainText(f"📊 Normals: {self.normals.shape[0]}")

            self._set_controls_enabled(True)
            self.btn_export.setEnabled(False)

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

            # Start the reconstruction in the background
            self.live_reconstruct()

        except Exception as e:
            self.console.appendPlainText(f"❌ Import error: {str(e)}")

    # =========================
    # Preprocessing (mirrors models/test.py)
    # =========================
    def _preprocess(self, points, normals, preprocess, voxel_size):
        """Normalize normals, optionally voxel-downsample, remove outliers
        and orient the normals consistently.

        This runs on the background thread, so every option is passed in as
        a plain value instead of reading the UI.
        """
        # Normalize normals
        normals = np.asarray(normals, dtype=float)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        if not preprocess:
            return points, normals

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Optional voxel downsampling (also averages the normals)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)

        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Re-estimate normals when they were lost (downsampling / filtering)
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(voxel_size * 4.0, 0.05),
                    max_nn=30,
                )
            )

        # Consistent orientation (best general choice)
        pcd.orient_normals_consistent_tangent_plane(k=30)

        return np.asarray(pcd.points), np.asarray(pcd.normals)

    # =========================
    # Poisson Reconstruction (heavy work, runs in the worker thread)
    # =========================
    def _build_mesh(self, params, points, normals):
        """Full pipeline -> (vertices, triangles)."""
        depth = params["depth"]
        width = params["width"]
        scale = params["scale"]
        linear_fit = params["linear_fit"]

        points, normals = self._preprocess(
            points, normals, params["preprocess"], params["voxel"]
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )

        # --- density filtering ---
        densities = np.asarray(densities)
        q = params["density_q"]
        if q > 0:
            if len(densities) == len(mesh.vertices):
                threshold = np.quantile(densities, q)
                mesh.remove_vertices_by_mask(densities < threshold)

        # --- crop to the point cloud bounding box ---
        if params["crop"]:
            mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

        # --- smoothing ---
        if params["smooth"] > 0:
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=params["smooth"]
            )

        mesh.compute_vertex_normals()
        self.mesh_o3d = mesh
        info = {"points_after_preprocess": len(points)}
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles), info

    # =========================
    # Loaders
    # =========================
    def _load_array(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
        else:  # .txt / .csv
            arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.asarray(arr, dtype=float)
# =========================
    # Live Reconstruction (background thread, debounced)
    # =========================
    def live_reconstruct(self, *_):
        """Called whenever a Poisson parameter changes."""
        if self.points is None or self.normals is None:
            return
        self._restart_reconstruction()

    def _restart_reconstruction(self):
        if self._worker is not None and self._worker.isRunning():
            # A reconstruction is running: mark it and it will be re-run
            # automatically as soon as the current job finishes.
            self._pending_run = True
            return
        self._start_worker()

    def _start_worker(self):
        self._pending_run = False
        self.console.appendPlainText("⏳ Reconstructing…")
        self.btn_export.setEnabled(False)

        # Snapshot parameters/inputs on the UI thread, pass copies to the worker
        params = {
            "depth": self.spinBox_depth.value(),
            "width": self.doubleSpinBox_width.value(),
            "scale": self.doubleSpinBox_scale.value(),
            "linear_fit": self.checkBox_linear_fit.isChecked(),
            "preprocess": self.checkBox_preprocess.isChecked(),
            "voxel": self.doubleSpinBox_voxel.value(),
            "density_q": self.doubleSpinBox_density.value() / 100.0,
            "crop": self.checkBox_crop.isChecked(),
            "smooth": self.spinBox_smooth.value(),
        }
        job = lambda: self._build_mesh(
            params,
            self.points.copy(),
            self.normals.copy(),
        )

        self._worker = ReconstructionWorker(self.vtk_widget)
        self._worker.set_job(job)
        self._worker.finished.connect(self._on_reconstruction_done)
        self._worker.failed.connect(self._on_reconstruction_failed)
        self._worker.start()

    def _on_reconstruction_done(self, vertices, triangles, info):
        pre_count = info.get("points_after_preprocess", "?")
        self.console.appendPlainText(
            f"   Points after preprocessing: {pre_count}"
        )
        self._render_mesh(vertices, triangles)
        if self._worker is not None and not self._worker.isRunning():
            self._worker.deleteLater()
            self._worker = None
        if self._pending_run:
            self._start_worker()

    def _on_reconstruction_failed(self, message):
        self.console.appendPlainText(f"❌ Reconstruction error: {message}")
        if self._worker is not None and not self._worker.isRunning():
            self._worker.deleteLater()
            self._worker = None

    # =========================
    # Background thread result -> VTK render (main thread)
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

            if not self.vtk_initialized:
                self.interactor.Initialize()
                self.vtk_initialized = True

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

            self.mesh_actor = self.actor
            n_v, n_t = len(vertices), len(triangles)
            self.console.appendPlainText(
                f"✅ Reconstruction done | vertices: {n_v} | triangles: {n_t}"
            )
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
            self.main_window, "Export Mesh", "reconstructed_mesh",
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