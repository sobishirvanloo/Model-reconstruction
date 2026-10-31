"""Tab 3 - Surface reconstruction (Open3D) with VTK rendering.

Three algorithms can be compared on the same data:

* **Poisson** - fits a smooth, watertight implicit surface
  (pipeline of ``models/test.py``);
* **Ball Pivoting** - rolls balls of radius r, 2r, 4r over the points
  and keeps the triangles they touch (interpolates the points);
* **Alpha Shape** - keeps Delaunay triangles whose circumsphere is
  smaller than alpha.

Ball radius and alpha are given as a multiple of the mean point spacing,
so the same setting works whatever the model size.

    optional voxel downsampling
    -> statistical outlier removal
    -> consistent normal orientation (global sign kept from the input)
    -> reconstruction (Poisson / Ball Pivoting / Alpha Shape)
    -> Poisson only: density filtering + optional crop to the bounding box
    -> optional Taubin smoothing
    -> quality evaluation (see ``quality_metrics``)

The heavy work runs on a background ``QThread`` so the UI stays responsive
while parameters are being tuned (live reconstruction, debounced).
"""

import hashlib
import math
import os
import time
from collections import OrderedDict

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QFileDialog, QLabel
from vtk.util import numpy_support

from modules.geometry_utils import (
    normalize,
    orient_consistently,
    polydata_from_numpy,
)
from modules.point_data import (
    add_button_after,
    ask_open_point_cloud,
    insert_widget,
    make_scrollable,
)
from modules.quality_metrics import evaluate
from modules.quality_report import ALGORITHM_NAMES, QualityReportDialog
from modules.vtk_controls import VTKViewport, RotationMixin


MAX_DEPTH = 12
MESH_FILTER = "Meshes (*.stl *.obj *.ply *.off *.gltf *.glb)"

POISSON = "poisson"
BALL_PIVOTING = "ball_pivoting"
ALPHA_SHAPE = "alpha_shape"
ALGORITHM_LABELS = ALGORITHM_NAMES   # display names, keyed by the ids above
# Default size factors (x mean point spacing), chosen on the Stanford bunny:
# BPA x1 was 7x slower with 48 pieces, x2 gave F 0.95; alpha x8-16 was best.
FACTOR_LABELS = {
    BALL_PIVOTING: "Ball radius (× point spacing)",
    ALPHA_SHAPE: "Alpha (× point spacing)",
}
DEFAULT_FACTORS = {BALL_PIVOTING: 2.0, ALPHA_SHAPE: 10.0}


class ReconstructionWorker(QThread):
    """Runs a reconstruction / evaluation job off the UI thread.

    Emits ``succeeded(result)`` or ``failed(error_message)``; the built-in
    ``QThread.finished`` fires afterwards and is used for cleanup.
    """

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self):
        try:
            result = self._job()
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.succeeded.emit(result)


# =========================
# Pipeline (pure functions - safe to run in the worker thread)
# =========================
def preprocess(points, normals, enabled, voxel_size):
    """Normalize normals, optionally voxel-downsample, remove outliers
    and orient the normals consistently."""
    normals = normalize(normals)
    if not enabled:
        return points, normals

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Optional voxel downsampling (also averages the normals)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.normals = o3d.utility.Vector3dVector(
            normalize(np.asarray(pcd.normals))
        )

    # Statistical outlier removal (keeps the matching normals)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Consistent orientation, without flipping the surface inside out
    reference = np.asarray(pcd.normals).copy()
    orient_consistently(pcd, reference)

    return np.asarray(pcd.points), np.asarray(pcd.normals)


def array_key(*arrays):
    """Content hash of numpy arrays (cache key)."""
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


# Preprocessing (outlier removal + normal orientation) is the slowest stage
# (~3 s on the bunny) but depends only on the input and two options, not on
# the reconstruction parameters: keep the last few results.  Only the
# worker thread touches the caches, and jobs run one at a time.
PREPROCESS_CACHE_SIZE = 4
_preprocess_cache = OrderedDict()


def cached_preprocess(points, normals, enabled, voxel_size):
    """``preprocess`` with an LRU cache -> (points, normals, was_cached)."""
    key = (array_key(points, normals), bool(enabled),
           float(voxel_size) if enabled else 0.0)
    if key in _preprocess_cache:
        _preprocess_cache.move_to_end(key)
        pre_points, pre_normals = _preprocess_cache[key]
        return pre_points, pre_normals, True

    pre_points, pre_normals = preprocess(points, normals, enabled, voxel_size)
    _preprocess_cache[key] = (np.array(pre_points), np.array(pre_normals))
    while len(_preprocess_cache) > PREPROCESS_CACHE_SIZE:
        _preprocess_cache.popitem(last=False)
    pre_points, pre_normals = _preprocess_cache[key]
    return pre_points, pre_normals, False


def depth_for_width(points, width, scale):
    """Octree depth whose finest cell is ~``width`` (PoissonRecon's rule).

    Open3D ignores ``width`` whenever ``depth`` is given, so the conversion
    is done here to make the *Width* control meaningful.
    """
    extent = float(np.max(points.max(axis=0) - points.min(axis=0))) * scale
    if extent <= 0:
        return 1
    depth = math.ceil(math.log2(extent / width))
    return int(min(max(depth, 1), MAX_DEPTH))


def reference_to_mesh(reference):
    """``(vertices, triangles)`` -> Open3D mesh with normals."""
    vertices, triangles = reference
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def evaluate_mesh(mesh, reference, points, normals):
    """Quality metrics vs. the reference mesh (or the input points)."""
    start = time.perf_counter()
    if reference is not None:
        metrics, errors = evaluate(
            mesh, reference_mesh=reference_to_mesh(reference), input_points=points
        )
    else:
        metrics, errors = evaluate(
            mesh, reference_points=points, reference_normals=normalize(normals)
        )
    return {
        "metrics": metrics,
        "vertex_errors": errors,
        "eval_time": time.perf_counter() - start,
    }


def mean_spacing(pcd):
    """Mean distance from each point to its nearest neighbour."""
    spacing = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    if spacing <= 0:
        raise ValueError("Cannot measure the point spacing (duplicate points?)")
    return spacing


def reconstruct_poisson(pcd, params):
    depth = params["depth"]
    if params["width"] > 0:
        depth = depth_for_width(
            np.asarray(pcd.points), params["width"], params["scale"]
        )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=params["scale"], linear_fit=params["linear_fit"]
    )

    # --- density filtering ---
    densities = np.asarray(densities)
    q = params["density_q"]
    if q > 0 and len(densities) == len(mesh.vertices):
        threshold = np.quantile(densities, q)
        mesh.remove_vertices_by_mask(densities < threshold)

    # --- crop to the point cloud bounding box ---
    if params["crop"]:
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    return mesh, {"depth": depth, "setting": f"depth {depth}"}


def reconstruct_ball_pivoting(pcd, params):
    spacing = mean_spacing(pcd)
    radius = params["factor"] * spacing
    radii = [radius, 2 * radius, 4 * radius]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh, {"spacing": spacing, "radii": radii,
                  "setting": f"r = {radius:.4g} (×{params['factor']:g})"}


# The Delaunay tetrahedralisation is the slow part of an alpha shape and
# does not depend on alpha: keep the last one (jobs run one at a time).
_tetra_cache = {}


def _tetra_mesh(pcd):
    key = array_key(np.asarray(pcd.points))
    if key not in _tetra_cache:
        _tetra_cache.clear()
        _tetra_cache[key] = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    return _tetra_cache[key]


def reconstruct_alpha_shape(pcd, params):
    spacing = mean_spacing(pcd)
    alpha = params["factor"] * spacing
    tetra, point_map = _tetra_mesh(pcd)
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra, point_map
        )
    except (IndexError, RuntimeError):
        # Open3D fails with a bare "_Map_base::at" when nothing survives
        mesh = o3d.geometry.TriangleMesh()
    return mesh, {"spacing": spacing, "alpha": alpha,
                  "setting": f"α = {alpha:.4g} (×{params['factor']:g})"}


RECONSTRUCTORS = {
    POISSON: reconstruct_poisson,
    BALL_PIVOTING: reconstruct_ball_pivoting,
    ALPHA_SHAPE: reconstruct_alpha_shape,
}


def build_mesh(params, points, normals, reference=None, run_evaluation=False):
    """Full pipeline -> result dict (mesh, arrays for rendering, timings
    and - when ``run_evaluation`` - quality metrics)."""
    timings = {}
    clock = time.perf_counter()

    pre_points, pre_normals, preprocess_cached = cached_preprocess(
        points, normals, params["preprocess"], params["voxel"]
    )
    if len(pre_points) < 4:
        raise ValueError("Too few points left after preprocessing")
    timings["preprocess"] = time.perf_counter() - clock
    clock = time.perf_counter()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pre_points)
    pcd.normals = o3d.utility.Vector3dVector(pre_normals)

    algorithm = params.get("algorithm", POISSON)
    mesh, info = RECONSTRUCTORS[algorithm](pcd, params)
    timings["reconstruct"] = time.perf_counter() - clock
    clock = time.perf_counter()

    # --- smoothing ---
    if params["smooth"] > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=params["smooth"]
        )

    if len(mesh.triangles) == 0:
        hint = {
            POISSON: "lower the density filter or disable cropping",
            BALL_PIVOTING: "increase the ball radius factor",
            ALPHA_SHAPE: "increase the alpha factor",
        }[algorithm]
        raise ValueError(f"Reconstruction produced an empty mesh - {hint}")

    mesh.compute_vertex_normals()
    timings["postprocess"] = time.perf_counter() - clock

    result = {
        "mesh": mesh,
        "vertices": np.asarray(mesh.vertices).copy(),
        "normals": np.asarray(mesh.vertex_normals).copy(),
        "triangles": np.asarray(mesh.triangles).copy(),
        "points_after_preprocess": len(pre_points),
        "preprocess_cached": preprocess_cached,
        "algorithm": algorithm,
        "depth": info.get("depth"),
        "setting": info["setting"],
        "timings": timings,
    }
    if run_evaluation:
        # Evaluate against the ORIGINAL input (before preprocessing)
        evaluation = evaluate_mesh(mesh, reference, points, normals)
        timings["evaluate"] = evaluation.pop("eval_time")
        result.update(evaluation)
    return result


COMBO_STYLE = """
QComboBox { background-color: #ffffff; color: #00171f; border: none;
            border-radius: 6px; padding: 6px 10px; font-size: 13px;
            font-weight: bold; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #00171f;
                              selection-background-color: #00a8e8; }
"""


class PoissonReconstruction(RotationMixin):
    """Import points + normals, reconstruct a mesh (Poisson, Ball Pivoting
    or Alpha Shape), measure its quality and export it."""

    NO_ACTOR_MESSAGE = "❌ No mesh loaded to rotate"
    LIVE_DEBOUNCE_MS = 400

    def __init__(self, main_window):
        self.main_window = main_window

        # ---------- UI Elements ----------
        self.btn_import = main_window.btn_import_pointsandnormals
        self.btn_export = main_window.btn_export_6
        self.pushButton = main_window.pushButton_6
        self.btn_import.setText("📂 Import Point Cloud")

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

        # Shown only for Poisson
        self.poisson_widgets = (
            main_window.Depth, self.spinBox_depth,
            main_window.width, self.doubleSpinBox_width,
            main_window.Scale, self.doubleSpinBox_scale,
            self.checkBox_linear_fit,
            main_window.label_density, self.doubleSpinBox_density,
            self.checkBox_crop,
        )
        self._build_algorithm_controls(main_window)

        # ---------- State ----------
        self.points = None
        self.normals = None
        self.mesh_o3d = None
        self._mesh_arrays = None      # (vertices, normals, triangles) shown
        self._last_build = None       # params / depth / timings of the mesh
        self._worker = None
        self._pending_run = False
        self._pending_eval = False
        self._data_generation = 0     # bumped on every import
        self._reset_camera_next = False
        self.init_rotation_state()

        # ---------- Quality evaluation state ----------
        self.reference = None         # (vertices, triangles) ground truth
        self.reference_name = ""
        self._reference_version = 0   # bumped whenever the reference changes
        self.evaluate_enabled = True
        self.heatmap_enabled = False
        self.vertex_errors = None
        self.last_record = None
        self.history = []
        self._run_counter = 0
        self._scalar_bar = None

        # Console + VTK render window + axes
        VTKViewport(self, main_window.frame_console_6, main_window.frame_view_6)

        # ---------- Configure parameter widgets ----------
        self.spinBox_depth.setMinimum(1)
        self.spinBox_depth.setMaximum(MAX_DEPTH)
        self.spinBox_depth.setValue(10)

        self.doubleSpinBox_width.setMinimum(0.0)
        self.doubleSpinBox_width.setMaximum(100.0)
        self.doubleSpinBox_width.setSingleStep(0.5)
        self.doubleSpinBox_width.setValue(0.0)          # 0 = use depth
        self.doubleSpinBox_width.setToolTip(
            "Finest octree cell size. 0 = use Depth; > 0 overrides Depth."
        )

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

        # Parameter changes are debounced so a burst of clicks runs once
        self._live_timer = QTimer(main_window)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(self.LIVE_DEBOUNCE_MS)
        self._live_timer.timeout.connect(self._restart_reconstruction)

        # ---------- Quality report window ----------
        self.report = QualityReportDialog(main_window)
        self.report.load_reference_clicked.connect(self.load_reference)
        self.report.use_points_clicked.connect(self.use_input_points)
        self.report.evaluate_toggled.connect(self.set_evaluate_enabled)
        self.report.heatmap_toggled.connect(self.set_heatmap_enabled)
        self.report.clear_history_clicked.connect(self.clear_history)
        self._update_reference_label()

        self.btn_quality = add_button_after(self.btn_export, "📊 Quality Report")
        self.btn_quality.clicked.connect(self.show_report)

        # The parameter panel is taller than a laptop screen: let it scroll
        make_scrollable(self.btn_export.parentWidget())

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(self.import_files)
        self.btn_export.clicked.connect(self.export_data)
        self.pushButton.clicked.connect(self.apply_rotation)

        self.setup_rotation_controls()
        self._connect_live_controls()

        self._set_controls_enabled(False)
        self.btn_export.setEnabled(False)

        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText("Surface Reconstruction ready ✅")
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(
            "📁 Import a point cloud with normals (.ply / .pcd / .txt / .csv"
            " / .npy) or send one from another tab"
        )

    # =========================
    # Algorithm selection
    # =========================
    def _build_algorithm_controls(self, main_window):
        """Algorithm menu + one size factor (ball radius / alpha), placed
        above the Poisson parameters in the side panel."""
        label_style = main_window.Depth.styleSheet()

        self.label_algorithm = QLabel("Algorithm")
        self.label_algorithm.setStyleSheet(label_style)
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.setStyleSheet(COMBO_STYLE)
        for key, label in ALGORITHM_LABELS.items():
            self.combo_algorithm.addItem(label, key)

        self.label_factor = QLabel()
        self.label_factor.setStyleSheet(label_style)
        self.doubleSpinBox_factor = QDoubleSpinBox()
        self.doubleSpinBox_factor.setStyleSheet(
            self.doubleSpinBox_scale.styleSheet()
        )
        self.doubleSpinBox_factor.setDecimals(2)
        self.doubleSpinBox_factor.setRange(0.5, 50.0)
        self.doubleSpinBox_factor.setSingleStep(0.5)
        self.doubleSpinBox_factor.setToolTip(
            "Multiple of the mean distance between neighbouring points."
        )

        for widget in (self.label_algorithm, self.combo_algorithm,
                       self.label_factor, self.doubleSpinBox_factor):
            insert_widget(main_window.Depth, widget, after=False)

        self.algorithm = POISSON
        self.factors = dict(DEFAULT_FACTORS)
        self._show_algorithm_controls()

    def _show_algorithm_controls(self):
        is_poisson = self.algorithm == POISSON
        for widget in self.poisson_widgets:
            widget.setVisible(is_poisson)
        self.label_factor.setVisible(not is_poisson)
        self.doubleSpinBox_factor.setVisible(not is_poisson)
        if not is_poisson:
            self.label_factor.setText(FACTOR_LABELS[self.algorithm])
            self.doubleSpinBox_factor.blockSignals(True)
            self.doubleSpinBox_factor.setValue(self.factors[self.algorithm])
            self.doubleSpinBox_factor.blockSignals(False)

    def _on_algorithm_changed(self, *_):
        self.algorithm = self.combo_algorithm.currentData()
        self._show_algorithm_controls()
        self.live_reconstruct()

    def _on_factor_changed(self, value):
        if self.algorithm in self.factors:
            self.factors[self.algorithm] = value
        self.live_reconstruct()

    # =========================
    # Signal wiring / enable state
    # =========================
    def _connect_live_controls(self):
        """Re-run the reconstruction whenever a parameter changes."""
        self.combo_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        self.doubleSpinBox_factor.valueChanged.connect(self._on_factor_changed)
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
        for widget in self.rotation_widgets():
            widget.setEnabled(enabled)

    # =========================
    # Import Files
    # =========================
    def import_files(self):
        """Prompt for a point cloud with normals (one file, or a points
        file plus a legacy normals file), then reconstruct."""
        try:
            data = ask_open_point_cloud(
                self.main_window, "Import Point Cloud with Normals",
                normals_required=True,
            )
        except Exception as e:
            self.console.appendPlainText(f"❌ Import error: {str(e)}")
            return
        if data is None:
            self.console.appendPlainText(
                "❌ Import cancelled (points with normals are required)"
            )
            return
        self.load_data(data, f"File imported ✅: {data.source}")

    def receive_data(self, data):
        """Called by the main window when another tab sends a cloud."""
        if not data.has_normals:
            data = data.with_normals()
        self.load_data(data, f"Received from another tab ✅: {data.source}")

    def load_data(self, data, title):
        self.points = data.points.copy()
        self.normals = data.normals.copy()
        self.mesh_o3d = None
        self._mesh_arrays = None
        self._data_generation += 1
        self._reset_camera_next = True

        # A mesh that travelled with the data is the ground truth
        self.reference = data.reference
        self.reference_name = (
            f"{data.source} (original mesh)" if data.reference is not None else ""
        )
        self._reference_version += 1
        self.vertex_errors = None
        self.last_record = None
        self._update_reference_label()
        self.report.show_record(None)
        self.report.set_status("Waiting for the reconstruction…")

        self.console.clear()
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(title)
        self.console.appendPlainText("=" * 50)
        self.console.appendPlainText(f"📊 Points:  {len(self.points)}")
        self.console.appendPlainText(f"📊 Normals: {len(self.normals)}")
        if self.reference is not None:
            self.console.appendPlainText(
                "📐 Ground-truth mesh received - quality is measured against it"
            )

        self._set_controls_enabled(True)
        self.btn_export.setEnabled(False)
        self.reset_rotation()

        # Start the reconstruction in the background right away
        self._live_timer.stop()
        self._restart_reconstruction()

    # =========================
    # Live Reconstruction (background thread, debounced)
    # =========================
    def live_reconstruct(self, *_):
        """Called whenever a Poisson parameter changes."""
        if self.points is None or self.normals is None:
            return
        self._live_timer.start()

    def _restart_reconstruction(self):
        if self._worker is not None:
            # A job is running: it is re-run with the latest parameters
            # as soon as the current job finishes.
            self._pending_run = True
            return
        self._start_worker()

    def _current_params(self):
        return {
            "algorithm": self.algorithm,
            "factor": self.factors.get(self.algorithm),
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

    def _start_worker(self):
        self._pending_run = False
        self._pending_eval = False
        self.console.appendPlainText(
            f"⏳ Reconstructing ({ALGORITHM_LABELS[self.algorithm]})…"
        )
        self.btn_export.setEnabled(False)

        # Snapshot parameters/inputs on the UI thread, pass copies to the worker
        params = self._current_params()
        points = self.points.copy()
        normals = self.normals.copy()
        reference = self.reference
        run_evaluation = self.evaluate_enabled
        tags = {
            "kind": "build",
            "params": params,
            "generation": self._data_generation,
            "reference_version": self._reference_version,
        }

        def job():
            result = build_mesh(params, points, normals, reference, run_evaluation)
            result.update(tags)
            return result

        self._launch(job)

    def _start_evaluation(self):
        """Evaluate the current mesh again (e.g. after a reference change)
        without rebuilding it."""
        if self.mesh_o3d is None or not self.evaluate_enabled:
            return
        if self._worker is not None:
            self._pending_eval = True
            return
        self._pending_eval = False
        self.console.appendPlainText("⏳ Evaluating quality…")

        mesh = o3d.geometry.TriangleMesh(self.mesh_o3d)
        points = self.points.copy()
        normals = self.normals.copy()
        reference = self.reference
        tags = {
            "kind": "evaluate",
            "generation": self._data_generation,
            "reference_version": self._reference_version,
        }

        def job():
            result = evaluate_mesh(mesh, reference, points, normals)
            result["timings"] = {"evaluate": result.pop("eval_time")}
            result.update(tags)
            return result

        self._launch(job)

    def _launch(self, job):
        self._worker = ReconstructionWorker(job, self.vtk_widget)
        self._worker.succeeded.connect(self._on_job_done)
        self._worker.failed.connect(self._on_reconstruction_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_job_done(self, result):
        if result["generation"] != self._data_generation:
            return  # result for data that has since been replaced

        if result["kind"] == "build":
            self.mesh_o3d = result["mesh"]
            self._mesh_arrays = (
                result["vertices"], result["normals"], result["triangles"]
            )
            self._last_build = {
                "params": result["params"],
                "algorithm": result["algorithm"],
                "depth": result["depth"],
                "setting": result["setting"],
                "timings": dict(result["timings"]),
            }
            # Metrics of the previous mesh no longer apply
            self.vertex_errors = None
            self.last_record = None
            if "metrics" not in result:
                self.report.show_record(None)
                self.report.set_status("Not evaluated (evaluation is off).")
            self.console.appendPlainText(
                f"   {ALGORITHM_LABELS[result['algorithm']]} | {result['setting']}"
                f" | points after preprocessing: {result['points_after_preprocess']}"
                + (" (cached)" if result["preprocess_cached"] else "")
            )

        evaluated = (
            "metrics" in result
            and result["reference_version"] == self._reference_version
        )
        if evaluated:
            self.vertex_errors = result["vertex_errors"]
            self._record_evaluation(result["metrics"], result["timings"])
        elif "metrics" in result:
            self._pending_eval = True  # reference changed while running

        if result["kind"] == "build":
            self._render_mesh()
            self.console.appendPlainText(
                f"✅ Reconstruction done | vertices: {len(result['vertices'])} | "
                f"triangles: {len(result['triangles'])}"
            )
            # A newer job may already be queued; export only the latest
            self.btn_export.setEnabled(not self._pending_run)
        elif self.heatmap_enabled:
            self._render_mesh()

    def _on_reconstruction_failed(self, message):
        self.console.appendPlainText(f"❌ Reconstruction error: {message}")

    def _on_worker_finished(self):
        # QThread.finished: the thread has fully stopped - safe to release
        self._worker.deleteLater()
        self._worker = None
        if self._pending_run:
            self._start_worker()
        elif self._pending_eval:
            self._start_evaluation()

    # =========================
    # Quality evaluation
    # =========================
    def _record_evaluation(self, metrics, eval_timings):
        timings = dict(self._last_build["timings"])
        timings.update(eval_timings)
        self._run_counter += 1
        record = {
            "run": self._run_counter,
            "params": self._last_build["params"],
            "algorithm": self._last_build["algorithm"],
            "depth": self._last_build["depth"],
            "setting": self._last_build["setting"],
            "n_points": len(self.points),
            "metrics": metrics,
            "timings": timings,
        }
        self.last_record = record
        self.history.append(record)

        against = "mesh" if metrics["reference"] == "mesh" else "points"
        self.console.appendPlainText(
            f"📏 {ALGORITHM_LABELS[record['algorithm']]} vs {against}: "
            f"Chamfer {metrics['chamfer_pct']:.3f}% · "
            f"H-95 {metrics['hausdorff_95_pct']:.3f}% · "
            f"F-score {metrics['f_score']:.3f} · "
            f"closed {'yes' if metrics['closed'] else 'no'}"
        )
        if metrics["likely_misaligned"]:
            self.console.appendPlainText(
                "⚠ Large distances: is the reference aligned with the data?"
            )
        self.report.show_record(record)
        self.report.set_history(self.history)

    def show_report(self):
        self.report.show()
        self.report.raise_()
        self.report.activateWindow()

    def _update_reference_label(self):
        if self.reference is not None:
            vertices, triangles = self.reference
            text = (f"Mesh: {self.reference_name}  "
                    f"({len(vertices):,} vertices, {len(triangles):,} triangles)")
        else:
            text = ("Input points (no ground-truth mesh) - measures how well "
                    "the surface fits the data.")
        self.report.set_reference(text, self.reference is not None)

    def load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self.report, "Load Reference Mesh (same coordinates as the data)",
            "", MESH_FILTER
        )
        if not path:
            return
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0:
            self.report.set_status(f"❌ {os.path.basename(path)}: no triangles")
            return
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        self.reference = (
            np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
        )
        self.reference_name = os.path.basename(path)
        self._reference_changed()

    def use_input_points(self):
        self.reference = None
        self.reference_name = ""
        self._reference_changed()

    def _reference_changed(self):
        self._reference_version += 1
        self._update_reference_label()
        self.console.appendPlainText(
            "📐 Reference: " + (self.reference_name or "input points")
        )
        self._start_evaluation()

    def set_evaluate_enabled(self, enabled):
        self.evaluate_enabled = enabled
        if enabled and self.last_record is None:
            self._start_evaluation()

    def set_heatmap_enabled(self, enabled):
        self.heatmap_enabled = enabled
        if enabled and self.vertex_errors is None:
            if self.evaluate_enabled:
                self.report.set_status("Heatmap needs an evaluation - running it…")
                self._start_evaluation()
            else:
                self.report.set_status(
                    "Heatmap needs an evaluation - enable evaluation first.")
        if self._mesh_arrays is not None:
            self._render_mesh()

    def clear_history(self):
        self.history = []
        self.report.set_history(self.history)

    # =========================
    # Background thread result -> VTK render (main thread)
    # =========================
    def _render_mesh(self):
        try:
            vertices, normals, triangles = self._mesh_arrays
            polydata = polydata_from_numpy(vertices, normals, triangles)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 0.6, 0.9)
            actor.GetProperty().SetAmbient(0.1)
            actor.GetProperty().SetDiffuse(0.8)
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(20)
            actor.GetProperty().SetInterpolationToPhong()

            show_errors = (
                self.heatmap_enabled
                and self.vertex_errors is not None
                and len(self.vertex_errors) == len(vertices)
            )
            if show_errors:
                self._apply_heatmap(polydata, mapper)
            else:
                mapper.ScalarVisibilityOff()
            self._set_scalar_bar(mapper if show_errors else None)

            # Keep the camera while tuning parameters; reset after import
            self.show_actor(actor, reset_camera=self._reset_camera_next)
            self._reset_camera_next = False

        except Exception as e:
            self.console.appendPlainText(f"❌ Render error: {str(e)}")

    def _apply_heatmap(self, polydata, mapper):
        """Color vertices by their distance to the reference (blue = 0,
        red >= 99th percentile)."""
        errors = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(self.vertex_errors, dtype=np.float32), deep=True
        )
        errors.SetName("Error")
        polydata.GetPointData().SetScalars(errors)

        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.Build()
        upper = max(float(np.percentile(self.vertex_errors, 99)), 1e-12)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0.0, upper)
        mapper.SetScalarModeToUsePointData()
        mapper.ScalarVisibilityOn()

    def _set_scalar_bar(self, mapper):
        if self._scalar_bar is None:
            bar = vtk.vtkScalarBarActor()
            bar.SetTitle("Error")
            bar.SetNumberOfLabels(5)
            bar.SetLabelFormat("%.3g")
            bar.SetPosition(0.88, 0.1)
            bar.SetWidth(0.1)
            bar.SetHeight(0.8)
            for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
                prop.SetColor(0.1, 0.1, 0.1)
                prop.ShadowOff()
            self.renderer.AddActor2D(bar)
            self._scalar_bar = bar
        if mapper is None:
            self._scalar_bar.VisibilityOff()
        else:
            self._scalar_bar.SetLookupTable(mapper.GetLookupTable())
            self._scalar_bar.VisibilityOn()

    # =========================
    # Export (STL / OBJ)
    # =========================
    def export_data(self):
        if self.mesh_o3d is None:
            self.console.appendPlainText("❌ No mesh to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.main_window, "Export Mesh", "reconstructed_mesh",
            "STL Files (*.stl);;OBJ Files (*.obj)"
        )
        if not file_path:
            return

        try:
            # Work on a copy so the displayed mesh is never modified
            mesh = o3d.geometry.TriangleMesh(self.mesh_o3d)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.remove_unreferenced_vertices()
            # Export what is displayed (rotation applied)
            mesh.transform(self.actor_matrix())
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in (".stl", ".obj"):
                ext = ".stl" if "stl" in selected_filter.lower() else ".obj"
                file_path += ext

            if ext == ".stl":
                success = o3d.io.write_triangle_mesh(
                    file_path, mesh, write_ascii=False
                )
            else:
                success = o3d.io.write_triangle_mesh(file_path, mesh)

            if success:
                self.console.appendPlainText(
                    f"✅ {ext[1:].upper()} exported (rotation applied): {file_path}"
                )
            else:
                self.console.appendPlainText(
                    f"❌ Export failed: could not write {file_path}"
                )
        except Exception as e:
            self.console.appendPlainText(f"❌ Export error: {str(e)}")
