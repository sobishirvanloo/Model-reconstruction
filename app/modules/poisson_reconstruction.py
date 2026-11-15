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

The pipeline itself lives in ``pipeline`` (no Qt) and runs in the helper
process; a background ``QThread`` waits for it, so the UI stays responsive
while parameters are being tuned (live reconstruction, debounced).
"""

import os
import threading
import time

import numpy as np
import open3d as o3d
import vtk
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QStyle,
)

from modules import settings
from vtk.util import numpy_support

from modules.degradation import SWEEP_KINDS, sweep_settings
from modules.geometry_utils import polydata_from_numpy, transfer_colors
from modules.point_data import (
    add_button_after,
    add_section,
    ask_open_point_cloud,
    insert_widget,
    make_scrollable,
)
from modules.pipeline import (  # noqa: F401  (re-exported for other modules)
    ALPHA_SHAPE,
    BALL_PIVOTING,
    DEFAULT_FACTORS,
    MAX_DEPTH,
    POISSON,
    RECONSTRUCTORS,
    RIMLS,
    SIGNED_DISTANCE,
    array_key,
    build_mesh,
    build_mesh_local,
    cached_preprocess,
    clear_all_caches,
    depth_for_width,
    evaluate_mesh,
    mean_spacing,
    preprocess,
    reference_to_mesh,
)
from modules.quality_report import (
    ALGORITHM_NAMES,
    QualityReportDialog,
    sweep_summary,
)
from modules.robustness_dialogs import SweepDialog
from modules.vtk_controls import TabMixin, VTKViewport, release_memory

MESH_FILTER = "Meshes (*.stl *.obj *.ply *.off *.gltf *.glb)"
ALGORITHM_LABELS = ALGORITHM_NAMES   # display names, keyed by the ids above
STOP_OLD_BUILD_AFTER = 8.0  # s: a longer build of old parameters is stopped
SWEEP_RUN_LIMIT = 900.0     # s per sweep run (a hung library call is stopped)
FACTOR_LABELS = {
    BALL_PIVOTING: "Ball radius (× point spacing)",
    ALPHA_SHAPE: "Alpha (× point spacing)",
    RIMLS: "Kernel radius h (× point spacing)",
    SIGNED_DISTANCE: "Influence radius (× point spacing)",
}


class ReconstructionWorker(QThread):
    """Runs a reconstruction / evaluation / sweep job off the UI thread.

    ``job(progress)`` may call ``progress(step)`` for intermediate results
    (emitted as ``progress``).  Emits ``succeeded(result)`` or
    ``failed(error_message)``; the built-in ``QThread.finished`` fires
    afterwards and is used for cleanup.
    """

    progress = pyqtSignal(object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self):
        try:
            result = self._job(self.progress.emit)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.succeeded.emit(result)



def run_sweep(points, normals, reference, base_params, factors, kind, levels,
              algorithms, seed, cancel, progress, base_note=""):
    """Robustness sweep: degrade the data at every level, reconstruct with
    every algorithm and evaluate.  Quality is always measured against the
    ground truth - the reference mesh, or the CLEAN input points - so it
    shows how much each algorithm suffers from the perturbation.

    ``progress(step)`` is called after every run; ``cancel`` (an Event) is
    checked between runs.
    """
    total = len(levels) * len(algorithms)
    done = 0
    for level in levels:
        degradation = sweep_settings(kind, level, seed)
        d_points, d_normals = degradation.apply(points, normals)
        note = degradation.describe()
        if base_note:
            note = f"{base_note} + {note}"
        for algorithm in algorithms:
            if cancel.is_set():
                return {"cancelled": True, "done": done, "total": total}
            params = dict(base_params, algorithm=algorithm,
                          factor=factors.get(algorithm))
            step = {"level": level, "algorithm": algorithm, "data": note,
                    "params": params, "n_points": len(d_points)}
            try:
                result = build_mesh(params, d_points, d_normals,
                                    timeout=SWEEP_RUN_LIMIT, cancel=cancel)
                evaluation = evaluate_mesh(result["mesh"], reference, points,
                                           normals, cancel=cancel)
                result["timings"]["evaluate"] = evaluation.pop("eval_time")
                step.update(metrics=evaluation["metrics"],
                            timings=result["timings"],
                            depth=result["depth"], setting=result["setting"])
            except Exception as e:
                step["error"] = str(e)
            done += 1
            step.update(done=done, total=total)
            progress(step)
    return {"cancelled": False, "done": done, "total": total}


PARAMETER_TIPS = {
    "combo_algorithm": "Poisson: smooth closed surface, robust to noise "
                       "- the most accurate overall.\n"
                       "Ball Pivoting: passes exactly through the points.\n"
                       "Alpha Shape: fastest, but many small pieces.\n"
                       "RIMLS: robust moving least squares - keeps sharp "
                       "features, ignores outliers (≈ 10 s).\n"
                       "Signed Distance: fast volumetric method, the best "
                       "choice for noisy data (no debris).",
    "spinBox_depth": "Octree depth: higher = more detail, slower (8–10 typical)",
    "doubleSpinBox_width": "Finest octree cell size. 0 = use Depth; "
                           "a value > 0 overrides Depth.",
    "doubleSpinBox_scale": "Size of the reconstruction cube relative to the "
                           "data's bounding box (1.1 avoids cut edges)",
    "checkBox_linear_fit": "Place vertices by linear interpolation "
                           "(slightly more accurate, slower)",
    "checkBox_preprocess": "Remove outliers and orient the normals "
                           "consistently before reconstructing",
    "doubleSpinBox_voxel": "Downsample the input first (0 = keep every point)",
    "doubleSpinBox_density": "Remove the least supported part of the Poisson "
                             "surface (% of vertices with the lowest density)",
    "checkBox_crop": "Cut the Poisson surface to the data's bounding box "
                     "(removes 'bubbles')",
    "spinBox_smooth": "Taubin smoothing iterations (0 = off)",
}


class PoissonReconstruction(TabMixin):
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
        self.colors = None            # RGB of the input points, if any
        self.source_name = ""
        self.mesh_o3d = None
        self._mesh_arrays = None      # (vertices, normals, triangles) shown
        self._mesh_colors = None      # vertex colours taken from the points
        self._last_build = None       # params / depth / timings of the mesh
        self._worker = None
        self._build_run = None        # {"cancel", "started"} of a running build
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
        self.data_note = ""           # how the loaded data was degraded
        self._sweep = None            # state of a running robustness sweep
        self._optimize = None         # state of a running optimisation
        self._send_after_build = False
        self.optimize_dialog = None   # created on first use

        # Console + toolbar + VTK render window + axes
        VTKViewport(
            self, main_window.frame_console_6, main_window.frame_view_6,
            empty_hint="Drop a point cloud with normals here\n"
                       "or press Ctrl+O to open one\n\n"
                       "(or send one from the Import mesh / Processing tab)",
        )

        # ---------- Configure parameter widgets ----------
        self.spinBox_depth.setMinimum(1)
        self.spinBox_depth.setMaximum(MAX_DEPTH)
        self.spinBox_depth.setValue(10)

        self.doubleSpinBox_width.setMinimum(0.0)
        self.doubleSpinBox_width.setMaximum(100.0)
        self.doubleSpinBox_width.setSingleStep(0.5)
        self.doubleSpinBox_width.setValue(0.0)          # 0 = use depth

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
        self.report.sweep_clicked.connect(self.show_sweep_dialog)
        self._update_reference_label()

        self.sweep_dialog = SweepDialog(main_window)
        self.sweep_dialog.run_requested.connect(self.start_sweep)
        self.sweep_dialog.cancel_requested.connect(self.cancel_sweep)

        self.btn_quality = add_button_after(self.btn_export, "Quality report")
        self.btn_quality.clicked.connect(self.show_report)
        self.btn_send_compare = add_button_after(self.btn_quality,
                                                 "Send to Compare")
        self.btn_send_compare.clicked.connect(self._send_to_compare)

        # ---------- Connect buttons ----------
        self.btn_import.clicked.connect(lambda: self.import_files())
        self.btn_export.clicked.connect(self.export_data)

        self.setup_rotation_controls()
        self._connect_live_controls()
        self._build_panel()

        # The parameter panel is taller than a laptop screen: let it scroll
        make_scrollable(self.btn_export.parentWidget())

        self._set_controls_enabled(False)
        self.btn_export.setEnabled(False)

        self.console.appendPlainText(
            "Ready. Open a point cloud with normals (.ply / .pcd / .txt / .csv"
            " / .npy) or send one from another tab."
        )

    def _build_panel(self):
        """Section headers, icons, tooltips and the file label."""
        mw = self.main_window
        self.btn_import.setText("Open point cloud…")
        self.decorate(self.btn_import, QStyle.SP_DialogOpenButton,
                      "Open a point cloud with normals (Ctrl+O)", "primary")
        self.file_label = QLabel("No data loaded")
        self.file_label.setProperty("role", "hint")
        self.file_label.setWordWrap(True)
        insert_widget(self.btn_import, self.file_label, after=True)
        self.add_remove_button(self.file_label)

        self.btn_export.setText("Export mesh…")
        self.decorate(self.btn_export, QStyle.SP_DialogSaveButton,
                      "Save the mesh as STL or OBJ, rotation applied (Ctrl+S)",
                      "primary")
        self.decorate(self.btn_send_compare, QStyle.SP_ArrowForward,
                      "Compare this mesh with the original in tab 4 "
                      "(7 distance methods, error statistics)")
        self.decorate(self.btn_optimize, "target")
        self.decorate(self.btn_quality, QStyle.SP_FileDialogDetailedView,
                      "Quality metrics, error heatmap, run history and "
                      "robustness sweeps (Ctrl+R)")

        add_section(self.btn_import, "File")
        add_section(mw.label_x_6, "Rotation")
        add_section(self.label_algorithm, "Reconstruction")
        add_section(mw.label_preproc, "Preprocessing")
        add_section(mw.label_density, "Post-processing")
        add_section(self.btn_export, "Output")
        mw.label_preproc.hide()       # the section header says it already
        for name, tip in PARAMETER_TIPS.items():
            getattr(self, name).setToolTip(tip)

    # =========================
    # Algorithm selection
    # =========================
    def _build_algorithm_controls(self, main_window):
        """Algorithm menu + one size factor (ball radius / alpha), placed
        above the Poisson parameters in the side panel."""
        self.label_algorithm = QLabel("Algorithm")
        self.combo_algorithm = QComboBox()
        for key, label in ALGORITHM_LABELS.items():
            self.combo_algorithm.addItem(label, key)

        self.label_factor = QLabel()
        self.doubleSpinBox_factor = QDoubleSpinBox()
        self.doubleSpinBox_factor.setDecimals(2)
        self.doubleSpinBox_factor.setRange(0.5, 50.0)
        self.doubleSpinBox_factor.setSingleStep(0.5)
        self.doubleSpinBox_factor.setToolTip(
            "Multiple of the mean distance between neighbouring points."
        )

        self.btn_optimize = QPushButton("Optimize parameters…")
        self.btn_optimize.setToolTip(
            "Search the settings that fit the reference mesh best "
            "(one algorithm or all of them), then apply them")
        self.btn_optimize.clicked.connect(self.show_optimize_dialog)
        for widget in (self.label_algorithm, self.combo_algorithm,
                       self.label_factor, self.doubleSpinBox_factor,
                       self.btn_optimize):
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
    def import_files(self, path=None):
        """Open a point cloud with normals (one file, or a points file plus
        a legacy normals file), then reconstruct.  ``path`` skips the first
        dialog (drag & drop, recent files)."""
        try:
            data = ask_open_point_cloud(
                self.main_window, "Open Point Cloud with Normals",
                normals_required=True, path=path,
            )
        except Exception as e:
            name = os.path.basename(path) if path else "the file"
            self.notify(f"❌ Could not open {name}: {e}", "error", dialog=True)
            return
        if data is None:
            self.notify("Open cancelled: points with normals are required",
                        "warning")
            return
        self.load_data(data, f"✅ Opened {data.source}")

    def receive_data(self, data):
        """Called by the main window when another tab sends a cloud."""
        if not data.has_normals:
            data = data.with_normals()
        self.load_data(data, f"✅ Received from another tab: {data.source}")

    def load_data(self, data, title):
        if self._sweep is not None:
            self.cancel_sweep()       # its results belong to the old data
        self.cancel_optimize()
        self.points = data.points.copy()
        self.normals = data.normals.copy()
        self.colors = None if data.colors is None else data.colors.copy()
        self.data_note = data.note
        self.mesh_o3d = None
        self._mesh_arrays = None
        self._mesh_colors = None
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
        self.console.appendPlainText(title)
        self.console.appendPlainText(f"   {len(self.points):,} points with normals")
        self.source_name = data.source
        self._show_data_info()
        self.set_tab_info(file=data.source, points=len(self.points), triangles=None)
        self.notify(f"Loaded {data.source or 'data'} - reconstructing…", "success")
        if self.reference is not None:
            self.console.appendPlainText(
                "📐 Ground-truth mesh received - quality is measured against it"
            )
        if self.data_note:
            self.console.appendPlainText(f"🧪 Data: {self.data_note}")

        self._set_controls_enabled(True)
        self.btn_export.setEnabled(False)
        self.btn_remove.setEnabled(True)
        self.reset_transform()

        # Start the reconstruction in the background right away
        self._live_timer.stop()
        self._restart_reconstruction()

    def _show_data_info(self):
        self.file_label.setText(
            f"{self.source_name or 'Received data'}\n{len(self.points):,} points"
            + (f"\n{self.data_note}" if self.data_note else "")
            + ("\nGround truth: original mesh" if self.reference is not None else ""))
        self.file_label.setProperty("role", "file")
        self.file_label.style().polish(self.file_label)

    def info_source(self):
        """What the Object info panel describes (the reconstructed mesh)."""
        if self._mesh_arrays is None:
            return None
        vertices, normals, triangles = self._mesh_arrays
        rows = [("Input points", f"{len(self.points):,}",
                 "Points with normals the mesh was built from")]
        if self._last_build:
            rows += [
                ("Algorithm", ALGORITHM_LABELS[self._last_build["algorithm"]],
                 "Reconstruction method"),
                ("Setting", self._last_build["setting"], "Main parameter"),
                ("Time", f"{sum(self._last_build['timings'].values()):.2f} s",
                 "Total reconstruction time"),
            ]
        if self.last_record:
            metrics = self.last_record.get("metrics", {})
            for key, label in (("chamfer_pct", "Chamfer (% diag)"),
                               ("hausdorff_95_pct", "Hausdorff-95 (% diag)"),
                               ("f_score", "F-score")):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    rows.append((label, f"{value:.4g}",
                                 "Quality against the reference (Quality report)"))
        return vertices, normals, triangles, ("Reconstruction", rows)

    # =========================
    # Checkpoints
    # =========================
    PARAMETER_WIDGETS = (
        "spinBox_depth", "doubleSpinBox_width", "doubleSpinBox_scale",
        "checkBox_linear_fit", "checkBox_preprocess", "doubleSpinBox_voxel",
        "doubleSpinBox_density", "checkBox_crop", "spinBox_smooth",
    )

    def snapshot(self):
        widgets = {}
        for name in self.PARAMETER_WIDGETS:
            widget = getattr(self, name)
            widgets[name] = (widget.isChecked() if hasattr(widget, "isChecked")
                             else widget.value())
        return {
            "points": self.points, "normals": self.normals,
            "colors": self.colors, "source": self.source_name,
            "data_note": self.data_note, "reference": self.reference,
            "reference_name": self.reference_name,
            "mesh": self.mesh_o3d, "mesh_arrays": self._mesh_arrays,
            "mesh_colors": self._mesh_colors, "last_build": self._last_build,
            "vertex_errors": self.vertex_errors, "last_record": self.last_record,
            "algorithm": self.algorithm, "factors": dict(self.factors),
            "widgets": widgets, **self.pose_state(),
        }

    def restore(self, state):
        """Back to a checkpoint without recomputing anything (a running
        reconstruction is ignored)."""
        if self._sweep is not None:
            self.cancel_sweep()
        self.cancel_optimize()
        self._live_timer.stop()
        self._pending_run = False
        self._pending_eval = False
        self._data_generation += 1
        self.points, self.normals = state["points"], state["normals"]
        self.colors, self.source_name = state["colors"], state["source"]
        self.data_note = state["data_note"]
        if state["reference"] is not self.reference:
            self._reference_version += 1
        self.reference = state["reference"]
        self.reference_name = state["reference_name"]
        self.mesh_o3d = state["mesh"]
        self._mesh_arrays = state["mesh_arrays"]
        self._mesh_colors = state["mesh_colors"]
        self._last_build = state["last_build"]
        self.vertex_errors = state["vertex_errors"]
        self.last_record = state["last_record"]

        # Parameters, without triggering a new reconstruction
        self.algorithm = state["algorithm"]
        self.factors = dict(state["factors"])
        self.combo_algorithm.blockSignals(True)
        self.combo_algorithm.setCurrentIndex(
            self.combo_algorithm.findData(self.algorithm))
        self.combo_algorithm.blockSignals(False)
        self._show_algorithm_controls()
        for name, value in state["widgets"].items():
            widget = getattr(self, name)
            widget.blockSignals(True)
            if hasattr(widget, "setChecked"):
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)

        self._show_data_info()
        self._update_reference_label()
        self.report.show_record(self.last_record)
        self._set_controls_enabled(True)
        self.btn_remove.setEnabled(True)
        self.btn_export.setEnabled(self.mesh_o3d is not None)
        self.restore_pose(state)
        if self._mesh_arrays is not None:
            self._reset_camera_next = False
            self._render_mesh()
            triangles = len(self._mesh_arrays[2])
        else:
            if self.actor is not None:
                self.renderer.RemoveActor(self.actor)
                self.actor = None
            triangles = None
        self.set_tab_info(file=self.source_name, points=len(self.points),
                          triangles=triangles)
        busy = getattr(self.main_window, "set_busy", None)
        if busy and self._worker is None:
            busy(False)

    # =========================
    # Remove data
    # =========================
    def has_data(self):
        return self.points is not None

    def clear_data(self):
        """Drop the points, mesh, reference and caches.  A job still
        running finishes in the background and its result is ignored;
        the run history (small) is kept for comparison."""
        if self._sweep is not None:
            self.cancel_sweep()
        self.cancel_optimize()
        self._live_timer.stop()
        self._pending_run = False
        self._pending_eval = False
        self._data_generation += 1    # ignore results of a running job
        self.points = None
        self.normals = None
        self.colors = None
        self.source_name = ""
        self.data_note = ""
        self.mesh_o3d = None
        self._mesh_arrays = None
        self._mesh_colors = None
        self._last_build = None
        self.reference = None
        self.reference_name = ""
        self._reference_version += 1
        self.vertex_errors = None
        self.last_record = None
        clear_all_caches()
        if self._scalar_bar is not None:
            self._scalar_bar.VisibilityOff()
        self._update_reference_label()
        self.report.show_record(None)
        self.report.set_status("No data loaded.")
        self.btn_export.setEnabled(False)
        self.clear_view()
        self.console.appendPlainText(
            "Ready. Open a point cloud with normals (.ply / .pcd / .txt / .csv"
            " / .npy) or send one from another tab."
        )

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
            # as soon as the current job finishes.  A slow build of the
            # old parameters is stopped instead of waited for.
            self._pending_run = True
            build = self._build_run
            if (build is not None and time.monotonic() - build["started"]
                    > STOP_OLD_BUILD_AFTER):
                build["cancel"].set()
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
        colors = self.colors
        reference = self.reference
        run_evaluation = self.evaluate_enabled
        tags = {
            "kind": "build",
            "params": params,
            "generation": self._data_generation,
            "reference_version": self._reference_version,
        }
        cancel = threading.Event()
        self._build_run = {"cancel": cancel, "started": time.monotonic()}

        def job(progress):
            result = build_mesh(params, points, normals, reference,
                                run_evaluation, cancel=cancel)
            # Each vertex takes the colour of the nearest input point
            result["colors"] = transfer_colors(points, colors, result["vertices"])
            if result["colors"] is not None:
                result["mesh"].vertex_colors = o3d.utility.Vector3dVector(
                    result["colors"])
            result.update(tags)
            return result

        self._launch(job, f"Reconstructing ({ALGORITHM_LABELS[self.algorithm]})…")

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

        def job(progress):
            result = evaluate_mesh(mesh, reference, points, normals)
            result["timings"] = {"evaluate": result.pop("eval_time")}
            result.update(tags)
            return result

        self._launch(job, "Evaluating quality…")

    def _launch(self, job, busy_text):
        set_busy = getattr(self.main_window, "set_busy", None)
        if set_busy:
            set_busy(True, busy_text)
        self._worker = ReconstructionWorker(job, self.vtk_widget)
        self._worker.progress.connect(self._on_sweep_step)
        self._worker.succeeded.connect(self._on_job_done)
        self._worker.failed.connect(self._on_reconstruction_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_job_done(self, result):
        if result["kind"] == "sweep_end":
            self._finish_sweep(result)
            return
        if result["kind"] == "optimize_end":
            self._finish_optimize(result)
            return
        if result["generation"] != self._data_generation:
            return  # result for data that has since been replaced

        if result["kind"] == "build":
            self.mesh_o3d = result["mesh"]
            self._mesh_arrays = (
                result["vertices"], result["normals"], result["triangles"]
            )
            self._mesh_colors = result.get("colors")
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
            self.notify(
                f"✅ Reconstruction done | vertices: {len(result['vertices'])} | "
                f"triangles: {len(result['triangles'])}", "success"
            )
            self.set_tab_info(points=len(self.points),
                              triangles=len(result["triangles"]))
            # A newer job may already be queued; export only the latest
            self.btn_export.setEnabled(not self._pending_run)
            self.record_checkpoint(
                f"{ALGORITHM_LABELS[result['algorithm']]} · {result['setting']}")
            if self._send_after_build and not self._pending_run:
                self._send_after_build = False      # "Apply & Send to Compare"
                send = getattr(self.main_window, "send_to_compare", None)
                if send:
                    send()
        elif self.heatmap_enabled:
            self._render_mesh()

    def _on_reconstruction_failed(self, message):
        build = self._build_run
        if build is not None and build["cancel"].is_set():
            self.console.appendPlainText(
                "⏹ Stopped the previous build - using the new parameters")
            return
        # Status bar only: while tuning live, a dialog per failure would nag
        self.notify(f"❌ Reconstruction failed: {message}", "error")
        if self._sweep is not None:
            self._finish_sweep({"cancelled": True})
        if self._optimize is not None:
            self._finish_optimize({"cancelled": True, "results": [], "best": None})

    def _on_worker_finished(self):
        # QThread.finished: the thread has fully stopped - safe to release
        self._worker.deleteLater()
        self._worker = None
        self._build_run = None
        if self._pending_run:
            self._start_worker()
        elif self._pending_eval:
            self._start_evaluation()
        set_busy = getattr(self.main_window, "set_busy", None)
        if set_busy and self._worker is None:
            set_busy(False)       # nothing new was started

    # =========================
    # Quality evaluation
    # =========================
    def _add_record(self, **fields):
        """Append one evaluated run to the history (and its table)."""
        self._run_counter += 1
        record = dict(fields, run=self._run_counter)
        self.history.append(record)
        self.report.set_history(self.history)
        return record

    def _record_evaluation(self, metrics, eval_timings):
        timings = dict(self._last_build["timings"])
        timings.update(eval_timings)
        record = self._add_record(
            params=self._last_build["params"],
            algorithm=self._last_build["algorithm"],
            depth=self._last_build["depth"],
            setting=self._last_build["setting"],
            n_points=len(self.points),
            metrics=metrics,
            timings=timings,
            data=self.data_note,
        )
        self.last_record = record
        self.object_info.schedule()

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

    # =========================
    # Robustness sweep
    # =========================
    def show_sweep_dialog(self):
        self.sweep_dialog.show()
        self.sweep_dialog.raise_()

    def start_sweep(self, kind, levels, algorithms, seed):
        if self.points is None:
            self.sweep_dialog.set_status("❌ Load data in this tab first.")
            return
        if self._sweep is not None:
            return
        if self._worker is not None:
            self.sweep_dialog.set_status(
                "⏳ A reconstruction is running - try again when it finishes.")
            return
        self._live_timer.stop()

        label = SWEEP_KINDS[kind][0]
        cancel = threading.Event()
        self._sweep = {
            "cancel": cancel, "kind": kind, "label": label,
            "levels": list(levels), "algorithms": list(algorithms),
            "records": [], "failures": [],
            "generation": self._data_generation,
        }
        self.sweep_dialog.set_running(True)
        self.sweep_dialog.set_status("Starting…")
        names = ", ".join(ALGORITHM_LABELS[a] for a in algorithms)
        self.console.appendPlainText(
            f"🧪 Sweep: {label} = {', '.join(f'{v:g}' for v in levels)} × {names}"
        )
        if self.reference is None:
            self.console.appendPlainText(
                "   (no reference mesh: measured against the clean input points)"
            )

        args = (self.points.copy(), self.normals.copy(), self.reference,
                self._current_params(), dict(self.factors), kind, list(levels),
                list(algorithms), seed, cancel)
        base_note = self.data_note
        generation = self._data_generation

        def job(progress):
            result = run_sweep(*args, progress, base_note=base_note)
            result.update(kind="sweep_end", generation=generation)
            return result

        self._launch(job, "Robustness sweep…")
        set_progress = getattr(self.main_window, "set_progress", None)
        if set_progress:
            set_progress(0, len(levels) * len(algorithms))

    # =========================
    # Parameter optimisation
    # =========================
    def show_optimize_dialog(self):
        if self.optimize_dialog is None:
            from modules.optimize_dialog import OptimizeDialog
            self.optimize_dialog = OptimizeDialog(self.main_window)
            self.optimize_dialog.run_requested.connect(self.start_optimize)
            self.optimize_dialog.cancel_requested.connect(self.cancel_optimize)
            self.optimize_dialog.apply_requested.connect(self.apply_trial)
        self.optimize_dialog.set_reference(self.reference is not None,
                                           self.algorithm)
        self.optimize_dialog.show()
        self.optimize_dialog.raise_()
        self.optimize_dialog.activateWindow()

    def start_optimize(self, options):
        dialog = self.optimize_dialog
        if self.points is None:
            dialog.set_status("❌ Load data in this tab first.")
            return
        if self._worker is not None or self._sweep is not None or self._optimize:
            dialog.set_status("⏳ Another job is running - try again when it "
                              "finishes.")
            return
        from modules.optimizer import run_optimization
        self._live_timer.stop()
        cancel = threading.Event()
        generation = self._data_generation
        self._optimize = {"cancel": cancel, "generation": generation}
        dialog.set_running(True)
        dialog.set_status("Starting…")
        names = ", ".join(ALGORITHM_LABELS[a] for a in options["algorithms"])
        self.console.appendPlainText(
            f"🎯 Optimising {names}: {options['budget']} runs each, objective "
            f"{options['objective']}, min recall {options['min_recall']:.0%}")
        args = (self.points.copy(), self.normals.copy(), self.reference,
                self._current_params())

        def job(progress):
            result = run_optimization(
                *args, options["algorithms"], options["budget"],
                options["objective"], options["min_recall"],
                options["validate_noise"], cancel, progress)
            result.update(kind="optimize_end", generation=generation)
            return result

        self._launch(job, "Optimising parameters…")

    def _on_optimize_step(self, step):
        if self._optimize is None:
            return
        trial = step["trial"]
        self.optimize_dialog.add_trial(step)
        name = ALGORITHM_LABELS[trial["algorithm"]]
        if "metrics" in trial:
            m = trial["metrics"]
            self.console.appendPlainText(
                f"🎯 {step['done']}/{step['total']} {name} {trial['setting']}: "
                f"F {m['f_score']:.4f} · Chamfer {m['chamfer_pct']:.4f} %")
        else:
            self.console.appendPlainText(
                f"🎯 {step['done']}/{step['total']} {name}: {trial['error']}")
        set_progress = getattr(self.main_window, "set_progress", None)
        if set_progress:
            set_progress(step["done"], step["total"],
                         f"🎯 Optimising {step['done']}/{step['total']}")

    def _finish_optimize(self, result):
        state, self._optimize = self._optimize, None
        if state is None:
            return
        release_memory()              # many runs: hand freed pages back
        dialog = self.optimize_dialog
        dialog.set_running(False)
        if result.get("generation", state["generation"]) != self._data_generation:
            dialog.set_status("The data changed - results discarded.")
            return
        dialog.show_result(result)
        best = result.get("best")
        if best and "metrics" in best:
            m = best["metrics"]
            text = (f"🎯 Best: {ALGORITHM_LABELS[best['algorithm']]} "
                    f"{best['setting']} - F {m['f_score']:.4f}, Chamfer "
                    f"{m['chamfer_pct']:.4f} %")
            self.console.appendPlainText(text)
            dialog.set_status(text.lstrip("🎯 ") + " - Apply to use it.")
            self.notify(text, "success")
        else:
            dialog.set_status("Optimisation finished without a usable run.")

    def cancel_optimize(self):
        if self._optimize is not None:
            self._optimize["cancel"].set()
            if self.optimize_dialog is not None:
                self.optimize_dialog.set_status("Cancelling after the current run…")

    def apply_trial(self, trial, send=False):
        """Set the parameters of an optimisation run and rebuild the mesh
        (``send``: then hand it to the Compare tab)."""
        if self.points is None or self._optimize is not None:
            return
        algorithm = trial["algorithm"]
        values = trial["values"]
        self.algorithm = algorithm
        self.combo_algorithm.blockSignals(True)
        self.combo_algorithm.setCurrentIndex(self.combo_algorithm.findData(algorithm))
        self.combo_algorithm.blockSignals(False)
        if "factor" in values:
            self.factors[algorithm] = values["factor"]
        self._show_algorithm_controls()
        widgets = {
            "depth": (self.spinBox_depth, int),
            "density_q": (self.doubleSpinBox_density, lambda v: 100.0 * v),
            "smooth": (self.spinBox_smooth, int),
            "scale": (self.doubleSpinBox_scale, float),
        }
        for name, value in values.items():
            if name in widgets:
                widget, convert = widgets[name]
                widget.blockSignals(True)
                widget.setValue(convert(value))
                widget.blockSignals(False)
        if algorithm == POISSON:
            self.doubleSpinBox_width.blockSignals(True)
            self.doubleSpinBox_width.setValue(0.0)       # the search used depth
            self.doubleSpinBox_width.blockSignals(False)
        self.console.appendPlainText(
            f"🎯 Applied: {ALGORITHM_LABELS[algorithm]} {trial['setting']}")
        self._send_after_build = send
        self._live_timer.stop()
        self._restart_reconstruction()

    def cancel_sweep(self):
        if self._sweep is not None:
            self._sweep["cancel"].set()
            self.sweep_dialog.set_status("Cancelling after the current run…")

    def _on_sweep_step(self, step):
        if step.get("kind") == "optimize":
            self._on_optimize_step(step)
            return
        sweep = self._sweep
        if sweep is None or sweep["generation"] != self._data_generation:
            return
        name = ALGORITHM_LABELS[step["algorithm"]]
        if "error" in step:
            sweep["failures"].append(step)
            self.console.appendPlainText(
                f"❌ {name} · {step['data']}: {step['error']}")
        else:
            record = self._add_record(
                params=step["params"], algorithm=step["algorithm"],
                depth=step["depth"], setting=step["setting"],
                n_points=step["n_points"], metrics=step["metrics"],
                timings=step["timings"], data=step["data"],
                level=step["level"],
            )
            sweep["records"].append(record)
            m = step["metrics"]
            self.console.appendPlainText(
                f"🧪 {name} · {step['data']}: Chamfer {m['chamfer_pct']:.3f}% · "
                f"F {m['f_score']:.3f}")
        text = f"Sweep {step['done']}/{step['total']}: {name} · {step['data']}"
        self.sweep_dialog.set_status(text)
        self.report.set_status(text)
        set_progress = getattr(self.main_window, "set_progress", None)
        if set_progress:
            set_progress(step["done"], step["total"], text)

    def _finish_sweep(self, result):
        sweep, self._sweep = self._sweep, None
        self.sweep_dialog.set_running(False)
        if sweep is None:
            return
        state = "cancelled" if result.get("cancelled") else "finished"
        title = f"Sweep {state}: {len(sweep['records'])} runs"
        if sweep["failures"]:
            title += f", {len(sweep['failures'])} failed"
        headers, rows = sweep_summary(
            sweep["records"], sweep["levels"], sweep["algorithms"], sweep["label"]
        )
        self.report.show_summary(title, headers, rows)
        self.sweep_dialog.set_status(title + " - summary in the Quality Report.")
        self.notify(f"🧪 {title}", "warning" if sweep["failures"] else "success")
        self.show_report()

    def _send_to_compare(self):
        if self._mesh_arrays is None:
            self.notify("❌ Reconstruct a mesh first", "warning")
            return
        send = getattr(self.main_window, "send_to_compare", None)
        if send:
            send()

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
            settings.last_dir(), MESH_FILTER
        )
        if not path:
            return
        settings.remember_path(path)
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
            polydata = polydata_from_numpy(
                vertices, normals, triangles, self._mesh_colors)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
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
            self.show_actor(actor, reset_camera=self._reset_camera_next,
                            kind="mesh")
            self._reset_camera_next = False

        except Exception as e:
            self.notify(f"❌ Render error: {e}", "error")

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
            bar.VisibilityOff()
            self.renderer.AddViewProp(bar)
            self._scalar_bar = bar
        if mapper is None:
            self._scalar_bar.VisibilityOff()
        else:
            self._scalar_bar.SetLookupTable(mapper.GetLookupTable())
            self._scalar_bar.VisibilityOn()
        self._style_scalar_bar()

    def _style_scalar_bar(self):
        color = self.viewport.color("text")
        for prop in (self._scalar_bar.GetTitleTextProperty(),
                     self._scalar_bar.GetLabelTextProperty()):
            prop.SetColor(*color)
            prop.ShadowOff()

    def refresh_theme(self):
        """Theme-dependent VTK colours that the viewport does not own."""
        if self._scalar_bar is not None:
            self._style_scalar_bar()
            self.viewport.render()

    # =========================
    # Export (STL / OBJ)
    # =========================
    def export_data(self):
        if self.mesh_o3d is None:
            self.notify("❌ No mesh to export", "warning")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.main_window, "Export Mesh",
            os.path.join(settings.last_dir(), "reconstructed_mesh"),
            "STL Files (*.stl);;OBJ Files (*.obj)"
        )
        if not file_path:
            return
        settings.remember_path(file_path)

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
                self.notify(
                    f"✅ {ext[1:].upper()} exported (rotation applied): {file_path}",
                    "success")
            else:
                self.notify(f"❌ Export failed: could not write {file_path}",
                            "error", dialog=True)
        except Exception as e:
            self.notify(f"❌ Export failed: {e}", "error", dialog=True)
