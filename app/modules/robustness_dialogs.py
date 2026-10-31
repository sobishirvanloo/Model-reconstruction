"""Windows for the robustness experiments.

* ``DegradeDialog`` - Processing tab: degrade the current point cloud by
  hand (noise, normal noise, holes, non-uniform sampling).
* ``SweepDialog``   - Reconstruction tab: run every selected algorithm on
  several levels of one perturbation and collect the quality metrics.

Both only collect settings and emit signals; the tabs do the work.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
)

from modules.degradation import AXES, SWEEP_KINDS, Degradation, parse_levels
from modules.quality_report import ALGORITHM_NAMES


def _double(minimum, maximum, step, decimals, suffix=""):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    return box


def _buttons(*specs):
    """Row of buttons from (label, QStyle icon) pairs; the first is the
    primary action."""
    row = QHBoxLayout()
    buttons = []
    for index, (label, icon) in enumerate(specs):
        button = QPushButton(label)
        button.setIcon(button.style().standardIcon(icon))
        if index == 0:
            button.setProperty("variant", "primary")
        row.addWidget(button)
        buttons.append(button)
    return row, buttons


class DegradeDialog(QDialog):
    applied = pyqtSignal(object)      # Degradation

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Degrade Data")

        self.noise = _double(0.0, 10.0, 0.05, 2, " %")
        self.normal = _double(0.0, 90.0, 5.0, 1, " °")
        self.holes = QSpinBox()
        self.holes.setRange(0, 50)
        self.hole_radius = _double(0.5, 30.0, 0.5, 1, " %")
        self.min_keep = _double(1.0, 100.0, 5.0, 1, " %")
        self.axis = QComboBox()
        self.axis.addItems(list(AXES))
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)

        form = QFormLayout()
        form.addRow("Position noise σ (% of diagonal)", self.noise)
        form.addRow("Normal noise σ", self.normal)
        form.addRow("Holes", self.holes)
        form.addRow("Hole radius (% of diagonal)", self.hole_radius)
        form.addRow("Non-uniform: keep at far end", self.min_keep)
        form.addRow("Non-uniform: along axis", self.axis)
        form.addRow("Random seed", self.seed)

        hint = QLabel(
            "Applied after voxel downsampling. A ground-truth mesh that came "
            "with the data stays clean, so the Reconstruction tab measures "
            "how robust each algorithm is."
        )
        hint.setWordWrap(True)
        self.summary = QLabel()

        row, (self.btn_apply, self.btn_reset, self.btn_close) = _buttons(
            ("Apply", QStyle.SP_DialogApplyButton),
            ("Reset to clean", QStyle.SP_DialogResetButton),
            ("Close", QStyle.SP_DialogCloseButton))
        self.btn_apply.clicked.connect(lambda: self.applied.emit(self.settings()))
        self.btn_reset.clicked.connect(self._reset)
        self.btn_close.clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self.summary)
        layout.addLayout(row)
        self.set_settings(Degradation())

    def settings(self):
        return Degradation(
            noise_pct=self.noise.value(),
            normal_deg=self.normal.value(),
            holes=self.holes.value(),
            hole_radius_pct=self.hole_radius.value(),
            min_keep_pct=self.min_keep.value(),
            axis=self.axis.currentIndex(),
            seed=self.seed.value(),
        )

    def set_settings(self, degradation):
        self.noise.setValue(degradation.noise_pct)
        self.normal.setValue(degradation.normal_deg)
        self.holes.setValue(degradation.holes)
        self.hole_radius.setValue(degradation.hole_radius_pct)
        self.min_keep.setValue(degradation.min_keep_pct)
        self.axis.setCurrentIndex(degradation.axis)
        self.seed.setValue(degradation.seed)
        self.summary.setText(f"Current: {degradation.describe()}")

    def _reset(self):
        self.set_settings(Degradation(seed=self.seed.value()))
        self.applied.emit(self.settings())


class SweepDialog(QDialog):
    run_requested = pyqtSignal(str, object, object, int)  # kind, levels, algorithms, seed
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robustness Sweep")
        self.resize(560, 320)

        self.kind = QComboBox()
        for key, (label, _, _) in SWEEP_KINDS.items():
            self.kind.addItem(label, key)
        self.levels = QLineEdit()
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.algorithm_checks = {}
        algorithms = QHBoxLayout()
        for key, label in ALGORITHM_NAMES.items():
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(self._update_count)
            self.algorithm_checks[key] = check
            algorithms.addWidget(check)

        form = QFormLayout()
        form.addRow("Perturbation", self.kind)
        form.addRow("Levels (comma separated)", self.levels)
        form.addRow("Algorithms", algorithms)
        form.addRow("Random seed", self.seed)

        hint = QLabel(
            "Each level degrades the data loaded in this tab and runs every "
            "selected algorithm with its current parameters. Results go to "
            "the Quality Report history. Quality is measured against the "
            "reference mesh, or the clean input points when there is none."
        )
        hint.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)

        row, (self.btn_run, self.btn_cancel) = _buttons(
            ("Run sweep", QStyle.SP_MediaPlay), ("Cancel", QStyle.SP_MediaStop))
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self.cancel_requested)
        self.kind.currentIndexChanged.connect(self._fill_default_levels)
        self.levels.textChanged.connect(self._update_count)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self.status)
        layout.addLayout(row)
        self._fill_default_levels()
        self.set_running(False)

    def _fill_default_levels(self, *_):
        _, _, defaults = SWEEP_KINDS[self.kind.currentData()]
        self.levels.setText(", ".join(f"{level:g}" for level in defaults))

    def _selected_algorithms(self):
        return [key for key, check in self.algorithm_checks.items()
                if check.isChecked()]

    def _update_count(self, *_):
        try:
            levels = parse_levels(self.levels.text(), self.kind.currentData())
        except ValueError as e:
            self.status.setText(f"❌ {e}")
            return
        runs = len(levels) * len(self._selected_algorithms())
        self.status.setText(f"{runs} reconstructions will run.")

    def _run(self):
        kind = self.kind.currentData()
        try:
            levels = parse_levels(self.levels.text(), kind)
        except ValueError as e:
            self.status.setText(f"❌ {e}")
            return
        algorithms = self._selected_algorithms()
        if not algorithms:
            self.status.setText("❌ Select at least one algorithm.")
            return
        self.run_requested.emit(kind, levels, algorithms, self.seed.value())

    def set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        for widget in (self.kind, self.levels, self.seed,
                       *self.algorithm_checks.values()):
            widget.setEnabled(not running)

    def set_status(self, text):
        self.status.setText(text)
