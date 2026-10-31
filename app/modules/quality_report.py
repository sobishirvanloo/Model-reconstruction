"""Non-modal "Quality Report" window of the Reconstruction tab.

Shows the latest quality metrics, a history of every evaluated run (to
compare parameter settings) and exports them as Markdown / CSV for a
report.  The window only displays data and emits signals; the
Reconstruction tab owns the state and does the work.
"""

import csv

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from modules.quality_metrics import metric_rows, rows_to_markdown


DIALOG_STYLE = """
QDialog { background-color: #003459; }
QLabel, QCheckBox, QGroupBox { color: #ffffff; font-size: 12px;
                               background: transparent; }
QGroupBox { font-weight: bold; border: 1px solid #007ea7;
            border-radius: 6px; margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QTableWidget { background-color: #ffffff; color: #00171f;
               alternate-background-color: #eef5f9;
               gridline-color: #c8d8e2; font-size: 12px; }
QHeaderView::section { background-color: #007ea7; color: #ffffff;
                       font-weight: bold; padding: 4px; border: none; }
"""

BUTTON_STYLE = """
QPushButton { background-color: #003459; color: #ffffff; border: none;
              border-radius: 6px; padding: 6px 12px; font-weight: bold; }
QPushButton:hover { background-color: #00a8e8; }
QPushButton:pressed { background-color: #00263a; }
QPushButton:disabled { background-color: #7f9aab; }
"""

ALGORITHM_NAMES = {
    "poisson": "Poisson",
    "ball_pivoting": "Ball Pivoting",
    "alpha_shape": "Alpha Shape",
}

# (column title, function(record) -> display value)
HISTORY_COLUMNS = (
    ("#", lambda r: r["run"]),
    ("Reference", lambda r: r["metrics"]["reference"]),
    ("Algorithm", lambda r: ALGORITHM_NAMES.get(r["algorithm"], r["algorithm"])),
    ("Setting", lambda r: r["setting"]),
    ("Voxel", lambda r: r["params"]["voxel"]),
    ("Density %", lambda r: round(r["params"]["density_q"] * 100, 2)),
    ("Smooth", lambda r: r["params"]["smooth"]),
    ("Crop", lambda r: "yes" if r["params"]["crop"] else "no"),
    ("Points", lambda r: r["n_points"]),
    ("Triangles", lambda r: r["metrics"]["triangles"]),
    ("Chamfer %", lambda r: round(r["metrics"]["chamfer_pct"], 4)),
    ("H-95 %", lambda r: round(r["metrics"]["hausdorff_95_pct"], 4)),
    ("F-score", lambda r: round(r["metrics"]["f_score"], 4)),
    ("Normal cons.", lambda r: (
        round(r["metrics"]["normal_consistency"], 4)
        if r["metrics"]["normal_consistency"] is not None else "-")),
    ("Closed", lambda r: "yes" if r["metrics"]["closed"] else "no"),
    ("Time (s)", lambda r: round(sum(r["timings"].values()), 2)),
)


def flatten_record(record):
    """One CSV row with every parameter and metric of a run."""
    row = {
        "run": record["run"],
        "algorithm": record["algorithm"],
        "setting": record["setting"],
        "points": record["n_points"],
        "effective_depth": record["depth"],
    }
    row.update({f"param_{k}": v for k, v in record["params"].items()})
    for key, value in record["metrics"].items():
        if isinstance(value, dict):
            row.update({f"{key}_{k}": v for k, v in value.items()})
        else:
            row[key] = value
    row.update({f"time_{k}": v for k, v in record["timings"].items()})
    return row


class QualityReportDialog(QDialog):
    load_reference_clicked = pyqtSignal()
    use_points_clicked = pyqtSignal()
    evaluate_toggled = pyqtSignal(bool)
    heatmap_toggled = pyqtSignal(bool)
    clear_history_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quality Report")
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(820, 700)
        self._latest_rows = []
        self._history = []

        # ---------- Reference ----------
        ref_box = QGroupBox("Reference (ground truth)")
        self.reference_label = QLabel("Input points")
        self.reference_label.setWordWrap(True)
        self.btn_load_reference = QPushButton("📐 Load reference mesh…")
        self.btn_use_points = QPushButton("Use input points")
        for button in (self.btn_load_reference, self.btn_use_points):
            button.setStyleSheet(BUTTON_STYLE)
        ref_buttons = QHBoxLayout()
        ref_buttons.addWidget(self.btn_load_reference)
        ref_buttons.addWidget(self.btn_use_points)
        ref_layout = QVBoxLayout(ref_box)
        ref_layout.addWidget(self.reference_label)
        ref_layout.addLayout(ref_buttons)

        # ---------- Options ----------
        self.check_evaluate = QCheckBox("Evaluate after every reconstruction")
        self.check_evaluate.setChecked(True)
        self.check_heatmap = QCheckBox("Color mesh by error (heatmap)")

        # ---------- Latest metrics ----------
        self.status_label = QLabel("No evaluation yet.")
        self.latest_table = QTableWidget(0, 2)
        self.latest_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.latest_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.latest_table.horizontalHeader().setStretchLastSection(True)
        self.latest_table.verticalHeader().setVisible(False)
        self.latest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.latest_table.setAlternatingRowColors(True)

        # ---------- History ----------
        self.history_table = QTableWidget(0, len(HISTORY_COLUMNS))
        self.history_table.setHorizontalHeaderLabels(
            [title for title, _ in HISTORY_COLUMNS])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)

        # ---------- Export ----------
        self.btn_copy = QPushButton("📋 Copy latest as Markdown")
        self.btn_save_csv = QPushButton("💾 Save history CSV…")
        self.btn_clear = QPushButton("🗑 Clear history")
        export_row = QHBoxLayout()
        for button in (self.btn_copy, self.btn_save_csv, self.btn_clear):
            button.setStyleSheet(BUTTON_STYLE)
            export_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(ref_box)
        layout.addWidget(self.check_evaluate)
        layout.addWidget(self.check_heatmap)
        layout.addWidget(self.status_label)
        layout.addWidget(self.latest_table, 3)
        layout.addWidget(QLabel("Run history (compare parameter settings):"))
        layout.addWidget(self.history_table, 2)
        layout.addLayout(export_row)

        # ---------- Signals ----------
        self.btn_load_reference.clicked.connect(self.load_reference_clicked)
        self.btn_use_points.clicked.connect(self.use_points_clicked)
        self.check_evaluate.toggled.connect(self.evaluate_toggled)
        self.check_heatmap.toggled.connect(self.heatmap_toggled)
        self.btn_copy.clicked.connect(self.copy_markdown)
        self.btn_save_csv.clicked.connect(self.save_csv)
        self.btn_clear.clicked.connect(self.clear_history_clicked)
        self._update_buttons()

    # =========================
    # Display
    # =========================
    def set_reference(self, text, is_mesh):
        self.reference_label.setText(text)
        self.btn_use_points.setEnabled(is_mesh)

    def set_status(self, text):
        self.status_label.setText(text)

    def show_record(self, record):
        """Show the latest evaluated run (``None`` clears the table)."""
        if record is None:
            self._latest_rows = []
        else:
            self._latest_rows = metric_rows(record["metrics"], record["timings"])
            warning = (
                "  ⚠ Large distance: is the reference aligned with the data?"
                if record["metrics"]["likely_misaligned"] else ""
            )
            self.set_status(f"Run #{record['run']}{warning}")
        self.latest_table.setRowCount(len(self._latest_rows))
        for row, (name, value) in enumerate(self._latest_rows):
            self.latest_table.setItem(row, 0, QTableWidgetItem(name))
            self.latest_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self._update_buttons()

    def set_history(self, history):
        self._history = list(history)
        self.history_table.setRowCount(len(self._history))
        for row, record in enumerate(self._history):
            for col, (_, getter) in enumerate(HISTORY_COLUMNS):
                item = QTableWidgetItem(str(getter(record)))
                item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, col, item)
        if self._history:
            self.history_table.scrollToBottom()
        self._update_buttons()

    def _update_buttons(self):
        self.btn_copy.setEnabled(bool(self._latest_rows))
        self.btn_save_csv.setEnabled(bool(self._history))
        self.btn_clear.setEnabled(bool(self._history))

    # =========================
    # Export
    # =========================
    def copy_markdown(self):
        QApplication.clipboard().setText(rows_to_markdown(self._latest_rows))
        self.set_status("Latest metrics copied to the clipboard as Markdown ✅")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Run History", "quality_history.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        rows = [flatten_record(record) for record in self._history]
        fields = []
        for row in rows:
            fields += [key for key in row if key not in fields]
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.set_status(f"History saved ✅: {path}")
        except OSError as e:
            self.set_status(f"❌ Could not save: {e}")
