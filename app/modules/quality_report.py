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
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import os

from modules import settings
from modules.quality_metrics import metric_rows, rows_to_markdown


def make_button(text, icon=None, tooltip="", variant=None):
    """Push button with a standard icon and an optional theme variant."""
    button = QPushButton(text)
    if icon is not None:
        button.setIcon(button.style().standardIcon(icon))
    if tooltip:
        button.setToolTip(tooltip)
    if variant:
        button.setProperty("variant", variant)
    return button


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
    ("Data", lambda r: r.get("data") or "clean"),
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


def sweep_summary(records, levels, algorithms, level_label):
    """Level x algorithm table of a sweep -> (headers, rows)."""
    headers = [level_label] + [ALGORITHM_NAMES[a] for a in algorithms]
    rows = []
    for level in levels:
        row = [f"{level:g}"]
        for algorithm in algorithms:
            match = [r for r in records
                     if r["level"] == level and r["algorithm"] == algorithm]
            if not match:
                row.append("failed")
                continue
            m = match[-1]["metrics"]
            row.append(f"Chamfer {m['chamfer_pct']:.3f}% · F {m['f_score']:.3f}")
        rows.append(row)
    return headers, rows


def table_to_markdown(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def flatten_record(record):
    """One CSV row with every parameter and metric of a run."""
    row = {
        "run": record["run"],
        "algorithm": record["algorithm"],
        "setting": record["setting"],
        "data": record.get("data") or "clean",
        "sweep_level": record.get("level", ""),
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
    sweep_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quality Report")
        self.resize(820, 700)
        self._latest_markdown = ""
        self._history = []

        # ---------- Reference ----------
        ref_box = QGroupBox("Reference (ground truth)")
        self.reference_label = QLabel("Input points")
        self.reference_label.setWordWrap(True)
        self.btn_load_reference = make_button(
            "Load reference mesh…", QStyle.SP_DialogOpenButton,
            "Compare against a mesh in the same coordinates as the data")
        self.btn_use_points = make_button(
            "Use input points", QStyle.SP_DialogResetButton,
            "Compare against the input points instead of a mesh")
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
        self.btn_sweep = make_button(
            "Robustness sweep…", QStyle.SP_MediaPlay,
            "Run every algorithm on several levels of noise, holes, …",
            "primary")
        options_row = QHBoxLayout()
        options_row.addWidget(self.check_heatmap)
        options_row.addStretch()
        options_row.addWidget(self.btn_sweep)

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
        self.btn_copy = make_button(
            "Copy table as Markdown", QStyle.SP_FileDialogContentsView,
            "Copy the table above, ready to paste into a report")
        self.btn_save_csv = make_button(
            "Save history CSV…", QStyle.SP_DialogSaveButton,
            "Every parameter and metric of every run")
        self.btn_clear = make_button(
            "Clear history", QStyle.SP_DialogDiscardButton)
        export_row = QHBoxLayout()
        for button in (self.btn_copy, self.btn_save_csv, self.btn_clear):
            export_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(ref_box)
        layout.addWidget(self.check_evaluate)
        layout.addLayout(options_row)
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
        self.btn_sweep.clicked.connect(self.sweep_clicked)
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
        rows = []
        if record is not None:
            rows = metric_rows(record["metrics"], record["timings"])
            warning = (
                "  ⚠ Large distance: is the reference aligned with the data?"
                if record["metrics"]["likely_misaligned"] else ""
            )
            data = record.get("data")
            self.set_status(
                f"Run #{record['run']}" + (f" · data: {data}" if data else "")
                + warning
            )
        self._show_table(["Metric", "Value"], [list(r) for r in rows])
        self._latest_markdown = rows_to_markdown(rows) if rows else ""
        self._update_buttons()

    def show_summary(self, title, headers, rows):
        """Show a sweep's level x algorithm summary in the top table."""
        self._show_table(headers, rows)
        self._latest_markdown = table_to_markdown(headers, rows)
        self.set_status(title)
        self._update_buttons()

    def _show_table(self, headers, rows):
        table = self.latest_table
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        header = table.horizontalHeader()
        for c in range(len(headers)):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

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
        self.btn_copy.setEnabled(bool(self._latest_markdown))
        self.btn_save_csv.setEnabled(bool(self._history))
        self.btn_clear.setEnabled(bool(self._history))

    # =========================
    # Export
    # =========================
    def copy_markdown(self):
        QApplication.clipboard().setText(self._latest_markdown)
        self.set_status("Table copied to the clipboard as Markdown ✅")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Run History",
            os.path.join(settings.last_dir(), "quality_history.csv"), "CSV (*.csv)")
        if not path:
            return
        settings.remember_path(path)
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
