"""Reconstruction tab: *Optimize parameters* window.

Settings (algorithm or all, objective, minimum recall, runs, noisy-data
check) -> Run; every run appears in the table and the convergence chart;
*Apply best* sets the tab's parameters (the selected row, or the best),
*Apply & Send to Compare* also hands the rebuilt mesh to tab 4.
"""

import csv
import os

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from modules import settings
from modules.optimizer import (
    OBJECTIVES,
    VALIDATION_NOISE_PCT,
    describe_values,
    objective_value,
)
from modules.quality_report import ALGORITHM_NAMES

ALL = "all"
COLORS = {"poisson": "#e15759", "ball_pivoting": "#4e79a7",
          "alpha_shape": "#76b7b2", "rimls": "#59a14f",
          "signed_distance": "#b07aa1"}
COLUMNS = ("Run", "Algorithm", "Parameters", "F-score", "Chamfer %",
           "Precision", "Recall", "Acc. RMS", "Pieces", "Time")


class ConvergenceChart(QWidget):
    """Best objective so far against the run number, per algorithm."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.series = {}          # algorithm -> [(run, best value)]
        self.label = ""

    def clear(self, label=""):
        self.series, self.label = {}, label
        self.update()

    def add(self, algorithm, run, value):
        points = self.series.setdefault(algorithm, [])
        best = value if not points else max(points[-1][1], value)
        points.append((run, best))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.Text)
        grid = QColor(text)
        grid.setAlpha(40)
        painter.fillRect(self.rect(), palette.color(palette.Base))
        font = QFont(self.font())
        font.setPixelSize(10)
        painter.setFont(font)
        points = [p for s in self.series.values() for p in s]
        if not points:
            painter.setPen(text)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Best value so far appears here while running")
            return
        left, right, top, bottom = 52, 12, 10, 24
        plot = QRectF(left, top, max(self.width() - left - right, 10),
                      max(self.height() - top - bottom, 10))
        runs = [p[0] for p in points]
        values = [p[1] for p in points]
        x0, x1 = min(runs), max(max(runs), min(runs) + 1)
        y0, y1 = min(values), max(values)
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 1e-3, y1 + 1e-3
        pad = (y1 - y0) * 0.1
        y0, y1 = y0 - pad, y1 + pad
        to_x = lambda v: plot.left() + (v - x0) / (x1 - x0) * plot.width()
        to_y = lambda v: plot.bottom() - (v - y0) / (y1 - y0) * plot.height()
        for i in range(5):
            value = y0 + (y1 - y0) * i / 4
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(plot.left(), to_y(value)),
                             QPointF(plot.right(), to_y(value)))
            painter.setPen(text)
            painter.drawText(QRectF(0, to_y(value) - 7, left - 5, 14),
                             Qt.AlignRight | Qt.AlignVCenter, f"{abs(value):.4g}")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 4, plot.width(), 14),
                         Qt.AlignHCenter, f"run · best {self.label} so far")
        for algorithm, series in self.series.items():
            color = QColor(COLORS.get(algorithm, "#888888"))
            path = QPainterPath(QPointF(to_x(series[0][0]), to_y(series[0][1])))
            for run, value in series[1:]:
                path.lineTo(QPointF(to_x(run), to_y(value)))
            painter.setPen(QPen(color, 2))
            painter.drawPath(path)
            painter.setBrush(color)
            for run, value in series:
                painter.drawEllipse(QPointF(to_x(run), to_y(value)), 2.5, 2.5)
            painter.setBrush(Qt.NoBrush)


class OptimizeDialog(QDialog):
    """Search window; the work itself runs in the Reconstruction tab."""

    run_requested = pyqtSignal(dict)
    cancel_requested = pyqtSignal()
    apply_requested = pyqtSignal(dict, bool)      # trial, send to Compare

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimize reconstruction parameters")
        self.resize(980, 720)
        self.trials = []
        self.result = None
        self.objective = "fscore"
        self.min_recall = 0.0
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Searches the parameters that fit the <b>reference mesh</b> best "
            "(coordinate descent: a coarse pass per parameter, then "
            "refinement around the best value). The objective is two-sided, "
            "so dropping hard parts of the surface does not pay off.")
        intro.setWordWrap(True)
        intro.setProperty("role", "hint")
        layout.addWidget(intro)

        form = QFormLayout()
        self.combo_algorithm = QComboBox()
        for key, name in ALGORITHM_NAMES.items():
            self.combo_algorithm.addItem(name, key)
        self.combo_algorithm.addItem("All algorithms (fair comparison)", ALL)
        self.combo_objective = QComboBox()
        for key, name in OBJECTIVES.items():
            self.combo_objective.addItem(name, key)
        self.spin_recall = QDoubleSpinBox()
        self.spin_recall.setRange(0, 100)
        self.spin_recall.setSuffix(" %")
        self.spin_recall.setValue(80)
        self.spin_recall.setToolTip(
            "Runs that cover less of the reference (within 1 % of the "
            "diagonal) are not eligible as best - keeps the result complete")
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(4, 300)
        self.spin_budget.setValue(25)
        self.spin_budget.setToolTip("Reconstructions tried per algorithm")
        self.check_validate = QCheckBox(
            f"Check the result on noisy data (σ = {VALIDATION_NOISE_PCT:g} % "
            "of the diagonal)")
        self.check_validate.setChecked(True)
        self.check_validate.setToolTip(
            "Re-run the starting and the best settings on a noisy copy: shows "
            "whether the gain is real or only fits this data set")
        form.addRow("Algorithm", self.combo_algorithm)
        form.addRow("Objective", self.combo_objective)
        form.addRow("Minimum recall", self.spin_recall)
        form.addRow("Runs per algorithm", self.spin_budget)
        form.addRow("", self.check_validate)
        layout.addLayout(form)
        self.reference_label = QLabel()
        self.reference_label.setWordWrap(True)
        layout.addWidget(self.reference_label)

        row = QHBoxLayout()
        self.btn_run = QPushButton("Run optimisation")
        self.btn_run.setProperty("variant", "primary")
        self.btn_cancel = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.progress, 1)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        splitter.addWidget(self.table)
        self.chart = ConvergenceChart()
        splitter.addWidget(self.chart)
        splitter.setSizes([420, 170])
        layout.addWidget(splitter, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        layout.addWidget(self.summary)

        row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply best to the tab")
        self.btn_apply.setToolTip(
            "Set the tab's parameters to the selected row (or the best run) "
            "and rebuild the mesh")
        self.btn_send = QPushButton("Apply && Send to Compare")
        self.btn_send.setProperty("variant", "primary")
        self.btn_copy = QPushButton("Copy table (Markdown)")
        self.btn_csv = QPushButton("Save CSV…")
        for button in (self.btn_apply, self.btn_send, self.btn_copy, self.btn_csv):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self.cancel_requested)
        self.btn_apply.clicked.connect(lambda: self._apply(False))
        self.btn_send.clicked.connect(lambda: self._apply(True))
        self.btn_copy.clicked.connect(self.copy_markdown)
        self.btn_csv.clicked.connect(self.save_csv)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.set_running(False)

    # ---------- state ----------
    def set_reference(self, has_reference, algorithm):
        if has_reference:
            self.reference_label.setText(
                "✅ Measured against the ground-truth (original) mesh.")
        else:
            self.reference_label.setText(
                "⚠ No reference mesh: the fit to the input points is optimised "
                "(weaker - load a reference in the Quality report, or send the "
                "data from Import mesh).")
        index = self.combo_algorithm.findData(algorithm)
        if index >= 0 and not self.trials:
            self.combo_algorithm.setCurrentIndex(index)

    def set_running(self, running):
        for widget in (self.combo_algorithm, self.combo_objective,
                       self.spin_recall, self.spin_budget, self.check_validate,
                       self.btn_run):
            widget.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self._running = running
        self._update_buttons()

    def _update_buttons(self):
        ready = not self._running and self._chosen() is not None
        self.btn_apply.setEnabled(ready)
        self.btn_send.setEnabled(ready)
        has_rows = bool(self.trials)
        self.btn_copy.setEnabled(has_rows)
        self.btn_csv.setEnabled(has_rows)

    def set_status(self, text):
        self.status.setText(text)

    # ---------- run ----------
    def _run(self):
        algorithm = self.combo_algorithm.currentData()
        algorithms = (list(ALGORITHM_NAMES) if algorithm == ALL else [algorithm])
        self.objective = self.combo_objective.currentData()
        self.min_recall = self.spin_recall.value() / 100.0
        self.trials, self.result = [], None
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.chart.clear(OBJECTIVES[self.objective].split(" (")[0])
        self.summary.setText("")
        self.progress.setRange(0, self.spin_budget.value() * len(algorithms))
        self.progress.setValue(0)
        self.run_requested.emit({
            "algorithms": algorithms,
            "budget": self.spin_budget.value(),
            "objective": self.objective,
            "min_recall": self.min_recall,
            "validate_noise": VALIDATION_NOISE_PCT if self.check_validate.isChecked() else 0.0,
        })

    def add_trial(self, step):
        trial = step["trial"]
        self.trials.append(trial)
        self.progress.setMaximum(step["total"])
        self.progress.setValue(step["done"])
        self._append_row(trial)
        if "metrics" in trial:
            self.chart.add(trial["algorithm"], trial["run"],
                           objective_value(trial["metrics"], self.objective))
        best = step["best"]
        if "metrics" in best:
            m = best["metrics"]
            self.set_status(
                f"Run {step['done']}/{step['total']} · best so far: "
                f"{ALGORITHM_NAMES[best['algorithm']]} "
                f"{describe_values(best['algorithm'], best['values'])} - "
                f"F {m['f_score']:.4f}, Chamfer {m['chamfer_pct']:.4f} %")

    def _append_row(self, trial):
        row = self.table.rowCount()
        self.table.insertRow(row)
        m = trial.get("metrics")
        cells = [
            trial["run"], ALGORITHM_NAMES[trial["algorithm"]],
            describe_values(trial["algorithm"], trial["values"]),
        ]
        if m:
            cells += [m["f_score"], m["chamfer_pct"], m["precision"], m["recall"],
                      m["accuracy"]["rms"], m["components"], trial["time"]]
        else:
            cells += ["failed: " + trial.get("error", "")[:60], "", "", "", "", "",
                      trial["time"]]
        for column, value in enumerate(cells):
            item = QTableWidgetItem()
            if isinstance(value, float):
                item.setData(Qt.DisplayRole,
                             round(value, 2 if column == 9 else 4))
            elif isinstance(value, int):
                item.setData(Qt.DisplayRole, value)
            else:
                item.setText(str(value))
            if column == 0:
                item.setData(Qt.UserRole, len(self.trials) - 1)
            if m and not trial["eligible"]:
                item.setForeground(QColor("#9aa6b2"))
                item.setToolTip("Below the minimum recall - not eligible")
            self.table.setItem(row, column, item)
        self.table.scrollToBottom()

    def show_result(self, result):
        """Final summary (best, gain over the start, noisy-data check)."""
        self.result = result
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)
        done = len(self.trials)
        self.progress.setMaximum(max(done, 1))
        self.progress.setValue(done)
        best = result.get("best")
        if best is None or "metrics" not in best:
            self.summary.setText("No successful run.")
            self._update_buttons()
            return
        # highlight the best row
        for row in range(self.table.rowCount()):
            index = self.table.item(row, 0).data(Qt.UserRole)
            if self.trials[index] is best:
                for column in range(self.table.columnCount()):
                    item = self.table.item(row, column)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.item(row, 1).setText("★ " + self.table.item(row, 1).text())
                self.table.selectRow(row)
        lines = []
        for part in result["results"]:
            initial, top = part["initial"], part["best"]
            if not top or "metrics" not in top:
                continue
            name = ALGORITHM_NAMES[part["algorithm"]]
            start = initial["metrics"] if initial and "metrics" in initial else None
            m = top["metrics"]
            gain = ""
            if start:
                gain = (f" (start F {start['f_score']:.4f}, Chamfer "
                        f"{start['chamfer_pct']:.4f} %)")
            lines.append(
                f"<b>{name}</b>: {describe_values(part['algorithm'], top['values'])}"
                f" → F <b>{m['f_score']:.4f}</b>, Chamfer {m['chamfer_pct']:.4f} %, "
                f"recall {m['recall']:.1%}{gain}")
        if not best["eligible"]:
            lines.append("⚠ No run reached the minimum recall - showing the best "
                         "one anyway; lower the minimum recall or try another "
                         "algorithm.")
        validation = result.get("validation")
        if validation:
            for algorithm, row in validation["results"].items():
                if "best" in row and "initial" in row and "f_score" in row["best"] \
                        and "f_score" in row["initial"]:
                    better = row["best"]["f_score"] >= row["initial"]["f_score"]
                    lines.append(
                        f"Noisy-data check ({validation['noise_pct']:g} %) · "
                        f"{ALGORITHM_NAMES[algorithm]}: start F "
                        f"{row['initial']['f_score']:.4f} → best F "
                        f"{row['best']['f_score']:.4f} "
                        + ("✅ the gain holds" if better
                           else "⚠ worse on noisy data - the settings may only "
                                "fit this data set"))
        if result.get("cancelled"):
            lines.append("Cancelled - the best of the runs done so far is shown.")
        self.summary.setText("<br>".join(lines))
        self._update_buttons()

    # ---------- output ----------
    def _chosen(self):
        rows = self.table.selectionModel().selectedRows() if self.table.model() else []
        if rows:
            index = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
            trial = self.trials[index]
            return trial if "metrics" in trial else None
        if self.result and self.result.get("best") and "metrics" in self.result["best"]:
            return self.result["best"]
        return None

    def _apply(self, send):
        trial = self._chosen()
        if trial is not None:
            self.apply_requested.emit(trial, send)

    def rows(self):
        out = []
        for trial in sorted(self.trials, key=lambda t: t["run"]):
            m = trial.get("metrics") or {}
            out.append([
                trial["run"], ALGORITHM_NAMES[trial["algorithm"]],
                describe_values(trial["algorithm"], trial["values"]),
                m.get("f_score", ""), m.get("chamfer_pct", ""),
                m.get("precision", ""), m.get("recall", ""),
                (m.get("accuracy") or {}).get("rms", ""), m.get("components", ""),
                round(trial["time"], 2), trial.get("error", ""),
            ])
        return out

    def copy_markdown(self):
        header = list(COLUMNS)
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for row in self.rows():
            cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row[:10]]
            lines.append("| " + " | ".join(cells) + " |")
        QApplication.clipboard().setText("\n".join(lines))
        self.set_status("Table copied as Markdown.")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save optimisation runs",
            os.path.join(settings.last_dir(), "optimization.csv"), "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        settings.remember_path(path)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["run", "algorithm", "parameters", "f_score",
                             "chamfer_pct", "precision", "recall",
                             "accuracy_rms", "pieces", "time_s", "error"])
            writer.writerows(self.rows())
        self.set_status(f"Saved {os.path.basename(path)}")
