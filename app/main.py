import os
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStyle,
)
from PyQt5.uic import loadUi

from modules import settings, theme
from modules.import_mesh import MESH_EXTENSIONS, Geometry
from modules.point_processing import POINT_EXTENSIONS, Processing
from modules.poisson_reconstruction import PoissonReconstruction
from modules.splash_screen import create_splash_and_show
from modules.vtk_controls import CAMERA_VIEWS, release_memory

TAB_TITLES = ("1   Import mesh", "2   Process points", "3   Reconstruct")
MESH_ONLY = (".stl", ".obj", ".off", ".gltf", ".glb")
TABLE_ONLY = (".txt", ".csv", ".npy")
MESSAGE_TIMEOUT_MS = {"info": 6000, "success": 6000, "warning": 10000,
                      "error": 15000}

SHORTCUTS = (
    ("Ctrl+O", "Open a file in the current tab"),
    ("Ctrl+Shift+O", "Open the sample Stanford bunny"),
    ("Ctrl+S", "Export from the current tab"),
    ("Ctrl+Shift+S", "Save a screenshot of the 3D view"),
    ("Ctrl+W", "Remove the current tab's data (frees memory)"),
    ("Ctrl+Shift+W", "Remove the data from all tabs"),
    ("Ctrl+0", "Reset the view"),
    ("Ctrl+1 … Ctrl+4", "Front / Top / Side / Iso view"),
    ("Ctrl+R", "Quality report"),
    ("Ctrl+T", "Switch light / dark theme"),
    ("Ctrl+Q", "Quit"),
    ("Mouse: left drag", "Rotate the model (anywhere in the view)"),
    ("Mouse: right / middle drag", "Move the model"),
    ("Mouse: wheel", "Zoom"),
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theme_name = settings.theme()
        theme.apply_theme(QApplication.instance(), self.theme_name)

        # Resolve the .ui file relative to this file so the app can be
        # launched from anywhere (not only from inside app/).
        ui_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ui", "main.ui"
        )
        loadUi(ui_path, self)
        self.setWindowTitle("Model Reconstruction")

        self._tab_info = {}
        self._build_status_bar()

        self.geometry = Geometry(self)
        self.processing = Processing(self)
        self.poisson = PoissonReconstruction(self)
        self.tabs = (self.geometry, self.processing, self.poisson)
        self.pages = (self.tab, self.tab_2, self.tab_5)

        for tab in self.tabs:
            panel = tab.btn_import.parentWidget()
            panel.setProperty("role", "panel")
        for index, title in enumerate(TAB_TITLES):
            self.tabWidget.setTabText(index, title)
        self.tabWidget.currentChanged.connect(self._refresh_status)

        self._build_menus()
        self.setAcceptDrops(True)
        self.set_theme(self.theme_name)
        self._refresh_status()

    # =========================
    # Tabs
    # =========================
    def current_tab(self):
        return self.tabs[self.tabWidget.currentIndex()]

    def show_tab(self, tab):
        self.tabWidget.setCurrentWidget(self.pages[self.tabs.index(tab)])

    def send_point_cloud(self, data, target):
        """Hand a ``PointCloudData`` to another tab in memory and show it.

        ``target`` is ``"processing"`` or ``"reconstruction"``.
        """
        tab = {"processing": self.processing,
               "reconstruction": self.poisson}[target]
        self.show_tab(tab)
        tab.receive_data(data)

    def open_path(self, path):
        """Open a file in the tab that can handle it (drag & drop, recent
        files): meshes go to *Import mesh*, tables to *Process points*,
        PLY / PCD to the current tab."""
        ext = os.path.splitext(path)[1].lower()
        current = self.current_tab()
        if ext in MESH_ONLY:
            target = self.geometry
        elif ext in TABLE_ONLY:
            target = current if current is not self.geometry else self.processing
        elif ext in MESH_EXTENSIONS or ext in POINT_EXTENSIONS:
            target = current
        else:
            self.show_message(f"❌ Unsupported file type: {ext or path}", "error")
            return
        self.show_tab(target)
        if target is self.poisson:
            target.import_files(path)
        else:
            target.import_file(path)

    def open_dialog(self):
        tab = self.current_tab()
        if tab is self.poisson:
            tab.import_files()
        else:
            tab.import_file()

    def open_sample(self):
        self.show_tab(self.geometry)
        self.geometry.open_sample()

    # =========================
    # Drag & drop
    # =========================
    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            event.acceptProposedAction()
            self.open_path(urls[0].toLocalFile())

    # =========================
    # Status bar
    # =========================
    def _build_status_bar(self):
        bar = self.statusbar
        self.status_message = QLabel()
        self.status_file = QLabel()
        self.status_counts = QLabel()
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        bar.addWidget(self.status_message, 1)
        bar.addPermanentWidget(self.status_progress)
        bar.addPermanentWidget(self.status_file)
        bar.addPermanentWidget(self.status_counts)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(lambda: self.status_message.clear())

    def show_message(self, text, level="info"):
        """Coloured, self-clearing message on the left of the status bar."""
        color = theme.status_color(self.theme_name, level)
        self.status_message.setStyleSheet(f"color: {color};")
        self.status_message.setText(text.splitlines()[0] if text else "")
        self._message_timer.start(MESSAGE_TIMEOUT_MS.get(level, 6000))

    def update_tab_info(self, tab, **info):
        self._tab_info.setdefault(tab, {}).update(info)
        index = self.tabs.index(tab) if hasattr(self, "tabs") else None
        if index is not None:
            has_data = bool(self._tab_info[tab].get("points"))
            self.tabWidget.setTabIcon(
                index,
                self.style().standardIcon(QStyle.SP_DialogApplyButton)
                if has_data else QIcon())
        self._refresh_status()

    def _refresh_status(self, *_):
        if not hasattr(self, "tabs"):
            return
        info = self._tab_info.get(self.current_tab(), {})
        name = info.get("file")
        self.status_file.setText(name or "No file")
        self.status_file.setToolTip(name or "")
        parts = []
        if info.get("points"):
            parts.append(f"{info['points']:,} points")
        if info.get("triangles"):
            parts.append(f"{info['triangles']:,} triangles")
        self.status_counts.setText(" · ".join(parts))

    def set_busy(self, busy, text=""):
        """Indeterminate progress bar while background work runs."""
        self.status_progress.setVisible(busy)
        if busy:
            self.status_progress.setRange(0, 0)
            self.status_progress.setFormat("")
            if text:
                self.show_message(f"⏳ {text}")

    def set_progress(self, done, total, text=""):
        self.status_progress.setVisible(True)
        self.status_progress.setRange(0, total)
        self.status_progress.setValue(done)
        self.status_progress.setFormat(f"{done}/{total}")
        if text:
            self.show_message(text)

    # =========================
    # Theme
    # =========================
    def set_theme(self, name):
        self.theme_name = name
        theme.apply_theme(QApplication.instance(), name)
        settings.set_theme(name)
        for tab in self.tabs:
            tab.viewport.apply_theme(name)
            refresh = getattr(tab, "refresh_theme", None)
            if refresh:
                refresh()
        if hasattr(self, "action_dark"):
            self.action_dark.setChecked(name == theme.DARK)

    def apply_view_settings(self, kind, values):
        """Object / background settings were saved: apply to every tab."""
        for tab in self.tabs:
            tab.viewport.apply_view_settings(kind, values)
        self.show_message(
            ("Object" if kind == "object" else "Background")
            + " settings saved", "success")

    def show_mouse_guide(self, visible):
        for tab in self.tabs:
            tab.viewport.mouse_guide.setVisible(visible)

    def toggle_theme(self, dark):
        self.set_theme(theme.DARK if dark else theme.LIGHT)

    # =========================
    # Menus
    # =========================
    def _action(self, menu, text, slot, shortcut=None, icon=None, tip=""):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if tip:
            action.setStatusTip(tip)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self):
        bar = self.menuBar()
        file_menu = self.menuFile
        file_menu.clear()
        file_menu.setTitle("&File")
        self._action(file_menu, "&Open…", self.open_dialog, "Ctrl+O",
                     QStyle.SP_DialogOpenButton)
        self._action(file_menu, "Open &Sample (Stanford bunny)", self.open_sample,
                     "Ctrl+Shift+O")
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        file_menu.addSeparator()
        self._action(file_menu, "&Export…", lambda: self.current_tab().export_data(),
                     "Ctrl+S", QStyle.SP_DialogSaveButton)
        self._action(file_menu, "Save Screens&hot…",
                     lambda: self.current_tab().viewport.save_screenshot(),
                     "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._action(file_menu, "Re&move Data from This Tab",
                     lambda: self.current_tab().remove_data(), "Ctrl+W",
                     QStyle.SP_TrashIcon,
                     "Remove the current tab's data and free its memory")
        self._action(file_menu, "Remove Data from &All Tabs",
                     self.remove_all_data, "Ctrl+Shift+W",
                     tip="Empty every tab and free the memory")
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, "Ctrl+Q")

        view_menu = bar.addMenu("&View")
        self._action(view_menu, "&Reset View",
                     lambda: self.current_tab().viewport.reset_camera(),
                     "Ctrl+0", QStyle.SP_BrowserReload)
        for number, (name, (direction, up)) in enumerate(CAMERA_VIEWS.items(), 1):
            self._action(
                view_menu, f"{name} View",
                lambda _=False, d=direction, u=up:
                    self.current_tab().viewport.set_view(d, u),
                f"Ctrl+{number}")
        view_menu.addSeparator()
        self.action_mouse_guide = self._action(
            view_menu, "&Mouse Guide", self.show_mouse_guide,
            tip="Show / hide the mouse help under the 3D views")
        self.action_mouse_guide.setCheckable(True)
        self.action_mouse_guide.setChecked(True)
        for tab in self.tabs:
            # ✕ on one strip hides them all (and unticks the menu)
            tab.viewport.mouse_guide.close_button.clicked.connect(
                lambda: self.action_mouse_guide.isChecked()
                and self.action_mouse_guide.trigger())
        self.action_dark = self._action(view_menu, "&Dark Theme",
                                        self.toggle_theme, "Ctrl+T")
        self.action_dark.setCheckable(True)
        self.action_dark.setChecked(self.theme_name == theme.DARK)
        view_menu.addSeparator()
        self._action(view_menu, "&Quality Report", self.open_quality_report,
                     "Ctrl+R", QStyle.SP_FileDialogDetailedView)

        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "&Keyboard Shortcuts", self.show_shortcuts, "F1")
        self._action(help_menu, "&About", self.show_about)

    def remove_all_data(self):
        """Empty every tab (one confirmation) and free the memory."""
        loaded = [tab for tab in self.tabs if tab.has_data()]
        if not loaded:
            self.show_message("Nothing to remove - all tabs are empty")
            return
        answer = QMessageBox.question(
            self, "Remove data",
            "Remove the data from all tabs?\n\n"
            "Unsaved results are lost; files on disk are not touched.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for tab in loaded:
            tab.clear_data()
        release_memory()
        self.show_message(f"🗑 Removed data from {len(loaded)} tab(s) - "
                          "memory freed", "success")

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        files = settings.recent_files()
        if not files:
            empty = self.recent_menu.addAction("No recent files")
            empty.setEnabled(False)
            return
        for path in files:
            action = self.recent_menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.setStatusTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_path(p))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear Recent Files").triggered.connect(
            settings.clear_recent)

    def open_quality_report(self):
        self.show_tab(self.poisson)
        self.poisson.show_report()

    def show_shortcuts(self):
        rows = "".join(f"<tr><td><b>{keys}</b>&nbsp;&nbsp;</td><td>{what}</td></tr>"
                       for keys, what in SHORTCUTS)
        QMessageBox.information(
            self, "Keyboard Shortcuts",
            f"<table cellpadding='3'>{rows}</table>"
            "<p>You can also drag & drop files onto the window.</p>")

    def show_about(self):
        QMessageBox.about(
            self, "About Model Reconstruction",
            "<h3>Model Reconstruction</h3>"
            "<p>Surface reconstruction from point clouds with Poisson, "
            "Ball Pivoting and Alpha Shapes (Open3D), rendered with VTK.</p>"
            "<p>Sobhan Shirvanloo</p>")


# ---------- Main ----------
if __name__ == "__main__":
    window, app = create_splash_and_show(MainWindow)
    sys.exit(app.exec_())
