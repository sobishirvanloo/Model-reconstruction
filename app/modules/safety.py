# MeshBench - Copyright (c) 2026 Sobhan Shirvanloo. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Crash protection and logging.

* An exception that escapes a Qt slot makes PyQt5 abort the whole
  program (core dump).  ``install`` replaces ``sys.excepthook`` (and the
  thread hook) so the error is written to the log, shown once in a
  dialog, and the application keeps running.
* Native crashes (segfaults in a C++ library) cannot be caught, but
  ``faulthandler`` writes their Python stack to ``crash.log`` so they can
  be diagnosed.
* VTK's warnings go to ``vtk.log`` instead of the console / a pop-up
  window.

Logs live in the per-user data folder (``log_folder``).
"""

import datetime
import faulthandler
import os
import sys
import threading
import traceback

from PyQt5.QtCore import QStandardPaths, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

_state = {"dialog_open": False, "crash_file": None}
MAX_LOG_BYTES = 2_000_000


def log_folder():
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    folder = os.path.join(base or os.path.expanduser("~"), "logs")
    os.makedirs(folder, exist_ok=True)
    return folder


def _trim(path):
    """Keep a log from growing without bound."""
    try:
        if os.path.getsize(path) > MAX_LOG_BYTES:
            os.replace(path, path + ".old")
    except OSError:
        pass


def write_log(text, name="errors.log"):
    path = os.path.join(log_folder(), name)
    _trim(path)
    try:
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            file.write(text)
    except OSError:
        pass
    return path


def _report(kind, value, tb):
    text = "".join(traceback.format_exception(kind, value, tb))
    path = write_log(text)
    sys.stderr.write(text)
    app = QApplication.instance()
    if app is None or _state["dialog_open"]:
        return
    # Show it from the event loop (never from inside the failing slot)
    def show():
        if _state["dialog_open"]:
            return
        _state["dialog_open"] = True
        try:
            QMessageBox.warning(
                app.activeWindow(), "MeshBench - unexpected error",
                f"An unexpected error occurred, the program keeps running:\n\n"
                f"{kind.__name__}: {value}\n\nDetails were saved to:\n{path}")
        finally:
            _state["dialog_open"] = False
    QTimer.singleShot(0, show)


def _excepthook(kind, value, tb):
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, value, tb)
        return
    _report(kind, value, tb)


def _thread_hook(args):
    if args.exc_type is SystemExit:
        return
    _report(args.exc_type, args.exc_value, args.exc_traceback)


def install():
    """Install the hooks (call once, after the QApplication exists)."""
    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    try:
        path = os.path.join(log_folder(), "crash.log")
        _trim(path)
        _state["crash_file"] = open(path, "a", encoding="utf-8")
        os.environ["MODELRECON_CRASH_LOG"] = path    # the helper logs here too
        faulthandler.enable(_state["crash_file"], all_threads=True)
    except OSError:
        faulthandler.enable()
    try:
        import vtk
        output = vtk.vtkFileOutputWindow()
        vtk_log = os.path.join(log_folder(), "vtk.log")
        _trim(vtk_log)
        output.SetFileName(vtk_log)
        output.AppendOn()
        vtk.vtkOutputWindow.SetInstance(output)
    except Exception:
        pass
