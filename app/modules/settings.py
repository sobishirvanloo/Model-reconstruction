"""Persistent user preferences (QSettings): theme, last folder, recent files."""

import json
import os

from PyQt5.QtCore import QSettings

MAX_RECENT = 8


def _store():
    return QSettings("ModelReconstruction", "ModelReconstruction")


def theme(default="light"):
    value = _store().value("theme", default)
    return value if value in ("light", "dark") else default


def set_theme(name):
    _store().setValue("theme", name)


def last_dir():
    """Folder of the last opened/saved file (falls back to the home dir)."""
    path = _store().value("last_dir", "")
    return path if path and os.path.isdir(path) else os.path.expanduser("~")


def remember_path(path):
    """Remember the folder of ``path`` for the next file dialog."""
    if path:
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if folder:
            _store().setValue("last_dir", folder)


def recent_files():
    files = _store().value("recent_files", []) or []
    if isinstance(files, str):          # a single entry comes back as str
        files = [files]
    return [f for f in files if os.path.isfile(f)]


def add_recent(path):
    path = os.path.abspath(path)
    files = [f for f in recent_files() if f != path]
    _store().setValue("recent_files", [path] + files[:MAX_RECENT - 1])
    remember_path(path)


def clear_recent():
    _store().setValue("recent_files", [])


def view_settings(kind):
    """Saved appearance settings (``"object"`` / ``"background"``) as a
    dict - empty when nothing was saved or the value is unreadable."""
    try:
        values = json.loads(_store().value(f"view/{kind}", "") or "{}")
    except (TypeError, ValueError):
        return {}
    return values if isinstance(values, dict) else {}


def set_view_settings(kind, values):
    _store().setValue(f"view/{kind}", json.dumps(values))


def layout_state(key):
    """Saved splitter states of a tab: {"panel": bytes, "console": bytes}."""
    state = {}
    for name in ("panel", "console"):
        value = _store().value(f"layout/{key}/{name}")
        if value:
            state[name] = value
    return state


def set_layout_state(key, state):
    for name, value in state.items():
        _store().setValue(f"layout/{key}/{name}", value)


def clear_layout_state():
    _store().remove("layout")
