# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Where the program's own files are - from the source tree, or from the
folder PyInstaller unpacks a Windows build into (``sys._MEIPASS``)."""

import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
HELPER_FLAG = "--meshbench-helper"      # MeshBench.exe started as the helper


def app_dir():
    """The ``app`` folder (``main.py``, ``ui/``, ``resources/``)."""
    if FROZEN:
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(*parts):
    return os.path.join(app_dir(), *parts)


def sample_mesh():
    """The Stanford bunny shipped with the program."""
    if FROZEN:
        return resource("models", "Stanford_Bunny.stl")
    return os.path.normpath(os.path.join(app_dir(), "..", "models",
                                         "Stanford_Bunny.stl"))
