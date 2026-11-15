# MeshBench - Copyright (c) 2026 Sobhan Shirvanloo. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Shared compute helpers: GPU (CUDA) neighbour search with a CPU
fallback, precise float32 queries for large coordinates, free memory.

* ``KNN`` - nearest-neighbour search (Open3D ``core.nns``).  It runs on
  the GPU when Open3D was built with CUDA and a card is present (and the
  user did not switch it off, *View ▸ Use GPU*), otherwise on all CPU
  cores.  Any GPU failure (out of GPU memory, unsupported query, ...)
  silently falls back to the CPU for the rest of the session.
* Coordinates are **centred** before they become float32: with
  georeferenced data (coordinates around 10⁶) float32 keeps only ~0.06
  units of precision, which made distances 6-16 x too large.
* ``available_memory`` - free RAM in bytes (``None`` when unknown), used
  to keep big grids within memory.
"""

import os
import threading

import numpy as np
import open3d as o3d

_lock = threading.Lock()
_state = {"enabled": True, "failed": False}


def cuda_available():
    try:
        return bool(o3d.core.cuda.is_available())
    except Exception:
        return False


def set_gpu_enabled(enabled):
    _state["enabled"] = bool(enabled)
    # inherited by the helper process (``isolated``) when it starts
    os.environ["MODELRECON_GPU"] = "1" if enabled else "0"


def gpu_in_use():
    return _state["enabled"] and not _state["failed"] and cuda_available()


def device_name():
    if gpu_in_use():
        return f"GPU (CUDA, {o3d.core.cuda.device_count()} device(s))"
    if cuda_available() and _state["failed"]:
        return "CPU (the GPU failed, fell back)"
    return f"CPU ({os.cpu_count() or 1} threads)"


def _device():
    return o3d.core.Device("CUDA:0") if gpu_in_use() else o3d.core.Device("CPU:0")


def _gpu_failed():
    with _lock:
        _state["failed"] = True


def available_memory():
    """Free physical memory in bytes, or ``None`` when it is unknown."""
    try:
        with open("/proc/meminfo") as file:              # Linux
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:                                                 # macOS / BSD
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    try:                                                 # Windows
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                        ("pagefile", ctypes.c_ulonglong),
                        ("free_pagefile", ctypes.c_ulonglong),
                        ("virtual", ctypes.c_ulonglong),
                        ("free_virtual", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.free)
    except Exception:
        pass
    return None


class KNN:
    """Nearest neighbours of query points among ``points``.

    The point set is centred on its mean before the float32 conversion;
    queries are shifted the same way, so precision does not depend on
    where the model sits in space.
    """

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        self.origin = self.points.mean(axis=0) if len(self.points) else np.zeros(3)
        self._local = np.ascontiguousarray(self.points - self.origin, dtype=np.float32)
        self._search = None
        self._knn_ready = False
        self._radius = None
        self._build()

    def _build(self):
        device = _device()
        self._search = o3d.core.nns.NearestNeighborSearch(
            o3d.core.Tensor(self._local, device=device))
        self._knn_ready = False
        self._radius = None
        self._on_gpu = device.get_type() == o3d.core.Device.DeviceType.CUDA

    def _query(self, query):
        local = np.ascontiguousarray(np.asarray(query, dtype=float) - self.origin,
                                     dtype=np.float32)
        return o3d.core.Tensor(local, device=_device() if self._on_gpu
                               else o3d.core.Device("CPU:0"))

    def _run(self, call):
        try:
            return call()
        except Exception:
            if not self._on_gpu:
                raise
            _gpu_failed()              # e.g. out of GPU memory: use the CPU
            self._build()
            return call()

    def knn(self, query, k, distances=False):
        """Indices (n, k) - and float64 distances when ``distances``."""
        k = max(1, min(int(k), len(self.points)))

        def call():
            if not self._knn_ready:
                self._search.knn_index()
                self._knn_ready = True
            index, _ = self._search.knn_search(self._query(query), k)
            return index.cpu().numpy().astype(np.int64).reshape(len(query), k)

        index = self._run(call)
        if not distances:
            return index
        diff = np.asarray(query, dtype=float)[:, None, :] - self.points[index]
        return index, np.sqrt(np.einsum("nkj,nkj->nk", diff, diff))

    def radius(self, query, radius, max_nn):
        """Up to ``max_nn`` neighbours within ``radius`` (-1 = none)."""

        def call():
            if self._radius is None or radius > self._radius:
                self._search.hybrid_index(radius)
                self._radius = radius
            index, _, _ = self._search.hybrid_search(
                self._query(query), radius, max_nn)
            return index.cpu().numpy().astype(np.int64).reshape(len(query), max_nn)

        return self._run(call)


def centred(*arrays, origin=None):
    """(origin, arrays shifted by -origin): float32-safe coordinates."""
    if origin is None:
        stacked = [np.asarray(a, dtype=float) for a in arrays if a is not None and len(a)]
        origin = (np.concatenate(stacked).mean(axis=0) if stacked else np.zeros(3))
    return origin, [None if a is None else np.asarray(a, dtype=float) - origin
                    for a in arrays]
