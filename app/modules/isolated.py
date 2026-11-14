"""Run crash-prone native code in a helper process.

Open3D's Poisson reconstruction occasionally segfaults inside the
library (an intermittent native crash on the worker thread).  A segfault
cannot be caught in-process - it would take the whole program down - so
``poisson`` runs in a separate, long-lived helper process: if Open3D
crashes there, only the helper dies; it is restarted and the run is
retried once.

The helper is a plain ``python -c`` that imports only this module
(numpy + Open3D, no Qt / VTK, and never the main script), started in the
background when the application opens (``warm_up``).  Requests and
results are pickled over its stdin / stdout.  In a frozen build (no
separate Python interpreter) the work simply runs in-process.
"""

import os
import pickle
import struct
import subprocess
import sys
import threading

import numpy as np

RETRIES = 1
_lock = threading.Lock()
_state = {"process": None}


# =========================
# Jobs (run in the helper)
# =========================
def _poisson_job(points, normals, depth, scale, linear_fit):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    return (np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy(),
            np.asarray(densities).copy())


JOBS = {"poisson": _poisson_job}


def _send(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _receive(stream):
    header = stream.read(8)
    if len(header) < 8:
        raise EOFError("helper closed")
    (size,) = struct.unpack("<Q", header)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError("helper closed")
    return pickle.loads(data)


def serve():
    """Helper loop: read (job, args), answer ("ok", result) / ("error", text)."""
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr            # stray prints must not corrupt the pipe
    import open3d  # noqa: F401  (load the library once, up front)
    _send(stdout, ("ready", None))
    while True:
        try:
            name, args = _receive(stdin)
        except EOFError:
            return
        try:
            _send(stdout, ("ok", JOBS[name](*args)))
        except Exception as e:
            _send(stdout, ("error", f"{type(e).__name__}: {e}"))


# =========================
# Client side (in the application)
# =========================
def _isolation_possible():
    return not getattr(sys, "frozen", False)


def _start():
    folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (f"import sys; sys.path.insert(0, {folder!r}); "
            "from modules.isolated import serve; serve()")
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    status, _ = _receive(process.stdout)       # wait for "ready"
    if status != "ready":
        raise RuntimeError("helper did not start")
    return process


def _process():
    process = _state["process"]
    if process is None or process.poll() is not None:
        process = _state["process"] = _start()
    return process


def _stop():
    process, _state["process"] = _state["process"], None
    if process is not None:
        try:
            process.stdin.close()
            process.wait(timeout=2)
        except Exception:
            process.kill()


def warm_up():
    """Start the helper in the background (hides its start-up time)."""
    if not _isolation_possible():
        return

    def start():
        with _lock:
            try:
                _process()
            except Exception:
                pass
    threading.Thread(target=start, name="isolated-warm-up", daemon=True).start()


def shutdown():
    with _lock:
        _stop()


def run(name, *args):
    """``JOBS[name](*args)`` in the helper; a crash there restarts it and
    retries ``RETRIES`` times before raising a clear error."""
    if not _isolation_possible():
        return JOBS[name](*args)
    with _lock:                                  # one request at a time
        for attempt in range(RETRIES + 1):
            try:
                process = _process()
                _send(process.stdin, (name, args))
                status, value = _receive(process.stdout)
            except (EOFError, BrokenPipeError, OSError):
                _stop()                          # the helper crashed
                if attempt == RETRIES:
                    raise RuntimeError(
                        "Open3D crashed while reconstructing (twice) - the "
                        "program is fine; try other parameters") from None
                continue
            if status == "error":
                raise RuntimeError(value)
            return value


def poisson(points, normals, depth, scale, linear_fit):
    """Open3D screened Poisson, isolated -> (vertices, triangles, densities)."""
    return run("poisson",
               np.ascontiguousarray(points, dtype=float),
               np.ascontiguousarray(normals, dtype=float),
               int(depth), float(scale), bool(linear_fit))
