# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Run crash-prone, GIL-holding native code in a helper process.

* Open3D occasionally segfaults inside the library (seen in Poisson; Ball
  Pivoting / alpha shapes are known to as well).  A segfault cannot be
  caught in-process - it would take the whole program down.
* Open3D also keeps Python's global lock (GIL) during its C++ calls, so
  the window froze for seconds per reconstruction even though the work
  ran on a background thread (measured: up to 4.5 s stalls, 60-90 % of
  every optimisation run).

So the whole reconstruction pipeline (``pipeline.build_mesh`` /
``evaluate_mesh``) runs in a separate, long-lived helper process: the
waiting thread holds no lock (the UI stays fluid), and if Open3D crashes
only the helper dies; it is restarted and the run is retried once.

The helper is a plain ``python -c`` that imports this module and the
pipeline (numpy, Open3D, VTK filters - no Qt, never the main script),
started in the background when the application opens (``warm_up``).
Requests and results are pickled over its stdin / stdout.  In a frozen
build (no separate Python interpreter) the work simply runs in-process.
"""

import os
import pickle
import struct
import subprocess
import sys
import threading
import time

import numpy as np

RETRIES = 1
_lock = threading.Lock()
_state = {"process": None, "in_helper": False, "closing": False}


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


def _build_job(*args):
    from modules.pipeline import build_job
    return build_job(*args)


def _evaluate_job(*args):
    from modules.pipeline import evaluate_job
    return evaluate_job(*args)


def _clear_job():
    from modules.pipeline import clear_job
    return clear_job()


def _compare_job(*args):
    from modules.comparison_metrics import compare_job
    return compare_job(*args)


def _compare_clear_job():
    from modules.comparison_metrics import compare_clear_job
    return compare_clear_job()


def _gpu_job(enabled):
    from modules import compute
    compute.set_gpu_enabled(enabled)


JOBS = {"poisson": _poisson_job, "build": _build_job,
        "evaluate": _evaluate_job, "clear": _clear_job, "gpu": _gpu_job,
        "compare": _compare_job, "compare_clear": _compare_clear_job}


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
    # (a windowed Windows build may start without sys.stdin / sys.stdout
    # objects although the pipes exist: open the descriptors directly)
    stdin = sys.stdin.buffer if sys.stdin is not None else os.fdopen(0, "rb")
    stdout = sys.stdout.buffer if sys.stdout is not None else os.fdopen(1, "wb")
    sys.stdout = sys.stderr            # stray prints must not corrupt the pipe
    _state["in_helper"] = True         # jobs run here, never in a sub-helper
    log = os.environ.get("MODELRECON_CRASH_LOG")
    if log:                            # native crashes of the helper -> crash.log
        try:
            import faulthandler
            _state["log"] = open(log, "a", encoding="utf-8")
            _state["log"].write("\n=== helper process started ===\n")
            _state["log"].flush()
            faulthandler.enable(_state["log"], all_threads=True)
        except OSError:
            pass
    from modules import compute
    compute.set_gpu_enabled(os.environ.get("MODELRECON_GPU", "1") == "1")
    import open3d  # noqa: F401  (load the libraries once, up front)
    import modules.pipeline  # noqa: F401
    _send(stdout, ("ready", None))
    while True:
        try:
            name, args = _receive(stdin)
        except EOFError:
            return
        try:
            _send(stdout, ("ok", JOBS[name](*args)))
        except Exception as e:
            _send(stdout, ("error", (type(e).__name__, str(e))))


# =========================
# Client side (in the application)
# =========================
def _isolation_possible():
    return not _state["in_helper"]


def _helper_command():
    """From the sources: this Python running ``serve``; in a Windows build
    (PyInstaller) there is no separate Python - MeshBench.exe itself is
    started again with the helper flag (``main.py`` checks it first)."""
    from modules import paths
    if paths.FROZEN:
        return [sys.executable, paths.HELPER_FLAG]
    folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (f"import sys; sys.path.insert(0, {folder!r}); "
            "from modules.isolated import serve; serve()")
    return [sys.executable, "-c", code]


def _start():
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # Windows: no console
    process = subprocess.Popen(
        _helper_command(), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
    status, _ = _receive(process.stdout)       # wait for "ready"
    if status != "ready":
        raise RuntimeError("helper did not start")
    return process


def _process():
    process = _state["process"]
    if process is None or process.poll() is not None:
        process = _state["process"] = _start()
    return process


def _kill(process):
    try:
        process.kill()
    except Exception:
        pass


def _stop():
    process, _state["process"] = _state["process"], None
    if process is not None:
        try:
            process.stdin.close()
            process.wait(timeout=2)
        except Exception:
            _kill(process)


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
    abort()
    with _lock:
        _stop()


class Stopped(RuntimeError):
    """The job was stopped on purpose (cancelled, closing)."""


class TooSlow(RuntimeError):
    """The job ran longer than its time limit and was stopped."""


def abort():
    """Stop the running job now (the program is closing): the helper is
    ended, the waiting call raises ``Stopped`` and no new helper starts."""
    _state["closing"] = True
    process = _state["process"]
    if process is not None:
        _kill(process)


def _wait_reply(process, timeout, cancel):
    """Read the reply, giving up when ``timeout`` (s) passes or ``cancel``
    is set - the helper is then ended (a native loop cannot be interrupted
    any other way) and a new one starts with the next job."""
    if timeout is None and cancel is None:
        return _receive(process.stdout)
    box = {}

    def read():
        try:
            box["reply"] = _receive(process.stdout)
        except BaseException as e:             # EOF when the helper ends
            box["error"] = e
    reader = threading.Thread(target=read, name="isolated-reply", daemon=True)
    reader.start()
    started = time.monotonic()
    while reader.is_alive():
        reader.join(0.1)
        if not reader.is_alive():
            break
        stop = None
        if cancel is not None and cancel.is_set():
            stop = Stopped("cancelled")
        elif _state.get("closing"):
            stop = Stopped("the program is closing")
        elif timeout is not None and time.monotonic() - started > timeout:
            stop = TooSlow(f"took longer than {timeout:.0f} s - stopped "
                           "(these settings are too slow for this data)")
        if stop is not None:
            _state["process"] = None
            _kill(process)
            raise stop
    if "error" in box:
        if _state.get("closing"):
            raise Stopped("the program is closing")
        raise box["error"]
    return box["reply"]


def run(name, *args, timeout=None, cancel=None):
    """``JOBS[name](*args)`` in the helper; a crash there restarts it and
    retries ``RETRIES`` times before raising a clear error.  ``timeout``
    (seconds) / ``cancel`` (an Event) stop a job that runs too long
    (``TooSlow``) or is no longer wanted (``Stopped``)."""
    if not _isolation_possible():
        return JOBS[name](*args)
    with _lock:                                  # one request at a time
        return _run_locked(name, args, timeout, cancel)


def _run_locked(name, args, timeout=None, cancel=None):
    queued, _state["queued"] = _state.get("queued", []), []
    for job in queued:                           # e.g. a GPU switch made while busy
        try:
            _run_once(*job)
        except Exception:
            pass
    return _run_once(name, args, timeout, cancel)


def _run_once(name, args, timeout=None, cancel=None):
    for attempt in range(RETRIES + 1):
        if _state.get("closing"):
            raise Stopped("the program is closing")
        try:
            process = _process()
            _send(process.stdin, (name, args))
            status, value = _wait_reply(process, timeout, cancel)
        except (EOFError, BrokenPipeError, OSError):
            _stop()                          # the helper crashed
            if attempt == RETRIES:
                raise RuntimeError(
                    "the reconstruction library crashed on this input "
                    "(twice) - the program is fine; try other "
                    "parameters") from None
            continue
        if status == "error":
            kind, message = value
            # keep the readable messages (e.g. "empty mesh - increase
            # the radius") as they were in-process
            raise (ValueError(message) if kind == "ValueError"
                   else RuntimeError(f"{kind}: {message}"))
        return value


def run_if_started(name, *args):
    """A small job (clear the caches, switch the GPU) for a running
    helper - never starts one just for that, and never waits: while a job
    runs it is queued and sent before the next one."""
    if not _isolation_possible():
        return JOBS[name](*args)
    process = _state["process"]
    if process is None or process.poll() is not None:
        return None
    if not _lock.acquire(blocking=False):
        _state.setdefault("queued", []).append((name, args))
        return None
    try:
        return _run_once(name, args)
    except Exception:
        return None
    finally:
        _lock.release()


def poisson(points, normals, depth, scale, linear_fit):
    """Open3D screened Poisson, isolated -> (vertices, triangles, densities)."""
    return run("poisson",
               np.ascontiguousarray(points, dtype=float),
               np.ascontiguousarray(normals, dtype=float),
               int(depth), float(scale), bool(linear_fit))
