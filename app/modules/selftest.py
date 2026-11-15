# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Self-test of a build (``MeshBench.exe --selftest`` / ``--selftest-gui``).

Used by the Windows build on GitHub Actions to check that the packaged
program works: every module imports, the helper process starts from the
executable and reconstructs, the comparison and the slice run; with the
GUI variant the main window opens, the sample bunny is reconstructed and
the program closes.  The result is written to ``meshbench-selftest.log``
in the working folder (a windowed build has no console); the exit code is
0 on success."""

import os
import sys
import time
import traceback

LOG = os.path.join(os.getcwd(), "meshbench-selftest.log")


def _log(text):
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def headless():
    start = time.time()
    try:
        import numpy as np
        import open3d as o3d
        from modules import (alignment, comparison_metrics, isolated, paths,
                             pipeline, slicing)
        _log(f"python {sys.version.split()[0]} frozen={paths.FROZEN} "
             f"open3d {o3d.__version__}")
        sphere = o3d.geometry.TriangleMesh.create_sphere(1.0, 40)
        sphere.compute_vertex_normals()
        cloud = sphere.sample_points_uniformly(20000)
        points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
        reference = (np.asarray(sphere.vertices), np.asarray(sphere.triangles))
        params = dict(algorithm="poisson", factor=None, depth=7, width=0.0,
                      scale=1.1, linear_fit=False, preprocess=True, voxel=0.0,
                      density_q=0.0, crop=True, smooth=2)
        result = pipeline.build_mesh(params, points, normals, reference, True)
        f_score = result["metrics"]["f_score"]
        _log(f"helper reconstruction: {len(result['triangles'])} triangles, F {f_score:.3f}")
        assert f_score > 0.9, "reconstruction quality too low"
        model = comparison_metrics.Model
        compared = model(result["vertices"], result["triangles"], result["normals"], "rec")
        original = model(reference[0], reference[1], None, "ref")
        out = comparison_metrics.run_comparison_isolated(compared, original, ("p2m",))
        _log(f"comparison p2m RMSE {out['results']['p2m']['stats']['rmse']:.4f}")
        half = slicing.slice_mesh(*reference, [slicing.axis_plane(
            "Z", reference[0].min(0), reference[0].max(0))])
        _log(f"slice: {len(half[1])} triangles")
        assert os.path.isfile(paths.sample_mesh()), "sample bunny missing"
        assert os.path.isfile(paths.resource("ui", "main.ui")), "main.ui missing"
        isolated.shutdown()
        _log(f"SELFTEST OK ({time.time() - start:.0f} s)")
        return 0
    except Exception:
        _log("SELFTEST FAILED\n" + traceback.format_exc())
        return 1


def gui(main_window_class, app):
    """Open the window, reconstruct the sample bunny, close."""
    from PyQt5.QtCore import QTimer
    state = {"start": time.time()}
    window = main_window_class()
    window.show()

    def step():
        try:
            tab = window.poisson
            elapsed = time.time() - state["start"]
            if "sent" not in state:
                window.geometry.open_sample()
                window.geometry.send_data("reconstruction")
                state["sent"] = True
            elif tab.mesh_o3d is not None and tab._worker is None:
                _log(f"GUI: reconstructed {len(tab._mesh_arrays[2])} triangles "
                     f"in {elapsed:.0f} s")
                _log("SELFTEST GUI OK")
                state["code"] = 0
                window.close()
                app.quit()
                return
            elif elapsed > 300:
                _log("SELFTEST GUI FAILED: no reconstruction after 300 s")
                state["code"] = 1
                window.close()
                app.quit()
                return
            QTimer.singleShot(500, step)
        except Exception:
            _log("SELFTEST GUI FAILED\n" + traceback.format_exc())
            state["code"] = 1
            app.quit()

    QTimer.singleShot(1000, step)
    app.exec_()
    return state.get("code", 1)
