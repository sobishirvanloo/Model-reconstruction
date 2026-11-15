"""Reconstruction pipeline - pure computation, no Qt.

    optional voxel downsampling
    -> statistical outlier removal
    -> consistent normal orientation (global sign kept from the input)
    -> reconstruction (Poisson / Ball Pivoting / Alpha Shape / RIMLS /
       Signed Distance)
    -> Poisson only: density filtering + optional crop to the bounding box
    -> optional Taubin smoothing
    -> quality evaluation (see ``quality_metrics``)

``build_mesh`` and ``evaluate_mesh`` run in the helper process
(``isolated``): Open3D keeps Python's global lock (GIL) during its C++
calls, which froze the window for seconds per run while optimising, and a
native crash in any algorithm (e.g. Ball Pivoting) only ends the helper -
it is restarted and the program keeps running.  The ``*_local`` versions
do the work in the calling process (inside the helper, frozen builds).
"""

import hashlib
import math
import time
from collections import OrderedDict

import numpy as np
import open3d as o3d

from modules import isolated
from modules.geometry_utils import normalize, orient_consistently
from modules.implicit_reconstruction import (
    reconstruct_rimls as _rimls,
    reconstruct_signed_distance as _signed_distance,
)
from modules.quality_metrics import evaluate

MAX_DEPTH = 12

POISSON = "poisson"
BALL_PIVOTING = "ball_pivoting"
ALPHA_SHAPE = "alpha_shape"
RIMLS = "rimls"
SIGNED_DISTANCE = "signed_distance"
# Default size factors (x mean point spacing), chosen on the Stanford bunny:
# BPA x1 was 7x slower with 48 pieces, x2 gave F 0.95; alpha x8-16 was best;
# RIMLS x6 beat x4 (Chamfer 0.43 vs 0.44 %, same time); signed distance
# x5 beat x3 (0.44 vs 0.48 %, and 0.53 vs 0.71 % with 0.5 % noise).
DEFAULT_FACTORS = {BALL_PIVOTING: 2.0, ALPHA_SHAPE: 10.0, RIMLS: 6.0,
                   SIGNED_DISTANCE: 5.0}


# =========================
# Pipeline (pure functions - safe to run in the worker thread)
# =========================
def preprocess(points, normals, enabled, voxel_size):
    """Normalize normals, optionally voxel-downsample, remove outliers
    and orient the normals consistently."""
    normals = normalize(normals)
    if not enabled:
        return points, normals

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Optional voxel downsampling (also averages the normals)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.normals = o3d.utility.Vector3dVector(
            normalize(np.asarray(pcd.normals))
        )

    # Statistical outlier removal (keeps the matching normals)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Consistent orientation, without flipping the surface inside out
    reference = np.asarray(pcd.normals).copy()
    orient_consistently(pcd, reference)

    return np.asarray(pcd.points), np.asarray(pcd.normals)


def array_key(*arrays):
    """Content hash of numpy arrays (cache key)."""
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


# Preprocessing (outlier removal + normal orientation) is the slowest stage
# (~3 s on the bunny) but depends only on the input and two options, not on
# the reconstruction parameters: keep the last few results.  Only the
# worker thread touches the caches, and jobs run one at a time.
PREPROCESS_CACHE_SIZE = 4
_preprocess_cache = OrderedDict()


def cached_preprocess(points, normals, enabled, voxel_size):
    """``preprocess`` with an LRU cache -> (points, normals, was_cached)."""
    key = (array_key(points, normals), bool(enabled),
           float(voxel_size) if enabled else 0.0)
    if key in _preprocess_cache:
        _preprocess_cache.move_to_end(key)
        pre_points, pre_normals = _preprocess_cache[key]
        return pre_points, pre_normals, True

    pre_points, pre_normals = preprocess(points, normals, enabled, voxel_size)
    _preprocess_cache[key] = (np.array(pre_points), np.array(pre_normals))
    while len(_preprocess_cache) > PREPROCESS_CACHE_SIZE:
        _preprocess_cache.popitem(last=False)
    pre_points, pre_normals = _preprocess_cache[key]
    return pre_points, pre_normals, False


def clear_caches():
    """Forget the cached preprocessing results and tetrahedralisation."""
    _preprocess_cache.clear()
    _tetra_cache.clear()


def depth_for_width(points, width, scale):
    """Octree depth whose finest cell is ~``width`` (PoissonRecon's rule).

    Open3D ignores ``width`` whenever ``depth`` is given, so the conversion
    is done here to make the *Width* control meaningful.
    """
    extent = float(np.max(points.max(axis=0) - points.min(axis=0))) * scale
    if extent <= 0:
        return 1
    depth = math.ceil(math.log2(extent / width))
    return int(min(max(depth, 1), MAX_DEPTH))


def reference_to_mesh(reference):
    """``(vertices, triangles)`` -> Open3D mesh with normals."""
    vertices, triangles = reference
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def evaluate_mesh_local(mesh, reference, points, normals):
    """Quality metrics vs. the reference mesh (or the input points)."""
    start = time.perf_counter()
    if reference is not None:
        metrics, errors = evaluate(
            mesh, reference_mesh=reference_to_mesh(reference), input_points=points
        )
    else:
        metrics, errors = evaluate(
            mesh, reference_points=points, reference_normals=normalize(normals)
        )
    return {
        "metrics": metrics,
        "vertex_errors": errors,
        "eval_time": time.perf_counter() - start,
    }


def mean_spacing(pcd):
    """Mean distance from each point to its nearest neighbour."""
    spacing = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    if spacing <= 0:
        raise ValueError("Cannot measure the point spacing (duplicate points?)")
    return spacing


def reconstruct_poisson(pcd, params):
    depth = params["depth"]
    if params["width"] > 0:
        depth = depth_for_width(
            np.asarray(pcd.points), params["width"], params["scale"]
        )
    # In a helper process: Open3D's Poisson occasionally segfaults, which
    # would otherwise take the whole program down (see ``isolated``)
    vertices, triangles, densities = isolated.poisson(
        np.asarray(pcd.points), np.asarray(pcd.normals), depth,
        params["scale"], params["linear_fit"])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles.astype(np.int32)))

    # --- density filtering ---
    densities = np.asarray(densities)
    q = params["density_q"]
    if q > 0 and len(densities) == len(mesh.vertices):
        threshold = np.quantile(densities, q)
        mesh.remove_vertices_by_mask(densities < threshold)

    # --- crop to the point cloud bounding box ---
    if params["crop"]:
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    return mesh, {"depth": depth, "setting": f"depth {depth}"}


def reconstruct_ball_pivoting(pcd, params):
    spacing = mean_spacing(pcd)
    radius = params["factor"] * spacing
    radii = [radius, 2 * radius, 4 * radius]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh, {"spacing": spacing, "radii": radii,
                  "setting": f"r = {radius:.4g} (×{params['factor']:g})"}


# The Delaunay tetrahedralisation is the slow part of an alpha shape and
# does not depend on alpha: keep the last one (jobs run one at a time).
_tetra_cache = {}


def _tetra_mesh(pcd):
    key = array_key(np.asarray(pcd.points))
    if key not in _tetra_cache:
        _tetra_cache.clear()
        _tetra_cache[key] = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    return _tetra_cache[key]


def reconstruct_alpha_shape(pcd, params):
    spacing = mean_spacing(pcd)
    alpha = params["factor"] * spacing
    tetra, point_map = _tetra_mesh(pcd)
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra, point_map
        )
    except (IndexError, RuntimeError):
        # Open3D fails with a bare "_Map_base::at" when nothing survives
        mesh = o3d.geometry.TriangleMesh()
    return mesh, {"spacing": spacing, "alpha": alpha,
                  "setting": f"α = {alpha:.4g} (×{params['factor']:g})"}


def reconstruct_rimls(pcd, params):
    """Robust implicit MLS (see ``implicit_reconstruction``)."""
    spacing = mean_spacing(pcd)
    return _rimls(np.asarray(pcd.points), np.asarray(pcd.normals),
                  spacing, params["factor"])


def reconstruct_signed_distance(pcd, params):
    """Volumetric signed distance (see ``implicit_reconstruction``)."""
    spacing = mean_spacing(pcd)
    return _signed_distance(np.asarray(pcd.points), np.asarray(pcd.normals),
                            spacing, params["factor"])


RECONSTRUCTORS = {
    POISSON: reconstruct_poisson,
    BALL_PIVOTING: reconstruct_ball_pivoting,
    ALPHA_SHAPE: reconstruct_alpha_shape,
    RIMLS: reconstruct_rimls,
    SIGNED_DISTANCE: reconstruct_signed_distance,
}


def build_mesh_local(params, points, normals, reference=None,
                     run_evaluation=False):
    """Full pipeline -> result dict (mesh, arrays for rendering, timings
    and - when ``run_evaluation`` - quality metrics)."""
    timings = {}
    clock = time.perf_counter()

    pre_points, pre_normals, preprocess_cached = cached_preprocess(
        points, normals, params["preprocess"], params["voxel"]
    )
    if len(pre_points) < 4:
        raise ValueError("Too few points left after preprocessing")
    timings["preprocess"] = time.perf_counter() - clock
    clock = time.perf_counter()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pre_points)
    pcd.normals = o3d.utility.Vector3dVector(pre_normals)

    algorithm = params.get("algorithm", POISSON)
    mesh, info = RECONSTRUCTORS[algorithm](pcd, params)
    timings["reconstruct"] = time.perf_counter() - clock
    clock = time.perf_counter()

    # --- smoothing ---
    if params["smooth"] > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=params["smooth"]
        )

    if len(mesh.triangles) == 0:
        hint = {
            POISSON: "lower the density filter or disable cropping",
            BALL_PIVOTING: "increase the ball radius factor",
            ALPHA_SHAPE: "increase the alpha factor",
            RIMLS: "increase the kernel radius",
            SIGNED_DISTANCE: "increase the influence radius",
        }[algorithm]
        raise ValueError(f"Reconstruction produced an empty mesh - {hint}")

    mesh.compute_vertex_normals()
    timings["postprocess"] = time.perf_counter() - clock

    result = {
        "mesh": mesh,
        "vertices": np.asarray(mesh.vertices).copy(),
        "normals": np.asarray(mesh.vertex_normals).copy(),
        "triangles": np.asarray(mesh.triangles).copy(),
        "points_after_preprocess": len(pre_points),
        "preprocess_cached": preprocess_cached,
        "algorithm": algorithm,
        "depth": info.get("depth"),
        "setting": info["setting"],
        "timings": timings,
    }
    if run_evaluation:
        # Evaluate against the ORIGINAL input (before preprocessing)
        evaluation = evaluate_mesh_local(mesh, reference, points, normals)
        timings["evaluate"] = evaluation.pop("eval_time")
        result.update(evaluation)
    return result



# =========================
# Isolated entry points (run in the helper process)
# =========================
def mesh_from_arrays(vertices, triangles, normals=None):
    """Open3D mesh from numpy arrays (vertex normals when given)."""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    if normals is not None and len(normals) == len(vertices):
        mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.asarray(normals, dtype=float))
    else:
        mesh.compute_vertex_normals()
    return mesh


MESH_KEYS = ("vertices", "normals", "triangles", "vertex_errors")


def build_job(params, points, normals, reference, run_evaluation,
              keep_mesh=True):
    """``build_mesh_local`` without the Open3D mesh (not picklable); the
    arrays it is made of are in the result - unless ``keep_mesh`` is off
    (the optimiser only needs the numbers: no mesh crosses the pipe)."""
    result = build_mesh_local(params, points, normals, reference,
                              run_evaluation)
    result.pop("mesh")
    if not keep_mesh:
        result["triangle_count"] = len(result["triangles"])
        for key in MESH_KEYS:
            result.pop(key, None)
    return result


def evaluate_job(vertices, triangles, vertex_normals, reference, points,
                 normals):
    mesh = mesh_from_arrays(vertices, triangles, vertex_normals)
    return evaluate_mesh_local(mesh, reference, points, normals)


def clear_job():
    clear_caches()


def _arrays(*values):
    return [None if v is None else np.ascontiguousarray(v) for v in values]


def build_mesh(params, points, normals, reference=None, run_evaluation=False,
               timeout=None, cancel=None, keep_mesh=True):
    """Full pipeline in the helper process -> result dict (``mesh``,
    arrays for rendering, timings, metrics when ``run_evaluation``).
    ``timeout`` / ``cancel`` stop it (``isolated.TooSlow`` / ``Stopped``)."""
    points, normals = _arrays(points, normals)
    if reference is not None:
        reference = tuple(_arrays(*reference))
    result = isolated.run("build", dict(params), points, normals, reference,
                          bool(run_evaluation), bool(keep_mesh),
                          timeout=timeout, cancel=cancel)
    if keep_mesh:
        result["mesh"] = mesh_from_arrays(
            result["vertices"], result["triangles"], result["normals"])
    return result


def evaluate_mesh(mesh, reference, points, normals, timeout=None, cancel=None):
    """Quality metrics of ``mesh`` (in the helper process)."""
    if reference is not None:
        reference = tuple(_arrays(*reference))
    vertex_normals = (np.asarray(mesh.vertex_normals)
                      if mesh.has_vertex_normals() else None)
    return isolated.run("evaluate", *_arrays(
        np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals),
        reference, *_arrays(points, normals), timeout=timeout, cancel=cancel)


def clear_all_caches():
    """Forget the cached preprocessing here and in the helper."""
    clear_caches()
    isolated.run_if_started("clear")
