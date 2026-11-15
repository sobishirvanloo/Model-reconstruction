# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Model-to-model comparison: a *compared* model (the reconstruction) is
measured against a *reference* (the original mesh).

Seven distance methods (all distances in model units):

==========  ======================  ======================================
key         name                    distance of each compared point to …
==========  ======================  ======================================
p2p         Point-to-point          the nearest reference vertex
p2pl        Point-to-plane          the tangent plane of the nearest
                                    reference vertex (signed, along its
                                    normal)
p2m         Point-to-mesh           the exact closest point on the
                                    reference triangles (signed)
c2c         Cloud-to-cloud          the nearest point of a dense uniform
                                    sampling of the reference surface;
                                    compared side sampled the same way
ac2c        Adaptive cloud-to-      a least-squares plane fitted to the
            cloud                   reference neighbours, the neighbourhood
                                    (6 / 12 / 24 points) chosen per point as
                                    the largest one that is still planar
c2m         Cloud-to-mesh           the reference triangles, for the dense
                                    sampling of the compared surface
m3c2        M3C2                    Lague et al. 2013: per core point, a
                                    normal fitted at the most planar of
                                    several scales, then the mean position
                                    of each cloud inside a cylinder along
                                    that normal; distance = difference of
                                    the means, with a 95% level of
                                    detection
==========  ======================  ======================================

Point-to-* methods use the vertices of the compared model; the cloud
methods use ``samples`` points drawn uniformly over both surfaces, which
makes them independent of how the meshes are triangulated.

Efficiency: nearest-neighbour searches use Open3D's multithreaded KD-tree
(``core.nns``), exact surface distances its BVH ``RaycastingScene``;
local plane fits are vectorised PCA (cumulative sums give every
neighbourhood size at once), processed in chunks so memory stays bounded.

Error statistics of a method (``error_stats``), over its values ``e``:

    mean e   mean of e (signed methods: bias inside / outside)
    SD       standard deviation of e
    MAE      mean |e|
    RMSE     sqrt(mean e²)
    R²       1 − Σe² / Σ‖q − q̄‖², q = matched reference positions: the
             share of the reference shape's spatial variance that the
             compared model reproduces (1 = perfect)
"""

import math
import time

import numpy as np
import open3d as o3d

from modules.compute import KNN
from modules.geometry_utils import estimate_oriented_normals, normalize

METHODS = {
    "p2p": ("Point-to-point", False,
            "Vertex to the nearest reference vertex"),
    "p2pl": ("Point-to-plane", True,
             "Vertex to the tangent plane of the nearest reference vertex"),
    "p2m": ("Point-to-mesh", True,
            "Vertex to the exact closest point on the reference triangles"),
    "c2c": ("Cloud-to-cloud", False,
            "Dense surface samples to the nearest reference sample"),
    "ac2c": ("Adaptive cloud-to-cloud", True,
             "Samples to a local plane fitted to 6-24 reference samples "
             "(largest planar neighbourhood)"),
    "c2m": ("Cloud-to-mesh", True,
            "Dense surface samples to the reference triangles"),
    "m3c2": ("M3C2", True,
             "Multiscale model-to-model cloud comparison along local "
             "normals, with a 95% level of detection"),
}
ORDER = tuple(METHODS)
MESH_METHODS = {"p2m", "c2m"}          # need reference triangles

DEFAULT_PARAMS = {
    "samples": 100_000,      # points per surface for the cloud methods
    "core_points": 20_000,   # M3C2 core points
    "m3c2_diameter": 6.0,    # projection diameter, x mean sample spacing
    # max cylinder half-length, % of the diagonal: just above the expected
    # change - a long cylinder reaches the far side of thin parts (on the
    # bunny's ears 5% raised the RMSE from 0.02 to 0.31)
    "m3c2_depth_pct": 1.0,
    "seed": 0,
}
ADAPTIVE_K = (6, 12, 24)
ADAPTIVE_PLANARITY = 0.01            # surface variation of a "planar" fit
M3C2_NORMAL_K = (12, 24, 48, 96)     # normal scales (neighbour counts)
CHUNK = 5000


# =========================
# Models and spatial indices
# =========================
class Model:
    """Vertices (+ triangles) with vertex normals; samples on demand."""

    def __init__(self, vertices, triangles=None, normals=None, name=""):
        self.vertices = np.ascontiguousarray(vertices, dtype=float)
        self.triangles = (None if triangles is None or len(triangles) == 0
                          else np.ascontiguousarray(triangles, dtype=np.int64))
        self.name = name
        self.mesh = None
        if self.triangles is not None:
            self.mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self.vertices),
                o3d.utility.Vector3iVector(self.triangles.astype(np.int32)))
        if normals is not None and len(normals) == len(self.vertices):
            self.normals = normalize(np.asarray(normals, dtype=float))
        elif self.mesh is not None:
            self.mesh.compute_vertex_normals()
            self.normals = normalize(np.asarray(self.mesh.vertex_normals))
        else:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.vertices))
            self.normals = np.asarray(estimate_oriented_normals(pcd).normals).copy()
        low, high = self.vertices.min(axis=0), self.vertices.max(axis=0)
        self.diagonal = float(np.linalg.norm(high - low))

    @property
    def has_triangles(self):
        return self.triangles is not None

    def translated(self, offset):
        """Same model moved by ``offset`` (normals reused, not recomputed)."""
        model = Model.__new__(Model)
        model.vertices = self.vertices + offset
        model.triangles = self.triangles
        model.normals = self.normals
        model.name = self.name
        model.diagonal = self.diagonal
        model.mesh = None
        if self.triangles is not None:
            model.mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(model.vertices),
                o3d.utility.Vector3iVector(self.triangles.astype(np.int32)))
        return model

    def samples(self, count, seed=0):
        """``count`` points spread uniformly over the surface (with the
        triangle normals); a point cloud is subsampled instead."""
        if self.mesh is None:
            if len(self.vertices) <= count:
                return self.vertices, self.normals
            pick = np.random.default_rng(seed).choice(
                len(self.vertices), count, replace=False)
            return self.vertices[pick], self.normals[pick]
        o3d.utility.random.seed(seed)
        pcd = self.mesh.sample_points_uniformly(count, use_triangle_normal=True)
        return (np.asarray(pcd.points).copy(),
                normalize(np.asarray(pcd.normals)))


# KD-tree over a point set: GPU when available, else all CPU cores
_Index = KNN


class _Context:
    """Lazily built samples / indices shared by the methods of one run."""

    def __init__(self, compared, reference, params):
        self.compared = compared
        self.reference = reference
        self.params = params
        self._cache = {}

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def ref_vertex_index(self):
        return self._get("rvi", lambda: _Index(self.reference.vertices))

    @property
    def ref_samples(self):
        return self._get("rs", lambda: self.reference.samples(
            self.params["samples"], self.params["seed"]))

    @property
    def cmp_samples(self):
        return self._get("cs", lambda: self.compared.samples(
            self.params["samples"], self.params["seed"] + 1))

    @property
    def ref_sample_index(self):
        return self._get("rsi", lambda: _Index(self.ref_samples[0]))

    @property
    def cmp_sample_index(self):
        return self._get("csi", lambda: _Index(self.cmp_samples[0]))

    @property
    def scene(self):
        def build():
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(
                o3d.t.geometry.TriangleMesh.from_legacy(self.reference.mesh))
            return scene
        return self._get("scene", build)

    @property
    def spacing(self):
        """Mean nearest-neighbour distance of the reference samples."""
        def build():
            points = self.ref_samples[0]
            pick = np.random.default_rng(0).choice(
                len(points), min(len(points), 10_000), replace=False)
            index = self.ref_sample_index.knn(points[pick], 2)[:, 1]
            return float(np.linalg.norm(points[pick] - points[index], axis=1).mean())
        return self._get("spacing", build)


# =========================
# Local plane fits (vectorised PCA)
# =========================
def _local_planes(relative, ks):
    """PCA of the first k neighbours, for every k in ``ks`` at once.

    ``relative``: (N, K, 3) neighbour positions relative to the query.
    Returns {k: (centroid (N, 3), normal (N, 3), variation (N,))} where
    variation = λ_min / Σλ (0 = perfectly planar).
    """
    s1 = np.cumsum(relative, axis=1)
    s2 = np.cumsum(np.einsum("nki,nkj->nkij", relative, relative), axis=1)
    planes = {}
    for k in ks:
        mean = s1[:, k - 1] / k
        cov = s2[:, k - 1] / k - np.einsum("ni,nj->nij", mean, mean)
        values, vectors = np.linalg.eigh(cov)
        total = np.maximum(values.sum(axis=1), 1e-30)
        planes[k] = (mean, vectors[:, :, 0], np.clip(values[:, 0], 0, None) / total)
    return planes


def _closest_on_mesh(ctx, query):
    """Signed distance to the reference triangles and the closest points
    (sign from the triangle normal: + outside, - inside)."""
    result = ctx.scene.compute_closest_points(
        o3d.core.Tensor(np.ascontiguousarray(query, dtype=np.float32)))
    closest = result["points"].numpy().astype(float)
    normals = result["primitive_normals"].numpy().astype(float)
    offset = query - closest
    distance = np.linalg.norm(offset, axis=1)
    sign = np.where(np.einsum("ij,ij->i", offset, normals) < 0, -1.0, 1.0)
    return sign * distance, closest


# =========================
# The seven methods: -> (values, query points, matched reference points)
# =========================
def point_to_point(ctx):
    query = ctx.compared.vertices
    nearest = ctx.ref_vertex_index.knn(query, 1)[:, 0]
    matched = ctx.reference.vertices[nearest]
    return np.linalg.norm(query - matched, axis=1), query, matched


def point_to_plane(ctx):
    query = ctx.compared.vertices
    nearest = ctx.ref_vertex_index.knn(query, 1)[:, 0]
    base = ctx.reference.vertices[nearest]
    normal = ctx.reference.normals[nearest]
    values = np.einsum("ij,ij->i", query - base, normal)
    return values, query, query - values[:, None] * normal


def point_to_mesh(ctx):
    query = ctx.compared.vertices
    values, closest = _closest_on_mesh(ctx, query)
    return values, query, closest


def cloud_to_cloud(ctx):
    query = ctx.cmp_samples[0]
    reference = ctx.ref_samples[0]
    nearest = ctx.ref_sample_index.knn(query, 1)[:, 0]
    matched = reference[nearest]
    return np.linalg.norm(query - matched, axis=1), query, matched


def adaptive_cloud_to_cloud(ctx):
    """Distance to a local least-squares plane; per point the largest
    neighbourhood (6 / 12 / 24 reference samples) that is still planar is
    used: large where the surface is flat (robust to sampling noise),
    small where it curves (follows the detail)."""
    query = ctx.cmp_samples[0]
    reference, reference_normals = ctx.ref_samples
    k_max = max(ADAPTIVE_K)
    values = np.empty(len(query))
    matched = np.empty_like(query)
    for start in range(0, len(query), CHUNK):
        q = query[start:start + CHUNK]
        neighbours = ctx.ref_sample_index.knn(q, k_max)
        planes = _local_planes(reference[neighbours] - q[:, None, :], ADAPTIVE_K)
        # choose the scale: largest planar one, else the smallest
        chosen_mean, chosen_normal, _ = planes[min(ADAPTIVE_K)]
        chosen_mean, chosen_normal = chosen_mean.copy(), chosen_normal.copy()
        for k in sorted(ADAPTIVE_K)[1:]:
            mean, normal, variation = planes[k]
            planar = variation < ADAPTIVE_PLANARITY
            chosen_mean[planar] = mean[planar]
            chosen_normal[planar] = normal[planar]
        # orient like the reference surface normals around the point
        around = reference_normals[neighbours[:, :min(ADAPTIVE_K)]].mean(axis=1)
        flip = np.einsum("ij,ij->i", chosen_normal, around) < 0
        chosen_normal[flip] *= -1
        # query sits at the origin of `relative`: distance = -mean · n
        d = -np.einsum("ij,ij->i", chosen_mean, chosen_normal)
        values[start:start + CHUNK] = d
        matched[start:start + CHUNK] = q - d[:, None] * chosen_normal
    return values, query, matched


def cloud_to_mesh(ctx):
    query = ctx.cmp_samples[0]
    values, closest = _closest_on_mesh(ctx, query)
    return values, query, closest


def m3c2(ctx):
    """M3C2 (Lague, Brodu & Leroux 2013).

    For each core point c (a subsample of the compared samples):
    1. normal n: PCA of the reference samples around c at several scales
       (12-96 neighbours); the most planar scale wins (multiscale);
    2. projection cylinder of radius r (``m3c2_diameter`` / 2 x spacing)
       and half-length h along n; it is covered by a stack of spheres
       (centres every r along the axis, each keeping only its own slab,
       so no point is counted twice);
    3. mean position of each cloud along n inside the cylinder:
       distance = mean_compared − mean_reference;
    4. level of detection LoD95 = 1.96 sqrt(σ1²/n1 + σ2²/n2); |d| > LoD95
       is a significant change.  Cores with < 3 points in a cloud get NaN.
    """
    params = ctx.params
    samples, _ = ctx.cmp_samples
    reference, reference_normals = ctx.ref_samples
    rng = np.random.default_rng(params["seed"])
    count = min(params["core_points"], len(samples))
    cores = samples[np.sort(rng.choice(len(samples), count, replace=False))]

    spacing = ctx.spacing
    r = max(params["m3c2_diameter"] / 2.0, 0.5) * spacing
    h = max(params["m3c2_depth_pct"] / 100.0 * ctx.reference.diagonal, r)
    steps = np.arange(-math.ceil(h / r), math.ceil(h / r) + 1) * r
    sphere = r * math.sqrt(1.25)             # covers a slab r long, radius r
    max_nn = 64

    values = np.full(count, np.nan)
    lod = np.full(count, np.nan)
    counts = np.zeros((count, 2), dtype=int)
    normals = np.empty((count, 3))
    k_max = max(M3C2_NORMAL_K)
    for start in range(0, count, CHUNK // 2):
        c = cores[start:start + CHUNK // 2]
        n_core = len(c)
        # 1. multiscale normal from the reference samples
        neighbours = ctx.ref_sample_index.knn(c, k_max)
        planes = _local_planes(reference[neighbours] - c[:, None, :], M3C2_NORMAL_K)
        best = np.full(n_core, np.inf)
        normal = np.zeros((n_core, 3))
        for k in M3C2_NORMAL_K:
            _, nk, variation = planes[k]
            better = variation < best
            best[better] = variation[better]
            normal[better] = nk[better]
        around = reference_normals[neighbours[:, :12]].mean(axis=1)
        normal[np.einsum("ij,ij->i", normal, around) < 0] *= -1
        normals[start:start + n_core] = normal

        # 2-3. cylinder statistics for both clouds
        stats = []
        for points, index in ((reference, ctx.ref_sample_index),
                              (samples, ctx.cmp_sample_index)):
            total = np.zeros(n_core)
            total2 = np.zeros(n_core)
            n_in = np.zeros(n_core, dtype=int)
            for step in steps:
                centres = c + step * normal
                found = index.radius(centres, sphere, max_nn)
                valid = found >= 0
                offset = points[np.where(valid, found, 0)] - c[:, None, :]
                t = np.einsum("nkj,nj->nk", offset, normal)
                lateral2 = np.einsum("nkj,nkj->nk", offset, offset) - t * t
                keep = (valid & (lateral2 <= r * r) & (np.abs(t) <= h)
                        & (t >= step - r / 2) & (t < step + r / 2))
                total += np.where(keep, t, 0).sum(axis=1)
                total2 += np.where(keep, t * t, 0).sum(axis=1)
                n_in += keep.sum(axis=1)
            mean = total / np.maximum(n_in, 1)
            var = np.maximum(total2 / np.maximum(n_in, 1) - mean ** 2, 0)
            var *= n_in / np.maximum(n_in - 1, 1)          # sample variance
            stats.append((mean, var, n_in))
        (m1, v1, n1), (m2, v2, n2) = stats
        ok = (n1 >= 3) & (n2 >= 3)
        block = slice(start, start + n_core)
        values[block] = np.where(ok, m2 - m1, np.nan)
        lod[block] = np.where(
            ok, 1.96 * np.sqrt(v1 / np.maximum(n1, 1) + v2 / np.maximum(n2, 1)),
            np.nan)
        counts[block, 0], counts[block, 1] = n1, n2
    # matched reference position: core shifted to the reference mean
    matched = cores - np.nan_to_num(values)[:, None] * normals
    extra = {
        "lod95": lod,
        "significant": np.abs(values) > lod,
        "projection_radius": r,
        "max_depth": h,
        "spacing": spacing,
        "points_per_cylinder": counts,
    }
    return values, cores, matched, extra


FUNCTIONS = {
    "p2p": point_to_point,
    "p2pl": point_to_plane,
    "p2m": point_to_mesh,
    "c2c": cloud_to_cloud,
    "ac2c": adaptive_cloud_to_cloud,
    "c2m": cloud_to_mesh,
    "m3c2": m3c2,
}


# =========================
# Statistics
# =========================
STAT_NAMES = {
    "n": "N", "mean": "Mean e", "sd": "SD", "mae": "MAE", "rmse": "RMSE",
    "r2": "R²", "median": "Median |e|", "p95": "P95 |e|", "max": "Max |e|",
}


def error_stats(values, matched):
    """mean e, SD, MAE, RMSE, R² (+ median / 95th percentile / max |e|)
    over the finite values."""
    finite = np.isfinite(values)
    e = values[finite]
    if len(e) == 0:
        return {key: float("nan") for key in STAT_NAMES} | {"n": 0, "valid": 0.0}
    q = matched[finite]
    abs_e = np.abs(e)
    total = float(((q - q.mean(axis=0)) ** 2).sum())
    return {
        "n": int(len(e)),
        "valid": float(finite.mean()),
        "mean": float(e.mean()),
        "sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
        "mae": float(abs_e.mean()),
        "rmse": float(np.sqrt((e * e).mean())),
        "r2": float(1.0 - (e * e).sum() / total) if total > 0 else float("nan"),
        "median": float(np.median(abs_e)),
        "p95": float(np.percentile(abs_e, 95)),
        "max": float(abs_e.max()),
    }


def to_vertices(values, points, vertices, cache=None):
    """Values given at ``points`` carried to ``vertices`` (nearest point);
    ``cache`` reuses the search of a point set several methods share."""
    if points is vertices or (len(points) == len(vertices)
                              and np.shares_memory(points, vertices)):
        return values
    key = id(points)
    if cache is None or key not in cache:
        nearest = _Index(points).knn(vertices, 1)[:, 0]
        if cache is None:
            return values[nearest]
        cache[key] = nearest
    return values[cache[key]]


# =========================
# Run
# =========================
def _prepare(compared, reference, params):
    """Shared state of one comparison: both models moved near the origin
    (the float32 KD-tree / BVH lose precision with large coordinates;
    results are moved back) and the lazily built samples / indices."""
    origin = reference.vertices.mean(axis=0)
    compared_local = compared.translated(-origin)
    reference_local = reference.translated(-origin)
    return {"origin": origin, "compared": compared_local,
            "reference": reference_local,
            "ctx": _Context(compared_local, reference_local, params),
            "vertex_cache": {}}


def _run_method(state, key):
    """One method -> its result entry (positions in the input coordinates)."""
    compared, reference = state["compared"], state["reference"]
    if key in MESH_METHODS and not reference.has_triangles:
        return {"skipped": "the reference has no triangles"}
    started = time.perf_counter()
    output = FUNCTIONS[key](state["ctx"])
    values, points, matched = output[:3]
    extra = output[3] if len(output) > 3 else {}
    return {
        "values": values,
        "points": points + state["origin"],
        "matched": matched + state["origin"],
        "vertex_values": to_vertices(values, points, compared.vertices,
                                     state["vertex_cache"]),
        "stats": error_stats(values, matched),
        "signed": METHODS[key][1],
        "extra": extra,
        "time": time.perf_counter() - started,
    }


def _finish(results, compared, reference, params):
    diagonal = max(compared.diagonal, reference.diagonal)
    warnings = []
    reference_result = results.get("p2m") or results.get("c2c") or results.get("p2p")
    if reference_result and "stats" in reference_result:
        median = reference_result["stats"]["median"]
        if median > 0.02 * diagonal:
            warnings.append(
                f"The models may not be aligned: median distance "
                f"{median:.4g} is {median / diagonal:.0%} of the diagonal.")
    return {"results": results, "params": params, "diagonal": diagonal,
            "warnings": warnings}


def run_comparison(compared, reference, methods=ORDER, params=None,
                   progress=None, cancel=None):
    """Run ``methods`` and return::

        {"results": {key: {...}}, "params": ..., "diagonal": ...,
         "warnings": [...]}

    Each result has ``values`` (per query point), ``points``, ``matched``,
    ``vertex_values`` (on the compared vertices, for colouring),
    ``stats``, ``time`` and, for M3C2, ``extra``; a method that cannot run
    has only ``skipped`` (the reason).
    """
    params = dict(DEFAULT_PARAMS, **(params or {}))
    state = _prepare(compared, reference, params)
    results = {}
    for number, key in enumerate(methods, 1):
        if cancel is not None and cancel.is_set():
            break
        results[key] = _run_method(state, key)
        if progress is not None and "skipped" not in results[key]:
            progress((number, len(methods), key))
    return _finish(results, compared, reference, params)


# =========================
# In the helper process (see ``isolated``): Open3D holds the GIL during
# its searches, which froze the window while comparing
# =========================
def _model_arrays(model):
    return (model.vertices, model.triangles, model.normals, model.name)


def _key(arrays, params):
    import hashlib
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        if array is not None and not isinstance(array, str):
            digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr(sorted(params.items())).encode())
    return digest.hexdigest()


_job_state = {"key": None}


def compare_job(compared_arrays, reference_arrays, key, params):
    """One method in the helper; the models, samples and indices are kept
    for the next method of the same run."""
    cache_key = _key(compared_arrays + reference_arrays, params)
    if _job_state["key"] != cache_key:
        _job_state.clear()
        _job_state["key"] = None
        compared = Model(*compared_arrays[:3], name=compared_arrays[3])
        reference = Model(*reference_arrays[:3], name=reference_arrays[3])
        _job_state.update(state=_prepare(compared, reference, params),
                          key=cache_key)
    return _run_method(_job_state["state"], key)


def run_comparison_isolated(compared, reference, methods=ORDER, params=None,
                            progress=None, cancel=None):
    """``run_comparison`` with the work in the helper process (one call
    per method, so progress and cancelling still work)."""
    from modules import isolated
    params = dict(DEFAULT_PARAMS, **(params or {}))
    compared_arrays = _model_arrays(compared)
    reference_arrays = _model_arrays(reference)
    results = {}
    for number, key in enumerate(methods, 1):
        if cancel is not None and cancel.is_set():
            break
        if key in MESH_METHODS and not reference.has_triangles:
            results[key] = {"skipped": "the reference has no triangles"}
            continue
        results[key] = isolated.run("compare", compared_arrays,
                                    reference_arrays, key, params)
        if progress is not None:
            progress((number, len(methods), key))
    isolated.run_if_started("compare_clear")
    return _finish(results, compared, reference, params)


def compare_clear_job():
    _job_state.clear()
    _job_state["key"] = None
