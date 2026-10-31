"""Quantitative quality metrics for a reconstructed mesh.

The reconstruction is compared with a *reference*:

* a reference **mesh** (ground truth, e.g. the original STL), or
* the **input point cloud** when no ground truth is available
  (measures how well the surface fits the data).

Distances are exact point-to-triangle distances (Open3D RaycastingScene)
wherever a surface is involved, nearest-neighbour distances otherwise.

    accuracy      reconstruction samples -> reference   (is it *correct*?)
    completeness  reference samples      -> reconstruction (is it *complete*?)

    Chamfer-L1    (mean accuracy + mean completeness) / 2
    Hausdorff     max(max accuracy, max completeness)
    Hausdorff-95  max(p95 accuracy, p95 completeness)  (robust to outliers)
    F-score@tau   2PR / (P + R), P = share of accuracy < tau,
                  R = share of completeness < tau (tau = 1% of bbox diagonal)
    Normal consistency  mean |n_recon . n_reference| at closest points

With only input points as reference, accuracy is pessimistic where the
points are sparse (a sample in the middle of a large triangle is far from
every point); a reference mesh does not have this bias.

All metrics are pure functions (no Qt) so they can run in a worker thread.
"""

import time

import numpy as np
import open3d as o3d

N_SAMPLES = 100_000          # surface samples per mesh
MAX_REFERENCE_POINTS = 200_000
TAU_RATIO = 0.01             # F-score threshold, fraction of bbox diagonal
# Alignment check: the INPUT points should lie on a reference mesh in the
# same coordinates, whatever the reconstruction quality.  Measured on the
# bunny: aligned / voxelized <= 0.002%, Gaussian noise sigma=1% diag
# 0.67%, rotated 15° 1.4%, 45° 3.6% (median distance, % of diagonal).
MISALIGNED_RATIO = 0.01
MAX_ALIGNMENT_POINTS = 50_000


# =========================
# Distance helpers
# =========================
class _SurfaceQuery:
    """Exact closest-point queries against a triangle mesh."""

    def __init__(self, mesh):
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def closest(self, points):
        query = o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
        result = self._scene.compute_closest_points(query)
        closest = result["points"].numpy().astype(float)
        normals = result["primitive_normals"].numpy().astype(float)
        distances = np.linalg.norm(closest - points, axis=1)
        return distances, normals


class _PointQuery:
    """Nearest-neighbour queries against a point cloud (with normals)."""

    def __init__(self, points, normals):
        self._normals = normals
        self._nns = o3d.core.nns.NearestNeighborSearch(
            o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
        )
        self._nns.knn_index()

    def closest(self, points):
        query = o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
        index, dist2 = self._nns.knn_search(query, 1)
        index = index.numpy().ravel()
        distances = np.sqrt(dist2.numpy().ravel().astype(float))
        normals = self._normals[index] if self._normals is not None else None
        return distances, normals


def _sample(mesh, n=N_SAMPLES):
    samples = mesh.sample_points_uniformly(number_of_points=n)
    return np.asarray(samples.points), np.asarray(samples.normals)


def _bbox_diagonal(points):
    extent = points.max(axis=0) - points.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    return diagonal if diagonal > 0 else 1.0


# =========================
# Mesh properties
# =========================
def mesh_properties(mesh):
    """Topology / size of the reconstructed mesh.

    ``is_watertight`` is avoided on purpose: its self-intersection test is
    far too slow for meshes with 100k+ triangles.  "Closed" here means
    edge-manifold without boundary edges.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    open_edges = len(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    closed = open_edges == 0
    _, cluster_sizes, _ = mesh.cluster_connected_triangles()

    volume = None
    if closed and len(triangles):
        v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
        volume = abs(float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)

    return {
        "vertices": len(vertices),
        "triangles": len(triangles),
        "surface_area": float(mesh.get_surface_area()),
        "volume": volume,
        "closed": closed,
        "open_edges": open_edges,
        "edge_manifold": mesh.is_edge_manifold(allow_boundary_edges=True),
        "vertex_manifold": mesh.is_vertex_manifold(),
        "components": len(cluster_sizes),
    }


# =========================
# Evaluation
# =========================
def _stats(distances):
    return {
        "mean": float(distances.mean()),
        "rms": float(np.sqrt((distances ** 2).mean())),
        "p95": float(np.percentile(distances, 95)),
        "max": float(distances.max()),
    }


def evaluate(mesh, reference_mesh=None, reference_points=None,
             reference_normals=None, input_points=None):
    """Compare ``mesh`` with a reference mesh, or with the input points.

    ``input_points`` (with a reference mesh) enables the alignment check.
    Returns ``(metrics, vertex_errors)`` where ``vertex_errors`` is the
    distance of every mesh vertex to the reference (for the heatmap).
    """
    start = time.perf_counter()
    if len(mesh.triangles) == 0:
        raise ValueError("cannot evaluate an empty mesh")
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    recon_query = _SurfaceQuery(mesh)
    recon_samples, recon_normals = _sample(mesh)

    if reference_mesh is not None:
        reference_kind = "mesh"
        ref_query = _SurfaceQuery(reference_mesh)
        ref_samples, ref_sample_normals = _sample(reference_mesh)
    else:
        reference_kind = "points"
        points = np.asarray(reference_points, dtype=float)
        normals = (
            np.asarray(reference_normals, dtype=float)
            if reference_normals is not None else None
        )
        if len(points) > MAX_REFERENCE_POINTS:
            keep = np.random.default_rng(0).choice(
                len(points), MAX_REFERENCE_POINTS, replace=False
            )
            points = points[keep]
            normals = normals[keep] if normals is not None else None
        ref_query = _PointQuery(points, normals)
        ref_samples, ref_sample_normals = points, normals

    accuracy, ref_normals_at_recon = ref_query.closest(recon_samples)
    completeness, _ = recon_query.closest(ref_samples)
    vertex_errors, _ = ref_query.closest(np.asarray(mesh.vertices))

    diagonal = _bbox_diagonal(ref_samples)
    tau = TAU_RATIO * diagonal
    precision = float((accuracy < tau).mean())
    recall = float((completeness < tau).mean())
    f_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    normal_consistency = None
    if ref_normals_at_recon is not None and len(recon_normals):
        dots = np.abs(np.einsum("ij,ij->i", recon_normals, ref_normals_at_recon))
        normal_consistency = float(dots.mean())

    # Alignment: median distance of the input points to the reference mesh
    input_offset = None
    if reference_kind == "mesh" and input_points is not None and len(input_points):
        sample = np.asarray(input_points, dtype=float)
        if len(sample) > MAX_ALIGNMENT_POINTS:
            sample = sample[np.random.default_rng(0).choice(
                len(sample), MAX_ALIGNMENT_POINTS, replace=False)]
        input_offset = float(np.median(ref_query.closest(sample)[0]))

    acc, comp = _stats(accuracy), _stats(completeness)
    chamfer = (acc["mean"] + comp["mean"]) / 2.0
    metrics = {
        "reference": reference_kind,
        "bbox_diagonal": diagonal,
        "chamfer_l1": chamfer,
        "chamfer_pct": 100.0 * chamfer / diagonal,
        "hausdorff": max(acc["max"], comp["max"]),
        "hausdorff_95": max(acc["p95"], comp["p95"]),
        "hausdorff_95_pct": 100.0 * max(acc["p95"], comp["p95"]) / diagonal,
        "accuracy": acc,
        "completeness": comp,
        "tau": tau,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "normal_consistency": normal_consistency,
        "input_to_reference_pct": (
            100.0 * input_offset / diagonal if input_offset is not None else None
        ),
        "likely_misaligned": (
            input_offset is not None and input_offset > MISALIGNED_RATIO * diagonal
        ),
        "eval_time": time.perf_counter() - start,
    }
    metrics.update(mesh_properties(mesh))
    return metrics, vertex_errors


# =========================
# Presentation
# =========================
def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def metric_rows(metrics, timings=None):
    """(name, value) rows for display / Markdown export."""
    ref = "reference mesh" if metrics["reference"] == "mesh" else "input points"
    rows = [
        ("Compared against", ref),
        ("Chamfer-L1", f"{_fmt(metrics['chamfer_l1'])}  "
                       f"({metrics['chamfer_pct']:.3f}% of diag.)"),
        ("Hausdorff (max)", _fmt(metrics["hausdorff"])),
        ("Hausdorff-95", f"{_fmt(metrics['hausdorff_95'])}  "
                         f"({metrics['hausdorff_95_pct']:.3f}% of diag.)"),
        ("Accuracy mean / RMS", f"{_fmt(metrics['accuracy']['mean'])} / "
                                f"{_fmt(metrics['accuracy']['rms'])}"),
        ("Completeness mean / RMS", f"{_fmt(metrics['completeness']['mean'])} / "
                                    f"{_fmt(metrics['completeness']['rms'])}"),
        (f"F-score @ τ={_fmt(metrics['tau'], 3)}",
         f"{metrics['f_score']:.4f}  (P {metrics['precision']:.3f}, "
         f"R {metrics['recall']:.3f})"),
        ("Normal consistency", _fmt(metrics["normal_consistency"])),
        ("Vertices / Triangles", f"{_fmt(metrics['vertices'])} / "
                                 f"{_fmt(metrics['triangles'])}"),
        ("Closed (no open edges)", f"{_fmt(metrics['closed'])}  "
                                   f"({_fmt(metrics['open_edges'])} open edges)"),
        ("Edge / Vertex manifold", f"{_fmt(metrics['edge_manifold'])} / "
                                   f"{_fmt(metrics['vertex_manifold'])}"),
        ("Connected components", _fmt(metrics["components"])),
        ("Surface area", _fmt(metrics["surface_area"])),
        ("Volume", _fmt(metrics["volume"])),
        ("Reference bbox diagonal", _fmt(metrics["bbox_diagonal"])),
    ]
    if metrics.get("input_to_reference_pct") is not None:
        rows.append(("Input points → reference (median)",
                     f"{metrics['input_to_reference_pct']:.3f}% of diag."
                     + ("  ⚠ misaligned?" if metrics["likely_misaligned"] else "")))
    if timings:
        rows.append(("Time: preprocess / reconstruct / post / eval",
                     " / ".join(f"{timings.get(k, 0):.2f}s" for k in
                                ("preprocess", "reconstruct", "postprocess", "evaluate"))))
    return rows


def rows_to_markdown(rows):
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)
