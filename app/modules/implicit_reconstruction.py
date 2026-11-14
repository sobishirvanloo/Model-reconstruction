"""Implicit-surface reconstructions (oriented points -> mesh).

* **RIMLS** - Robust Implicit Moving Least Squares (Öztireli, Guennebaud
  & Gross 2009, as in MeshLab): around every point x the surface is the
  zero set of a locally fitted function

      f(x) = Σ wᵢ nᵢ·(x − pᵢ) / Σ wᵢ ,   wᵢ = αᵢ φ(‖x − pᵢ‖ / h)

  refitted a few times with robust weights αᵢ that ignore neighbours
  whose plane or normal disagrees (outliers, the far side of thin parts),
  so sharp features survive and noise is averaged out.
* **Signed distance** - volumetric signed distance to the oriented
  points, accumulated over a radius (VTK ``vtkSignedDistance``, the
  approach of Curless & Levoy 1996): fast and robust to noise.

Both evaluate the function on a regular grid but only in a thin band
around the points (cells farther away stay "empty"), then extract the
zero level with ``vtkExtractSurface``, which never builds a surface
between a value and an empty cell - so no fake closing surfaces appear.
Small pieces (< ``MIN_PIECE`` of the largest) are removed afterwards.
"""

import math

import numpy as np
import open3d as o3d
import vtk
from vtk.util import numpy_support

from modules.compute import KNN, available_memory

CELL_FACTOR = 1.5          # grid cell, x mean point spacing
MAX_CELLS = 300            # cells along the longest side (memory cap)
BAND_CELLS = 2.5           # band around the points where f is evaluated
MIN_PIECE = 0.01           # pieces smaller than 1 % of the largest go
RIMLS_K = 16               # neighbours per evaluation
RIMLS_ITERATIONS = 3
RIMLS_SIGMA_R = 0.5        # robustness to residuals (x h)
RIMLS_SIGMA_N = 0.8        # robustness to normal differences
CHUNK = 60_000
EMPTY = 1e30


GRID_MEMORY_SHARE = 0.25   # a grid may use at most 1/4 of the free RAM
BYTES_PER_CELL = 9         # float32 value + band flag + VTK copy (approx.)


def _grid(points, spacing, pad):
    """Origin, cell size and dimensions of the evaluation grid; the cell
    grows when the grid would not fit comfortably in the free memory."""
    low = points.min(axis=0) - pad
    high = points.max(axis=0) + pad
    cell = max(CELL_FACTOR * spacing, float((high - low).max()) / MAX_CELLS)
    free = available_memory()
    if free:
        budget = GRID_MEMORY_SHARE * free / BYTES_PER_CELL
        cells = np.prod(np.ceil((high - low) / cell) + 1)
        if cells > budget:
            cell *= (cells / budget) ** (1 / 3) * 1.01
    dims = np.ceil((high - low) / cell).astype(np.int64) + 1
    return low, cell, dims


def _band(points, low, cell, dims, radius_cells):
    """Flat indices (x slowest) of the grid nodes within ``radius_cells``
    of a point."""
    nodes = np.unique(np.floor((points - low) / cell).astype(np.int64), axis=0)
    r = int(math.ceil(radius_cells))
    offsets = np.array(np.meshgrid(*[np.arange(-r, r + 2)] * 3, indexing="ij")
                       ).reshape(3, -1).T
    centre = np.full(3, 0.5)
    offsets = offsets[np.linalg.norm(offsets - centre, axis=1) <= radius_cells + 1]
    # mark a boolean grid instead of sorting millions of duplicates
    marked = np.zeros(int(np.prod(dims)), dtype=bool)
    for offset in offsets:
        index = nodes + offset
        index = index[((index >= 0) & (index < dims)).all(axis=1)]
        marked[(index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]] = True
    return np.flatnonzero(marked)


def _extract(values, low, cell, dims):
    """Zero level of a grid (``EMPTY`` cells ignored) -> Open3D mesh."""
    image = vtk.vtkImageData()
    image.SetDimensions(*(int(d) for d in dims))
    image.SetOrigin(*low)
    image.SetSpacing(cell, cell, cell)
    # numpy flat order is x-slowest; VTK wants x fastest
    volume = values.reshape(dims).transpose(2, 1, 0)
    array = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(volume.reshape(-1), dtype=np.float32), deep=True)
    array.SetName("f")
    image.GetPointData().SetScalars(array)
    return _surface(image, EMPTY * 0.5)


def _surface(image_or_port, radius):
    extract = vtk.vtkExtractSurface()
    if isinstance(image_or_port, vtk.vtkImageData):
        extract.SetInputData(image_or_port)
    else:
        extract.SetInputConnection(image_or_port)
    extract.SetRadius(radius)
    extract.HoleFillingOn()
    extract.ComputeNormalsOff()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(extract.GetOutputPort())
    triangles.Update()
    poly = triangles.GetOutput()
    if poly.GetNumberOfPolys() == 0:
        return o3d.geometry.TriangleMesh()
    vertices = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    connectivity = numpy_support.vtk_to_numpy(
        poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(connectivity.astype(np.int32)))


def _sign_inverted(values, points, normals, low, cell, dims, radius):
    """True when the grid is negative on the side the normals point to
    (it must be positive outside)."""
    pick = np.random.default_rng(0).choice(len(points), min(len(points), 2000),
                                           replace=False)
    probe = points[pick] + normals[pick] * (0.5 * radius)
    index = np.clip(np.rint((probe - low) / cell).astype(np.int64), 0, dims - 1)
    flat = (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]
    sample = values[flat]
    sample = sample[np.abs(sample) < EMPTY * 0.5]
    return len(sample) > 0 and np.median(sample) < 0


def remove_small_pieces(mesh, share=MIN_PIECE):
    """Drop connected pieces with fewer than ``share`` of the triangles of
    the largest one (debris of the band / noise)."""
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < share * counts.max())
    mesh.remove_unreferenced_vertices()
    return mesh


# =========================
# RIMLS
# =========================
def rimls_values(points, normals, queries, h):
    """Robust IMLS function value at ``queries`` (NaN where no point lies
    within h)."""
    search = KNN(points)                       # GPU when available
    out = np.full(len(queries), np.nan)
    k = min(RIMLS_K, len(points))
    # float32 halves the memory traffic; values are relative to x anyway
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    h = np.float32(h)
    h2 = h * h
    for start in range(0, len(queries), CHUNK):
        x = np.asarray(queries[start:start + CHUNK], dtype=np.float32)
        index = search.knn(x, k)
        diff = x[:, None, :] - points[index]                 # x − pᵢ
        n = normals[index]
        dist2 = np.einsum("mkj,mkj->mk", diff, diff)
        plane = np.einsum("mkj,mkj->mk", diff, n)            # nᵢ·(x − pᵢ)
        t = np.clip(1.0 - dist2 / h2, 0.0, None)
        phi = t ** 4                                         # Wendland-like
        dphi = (np.float32(-8.0) / h2) * t ** 3              # dφ/dx / (x − pᵢ)
        f = np.zeros(len(x), dtype=np.float32)
        grad = np.zeros((len(x), 3), dtype=np.float32)
        for iteration in range(RIMLS_ITERATIONS):
            if iteration == 0:
                alpha = 1.0
            else:
                residual = (plane - f[:, None]) / (RIMLS_SIGMA_R * h)
                normal_gap = np.einsum("mkj,mkj->mk", n - grad[:, None, :],
                                       n - grad[:, None, :])
                alpha = np.exp(-residual ** 2 - normal_gap / RIMLS_SIGMA_N ** 2)
            w = alpha * phi
            gw = (alpha * dphi)[..., None] * diff
            sum_w = w.sum(axis=1)
            ok = sum_w > 1e-12
            safe = np.where(ok, sum_w, 1.0)
            f = np.where(ok, (w * plane).sum(axis=1) / safe, 0.0)
            grad = np.where(
                ok[:, None],
                ((gw * plane[..., None]).sum(axis=1) - f[:, None] * gw.sum(axis=1)
                 + (w[..., None] * n).sum(axis=1)) / safe[:, None],
                0.0)
        out[start:start + CHUNK] = np.where(ok, f, np.nan)
    return out


def reconstruct_rimls(points, normals, spacing, factor):
    """RIMLS surface; ``factor`` = kernel radius h in mean point spacings
    (larger = smoother, slower)."""
    # centred: float32 math stays precise for any coordinates
    origin = np.asarray(points, dtype=float).mean(axis=0)
    points = np.asarray(points, dtype=float) - origin
    h = factor * spacing
    low, cell, dims = _grid(points, spacing, pad=2 * h)
    band = _band(points, low, cell, dims, BAND_CELLS)
    ix, rest = np.divmod(band, dims[1] * dims[2])
    iy, iz = np.divmod(rest, dims[2])
    queries = low + np.column_stack([ix, iy, iz]) * cell
    values = rimls_values(points, normals, queries, h)
    grid = np.full(int(np.prod(dims)), EMPTY, dtype=np.float32)
    finite = np.isfinite(values)
    grid[band[finite]] = values[finite]
    mesh = remove_small_pieces(_extract(grid, low, cell, dims))
    mesh.translate(origin)
    return mesh, {"h": h, "cell": cell, "grid": tuple(int(d) for d in dims),
                  "setting": f"h = {h:.4g} (×{factor:g})"}


# =========================
# Signed distance (VTK)
# =========================
def reconstruct_signed_distance(points, normals, spacing, factor):
    """Volumetric signed distance; ``factor`` = influence radius in mean
    point spacings (larger = smoother, fills small gaps)."""
    origin = np.asarray(points, dtype=float).mean(axis=0)
    points = np.asarray(points, dtype=float) - origin
    radius = factor * spacing
    low, cell, dims = _grid(points, spacing, pad=2 * radius)
    high = low + (dims - 1) * cell
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(points, dtype=float), deep=True))
    poly.SetPoints(vtk_points)
    vtk_normals = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(normals, dtype=np.float32), deep=True)
    vtk_normals.SetName("Normals")
    poly.GetPointData().SetNormals(vtk_normals)
    distance = vtk.vtkSignedDistance()
    distance.SetInputData(poly)
    distance.SetRadius(radius)
    distance.SetDimensions(*(int(d) for d in dims))
    distance.SetBounds(low[0], high[0], low[1], high[1], low[2], high[2])
    distance.Update()
    # VTK marks unseen cells with -radius and, in current versions, makes
    # the distance positive *inside* - then the inner edge of the band is
    # a real zero crossing and a phantom surface appears.  Use VTK's values
    # with our own empty marker and the sign checked against the normals.
    values = numpy_support.vtk_to_numpy(
        distance.GetOutput().GetPointData().GetScalars()).astype(np.float32)
    nx, ny, nz = (int(d) for d in dims)
    values = values.reshape(nz, ny, nx).transpose(2, 1, 0).reshape(-1)  # x slowest
    empty = np.isclose(np.abs(values), radius, rtol=1e-5)
    values[empty] = EMPTY
    if _sign_inverted(values, points, normals, low, cell, dims, radius):
        seen = ~empty
        values[seen] = -values[seen]
    mesh = remove_small_pieces(_extract(values, low, cell, dims))
    mesh.translate(origin)
    return mesh, {"radius": radius, "cell": cell,
                  "grid": tuple(int(d) for d in dims),
                  "setting": f"r = {radius:.4g} (×{factor:g})"}
