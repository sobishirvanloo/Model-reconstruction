"""Numpy / Open3D / VTK helpers shared by the three tabs."""

import os

import numpy as np
import open3d as o3d
import vtk
from vtk.util import numpy_support


# =========================
# Loading
# =========================
def load_array(path):
    """Load a numeric 2-D array from .npy, .csv or whitespace/comma .txt."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        delimiter = "," if ext == ".csv" else None
        try:
            arr = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        except ValueError:
            # .txt written with commas / .csv with a header row
            arr = np.genfromtxt(
                path, delimiter=",", skip_header=0, invalid_raise=True
            )
            arr = arr[~np.isnan(arr).all(axis=1)]
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(
            f"{os.path.basename(path)}: needs at least 3 columns, "
            f"got {arr.shape[1]}"
        )
    return arr


# =========================
# Normals
# =========================
def normalize(normals):
    normals = np.asarray(normals, dtype=float)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def bbox_diagonal(points):
    points = np.asarray(points)
    if len(points) == 0:
        return 1.0
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return diag if diag > 0 else 1.0


def normal_search_radius(points, voxel_size=0.0):
    """Neighbourhood radius that scales with the model size."""
    return max(voxel_size * 4.0, bbox_diagonal(points) * 0.02)


def estimate_oriented_normals(pcd, voxel_size=0.0, reference=None):
    """Estimate normals on ``pcd`` and orient them consistently.

    When ``reference`` normals (same length as the cloud) are given, the
    global sign is chosen so the result agrees with them; otherwise the
    normals are pointed away from the cloud centroid.
    """
    points = np.asarray(pcd.points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_search_radius(points, voxel_size), max_nn=30
        )
    )
    orient_consistently(pcd, reference)
    return pcd


def orient_consistently(pcd, reference=None):
    """``orient_normals_consistent_tangent_plane`` picks an arbitrary global
    sign, which can turn the whole surface inside out.  Fix the sign
    against ``reference`` normals or the outward (from centroid) direction.
    """
    if len(pcd.points) >= 4:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=30)
        except RuntimeError:
            pass

    normals = np.asarray(pcd.normals)
    if reference is not None and len(reference) == len(normals):
        agreement = np.einsum("ij,ij->i", normals, reference).sum()
    else:
        points = np.asarray(pcd.points)
        outward = points - points.mean(axis=0)
        agreement = np.einsum("ij,ij->i", normals, outward).sum()
    if agreement < 0:
        pcd.normals = o3d.utility.Vector3dVector(-normals)
    return pcd


# =========================
# Transforms
# =========================
def vtk_matrix_to_numpy(matrix):
    return np.array(
        [[matrix.GetElement(r, c) for c in range(4)] for r in range(4)]
    )


def apply_transform(matrix, points, normals=None):
    """Apply a 4x4 transform to points (and the matching normal transform,
    the inverse-transpose of the linear part, to normals)."""
    points = np.asarray(points, dtype=float)
    out_points = points @ matrix[:3, :3].T + matrix[:3, 3]
    if normals is None:
        return out_points, None
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    out_normals = normalize(np.asarray(normals, dtype=float) @ normal_matrix.T)
    return out_points, out_normals


# =========================
# VTK conversion (vectorised - no Python loops)
# =========================
def polydata_from_numpy(points, normals=None, triangles=None, colors=None):
    """Build vtkPolyData from numpy arrays.

    With ``triangles`` the result is a surface, otherwise every point is
    emitted as a vertex cell (point cloud).  ``colors`` (RGB in 0..1) are
    stored as the active ``Colors`` scalars.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    n = len(points)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    if triangles is not None:
        triangles = np.asarray(triangles, dtype=np.int64)
        cells = np.hstack(
            [np.full((len(triangles), 1), 3, dtype=np.int64), triangles]
        ).ravel()
        cell_array = vtk.vtkCellArray()
        cell_array.SetCells(
            len(triangles),
            numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True),
        )
        polydata.SetPolys(cell_array)
    else:
        cells = np.column_stack(
            [np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]
        ).ravel()
        cell_array = vtk.vtkCellArray()
        cell_array.SetCells(
            n, numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True)
        )
        polydata.SetVerts(cell_array)

    if normals is not None and len(normals) == n:
        vtk_normals = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(normals, dtype=np.float32), deep=True
        )
        vtk_normals.SetName("Normals")
        polydata.GetPointData().SetNormals(vtk_normals)

    if colors is not None and len(colors) == n:
        rgb = np.ascontiguousarray(
            np.clip(np.asarray(colors)[:, :3] * 255.0, 0, 255), dtype=np.uint8)
        vtk_colors = numpy_support.numpy_to_vtk(rgb, deep=True)
        vtk_colors.SetName("Colors")
        polydata.GetPointData().SetScalars(vtk_colors)

    return polydata


def transfer_colors(source_points, source_colors, target_points):
    """Colour each target point like its nearest source point - keeps the
    original colours through downsampling, noise, shuffling and
    reconstruction.  ``None`` when there are no colours."""
    if source_colors is None or target_points is None or len(target_points) == 0:
        return None
    search = o3d.core.nns.NearestNeighborSearch(
        o3d.core.Tensor(np.ascontiguousarray(source_points, dtype=np.float32)))
    search.knn_index()
    index, _ = search.knn_search(
        o3d.core.Tensor(np.ascontiguousarray(target_points, dtype=np.float32)), 1)
    return np.asarray(source_colors)[index.numpy().ravel()]
