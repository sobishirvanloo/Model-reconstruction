# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Slicing a mesh or a point cloud with up to six planes (no Qt).

A plane is given by six numbers in the model's own coordinates: a point
(position X, Y, Z) and three angles (X, Y, Z, in degrees, applied like
the model rotation: R = Rz · Rx · Ry) that turn the base normal +Z.  The
part kept is where every enabled plane's signed distance is >= 0
(``flip`` keeps the other side): the intersection of half-spaces - one
plane cuts the model in two, six make a box.

* points: kept by a mask;
* meshes: clipped exactly (triangles crossing a plane are cut along it,
  ``vtkClipPolyData``); with ``cap`` the opening is closed by the
  triangulated cross-section of each plane (``vtkCutter`` →
  ``vtkStripper`` → ``vtkContourTriangulator``), facing outward, so a
  closed mesh stays closed and keeps a meaningful volume.
* ``section``: the cut line of one plane - its length, and the area it
  encloses (meaningful for closed meshes).
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import vtk
from vtk.util import numpy_support

MAX_PLANES = 6


@dataclass(frozen=True)
class Plane:
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)     # degrees, X / Y / Z
    enabled: bool = True
    flip: bool = False

    @property
    def normal(self):
        """Unit normal of the kept side."""
        n = rotation_matrix(self.rotation) @ np.array([0.0, 0.0, 1.0])
        return -n if self.flip else n

    @property
    def origin(self):
        return np.asarray(self.position, dtype=float)

    def distance(self, points):
        """Signed distance of ``points`` (>= 0 = kept)."""
        return (np.asarray(points, dtype=float) - self.origin) @ self.normal

    def vtk_plane(self, matrix=None):
        """``vtkPlane`` (in world coordinates when ``matrix`` - the
        model's 4x4 pose - is given)."""
        origin, normal = self.origin, self.normal
        if matrix is not None:
            origin = matrix[:3, :3] @ origin + matrix[:3, 3]
            normal = np.linalg.inv(matrix[:3, :3]).T @ normal
            normal /= np.linalg.norm(normal)
        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        return plane


def rotation_matrix(rotation):
    """3x3 rotation of VTK angles (degrees): Rz · Rx · Ry."""
    ax, ay, az = (math.radians(a) for a in rotation)
    cx, sx, cy, sy, cz, sz = (math.cos(ax), math.sin(ax), math.cos(ay),
                              math.sin(ay), math.cos(az), math.sin(az))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ rx @ ry


def angles_for_normal(normal):
    """Angles (Z = 0) whose plane has ``normal``: n = (sin β, −sin α cos β,
    cos α cos β)."""
    n = np.asarray(normal, dtype=float)
    n = n / max(np.linalg.norm(n), 1e-12)
    beta = math.degrees(math.asin(max(-1.0, min(1.0, n[0]))))
    alpha = math.degrees(math.atan2(-n[1], n[2]))
    return (alpha, beta, 0.0)


# ---------- presets (model box low / high) ----------
AXIS_NORMALS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def axis_plane(axis, low, high):
    """Plane through the box centre, normal along ``axis`` (keeps +)."""
    centre = (np.asarray(low) + np.asarray(high)) / 2
    return Plane(tuple(centre), angles_for_normal(AXIS_NORMALS[axis]))


def box_planes(low, high, share=0.6):
    """Six planes around the centre ``share`` of the box (kept inside)."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    centre, half = (low + high) / 2, (high - low) * share / 2
    planes = []
    for i, axis in enumerate("XYZ"):
        normal = np.array(AXIS_NORMALS[axis])
        lower = centre.copy()
        lower[i] -= half[i]
        upper = centre.copy()
        upper[i] += half[i]
        planes.append(Plane(tuple(lower), angles_for_normal(normal)))
        planes.append(Plane(tuple(upper), angles_for_normal(-normal)))
    return planes


# =========================
# Slicing
# =========================
def active(planes):
    return [p for p in planes if p.enabled]


def point_mask(points, planes):
    """True for the points kept by every enabled plane."""
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for plane in active(planes):
        keep &= plane.distance(points) >= 0
    return keep


def _polydata(vertices, triangles):
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    poly.SetPoints(points)
    triangles = np.asarray(triangles, dtype=np.int64)
    cells = np.hstack([np.full((len(triangles), 1), 3, np.int64), triangles]).ravel()
    array = vtk.vtkCellArray()
    array.SetCells(len(triangles), numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
    poly.SetPolys(array)
    return poly


def _arrays(poly):
    """(vertices, triangles) of a triangle polydata."""
    triangles_filter = vtk.vtkTriangleFilter()
    triangles_filter.SetInputData(poly)
    triangles_filter.PassLinesOff()
    triangles_filter.PassVertsOff()
    triangles_filter.Update()
    poly = triangles_filter.GetOutput()
    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfPolys() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    vertices = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    triangles = numpy_support.vtk_to_numpy(
        poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    return vertices, triangles


def _cross_section(poly, plane):
    """Closed cut contours of ``poly`` by ``plane`` (polylines)."""
    cutter = vtk.vtkCutter()
    cutter.SetInputData(poly)
    cutter.SetCutFunction(plane)
    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    return stripper.GetOutput()


def _cap(poly, plane):
    """Triangles closing the cut of ``poly`` by ``plane``, facing away
    from the kept side (outward)."""
    contours = _cross_section(poly, plane)
    if contours.GetNumberOfPoints() < 3:
        return None
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(contours)
    triangulate = vtk.vtkContourTriangulator()
    triangulate.SetInputConnection(clean.GetOutputPort())
    triangulate.Update()
    vertices, triangles = _arrays(triangulate.GetOutput())
    if len(triangles) == 0:
        return None
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    outward = -np.asarray(plane.GetNormal())
    wrong = normals @ outward < 0
    triangles[wrong] = triangles[wrong][:, [0, 2, 1]]
    return _polydata(vertices, triangles)


def slice_mesh(vertices, triangles, planes, cap=True):
    """Keep the part of the mesh on the kept side of every enabled plane
    -> (vertices, triangles); the cut is exact, ``cap`` closes it."""
    poly = _polydata(vertices, triangles)
    vertices = np.asarray(vertices, dtype=float)
    size = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) if len(vertices) else 1.0
    tolerance = 1e-7 * max(size, 1e-12)
    for plane in active(planes):
        vtk_plane = plane.vtk_plane()
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(poly)
        clip.SetClipFunction(vtk_plane)
        clip.SetValue(0.0)
        clip.Update()
        clipped = clip.GetOutput()
        pieces = [clipped]
        if cap:
            closing = _cap(poly, vtk_plane)
            if closing is not None:
                # the cap of this plane is cut by the planes already applied
                # (poly is the result so far), so it fits the kept part
                pieces.append(closing)
        append = vtk.vtkAppendPolyData()
        for piece in pieces:
            only_geometry = vtk.vtkPolyData()
            only_geometry.SetPoints(piece.GetPoints())
            only_geometry.SetPolys(piece.GetPolys())
            append.AddInputData(only_geometry)
        merge = vtk.vtkCleanPolyData()
        merge.SetInputConnection(append.GetOutputPort())
        merge.PointMergingOn()
        # the cap's edge points and the clipped edge points are the same
        # intersections computed twice - join them (tiny absolute tolerance)
        merge.ToleranceIsAbsoluteOn()
        merge.SetAbsoluteTolerance(tolerance)
        merge.Update()
        poly = merge.GetOutput()
        if poly.GetNumberOfPolys() == 0:
            break
    return _arrays(poly)


def section(vertices, triangles, plane):
    """Cut line of ``plane`` through the mesh -> {"length", "area",
    "loops"} (area = enclosed by closed loops; a closed mesh's cross
    section)."""
    poly = _polydata(vertices, triangles)
    vtk_plane = plane.vtk_plane()
    contours = _cross_section(poly, vtk_plane)
    length = 0.0
    if contours.GetNumberOfPoints():
        points = numpy_support.vtk_to_numpy(contours.GetPoints().GetData())
        lines = contours.GetLines()
        lines.InitTraversal()
        ids = vtk.vtkIdList()
        while lines.GetNextCell(ids):
            index = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
            segment = points[index]
            length += float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
    area = 0.0
    closing = _cap(poly, vtk_plane)
    if closing is not None:
        v, t = _arrays(closing)
        a, b, c = (v[t[:, i]] for i in range(3))
        area = float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)
    return {"length": length, "area": area,
            "loops": contours.GetNumberOfLines() if contours else 0}


def slice_data(kind, vertices, triangles, planes, cap=True):
    """Slice either kind -> (vertices, triangles or None, kept-vertex
    mask or None).  For points the mask selects the kept rows (normals,
    colours follow); a mesh gets new vertices (normals recomputed,
    colours carried by nearest neighbour)."""
    if kind == "points" or triangles is None:
        mask = point_mask(vertices, planes)
        return np.asarray(vertices)[mask], None, mask
    new_vertices, new_triangles = slice_mesh(vertices, triangles, planes, cap)
    return new_vertices, new_triangles, None


def with_changes(plane, **changes):
    return replace(plane, **changes)
