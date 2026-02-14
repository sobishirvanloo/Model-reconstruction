import os
import numpy as np
import open3d as o3d
import vtk


class GeometryHandler:
    def __init__(self):
        self.geometry = None
        self.points = None
        self.normals = None
        self.is_pointcloud = False
        self.actor = None
        self.vtk_initialized = False
        self.renderer = None
        self.interactor = None

    # =========================
    # Import Geometry
    # =========================
    def import_geometry(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pcd":
            self.is_pointcloud = True
            self.geometry = o3d.io.read_point_cloud(file_path)
            if not self.geometry.has_normals():
                self.geometry.estimate_normals()
            self.points = np.asarray(self.geometry.points)
            self.normals = np.asarray(self.geometry.normals)
        else:
            self.is_pointcloud = False
            self.geometry = o3d.io.read_triangle_mesh(file_path)
            self.geometry.compute_vertex_normals()
            if len(self.geometry.triangles) == 0:
                raise ValueError("Mesh has no triangles!")
            self.points = np.asarray(self.geometry.vertices)
            self.normals = np.asarray(self.geometry.vertex_normals)

    # =========================
    # Export Geometry
    # =========================
    def export_geometry(self, file_path_points, file_path_normals):
        if self.geometry is None:
            return
        np.savetxt(file_path_points, self.points, fmt="%.6f")
        np.savetxt(file_path_normals, self.normals, fmt="%.6f")

    # =========================
    # Render with VTK - اصلاح شده
    # =========================
    def render(self, renderer, interactor):
        """Render geometry and return the actor"""
        if self.geometry is None:
            print("[ERROR] Geometry is None in render()")
            return None

        self.renderer = renderer
        self.interactor = interactor

        # Remove previous actor
        if self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None

        # ========== ساخت VTK Points ==========
        vtk_points = vtk.vtkPoints()
        vtk_normals = vtk.vtkFloatArray()
        vtk_normals.SetNumberOfComponents(3)
        vtk_normals.SetName("Normals")

        for p, n in zip(self.points, self.normals):
            vtk_points.InsertNextPoint(p[0], p[1], p[2])
            vtk_normals.InsertNextTuple((n[0], n[1], n[2]))

        # ========== ساخت PolyData ==========
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetNormals(vtk_normals)

        # اضافه کردن cells (فقط برای Mesh، نه PointCloud)
        if not self.is_pointcloud:
            vtk_cells = vtk.vtkCellArray()
            for tri in np.asarray(self.geometry.triangles):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, tri[0])
                triangle.GetPointIds().SetId(1, tri[1])
                triangle.GetPointIds().SetId(2, tri[2])
                vtk_cells.InsertNextCell(triangle)
            polydata.SetPolys(vtk_cells)

        # PointCloud رو به Glyphs تبدیل کن
        if self.is_pointcloud:
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            polydata = glyph.GetOutput()

        # ========== ساخت Mapper و Actor ==========
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        self.actor.GetProperty().SetPointSize(2)
        self.actor.GetProperty().SetInterpolationToPhong()

        # اضافه کردن به Renderer
        self.renderer.AddActor(self.actor)
        self.renderer.ResetCamera()

        # Initialize VTK
        if not self.vtk_initialized:
            self.interactor.Initialize()
            self.vtk_initialized = True

        self.interactor.GetRenderWindow().Render()

        # ✅ برگرداندن actor - این بود که فقط کافی بود!
        print(f"[DEBUG] Actor created and returned: {self.actor}")
        return self.actor