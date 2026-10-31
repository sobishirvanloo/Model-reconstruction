"""Standalone version of the Reconstruction-tab pipeline.

Usage:
    python test.py [points.txt] [normals.txt]

The points and normals files must come from the same export (same row
order).  A points file with 6 columns (x y z nx ny nz) can be passed alone.
"""

import os
import sys

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))


def orient_keep_sign(pcd, reference):
    """Consistent orientation without flipping the surface inside out."""
    pcd.orient_normals_consistent_tangent_plane(k=30)
    normals = np.asarray(pcd.normals)
    if np.einsum("ij,ij->i", normals, reference).sum() < 0:
        pcd.normals = o3d.utility.Vector3dVector(-normals)


def main():
    points_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "points.txt")
    normals_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "normals.txt")

    # =====================
    # 1. Load data
    # =====================
    data = np.loadtxt(points_path, ndmin=2)
    points = data[:, :3]
    if data.shape[1] >= 6 and len(sys.argv) <= 2:
        normals = data[:, 3:6]
    else:
        normals = np.loadtxt(normals_path, ndmin=2)[:, :3]
    if len(points) != len(normals):
        sys.exit(f"Mismatch: {len(points)} points vs {len(normals)} normals")

    # نرمال‌سازی نرمال‌ها
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # =====================
    # 2. Preprocessing (خیلی مهم)
    # =====================

    # Downsample برای یکنواختی
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # حذف نویز
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # جهت‌دهی درست نرمال‌ها (علامت کلی نرمال‌های ورودی حفظ می‌شود)
    orient_keep_sign(pcd, np.asarray(pcd.normals).copy())

    # =====================
    # 3. Poisson Reconstruction
    # =====================
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=10,        # تعادل عالی بین کیفیت و سرعت
        scale=1.1,       # جلوگیری از برش لبه‌ها
        linear_fit=False
    )

    # =====================
    # 4. Density Filtering (کلیدی)
    # =====================
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.02)   # sweet spot معمول
    mesh.remove_vertices_by_mask(densities < threshold)

    # =====================
    # 5. Crop (خیلی مهم برای حذف حباب Poisson)
    # =====================
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

    # =====================
    # 6. Smoothing
    # =====================
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)

    # =====================
    # 7. Final touches + visualization
    # =====================
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    print(f"vertices: {len(mesh.vertices)} | triangles: {len(mesh.triangles)}")

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()
