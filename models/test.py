# import open3d as o3d
# import numpy as np

# points  = np.loadtxt("points.txt")
# normals = np.loadtxt("normals.txt")

# pcd = o3d.geometry.PointCloud()
# pcd.points  = o3d.utility.Vector3dVector(points)
# pcd.normals = o3d.utility.Vector3dVector(normals)

# pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 100.0]))
# # pcd.orient_normals_consistent_tangent_plane(k=30)

# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# densities_np = np.asarray(densities)
# mesh.remove_vertices_by_mask(densities_np < np.quantile(densities_np, 0.01))

# # mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
# mesh = mesh.filter_smooth_taubin(number_of_iterations=5)

# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([0.8, 0.8, 0.8])

# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

import open3d as o3d
import numpy as np

# =====================
# 1. Load data
# =====================
points  = np.loadtxt("points_changed.txt")
normals = np.loadtxt("normals.txt")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# اگر نرمال داری → نرمال‌سازی کن
normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
pcd.normals = o3d.utility.Vector3dVector(normals)

# =====================
# 2. Preprocessing (خیلی مهم)
# =====================

# Downsample برای یکنواختی
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# حذف نویز
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# اگر نرمال‌هات خوب نیستن، اینو فعال کن:
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
# )

# جهت‌دهی درست نرمال‌ها (بهترین حالت عمومی)
pcd.orient_normals_consistent_tangent_plane(k=30)

# =====================
# 3. Poisson Reconstruction
# =====================
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=10,        # تعادل عالی بین کیفیت و سرعت
    width=0,         # auto
    scale=1.1,       # جلوگیری از برش لبه‌ها
    linear_fit=False
)

# =====================
# 4. Density Filtering (کلیدی)
# =====================
densities = np.asarray(densities)

# sweet spot معمول
threshold = np.quantile(densities, 0.02)

mesh.remove_vertices_by_mask(densities < threshold)

# =====================
# 5. Crop (خیلی مهم برای حذف حباب Poisson)
# =====================
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

# =====================
# 6. Smoothing (حرفه‌ای)
# =====================
mesh = mesh.filter_smooth_taubin(
    number_of_iterations=5
)

# =====================
# 7. Final touches
# =====================
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.8, 0.8, 0.8])

# =====================
# 8. Visualization
# =====================
o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_back_face=True
)
