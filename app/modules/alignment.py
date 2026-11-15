# MeshBench - Copyright (c) 2026 Sobhan Shirvanloo. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Rigid alignment of a compared model onto a reference (registration).

Used by the Compare tab when the two files are not in the same
coordinates (e.g. the reconstruction was exported after being rotated):

1. **check** - median distance of compared surface samples to the
   reference; below ``ALIGNED_SHARE`` of the diagonal the models are
   taken as aligned and left alone (re-fitting an already aligned
   reconstruction would hide part of its real error);
2. **global** - FPFH features + RANSAC on a coarse sampling (finds the
   pose from any starting angle / position), plus two cheap candidates
   (unchanged, centroids matched);
3. **refine** - point-to-plane ICP at three decreasing distances on a
   denser sampling; the candidate with the best final fit wins.

Everything is seeded, so the same files always give the same result.
"""

import numpy as np
import open3d as o3d

ALIGNED_SHARE = 0.005      # median gap < 0.5 % of the diagonal = aligned
SAMPLES = 30_000
GLOBAL_VOXELS = 40         # coarse voxel = diagonal / 40
FINE_VOXELS = 150          # ICP sampling voxel = diagonal / 150

reg = o3d.pipelines.registration


def _cloud(model, count, seed):
    points, normals = model.samples(count, seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def misalignment(compared, reference, seed=0):
    """Median distance (model units) from compared samples to the
    reference surface (or its points)."""
    query = np.asarray(_cloud(compared, 5000, seed).points)
    if reference.has_triangles:
        origin = reference.vertices.mean(axis=0)   # float32-safe
        local = o3d.geometry.TriangleMesh(reference.mesh)
        local.translate(-origin)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(local))
        distance = scene.compute_distance(
            o3d.core.Tensor((query - origin).astype(np.float32))).numpy()
    else:
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference.vertices))
        distance = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(query)).compute_point_cloud_distance(target))
    return float(np.median(distance))


def is_aligned(compared, reference):
    diagonal = max(compared.diagonal, reference.diagonal)
    return misalignment(compared, reference) <= ALIGNED_SHARE * diagonal


def _features(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(2 * voxel, 30))
    feature = reg.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(5 * voxel, 100))
    return down, feature


def align(compared, reference, seed=0):
    """4x4 matrix moving ``compared`` onto ``reference`` and a summary
    {fitness, rmse, method}."""
    o3d.utility.random.seed(seed)
    diagonal = max(compared.diagonal, reference.diagonal)
    source = _cloud(compared, SAMPLES, seed)
    target = _cloud(reference, SAMPLES, seed + 1)

    # --- candidates: unchanged, centroids matched, FPFH + RANSAC ---
    candidates = {"unchanged": np.eye(4)}
    shift = np.eye(4)
    shift[:3, 3] = target.get_center() - source.get_center()
    candidates["centroids"] = shift
    voxel = diagonal / GLOBAL_VOXELS
    source_down, source_feature = _features(source, voxel)
    target_down, target_feature = _features(target, voxel)
    ransac = reg.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_feature, target_feature, True,
        1.5 * voxel, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         reg.CorrespondenceCheckerBasedOnDistance(1.5 * voxel)],
        reg.RANSACConvergenceCriteria(100_000, 0.999))
    candidates["features (RANSAC)"] = np.asarray(ransac.transformation)

    # --- refine each with point-to-plane ICP, keep the best fit ---
    fine = diagonal / FINE_VOXELS
    source_fine = source.voxel_down_sample(fine)
    target_fine = target.voxel_down_sample(fine)
    best = None
    for method, start in candidates.items():
        matrix = start
        for distance in (4 * voxel, voxel, 2 * fine):
            result = reg.registration_icp(
                source_fine, target_fine, distance, matrix,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=60))
            matrix = result.transformation
        score = (round(result.fitness, 3), -result.inlier_rmse)
        if best is None or score > best[0]:
            best = (score, np.asarray(matrix), result, method)
    _, matrix, result, method = best
    return matrix, {"fitness": float(result.fitness),
                    "rmse": float(result.inlier_rmse), "method": method}


def describe(matrix):
    """Rotation angles (degrees, X / Y / Z as vtkProp3D uses them) and
    the translation of a rigid 4x4 matrix."""
    rotation = matrix[:3, :3]
    # vtkProp3D order: R = Rz · Rx · Ry
    x = np.degrees(np.arcsin(np.clip(rotation[2, 1], -1, 1)))
    y = np.degrees(np.arctan2(-rotation[2, 0], rotation[2, 2]))
    z = np.degrees(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    return (x, y, z), matrix[:3, 3]
