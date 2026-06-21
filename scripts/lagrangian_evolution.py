import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

from scripts.geometry.objects import PointCloud, VoxelCube
from scripts.model.model import Model
from scripts.utils.clouds import generate_point_cloud

def fill_voxel_cube(PC: PointCloud, VC: VoxelCube):
    tree = cKDTree(PC.points_arr)
    for i in range(VC.size):
        for j in range(VC.size):
            for k in range(VC.size):
                x, y, z, = VC.get_coords(i, j, k)
                dist, _ = tree.query([x, y, z], k=1)
                VC.D[i, j, k] = dist


if __name__ == "__main__":
    M = Model("data/ModelNet10/toilet/train/toilet_0001.off")
    M.normalize()

    PC = generate_point_cloud(M,2000)
    VC = VoxelCube(size=64, min_bound=-1.3, max_bound=1.3)
    fill_voxel_cube(PC, VC)
    VC.gradient_all(VC.D)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.2, resolution=80)
    sphere.compute_vertex_normals()
    sphere_vertices = np.asarray(sphere.vertices)

    max_step = 0.01
    alpha = 0.01

    for _ in range(500):
        sphere_vertices = np.asarray(sphere.vertices)

        for idx in range(sphere_vertices.shape[0]):
            v = sphere_vertices[idx]

            dist = VC.interpolate_scalar_field(VC.D, v)
            if dist < 1.5 * VC.h:
                continue

            grad = VC.interpolate_vector_field(VC.grad, v)
            direction = -grad

            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                continue

            direction = direction / norm * alpha

            step_norm = np.linalg.norm(direction)
            if step_norm > max_step:
                direction = direction / step_norm * max_step

            sphere_vertices[idx] = v + direction

        sphere.vertices = o3d.utility.Vector3dVector(sphere_vertices)
        sphere.compute_vertex_normals()
    o3d.visualization.draw_geometries([sphere])