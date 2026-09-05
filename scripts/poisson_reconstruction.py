import open3d as o3d
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from skimage.measure import marching_cubes

from scripts.geometry.objects import PointCloud, VoxelCube
from scripts.model.model import Model
from scripts.utils.clouds import generate_point_cloud
from tqdm import tqdm

def fill_voxel_cube(PC: PointCloud, VC: VoxelCube):
    arr = PC.points_arr
    n_arr = PC.normals_arr
    for p in range(arr.shape[0]):
        x, y, z = arr[p]
        n_x, n_y, n_z = n_arr[p]
        i, j, k = VC.get_voxel(x, y, z)
        VC.V[i, j, k] += np.array([n_x, n_y, n_z])
        VC.counts[i, j, k] += 1
    mask = VC.counts > 0
    VC.V[mask] = VC.V[mask] / VC.counts[mask][:, None]

def create_sparse_matrix(VC: VoxelCube):
    N = VC.size
    sparse_matrix = lil_matrix((N**3, N**3))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                p = i * N ** 2 + j * N + k
                neighbors = 0
                if i + 1 < N:
                    p_in1 = (i + 1) * N ** 2 + j * N + k
                    sparse_matrix[p, p_in1] = 1
                    neighbors += 1
                if i - 1 >= 0:
                    p_in2 = (i - 1) * N ** 2 + j * N + k
                    sparse_matrix[p, p_in2] = 1
                    neighbors += 1
                if j + 1 < N:
                    p_jn1 = i * N ** 2 + (j + 1) * N + k
                    sparse_matrix[p, p_jn1] = 1
                    neighbors += 1
                if j - 1 >= 0:
                    p_jn2 = i * N ** 2 + (j - 1) * N + k
                    sparse_matrix[p, p_jn2] = 1
                    neighbors += 1
                if k + 1 < N:
                    p_kn1 = i * N ** 2 + j * N + k + 1
                    sparse_matrix[p, p_kn1] = 1
                    neighbors += 1
                if k - 1 >= 0:
                    p_kn2 = i * N ** 2 + j * N + k - 1
                    sparse_matrix[p, p_kn2] = 1
                    neighbors += 1
                sparse_matrix[p, p] = -neighbors
    sparse_matrix[0, :] = 0
    sparse_matrix [0, 0] = 1
    return sparse_matrix.tocsr()

def marching_cubes_to_mesh(VC: VoxelCube, ans, sgm):
    verts, faces, normals, values = marching_cubes(ans, level=sgm)

    verts_world = np.zeros_like(verts, dtype=float)
    verts_world[:, 0] = -1 + 2 * verts[:, 0] / (VC.size - 1)
    verts_world[:, 1] = -1 + 2 * verts[:, 1] / (VC.size - 1)
    verts_world[:, 2] = -1 + 2 * verts[:, 2] / (VC.size - 1)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    return mesh, verts_world, faces

if __name__ == "__main__":
    M = Model("data/ModelNet10/toilet/train/toilet_0001.off")
    M.normalize()

    PC = generate_point_cloud(M,8000)
    VC = VoxelCube(size=32, min_bound=-1, max_bound=1)
    fill_voxel_cube(PC, VC)

    PC.visualize_oriented_cloud()
    VC.divergence_all(VC.V)
    b = VC.div.flatten()
    b *= VC.h ** 2
    b[0] = 0
    ans = spsolve(create_sparse_matrix(VC), b)
    ans = ans.reshape((VC.size, VC.size, VC.size))

    sgm = 0
    for point in PC.points_arr:
        x, y, z = point
        i, j, k = VC.get_voxel(x, y, z)
        sgm += ans[i, j, k]
    sgm /= PC.points_arr.shape[0]

    mesh, verts, faces = marching_cubes_to_mesh(VC, ans, sgm)
    o3d.visualization.draw_geometries([mesh])
