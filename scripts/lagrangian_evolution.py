import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import identity, diags
from scipy.sparse.linalg import spsolve

from scripts.geometry.objects import PointCloud, VoxelCube, MeshGeometry
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

def compute_neg_gradient(VC: VoxelCube):
    grad = VC.grad
    norm = np.linalg.norm(grad, axis=-1, keepdims=True)
    VC.neg_grad = -grad / (norm + 1e-12)
    return VC.neg_grad

def epsilon_func(d, C1=1.0, C2=1.0):
    if d < 0:
        return 0.0
    return C1 * (1.0 - np.exp(-(d * d) / C2))

def rho_func(d, A=1.0):
    if d < 0:
        return 0.0
    return A * (1.0 - np.exp(-(d * d)))

def eta_func(d, neg_grad, norm, D1=1.0, D2=1.0):
    if d < 0:
        return 0.0
    
    s = np.dot(neg_grad, norm)
    s = np.clip(s, -1.0, 1.0)

    return D1 * d * (s - D2 * np.sqrt(max(0.0, 1.0 - s * s)))

def lsw_step(MG:MeshGeometry, VC: VoxelCube, iso_level=0.0, tau=0.01, lmbd = 1):
    normals = MG.compute_vertex_normals()
    areas = MG.compute_vertex_areas()
    neighbors = MG.build_vertex_ordered_neighbors()
    vt = MG.compute_tangential_velocities(normals, neighbors)

    n = MG.V.shape[0]
    d = np.zeros(n)
    neg_grad = np.zeros((n, 3))
    eps = np.zeros(n)
    eta = np.zeros(n)
    rho = np.zeros(n)
    M = diags(areas, format='csr')

    for i in range(n):
        p = MG.V[i]

        d[i] = VC.interpolate_scalar_field(VC.D, p) - iso_level
        neg_grad[i] = VC.interpolate_vector_field(VC.neg_grad, p[None, :])[0]

        eps[i] = epsilon_func(d[i])
        eta[i] = eta_func(d[i], neg_grad[i], normals[i])
        rho[i] = rho_func(d[i])
    print(f"meand d: {np.mean(d)}, min d: {np.min(d)}, max d: {np.max(d)}")
    print(f"min eta: {np.min(eta)}, max eta: {np.max(eta)}, min epsilon: {np.min(eps)}, max epsilon: {np.max(eps)}")
    
    normal_vel = eta[:, None] * normals
    tangent_vel = lmbd * vt

    print("mean |normal|:", np.mean(np.linalg.norm(normal_vel, axis=1)))
    print("mean |tangent|:", np.mean(np.linalg.norm(tangent_vel, axis=1)))
    print("max |normal|:", np.max(np.linalg.norm(normal_vel, axis=1)))
    print("max |tangent|:", np.max(np.linalg.norm(tangent_vel, axis=1)))

    rho = np.maximum(rho, 0.05)

    L = MG.cotangent_laplacian()

    rhs = M @ MG.V.copy()
    rhs += tau * M @ (eta[:, None] * normals + lmbd * vt)

    A = M + tau * diags(eps, format='csr') @ L

    V_new = np.zeros_like(MG.V)
    V_new[:, 0] = spsolve(A, rhs[:, 0])
    V_new[:, 1] = spsolve(A, rhs[:, 1])
    V_new[:, 2] = spsolve(A, rhs[:, 2])

    return V_new, np.mean(d)

if __name__ == "__main__":
    M = Model("data/ModelNet10/toilet/train/toilet_0001.off")
    M.normalize()

    PC = generate_point_cloud(M,10000)
    VC = VoxelCube(size=64, min_bound=-1.3, max_bound=1.3)
    fill_voxel_cube(PC, VC)
    VC.gradient_all(VC.D)
    compute_neg_gradient(VC)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.2, resolution=30)
    sphere.compute_vertex_normals()
    sphere_vertices = np.asarray(sphere.vertices)
    sphere_triangles = np.asarray(sphere.triangles)

    MG = MeshGeometry(sphere_vertices, sphere_triangles)
    ordered_neighbors = MG.build_vertex_ordered_neighbors()

    bad = sum(1 for ring in ordered_neighbors if len(ring) < 3)
    print(bad)

    tau = 0.03
    remesh_every = 5

    for step in range(70):
        MG.V, mean_d = lsw_step(
            MG,
            VC,
            iso_level=0.0,
            tau=tau,
            lmbd=0.2,
        )

        if not np.all(np.isfinite(MG.V)):
            print("NaN detected, stop")
            break

        if step % remesh_every == 0 and step > 0:
            collapsed, splitted, flipped = MG.adaptive_remesh(
                VC,
                iso_level=0.0,
                min_len=0.25 * VC.h,
                max_len=4.0 * VC.h,
                gamma_crit=0.4,
            )

            lengths = MG.edge_lengths()

            print("after adaptive remesh")
            print("collapsed:", collapsed, "splitted:", splitted, "flipped:", flipped)
            print("vertices:", MG.V.shape[0], "faces:", MG.F.shape[0])
            print("edge min:", lengths.min())
            print("edge mean:", lengths.mean())
            print("edge max:", lengths.max())

    sphere.vertices = o3d.utility.Vector3dVector(MG.V)
    sphere.triangles = o3d.utility.Vector3iVector(MG.F)
    sphere.compute_vertex_normals()

    o3d.visualization.draw_geometries([sphere])

    max_step = 0.01
    alpha = 0.01

    # for _ in range(500):
    #     sphere_vertices = np.asarray(sphere.vertices)

    #     for idx in range(sphere_vertices.shape[0]):
    #         v = sphere_vertices[idx]

    #         dist = VC.interpolate_scalar_field(VC.D, v)
    #         if dist < 1.5 * VC.h:
    #             continue

    #         grad = VC.interpolate_vector_field(VC.grad, v)
    #         direction = -grad

    #         norm = np.linalg.norm(direction)
    #         if norm < 1e-8:
    #             continue

    #         direction = direction / norm * alpha

    #         step_norm = np.linalg.norm(direction)
    #         if step_norm > max_step:
    #             direction = direction / step_norm * max_step

    #         sphere_vertices[idx] = v + direction

    #     sphere.vertices = o3d.utility.Vector3dVector(sphere_vertices)
    #     sphere.compute_vertex_normals()
    # o3d.visualization.draw_geometries([sphere])