import numpy as np
import open3d as o3d

from scripts.geometry.objects import OctreeNode, Octree
from scripts.model.model import Model
from scripts.utils.clouds import generate_point_cloud

from scipy.sparse.linalg import lsqr
from skimage.measure import marching_cubes

def node_to_lineset(node, color=(0, 1, 0)):
    c = node.center
    h = node.half_size

    corners = np.array([
        [c[0]-h, c[1]-h, c[2]-h],
        [c[0]+h, c[1]-h, c[2]-h],
        [c[0]+h, c[1]+h, c[2]-h],
        [c[0]-h, c[1]+h, c[2]-h],
        [c[0]-h, c[1]-h, c[2]+h],
        [c[0]+h, c[1]-h, c[2]+h],
        [c[0]+h, c[1]+h, c[2]+h],
        [c[0]-h, c[1]+h, c[2]+h],
    ])

    lines = np.array([
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7],
    ])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color), (len(lines), 1))
    )

    return ls


def visualize_octree(tree, points=None, only_non_empty=True):
    leaves = tree.collect_leaves()
    geometries = []

    for leaf in leaves:
        if only_non_empty and len(leaf.indeces) == 0:
            continue
        geometries.append(node_to_lineset(leaf))

    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0, 0])
        geometries.append(pcd)

    o3d.visualization.draw_geometries(geometries)

def marching_cubes_to_mesh(tree, ans, sgm):
    verts, faces, normals, values = marching_cubes(ans, level=sgm)

    q = ans.shape[0]

    root_min = tree.root.center - tree.root.half_size
    step = tree.root.size / (q - 1)

    verts_world = root_min + verts * step

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    return mesh, verts_world, faces

M = Model("data/ModelNet10/desk/train/desk_0001.off")
M.normalize()
D = 6
Q = 4

PC = generate_point_cloud(M,10000)
root = OctreeNode(np.array([0.0, 0.0, 0.0]), 2.2, 0)
tree = Octree(root, PC.points_arr, max_depth=D, min_points=1)
tree.build()
tree.fill_depth_neighborhood(D)
leaves = tree.collect_leaves()

#visualize_octree(tree, points=PC.points_arr, only_non_empty=True)
tree.calculate_normals(PC.points_arr, PC.normals_arr, depth=D)

L = tree.assemble_L(qq=Q, depth=D)
v = tree.calculate_v(qq=Q, depth=D)

x = lsqr(L, v)[0]

field, gamma = tree.get_dense_field(PC.points_arr, depth=D, x=x, q=128)

print(x.min(), x.max())
print(np.linalg.norm(x))
print(field.min(), field.max())
print(gamma)

mesh, verts, faces = marching_cubes_to_mesh(tree, field, gamma)
o3d.visualization.draw_geometries([mesh])