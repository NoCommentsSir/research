import open3d as o3d
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm

class Point:
    def __init__(self, x, y, z, triangle_id):
        self.x = x
        self.y = y
        self.z = z
        self.triangle_id = triangle_id

class PointCloud:
    def __init__(self, points_arr, normals_arr):
        self.points_arr = points_arr.astype(float)
        self.normals_arr = normals_arr.astype(float)

    def visualize_oriented_cloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_arr)
        pcd.normals = o3d.utility.Vector3dVector(self.normals_arr)
        o3d.visualization.draw_geometries([pcd])

def gradient_at_vector(field, i, j, k, size, h):
        if i > 0 and i < size - 1 and j > 0 and j < size - 1 and k > 0 and k < size - 1:
            dVx = (field[i + 1, j, k, 0] - field[i - 1, j, k, 0]) / (2 * h)
            dVy = (field[i, j + 1, k, 1] - field[i, j - 1, k, 1]) / (2 * h)
            dVz = (field[i, j, k + 1, 2] - field[i, j, k - 1, 2]) / (2 * h)
            return np.array([dVx, dVy, dVz])
        else:
            return np.array([0.0, 0.0, 0.0])
        
def gradient_at_scalar(field, i, j, k, size, h):
        if i > 0 and i < size - 1 and j > 0 and j < size - 1 and k > 0 and k < size - 1:
            dVx = (field[i + 1, j, k] - field[i - 1, j, k]) / (2 * h)
            dVy = (field[i, j + 1, k] - field[i, j - 1, k]) / (2 * h)
            dVz = (field[i, j, k + 1] - field[i, j, k - 1]) / (2 * h)
            return np.array([dVx, dVy, dVz])
        else:
            return np.array([0.0, 0.0, 0.0])

def get_gradient_at(matrix, i, j, k, size, h):
        if matrix.ndim == 4:
            return gradient_at_vector(matrix, i, j, k, size, h)
        else:
            return gradient_at_scalar(matrix, i, j, k, size, h)

def divergence_at(field, i, j, k, size, h):
        if i > 0 and i < size - 1 and j > 0 and j < size - 1 and k > 0 and k < size - 1:
            dVx, dVy, dVz = get_gradient_at(field, i, j, k, size, h)
            return dVx + dVy + dVz
        else:
            return 0
        
def support_bounds(node):
    r = 2 * node.half_size
    return node.center - r, node.center + r


def intersect_bounds(a_min, a_max, b_min, b_max):
    mn = np.maximum(a_min, b_min)
    mx = np.minimum(a_max, b_max)

    if np.any(mx <= mn):
        return None, None

    return mn, mx
        
class VoxelCube:
    def __init__(self, size=32, min_bound=-1.0, max_bound=1.0):
        self.size = size
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.length = max_bound - min_bound

        self.V = np.zeros((size, size, size, 3))
        self.counts = np.zeros((size, size, size), dtype=int)
        self.div = np.zeros((size, size, size))
        self.grad = np.zeros((size, size, size, 3))
        self.D = np.zeros((size, size, size))

        self.h = self.length / (self.size - 1)

    def get_voxel(self, x, y, z):
        i = np.floor(((x - self.min_bound) / self.length) * (self.size - 1)).astype(int)
        j = np.floor(((y - self.min_bound) / self.length) * (self.size - 1)).astype(int)
        k = np.floor(((z - self.min_bound) / self.length) * (self.size - 1)).astype(int)

        i = np.clip(i, 0, self.size - 1)
        j = np.clip(j, 0, self.size - 1)
        k = np.clip(k, 0, self.size - 1)

        return int(i), int(j), int(k)

    def get_coords(self, i, j, k):
        x = self.min_bound + (i / (self.size - 1)) * self.length
        y = self.min_bound + (j / (self.size - 1)) * self.length
        z = self.min_bound + (k / (self.size - 1)) * self.length
        return np.array([x, y, z], dtype=float)
        
    def divergence_all(self, field):
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    self.div[i, j, k] = divergence_at(field, i, j, k, self.size, self.h)
        return self.div
    
    def gradient_all(self, field):
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    self.grad[i, j, k] = get_gradient_at(field, i, j, k, self.size, self.h)
        return self.grad

    def extract_iso_surface_points(self, field, iso_level):
        surface_points = []

        cube_corners = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]

        cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for i in range(self.size - 1):
            for j in range(self.size - 1):
                for k in range(self.size - 1):
                    corner_positions = []
                    corner_values = []

                    for dx, dy, dz in cube_corners:
                        gi = i + dx
                        gj = j + dy
                        gk = k + dz

                        corner_positions.append(self.get_coords(gi, gj, gk))
                        corner_values.append(field[gi, gj, gk])

                    corner_values = np.array(corner_values)

                    if not (corner_values.min() <= iso_level <= corner_values.max()):
                        continue

                    intersections = []

                    for a, b in cube_edges:
                        va = corner_values[a]
                        vb = corner_values[b]

                        fa = va - iso_level
                        fb = vb - iso_level

                        if fa == 0:
                            intersections.append(corner_positions[a])
                            continue

                        if fb == 0:
                            intersections.append(corner_positions[b])
                            continue

                        if fa * fb < 0:
                            pa = corner_positions[a]
                            pb = corner_positions[b]

                            t = (iso_level - va) / (vb - va)
                            p = pa + t * (pb - pa)

                            intersections.append(p)

                    if len(intersections) > 0:
                        surface_point = np.mean(intersections, axis=0)
                        surface_points.append(surface_point)

        return np.array(surface_points, dtype=float)
    
    def interpolate_vector_field(self, field, p):
        gx = ((p[0] - self.min_bound) / self.length) * (self.size - 1)
        gy = ((p[1] - self.min_bound) / self.length) * (self.size - 1)
        gz = ((p[2] - self.min_bound) / self.length) * (self.size - 1)
        i, j, k = int(np.floor(gx)), int(np.floor(gy)), int(np.floor(gz))
        i1, j1, k1 = min(i + 1, self.size - 1), min(j + 1, self.size - 1), min(k + 1, self.size - 1)
        tx = gx - i
        ty = gy - j
        tz = gz - k
        w000 = (1 - tx) * (1 - ty) * (1 - tz)
        w001 = (1 - tx) * (1 - ty) * tz
        w010 = (1 - tx) * ty * (1 - tz)
        w011 = (1 - tx) * ty * tz
        w100 = tx * (1 - ty) * (1 - tz)
        w101 = tx * (1 - ty) * tz
        w110 = tx * ty * (1 - tz)
        w111 = tx * ty * tz
        c000 = field[i, j, k]
        c001 = field[i, j, k1]
        c010 = field[i, j1, k]
        c011 = field[i, j1, k1]
        c100 = field[i1, j, k]
        c101 = field[i1, j, k1]
        c110 = field[i1, j1, k]
        c111 = field[i1, j1, k1]
        grad_total = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011 + w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)
        return grad_total 
    
    def interpolate_scalar_field(self, field, p):
        gx = ((p[0] - self.min_bound) / self.length) * (self.size - 1)
        gy = ((p[1] - self.min_bound) / self.length) * (self.size - 1)
        gz = ((p[2] - self.min_bound) / self.length) * (self.size - 1)

        gx = np.clip(gx, 0, self.size - 2)
        gy = np.clip(gy, 0, self.size - 2)
        gz = np.clip(gz, 0, self.size - 2)

        i, j, k = int(np.floor(gx)), int(np.floor(gy)), int(np.floor(gz))
        i1, j1, k1 = i + 1, j + 1, k + 1

        tx, ty, tz = gx - i, gy - j, gz - k

        c000 = field[i, j, k]
        c001 = field[i, j, k1]
        c010 = field[i, j1, k]
        c011 = field[i, j1, k1]
        c100 = field[i1, j, k]
        c101 = field[i1, j, k1]
        c110 = field[i1, j1, k]
        c111 = field[i1, j1, k1]

        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tz) + c1 * tz
    
class OctreeNode:

    def __init__(self, center: np.ndarray, size: float, depth=0):
        self.center = center
        self.size = size
        self.half_size = size / 2
        self.children = []
        self.depth = depth
        self.indeces = []
        self.vector_field = np.zeros(3)
        self.grid_index = None

    def _is_leaf(self) -> bool:
        return len(self.children) == 0

    def contains_point(self, point: np.ndarray) -> bool:
        return np.all(np.abs(point - self.center) <= self.half_size)

    def subdivide(self):
        child_size = self.size / 2
        child_half_size = self.half_size / 2
        ix, iy, iz = self.grid_index

        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0,1]:
                    dx = -child_half_size if i == 0 else child_half_size
                    dy = -child_half_size if j == 0 else child_half_size
                    dz = -child_half_size if k == 0 else child_half_size

                    child_center = self.center + np.array([dx, dy, dz])
                    child = OctreeNode(child_center, child_size, self.depth + 1)
                    child.grid_index = (2*ix + i, 2*iy + j, 2*iz + k)

                    self.children.append(child)

    def local_coords(self, point):
        return (point - self.center) / self.half_size
    
    def B1(self, t):
        a = abs(t)
        if a < 1:
            return (4 - 6 * t**2 + 3 * a**3) / 6
        if a < 2:
            return (2 - a)**3 / 6
        return 0.0

    def dB1(self, t):
        a = abs(t)
        if a < 1:
            return -2 * t + 1.5 * a * t
        if a < 2:
            return -0.5 * (2 - a)**2 * np.sign(t)
        return 0.0
    
    def ddB1(self, t):
        a = abs(t)
        if a < 1:
            return -2 + 3 * a 
        if a < 2:
            return 2 - a
        return 0.0

    def basis_func(self, point):
        tx, ty, tz = self.local_coords(point)
        return self.B1(tx) * self.B1(ty) * self.B1(tz)
    
    def basis_grad(self, point):
        tx, ty, tz = self.local_coords(point)
        bx, by, bz = self.B1(tx), self.B1(ty), self.B1(tz)
        dbx, dby, dbz = self.dB1(tx), self.dB1(ty), self.dB1(tz)
        return np.array([
            dbx * by * bz / self.half_size,
            bx * dby * bz / self.half_size,
            bx * by * dbz / self.half_size
        ])
    
    def basis_laplasian(self, point):
        tx, ty, tz = self.local_coords(point)
        bx, by, bz = self.B1(tx), self.B1(ty), self.B1(tz)
        ddbx, ddby, ddbz = self.ddB1(tx), self.ddB1(ty), self.ddB1(tz)
        return (ddbx * by * bz + ddby * bx * bz + ddbz * bx * by) / self.half_size ** 2 

class Octree:

    def __init__(self, root: OctreeNode, points: np.ndarray, max_depth: int, min_points: int = 20):
        self.root = root
        self.points = points
        self.max_depth = max_depth
        self.min_points = min_points
        self.root.indeces = list(range(points.shape[0]))
        self.depth_map = {}
        root.grid_index = (0,0,0)
        self.v = None

    def get_child_id(self, center: np.ndarray, point: np.ndarray):
        child_id = 0

        if point[0] >= center[0]:
            child_id += 4
        if point[1] >= center[1]:
            child_id += 2
        if point[2] >= center[2]:
            child_id += 1

        return child_id

    def subdivide_node(self, node: OctreeNode):
        if node.depth >= self.max_depth:
            return
        
        node.subdivide()

        for point_index in node.indeces:
            p = self.points[point_index]
            child_id = self.get_child_id(node.center, p)
            node.children[child_id].indeces.append(point_index)

        node.indeces = []

        for child in node.children:
            self.subdivide_node(child)
    
    def build(self):
        self.subdivide_node(self.root)
        self.depth_map = self.get_nodes_by_depth()

    def collect_leaves(self, node=None, leaves=None):
        if node is None:
            node = self.root

        if leaves is None:
            leaves = []

        if node._is_leaf():
            leaves.append(node)
            return leaves

        for child in node.children:
            self.collect_leaves(child, leaves)

        return leaves
    
    def get_nodes_by_depth(self):
        nodes_by_depth = {}
        arr = [self.root]

        while len(arr) > 0:
            node = arr.pop()

            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = {}

            nodes_by_depth[node.depth][node.grid_index] = node

            for child in node.children:
                arr.append(child)
        
        return nodes_by_depth
        
    def get_8_depth_nodes_with_weights(self, p, depth:int):
        root_min = self.root.center - self.root.half_size
        root_size = self.root.size
        R = 2 ** depth
        gx = ((p[0] - root_min[0]) / root_size) * (R - 1)
        gy = ((p[1] - root_min[1]) / root_size) * (R - 1)
        gz = ((p[2] - root_min[2]) / root_size) * (R - 1)

        gx = np.clip(gx, 0, R - 2)
        gy = np.clip(gy, 0, R - 2)
        gz = np.clip(gz, 0, R - 2)

        i, j, k = int(np.floor(gx)), int(np.floor(gy)), int(np.floor(gz))
        i1, j1, k1 = i + 1, j + 1, k + 1

        tx, ty, tz = gx - i, gy - j, gz - k
        w000 = (1 - tx) * (1 - ty) * (1 - tz)
        w001 = (1 - tx) * (1 - ty) * tz
        w010 = (1 - tx) * ty * (1 - tz)
        w011 = (1 - tx) * ty * tz
        w100 = tx * (1 - ty) * (1 - tz)
        w101 = tx * (1 - ty) * tz
        w110 = tx * ty * (1 - tz)
        w111 = tx * ty * tz

        candidates = [
            ((i, j, k), w000),
            ((i, j, k1), w001),
            ((i, j1, k), w010),
            ((i, j1, k1), w011),
            ((i1, j, k), w100),
            ((i1, j, k1), w101),
            ((i1, j1, k), w110),
            ((i1, j1, k1), w111),
        ]

        result = []
        mp = self.depth_map[depth]

        for idx, weight in candidates:
            node = mp.get(idx)
            if node is not None:
                result.append((node, weight))

        return result
    
    def calculate_normals(self, points, normals, depth: int):
        for p, n in zip(points, normals):
            nodes = self.get_8_depth_nodes_with_weights(p, depth)

            for node, weight in nodes:
                node.vector_field += weight * n

    def calculate_V(self, point, depth):
        res = np.zeros(3, dtype=np.float64)

        root_min = self.root.center - self.root.half_size
        root_size = self.root.size
        R = 2 ** depth

        g = ((point - root_min) / root_size) * R
        ix, iy, iz = np.floor(g).astype(int)

        mp = self.depth_map[depth]

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    node = mp.get((ix + dx, iy + dy, iz + dz))
                    if node is None:
                        continue

                    f = node.basis_func(point)
                    if f != 0:
                        res += f * node.vector_field

        return res
    
    def calculate_v(self, qq, depth):
        res = []
        for node in self.depth_map[depth].values():
            min_bound = node.center - node.size
            max_bound = node.center + node.size

            l = max_bound - min_bound
            sv = l[0] * l[1] * l[2]
            w = sv / (qq ** 3)
            step = l / qq
            rhs = 0.0

            for a in range(qq):
                for b in range(qq):
                    for c in range(qq):
                        q = np.array([min_bound[0] + step[0] * (a + 0.5), min_bound[1] + step[1] * (b + 0.5), min_bound[2] + step[2] * (c + 0.5)])
                        Vq = self.calculate_V(q, depth)
                        gradF = node.basis_grad(q)
                        rhs += -np.dot(Vq, gradF) * w

            res.append(rhs)
        return np.array(res)

    def assemble_L(self, qq, depth):
        nodes = list(self.depth_map[depth].values())

        for idx, node in enumerate(nodes):
            node.node_id = idx

        L = lil_matrix((len(nodes), len(nodes)), dtype=np.float64)
        mp = self.depth_map[depth]

        for node_i in nodes:
            i = node_i.node_id
            ix, iy, iz = node_i.grid_index

            min_i, max_i = support_bounds(node_i)

            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    for dz in range(-4, 5):
                        node_j = mp.get((ix + dx, iy + dy, iz + dz))
                        if node_j is None:
                            continue

                        j = node_j.node_id

                        min_j, max_j = support_bounds(node_j)

                        mn, mx = intersect_bounds(min_i, max_i, min_j, max_j)
                        if mn is None:
                            continue

                        lengths = mx - mn
                        volume = lengths[0] * lengths[1] * lengths[2]
                        w = volume / (qq ** 3)
                        step = lengths / qq

                        value = 0.0

                        for a in range(qq):
                            for b in range(qq):
                                for c in range(qq):
                                    q = np.array([
                                        mn[0] + step[0] * (a + 0.5),
                                        mn[1] + step[1] * (b + 0.5),
                                        mn[2] + step[2] * (c + 0.5),
                                    ])

                                    value += (
                                        node_j.basis_laplasian(q)
                                        * node_i.basis_func(q)
                                        * w
                                    )

                        if abs(value) > 1e-14:
                            L[i, j] = value

        return L.tocsr()
    
    def evaluate_chi(self, point, depth, x):
        chi = 0.0

        root_min = self.root.center - self.root.half_size
        root_size = self.root.size
        R = 2 ** depth

        g = ((point - root_min) / root_size) * R
        ix, iy, iz = np.floor(g).astype(int)

        mp = self.depth_map[depth]

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    node = mp.get((ix + dx, iy + dy, iz + dz))
                    if node is None:
                        continue

                    f = node.basis_func(point)
                    if f != 0:
                        chi += x[node.node_id] * f

        return chi
    
    def get_dense_field(self, points, depth, x, q):
        cnt = len(points)
        chi = 0.0
        for point in tqdm(points, desc='Processing'):
            chi += self.evaluate_chi(point, depth, x)
        chi /= cnt

        field = np.zeros((q, q, q), dtype=np.float64)

        root_min = self.root.center - self.root.half_size
        step = self.root.size / (q - 1)

        for i in range(q):
            for j in range(q):
                for k in range(q):
                    p = np.array([
                        root_min[0] + i * step,
                        root_min[1] + j * step,
                        root_min[2] + k * step,
                    ], dtype=np.float64)

                    field[i, j, k] = self.evaluate_chi(p, depth, x)

        return field, chi






    

        
if __name__ == '__main__':
    pass