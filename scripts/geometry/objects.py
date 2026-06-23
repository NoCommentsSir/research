import open3d as o3d
import numpy as np
from scipy.sparse import coo_matrix
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

def order_ring_from_pairs(pairs):
        dct = {}

        for a, b in pairs:

            if a == b:
                continue

            if a not in dct:
                dct[a] = []
            if b not in dct:
                dct[b] = []

            if b not in dct[a]:
                dct[a].append(b)
            if a not in dct[b]:
                dct[b].append(a)

        if len(dct) == 0:
            return []

        for v in dct:
            if len(dct[v]) != 2:
                return []

        start = next(iter(dct.keys()))

        ordered = [start]
        prev = None
        cur = start

        while True:
            a, b = dct[cur]

            if a != prev:
                nxt = a
            else:
                nxt = b

            if nxt == start:
                break

            if nxt in ordered:
                return []

            ordered.append(nxt)

            prev = cur
            cur = nxt

            if len(ordered) > len(dct):
                return []

        if len(ordered) != len(dct):
            return []

        return ordered

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
        
def gradient_scalar_all(field, h):
    grad = np.zeros(field.shape + (3,), dtype=np.float64)

    grad[1:-1, 1:-1, 1:-1, 0] = (
        field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]
    ) / (2 * h)

    grad[1:-1, 1:-1, 1:-1, 1] = (
        field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]
    ) / (2 * h)

    grad[1:-1, 1:-1, 1:-1, 2] = (
        field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]
    ) / (2 * h)

    return grad


def divergence_vector_all(field, h):
    div = np.zeros(field.shape[:3], dtype=np.float64)

    div[1:-1, 1:-1, 1:-1] = (
        (field[2:, 1:-1, 1:-1, 0] - field[:-2, 1:-1, 1:-1, 0]) / (2 * h)
        + (field[1:-1, 2:, 1:-1, 1] - field[1:-1, :-2, 1:-1, 1]) / (2 * h)
        + (field[1:-1, 1:-1, 2:, 2] - field[1:-1, 1:-1, :-2, 2]) / (2 * h)
    )

    return div
        
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
        self.grad = np.zeros((size, size, size, 3), dtype=np.float64)
        self.neg_grad = np.zeros((size, size, size, 3), dtype=np.float64)
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
        
    def gradient_all(self, field):
        if field.ndim == 3:
            self.grad = gradient_scalar_all(field, self.h)
            return self.grad

        raise ValueError("gradient_all сейчас ожидает scalar field shape=(N,N,N)")

    def divergence_all(self, field):
        if field.ndim != 4 or field.shape[-1] != 3:
            raise ValueError("divergence_all ожидает vector field shape=(N,N,N,3)")

        self.div = divergence_vector_all(field, self.h)
        return self.div
    
    def interpolate_vector_field(self, field, p):
        p = np.asarray(p, dtype=np.float64)

        gx = ((p[:, 0] - self.min_bound) / self.length) * (self.size - 1)
        gy = ((p[:, 1] - self.min_bound) / self.length) * (self.size - 1)
        gz = ((p[:, 2] - self.min_bound) / self.length) * (self.size - 1)

        gx = np.clip(gx, 0, self.size - 2)
        gy = np.clip(gy, 0, self.size - 2)
        gz = np.clip(gz, 0, self.size - 2)

        i0 = np.floor(gx).astype(int)
        j0 = np.floor(gy).astype(int)
        k0 = np.floor(gz).astype(int)

        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        tx = gx - i0
        ty = gy - j0
        tz = gz - k0

        c000 = field[i0, j0, k0]
        c001 = field[i0, j0, k1]
        c010 = field[i0, j1, k0]
        c011 = field[i0, j1, k1]
        c100 = field[i1, j0, k0]
        c101 = field[i1, j0, k1]
        c110 = field[i1, j1, k0]
        c111 = field[i1, j1, k1]

        w000 = ((1 - tx) * (1 - ty) * (1 - tz))[:, None]
        w001 = ((1 - tx) * (1 - ty) * tz)[:, None]
        w010 = ((1 - tx) * ty * (1 - tz))[:, None]
        w011 = ((1 - tx) * ty * tz)[:, None]
        w100 = (tx * (1 - ty) * (1 - tz))[:, None]
        w101 = (tx * (1 - ty) * tz)[:, None]
        w110 = (tx * ty * (1 - tz))[:, None]
        w111 = (tx * ty * tz)[:, None]

        return (
            w000 * c000
            + w001 * c001
            + w010 * c010
            + w011 * c011
            + w100 * c100
            + w101 * c101
            + w110 * c110
            + w111 * c111
        )
    
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
    
class MeshGeometry:

    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.V = V
        self.F = F

    def compute_vertex_normals(self):
        N = self.V.shape[0]
        normals = np.zeros((N, 3), dtype=np.float64)

        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        for face_id, tri in enumerate(self.F):
            i, j, k = tri

            normals[i] += face_normals[face_id]
            normals[j] += face_normals[face_id]
            normals[k] += face_normals[face_id]

        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (nrm + 1e-12)

        return normals

    def build_vertex_neighbors(self):
        N = self.V.shape[0]
        arr = [set() for _ in range(N)]

        for tri in self.F:
            i, j, k = tri

            arr[i].add(j)
            arr[i].add(k)

            arr[j].add(i)
            arr[j].add(k)

            arr[k].add(i)
            arr[k].add(j)

        return [sorted(list(s)) for s in arr]

    def build_vertex_ordered_neighbors(self):
        N = self.V.shape[0]
        arr = [list() for _ in range(N)]

        for tri in self.F:
            i, j, k = tri

            arr[i].append((j,k))
            arr[j].append((i,k))
            arr[k].append((i,j))

        ordered_neighbors = []

        for i in range(N):
            ring = order_ring_from_pairs(arr[i])
            ordered_neighbors.append(ring)

        return ordered_neighbors
    
    def compute_tangential_velocities(self, normals, ordered_neighbors):
        N = self.V.shape[0]
        vt = np.zeros((N, 3), dtype=np.float64)

        for i in range(N):

            arr = ordered_neighbors[i]

            if len(arr) < 3:
                continue

            p = self.V[i]
            n = normals[i]

            r = np.zeros(3, dtype=np.float64)
            m = len(arr)

            for t in range(m):
                e1 = arr[t]
                e2 = arr[(t+1)%m]

                edge1 = self.V[e1] - p
                edge2 = self.V[e2] - p
                nrm1 = np.linalg.norm(edge1)
                nrm2 = np.linalg.norm(edge2)

                if nrm1 < 1e-12 or nrm2 < 1e-12:
                    continue
                edge1_n = edge1 / nrm1
                edge2_n = edge2 / nrm2


                r += (1 + np.dot(edge1_n, edge2_n)) * (edge1 + edge2)
            
            r /= m
            vt[i] = r - np.dot(r,n) * n
        return vt
    
    def triangle_quality(self, i, j, k):
        v0 = self.V[i]
        v1 = self.V[j]
        v2 = self.V[k]

        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)

        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        denom = a * a + b * b + c * c

        if denom < 1e-12:
            return 0.0

        return 4.0 * np.sqrt(3.0) * area / denom


    def collect_edges(self):
        edges = set()

        for tri in self.F:
            i, j, k = map(int, tri)

            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))

        return list(edges)


    def edge_lengths(self):
        edges = self.collect_edges()
        lengths = []

        for i, j in edges:
            lengths.append(np.linalg.norm(self.V[i] - self.V[j]))

        return np.array(lengths, dtype=np.float64)


    def remove_degenerate_faces(self, area_eps=1e-12, edge_eps=1e-8):
        new_faces = []

        for tri in self.F:
            i, j, k = map(int, tri)

            if i == j or j == k or k == i:
                continue

            v0 = self.V[i]
            v1 = self.V[j]
            v2 = self.V[k]

            if not np.all(np.isfinite(v0)):
                continue
            if not np.all(np.isfinite(v1)):
                continue
            if not np.all(np.isfinite(v2)):
                continue

            e01 = np.linalg.norm(v0 - v1)
            e12 = np.linalg.norm(v1 - v2)
            e20 = np.linalg.norm(v2 - v0)

            if e01 < edge_eps or e12 < edge_eps or e20 < edge_eps:
                continue

            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            if area < area_eps:
                continue

            new_faces.append([i, j, k])

        self.F = np.array(new_faces, dtype=np.int64)

    def remove_unreferenced_vertices(self):
        if self.F.shape[0] == 0:
            return

        used = np.unique(self.F.reshape(-1))

        old_to_new = -np.ones(self.V.shape[0], dtype=np.int64)
        old_to_new[used] = np.arange(len(used))

        self.V = self.V[used]
        self.F = old_to_new[self.F]


    def cleanup_mesh(self):
        self.remove_degenerate_faces()
        self.remove_unreferenced_vertices()

    def remove_degenerate_faces(self, area_eps=1e-12, edge_eps=1e-8):
        new_faces = []

        for tri in self.F:
            i, j, k = map(int, tri)

            if i == j or j == k or k == i:
                continue

            v0 = self.V[i]
            v1 = self.V[j]
            v2 = self.V[k]

            if not np.all(np.isfinite(v0)):
                continue
            if not np.all(np.isfinite(v1)):
                continue
            if not np.all(np.isfinite(v2)):
                continue

            e01 = np.linalg.norm(v0 - v1)
            e12 = np.linalg.norm(v1 - v2)
            e20 = np.linalg.norm(v2 - v0)

            if e01 < edge_eps or e12 < edge_eps or e20 < edge_eps:
                continue

            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            if area < area_eps:
                continue

            new_faces.append([i, j, k])

        self.F = np.array(new_faces, dtype=np.int64)

    def compute_mean_curvature(self):
        areas = self.compute_vertex_areas()
        areas = np.maximum(areas, 1e-12)

        L = self.cotangent_laplacian()

        LV = L @ self.V

        H = 0.5 * np.linalg.norm(LV / areas[:, None], axis=1)

        return H


    def compute_vertex_mean_edge_lengths(self):
        neighbors = self.build_vertex_neighbors()
        N = self.V.shape[0]

        mean_lengths = np.zeros(N, dtype=np.float64)

        for i in range(N):
            neigh = neighbors[i]

            if len(neigh) == 0:
                continue

            lengths = []

            for j in neigh:
                lengths.append(np.linalg.norm(self.V[i] - self.V[j]))

            mean_lengths[i] = np.mean(lengths)

        return mean_lengths


    def detect_feature_vertices(self, gamma_crit=0.4, max_feature_valence=6):
        H = self.compute_mean_curvature()
        mean_l = self.compute_vertex_mean_edge_lengths()
        neighbors = self.build_vertex_neighbors()

        gamma = 2.0 * np.abs(H) * mean_l

        feature_mask = gamma > gamma_crit

        for i in range(self.V.shape[0]):
            if len(neighbors[i]) > max_feature_valence:
                feature_mask[i] = False

        return feature_mask
    
    def collapse_short_edges(self, min_len, feature_mask=None):
        edges = self.collect_edges()

        if len(edges) == 0:
            return 0

        parent = np.arange(self.V.shape[0], dtype=np.int64)
        used = np.zeros(self.V.shape[0], dtype=bool)

        edges_with_len = []

        for i, j in edges:
            length = np.linalg.norm(self.V[i] - self.V[j])
            edges_with_len.append((length, i, j))

        edges_with_len.sort(key=lambda x: x[0])

        collapsed = 0

        for length, i, j in edges_with_len:
            if length >= min_len:
                break

            if used[i] or used[j]:
                continue

            if feature_mask is not None:
                if feature_mask[i] or feature_mask[j]:
                    continue

            mid = 0.5 * (self.V[i] + self.V[j])

            self.V[i] = mid
            parent[j] = i

            used[i] = True
            used[j] = True

            collapsed += 1

        for idx in range(len(parent)):
            while parent[idx] != parent[parent[idx]]:
                parent[idx] = parent[parent[idx]]

        self.F = parent[self.F]

        self.cleanup_mesh()

        return collapsed
    
    def split_long_edges(self, max_len, feature_mask=None):
        V_list = self.V.tolist()
        new_faces = []

        edge_midpoint = {}
        new_vertex_ids = []

        def can_split(a, b):
            if feature_mask is None:
                return True

            # не режем ребро, если оба конца являются feature
            if feature_mask[a] and feature_mask[b]:
                return False

            return True

        def get_midpoint(i, j):
            edge = tuple(sorted((int(i), int(j))))

            if edge in edge_midpoint:
                return edge_midpoint[edge]

            mid = 0.5 * (self.V[edge[0]] + self.V[edge[1]])

            idx = len(V_list)
            V_list.append(mid.tolist())

            edge_midpoint[edge] = idx
            new_vertex_ids.append(idx)

            return idx

        for tri in self.F:
            i, j, k = map(int, tri)

            lij = np.linalg.norm(self.V[i] - self.V[j])
            ljk = np.linalg.norm(self.V[j] - self.V[k])
            lki = np.linalg.norm(self.V[k] - self.V[i])

            split_ij = lij > max_len and can_split(i, j)
            split_jk = ljk > max_len and can_split(j, k)
            split_ki = lki > max_len and can_split(k, i)

            count = int(split_ij) + int(split_jk) + int(split_ki)

            if count == 0:
                new_faces.append([i, j, k])

            elif count == 1:
                if split_ij:
                    m = get_midpoint(i, j)
                    new_faces.append([i, m, k])
                    new_faces.append([m, j, k])

                elif split_jk:
                    m = get_midpoint(j, k)
                    new_faces.append([j, m, i])
                    new_faces.append([m, k, i])

                else:
                    m = get_midpoint(k, i)
                    new_faces.append([k, m, j])
                    new_faces.append([m, i, j])

            else:
                mij = get_midpoint(i, j)
                mjk = get_midpoint(j, k)
                mki = get_midpoint(k, i)

                new_faces.append([i, mij, mki])
                new_faces.append([mij, j, mjk])
                new_faces.append([mki, mjk, k])
                new_faces.append([mij, mjk, mki])

        self.V = np.array(V_list, dtype=np.float64)
        self.F = np.array(new_faces, dtype=np.int64)

        return new_vertex_ids
    
    def project_vertices_to_distance_field(
        self,
        VC,
        iso_level=0.0,
        alpha=0.3,
        vertex_ids=None,
        max_step=None,
    ):
        if vertex_ids is None:
            vertex_ids = range(self.V.shape[0])

        lo = VC.min_bound + 2.0 * VC.h
        hi = VC.max_bound - 2.0 * VC.h

        for idx in vertex_ids:
            p = self.V[idx]

            if not np.all(np.isfinite(p)):
                continue

            p_safe = np.clip(p, lo, hi)

            d = VC.interpolate_scalar_field(VC.D, p_safe) - iso_level
            neg_grad = VC.interpolate_vector_field(VC.neg_grad, p_safe[None, :])[0]

            step = alpha * d * neg_grad

            if max_step is not None:
                step_norm = np.linalg.norm(step)

                if step_norm > max_step:
                    step = step / (step_norm + 1e-12) * max_step

            p_new = p + step
            p_new = np.clip(p_new, lo, hi)

            if np.all(np.isfinite(p_new)):
                self.V[idx] = p_new

    def flip_edges(self, feature_mask=None, min_improvement=1e-4, max_flips=1000):
        edge_to_faces = {}

        for face_id, tri in enumerate(self.F):
            i, j, k = map(int, tri)

            for a, b in [(i, j), (j, k), (k, i)]:
                edge = tuple(sorted((a, b)))

                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []

                edge_to_faces[edge].append(face_id)

        removed = np.zeros(self.F.shape[0], dtype=bool)
        flipped = 0

        existing_edges = set(self.collect_edges())

        def orient_face(a, b, c, ref_n):
            n = np.cross(self.V[b] - self.V[a], self.V[c] - self.V[a])

            if np.dot(n, ref_n) < 0:
                return [a, c, b]

            return [a, b, c]

        for edge, faces in edge_to_faces.items():
            if flipped >= max_flips:
                break

            if len(faces) != 2:
                continue

            f0, f1 = faces

            if removed[f0] or removed[f1]:
                continue

            i, j = edge

            if feature_mask is not None:
                if feature_mask[i] or feature_mask[j]:
                    continue

            tri0 = list(map(int, self.F[f0]))
            tri1 = list(map(int, self.F[f1]))

            opp0 = [v for v in tri0 if v not in edge]
            opp1 = [v for v in tri1 if v not in edge]

            if len(opp0) != 1 or len(opp1) != 1:
                continue

            k = opp0[0]
            l = opp1[0]

            if k == l:
                continue

            new_edge = tuple(sorted((k, l)))

            if new_edge in existing_edges:
                continue

            old_q1 = self.triangle_quality(*tri0)
            old_q2 = self.triangle_quality(*tri1)
            old_min_q = min(old_q1, old_q2)

            new_a = [k, l, i]
            new_b = [l, k, j]

            new_q1 = self.triangle_quality(*new_a)
            new_q2 = self.triangle_quality(*new_b)
            new_min_q = min(new_q1, new_q2)

            if new_min_q <= old_min_q + min_improvement:
                continue

            n0 = np.cross(
                self.V[tri0[1]] - self.V[tri0[0]],
                self.V[tri0[2]] - self.V[tri0[0]],
            )
            n1 = np.cross(
                self.V[tri1[1]] - self.V[tri1[0]],
                self.V[tri1[2]] - self.V[tri1[0]],
            )
            ref_n = n0 + n1

            if np.linalg.norm(ref_n) < 1e-12:
                continue

            self.F[f0] = orient_face(k, l, i, ref_n)
            self.F[f1] = orient_face(l, k, j, ref_n)

            existing_edges.discard(edge)
            existing_edges.add(new_edge)

            removed[f0] = True
            removed[f1] = True

            flipped += 1

        return flipped
    
    def adaptive_remesh(
        self,
        VC,
        iso_level=0.0,
        min_len=None,
        max_len=None,
        gamma_crit=0.4,
    ):
        if min_len is None:
            min_len = 0.25 * VC.h

        if max_len is None:
            max_len = 4.0 * VC.h

        self.cleanup_mesh()

        feature_mask = self.detect_feature_vertices(gamma_crit=gamma_crit)

        collapsed = self.collapse_short_edges(
            min_len=min_len,
            feature_mask=feature_mask,
        )

        feature_mask = self.detect_feature_vertices(gamma_crit=gamma_crit)

        new_ids = self.split_long_edges(
            max_len=max_len,
            feature_mask=feature_mask,
        )

        self.project_vertices_to_distance_field(
            VC,
            iso_level=iso_level,
            alpha=0.3,
            vertex_ids=new_ids,
            max_step=0.5 * VC.h,
        )

        self.cleanup_mesh()

        feature_mask = self.detect_feature_vertices(gamma_crit=gamma_crit)

        flipped = self.flip_edges(
            feature_mask=feature_mask,
            min_improvement=1e-4,
            max_flips=1000,
        )

        self.cleanup_mesh()

        return collapsed, len(new_ids), flipped

    def cotangent_laplacian(self):
        N = self.V.shape[0]

        weights = {}

        def cotangent(a, b):
            cross_norm = np.linalg.norm(np.cross(a, b))
            if cross_norm < 1e-12:
                return 0.0

            return np.dot(a, b) / cross_norm

        def add_weight(i, j, w):
            if i > j:
                i, j = j, i

            weights[(i, j)] = weights.get((i, j), 0.0) + w

        for tri in self.F:
            i, j, k = tri

            vi = self.V[i]
            vj = self.V[j]
            vk = self.V[k]

            # Угол при k лежит напротив ребра (i, j)
            ctg_k = cotangent(vi - vk, vj - vk)

            # Угол при j лежит напротив ребра (i, k)
            ctg_j = cotangent(vi - vj, vk - vj)

            # Угол при i лежит напротив ребра (j, k)
            ctg_i = cotangent(vj - vi, vk - vi)

            add_weight(i, j, 0.5 * ctg_k)
            add_weight(i, k, 0.5 * ctg_j)
            add_weight(j, k, 0.5 * ctg_i)

        rows = []
        cols = []
        data = []

        diagonal = np.zeros(N, dtype=np.float64)

        for (i, j), w in weights.items():
            if abs(w) < 1e-14:
                continue

            diagonal[i] += w
            diagonal[j] += w

            rows.append(i)
            cols.append(j)
            data.append(-w)

            rows.append(j)
            cols.append(i)
            data.append(-w)

        for i in range(N):
            rows.append(i)
            cols.append(i)
            data.append(diagonal[i])

        L = coo_matrix(
            (data, (rows, cols)),
            shape=(N, N)
        ).tocsr()

        return L

    def compute_vertex_areas(self):
        N = self.V.shape[0]
        areas = np.zeros(N, dtype=np.float64)

        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]

        face_cross = np.cross(v1 - v0, v2 - v0)

        # Площадь каждого треугольника
        face_areas = 0.5 * np.linalg.norm(face_cross, axis=1)

        for face_id, tri in enumerate(self.F):
            i, j, k = tri

            area_part = face_areas[face_id] / 3.0

            areas[i] += area_part
            areas[j] += area_part
            areas[k] += area_part

        return areas
    
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
    
    def _B1_arr(self, t):
        a = np.abs(t)
        res = np.zeros_like(t, dtype=np.float64)

        m1 = a < 1
        m2 = (a >= 1) & (a < 2)

        res[m1] = (4 - 6 * t[m1]**2 + 3 * a[m1]**3) / 6
        res[m2] = (2 - a[m2])**3 / 6
        return res

    def _ddB1_arr(self, t):
        a = np.abs(t)
        res = np.zeros_like(t, dtype=np.float64)

        m1 = a < 1
        m2 = (a >= 1) & (a < 2)

        res[m1] = -2 + 3 * a[m1]
        res[m2] = 2 - a[m2]
        return res

    def basis_func_batch(self, points):
        local = (points - self.center) / self.half_size
        bx = self._B1_arr(local[:, 0])
        by = self._B1_arr(local[:, 1])
        bz = self._B1_arr(local[:, 2])
        return bx * by * bz

    def basis_laplasian_batch(self, points):
        local = (points - self.center) / self.half_size

        bx = self._B1_arr(local[:, 0])
        by = self._B1_arr(local[:, 1])
        bz = self._B1_arr(local[:, 2])

        ddx = self._ddB1_arr(local[:, 0])
        ddy = self._ddB1_arr(local[:, 1])
        ddz = self._ddB1_arr(local[:, 2])

        h2 = self.half_size ** 2
        return (ddx * by * bz + bx * ddy * bz + bx * by * ddz) / h2

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
        
        if len(node.indeces) <= self.min_points:
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
    
    def create_node_at_depth(self, idx, depth:int) -> OctreeNode:
        idx = np.array(idx, dtype=np.float64)
        size = self.root.size / 2 ** depth
        center = - self.root.half_size + (idx + 0.5) * size
        res =  OctreeNode(
            center=center,
            size = size,
            depth = depth
        )
        res.grid_index = idx
        return res

    
    def fill_depth_neighborhood(self, depth):
        nodes = set(self.depth_map[depth].keys())
        to_add = set()
        offsets = range(-4, 5)
        R = 2 ** depth

        for node in nodes:
                for dx in offsets:
                    for dy in offsets:
                        for dz in offsets:
                            idx = (node[0] + dx, node[1] + dy, node[2] + dz)

                            if not (
                                0 <= idx[0] < R and
                                0 <= idx[1] < R and
                                0 <= idx[2] < R
                            ):
                                continue

                            to_add.add(idx)

        for idx in to_add:                    
            if idx not in self.depth_map[depth]:
                self.depth_map[depth][idx] = self.create_node_at_depth(idx, depth)

        for i, node in enumerate(self.depth_map[depth].values()):
            node.node_id = i

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
        for node in tqdm(self.depth_map[depth].values(), "v calc"):
            min_bound, max_bound = support_bounds(node)

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

        rows, cols, data = [], [], []
        mp = self.depth_map[depth]
        n = len(nodes)

        offsets = range(-4, 5)

        for node_i in tqdm(nodes, desc="Assembling L"):
            i = node_i.node_id
            ix, iy, iz = node_i.grid_index

            min_i, max_i = support_bounds(node_i)

            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        node_j = mp.get((ix + dx, iy + dy, iz + dz))
                        if node_j is None:
                            continue

                        min_j, max_j = support_bounds(node_j)

                        mn, mx = intersect_bounds(min_i, max_i, min_j, max_j)
                        if mn is None:
                            continue

                        lengths = mx - mn
                        volume = np.prod(lengths)
                        step = lengths / qq
                        weight = volume / (qq ** 3)

                        xs = mn[0] + step[0] * (np.arange(qq) + 0.5)
                        ys = mn[1] + step[1] * (np.arange(qq) + 0.5)
                        zs = mn[2] + step[2] * (np.arange(qq) + 0.5)

                        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
                        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

                        fi = node_i.basis_func_batch(pts)
                        lap_j = node_j.basis_laplasian_batch(pts)

                        value = np.sum(lap_j * fi) * weight

                        if abs(value) > 1e-14:
                            rows.append(i)
                            cols.append(node_j.node_id)
                            data.append(value)

        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    
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
        arr = []
        for point in tqdm(points, desc='Processing'):
            arr.append(self.evaluate_chi(point, depth, x))
        chi = np.median(arr)

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