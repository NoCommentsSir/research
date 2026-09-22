import numpy as np
import open3d as o3d
import torch
import random
from collections import deque

from ..utils.tools import read_file

def triangle_area(A, B, C):
    return np.linalg.norm(np.cross(B - A, C - A)) / 2

def signed_volume(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:

    A = vertices[triangles[:, 0]]
    B = vertices[triangles[:, 1]]
    C = vertices[triangles[:, 2]]

    volumes = torch.sum(
        A * torch.linalg.cross(B, C, dim=1),
        dim=1
    ) / 6.0

    return volumes.sum()

class Model:
    def __init__(self, file:str):
        self.vertices = []
        self.triangles = []
        self.edges = {}
        self.areas = []
        self.point_cloud = torch.tensor([], dtype=torch.float32)
        self.normals = torch.tensor([], dtype=torch.float32)

        self.v = 0
        self.t = 0

        model_gen = read_file(file)
        next(model_gen)
        self.v, self.t, _ = list(map(int, next(model_gen).split()))

        for _ in range(self.v):
            self.vertices.append(list(map(float, next(model_gen).split())))

        for _ in range(self.t):
            tri = list(map(int, next(model_gen).split()))[1:]
            self.triangles.append(tri)

            self.areas.append(triangle_area(
                np.array(self.vertices[tri[0]]),
                np.array(self.vertices[tri[1]]),
                np.array(self.vertices[tri[2]])
            ))

        for tri_idx, tri in enumerate(self.triangles):
            a, b, c = tri

            for u, v in [(a, b), (b, c), (c, a)]:
                key = (min(u, v), max(u, v))

                sign = 1 if (u, v) == key else -1

                if key not in self.edges:
                    self.edges[key] = []

                self.edges[key].append((tri_idx, sign))


        self.vertices = torch.tensor(self.vertices, dtype = torch.float32)
        self.triangles = torch.tensor(self.triangles, dtype = torch.long)
        self.areas = torch.tensor(self.areas, dtype = torch.float32)
        self.triangles_graph = {}
        
        for key in self.edges:
            if len(self.edges[key]) == 2:
                tri1, sign1 = self.edges[key][0]
                tri2, sign2 = self.edges[key][1]

                if tri1 not in self.triangles_graph:
                    self.triangles_graph[tri1] = []

                if tri2 not in self.triangles_graph:
                    self.triangles_graph[tri2] = []

                need_flip = sign1 == sign2

                self.triangles_graph[tri1].append([tri2, need_flip])
                self.triangles_graph[tri2].append([tri1, need_flip])

    def normalize(self):
        v_arr = self.vertices
        center = torch.mean(v_arr, dim=0)
        v_arr -= center
        
        max_distance = torch.max(torch.norm(v_arr, dim=1))
        if max_distance > 1e-12:
            v_arr /= max_distance

        self.vertices = v_arr

    def sample_points(self, n: int):
        probs = self.areas / self.areas.sum()
        triangles_indexes = torch.multinomial(probs, num_samples=n, replacement=True)
        triangles = self.triangles[triangles_indexes]

        points = []
        normals = []

        for tri in triangles:
            A = self.vertices[tri[0]]
            B = self.vertices[tri[1]]
            C = self.vertices[tri[2]]
        
            AB = B - A
            AC = C - A

            normal = torch.linalg.cross(AB, AC)
            length = torch.norm(normal)

            if length < 1e-12:
                continue

            normal = normal / length

            u = random.random()
            v = random.random()

            if u + v > 1:
                u = 1 - u
                v = 1 - v

            points.append(A + u * AB + v * AC)
            normals.append(normal)

        self.point_cloud = torch.stack(points)
        self.normals = torch.stack(normals)

    def reverse_triangles(self):
        flags = [None] * self.t

        for i in range(self.t):

            if flags[i] is not None:
                continue

            flags[i] = False
            q = deque([i])

            while q:
                current = q.popleft()

                for tri, flg in self.triangles_graph.get(current, []):
                    to_flip = flags[current] ^ flg
                    if flags[tri] is None:
                        flags[tri] = to_flip
                        q.append(tri)

        for i, need_flip in enumerate(flags):
            if need_flip:
                tmp = self.triangles[i, 1].clone()
                self.triangles[i, 1] = self.triangles[i, 2]
                self.triangles[i, 2] = tmp      

    def orient_outward(self):
        volume = signed_volume(self.vertices, self.triangles)
        if volume < 0:
            triangles = self.triangles.clone()
            triangles[:, [1, 2]] = triangles[:, [2, 1]]

            self.triangles = triangles

    def get_point_cloud(self, n:int):
        self.normalize()
        self.reverse_triangles()
        self.orient_outward()
        self.sample_points(n)

    def visualize_meshes(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
        

if __name__ == '__main__':
    m = Model('data/ModelNet10/bed/test/bed_0516.off')
    m.get_point_cloud(2000)