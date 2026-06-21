import random
import numpy as np

from scripts.geometry.objects import Point, PointCloud
from scripts.model.model import Model

def sample_points(A, B, C, triangle_id):
    u = random.random()
    v = random.random()

    if u + v > 1:
        u = 1 - u
        v = 1 - v

    p = Point(0, 0, 0, triangle_id)

    p.x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    p.y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    p.z = A[2] + u * (B[2] - A[2]) + v * (C[2] - A[2])

    return p

def generate_point_cloud(M: Model, n: int):
    vertices = np.array(M.vertices, dtype=float)
    triangles = np.array(M.triangles, dtype=int)
    probs = np.array(M.areas) / (sum(M.areas) + 1e-12)
    triangles_indexes = np.random.choice(len(M.triangles), size=(n,), p=probs)
    triangles = triangles[triangles_indexes]

    points_chunks = []
    normals_chunks = []

    for tri in triangles:
        A = vertices[tri[0]]
        B = vertices[tri[1]]
        C = vertices[tri[2]]
    
        AB = B - A
        AC = C - A

        normal = np.cross(AB, AC)
        length = np.linalg.norm(normal)

        if length < 1e-12:
            continue

        normal = normal / length

        u = random.random()
        v = random.random()

        if u + v > 1:
            u = 1 - u
            v = 1 - v

        points = A + u * AB + v * AC
        normals = normal[None, :]

        points_chunks.append(points)
        normals_chunks.append(normals)

    points_arr = np.vstack(points_chunks)
    normals_arr = np.vstack(normals_chunks)

    return PointCloud(points_arr, normals_arr)