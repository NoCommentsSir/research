import numpy as np
import open3d as o3d

def triangle_area(A, B, C):
    return np.linalg.norm(np.cross(B - A, C - A)) / 2

class Model:
    vertices = []
    triangles = []
    edges = set()
    areas = []
    v = 0
    t = 0

    def __init__(self, file):
        self.vertices = []
        self.triangles = []
        self.edges = set()

        with open(file, 'r') as f:
            f.readline()
            self.v, self.t, _ = list(map(int, f.readline().split()))

            for i in range(self.v):
                self.vertices.append(list(map(float, f.readline().split())))

            for i in range(self.t):
                self.triangles.append(list(map(int, f.readline().split()))[1:])
                self.areas.append(triangle_area(
                    np.array(self.vertices[self.triangles[-1][0]]),
                    np.array(self.vertices[self.triangles[-1][1]]),
                    np.array(self.vertices[self.triangles[-1][2]])
                ))

            for i in self.triangles:
                self.edges.add((i[0], i[1]))
                self.edges.add((i[1], i[2]))
                self.edges.add((i[2], i[0]))

    def normalize(self):
        v_arr = np.array(self.vertices)
        center = np.mean(v_arr, axis=0)
        v_arr -= center
        max_distance = np.max(np.linalg.norm(v_arr, axis=1))    
        v_arr /= max_distance
        self.vertices = v_arr.tolist()

    def save_obj(self, path):
        with open(path, "w") as f:

            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            for tri in self.triangles:
                a, b, c = tri
                f.write(f"f {a+1} {b+1} {c+1}\n")

    def visualize_meshes(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])