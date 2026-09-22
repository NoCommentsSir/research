import torch
import pathlib
from torch.utils.data import Dataset

from .model import Model
from ..geometry.neighbors import get_edge_features, get_points_neiborhood

class ModelData:

    def __init__(self, root_dir:str, split:str, points_cnt:int, k_neighbors:int):
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.points_cnt = points_cnt
        self.k_neighbors = k_neighbors
        self.files = sorted(
            self.root_dir.glob(f"*/{split}/*.off")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]

        model = Model(file)
        model.get_point_cloud(self.points_cnt)
        points = model.point_cloud
        normals = model.normals
        edges = get_points_neiborhood(model.point_cloud, self.k_neighbors)
        edge_features = get_edge_features(model.point_cloud, edges)

        return {
            "points": points,
            "normals": normals,
            "edges": edges,
            "edge_features": edge_features,
            "path": str(file),
        }


if __name__ == '__main__':
    D = DataLoader('data\\ModelNet10', 'train', 2000, 16)
    print(len(D))