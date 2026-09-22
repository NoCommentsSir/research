import torch

from ..dataset.model import Model

def get_points_neiborhood(points, k:int):
    D = torch.cdist(points, points)
    N = torch.topk(D, k+1, largest=False).indices[:, 1:]

    n = points.shape[0]

    source = torch.arange(n).repeat_interleave(k)
    target = N.reshape(-1)

    edge_index = torch.stack([source, target], dim=0)

    return edge_index

def get_edge_features(points, edges):
    src = edges[0]
    tgt = edges[1]

    dlt = points[tgt] - points[src]
    dot = torch.sum(points[tgt] * points[src], dim=1, keepdim=True)
    dist = torch.norm(points[tgt] - points[src], dim=1, keepdim=True)
    nrm_tgt = torch.norm(points[tgt], dim=1, keepdim=True)
    nrm_src = torch.norm(points[src], dim=1, keepdim=True)

    cos_angle = dot / (nrm_src * nrm_tgt + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angles = torch.acos(cos_angle)

    return torch.cat([dlt, dist], dim=1)

if __name__ == '__main__':
    m = Model('data/ModelNet10/bed/test/bed_0516.off')
    m.get_point_cloud(2000)
    arr = get_points_neiborhood(m.point_cloud, 16)
    res = get_edge_features(m.point_cloud, arr)
    print(res.shape)