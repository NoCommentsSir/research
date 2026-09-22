import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset.dataloader import ModelData
from .geometry.neighbors import *

def single_graph_collate(batch):

   points_list = []
   normals_list = []
   edges_list = []
   edge_features_list = []
   batch_list = []

   cnt = 0

   for idx, item in enumerate(batch):
      points = item["points"]
      normals = item["normals"]
      edges = item["edges"]
      edge_features = item["edge_features"]
      edges = edges + cnt

      points_list.append(points)
      normals_list.append(normals)
      edges_list.append(edges)
      edge_features_list.append(edge_features)

      batch_list.append(
         torch.full(
            (points.shape[0],),
            idx,
            dtype=torch.long
         )
      )
      
      cnt += points.shape[0]

   return {
        "points": torch.cat(points_list, dim=0),
        "normals": torch.cat(normals_list, dim=0),
        "edges": torch.cat(edges_list, dim=1),
        "edge_features": torch.cat(edge_features_list, dim=0),
        "batch": torch.cat(batch_list, dim=0),
    }

class EdgeConv(nn.Module):
     
   def __init__(self, in_features, out_features):
      super().__init__()
      self.layer = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_features),
        )

   def forward(self, node_features:torch.Tensor, edges:torch.Tensor, edge_features:torch.Tensor):
      v_src = edges[0]
      v_tgt = edges[1]

      h_src = node_features[v_src]
      h_tgt = node_features[v_tgt]

      dlt = h_tgt - h_src

      z = torch.cat([h_src, dlt, edge_features], dim=1)
      messages = self.layer(z)

      new_features = torch.zeros(node_features.shape[0], messages.shape[1], dtype=messages.dtype, device=messages.device)
      idx = v_src[:, None].expand_as(messages)
      new_features.scatter_reduce_(0, idx, messages, reduce="amax", include_self=False)
      return new_features

class NormalPredictor(nn.Module):
     
   def __init__(self):
      super(NormalPredictor, self).__init__()
      self.e1 = EdgeConv(10, 128)
      self.lrl1 = nn.LeakyReLU()
      self.e2 = EdgeConv(260, 128)
      self.lrl2 = nn.LeakyReLU()
      self.e3 = EdgeConv(260, 64)
      self.lrl3 = nn.LeakyReLU()
      self.l1 = nn.Linear(195, 256)
      self.lrl4 = nn.LeakyReLU()
      self.l2 = nn.Linear(256, 64)
      self.lrl5 = nn.LeakyReLU()
      self.l3 = nn.Linear(64, 3)

   def forward(self, points, edges, features, batch):
      batch_size = int(batch.max().item()) + 1

      x1 = self.e1(points, edges, features)
      y1 = self.lrl1(x1)
      x2 = self.e2(y1, edges, features)
      y2 = self.lrl2(x2)
      x3 = self.e3(y2, edges, features)
      y3 = self.lrl3(x3)
      g_mx = torch.zeros(batch_size, y3.shape[1], device=y3.device, dtype=y3.dtype)
      idx = batch[:, None].expand_as(y3)
      g_mx.scatter_reduce_(0, idx, y3, reduce="amax", include_self=False)
      g_mn = torch.zeros(batch_size, y3.shape[1], device=y3.device, dtype=y3.dtype)
      g_mn.index_add_(0, batch, y3)
      counts = torch.bincount(batch, minlength=batch_size).to(y3.dtype)
      g_mn = g_mn / counts[:, None]
      g_mx = g_mx[batch]
      g_mn = g_mn[batch]
      f = torch.cat([y3, g_mx, g_mn, points], dim=1)
      x4 = self.l1(f)
      y4 = self.lrl4(x4)
      x5 = self.l2(y4)
      y5 = self.lrl5(x5)
      x6 = self.l3(y5)
      return F.normalize(x6, 2, dim=1)

   def fit(self, opt:optim.Optimizer, params:dict, train_loader:DataLoader):

      for epoch in tqdm(range(params['num_epochs'])):

         self.train()
         total_loss = 0.0

         for batch in tqdm(train_loader):

            points = batch["points"].to(device)
            normals = batch["normals"].to(device)
            edges = batch["edges"].to(device)
            edge_features = batch["edge_features"].to(device)
            idx = batch['batch']
            opt.zero_grad()
      
            pred_normals = self(
               points,
               edges,
               edge_features,
               idx
            )
      
            dot = (pred_normals * normals).sum(dim=1)
            loss = 1 - dot.mean()
      
            uloss = 1 - dot.abs().mean() 
      
            l = loss #+ 0.5 * uloss
            
            angular_error = torch.rad2deg(
               torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))
            ).mean()
            
            signed_angle = torch.rad2deg(torch.acos(dot)).mean()
      
            unsigned_angle = torch.rad2deg(
               torch.acos(dot.abs())
            ).mean()

            l.backward()
            opt.step()
            total_loss += l.item()

         total_loss /= len(train_loader)
         print(
            f"{epoch + 1}/{params['num_epochs']} | "
            f"loss={total_loss:.4f}"
         )
      


if __name__ == '__main__':

   device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
   )

   print(device)
   model = NormalPredictor().to(device)

   train_data = ModelData('data\\ModelNet10', 'train', 2000, 16)
   test_data = ModelData('data\\ModelNet10', 'test', 2000, 16)

   train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=single_graph_collate)
   test_loader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=single_graph_collate)
   
   params = {
      'lr': 3e-4, 
      'eps': 1e-8, 
      'num_epochs': 300,
      'betas': (0.9, 0.999)
   }

   optimizer = optim.Adam(
      model.parameters(),
      lr=params['lr'],
      betas=params['betas'],
      eps=params['eps']
   )

   model.fit(optimizer, params, train_loader)

   
   
   


