import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import mlflow
import google.protobuf

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .dataset.dataloader import ModelData
from .geometry.neighbors import *

def global_max_pool(x, batch, batch_size):
    g_max = torch.zeros(batch_size, x.shape[1], device=x.device, dtype=x.dtype)
    idx = batch[:, None].expand_as(x)
    g_max.scatter_reduce_(0, idx, x, reduce="amax", include_self=False)
    return g_max[batch]

def evaluate(model, dataloader, device):
    model.eval()

    total_metrics = {
        "cos": 0.0,
        "angular": 0.0,
        "sign_err": 0.0,
        "axis": 0.0,
        "axis_angular": 0.0,
    }

    with torch.no_grad():
        for batch in dataloader:
            points = batch["points"].to(device)
            normals = batch["normals"].to(device)
            edges = batch["edges"].to(device)
            edge_features = batch["edge_features"].to(device)
            idx = batch["batch"].to(device)
            batch_size = batch["num_graph"]

            pred_normals = model(
                points,
                edges,
                edge_features,
                idx,
                batch_size
            )

            dot = (pred_normals * normals).sum(dim=1)
            dot = dot.clamp(-1.0, 1.0)

            cos_loss = 1 - dot.mean()
            axis_loss = 1 - dot.abs().mean()
            sign_error = (dot < 0).float().mean()

            angular_error = torch.rad2deg(
                torch.acos(dot)
            ).mean()

            axis_angular_error = torch.rad2deg(
                torch.acos(dot.abs())
            ).mean()

            total_metrics["cos"] += cos_loss.item()
            total_metrics["angular"] += angular_error.item()
            total_metrics["sign_err"] += sign_error.item()
            total_metrics["axis"] += axis_loss.item()
            total_metrics["axis_angular"] += axis_angular_error.item()

    for key in total_metrics:
        total_metrics[key] /= len(dataloader)

    return total_metrics

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
        "num_graph": len(batch)
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
      self.e2 = EdgeConv(516, 128)
      self.lrl2 = nn.LeakyReLU()
      self.e3 = EdgeConv(516, 64)
      self.lrl3 = nn.LeakyReLU()
      self.e4 = EdgeConv(394, 128)
      self.lrl4 = nn.LeakyReLU()
      self.l1 = nn.Linear(128, 256)
      self.lrl5 = nn.LeakyReLU()
      self.l2 = nn.Linear(256, 64)
      self.lrl6 = nn.LeakyReLU()
      self.l3 = nn.Linear(64, 3)

   def forward(self, points, edges, features, batch, batch_size):
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
      x4 = self.e4(f, edges, features)
      y4 = self.lrl4(x4)
      x5 = self.l1(y4)
      y5 = self.lrl4(x5)
      x6 = self.l2(y5)
      y6 = self.lrl5(x6)
      x7 = self.l3(y6)
      return F.normalize(x7, 2, dim=1)

   def fit(self, opt:optim.Optimizer, scheduler:optim.lr_scheduler, params:dict, train_loader:DataLoader, valid_loader:DataLoader, device):

      for epoch in tqdm(range(params['num_epochs'])):

         self.train()
         total_loss = 0.0
         total_unsigned_loss = 0.0
         angular_loss = 0.0
         angle_loss = 0.0
         angle_unsigned_loss = 0.0
         wrong_sign = 0.0

         for batch in tqdm(train_loader):

            points = batch["points"].to(device)
            normals = batch["normals"].to(device)
            edges = batch["edges"].to(device)
            edge_features = batch["edge_features"].to(device)
            idx = batch['batch'].to(device)
            l = batch["num_graph"]
            opt.zero_grad()
      
            pred_normals = self(
               points,
               edges,
               edge_features,
               idx,
               l
            )
      
            dot = (pred_normals * normals).sum(dim=1)
            src = edges[0]
            tgt = edges[1]
            pred_pair_nrm = (pred_normals[src] * pred_normals[tgt]).sum(dim=1)
            pair_nrm = (normals[src] * normals[tgt]).sum(dim=1)
            # mse = F.mse_loss(pred_pair_nrm, pair_nrm)
            loss = 1 - dot.mean()
            uloss = 1 - dot.abs().mean() 
            ws = (dot < 0).float().mean()
      
            ls = loss# + params['lmb'] * mse
            
            angular_error = torch.rad2deg(
               torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))
            ).mean()
            
            signed_angle = torch.rad2deg(torch.acos(dot.clamp(-1.0, 1.0))).mean()
      
            unsigned_angle = torch.rad2deg(
               torch.acos(dot.abs().clamp(-1.0, 1.0))
            ).mean()

            ls.backward()
            opt.step()
        
            total_loss += ls.item()
            total_unsigned_loss += uloss.item()
            angular_loss += angular_error.item()
            angle_loss += signed_angle.item()
            angle_unsigned_loss += unsigned_angle.item()
            wrong_sign += ws.item()
            

         total_loss /= len(train_loader)
         total_unsigned_loss /= len(train_loader)
         angular_loss /= len(train_loader)
         angle_loss /= len(train_loader)
         angle_unsigned_loss /= len(train_loader)
         wrong_sign /= len(train_loader)
         valid_metrics = evaluate(self, valid_loader, device)
         scheduler.step(valid_metrics["cos"])
         mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=epoch + 1)
         mlflow.log_metric("train_loss", total_loss, step=epoch + 1)
         mlflow.log_metric("train_unsigned_loss", total_unsigned_loss, step=epoch + 1)
         mlflow.log_metric("angular_loss", angular_loss, step=epoch + 1)
         mlflow.log_metric("wrong_train_sign", wrong_sign, step=epoch + 1)
         mlflow.log_metric("val_cos_err", valid_metrics['cos'], step=epoch + 1)
         mlflow.log_metric("val_angular_err", valid_metrics['angular'], step=epoch + 1)
         mlflow.log_metric("val_axis_err", valid_metrics['axis'], step=epoch + 1)
         mlflow.log_metric("val_sign_err", valid_metrics['sign_err'], step=epoch + 1)
         mlflow.log_metric("val_axis_angular", valid_metrics['axis_angular'], step=epoch + 1)
          
         print(
            f"{epoch + 1}/{params['num_epochs']} | "
            f"loss={total_loss:.4f}"
         )
          
if __name__ == '__main__':

   device = torch.device(
      "cuda:2" if torch.cuda.is_available() else "cpu"
   )

   #print(device)
   model = NormalPredictor().to(device)
   path = '../data-storage/data/ModelNet10'
   print(os.path.exists(path))
   full_train_data = ModelData(path, 'train', 2000, 16)
   test_data = ModelData(path, 'test', 2000, 16)

   val_size = int(0.1 * len(full_train_data))
   train_size = len(full_train_data) - val_size

   train_data, val_data = random_split(
        full_train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
   

   train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=12, collate_fn=single_graph_collate)
   valid_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=12, collate_fn=single_graph_collate)
   test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=12, collate_fn=single_graph_collate)
   
   params = {
      "architecture": "global pooling after each lay",
      "edgeconv_layers": 4,
      "edgeconv_dims": "128-128-64-128",
      "mlp_dims": "128-256-64-3",
      "aggregation": "max+mean",
      "output_norm": "L2",
      'lr': 3e-4, 
      'eps': 1e-8, 
      'num_epochs': 15,
      'betas': (0.9, 0.999),
      'lmb': 0.1
   }

   optimizer = optim.Adam(
      model.parameters(),
      lr=params['lr'],
      betas=params['betas'],
      eps=params['eps']
   )

   scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
    
   experiment_folder = f"{os.environ['S3_ARTIFACT_ROOT']}/exp1"
   experiment_id = 144
   
   with mlflow.start_run(experiment_id=experiment_id):
      mlflow.log_params(params)
      model.fit(optimizer, scheduler, params, train_loader, valid_loader, device)
    

   
   
   


