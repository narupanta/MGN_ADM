import numpy as np
from torch_geometric.data import Data
import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from core.utils import * 

import numpy as np

def rollout(model, data, metadata, device="cuda"):
    """Roll out predictions by stepping `time_window` ahead at each iteration."""
    # data = list(data)
    data = data.to(device)
    initial_state = data.clone()

    initial_state["mesh_pos"] = data["mesh_pos"].unsqueeze(0)
    initial_state["node_type"] = data["node_type"].unsqueeze(0)

    initial_state = Data(**initial_state)

    timesteps = len(data.u)

    curr_graph = initial_state.clone()
    curr_graph.u = data.u[0].unsqueeze(0)

    start_point = 0
    pred_u_list = [curr_graph.u]
    progress = tqdm(range(start_point, timesteps, model.time_window), desc="Rollout")

    for t in progress:
        # === Model predicts time_window steps ===
        with torch.no_grad():

            curr_graph.load = data["load"][t].unsqueeze(0)
            # curr_graph.u = data["u"][t].unsqueeze(0)
            pred_u = model.predict(curr_graph.to(device), metadata)
            curr_graph.u = pred_u[-1:]  # advance to last prediction
        pred_u_list.append(pred_u)

    pred_u_tensor = torch.cat(pred_u_list, dim=0)[:timesteps]
    gt_u_tensor = data.u

    min_vals = torch.amin(pred_u_tensor, dim = (0,1))  # shape: (1, 1, 3)
    max_vals = torch.amax(pred_u_tensor, dim = (0,1))  # shape: (1, 1, 3)

    # Normalize per dimension
    norm_pred = (pred_u_tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid div by 0

    # Do the same for GT
    min_vals_gt = torch.amin(gt_u_tensor, dim=(0, 1), keepdim=True)
    max_vals_gt = torch.amax(gt_u_tensor, dim=(0, 1), keepdim=True)
    norm_gt = (gt_u_tensor - min_vals_gt) / (max_vals_gt - min_vals_gt + 1e-8)
    rmse_per_frame = torch.sqrt(torch.mean((pred_u_tensor - gt_u_tensor)**2, dim = 1))
    rmse_per_trajectory = torch.mean(rmse_per_frame, dim = 0)
    rel_rmse_per_frame = torch.sqrt(torch.mean((norm_pred - norm_gt)**2, dim = 1))
    rel_rmse_trajectory = torch.mean(rel_rmse_per_frame, dim = 0)
    output = {"mesh_pos": initial_state.mesh_pos.squeeze(0),
              "node_type": initial_state.node_type.squeeze(0),
              "cells": initial_state.cells}
    for target, index, _, _ in metadata :
        output[f"{target}_rel_rmse_per_frame"] = rel_rmse_per_frame[:, index]
        output[f"{target}_rel_rmse_trajectory"] = rel_rmse_trajectory[index]
        output[f"{target}__rmse_per_frame"] = rmse_per_frame[:, index]
        output[f"{target}__rmse_trajectory"] = rmse_per_trajectory[index]
    output[f"pred_u"] = pred_u_tensor
    output[f"gt_u"] = gt_u_tensor
    return output
