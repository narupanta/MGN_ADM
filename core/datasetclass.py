import json
import torch
from torch_geometric.data import Data, Dataset
import os
from .utils import load_config, triangles_to_edges, tetrahedral_to_edges
import numpy as np

def assemble_node_value(data, data_config, float_type) :
    target_values = data_config["dynamic"]["target"].keys()
    to_stack = []
    for value in target_values :
        to_stack.append(torch.tensor(data[value]) if len(data[value].shape) == 3 else torch.tensor(data[value]).unsqueeze(-1))
    u = torch.cat(to_stack, dim = -1).to(dtype = float_type)
    return u

def get_target_metadata(data_config):
    targets = data_config["dynamic"]["target"]
    target_names = targets.keys()
    u_metadata = []
    dim_count = 0

    for v in target_names:
        target_dim = targets[v]["dim"]
        dbc_map = targets[v].get("dbc_at_node_type", {})

        for i in range(target_dim):
            quantity_name = f"{v}_{i}"
            if i in dbc_map.values():
                # Add an entry for each matching node type
                for node_type, comp_idx in dbc_map.items():
                    if comp_idx == i:
                        u_metadata.append((quantity_name, dim_count, node_type))
            dim_count += 1

    return u_metadata

def assemble_node_load(data, data_config, float_type) :
    load_values = data_config["dynamic"]["load"].keys()
    to_stack = []
    for value in load_values :
        to_stack.append(torch.tensor(data[value]) if len(data[value].shape) == 3 else torch.tensor(data[value]).unsqueeze(-1))
    load = torch.cat(to_stack, dim = -1).to(dtype = float_type)
    return load
def get_load_metadata(data_config) :
    load_values = data_config["dynamic"]["load"].keys()
    load_metadata = []
    dim_count = 0
    for _, v in enumerate(load_values) :
        load_dim = data_config["dynamic"]["load"][v]["dim"]
        load_metadata.append((v, dim_count, load_dim))
        dim_count += load_dim
    return load_metadata

def convert_cells_to_edges(cells, cell_type) :
    if cell_type == "tri" :
        edge_index, senders, receivers = triangles_to_edges(cells).values()
    elif cell_type == "tetra":
        edge_index, senders, receivers = tetrahedral_to_edges(cells).values()
    else :
        print("Cell Type doesn't exist")
    return edge_index, senders, receivers

class DatasetMGN(Dataset):
    def __init__(
        self,
        data_dir,
        add_targets=True,
        add_history=False,
        split_frames=True,
        add_noise=True,
        train_test_split= "train",
        time_window=1
    ):
        super().__init__()
        self.data_dir = data_dir
        self.add_targets = add_targets
        self.add_history = add_history
        self.split_frames = split_frames
        self.add_noise = add_noise
        self.time_window = time_window
        self.data_config = load_config(os.path.join(self.data_dir, "data_config.yml"))
        self.u_metadata = get_target_metadata(self.data_config)
        self.load_metadata = get_load_metadata(self.data_config)
        self.train_test_split = train_test_split
        self.file_name_list = sorted([
            f for f in os.listdir(os.path.join(data_dir, train_test_split))])

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        data_config = load_config(os.path.join(self.data_dir, "data_config.yml"))
        data = np.load(os.path.join(self.data_dir, self.train_test_split, self.file_name_list[idx]))
        float_type = torch.float64
        mesh_pos = torch.tensor(data["mesh_pos"], dtype = float_type)
        cells = torch.tensor(data["node_connectivity"], dtype = torch.long)
        node_type = torch.tensor(data["node_type"]) if "node_type" in data.keys() else torch.zeros((mesh_pos.shape[0], 1))
        edge_index, senders, receivers = convert_cells_to_edges(cells, cell_type = data_config["static"]["cells"]["cell_type"])
        load = assemble_node_load(data, data_config, float_type)
        u = assemble_node_value(data, data_config, float_type)

        trajectory = Data(mesh_pos = mesh_pos,
                     node_type = node_type,
                     cells = cells,
                     edge_index = edge_index,
                     senders = senders, 
                     receivers = receivers,
                     load = load,
                     u = u)
        
        # only u is prediction var
        # split target, current, previous
        # target size depends on time_window
        # previous only onestep back from current
        u_prev = u[:-2]
        u_target = torch.tensor(np.array([u[i:i + self.time_window] for i in range(2, u.shape[0] - self.time_window + 1)]))
        u_curr = u[1: -self.time_window] 

        load_prev = load[:-2]
        load_next = torch.tensor(np.array([load[i:i + self.time_window] for i in range(2, load.shape[0] - self.time_window + 1)]))
        load_curr = load[1: -self.time_window] 

        #split into frames
        if self.split_frames :
            frames = []
            for t in range(u_curr.shape[0]) :
                frame_data = {
                        "mesh_pos": mesh_pos.unsqueeze(0),
                        "senders": senders,
                        "receivers": receivers,
                        "cells": cells.unsqueeze(0),
                        "node_type": node_type.unsqueeze(0),

                        "u" : u_curr[t].unsqueeze(0), # shape (1, #nodes, #u_dim)
                        "u_prev": u_prev[t].unsqueeze(0), # shape (1, #nodes, #u_dim)
                        "u_target" : u_target[t], # shape (time_window, #nodes, #u_dim)

                        "load": load_curr[t].unsqueeze(0), # shape (1, #nodes, #load_dim)
                        "load_prev" : load_prev[t].unsqueeze(0), # shape (1, #nodes, #load_dim)
                        "load_next": load_next[t] # shape (time_window, #nodes, #load_dim)
                    }
                frames.append(Data(**frame_data))
            return frames
        
        return trajectory

    def get_name(self, idx):
        return self.file_name_list[idx]