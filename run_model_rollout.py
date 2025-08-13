import os
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import torch
from torch_geometric.loader import DataLoader

from core.datasetclass import DatasetMGN
from core.model import EncodeProcessDecode
from core.rollout import rollout
from core.utils import load_config, prepare_directories, logger_setup
from core.postprocessing import export_to_paraview
def main() :
    cfg = load_config("/mnt/c/Users/narun/OneDrive/Desktop/Project/MGN_ADM/trained_models/hydrogel2D/2025-08-12T16h02m49s/config_summary.yaml")
    paths = cfg["paths"]
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    test_dataset = DatasetMGN(paths["data_dir"], 
                             time_window = cfg["model"]["time_window"],
                             split_frames = False)
    model = build_model(cfg, test_dataset.data_config, device)
    model.load_model("/mnt/c/Users/narun/OneDrive/Desktop/Project/MGN_ADM/trained_models/hydrogel2D/2025-08-12T16h02m49s/model_checkpoint")
    data = test_dataset[0]

    output = rollout(model, data, test_dataset.u_metadata, device)
    mesh_pos = output["mesh_pos"].detach().cpu().numpy()
    cells = output["cells"].detach().cpu().numpy()
    node_type = output["node_type"].detach().cpu().numpy()
    pred_u = output["pred_u"].detach().cpu().numpy()
    gt_u = output["gt_u"].detach().cpu().numpy()
    export_to_paraview(mesh_pos, cells, pred_u, gt_u, "/mnt/c/Users/narun/OneDrive/Desktop/Project/MGN_ADM/rollouts/hydrogel2D")
    check = output
def build_model(cfg, data_config, device):
    model_config = cfg["model"]
    model = EncodeProcessDecode(
        latent_size = model_config["latent_size"],
        timestep = float(model_config["timestep"]),
        time_window = model_config["time_window"],
        message_passing_steps = model_config["message_passing_steps"],
        include_push_forward = model_config["include_push_forward"],
        voxel_size = model_config["voxel_size"],
        data_config = data_config,
        device = device)
    return model.to(device)
def prepare_rollout_directory() :
    pass

if __name__ == "__main__" :
    main()