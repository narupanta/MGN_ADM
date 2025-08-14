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
from core.postprocessing import export_to_xdmf
import os
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import torch
import numpy as np
from datetime import datetime
from torch_geometric.loader import DataLoader

from core.datasetclass import DatasetMGN
from core.model import EncodeProcessDecode
from core.rollout import rollout
from core.utils import load_config, prepare_directories, logger_setup
from core.postprocessing import export_to_paraview


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rollout EncodeProcessDecode model")
    parser.add_argument('--config', type=str, default="/mnt/c/Users/narun/OneDrive/Desktop/Project/MGN_ADM/rollout_configs/lpbf3D_config.yml", help="Path to the config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    paths = cfg["paths"]
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    # Make sure dirs exist
    model_load_dir = Path(paths["model_load_dir"])
    data_dir = Path(paths["data_dir"])
    rollout_root = Path(paths["rollout_dir"])

    if not model_load_dir.exists():
        raise FileNotFoundError(f"Model load dir not found: {model_load_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    rollout_root.mkdir(parents=True, exist_ok=True)
    model_config = load_config(model_load_dir / "config_summary.yaml")
    # Create dated subfolder
    date_str = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    rollout_date_dir = rollout_root / date_str
    rollout_date_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    test_dataset = DatasetMGN(
        data_dir,
        time_window=model_config["model"]["time_window"],
        split_frames=False,
        train_test_split=paths["split"]
    )

    # Build and load model
    model = build_model(model_config, test_dataset.data_config, device)
    model.load_model(model_load_dir / "model_checkpoint")
    model.eval()
    model = torch.compile(model)
    # Loop through dataset
    for traj_idx in range(len(test_dataset)):
        data = test_dataset[traj_idx]
        x = test_dataset.get_name(traj_idx)
        # Run rollout
        output = rollout(model, data, test_dataset.u_metadata, device)

        # Prepare trajectory folder
        traj_dir = rollout_date_dir / f"trajectory_{traj_idx:03d}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Save NPZ
        npz_path = traj_dir / "rollout_data.npz"
        np.savez_compressed(
            npz_path,
            mesh_pos=output["mesh_pos"].detach().cpu().numpy(),
            cells=output["cells"].detach().cpu().numpy(),
            node_type=output["node_type"].detach().cpu().numpy(),
            pred_u=output["pred_u"].detach().cpu().numpy(),
            gt_u=output["gt_u"].detach().cpu().numpy(),
        )

        # Save Paraview files
        export_to_xdmf(
            mesh_pos=output["mesh_pos"].detach().cpu().numpy(),
            cells=output["cells"].detach().cpu().numpy(),
            pred_u=output["pred_u"].detach().cpu().numpy(),
            gt_u=output["gt_u"].detach().cpu().numpy(),
            save_path=Path(traj_dir)
        )

        print(f"Saved trajectory {traj_idx} to {traj_dir}")


def build_model(cfg, data_config, device):
    model_config = cfg["model"]
    model = EncodeProcessDecode(
        latent_size=model_config["latent_size"],
        timestep=float(model_config["timestep"]),
        time_window=model_config["time_window"],
        message_passing_steps=model_config["message_passing_steps"],
        include_push_forward=model_config["include_push_forward"],
        voxel_size=model_config["voxel_size"],
        data_config=data_config,
        device=device,
    )
    return model.to(device)


if __name__ == "__main__":
    main()
