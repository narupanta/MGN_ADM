import os
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import torch
from torch_geometric.loader import DataLoader
from core.postprocessing import export_to_xdmf
from core.datasetclass import DatasetMGN
from core.model import EncodeProcessDecode
from core.rollout import rollout
from core.utils import load_config, prepare_directories, logger_setup


def load_config(path="train_config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def train(model, train_dataset, val_dataset, optimizer, run_dir, model_dir, logs_dir, cfg, device):
    logger_setup(os.path.join(logs_dir, "logs.txt"))
    logger = logging.getLogger()

    best_val_loss = float("inf")
    start_epoch = 0
    #start_epoch = from load model dir to perform continue training 
    num_epochs = cfg["training"]["num_epochs"]
    time_window = cfg["model"]["time_window"]

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_total_loss = 0

        logger.info(f"==== Epoch {epoch + 1} ====")

        for traj_idx, trajectory in enumerate(train_dataset):
            traj_total_loss = 0
            traj_loss_split = {f"{target_name} loss" : 0.0 for target_name, _, _, _ in train_dataset.u_metadata}
            train_loader = DataLoader(trajectory, batch_size=1, shuffle=True)
            loop = tqdm(train_loader, leave=False)

            for batch in loop:
                optimizer.zero_grad()
                batch = batch.to(device)
                start = time.perf_counter()
                delta = model(batch, train_dataset.u_metadata)
                t_delta = time.perf_counter() - start

                start = time.perf_counter()
                total_loss, loss = model.loss(delta, batch, train_dataset.u_metadata)
                t_loss = time.perf_counter() - start

                # print(f"delta() took {t_delta:.6f}s, loss() took {t_loss:.6f}s")
                if epoch > 0:
                    total_loss.backward()
                    optimizer.step()

                    traj_total_loss += total_loss.item()

                    loop.set_description(f"Epoch {epoch + 1}, Traj {traj_idx + 1}")
                    postfix = {}
                    for k,v in loss.items() :
                        postfix[f"{k} loss"] =  f"{v.item():.4f}"
                        traj_loss_split[f"{k} loss"] += v.item()

                    postfix["total loss"] = f"{total_loss.item():.4f}"
                    loop.set_postfix(postfix)

            loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in traj_loss_split.items())

            log_line = (
                f"Epoch {epoch + 1} "
                f"Trajectory {traj_idx + 1}: "
                f"Total Loss: {traj_total_loss:.4f}, "
                f"{loss_str}"
            )
            train_total_loss += traj_total_loss
            logger.info(log_line)

        # === Validation ===
        if epoch > 0 :
            val_total_loss = 0.0
            for traj_idx, trajectory in enumerate(val_dataset):
                output = rollout(model, trajectory, train_dataset.u_metadata, device)
                val_loss = 0.0
                val_i = 0
                val_traj_loss_split = {}
                for k, v in output.items() :
                    if k.endswith("__rmse_trajectory") :
                        val_loss += v
                        val_i += 1
                        val_traj_loss_split[f"Rollout {k}"] = v
                val_total_loss += val_loss.item()/val_i
                val_loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_traj_loss_split.items())
                logger.info(
                    f"Rollout Trajectory {traj_idx + 1}: Rollout RMSE: {val_loss:.6e}, {val_loss_str}"
                )

            avg_train_loss = train_total_loss / len(train_dataset)
            avg_val_loss = val_total_loss / len(val_dataset)

            logger.info(f"Epoch {epoch + 1} Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.6e}")
            print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.6e}")

            if avg_val_loss < best_val_loss:
                model.save_model(model_dir)
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_state_dict.pth"))
                best_val_loss = avg_val_loss
                export_to_xdmf(
                    mesh_pos=output["mesh_pos"].detach().cpu().numpy(),
                    cells=output["cells"].detach().cpu().numpy(),
                    pred_u=output["pred_u"].detach().cpu().numpy(),
                    gt_u=output["gt_u"].detach().cpu().numpy(),
                    save_path=Path("/home/y0113799/Hiwi/MGN_ADM/rollouts/lpbf3D/c")
                )
                print(torch.sqrt(torch.mean((output["pred_u"] - output["gt_u"])**2)))
                logger.info("Checkpoint saved (best model so far).")


def load_checkpoint_if_available(model, optimizer, model_dir):
    optim_path = os.path.join(model_dir, "optimizer_state_dict.pth")

    if os.path.exists(model_dir) and os.path.exists(optim_path):
        model.load_model(model_dir)
        optimizer.load_state_dict(torch.load(optim_path))
        print(f"Resumed training from checkpoint at: {model_dir}")
        return True
    else:
        print(f"Model path not found in: {model_dir}. Starting fresh.")
        return False

def setup_training_environment(cfg):
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    paths = cfg["paths"]

    # Decide run_dir and logging/model paths
    if paths.get("model_load_dir"):  # Resume from checkpoint
        run_dir = prepare_directories(paths["model_save_dir"])
        model_dir = paths["model_dir"]
        logs_dir = os.path.join(run_dir, "logs")

        config_summary_path = os.path.join(run_dir, "config_summary.yaml")
        with open(config_summary_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    else:  # Start fresh training
        run_dir = prepare_directories(paths["model_save_dir"])
        model_dir = os.path.join(run_dir, "model_checkpoint")
        logs_dir = os.path.join(run_dir, "logs")

        # Save a clean summary of the config at start of training
        config_summary_path = os.path.join(run_dir, "config_summary.yaml")
        with open(config_summary_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    noise_level = cfg["training"]["noise_level"]
    train_dataset = DatasetMGN(paths["data_dir"], 
                               time_window = cfg["model"]["time_window"],
                               noise_level = noise_level,
                               split_frames = True)

    val_dataset = DatasetMGN(paths["data_dir"], 
                             time_window = cfg["model"]["time_window"],
                             split_frames = False)
    model = build_model(cfg, train_dataset.data_config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    resumed = False
    if paths.get("model_dir"):
        resumed = load_checkpoint_if_available(model, optimizer, model_dir)

    return model, optimizer, train_dataset, val_dataset, run_dir, model_dir, logs_dir, cfg, device


def main() :
    import argparse
    parser = argparse.ArgumentParser(description="Train EncodeProcessDecode model")
    parser.add_argument('--config', type=str, default="./train_configs/hydrogel2D_config.yml", help="Path to the config YAML file")
    args = parser.parse_args()
    # config_dir = "/mnt/c/Users/narun/OneDrive/Desktop/Project/MGN_ADM/train_configs/hydrogel2D_config.yml"
    cfg = load_config(args.config)
    model, optimizer, train_dataset, val_dataset, run_dir, model_dir, logs_dir, cfg, device = setup_training_environment(cfg)

    train(model, train_dataset, val_dataset, optimizer, run_dir, model_dir, logs_dir, cfg, device)
if __name__ == "__main__" :
    main()