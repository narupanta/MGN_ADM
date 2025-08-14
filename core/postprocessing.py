import os 
import numpy as np
import meshio
from contextlib import contextmanager
from pathlib import Path
# def plot_paraview(output, rollout_dir) :
#     base_dir = "/mnt/c/Users/narun/OneDrive/Desktop/Project/Hydrogel_MGN/Hydrogel_MGN/rollout/nonlinear_rollout/bending"
#     pred_dir = os.path.join(base_dir, "pred")
#     gt_dir = os.path.join(base_dir, "gt")

#     os.makedirs(pred_dir, exist_ok=True)
#     os.makedirs(gt_dir, exist_ok=True)

#     # Ensure displacements are padded to 3D
#     def pad_to_3d(arr):
#         if arr.shape[1] == 2:
#             return np.hstack([arr, np.zeros((arr.shape[0], 1))])
#         return arr
#     mesh_pos = output["mesh_pos"].detach().cpu().numpy()
#     cells = output["cells"].detach().cpu().numpy()
#     for timestep in range(len(output["gt_displacement"])):
#         # Base mesh

#         # Predicted
#         pred_disp = output["predict_displacement"][timestep].squeeze(0).detach().cpu().numpy()
#         pred_pvf = output["predict_pvf"][timestep].squeeze(0).detach().cpu().numpy()

#         # Ground truth
#         gt_disp = output["gt_displacement"][timestep].squeeze(0).detach().cpu().numpy()
#         gt_pvf = output["gt_pvf"][timestep].squeeze(0).detach().cpu().numpy()
#         pvf_env = output["pvf_env"][timestep].squeeze(0).detach().cpu().numpy()
#         # Pad displacements to 3D if needed
#         pred_disp_3d = pad_to_3d(pred_disp)
#         gt_disp_3d = pad_to_3d(gt_disp)

#         # Write predicted file
#         meshio.write_points_cells(
#             os.path.join(pred_dir, f"pred_{timestep:04d}.vtu"),
#             points=pred_disp_3d,
#             cells=[("triangle", cells)],
#             point_data={"pvf": pred_pvf, "pvf_env": pvf_env}
#         )

#         # Write ground truth file
#         meshio.write_points_cells(
#             os.path.join(gt_dir, f"gt_{timestep:04d}.vtu"),
#             points=gt_disp_3d,
#             cells=[("triangle", cells)],
#             point_data={"pvf": gt_pvf, "pvf_env": pvf_env}

#         )

import meshio
import numpy as np
import os

def export_to_paraview(mesh_pos, cells, pred, gt, out_dir):
    """
    mesh_pos: (num_nodes, domain_dim)  -> coordinates of nodes
    cells: dict or list of tuples, e.g., [("triangle", [[...], ...])]
    pred: (num_steps, num_nodes, num_dims) -> predictions over time
    gt:   (num_steps, num_nodes, num_dims) -> ground truth over time
    out_dir: str, output folder
    """
    os.makedirs(out_dir, exist_ok=True)
    num_steps = pred.shape[0]

    # Determine cell type string for meshio
    if isinstance(cells, np.ndarray):
        # If you just have indices and know type
        cell_block = [("triangle", cells)]  # change to "tetra", "quad", etc. if needed
    elif isinstance(cells, list):
        cell_block = cells
    elif isinstance(cells, dict):
        # If stored like {"triangle": array}
        cell_block = [(k, v) for k, v in cells.items()]
    else:
        raise ValueError("Unsupported cells format")

    pvd_entries = []

    for t in range(num_steps):
        point_data = {
            "prediction_world_pos": pred[t][:, :2],  # shape: (num_nodes, num_dims)
            "groundtruth_world_pos": gt[t][:, :2],   # shape: (num_nodes, num_dims)
            "prediction_pvf": pred[t][:, 2:],  # shape: (num_nodes, num_dims)
            "groundtruth_pvf": gt[t][:, 2:],   # shape: (num_nodes, num_dims)
            "error": (pred[t] - gt[t])
        }

        filename = os.path.join(out_dir, f"step_{t:04d}.vtu")
        mesh = meshio.Mesh(points=pred[t][:, :2], cells=cell_block, point_data=point_data)
        meshio.write(filename, mesh)

        pvd_entries.append((t, f"step_{t:04d}.vtu"))

    # Write PVD file for animation
    pvd_path = os.path.join(out_dir, "timesteps.pvd")
    meshio.write_points_cells(pvd_path, mesh_pos, cell_block)  # minimal to create file
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t, fname in pvd_entries:
            f.write(f'    <DataSet timestep="{t}" part="0" file="{fname}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"Exported {num_steps} timesteps to {out_dir} (open timesteps.pvd in Paraview)")


def plot_gif() :
    pass
def plot_training_progress() :
    pass
def plot_rollout_error() :
    pass
def export_to_paraview_xdmf(mesh_pos, cells, pred_u, gt_u, save_path, trajectory_name="trajectory"):
    """
    Export rollout results to a single XDMF (HDF5-backed) file with all timesteps.

    Args:
        mesh_pos: (num_nodes, dim)
        cells: (num_cells, nodes_per_cell) int array
        pred_u: (num_steps, num_nodes, num_features)
        gt_u: (num_steps, num_nodes, num_features)
        save_path: folder to save file in
        trajectory_name: file name without extension
    """
    os.makedirs(save_path, exist_ok=True)

    num_steps, num_nodes, num_features = pred_u.shape
    cells = [("triangle", cells)]  # change type if not triangles

    # Repeat mesh coordinates for each timestep
    # points_time = np.tile(mesh_pos, (num_steps, 1))

    # # Repeat cell connectivity for each timestep
    # cells_time = cells * num_steps

    # Create Mesh object
    mesh = meshio.Mesh(
        points=mesh_pos,
        cells=cells,
        point_data={
            "prediction": pred_u,
            "groundtruth": gt_u,
        }
    )

    # Write to XDMF (this will generate .xdmf + .h5 files)
    out_file = os.path.join(save_path, f"{trajectory_name}.xdmf")
    meshio.write(out_file, mesh, file_format="xdmf")

    print(f"✅ Saved rollout to {out_file}")


@contextmanager
def working_directory(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(old)

def export_to_xdmf(mesh_pos, cells, pred_u, gt_u, save_path: Path, *,
                   cell_type: str | None = None, t0: float = 0.0, dt: float = 1.0):
    """
    Write a single XDMF time series: save_path/rollout.xdmf + rollout.h5 (same folder).

    mesh_pos: (N, dim) float
    cells:    (M, nodes_per_elem) int
    pred_u:   (T, N, F) float
    gt_u:     (T, N, F) float
    cell_type: override (e.g. "triangle", "quad", "tetra"); otherwise guessed from nodes_per_elem
    t0, dt:   time origin and step for the XDMF time values (t = t0 + k*dt)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Dtypes and contiguity (meshio/h5py can be picky)
    mesh_pos = np.asarray(mesh_pos, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    pred_u = np.asarray(pred_u, dtype=np.float32)
    gt_u   = np.asarray(gt_u,   dtype=np.float32)

    # Determine cell type if not provided
    if cell_type is None:
        n_per = int(cells.shape[1])
        # 2D meshes
        if n_per == 3:
            cell_type = "triangle"
        elif n_per == 4:
            # change to "tetra" if your mesh is 3D with 4-node tets
            cell_type = "tetra"
        elif n_per == 2:
            cell_type = "line"
        elif n_per == 1:
            cell_type = "vertex"
        else:
            raise ValueError(f"Unsupported cells with {n_per} nodes/element. "
                             f"Pass cell_type explicitly.")

    cell_block = [(cell_type, cells)]
    T = int(pred_u.shape[0])

    # Force meshio to write into save_path by changing CWD just for the write
    with working_directory(save_path):
        xdmf_name = "rollout.xdmf"
        xdmf_abs = Path.cwd() / xdmf_name  # absolute, inside save_path

        with meshio.xdmf.TimeSeriesWriter(str(xdmf_abs)) as writer:
            writer.write_points_cells(mesh_pos, cell_block)

            for k in range(T):
                t_val = float(t0 + k * dt)
                writer.write_data(
                    t_val,
                    point_data={
                        "Predicted_U":   pred_u[k],           # (N, F)
                        "Ground_Truth_U": gt_u[k],            # (N, F)
                        "Error":         (pred_u[k] - gt_u[k])# (N, F)
                    }
                )

        # Sanity check: both files in same folder
        h5_abs = xdmf_abs.with_suffix(".h5")
        if not h5_abs.exists():
            raise RuntimeError(
                f"HDF5 not found next to XDMF.\nExpected: {h5_abs}\n"
                f"Actual dir listing: {list(Path.cwd().iterdir())}"
            )

    print(f"✅ Wrote:\n  {xdmf_abs}\n  {h5_abs}\n(Open the .xdmf in ParaView)")
