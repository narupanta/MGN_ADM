import os 
import numpy as np
import meshio
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
            "prediction": pred[t],  # shape: (num_nodes, num_dims)
            "groundtruth": gt[t],   # shape: (num_nodes, num_dims)
            "error": (pred[t] - gt[t])
        }

        filename = os.path.join(out_dir, f"step_{t:04d}.vtu")
        mesh = meshio.Mesh(points=mesh_pos, cells=cell_block, point_data=point_data)
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
