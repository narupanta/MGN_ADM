import os
import time
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm, Conv1d, LazyLinear, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch_scatter
from scipy.spatial import Delaunay
from .normalization import Normalizer
from torch_geometric.nn import knn_graph
torch.set_default_dtype(torch.float64)
class Swish(torch.nn.Module):
    """Swish activation function."""
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
class MLP(torch.nn.Module) :
    def __init__(self, input_size, latent_size, device) :
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.device = device
        self.mlp = Sequential(
            Linear(self.input_size, self.latent_size),
            ReLU(),
            Linear(self.latent_size, self.latent_size),
            ReLU(),
            LayerNorm(self.latent_size)).to(self.device)
    def forward(self, input) :
        return self.mlp(input)

class GraphNetBlock(torch.nn.Module):
    """Graph Network block with residual connections."""
    def __init__(self, latent_size, graph_amount, device):
        super().__init__()
        self.latent_size = latent_size
        self.device = device
        self.edge_feature_nets = torch.nn.ModuleList([MLP(latent_size * 3, self.latent_size, device) for _ in range(graph_amount)])
        self.node_feature_net = MLP(latent_size * (1 + graph_amount), self.latent_size, device)

    def forward(self, graph):
        node_latents, edge_latents = graph.node_latents, graph.edge_latents
        new_edge_latents = []
        aggrs = []
        for idx, edge_latent in enumerate(edge_latents) :
            edge_input = torch.cat([node_latents[:, edge_latent["senders"], :], node_latents[:, edge_latent["receivers"], :], edge_latent["features"]], dim=-1)
            new_edge_latent = self.edge_feature_nets[idx](edge_input)
            aggr = torch_scatter.scatter_add(new_edge_latent.to(torch.float64), edge_latent["receivers"], dim = 1, dim_size = node_latents.shape[1])
            new_edge_latents.append(new_edge_latent)
            aggrs.append(aggr)
        node_input = torch.cat([node_latents] + aggrs, dim=-1)
        new_node_latents = self.node_feature_net(node_input)
        return Data(
            node_latents = new_node_latents + node_latents,
            edge_latents = [{"features": edge_latent["features"] + new_edge_latents[idx], 
                             "senders": edge_latent["senders"], 
                             "receivers": edge_latent["receivers"]} for idx, edge_latent in enumerate(graph.edge_latents)],
        )


class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode architecture for GNN."""

    def __init__(self,
                 latent_size,
                 timestep,
                 time_window,
                 message_passing_steps,
                 voxel_size,
                 include_push_forward,
                 data_config,
                 device,
                 name='EncodeProcessDecode'):
        super().__init__()
        self.name = name
        self.device = device

        self.latent_size = latent_size
        self.time_window = time_window
        self.timestep = timestep
        self.message_passing_steps = message_passing_steps
        self.voxel_size = voxel_size
        self.include_push_forward = include_push_forward

        self.data_config = data_config
        self.mesh_pos_dim = data_config["static"]["mesh_pos"]["dim"]
        self.node_type_dim = data_config["static"]["node_type"]["dim"]
        self.target_dim = sum([v["dim"] for v in data_config["dynamic"]["target"].values()])
        self.load_dim = sum([v["dim"] for v in data_config["dynamic"]["load"].values()])

        self.node_encoder = MLP(self.target_dim + self.load_dim + self.node_type_dim, self.latent_size, device)
        self.edge_encoder = MLP(self.mesh_pos_dim + self.target_dim + 1, self.latent_size, device)
        # Normalizers
        self.output_normalizer = Normalizer(time_window, self.target_dim, 'output_normalizer', device)
        self.node_normalizer = Normalizer(1, self.target_dim + self.load_dim + self.node_type_dim, 'node_features_normalizer', device)
        self.graph_amount = 1
        self.edge_normalizer = Normalizer(1, self.mesh_pos_dim + self.target_dim + 1, 'edge_normalizer', device)
        if self.voxel_size : 
            self.graph_amount = 2 
            self.coarse_edge_encoder = MLP(self.mesh_pos_dim + self.target_dim + 1, self.latent_size, device)
            self.coarse_edge_normalizer = Normalizer(1, self.mesh_pos_dim + self.target_dim + 1, 'coarse_edge_normalizer', device)
        # GNN core
        self.graphnet_blocks = torch.nn.ModuleList([
            GraphNetBlock(latent_size, self.graph_amount, device = device)
            for _ in range(message_passing_steps)
        ])

        # Decoder
        self.node_decoder = Sequential(
            Conv1d(latent_size, 8, 1),
            Swish(),
            Conv1d(8, self.target_dim * time_window, 1)
        ).to(device)

    def _forward(self, mesh_pos, senders, receivers, node_type, u, load) :
        latent_graph = self._encode(mesh_pos, senders, receivers, node_type, u, load)

        for block in self.graphnet_blocks:
            latent_graph = block(latent_graph)

        node_latents = latent_graph.node_latents.permute(0, 2, 1)
        decoded = self.node_decoder(node_latents).permute(0, 2, 1)

        dt = torch.arange(1, self.time_window + 1).repeat_interleave(self.target_dim).to(self.device)
        delta = (decoded * dt).reshape(-1, self.time_window, self.target_dim).permute(1, 0, 2)

        return delta
    def forward(self, graph):
        if self.include_push_forward :
            with torch.no_grad():
                norm_u_dot_prev = self._forward(graph.mesh_pos, graph.senders, graph.receivers, graph.node_type, 
                                            graph.u_prev, graph.load_prev)
            u_dot_prev = self.output_normalizer.inverse(norm_u_dot_prev)[0]
            pred_u = graph.u_prev + u_dot_prev
            delta_u = self._forward(graph.mesh_pos, graph.senders, graph.receivers, graph.node_type, 
                                    pred_u, graph.load)
        else :
            delta_u = self._forward(graph.mesh_pos, graph.senders, graph.receivers, graph.node_type, 
                                  graph.u, graph.load)
        return delta_u

    def predict(self, graph, u_metadata):
        self.eval()
        node_type = graph.node_type
        output = self._forward(graph.mesh_pos, graph.senders, graph.receivers, node_type, graph.u, graph.load)
        delta_u = self.output_normalizer.inverse(output)
        for _, dim_idx, bc_node_type, _ in u_metadata:
            # Mask out nodes that are Dirichlet BC for this component
            mask = (node_type[:, :, bc_node_type] == 1).squeeze(0)
            delta_u[:, mask, dim_idx] = 0
        u_pred = graph.u + delta_u
        return u_pred
    
   
    def loss(self, pred_delta_u, graph, u_metadata):
        # node_type: (1, num_nodes, num_node_types) -> integer type per node
        node_type = graph.node_type # shape: (num_nodes,)

        u_curr = graph.u            # (1, num_nodes, dim)
        u_target = graph.u_target   # (1, num_nodes, dim)

        target_u_dot = (u_target - u_curr)
        normalized_target_delta_u = self.output_normalizer(target_u_dot)

        # error shape: (1, num_nodes, dim)
        error = (pred_delta_u - normalized_target_delta_u) ** 2

        loss_dict = {}
        comp_losses = []

        for name, dim_idx, bc_node_type, _ in u_metadata:
            # Mask out nodes that are Dirichlet BC for this component
            mask = ~(node_type[:, :, bc_node_type] == 1).squeeze(0)
            masked_error = error[:, mask, dim_idx]

            if masked_error.numel() > 0:
                comp_loss = masked_error.mean()
            else:
                comp_loss = 0.0  # no valid nodes, loss = 0

            loss_dict[name] = comp_loss
            comp_losses.append(comp_loss)

        # Average over components instead of sum
        total_loss = sum(comp_losses) / len(comp_losses) if comp_losses else 0.0

        return total_loss, loss_dict

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_normalizer, os.path.join(path, "edge_features_normalizer.pth"))
        if self.voxel_size :
            torch.save(self.coarse_edge_normalizer, os.path.join(path, "coarse_edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth")))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
        if self.voxel_size :
            self.coarse_edge_normalizer = torch.load(os.path.join(path, "coarse_edge_features_normalizer.pth"))

    def _encode(self, mesh_pos, senders, receivers, node_type, u, load):
        encoded_edges = []
        node_feats = self._build_node_features(node_type, u, load)
        edge_feats = self._build_edge_features(mesh_pos, senders, receivers, u)
        encoded_edges.append({"features":self.edge_encoder(self.edge_normalizer(edge_feats["features"])),
                              "senders":edge_feats["senders"], 
                              "receivers": edge_feats["receivers"]})
        if self.voxel_size :
            coarse_edge_feats = self._build_coarse_edge_features(mesh_pos, u)
            encoded_edges.append({"features":self.coarse_edge_encoder(self.coarse_edge_normalizer(coarse_edge_feats["features"])),
                                  "senders":coarse_edge_feats["senders"], 
                                  "receivers": coarse_edge_feats["receivers"]})
        return Data(
            node_latents = self.node_encoder(self.node_normalizer(node_feats)),
            edge_latents = encoded_edges
        )

    def _build_node_features(self, node_type, u, load):
        # one_hot_type = F.one_hot(graph.node_type).float()[:, 1,:]
        one_hot_type = node_type
        return torch.cat([u, load, one_hot_type], dim=-1)

    def _build_edge_features(self, mesh_pos, senders, receivers, u):
        rel_mesh = mesh_pos[:, senders, :] - mesh_pos[:, receivers, :]
        dist_mesh = torch.norm(rel_mesh, dim=-1, keepdim=True)
        grad_u = u[:, senders, :] - u[:, receivers, :]
        return {"features": torch.cat([rel_mesh, dist_mesh, grad_u], dim=-1),
                "senders": senders, 
                "receivers": receivers}
    
    def _build_coarse_edge_features(self, mesh_pos, u):

        def voxel_grid_sampling(points, voxel_size_percentage):
            device = points.device
            points = points.to(torch.float64)
            coord_range = torch.max(points, dim=1).values - torch.min(points, dim=1).values
            mean_range = torch.mean(torch.abs(coord_range))
            voxel_size = mean_range * voxel_size_percentage
            # Integer voxel coordinates
            coords = torch.floor(points / voxel_size)

            # Unique voxel positions + first index
            coords_unique, inverse_indices = torch.unique(coords, dim=1, return_inverse=True)
            first_indices = torch.zeros(coords_unique.shape[1], dtype=torch.long, device=device)

            for i in range(coords_unique.shape[1]):
                first_indices[i] = torch.nonzero(inverse_indices == i, as_tuple=False)[0]

            return points[:, first_indices, :], first_indices
        if mesh_pos.shape[-1] == 3 :
            k = 4
        else :
            k = 3
        voxel_size = self.voxel_size
        sampled_points, first_indices = voxel_grid_sampling(mesh_pos, voxel_size)

        sampled_edge_index = knn_graph(sampled_points.squeeze(0), k=k, loop=False).T
        sampled_edge_index_by_original = first_indices[sampled_edge_index]
        sampled_senders, sampled_receivers = sampled_edge_index_by_original[:, 0], sampled_edge_index_by_original[:, 1]

        rel_mesh = mesh_pos[:, sampled_senders, :] - mesh_pos[:, sampled_receivers, :]
        dist_mesh = torch.norm(rel_mesh, dim=-1, keepdim=True)
        grad_u = u[:, sampled_senders, :] - u[:, sampled_receivers, :]
        return {"features" : torch.cat([rel_mesh, dist_mesh, grad_u], dim=-1), 
                "senders": sampled_senders, 
                "receivers": sampled_receivers}