import torch
from dgl.nn.pytorch import EGTLayer
from torch_geometric.utils import to_dense_batch, to_dense_adj


class EGTNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        num_layers,
        dropout,
        beta=True,
        heads=1,
        edge_output_dim=None,
    ):
        super(EGTNetwork, self).__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.heads = heads
        if edge_output_dim is None:
            self.edge_output_dim = edge_dim
            edge_output_dim = edge_dim
        else:
            self.edge_output_dim = edge_output_dim

        self.node_input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_input_proj = torch.nn.Linear(edge_dim, hidden_dim)

        conv_layers = []
        for _ in range(num_layers):
            conv_layers.append(
                EGTLayer(
                    feat_size=hidden_dim,
                    edge_feat_size=hidden_dim,
                    num_heads=heads,
                    num_virtual_nodes=0,
                    dropout=dropout,
                    edge_update=True,
                )
            )
        self.convs = torch.nn.ModuleList(conv_layers)

        # Layer Normalization
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        )

        # Output Proj
        self.node_output_proj = torch.nn.Linear(hidden_dim, output_dim)
        self.edge_output_proj = torch.nn.Linear(hidden_dim, edge_output_dim)

        self.dropout = dropout

    def reset_parameters(self):
        self.node_input_proj.reset_parameters()
        self.edge_input_proj.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

        self.node_output_proj.reset_parameters()
        self.edge_output_proj.reset_parameters()

    def prepare_inputs(self, x, edge_index, edge_attr, batch):
        x_dense, mask_nodes = to_dense_batch(x, batch)

        if edge_attr is not None:
            edge_dense = to_dense_adj(edge_index, batch, edge_attr)
        else:
            edge_dense = to_dense_adj(edge_index, batch)
            edge_dense = edge_dense.unsqueeze(-1)

        mask_2d = mask_nodes.unsqueeze(-1) & mask_nodes.unsqueeze(-2)
        attention_mask = torch.where(
            mask_2d,
            torch.zeros_like(mask_2d, dtype=torch.float),
            torch.full_like(mask_2d, -1e9, dtype=torch.float),
        )

        return x_dense, edge_dense, attention_mask, mask_nodes, mask_2d

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_input_proj(x)
        if edge_attr is not None:
            edge_attr = self.edge_input_proj(edge_attr)

        x, edge_attr, attn_mask, mask_nodes, mask_adj = self.prepare_inputs(x, edge_index, edge_attr, batch)

        for i in range(self.num_layers - 1):
            x_new, edge_attr_new = self.convs[i](x, edge_attr, attn_mask)
            x = self.norms[i](x_new)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            edge_attr = edge_attr_new

        x, edge_attr = self.convs[-1](x, edge_attr, attn_mask)

        x = self.node_output_proj(x)
        edge_attr = self.edge_output_proj(edge_attr)

        return x, edge_attr, mask_nodes
