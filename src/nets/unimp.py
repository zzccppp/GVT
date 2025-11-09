from torch_geometric.nn import TransformerConv
import torch
import torch_geometric.nn as pygnn


class UniMP(torch.nn.Module):

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
    ):
        super(UniMP, self).__init__()

        self.num_layers = num_layers
        conv_layers = [
            TransformerConv(input_dim, hidden_dim // heads, heads=heads, beta=beta, edge_dim=edge_dim)
        ]
        conv_layers += [
            TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, beta=beta, edge_dim=edge_dim)
            for _ in range(num_layers - 2)
        ]

        # setting concat to True.
        conv_layers.append(
            TransformerConv(hidden_dim, output_dim, heads=heads, beta=beta, concat=True, edge_dim=edge_dim)
        )
        self.convs = torch.nn.ModuleList(conv_layers)

        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)
        # Probability of an element getting zeroed.
        self.dropout = dropout

        self.output_lin = torch.nn.Linear(output_dim * heads, output_dim)

    def reset_parameters(self):
        """
        Resets the parameters of the convolutional and normalization layers,
        ensuring they are re-initialized when needed.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

        self.output_lin.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        # x = pygnn.global_max_pool(x, batch)
        # x = pygnn.global_mean_pool(x, batch)
        x = self.output_lin(x)

        return x
