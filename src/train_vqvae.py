import wandb
from load_data import get_dataset
from utils.commons import (
    init_wandb,
    setseed,
    generate_run_name,
    generate_save_path,
    save_model,
    setup_save_dir,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
import matplotlib
from utils.args import parse_args, update_config
from vqvae.vq import VectorQuantize
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, get_laplacian
from tqdm.auto import tqdm
from vqvae.decoder import EGTDecoderRoPE
from utils.plots import (
    plot_distribution_individual_bins_histogram as plot_distribution,
)
from nets.gt_pyg import GraphTransformerNet
from accelerate import Accelerator
from torch.amp import autocast, GradScaler

matplotlib.use("WebAgg")


class Encoder(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        num_layers,
        pe_dim,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.net = GraphTransformerNet(
            node_dim_in=node_feat_dim,
            edge_dim_in=edge_dim,
            pe_in_dim=pe_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            norm="bn",
            num_gt_layers=num_layers,
            num_heads=8,
            act="relu",
            dropout=0.1,
        )

        self.fusion_layer = nn.Linear(output_dim + output_dim, output_dim)

    def forward(self, x, edge_attr, edge_index, pe, batch):
        encoded_x, encoded_edge_attr = self.net(
            x, edge_index, batch=batch, edge_attr=edge_attr, pe=pe
        )
        aggregated_edge_info = scatter_mean(
            encoded_edge_attr, edge_index[1], dim=0, dim_size=x.size(0)
        )
        fused_node_features = torch.cat([encoded_x, aggregated_edge_info], dim=-1)
        final_node_features_for_vq = self.fusion_layer(fused_node_features)
        return final_node_features_for_vq, encoded_edge_attr


class VQVAE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        dropout_ratio,
        activation,
        norm_type,
        codebook_size,
        lamb_edge,
        lamb_node,
        pe_dim=6,
        vq_decay=0.8,
        vq_commitment_weight=0.25,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.norm_type = norm_type
        self.codebook_size = codebook_size
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
        self.edge_dim = edge_dim

        self.encoder = Encoder(
            input_dim,
            hidden_dim,
            output_dim,
            edge_dim,
            num_layers,
            pe_dim,
        )
        self.vq = VectorQuantize(
            output_dim,
            codebook_size,
            decay=vq_decay,
            commitment_weight=vq_commitment_weight,
            use_cosine_sim=True,
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self.decoder = EGTDecoderRoPE(
            num_layers, output_dim, hidden_dim, input_dim, edge_dim
        )

        self.linear_fuse = nn.Linear(output_dim + output_dim, output_dim)

    def forward(self, h, edge_attr, edge_index, pe, batch):
        batch_size = batch.max() + 1
        mask = (batch.unsqueeze(-1) == batch.unsqueeze(-2)).float()

        # plot mask as hot image
        # plt.imshow(mask.cpu().squeeze().numpy())
        # plt.show()

        encoded_x, encoded_edge = self.encoder(h, edge_attr, edge_index, pe, batch)

        quantized, embed_ind, commit_loss, dist, codebook = self.vq(encoded_x)

        quantized_node, adj_quantized, mask_node = self.decoder(quantized, batch, mask)

        # feature_rec_loss = self.lamb_node * F.mse_loss(quantized_node, h)
        feature_rec_loss = self.lamb_node * F.cross_entropy(
            quantized_node, torch.argmax(h, dim=-1)
        )
        # feature_rec_loss = torch.tensor(0.0, device=h.device)  # disable feature loss

        # if self.adj_loss_type == "mse":
        #     edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))
        # elif self.adj_loss_type == "bce":
        #     edge_rec_loss = self.lamb_edge * F.binary_cross_entropy(adj_quantized, adj)

        gt_adj = torch.zeros_like(adj_quantized, device=adj_quantized.device)
        gt_adj[:, :, :, -1] = 0.9
        gt_adj[:, :, :, :-1] = to_dense_adj(
            edge_index, batch=batch, edge_attr=edge_attr
        )
        labels = torch.argmax(gt_adj, dim=-1)
        mask_edge = mask_node.unsqueeze(-1) & mask_node.unsqueeze(-2)

        diff_per_graph = (adj_quantized.argmax(dim=-1) != labels).reshape(
            gt_adj.shape[0], -1
        ).sum(dim=-1) / 2

        edge_rec_loss = self.lamb_edge * F.cross_entropy(
            adj_quantized[mask_edge], labels[mask_edge]
        )

        loss = feature_rec_loss + edge_rec_loss + commit_loss
        return (
            (loss, feature_rec_loss, edge_rec_loss, commit_loss),
            quantized_node,
            adj_quantized,
            diff_per_graph,
            torch.unique(embed_ind).tolist(),
        )

    def decode(self, quantized, batch):
        mask = (batch.unsqueeze(-1) == batch.unsqueeze(-2)).float()
        quantized_node, adj_quantized, mask_node = self.decoder(quantized, batch, mask)
        return quantized_node, adj_quantized, mask_node

    def encode(self, h, edge_attr, edge_index, pe, batch):
        encoded_x, encoded_edge = self.encoder(h, edge_attr, edge_index, pe, batch)
        quantized, embed_ind, commit_loss, dist, codebook = self.vq(encoded_x)
        return quantized, embed_ind, commit_loss, dist, codebook


def init_data(data, config_sample, test_data=None):
    if test_data is None:
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            data,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(config_sample["train"]["seed"]),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config_sample["train"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config_sample["train"]["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        return train_loader, test_loader
    else:
        train_loader = DataLoader(
            data,
            batch_size=config_sample["train"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=config_sample["train"]["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        return train_loader, test_loader


def testset_metrics(model, test_loader, accelerator):
    all_diffs_local = []
    all_x_diffs_local = []

    model.eval()

    unwrapped_model = accelerator.unwrap_model(model)

    for batch_graph in tqdm(
        test_loader, desc="Testing", disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            with autocast(device_type=accelerator.device.type):
                quantized, embed_ind, commit_loss, dist, codebook = (
                    unwrapped_model.encode(
                        batch_graph.x,
                        batch_graph.edge_attr,
                        batch_graph.edge_index,
                        batch_graph.pe,
                        batch_graph.batch,
                    )
                )
                quantized_node, adj_quantized, node_mask = unwrapped_model.decode(
                    quantized, batch_graph.batch
                )

            # Calculate node errors per graph on this process's data
            x_diff_per_graph_batch = scatter_sum(
                (quantized_node.argmax(dim=-1) != batch_graph.x.argmax(dim=-1)).int(),
                batch_graph.batch,
                dim=0,
            )
            all_x_diffs_local.append(x_diff_per_graph_batch)

            # Calculate edge errors per graph on this process's data
            gt_adj = torch.zeros_like(adj_quantized, device=adj_quantized.device)
            gt_adj[:, :, :, -1] = 0.9
            gt_adj[:, :, :, :-1] = to_dense_adj(
                batch_graph.edge_index,
                batch=batch_graph.batch,
                edge_attr=batch_graph.edge_attr,
            )
            gt_adj = gt_adj.argmax(dim=-1)
            adj_quantized = adj_quantized.argmax(dim=-1)

            diff_per_graph_batch = (adj_quantized != gt_adj).reshape(
                gt_adj.shape[0], -1
            ).sum(dim=-1) / 2
            all_diffs_local.append(diff_per_graph_batch)

    # Concatenate results from all batches on this process
    all_diffs_local_tensor = torch.cat(all_diffs_local, dim=0)
    all_x_diffs_local_tensor = torch.cat(all_x_diffs_local, dim=0)

    # 3. Gather results from all processes to the main process
    gathered_diffs = accelerator.gather(all_diffs_local_tensor)
    gathered_x_diffs = accelerator.gather(all_x_diffs_local_tensor)

    # 4. Only calculate final metrics and plot on the main process
    if accelerator.is_main_process:
        # Convert gathered tensors to numpy for calculations and plotting
        gathered_diffs_np = gathered_diffs.cpu().numpy()
        gathered_x_diffs_np = gathered_x_diffs.cpu().numpy()

        # Calculate average errors over the entire test set (all gathered data)
        test_set_edge_error = (
            sum(gathered_diffs_np) / len(gathered_diffs_np)
            if len(gathered_diffs_np) > 0
            else 0.0
        )
        test_set_node_error = (
            sum(gathered_x_diffs_np) / len(gathered_x_diffs_np)
            if len(gathered_x_diffs_np) > 0
            else 0.0
        )

        # Generate plots
        edge_error_dist_plot = plot_distribution(
            gathered_diffs_np, "Testset Edge Error Dist"
        )
        node_error_dist_plot = plot_distribution(
            gathered_x_diffs_np, "Testset Node Error Dist"
        )

        # 5. Return results only on the main process
        return (
            test_set_edge_error,
            test_set_node_error,
            edge_error_dist_plot,
            node_error_dist_plot,
        )
    else:
        # 5. Non-main processes return None
        return (None, None, None, None)


config_template = {
    "model": {
        "name": "VQVAE",
        "config": {
            "num_layers": 4,
            "input_dim": 4,
            "hidden_dim": 128,
            "output_dim": 128,
            "edge_dim": 3,
            "dropout_ratio": 0.1,
            "activation": "relu",
            "norm_type": "batch",
            "codebook_size": 8192,
            "lamb_edge": 1.0,
            "lamb_node": 1.0,
            "vq_decay": 0.8,
            "vq_commitment_weight": 0.25,
            "pe_dim": 6,
        },
    },
    "optimizer": {
        "name": "AdamW",
        "config": {
            "lr": 1e-4,
        },
    },
    "train": {
        "batch_size": 32 * 2,
        "num_epochs": 500,
        "ckpt_freq": 20,
        "seed": 42,
        "gradient_accumulation_steps": 1,
    },
    "dataset": {
        "name": "QM9",
    },
    "wandb_project": "MORGAN-VQVAE",
    "mixed_precision": "fp16",
    "device": "cuda",
}


def main():
    from accelerate.utils import DistributedDataParallelKwargs

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config_template["mixed_precision"], kwargs_handlers=[kwargs]
    )
    args = parse_args(config_template)
    config_sample = update_config(config_template, args)
    accelerator.print("Using Config: ", config_sample)
    setseed(config_sample["train"]["seed"])

    if accelerator.is_main_process:
        init_wandb(
            config_sample, tags=["VQVAE"], group=config_sample["dataset"]["name"]
        )
        # init save path
        runname = None
        if wandb.run.name is None or wandb.run.name == "":
            runname = generate_run_name(config_sample["model"]["name"])
        else:
            runname = wandb.run.name
        save_dir = setup_save_dir(
            config_sample["model"]["name"],
            f"./runs/vqvae/{config_sample['dataset']['name']}/",
            runname,
        )
    else:
        save_dir = None
        runname = None

    accelerator.print(f"Using device: {accelerator.device}")

    data = get_dataset(
        name=config_sample["dataset"]["name"],
        split="train",
        pe_dim=config_sample["model"]["config"]["pe_dim"],
    )
    test_data = get_dataset(
        name=config_sample["dataset"]["name"],
        split="test",
        pe_dim=config_sample["model"]["config"]["pe_dim"],
    )
    loader, test_loader = init_data(data, config_sample, test_data=test_data)

    model = VQVAE(**config_sample["model"]["config"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config_sample["optimizer"]["config"]["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config_sample["train"]["num_epochs"]
    )

    model, optimizer, loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, loader, test_loader, scheduler
    )

    scalar = GradScaler()

    for epoch in tqdm(
        range(config_sample["train"]["num_epochs"]),
        desc="Epoch Progress",
        disable=not accelerator.is_local_main_process,
    ):
        model.train()
        losses = []
        feature_rec_losses = []
        edge_rec_losses = []
        commit_losses = []
        diffs_per_graph = []
        codebook_used_inds = set()
        for step, batch_graph in enumerate(
            tqdm(
                loader,
                desc="Batch Progress",
                disable=not accelerator.is_local_main_process,
            )
        ):
            with autocast(device_type=accelerator.device.type):
                (
                    (loss, feature_rec_loss, edge_rec_loss, commit_loss),
                    quantized_node,
                    quantized_adj,
                    diff_per_graph,
                    unique_embed_inds,
                ) = model(
                    batch_graph.x,
                    batch_graph.edge_attr,
                    batch_graph.edge_index,
                    batch_graph.pe,
                    batch=batch_graph.batch,
                )

            accelerator.backward(loss)

            if (step + 1) % config_sample["train"]["gradient_accumulation_steps"] == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            feature_rec_losses.append(feature_rec_loss.item())
            edge_rec_losses.append(edge_rec_loss.item())
            commit_losses.append(commit_loss.item())
            diffs_per_graph.append(diff_per_graph)

            codebook_used_inds.update(unique_embed_inds)

        scheduler.step()

        diffs_per_graph = torch.cat(diffs_per_graph, dim=0).cpu().numpy()
        if accelerator.is_main_process:
            # log metrics in main process
            wandb.log(
                {
                    "loss": sum(losses) / len(losses),
                    "feature_rec_loss": sum(feature_rec_losses)
                    / len(feature_rec_losses),
                    "edge_rec_loss": sum(edge_rec_losses) / len(edge_rec_losses),
                    "commit_loss": sum(commit_losses) / len(commit_losses),
                    "train_edge_error_dist": plot_distribution(
                        diffs_per_graph, "Train Edge Error Dist"
                    ),
                    "codebook_usage": len(codebook_used_inds)
                    / config_sample["model"]["config"]["codebook_size"],
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            # save model in main process
            if epoch % config_sample["train"]["ckpt_freq"] == 0:
                save_model(
                    accelerator.unwrap_model(model),
                    epoch,
                    optimizer,
                    generate_save_path(
                        save_dir,
                        config_sample["model"]["name"],
                        epoch,
                    ),
                    ema=None,
                    config=config_sample,
                )
        if epoch % config_sample["train"]["ckpt_freq"] == 0:
            (
                test_set_edge_error,
                test_set_node_error,
                edge_error_dist,
                node_error_dist,
            ) = testset_metrics(model, test_loader, accelerator)
            if accelerator.is_main_process and test_set_edge_error is not None:
                wandb.log(
                    {
                        "edge_error_dist": edge_error_dist,
                        "node_error_dist": node_error_dist,
                        "test_set_edge_error": test_set_edge_error,
                        "test_set_node_error": test_set_node_error,
                    },
                    step=epoch,
                )

    if accelerator.is_main_process:
        wandb.finish()

    pass


if __name__ == "__main__":
    main()
