from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_adj
from load_data import get_dataset
from train_vqvae import VQVAE
import torch
import matplotlib
from utils.args import parse_args, update_config
from torch_geometric.loader import DataLoader
import utils.logger as log
from tqdm import tqdm
import os

matplotlib.use("WebAgg")


config_template = {
    "ckpt_path": "runs/vqvae/QM9/crisp-wave-7/VQVAE_epoch_580.pt",
    "seed": 42,
    "batch_size": 512,
    "device": "cuda",
    "save_path": "", # Will be updated based on ckpt_path
    "dataset": "", # Will be updated based on the checkpoint config
}


if __name__ == "__main__":
    args = parse_args(config_template)
    config_sample = update_config(config_template, args)
    log.info("Using Config: ", config_sample)

    device = torch.device(config_sample["device"])
    log.info("Using Device: ", str(device))

    ckpt = torch.load(config_sample["ckpt_path"], map_location=device)
    ckpt_config = ckpt["config"]

    dataset_name = ckpt_config["dataset"]["name"]
    config_sample["dataset"] = dataset_name
    dataset = get_dataset(name=dataset_name, split="train", pe_dim=ckpt_config["model"]["config"]["pe_dim"])
    loader = DataLoader(dataset, batch_size=config_sample["batch_size"], shuffle=False)

    model = VQVAE(**ckpt_config["model"]["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # get ckpt_path directory
    ckpt_dir = os.path.dirname(config_sample["ckpt_path"])
    log.info("save to ", ckpt_dir)
    config_sample["save_path"] = ckpt_dir

    all_x = []
    all_embed_ind = []
    slices = []

    all_valid_x = []
    all_valid_embed_ind = []
    all_valid_slices = []

    for batch_graph in tqdm(loader):
        batch_graph = batch_graph.to(device)

        with torch.no_grad():
            quantized, embed_ind, commit_loss, dist, codebook = model.encode(
                batch_graph.x,
                batch_graph.edge_attr,
                batch_graph.edge_index,
                batch_graph.pe,
                batch_graph.batch,
            )
            quantized_node, adj_quantized, mask_node = model.decode(
                quantized, batch=batch_graph.batch
            )

            x_diff = scatter_sum(
                (quantized_node.argmax(dim=-1) != batch_graph.x.argmax(dim=-1)).int(),
                batch_graph.batch,
                dim=0,
            )
            gt_adj = torch.zeros_like(adj_quantized, device=adj_quantized.device)
            gt_adj[:, :, :, -1] = 0.9
            gt_adj[:, :, :, :-1] = to_dense_adj(
                batch_graph.edge_index,
                batch=batch_graph.batch,
                edge_attr=batch_graph.edge_attr,
            )
            gt_adj = gt_adj.argmax(dim=-1)
            adj_quantized = adj_quantized.argmax(dim=-1)

            diff_per_graph = (adj_quantized != gt_adj).reshape(gt_adj.shape[0], -1).sum(
                dim=-1
            ) / 2
            recon_flags = torch.logical_and((x_diff == 0), (diff_per_graph == 0))

        for i in range(batch_graph.num_graphs):
            ind = batch_graph.batch == i
            graph_data = {
                "x": batch_graph.x[ind].cpu(),
                "embed_ind": embed_ind[ind].cpu(),
            }
            num_nodes = graph_data["x"].size(0)

            all_x.append(graph_data["x"])
            all_embed_ind.append(graph_data["embed_ind"])
            slices.append(num_nodes)

            if recon_flags[i]:
                all_valid_x.append(graph_data["x"])
                all_valid_embed_ind.append(graph_data["embed_ind"])
                all_valid_slices.append(num_nodes)

    x_tensor = torch.cat(all_x, dim=0)
    embed_ind_tensor = torch.cat(all_embed_ind, dim=0)
    slices_tensor = torch.tensor(slices, dtype=torch.long)

    codebook = model.vq.codebook.cpu().clone()

    processed_data = {
        "x": x_tensor,
        "embed_ind": embed_ind_tensor,
        "slices": slices_tensor,
        "codebook": codebook,
        "generation_config": config_sample,
        "is_reduced": False,
    }

    # Ensure the save path exists
    if not os.path.exists(config_sample["save_path"]):
        os.makedirs(config_sample["save_path"])

    torch.save(processed_data, f"{config_sample['save_path']}/{dataset_name}_vqdataset.pt")

    unique_indices = torch.unique(embed_ind_tensor)

    original_codebook_size = model.vq.codebook.size(0)
    mapping = torch.full((original_codebook_size,), -1, dtype=torch.long, device="cpu")
    mapping[unique_indices] = torch.arange(
        len(unique_indices), dtype=torch.long, device="cpu"
    )

    embed_ind_tensor_reduced = mapping[embed_ind_tensor]

    reduced_codebook = codebook[unique_indices]

    processed_data_reduced = {
        "x": x_tensor,
        "embed_ind": embed_ind_tensor_reduced,
        "slices": slices_tensor,
        "codebook": reduced_codebook,
        "generation_config": config_sample,
        "is_reduced": True,
    }

    save_path = os.path.join(config_sample["save_path"], f"{dataset_name}_vqdataset_reduced.pt")
    torch.save(processed_data_reduced, save_path)
    log.info(f"Reduced dataset saved to {save_path}")
    log.info(f"Original codebook size: {original_codebook_size}")
    log.info(f"Reduced codebook size: {len(unique_indices)}")

    # save the valid dataset
    valid_x_tensor = torch.cat(all_valid_x, dim=0)
    valid_embed_ind_tensor = torch.cat(all_valid_embed_ind, dim=0)
    valid_slices_tensor = torch.tensor(all_valid_slices, dtype=torch.long)
    valid_codebook = model.vq.codebook.cpu().clone()

    valid_processed_data = {
        "x": valid_x_tensor,
        "embed_ind": valid_embed_ind_tensor,
        "slices": valid_slices_tensor,
        "codebook": valid_codebook,
        "generation_config": config_sample,
        "is_reduced": False,
    }
    valid_save_path = os.path.join(
        config_sample["save_path"], f"{dataset_name}_vqdataset_valid.pt"
    )
    torch.save(valid_processed_data, valid_save_path)
    log.info(f"Valid dataset saved to {valid_save_path}")
    log.info(f"Valid dataset size: {len(all_valid_x)}")
    log.info(f"Drop ratio: {1 - len(all_valid_x) / len(all_x)}")

    # reduce the valid dataset
    valid_unique_indices = torch.unique(valid_embed_ind_tensor)
    valid_mapping = torch.full(
        (original_codebook_size,), -1, dtype=torch.long, device="cpu"
    )
    valid_mapping[valid_unique_indices] = torch.arange(
        len(valid_unique_indices), dtype=torch.long, device="cpu"
    )
    valid_embed_ind_tensor_reduced = valid_mapping[valid_embed_ind_tensor]
    valid_reduced_codebook = valid_codebook[valid_unique_indices]
    valid_processed_data_reduced = {
        "x": valid_x_tensor,
        "embed_ind": valid_embed_ind_tensor_reduced,
        "slices": valid_slices_tensor,
        "codebook": valid_reduced_codebook,
        "generation_config": config_sample,
        "is_reduced": True,
    }
    valid_reduced_save_path = os.path.join(
        config_sample["save_path"], f"{dataset_name}_vqdataset_valid_reduced.pt"
    )
    torch.save(valid_processed_data_reduced, valid_reduced_save_path)
    log.info(f"Valid reduced dataset saved to {valid_reduced_save_path}")
    log.info(f"Valid reduced codebook size: {len(valid_unique_indices)}")
