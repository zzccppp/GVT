import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem
import requests
from typing import Tuple, List
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, get_laplacian
from utils.graphs import rcm_reorder_zinc
import numpy as np


def zinc_transform_pe(data, pe_dim=6):
    x = data.x.long()
    data.x = F.one_hot(x, num_classes=28).squeeze(1).float()
    edge_attr = data.edge_attr.long()
    edge_attr = edge_attr - 1
    data.edge_attr = F.one_hot(edge_attr, num_classes=3).squeeze(1).float()

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    L = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_dense_adj(L[0], edge_attr=L[1]).squeeze(0)

    try:
        eig_vals, eig_vecs = torch.linalg.eigh(L)
    except torch.linalg.LinAlgError:
        print(
            f"Warning: torch.linalg.eigh failed for graph with {num_nodes} nodes. Falling back to NumPy."
        )
        try:
            L_np = L.cpu().numpy()
            eig_vals_np, eig_vecs_np = np.linalg.eigh(L_np)
            eig_vals = torch.from_numpy(eig_vals_np).to(L.device).float()
            eig_vecs = torch.from_numpy(eig_vecs_np).to(L.device).float()
        except np.linalg.LinAlgError:
            # 如果 NumPy 也失败，返回零 PE
            print(
                f"Error: NumPy linalg.eigh also failed for graph with {num_nodes} nodes. Returning zero PE."
            )
            data.pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)
            return data

    k = min(pe_dim + 1, num_nodes)  # 我们最多能获取 num_nodes 个特征向量
    selected_eig_vecs = eig_vecs[:, 1:k]

    # 则需要用零向量进行填充
    if selected_eig_vecs.shape[1] < pe_dim:
        padding_needed = pe_dim - selected_eig_vecs.shape[1]
        padding = torch.zeros(
            num_nodes, padding_needed, device=eig_vecs.device, dtype=torch.float
        )
        pe = torch.cat([selected_eig_vecs, padding], dim=1)
    else:
        pe = selected_eig_vecs

    # 确保输出维度总是 (num_nodes, pe_dim)
    assert pe.shape[1] == pe_dim, f"PE shape mismatch: {pe.shape[0]} != {num_nodes}"

    if pe.shape[1] > 0:
        # 计算每个 PE 维度的均值和标准差
        mean = pe.mean(dim=0, keepdim=True)
        std = pe.std(dim=0, keepdim=True)
        # 标准化，加上一个小的 epsilon 防止除以零
        pe = (pe - mean) / (std + 1e-6)
        # 处理可能因 std=0 产生的 NaN 值
        pe = torch.nan_to_num(pe, nan=0.0)
    data.pe = pe.float()  # 确保是 float 类型

    # 重新排序数据
    data, perm = rcm_reorder_zinc(data)
    data.perm = perm

    return data


class ZincCSVDataset(Dataset):
    csv_url = "https://data.zzdirty.com/250k_rndm_zinc_drugs_clean_3.csv"
    filename = "zinc_drugs.csv"

    def __init__(self, root: str, force_download: bool = False):
        self.root = root
        self.filepath = os.path.join(self.root, self.filename)
        self.processed_dir = os.path.join(
            self.root, self.__class__.__name__ + "_processed"
        )
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_file = os.path.join(self.processed_dir, "data.pt")
        self.smiles_list: List[str] = []
        self.mols_list: List[Chem.Mol] = []

        self._download_and_process(force_download)

    def _download_and_process(self, force_download: bool):
        if not os.path.exists(self.filepath) or force_download:
            print(f"Downloading CSV file from {self.csv_url} to {self.filepath}...")
            response = requests.get(self.csv_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(self.filepath, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("Download complete.")
        else:
            print(f"CSV file already exists at {self.filepath}.")

        if not os.path.exists(self.processed_file):
            print("Processing SMILES and converting to Mols...")
            df = pd.read_csv(self.filepath)
            if "smiles" not in df.columns:
                raise ValueError("CSV file must contain a column named 'smiles'.")

            for index, row in tqdm(df.iterrows()):
                smiles = row["smiles"]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    self.smiles_list.append(smiles)
                    self.mols_list.append(mol)
                else:
                    print(
                        f"Warning: Could not parse SMILES: {smiles} at index {index}."
                    )

            torch.save((self.smiles_list, self.mols_list), self.processed_file)
            print(
                f"Processed and saved {len(self.smiles_list)} samples to {self.processed_file}."
            )
        else:
            print(f"Loading processed data from {self.processed_file}.")
            self.smiles_list, self.mols_list = torch.load(self.processed_file)

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tuple[str, Chem.Mol]:
        return self.smiles_list[idx], self.mols_list[idx]


# 示例用法
if __name__ == "__main__":
    zinc_dataset = ZincCSVDataset(root="data/zinc_smiles", force_download=False)

    print(f"数据集大小: {len(zinc_dataset)}")

    # 获取一个样本
    smiles, mol = zinc_dataset[0]
    print(f"第一个样本的SMILES: {smiles}")
    print(f"第一个样本的Mol对象: {mol}")
