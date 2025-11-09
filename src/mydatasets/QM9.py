# part2_custom_qm9_dataset_fixed_maps.py

import os
import os.path as osp
from typing import Callable, List, Optional, Dict, Any, Set, Tuple, Union
import csv
import collections

import requests  # For downloading
import torch
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    # download_url, # Replaced with requests for more control potentially
)

# Utils for PE transform if used
from torch_geometric.utils import to_dense_adj, get_laplacian, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee


# Attempt to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT

    RDKitMol = Chem.rdchem.Mol  # For type hinting
except ImportError:
    Chem = None
    BT = None
    RDKitMol = Any  # type: ignore
    print("RDKit not found. Please install RDKit to use this dataset.")
    print("You can install it via pip: pip install rdkit-pypi")


class CustomQM9Dataset(InMemoryDataset):
    r"""
    A custom QM9 dataset implementation that loads data from a CSV file
    (containing SMILES, splits, and 19 target properties), which is
    downloaded from a URL.
    Hydrogen atoms are removed during graph construction.
    Node features are one-hot encoded heavy atom types based on a predefined map.
    Edge features are one-hot encoded bond types based on a predefined map.
    """

    DATASET_CSV_URL = "https://data.zzdirty.com/qm9_custom_splits_targets_cleaned.csv"
    RAW_FILENAME = "qm9_custom_splits_targets.csv"
    TARGET_SPLITS = ["train", "test"]

    PREDEFINED_HEAVY_ATOM_MAP: Dict[str, int] = {
        "C": 0,
        "N": 1,
        "O": 2,
        "F": 3,
    }

    PREDEFINED_BOND_TYPES_RDKIT: List[Any] = [
        BT.SINGLE,
        BT.DOUBLE,
        BT.TRIPLE,
        # BT.AROMATIC,
    ]

    def _get_processed_filename_for_split(self, split_name: str) -> str:
        return f"qm9_custom_{split_name}_processed_v2_fixedmaps.pt"  # Changed version

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if Chem is None or BT is None:
            raise ImportError(
                "RDKit is required to use CustomQM9Dataset. Please install it."
            )

        if split not in self.TARGET_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {self.TARGET_SPLITS}"
            )
        self.current_split_name = split

        # Use predefined maps
        self.atom_map: Dict[str, int] = self.PREDEFINED_HEAVY_ATOM_MAP
        self.num_atom_types: int = len(self.atom_map)

        self.bond_map: Dict[Any, int] = {
            bond_type: i for i, bond_type in enumerate(self.PREDEFINED_BOND_TYPES_RDKIT)
        }
        self.num_bond_types: int = len(self.bond_map)

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        # No need to load atom_map from file anymore as it's predefined.
        # Load data for the current split
        path_to_load = osp.join(
            self.processed_dir,
            self._get_processed_filename_for_split(self.current_split_name),
        )
        self.load(path_to_load)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.RAW_FILENAME]

    @property
    def processed_file_names(self) -> List[str]:
        # ATOM_MAP_FILENAME is removed as the map is now hardcoded
        return [
            self._get_processed_filename_for_split(self.current_split_name),
        ]

    def download(self) -> None:
        print(
            f"CustomQM9Dataset: Checking for {self.RAW_FILENAME} in {self.raw_dir} or downloading from {self.DATASET_CSV_URL}..."
        )
        raw_file_path = osp.join(self.raw_dir, self.RAW_FILENAME)

        # Only download if the file doesn't already exist in the raw directory
        if osp.exists(raw_file_path):
            print(
                f"{self.RAW_FILENAME} already exists in {self.raw_dir}. Skipping download."
            )
            return

        try:
            os.makedirs(self.raw_dir, exist_ok=True)  # Ensure raw_dir exists
            print(f"Attempting to download to {raw_file_path}")
            response = requests.get(
                self.DATASET_CSV_URL, stream=True, timeout=30
            )  # Added timeout
            response.raise_for_status()  # Raise an exception for HTTP errors

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with (
                open(raw_file_path, "wb") as f,
                tqdm(
                    desc=self.RAW_FILENAME,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete.")
        except requests.exceptions.HTTPError as e_http:
            print(f"HTTP error downloading {self.RAW_FILENAME}: {e_http}")
            if osp.exists(raw_file_path):
                os.remove(raw_file_path)
            raise IOError(
                f"Failed to download {self.RAW_FILENAME} due to HTTP error."
            ) from e_http
        except requests.exceptions.RequestException as e_req:
            print(f"Network error downloading {self.RAW_FILENAME}: {e_req}")
            if osp.exists(raw_file_path):
                os.remove(raw_file_path)  # Clean up partial
            raise IOError(
                f"Failed to download {self.RAW_FILENAME} due to network error."
            ) from e_req
        except Exception as e:
            print(f"An unexpected error during download of {self.RAW_FILENAME}: {e}")
            if osp.exists(raw_file_path):
                os.remove(raw_file_path)  # Clean up partial
            raise IOError(f"Download failed due to an unexpected error.") from e

    def process(self) -> None:
        print(
            f"CustomQM9Dataset: Processing {self.RAW_FILENAME} to generate graph data using predefined maps..."
        )
        csv_path = osp.join(self.raw_dir, self.RAW_FILENAME)
        if not osp.exists(csv_path):
            raise FileNotFoundError(
                f"Raw CSV file {csv_path} not found. "
                "It should have been downloaded by the `download` method. "
                "Please check the URL or manually place the file and re-run, or try `force_reload=True`."
            )

        # Atom map is now predefined, no need to build or save it.
        print(
            f"Using predefined heavy atom map with {self.num_atom_types} types: {self.atom_map}"
        )
        print(f"Using predefined bond map with {self.num_bond_types} types.")

        raw_data_by_split: Dict[str, List[Tuple[str, List[float]]]] = (
            collections.defaultdict(list)
        )
        property_names_in_order = [f"prop_{i}" for i in range(19)]

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Reading CSV"):
                smiles = row["smiles"].strip()
                split_from_csv = row["split"].strip()
                if not smiles or split_from_csv not in self.TARGET_SPLITS:
                    continue
                try:
                    targets = [
                        float(row[prop_name]) for prop_name in property_names_in_order
                    ]
                    raw_data_by_split[split_from_csv].append((smiles, targets))
                except (ValueError, KeyError) as e:
                    print(
                        f"Warning: Error parsing row for SMILES {smiles}: {e}. Skipping."
                    )
                    continue

        print("CSV reading complete. Starting graph conversion...")

        for target_split_name in self.TARGET_SPLITS:
            print(
                f"Converting molecules to graph Data objects for split: {target_split_name}..."
            )
            data_list_for_current_split = []
            items_to_convert = raw_data_by_split[target_split_name]

            if not items_to_convert:
                print(f"No molecules to process for split '{target_split_name}'.")

            for original_smiles, targets_list_float in tqdm(
                items_to_convert, desc=f"Converting {target_split_name}"
            ):
                mol = Chem.MolFromSmiles(original_smiles)
                if mol is None:
                    print(
                        f"Warning: Could not parse SMILES: {original_smiles} (split: {target_split_name}). Skipping."
                    )
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    print(
                        f"Warning: Sanitization failed for SMILES: {original_smiles} (split: {target_split_name}). Error: {e}. Skipping."
                    )
                    continue

                Chem.Kekulize(mol)

                node_feature_vectors = []
                original_to_new_idx_map: Dict[int, int] = {}
                new_idx_counter = 0
                valid_molecule_for_features = True

                for atom_idx_orig, atom in enumerate(mol.GetAtoms()):
                    if atom.GetAtomicNum() == 1:
                        continue  # Skip hydrogen

                    original_to_new_idx_map[atom_idx_orig] = new_idx_counter
                    new_idx_counter += 1

                    atom_symbol = atom.GetSymbol()
                    atom_type_mapped_idx = self.atom_map.get(atom_symbol)
                    if atom_type_mapped_idx is None:
                        print(
                            f"Warning: Atom symbol '{atom_symbol}' in SMILES '{original_smiles}' "
                            f"is not in the PREDEFINED_HEAVY_ATOM_MAP: {list(self.atom_map.keys())}. Skipping molecule."
                        )
                        valid_molecule_for_features = False
                        break
                    node_feature_vectors.append(
                        torch.nn.functional.one_hot(
                            torch.tensor(atom_type_mapped_idx),
                            num_classes=self.num_atom_types,
                        )
                    )

                if not valid_molecule_for_features or not node_feature_vectors:
                    if (
                        valid_molecule_for_features
                    ):  # Implies only H atoms or empty after H removal
                        print(
                            f"Warning: No heavy atom features for SMILES: {original_smiles}. Skipping."
                        )
                    continue

                x = torch.stack(node_feature_vectors, dim=0).float()
                num_heavy_atoms = x.shape[0]

                edge_indices_list = []
                edge_attrs_list = []

                for bond in mol.GetBonds():
                    start_atom_orig_idx = bond.GetBeginAtomIdx()
                    end_atom_orig_idx = bond.GetEndAtomIdx()

                    if (
                        start_atom_orig_idx in original_to_new_idx_map
                        and end_atom_orig_idx in original_to_new_idx_map
                    ):
                        new_start_idx = original_to_new_idx_map[start_atom_orig_idx]
                        new_end_idx = original_to_new_idx_map[end_atom_orig_idx]
                        bond_type_rdkit = bond.GetBondType()
                        bond_type_mapped_idx = self.bond_map.get(bond_type_rdkit)

                        if bond_type_mapped_idx is None:
                            # This bond type is not in our PREDEFINED_BOND_TYPES_RDKIT
                            # For QM9, this should be rare with the standard set.
                            print(
                                f"Warning: Bond type {bond_type_rdkit} in SMILES {original_smiles} "
                                f"is not in PREDEFINED_BOND_TYPES_RDKIT. Skipping bond."
                            )
                            continue

                        one_hot_bond_attr = torch.nn.functional.one_hot(
                            torch.tensor(bond_type_mapped_idx),
                            num_classes=self.num_bond_types,
                        ).float()

                        edge_indices_list.extend(
                            [(new_start_idx, new_end_idx), (new_end_idx, new_start_idx)]
                        )
                        edge_attrs_list.extend([one_hot_bond_attr, one_hot_bond_attr])

                if edge_indices_list:
                    edge_index = (
                        torch.tensor(edge_indices_list, dtype=torch.long)
                        .t()
                        .contiguous()
                    )
                    edge_attr = torch.stack(edge_attrs_list, dim=0)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, self.num_bond_types), dtype=torch.float)

                y = torch.tensor(targets_list_float, dtype=torch.float)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    smiles=original_smiles,
                    split=target_split_name,
                    num_heavy_atoms=num_heavy_atoms,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list_for_current_split.append(data)

            processed_file_path = osp.join(
                self.processed_dir,
                self._get_processed_filename_for_split(target_split_name),
            )
            if not data_list_for_current_split and items_to_convert:
                print(
                    f"Warning: No Data objects created for split '{target_split_name}'. Check SMILES, filters, or atom/bond maps."
                )

            self.save(data_list_for_current_split, processed_file_path)
            print(
                f"Processed data for split '{target_split_name}' saved to {processed_file_path} ({len(data_list_for_current_split)} graphs)"
            )

        print("All target splits processed and saved.")

    def get_atom_type_map(self) -> Dict[str, int]:
        return self.atom_map  # Returns the predefined map

    def get_bond_type_map(self) -> Dict[Any, int]:
        return self.bond_map  # Returns the predefined map

    def get_num_atom_features(self) -> int:
        return self.num_atom_types  # Based on predefined map

    def get_num_edge_features(self) -> int:
        return self.num_bond_types  # Based on predefined map

    def get_current_split_name(self) -> str:
        return self.current_split_name

    def len(self) -> int:
        return super().len()

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        # Ensure split attribute is present, though it should be from processing
        if not hasattr(data, "split") or data.split is None:
            data.split = self.current_split_name
        return data


def rcm_reorder(data):
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    perm = reverse_cuthill_mckee(adj_matrix)[::-1]

    perm_tensor = torch.from_numpy(perm.copy()).long()
    inv_perm = torch.from_numpy(np.argsort(perm).copy()).long()

    data.x = data.x[perm_tensor]
    data.edge_index = inv_perm[data.edge_index]

    if "pe" in data.keys():
        data.pe = data.pe[perm_tensor]

    return data, perm


def transform_pe_reorder(data, pe_dim=6):
    if data.num_nodes > 1:
        data, perm = rcm_reorder(data)

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

    return data


# Example Usage (main block):
if __name__ == "__main__":
    QM9_CUSTOM_ROOT = "data/QM9_Custom_FixedMaps"

    # Ensure RDKit is available for the main script execution
    if Chem is None:
        print(
            "RDKit is not installed. Cannot run the main example. Please install RDKit."
        )
    else:
        try:
            print(f"\n--- Loading Custom QM9 Training Set (Fixed Maps) ---")
            # Set force_reload=True for the very first run for a given root directory
            # or if you change processing logic significantly.
            train_dataset = CustomQM9Dataset(
                root=QM9_CUSTOM_ROOT,
                split="train",
                force_reload=False,  # Set to True for initial processing or reprocessing
                pre_transform=transform_pe_reorder,  # Using the PE transform
            )
            print(f"Number of graphs in training set: {len(train_dataset)}")
            if len(train_dataset) > 0:
                sample_train_data = train_dataset[0]
                print("\nSample Training Data (after pre_transform):")
                print(f"  SMILES: {sample_train_data.smiles}")
                print(f"  Split: {sample_train_data.split}")
                print(f"  y shape: {sample_train_data.y.shape}")
                print(f"  x shape: {sample_train_data.x.shape}")
                print(f"  edge_index shape: {sample_train_data.edge_index.shape}")
                print(f"  edge_attr shape: {sample_train_data.edge_attr.shape}")
                if hasattr(sample_train_data, "pe"):
                    print(f"  pe shape: {sample_train_data.pe.shape}")
                print(f"  Num heavy atoms: {sample_train_data.num_heavy_atoms}")
                print(
                    f"  Node feature dim (atom types): {train_dataset.get_num_atom_features()}"
                )
                print(
                    f"  Edge feature dim (bond types): {train_dataset.get_num_edge_features()}"
                )
                print(f"  Atom map (predefined): {train_dataset.get_atom_type_map()}")

            print(f"\n--- Loading Custom QM9 Test Set (Fixed Maps) ---")
            test_dataset = CustomQM9Dataset(
                root=QM9_CUSTOM_ROOT,
                split="test",
                force_reload=False,
                pre_transform=transform_pe_reorder,  # Apply same transform
            )
            print(f"Number of graphs in test set: {len(test_dataset)}")
            if len(test_dataset) > 0:
                sample_test_data = test_dataset[0]
                print("\nSample Test Data (after pre_transform):")
                print(f"  SMILES: {sample_test_data.smiles}")
                print(f"  Split: {sample_test_data.split}")
                print(f"  y shape: {sample_test_data.y.shape}")
                if hasattr(sample_test_data, "pe"):
                    print(f"  pe shape: {sample_test_data.pe.shape}")

        except ImportError as e_imp:
            print(f"ImportError in main: {e_imp}")
        except FileNotFoundError as e_fnf:
            print(f"FileNotFoundError in main: {e_fnf}")
            print(
                f"Ensure CSV is at {CustomQM9Dataset.DATASET_CSV_URL} and downloadable, or manually place "
                f"'{CustomQM9Dataset.RAW_FILENAME}' in '{osp.join(QM9_CUSTOM_ROOT, 'raw')}' "
                "if download fails and run with force_reload=True once."
            )
        except Exception as e_main:
            print(f"An error occurred in the main example usage: {e_main}")
            import traceback

            traceback.print_exc()
