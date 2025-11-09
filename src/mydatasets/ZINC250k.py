import os
import os.path as osp
import random
from typing import Callable, List, Optional, Dict, Any, Tuple
import csv
import collections

import requests  # For downloading
import torch
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
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


class CustomZINCDataset(InMemoryDataset):
    r"""
    A custom ZINC dataset implementation that loads data from a CSV file
    (containing SMILES and 3 target properties: logP, qed, SAS), which is
    downloaded from a URL.

    Since the original CSV does not contain data splits, this class
    deterministically creates an 80/20 train/test split using a fixed
    random seed.

    Hydrogen atoms are removed during graph construction.
    Node features are one-hot encoded heavy atom types based on a predefined map.
    Edge features are one-hot encoded bond types based on a predefined map.
    """

    DATASET_CSV_URL = (
        "https://raw.githubusercontent.com/harryjo97/GDSS/master/data/zinc250k.csv"
    )
    RAW_FILENAME = "zinc250k.csv"
    TARGET_SPLITS = ["train", "test"]

    # ZINC contains a wider variety of atoms than QM9
    PREDEFINED_HEAVY_ATOM_MAP: Dict[str, int] = {
        "C": 0,
        "N": 1,
        "O": 2,
        "F": 3,
        "S": 4,
        "Cl": 5,
        "Br": 6,
        "I": 7,
        "P": 8,
    }

    # Aromatic bonds are common in ZINC
    PREDEFINED_BOND_TYPES_RDKIT: List[Any] = [
        BT.SINGLE,
        BT.DOUBLE,
        BT.TRIPLE,
    ]

    def _get_processed_filename_for_split(self, split_name: str) -> str:
        return f"zinc_custom_{split_name}_processed_v1.pt"

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
                "RDKit is required to use CustomZINCDataset. Please install it."
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
        # Return processed files for all splits so `process` runs only if any is missing
        return [self._get_processed_filename_for_split(s) for s in self.TARGET_SPLITS]

    def download(self) -> None:
        print(
            f"CustomZINCDataset: Checking for {self.RAW_FILENAME} in {self.raw_dir} or downloading..."
        )
        raw_file_path = osp.join(self.raw_dir, self.RAW_FILENAME)

        if osp.exists(raw_file_path):
            print(f"{self.RAW_FILENAME} already exists. Skipping download.")
            return

        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            print(
                f"Attempting to download from {self.DATASET_CSV_URL} to {raw_file_path}"
            )
            response = requests.get(self.DATASET_CSV_URL, stream=True, timeout=60)
            response.raise_for_status()

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
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {self.RAW_FILENAME}: {e}")
            if osp.exists(raw_file_path):
                os.remove(raw_file_path)
            raise IOError(f"Failed to download {self.RAW_FILENAME}.") from e

    def process(self) -> None:
        print(
            f"CustomZINCDataset: Processing {self.RAW_FILENAME} to generate graph data..."
        )
        csv_path = osp.join(self.raw_dir, self.RAW_FILENAME)
        if not osp.exists(csv_path):
            raise FileNotFoundError(f"Raw CSV file {csv_path} not found.")

        print(f"Using predefined heavy atom map with {self.num_atom_types} types.")
        print(f"Using predefined bond map with {self.num_bond_types} types.")

        # Read all data from CSV first
        all_raw_data: List[Tuple[str, List[float]]] = []
        property_names_in_order = ["logP", "qed", "SAS"]

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Reading ZINC CSV"):
                smiles = row["smiles"].strip()
                if not smiles:
                    continue
                try:
                    targets = [
                        float(row[prop_name]) for prop_name in property_names_in_order
                    ]
                    all_raw_data.append((smiles, targets))
                except (ValueError, KeyError) as e:
                    print(
                        f"Warning: Error parsing row for SMILES {smiles}: {e}. Skipping."
                    )
                    continue

        # --- Deterministic Train/Test Split ---
        print("Creating deterministic 80/20 train/test split...")
        num_molecules = len(all_raw_data)
        indices = list(range(num_molecules))
        # Use a fixed seed to ensure the split is the same every time
        random.seed(42)
        random.shuffle(indices)

        split_idx = int(num_molecules * 0.8)
        train_indices = set(indices[:split_idx])
        test_indices = set(indices[split_idx:])

        print(
            f"Total molecules: {num_molecules}. Train: {len(train_indices)}, Test: {len(test_indices)}."
        )

        raw_data_by_split: Dict[str, List[Tuple[str, List[float]]]] = (
            collections.defaultdict(list)
        )
        for i, (smiles, targets) in enumerate(all_raw_data):
            if i in train_indices:
                raw_data_by_split["train"].append((smiles, targets))
            elif i in test_indices:
                raw_data_by_split["test"].append((smiles, targets))

        print("CSV reading and splitting complete. Starting graph conversion...")

        for target_split_name in self.TARGET_SPLITS:
            print(
                f"Converting molecules to graph Data objects for split: {target_split_name}..."
            )
            data_list_for_current_split = []
            items_to_convert = raw_data_by_split[target_split_name]

            for original_smiles, targets_list_float in tqdm(
                items_to_convert, desc=f"Converting {target_split_name}"
            ):
                mol = Chem.MolFromSmiles(original_smiles)
                if mol is None:
                    continue

                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue

                # It is recommended to kekulize molecules for graph processing
                Chem.Kekulize(mol)

                node_feature_vectors = []
                original_to_new_idx_map: Dict[int, int] = {}
                new_idx_counter = 0
                valid_molecule = True

                for atom_idx_orig, atom in enumerate(mol.GetAtoms()):
                    if atom.GetAtomicNum() == 1:
                        continue  # Skip H

                    original_to_new_idx_map[atom_idx_orig] = new_idx_counter
                    new_idx_counter += 1

                    atom_symbol = atom.GetSymbol()
                    atom_type_mapped_idx = self.atom_map.get(atom_symbol)
                    if atom_type_mapped_idx is None:
                        print(
                            f"Warning: Atom '{atom_symbol}' in SMILES '{original_smiles}' not in map. Skipping."
                        )
                        valid_molecule = False
                        break
                    node_feature_vectors.append(
                        torch.nn.functional.one_hot(
                            torch.tensor(atom_type_mapped_idx),
                            num_classes=self.num_atom_types,
                        )
                    )

                if not valid_molecule or not node_feature_vectors:
                    continue

                x = torch.stack(node_feature_vectors, dim=0).float()
                num_heavy_atoms = x.shape[0]

                edge_indices_list = []
                edge_attrs_list = []
                for bond in mol.GetBonds():
                    start_idx_orig = bond.GetBeginAtomIdx()
                    end_idx_orig = bond.GetEndAtomIdx()

                    if (
                        start_idx_orig in original_to_new_idx_map
                        and end_idx_orig in original_to_new_idx_map
                    ):
                        new_start_idx = original_to_new_idx_map[start_idx_orig]
                        new_end_idx = original_to_new_idx_map[end_idx_orig]
                        bond_type_rdkit = bond.GetBondType()
                        bond_type_mapped_idx = self.bond_map.get(bond_type_rdkit)

                        if bond_type_mapped_idx is None:
                            # This should be rare with the current bond map
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

                # ZINC has 3 regression targets
                y = torch.tensor(targets_list_float, dtype=torch.float).view(1, -1)

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
            self.save(data_list_for_current_split, processed_file_path)
            print(
                f"Processed data for '{target_split_name}' saved to {processed_file_path} ({len(data_list_for_current_split)} graphs)"
            )

    def get_atom_type_map(self) -> Dict[str, int]:
        return self.atom_map

    def get_bond_type_map(self) -> Dict[Any, int]:
        return self.bond_map

    def get_num_atom_features(self) -> int:
        return self.num_atom_types

    def get_num_edge_features(self) -> int:
        return self.num_bond_types

    def get_current_split_name(self) -> str:
        return self.current_split_name


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
            print(
                f"Error: NumPy linalg.eigh also failed for graph with {num_nodes} nodes. Returning zero PE."
            )
            data.pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)
            return data

    k = min(pe_dim + 1, num_nodes)
    selected_eig_vecs = eig_vecs[:, 1:k]

    if selected_eig_vecs.shape[1] < pe_dim:
        padding_needed = pe_dim - selected_eig_vecs.shape[1]
        padding = torch.zeros(
            num_nodes, padding_needed, device=eig_vecs.device, dtype=torch.float
        )
        pe = torch.cat([selected_eig_vecs, padding], dim=1)
    else:
        pe = selected_eig_vecs

    assert pe.shape[1] == pe_dim, f"PE shape mismatch: {pe.shape[0]} != {num_nodes}"

    if pe.shape[1] > 0:
        mean = pe.mean(dim=0, keepdim=True)
        std = pe.std(dim=0, keepdim=True)
        pe = (pe - mean) / (std + 1e-6)
        pe = torch.nan_to_num(pe, nan=0.0)
    data.pe = pe.float()

    return data

def transform_pe_reorder_custom_reorder(data, reorder_fun, pe_dim=6):
    if data.num_nodes > 1:
        data, perm = reorder_fun(data)

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
            print(
                f"Error: NumPy linalg.eigh also failed for graph with {num_nodes} nodes. Returning zero PE."
            )
            data.pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)
            return data

    k = min(pe_dim + 1, num_nodes)
    selected_eig_vecs = eig_vecs[:, 1:k]

    if selected_eig_vecs.shape[1] < pe_dim:
        padding_needed = pe_dim - selected_eig_vecs.shape[1]
        padding = torch.zeros(
            num_nodes, padding_needed, device=eig_vecs.device, dtype=torch.float
        )
        pe = torch.cat([selected_eig_vecs, padding], dim=1)
    else:
        pe = selected_eig_vecs

    assert pe.shape[1] == pe_dim, f"PE shape mismatch: {pe.shape[0]} != {num_nodes}"

    if pe.shape[1] > 0:
        mean = pe.mean(dim=0, keepdim=True)
        std = pe.std(dim=0, keepdim=True)
        pe = (pe - mean) / (std + 1e-6)
        pe = torch.nan_to_num(pe, nan=0.0)
    data.pe = pe.float()

    return data

# Example Usage
if __name__ == "__main__":
    ZINC_CUSTOM_ROOT = "data/ZINC_Custom"

    if Chem is None:
        print("RDKit is not installed. Cannot run the main example.")
    else:
        try:
            print(f"\n--- Loading Custom ZINC Training Set ---")
            # Set force_reload=True on the first run or if you change processing logic
            train_dataset = CustomZINCDataset(
                root=ZINC_CUSTOM_ROOT,
                split="train",
                force_reload=False,  # Set to True for initial processing
                pre_transform=transform_pe_reorder,
            )
            print(f"Number of graphs in training set: {len(train_dataset)}")
            if len(train_dataset) > 0:
                sample_train_data = train_dataset[0]
                print("\nSample Training Data (after pre_transform):")
                print(f"  SMILES: {sample_train_data.smiles}")
                print(f"  Split: {sample_train_data.split}")
                print(f"  y shape: {sample_train_data.y.shape} (logP, qed, SAS)")
                print(f"  y values: {sample_train_data.y}")
                print(f"  x shape: {sample_train_data.x.shape}")
                print(f"  edge_index shape: {sample_train_data.edge_index.shape}")
                print(f"  edge_attr shape: {sample_train_data.edge_attr.shape}")
                if hasattr(sample_train_data, "pe"):
                    print(f"  pe shape: {sample_train_data.pe.shape}")
                print(
                    f"  Node feature dim (atom types): {train_dataset.get_num_atom_features()}"
                )
                print(
                    f"  Edge feature dim (bond types): {train_dataset.get_num_edge_features()}"
                )
                print(f"  Atom map (predefined): {train_dataset.get_atom_type_map()}")

            print(f"\n--- Loading Custom ZINC Test Set ---")
            test_dataset = CustomZINCDataset(
                root=ZINC_CUSTOM_ROOT,
                split="test",
                force_reload=False,
                pre_transform=transform_pe_reorder,
            )
            print(f"Number of graphs in test set: {len(test_dataset)}")
            if len(test_dataset) > 0:
                sample_test_data = test_dataset[10]  # Get a different sample
                print("\nSample Test Data (after pre_transform):")
                print(f"  SMILES: {sample_test_data.smiles}")
                print(f"  Split: {sample_test_data.split}")
                print(f"  y shape: {sample_test_data.y.shape}")
                if hasattr(sample_test_data, "pe"):
                    print(f"  pe shape: {sample_test_data.pe.shape}")

        except (IOError, FileNotFoundError) as e:
            print(f"\nAn error occurred: {e}")
            print(
                f"Please ensure '{CustomZINCDataset.RAW_FILENAME}' is downloadable or manually "
                f"place it in '{osp.join(ZINC_CUSTOM_ROOT, 'raw')}' and run with force_reload=True."
            )
        except Exception as e:
            print(f"An unexpected error occurred in the main example: {e}")
            import traceback

            traceback.print_exc()
