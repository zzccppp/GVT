from collections import defaultdict
import os
import os.path as osp
from typing import Callable, List, Optional, Dict, Any, Set, Tuple, Union
import csv  # For reading CSV files

import torch
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
)
from torch_geometric.utils import to_dense_adj, get_laplacian
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Attempt to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT

    RDKitMol = Chem.rdchem.Mol  # For type hinting
except ImportError:
    Chem = None
    BT = None
    RDKitMol = None  # type: ignore
    print("RDKit not found. Please install RDKit to use this dataset.")
    print("You can install it via pip: pip install rdkit-pypi")
    print("Or via conda: conda install -c conda-forge rdkit")


class MOSESDataset(InMemoryDataset):
    r"""The MOSES dataset, consisting of SMILES strings and their splits.
    The dataset is downloaded as a CSV file and processed into graph structures.
    Data for each predefined split ('train', 'test', 'test_scaffolds') is stored
    in separate processed files. An instance of this class loads data for one specific split.

    SMILES are processed once. Atom type mapping is dynamically generated from the dataset
    and stored globally. Node features are one-hot encoded atom types.

    Bond types are fixed (SINGLE, DOUBLE, TRIPLE, AROMATIC).
    Edge features are one-hot encoded bond types.

    Each data object will have an additional attribute `split` indicating
    which of the predefined splits it belongs to.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str): The specific split to load. Must be one of 'train', 'test',
            or 'test_scaffolds'.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk during the `process` phase. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset (applied during `process` phase). (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            If True, the `process` method will be executed even if processed
            files for all splits exist, regenerating them. (default: :obj:`False`)
    """

    URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
    RAW_FILENAME = "moses_dataset_v1.csv"
    ATOM_MAP_FILENAME = "moses_atom_map_v1.pt"
    TARGET_SPLITS = ["train", "test", "test_scaffolds"]

    # BOND_TYPES_RDKIT = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
    BOND_TYPES_RDKIT = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE]

    def _get_processed_filename_for_split(self, split_name: str) -> str:
        return f"moses_{split_name}_processed_data_v1.pt"

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if Chem is None or BT is None or RDKitMol is None:
            raise ImportError(
                "RDKit is required to use the MOSESDataset. Please install it."
            )

        if split not in self.TARGET_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {self.TARGET_SPLITS}"
            )
        self.current_split_name = split

        self.bond_map: Dict[Any, int] = {
            bond_type: i for i, bond_type in enumerate(self.BOND_TYPES_RDKIT)
        }
        self.num_bond_types: int = len(self.bond_map)

        self.atom_map: Dict[str, int] = {}
        self.num_atom_types: int = 0

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        # Load atom_map (common for all splits)
        # This occurs after super().__init__(), which calls self.process() if necessary.
        # If process() was called, self.atom_map is already populated.
        # If not (files existed), we load it here.
        atom_map_path = osp.join(self.processed_dir, self.ATOM_MAP_FILENAME)
        if not self.atom_map and osp.exists(
            atom_map_path
        ):  # Check if not already set by process()
            try:
                loaded_atom_map = torch.load(atom_map_path)
                if isinstance(loaded_atom_map, dict):
                    self.atom_map = loaded_atom_map
                else:
                    # For potential backward compatibility if it was saved as (map, splits_set)
                    # Or if format is unexpected.
                    print(
                        f"Warning: Atom map file at {atom_map_path} was not a simple dict. Trying to extract dict."
                    )
                    if (
                        isinstance(loaded_atom_map, tuple)
                        and len(loaded_atom_map) > 0
                        and isinstance(loaded_atom_map[0], dict)
                    ):
                        self.atom_map = loaded_atom_map[0]
                    else:
                        raise ValueError(
                            f"Unexpected format in atom map file: {atom_map_path}. Expected a dict."
                        )
                self.num_atom_types = len(self.atom_map)
            except Exception as e:
                raise RuntimeError(
                    f"Error loading atom map file {atom_map_path}: {e}. Consider force_reload=True."
                )

        elif not osp.exists(atom_map_path) and not force_reload:
            # This implies data for the split might exist, but atom_map is missing.
            # process() should have created it.
            raise RuntimeError(
                f"Atom map file not found at {atom_map_path} but not forcing reload. "
                "This indicates an inconsistent state. Consider using force_reload=True."
            )

        # Ensure num_atom_types is set if atom_map is populated
        if self.atom_map and self.num_atom_types == 0:
            self.num_atom_types = len(self.atom_map)

        self.load(osp.join(self.processed_dir, self.processed_file_names[0]))

    @property
    def raw_file_names(self) -> List[str]:
        return [self.RAW_FILENAME]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Files relevant for THIS specific instance (one split's data + global atom map).
        The `process` method ensures all TARGET_SPLITS files are created.
        The `InMemoryDataset` checks for existence of these to decide if `process` runs.
        If `force_reload=False`, and these files for the *current split* exist, `process` is skipped.
        If any are missing for the current split, `process` runs and (re)generates files for ALL splits.
        """
        return [
            self._get_processed_filename_for_split(self.current_split_name),
            self.ATOM_MAP_FILENAME,
        ]

    def download(self) -> None:
        print(f"Downloading {self.URL} to {osp.join(self.raw_dir, self.RAW_FILENAME)}")
        download_url(self.URL, self.raw_dir, filename=self.RAW_FILENAME)
        print("Download complete.")

    def process(self) -> None:
        print("Processing raw MOSES CSV data for all target splits...")
        csv_path = osp.join(self.raw_dir, self.RAW_FILENAME)

        smiles_data_from_csv: Dict[str, List[str]] = {
            split: [] for split in self.TARGET_SPLITS
        }
        temp_available_splits_in_csv: Set[str] = set()

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "SMILES" not in reader.fieldnames or "SPLIT" not in reader.fieldnames:
                raise ValueError("CSV file must contain 'SMILES' and 'SPLIT' columns.")
            for row in reader:
                smiles = row["SMILES"].strip()
                split_from_csv = row["SPLIT"].strip()
                temp_available_splits_in_csv.add(split_from_csv)
                if smiles and split_from_csv in self.TARGET_SPLITS:
                    smiles_data_from_csv[split_from_csv].append(smiles)

        print(f"Found the following splits in CSV: {temp_available_splits_in_csv}")
        print(f"Processing for target splits: {self.TARGET_SPLITS}")

        # --- Pass 1: Parse ALL relevant SMILES (from target splits) to RDKit mol objects and collect unique atom symbols ---
        print("Parsing SMILES and collecting atom types for atom map generation...")
        # Store as: Dict[split_name, List[Tuple[original_smiles, rdkit_mol_object]]]
        parsed_molecules_by_split: Dict[str, List[Tuple[str, RDKitMol]]] = {
            split: [] for split in self.TARGET_SPLITS
        }  # type: ignore
        unique_atom_symbols: Set[str] = set()

        total_smiles_to_parse = sum(
            len(s_list) for s_list in smiles_data_from_csv.values()
        )
        pbar_parse = tqdm(
            total=total_smiles_to_parse, desc="Parsing SMILES for atom map"
        )

        for split_name in self.TARGET_SPLITS:
            for smiles_string in smiles_data_from_csv[split_name]:
                mol = Chem.MolFromSmiles(smiles_string)  # type: ignore

                if mol is None:
                    print(
                        f"Warning: Could not parse SMILES: {smiles_string} (split: {split_name}). Skipping."
                    )
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    print(
                        f"Warning: Sanitization failed for SMILES: {smiles_string} (split: {split_name}). Error: {e}. Skipping."
                    )
                    continue

                Chem.Kekulize(mol)
                
                if mol is not None:
                    if (
                        mol.GetNumAtoms() < 2
                    ):
                        print(
                            f"Warning: Molecule with no atoms (or error) for SMILES: {smiles_string} in split {split_name}. Skipping."
                        )
                        pbar_parse.update(1)
                        continue
                    parsed_molecules_by_split[split_name].append((smiles_string, mol))
                    for atom in mol.GetAtoms():
                        unique_atom_symbols.add(atom.GetSymbol())
                else:
                    print(
                        f"Warning: Could not parse SMILES: {smiles_string} in split {split_name}"
                    )
                pbar_parse.update(1)
        pbar_parse.close()

        # --- Create and save global atom map ---
        sorted_atom_symbols = sorted(list(unique_atom_symbols))
        self.atom_map = {symbol: i for i, symbol in enumerate(sorted_atom_symbols)}
        self.num_atom_types = len(self.atom_map)
        print(
            f"Generated atom map with {self.num_atom_types} unique atom types: {self.atom_map}"
        )
        torch.save(self.atom_map, osp.join(self.processed_dir, self.ATOM_MAP_FILENAME))
        print(
            f"Atom map saved to {osp.join(self.processed_dir, self.ATOM_MAP_FILENAME)}"
        )

        # --- Pass 2: Convert stored RDKit mol objects to Data objects and save per split ---
        for target_split_name in self.TARGET_SPLITS:
            print(
                f"Converting RDKit mols to graph Data objects for split: {target_split_name}..."
            )
            data_list_for_current_split = []

            molecules_to_convert = parsed_molecules_by_split[target_split_name]
            if not molecules_to_convert:
                print(f"No molecules to process for split '{target_split_name}'.")

            pbar_convert = tqdm(
                total=len(molecules_to_convert),
                desc=f"Converting for {target_split_name}",
            )
            for original_smiles, mol in molecules_to_convert:
                # Get node features (atom types)
                atom_features = []
                valid_molecule = True
                for atom in mol.GetAtoms():
                    atom_symbol = atom.GetSymbol()
                    atom_idx = self.atom_map.get(atom_symbol, None)
                    if atom_idx is None:
                        print(
                            f"Critical Error: Atom symbol '{atom_symbol}' from SMILES '{original_smiles}' "
                            f"not found in generated atom_map. This should not happen. Skipping molecule."
                        )
                        valid_molecule = False
                        break
                    atom_features.append(
                        torch.nn.functional.one_hot(
                            torch.tensor(atom_idx), num_classes=self.num_atom_types
                        )
                    )

                if not valid_molecule or not atom_features:
                    if (
                        valid_molecule
                    ):  # implies no atom_features but was valid_molecule
                        print(
                            f"Warning: No atom features generated for SMILES: {original_smiles} (split: {target_split_name}). Skipping."
                        )
                    pbar_convert.update(1)
                    continue
                x = torch.stack(atom_features, dim=0).float()

                # Get edge features (bond types) and edge_index
                edge_indices = []
                edge_attrs = []
                bond_error = False
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_type_rdkit = bond.GetBondType()
                    bond_idx = self.bond_map.get(bond_type_rdkit, None)
                    if bond_idx is None:
                        print(
                            f"Error: Unknown bond type: {bond_type_rdkit} in SMILES {original_smiles} (split: {target_split_name}). Skipping molecule."
                        )
                        bond_error = True
                        break
                    one_hot_bond = torch.nn.functional.one_hot(
                        torch.tensor(bond_idx), num_classes=self.num_bond_types
                    ).float()
                    edge_indices.extend([(i, j), (j, i)])
                    edge_attrs.extend([one_hot_bond, one_hot_bond])

                if bond_error:
                    pbar_convert.update(1)
                    continue

                if edge_indices:
                    edge_index = (
                        torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                    )
                    edge_attr = torch.stack(edge_attrs, dim=0).float()
                else:  # Molecule with no bonds (e.g., single atom)
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, self.num_bond_types), dtype=torch.float)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    smiles=original_smiles,
                    split=target_split_name,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    pbar_convert.update(1)
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list_for_current_split.append(data)
                pbar_convert.update(1)
            pbar_convert.close()

            processed_file_path = osp.join(
                self.processed_dir,
                self._get_processed_filename_for_split(target_split_name),
            )
            if not data_list_for_current_split and molecules_to_convert:
                print(
                    f"Warning: No Data objects were created for split '{target_split_name}' "
                    f"despite having {len(molecules_to_convert)} parsed molecules. "
                    "Check filtering/transformation logic or atom/bond mapping if this is unexpected."
                )

            # Save the list of Data objects for the current split
            # InMemoryDataset's save method handles collating and saving.
            self.save(data_list_for_current_split, processed_file_path)
            print(
                f"Processed graph data for split '{target_split_name}' saved to {processed_file_path} ({len(data_list_for_current_split)} graphs)"
            )

        print("All target splits processed and saved.")

    def get_atom_type_map(self) -> Dict[str, int]:
        return self.atom_map

    def get_bond_type_map(self) -> Dict[Any, int]:
        return self.bond_map

    def get_num_atom_features(self) -> int:
        if self.num_atom_types == 0 and self.atom_map:
            self.num_atom_types = len(self.atom_map)
        return self.num_atom_types

    def get_num_edge_features(self) -> int:
        return self.num_bond_types

    def get_available_splits(self) -> List[str]:
        """Returns a list of all predefined target split names for this dataset class."""
        return sorted(list(self.TARGET_SPLITS))

    def get_current_split_name(self) -> str:
        """Returns the name of the split this dataset instance currently holds."""
        return self.current_split_name

    def get_split_data(self, split_name: str) -> List[Data]:
        """
        Returns a list of Data objects for the specified split.
        If the requested split is the one this instance holds, it returns its data.
        Otherwise, it raises an error, as this instance is specific to one split.
        To get data for another split, instantiate MOSESDataset for that split.
        """
        if split_name == self.current_split_name:
            # self.data is already the collated data for the current split.
            # We need to de-collate it if the user wants a list.
            return [self.get(i) for i in range(len(self))]
        else:
            raise ValueError(
                f"This dataset instance is for split '{self.current_split_name}'. "
                f"To get data for '{split_name}', please create a new MOSESDataset "
                f"instance with split='{split_name}'."
            )

    def get_split_dataset(self, split_name: str) -> "MOSESDataset":
        """
        Returns a MOSESDataset instance for the specified split.
        If the requested split is the one this instance represents, it returns self.
        Otherwise, it creates and returns a new MOSESDataset instance for that split.
        """
        if split_name not in self.TARGET_SPLITS:
            raise ValueError(
                f"Split '{split_name}' not found. Available splits are: {self.TARGET_SPLITS}"
            )
        if split_name == self.current_split_name:
            return self
        else:
            # Create a new instance for the other split.
            # It will share the same root and transforms.
            # force_reload=False because processing should have already happened.
            print(f"Creating a new dataset instance for split: {split_name}")
            return MOSESDataset(
                root=self.root,
                split=split_name,
                transform=self.transform,
                pre_transform=self.pre_transform,  # Applied during initial processing
                pre_filter=self.pre_filter,  # Applied during initial processing
                force_reload=False,  # Data should exist
            )


def rcm_reorder_moses(data):
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    perm = reverse_cuthill_mckee(adj_matrix)[::-1]

    # 生成逆排列：旧节点 -> 新节点
    # inv_perm = torch.from_numpy(np.argsort(perm)).long()
    perm_tensor = torch.from_numpy(perm.copy()).long()
    inv_perm = torch.from_numpy(np.argsort(perm).copy()).long()

    # 重新排序节点和边
    data.x = data.x[perm_tensor]
    data.edge_index = inv_perm[data.edge_index]

    if "pe" in data.keys():
        data.pe = data.pe[perm_tensor]

    return data, perm


def moses_transform_pe_reorder(data, pe_dim=6):
    data, perm = rcm_reorder_moses(data)

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


if __name__ == "__main__":
    try:
        dataset = MOSESDataset(
            root="data/MOSES",
            force_reload=False,
            pre_transform=moses_transform_pe_reorder,
            split="train",
        )

        print(f"\n--- Dataset Info ---")
        print(f"Total number of graphs in dataset: {len(dataset)}")
        print(f"Number of atom types: {dataset.get_num_atom_features()}")
        print(f"Number of bond types: {dataset.get_num_edge_features()}")
        print(f"Atom type map: {dataset.get_atom_type_map()}")
        # print(f"Bond type map: {dataset.get_bond_type_map()}") # RDKit objects as keys, can be verbose
        print(f"Available splits: {dataset.get_available_splits()}")

        if len(dataset) > 0:
            print(f"\n--- Sample Data Object (Overall Dataset) ---")
            sample_data = dataset[0]
            print(sample_data)
            print(f"SMILES: {sample_data.smiles}")
            print(f"Split: {sample_data.split}")
            print(f"Node features shape: {sample_data.x.shape}")
            print(f"Edge index shape: {sample_data.edge_index.shape}")
            print(f"Edge attributes shape: {sample_data.edge_attr.shape}")

        # Test getting specific splits
        for split_name in dataset.get_available_splits():
            print(f"\n--- Testing Split: {split_name} ---")
            try:
                # Option 1: Get list of Data objects for the split
                # split_specific_data_list = dataset.get_split_data(split_name)
                # print(f"Number of graphs in '{split_name}' split (direct list): {len(split_specific_data_list)}")
                # if split_specific_data_list:
                #     print(f"First item in '{split_name}' split: {split_specific_data_list[0]}")
                #     print(f"Its SMILES: {split_specific_data_list[0].smiles}, Its split property: {split_specific_data_list[0].split}")

                # Option 2: Get a new Dataset object for the split
                split_specific_dataset = dataset.get_split_dataset(split_name)
                print(
                    f"Number of graphs in '{split_name}' split (as new dataset): {len(split_specific_dataset)}"
                )
                if len(split_specific_dataset) > 0:
                    print(
                        f"First item in '{split_name}' split dataset: {split_specific_dataset[0]}"
                    )
                    print(
                        f"Its SMILES: {split_specific_dataset[0].smiles}, Its split property: {split_specific_dataset[0].split}"
                    )
                    # Verify it only contains the correct split
                    all_splits_in_subset = set(
                        d.split
                        for d in [
                            split_specific_dataset.get(i)
                            for i in range(len(split_specific_dataset))
                        ]
                    )
                    print(f"Unique splits in this subset: {all_splits_in_subset}")
                    assert (
                        len(all_splits_in_subset) == 1
                        and split_name in all_splits_in_subset
                    )

            except ValueError as e:
                print(f"Error processing split {split_name}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with split {split_name}: {e}")

        # Example of how to get train, test, validation sets
        if "train" in dataset.get_available_splits():
            train_dataset = dataset.get_split_dataset("train")
            print(f"\nNumber of training samples: {len(train_dataset)}")
        if "test" in dataset.get_available_splits():
            test_dataset = dataset.get_split_dataset("test")
            print(f"Number of test samples: {len(test_dataset)}")
        # MOSES uses 'valid' or 'validation'. Check which one is present.
        validation_split_name = None
        if "validation" in dataset.get_available_splits():
            validation_split_name = "validation"
        elif "valid" in dataset.get_available_splits():
            validation_split_name = "valid"

        if validation_split_name:
            validation_dataset = dataset.get_split_dataset(validation_split_name)
            print(
                f"Number of {validation_split_name} samples: {len(validation_dataset)}"
            )
        else:
            print("No 'validation' or 'valid' split found in the dataset.")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure RDKit is installed.")
    except Exception as e:
        print(f"An error occurred during dataset processing: {e}")
        import traceback

        traceback.print_exc()
