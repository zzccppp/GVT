from functools import partial
from rdkit import Chem
from torch_geometric.utils import to_dense_adj
import torch

from utils.chems import check_validity, tensor_to_mol
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import numpy as np

def get_atom_types(dataset_name):
    if dataset_name == "QM9":
        return {
            0: "C",
            1: "N",
            2: "O",
            3: "F",
        }
    elif dataset_name == "ZINC":
        r = {
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
        return {v: k for k, v in r.items()}

    elif dataset_name == "MOSES":
        r = {"Br": 0, "C": 1, "Cl": 2, "F": 3, "N": 4, "O": 5, "S": 6}
        return {v: k for k, v in r.items()}

    elif dataset_name == "Guacamol":
        atom_types = {
            "B": 0,
            "Br": 1,
            "C": 2,
            "Cl": 3,
            "F": 4,
            "I": 5,
            "N": 6,
            "O": 7,
            "P": 8,
            "S": 9,
            "Se": 10,
            "Si": 11,
        }
        return {v: k for k, v in atom_types.items()}
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets are: QM9, ZINC, MOSES, Guacamol."
        )


def get_zinc_dataset(split, root=None, pe_dim=6, reorder_func="dfs"):
    def filter_fn(data):
        atom_types = {
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
        atom_types = {v: k for k, v in atom_types.items()}

        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            # 4: Chem.BondType.AROMATIC,
        }

        x = data.x.argmax(dim=-1)
        adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze()
        adj = torch.cat(
            [torch.full((adj.shape[0], adj.shape[0], 1), 0.9), adj], dim=2
        ).argmax(dim=-1)

        mol = tensor_to_mol(x, adj, atom_types=atom_types, bond_types=bond_types)
        return check_validity(mol)

    from mydatasets.ZINC250k import CustomZINCDataset, transform_pe_reorder_custom_reorder

    def re_func(data):
        if reorder_func == "dfs":
            num_nodes = data.num_nodes
            G = to_networkx(data, to_undirected=True)
            dfs_order_nodes = list(nx.dfs_preorder_nodes(G, source=0))
            if len(dfs_order_nodes) < num_nodes:
                remaining_nodes = list(set(range(num_nodes)) - set(dfs_order_nodes))
                dfs_order_nodes.extend(remaining_nodes)

            perm = torch.tensor(dfs_order_nodes, dtype=torch.long)
            inv_perm = torch.from_numpy(np.argsort(np.array(dfs_order_nodes)).copy()).long()

            data.x = data.x[perm]
            data.edge_index = inv_perm[data.edge_index]

            return data, perm

        elif reorder_func == "bfs":
            num_nodes = data.num_nodes
            G = to_networkx(data, to_undirected=True)
            bfs_edges = nx.bfs_edges(G, source=0)
            bfs_order_nodes = [0] + [v for u, v in bfs_edges]
            if len(bfs_order_nodes) < num_nodes:
                remaining_nodes = list(set(range(num_nodes)) - set(bfs_order_nodes))
                bfs_order_nodes.extend(remaining_nodes)

            perm = torch.tensor(bfs_order_nodes, dtype=torch.long)
            inv_perm = torch.from_numpy(np.argsort(np.array(bfs_order_nodes)).copy()).long()

            data.x = data.x[perm]
            data.edge_index = inv_perm[data.edge_index]

            return data, perm
        elif reorder_func == "random":
            num_nodes = data.num_nodes
            dfs_order_nodes = np.random.permutation(num_nodes).tolist()
            perm = torch.tensor(dfs_order_nodes, dtype=torch.long)
            inv_perm = torch.from_numpy(np.argsort(np.array(dfs_order_nodes)).copy()).long()

            data.x = data.x[perm]
            data.edge_index = inv_perm[data.edge_index]

            return data, perm
        else:
            raise ValueError(f"Unsupported reorder function: {reorder_func}")

    root = root or "data/ZINC_custom_reorder"
    dataset = CustomZINCDataset(
        root=root,
        pre_transform=partial(transform_pe_reorder_custom_reorder, reorder_fun=re_func, pe_dim=pe_dim),
        pre_filter=filter_fn,
        split=split,
    )

    return dataset


def get_dataset(name, split, root=None, pe_dim=6):
    if name == "QM9":
        from mydatasets.QM9 import CustomQM9Dataset, transform_pe_reorder

        root = root or "data/QM9"
        dataset = CustomQM9Dataset(
            root=root,
            split=split,
            force_reload=False,
            pre_transform=partial(transform_pe_reorder, pe_dim=pe_dim),
        )
        return dataset
    elif name == "ZINC":

        def filter_fn(data):
            atom_types = {
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
            atom_types = {v: k for k, v in atom_types.items()}

            bond_types = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
                # 4: Chem.BondType.AROMATIC,
            }

            x = data.x.argmax(dim=-1)
            adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze()
            adj = torch.cat(
                [torch.full((adj.shape[0], adj.shape[0], 1), 0.9), adj], dim=2
            ).argmax(dim=-1)

            mol = tensor_to_mol(x, adj, atom_types=atom_types, bond_types=bond_types)
            return check_validity(mol)

        from mydatasets.ZINC250k import CustomZINCDataset, transform_pe_reorder

        root = root or "data/ZINC"
        dataset = CustomZINCDataset(
            root=root,
            pre_transform=partial(transform_pe_reorder, pe_dim=pe_dim),
            pre_filter=filter_fn,
            split=split,
        )

        return dataset

    elif name == "MOSES":
        from mydatasets.MOSES import MOSESDataset, moses_transform_pe_reorder

        root = root or "data/MOSES"
        dataset = MOSESDataset(
            root=root,
            split=split,
            force_reload=False,
            pre_transform=partial(moses_transform_pe_reorder, pe_dim=pe_dim),
        )
        return dataset

    elif name == "Guacamol":
        from mydatasets.GuacaMol import GuacaMolDataset, guacamol_transform_pe_reorder

        def filter_fn(data):
            atom_types = {
                "B": 0,
                "Br": 1,
                "C": 2,
                "Cl": 3,
                "F": 4,
                "I": 5,
                "N": 6,
                "O": 7,
                "P": 8,
                "S": 9,
                "Se": 10,
                "Si": 11,
            }
            atom_types = {v: k for k, v in atom_types.items()}

            bond_types = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }

            x = data.x.argmax(dim=-1)
            adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze()
            adj = torch.cat(
                [torch.full((adj.shape[0], adj.shape[0], 1), 0.9), adj], dim=2
            ).argmax(dim=-1)

            mol = tensor_to_mol(x, adj, atom_types=atom_types, bond_types=bond_types)
            return check_validity(mol)

        root = root or "data/GuacaMol"
        dataset = GuacaMolDataset(
            root=root,
            split=split,
            force_reload=False,
            pre_transform=partial(guacamol_transform_pe_reorder, pe_dim=pe_dim),
            pre_filter=filter_fn,
        )
        return dataset

    else:
        raise ValueError(
            f"Unsupported dataset: {name}. Supported datasets are: QM9, ZINC, MOSES."
        )


if __name__ == "__main__":
    # Example usage
    dataset_name = "Guacamol"
    split = "train"
    dataset = get_dataset(dataset_name, split)
    print(f"Loaded {len(dataset)} samples from {dataset_name} dataset.")
