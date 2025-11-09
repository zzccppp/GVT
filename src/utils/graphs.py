import torch
from torch_cluster import graclus_cluster
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np

def rcm_reorder_qm9(data):
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    perm = reverse_cuthill_mckee(adj_matrix)[::-1]
    
    # inv_perm = torch.from_numpy(np.argsort(perm)).long()
    perm_tensor = torch.from_numpy(perm.copy()).long()
    inv_perm = torch.from_numpy(np.argsort(perm).copy()).long()
    
    data.x = data.x[perm_tensor]
    data.edge_index = inv_perm[data.edge_index]
    data.pos = data.pos[perm_tensor]
    data.z = data.z[perm_tensor]
    
    return data, perm

def rcm_reorder_zinc(data):
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    perm = reverse_cuthill_mckee(adj_matrix)[::-1]
    
    # inv_perm = torch.from_numpy(np.argsort(perm)).long()
    perm_tensor = torch.from_numpy(perm.copy()).long()
    inv_perm = torch.from_numpy(np.argsort(perm).copy()).long()
    
    data.x = data.x[perm_tensor]
    data.edge_index = inv_perm[data.edge_index]

    if "pe" in data.keys():
        data.pe = data.pe[perm_tensor]
    
    return data, perm

def rcm_reorder_guacamol(data):
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    perm = reverse_cuthill_mckee(adj_matrix)[::-1]
    
    # inv_perm = torch.from_numpy(np.argsort(perm)).long()
    perm_tensor = torch.from_numpy(perm.copy()).long()
    inv_perm = torch.from_numpy(np.argsort(perm).copy()).long()
    
    data.x = data.x[perm_tensor]
    data.edge_index = inv_perm[data.edge_index]

    if "pe" in data.keys():
        data.pe = data.pe[perm_tensor]
    
    return data, perm


def random_reorder_qm9(data):
    perm = torch.randperm(data.num_nodes)
    
    inv_perm = torch.argsort(perm)
    
    data.x = data.x[perm]
    data.edge_index = inv_perm[data.edge_index]
    data.pos = data.pos[perm]
    data.z = data.z[perm]
    
    return data, perm