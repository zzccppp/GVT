from rdkit import Chem
import numpy as np
import networkx as nx


### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    from sklearn.metrics.pairwise import pairwise_kernels
    from eden.graph import vectorize

    def kernel_compute(X, Y=None, is_hist=True, metric="linear", n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric="linear", n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(
        samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs
    )

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    # prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(
        graph_ref_list,
        graph_pred_list_remove_empty,
        metric="nspdk",
        is_hist=False,
        n_jobs=20,
    )
    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def tensor_to_mol(x, adj, atom_types, bond_types):

    x = x.cpu().numpy()
    adj = adj.cpu().numpy()
    mol = Chem.RWMol()

    for atom_type in x:
        atom = Chem.Atom(atom_types[atom_type])
        mol.AddAtom(atom)

    for i, j in zip(*adj.nonzero()):
        if i < j:
            bond_type = bond_types[adj[i, j]]
            mol.AddBond(int(i), int(j), bond_type)

    mol = mol.GetMol()

    return mol


def is_validate_mol(mols, return_valid_smiles=False):
    if not return_valid_smiles:
        return [check_validity(mol) for mol in mols]

    all_smiles = []
    valids = []
    for mol in mols:
        is_valid = check_validity(mol)
        valids.append(is_valid)
        if is_valid:
            smiles = Chem.MolToSmiles(mol)
            all_smiles.append(smiles)

    return valids, all_smiles


def check_validity(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except:
        return False
