from typing import Literal
import torch
from tqdm import trange
from load_data import get_atom_types, get_dataset
from mydatasets.vq_gen_dataset import VQ_Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader
import os
from train_ar_model import create_custom_model
from train_vqvae import VQVAE
from rdkit import Chem
from rdkit.Chem import rdmolops, Draw, AllChem, Descriptors, QED
from rdkit.Chem.Draw import SimilarityMaps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from fcd_torch import FCD
import networkx as nx
from utils.chems import nspdk_stats
import time


def check_validity(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except:
        return False


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
            #    atomic_num=atom.GetAtomicNum(),
            #    formal_charge=atom.GetFormalCharge(),
            #    chiral_tag=atom.GetChiralTag(),
            #    hybridization=atom.GetHybridization(),
            #    num_explicit_hs=atom.GetNumExplicitHs(),
            #    is_aromatic=atom.GetIsAromatic())

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                label=int(bond.GetBondTypeAsDouble()),
            )
            #    bond_type=bond.GetBondType())

        nx_graphs.append(G)
    return nx_graphs


def get_testset_smiles(dataset_name: str):
    if dataset_name == "QM9":
        dataset = get_dataset(dataset_name, "test")
        smiles = [i.smiles for i in dataset]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
    elif dataset_name == "ZINC":
        dataset = get_dataset(dataset_name, "test")
        smiles = [i.smiles for i in dataset]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} is not supported for test set retrieval."
        )

    return smiles, mols


def load_trained_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # dataset = torch.load("data/pyg/qm9_vq/processed_data.pt", map_location="cpu")
    config_sample = checkpoint["config"]

    model = create_custom_model(
        config_sample["codebook_size"][0],
        config_sample["codebook_size"][1],
        config_sample["model"]["config"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = VQ_Dataset(config_sample["train"]["dataset_path"])
    vae_ckpt_path = dataset.generate_config["ckpt_path"]

    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    vae_config_sample = vae_ckpt["config"]
    vae_model = VQVAE(**vae_config_sample["model"]["config"])
    vae_model.load_state_dict(vae_ckpt["model_state_dict"])
    vae_model.eval()

    vae_model.to(device)

    return (
        model.to(device),
        config_sample["codebook_size"],
        dataset,
        vae_model,
        vae_config_sample["dataset"]["name"],
    )


def generate_sequences(
    model, codebook_size, num_samples=5, max_length=64, top_k=30, temperature=1.0
):
    device = next(model.parameters()).device

    generated_sequences = []
    input_ids = torch.tensor([[codebook_size]]).to(device)
    input_ids = input_ids.repeat(1, num_samples).view(-1, 1)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length,
            # eos_token_id=codebook_size + 1,  # EOS token
            # pad_token_id=codebook_size + 2,  # PAD token
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
        )

    sequence = output[:, 1:]
    # find the EOS token
    EOS = codebook_size + 1
    PAD = codebook_size + 2

    batch_size = sequence.shape[0]

    for i in range(batch_size):
        current_sequence_tensor = sequence[i]
        is_eos_or_pad_mask = torch.logical_or(
            current_sequence_tensor == EOS, current_sequence_tensor == PAD
        )
        eos_or_pad_indices = torch.where(is_eos_or_pad_mask)[0]
        truncate_at_index = len(current_sequence_tensor)
        if (
            eos_or_pad_indices.numel() > 0
        ):
            truncate_at_index = eos_or_pad_indices[0].item()
        truncated_sequence_tensor = current_sequence_tensor[:truncate_at_index]
        generated_sequences.append(truncated_sequence_tensor.tolist())

    return generated_sequences


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


def decode_sequences(sequences, codebook, vae_model, dataset_name, device="cuda"):
    codebook = codebook.to(device)

    sequences = [torch.tensor(seq).to(device) for seq in sequences]

    batch = torch.cat(
        [
            torch.full((len(seq),), i, dtype=torch.long, device=device)
            for i, seq in enumerate(sequences)
        ]
    )

    combined_sequence = torch.cat(sequences)
    quantized_vec = codebook[combined_sequence]
    quantized_vec.to(device)

    with torch.no_grad():
        quantized_node, adj_quantized, node_mask = vae_model.decode(
            quantized_vec, batch
        )

    quantized_node, adj_quantized, node_mask = (
        quantized_node.argmax(dim=-1),
        adj_quantized.argmax(dim=-1),
        node_mask,
    )

    chems = []

    atom_types = get_atom_types(dataset_name)

    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(len(sequences)):
        x = quantized_node[batch == i]
        adj = (adj_quantized[i][: x.shape[0], : x.shape[0]] + 1) % 4

        chem = tensor_to_mol(x, adj, atom_types, bond_types)
        chems.append(chem)

    return chems


def evaluate_molecules(mols, dataset_name):
    results = defaultdict(list)
    valid_mols = []

    print(f"All Molecules: {len(mols)}")

    for i, mol in enumerate(mols):
        try:
            if mol is None:
                results["valid"].append(False)
                continue

            is_valid = check_validity(mol)
            results["valid"].append(is_valid)
            if not is_valid:
                continue

            smiles = Chem.MolToSmiles(mol)

            if is_valid:
                valid_mols.append(mol)
                results["smiles"].append(smiles)
        except Exception as e:
            print(f"Error on: {str(e)}")
            results["valid"].append(False)

    validity_rate = sum(results["valid"]) / len(mols) if mols else 0
    print(f"Valid Rate: {validity_rate:.2%}")

    unique_smiles = list(set(results["smiles"]))
    unique_rate = (
        len(unique_smiles) / len(results["smiles"]) if results["smiles"] else 0
    )

    test_smiles, test_mols = get_testset_smiles(dataset_name)

    fcd = FCD(device="cuda:0", n_jobs=16)
    fcd_score = fcd(gen=unique_smiles, ref=test_smiles)
    nspdk = nspdk_stats(mols_to_nx(test_mols), mols_to_nx(valid_mols))

    return {
        "valid_count": sum(results["valid"]),
        "total_count": len(mols),
        "validity_rate": validity_rate,
        "property_values": results,
        # "valid_mols": valid_mols,
        "unique_rate": unique_rate,
        "fcd": fcd_score,
        "nspdk": nspdk,
        "unique_smiles": unique_smiles,
    }


def test_generation(
    ar_ckpt_path,
    num_batch=50,
    num_samples_batch=200,
    device="cuda",
    temperature=1.0,
    top_k=30,
):
    model, codebook_shape, dataset, vae_model, dataset_name = load_trained_model(
        ar_ckpt_path, device
    )

    all_mols = []

    start_time = time.time()

    for _ in trange(num_batch):
        sequences = generate_sequences(
            model,
            codebook_shape[0],
            num_samples=num_samples_batch,
            max_length=30,
            temperature=temperature,
            top_k=top_k,
        )
        mols = decode_sequences(
            sequences, dataset.codebook, vae_model, dataset_name, device
        )

        all_mols.extend(mols)

    end_time = time.time()

    print(f"Generation Elasped: {end_time - start_time} s.")

    evaluation_results = evaluate_molecules(all_mols, dataset_name)

    return mols, evaluation_results


if __name__ == "__main__":
    ar_ckpt_path = "runs/ar/robust-water-12/GPT2LMHeadModel_epoch_2000.pt"
    num_batch = 10
    num_samples_batch = 1000
    device = "cuda"
    temperature = 1.0
    top_k = 100

    mols, eval_results = test_generation(
        ar_ckpt_path,
        num_batch,
        num_samples_batch=num_samples_batch,
        device=device,
        temperature=temperature,
        top_k=top_k,
    )

    print("Results:")
    print(f"FCD: {eval_results['fcd']:.4f}")
    print(f"NSPDK: {eval_results['nspdk']:.4f}")
    print(f"Mols Count: {eval_results['total_count']}")
    print(f"Valid Count: {eval_results['valid_count']}")
    print(f"Valid Rate: {eval_results['validity_rate']:.2%}")
    print(f"Unique Rate: {eval_results['unique_rate']:.2%}")

    import json

    result_path = ar_ckpt_path.replace(".pt", "_results.json")
    with open(result_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Result save to: {result_path}")
