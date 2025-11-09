from typing import Literal
import moses
import torch
from tqdm import trange
from mydatasets.vq_gen_dataset import VQ_Dataset
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
import pandas as pd
import os

def check_validity(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
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


def load_trained_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    config_sample = checkpoint["config"]

    model = create_custom_model(
        config_sample["codebook_size"][0],
        config_sample["codebook_size"][1],
        config_sample["model"]["config"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model.to(device),
        config_sample["codebook_size"],
        config_sample["train"]["dataset_path"],
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


def decode_sequences(sequences, codebook, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config_sample = ckpt["config"]
    model = VQVAE(**config_sample["model"]["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model.to(device)

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
        quantized_node, adj_quantized, node_mask = model.decode(quantized_vec, batch)

    quantized_node, adj_quantized, node_mask = (
        quantized_node.argmax(dim=-1),
        adj_quantized.argmax(dim=-1),
        node_mask,
    )

    chems = []

    atom_types = {"Br": 0, "C": 1, "Cl": 2, "F": 3, "N": 4, "O": 5, "S": 6}

    atom_types = {v: k for k, v in atom_types.items()}

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


def test_generation(
    ar_ckpt_path,
    num_samples=32,
    device="cuda",
    temperature=1.0,
    top_k=30,
    max_length=64,
):
    model, codebook_shape, dataset_path = load_trained_model(ar_ckpt_path, device)

    dataset_path = dataset_path

    dataset = VQ_Dataset(dataset_path)
    vae_ckpt_path = dataset.generate_config["ckpt_path"]

    all_smiles = []
    max_batch = 200
    sample_times = num_samples // max_batch
    if num_samples % max_batch != 0:
        sample_times += 1

    for i in trange(sample_times):
        sequences = generate_sequences(
            model,
            codebook_shape[0],
            num_samples=max_batch,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

        mols = decode_sequences(sequences, dataset.codebook, vae_ckpt_path, device)

        smiles = [Chem.MolToSmiles(mol) for mol in mols]

        all_smiles.extend(smiles)

    return all_smiles

def do_test(ckpt_path, temperature=1.0, top_k=100, num_samples=10000, max_length=64):
    smiles = test_generation(
        ckpt_path,
        num_samples=num_samples,
        device="cuda",
        temperature=temperature,
        top_k=top_k,
        max_length=max_length,
    )
    metrics = moses.get_all_metrics(smiles)

    print("Ckpt Path:", ckpt_path)
    print("Valid ", metrics["valid"] * 100)
    print("Unique ", metrics["unique@1000"] * 100)
    print(
        "Novel ",
        metrics["Novelty"] * 100,
    )
    print(
        "Filters ",
        metrics["Filters"] * 100,
    )
    print("FCD ", metrics["FCD/Test"])
    print("SNN ", metrics["SNN/Test"])
    print("Scaf ", metrics["Scaf/TestSF"] * 100)

    return {
        "valid": metrics["valid"],
        "unique@1000": metrics["unique@1000"],
        "Novelty": metrics["Novelty"],
        "Filters": metrics["Filters"],
        "FCD/Test": metrics["FCD/Test"],
        "SNN/Test": metrics["SNN/Test"],
        "Scaf/TestSF": metrics["Scaf/TestSF"],
        "ckpt_path": ckpt_path,
        "temperature": temperature,
        "top_k": top_k,
    }, smiles



if __name__ == "__main__":
    output_filename = "ar_model_moses_results_stream.csv"

    all_temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    all_top_k = [50, 100, 200]

    for temp in all_temps:
        for top_k in all_top_k:
            print(f"--> Start: temperature={temp}, top_k={top_k}")
            result, smiles = do_test(
                "runs/ar/fanciful-leaf-7/GPT2LMHeadModel_epoch_40.pt",
                temperature=temp,
                top_k=top_k,
                num_samples=10000,
                max_length=64,
            )
            
            df_row = pd.DataFrame([result])

            write_header = not os.path.exists(output_filename)
            
            df_row.to_csv(
                output_filename, 
                mode='a', 
                header=write_header, 
                index=False
            )

            print(f"--> Finished: temperature={temp}, top_k={top_k}")



    pass
