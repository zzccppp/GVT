import torch
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning
import os
from rdkit import Chem
from tqdm import trange
from test_ar_model import decode_sequences, generate_sequences, load_trained_model


class CustomGenerator(DistributionMatchingGenerator):
    def __init__(
        self,
        ckpt_path,
        temperature,
        top_k,
        max_length=90,
        device="cuda",
        pool_size=12000,
    ):
        super().__init__()

        self.ckpt_path = ckpt_path
        self.temperature = temperature
        self.top_k = top_k
        self.max_length = max_length
        self.device = device
        self.pool_size = pool_size

        (
            self.model,
            self.codebook_shape,
            self.dataset,
            self.vae_model,
            self.dataset_name,
        ) = load_trained_model(ckpt_path, device)

        assert self.dataset_name == "Guacamol"
        assert pool_size > 0

        self.molecule_pool = []
        self.pool_initialized = False
        self._current_pool_index = 0

        self.max_batch_size = 200

        self._initialize_molecule_pool()

    def _initialize_molecule_pool(self):
        if self.pool_initialized:
            return

        self.molecule_pool = self.generate_new_molecules(self.pool_size)
        self.pool_initialized = True
        self._current_pool_index = 0

    def generate_new_molecules(self, num_samples):
        all_mols = []

        num_samples_k = num_samples // self.max_batch_size + 1

        for _ in trange(num_samples_k):
            # 生成序列
            sequences = generate_sequences(
                self.model,
                self.codebook_shape[0],
                num_samples=self.max_batch_size,
                max_length=self.max_length,
                temperature=temperature,
                top_k=top_k,
            )

            mols = decode_sequences(
                sequences,
                self.dataset.codebook,
                self.vae_model,
                self.dataset_name,
                self.device,
            )

            smiles = [Chem.MolToSmiles(mol) for mol in mols if mol is not None]

            all_mols.extend(smiles)

        return all_mols

    def generate(self, number_samples):
        if not self.pool_initialized:
            self._initialize_molecule_pool()

        if number_samples > self.pool_size:
            print(
                f"(Warning: Requested number of molecules ({number_samples}) exceeds the defined pool size ({self.pool_size}))."
            )

        generated_smiles = []
        pool_actual_size = len(self.molecule_pool)

        for _ in range(number_samples):
            molecule_to_yield = self.molecule_pool[self._current_pool_index]
            generated_smiles.append(molecule_to_yield)
            self._current_pool_index = (self._current_pool_index + 1) % pool_actual_size

        print(
            f"(Retrieved {len(generated_smiles)} molecules from the pool. Current pool index: {self._current_pool_index})."
        )
        return generated_smiles


if __name__ == "__main__":
    ckpt_path = "runs/ar/clear-dew-10/GPT2LMHeadModel_epoch_80.pt"
    temperature = 1.0
    top_k = 100
    device = "cuda:1"
    chembl_training_file = "data/GuacaMol/raw/guacamol_v1_train.smiles"

    epoch = ckpt_path.split("_")[-1].split(".")[0]
    ckpt_dir = os.path.dirname(ckpt_path)
    json_save_path = os.path.join(
        ckpt_dir, f"GPT2LMHeadModel_{epoch}_{temperature}_{top_k}.json"
    )
    print(f"Results saved to {json_save_path}")

    generator = CustomGenerator(
        ckpt_path, temperature, top_k, max_length=90, device=device, pool_size=12000
    )

    assess_distribution_learning(
        generator,
        chembl_training_file=chembl_training_file,
        json_output_file=json_save_path,
    )
