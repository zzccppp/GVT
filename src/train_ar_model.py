import os
import torch
from torch.amp import autocast, GradScaler
from mydatasets.vq_gen_dataset import VQ_Dataset
import torchinfo
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.args import parse_args, update_config
from utils.commons import (
    generate_run_name,
    generate_save_path,
    init_wandb,
    save_model,
    setseed,
    setup_save_dir,
    to_cuda,
)
from functools import partial
import utils.logger as log
import wandb
from nets.minigpt import GPT, GPTPosEmb
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator


config_template = {
    "wandb_project": "MORGAN-AR-Generation",
    "train": {
        "batch_size": 512,
        "device": "cuda",
        "seed": 42,
        "ckpt_freq": 20,
        "epochs": 2001,
        "lr": 1e-4,
        "permute": False,
        "dataset_path": "runs/vqvae/QM9/dutiful-butterfly-3/QM9_vqdataset_valid_reduced.pt",
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1,
    },
    "model": {
        "name": "GPT2LMHeadModel",
        "config": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 384,
            "n_positions": 90,  # 最大序列长度
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "pos_emb": True,
        },
    },
}


def calc_unique_index(dataset_path):
    dataset = VQ_Dataset(dataset_path)
    unique_index = set()
    for seq in dataset.sequences:
        unique_index.update(seq.tolist())
        pass

    print(f"Unique index count: {len(unique_index)}")
    print(f"Codebook size: {len(dataset.codebook)}")
    print(unique_index)


def create_custom_model(codebook_size, code_dim, model_config):
    """创建适合codebook尺寸的GPT模型"""
    config = GPT.get_default_config()
    config.model_type = None
    config.n_layer = model_config["n_layer"]
    config.n_head = model_config["n_head"]
    config.n_embd = model_config["n_embd"]
    # these options must be filled in externally
    config.vocab_size = codebook_size + 3
    config.block_size = model_config["n_positions"]
    # dropout hyperparameters
    config.embd_pdrop = 0.1
    config.resid_pdrop = model_config["resid_pdrop"]
    config.attn_pdrop = model_config["attn_pdrop"]

    if model_config.get("pos_emb", False):
        return GPTPosEmb(config)
    else:
        return GPT(config)


def dynamic_padding_collator(batch, pad_token_id):
    """处理变长序列的批处理函数 (带attention mask生成)"""
    # 解包输入和标签序列
    seq_list = [item["origin_seq"] for item in batch]

    # 使用pad_sequence动态填充
    padded_seq = pad_sequence(
        sequences=seq_list,
        batch_first=True,  # 输出形状为 (B, T)
        padding_value=pad_token_id,
    )
    input_ids = padded_seq[:, :-1]  # [batch_size, seq_len-1]
    labels = padded_seq[:, 1:].clone()  # [batch_size, seq_len-1]
    labels[labels == pad_token_id] = -100

    # 生成attention mask（标记有效位置）
    attention_mask = (input_ids != pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


if __name__ == "__main__":
    args = parse_args(config_template)
    config_sample = update_config(config_template, args)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=config_template["train"]["mixed_precision"],
        kwargs_handlers=[kwargs],
    )
    accelerator.print("Using Config: ", config_sample)
    setseed(config_sample["train"]["seed"])

    dataset_path = config_sample["train"]["dataset_path"]

    dataset = VQ_Dataset(dataset_path, random_perm=config_sample["train"]["permute"])

    if accelerator.is_main_process:
        init_wandb(
            config_sample,
            tags=[config_sample["model"]["name"]],
            group=dataset.generate_config["dataset"],
        )

        runname = None
        if wandb.run.name is None or wandb.run.name == "":
            runname = generate_run_name(config_sample["model"]["name"])
        else:
            runname = wandb.run.name

        save_dir = setup_save_dir(
            config_sample["model"]["name"],
            "./runs/ar/",
            runname,
        )
        log.info(f"Saving to {save_dir}")
    else:
        save_dir = None
        runname = None

    accelerator.print(f"Using device: {accelerator.device}")

    config_sample["codebook_size"] = dataset.codebook.shape
    model = create_custom_model(
        dataset.codebook.shape[0],
        dataset.codebook.shape[1],
        config_sample["model"]["config"],
    )

    # summary the model
    if accelerator.is_main_process:
        torchinfo.summary(
            model, input_data=torch.randint(0, dataset.codebook.shape[0], (1, 5))
        )

    loader = DataLoader(
        dataset,
        batch_size=config_sample["train"]["batch_size"],
        shuffle=True,
        collate_fn=partial(
            dynamic_padding_collator, pad_token_id=dataset.codebook.size(0) + 2
        ),
        num_workers=8,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_sample["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config_sample["train"]["epochs"]
    )

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    for epoch in tqdm(
        range(config_sample["train"]["epochs"]), disable=not accelerator.is_main_process
    ):
        losses = []
        for batch in tqdm(loader, disable=not accelerator.is_main_process):
            with autocast(device_type=accelerator.device.type):
                logits, loss = model(
                    idx=batch["input_ids"],
                    targets=batch["labels"],
                )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
        scheduler.step()

        if accelerator.is_main_process:
            wandb.log(
                {
                    "loss": sum(losses) / len(losses),
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            if epoch % config_sample["train"]["ckpt_freq"] == 0:
                save_model(
                    accelerator.unwrap_model(model),
                    epoch,
                    optimizer,
                    generate_save_path(
                        save_dir,
                        config_sample["model"]["name"],
                        epoch,
                    ),
                    ema=None,
                    config=config_sample,
                )
    pass
