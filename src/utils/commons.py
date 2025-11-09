import datetime
import torch
import os
import numpy as np
import wandb
from utils.ema import ExponentialMovingAverage
from accelerate.utils import set_seed

def to_cuda(x, device):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=device, non_blocking=True)

def save_model(model, epoch, optim, path, config=None, ema=None, scheduler=None, other_info=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "config": config,
            "ema": ema.state_dict() if ema is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "other_info": other_info if other_info is not None else {},
        },
        path,
    )


def load_model(model, optim, path, ema=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    return model, optim, checkpoint["epoch"], ema



def generate_save_path(parent_path, model_name, epoch):
    return f"{parent_path}/{model_name}_epoch_{epoch}.pt"


def generate_run_name(model_name):
    return f"{model_name}_{datetime.datetime.now().now().strftime('%Y-%m-%d_%H-%M-%S')}"


def create_save_dir(run_name, parent_path="./runs"):
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    path = f"{parent_path}/{run_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def setup_save_dir(model_name, parent_path="./runs", run_name=None):
    if run_name is None or run_name == "":
        run_name = generate_run_name(model_name)
    return create_save_dir(run_name, parent_path)


def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    set_seed(seed)


def init_wandb(config, tags=[], group="default"):
    import sys

    if "debugpy" in sys.modules:
        print("Debugger detected, wandb set to offline")
        wandb.init(project=config["wandb_project"], config=config, mode="offline", tags=['debug'] + tags)
    else:
        wandb.init(
            project=config["wandb_project"],
            config=config,
            tags=tags,
            save_code=True,
            group=group
        )
        wandb.run.log_code("src/")
