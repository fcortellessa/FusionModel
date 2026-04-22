import os   
import sys
from datetime import datetime
import logging

import torch
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("GripFusionNet")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if main() is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler (STDOUT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, log_file


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoints", interval=1):
    if epoch % interval == 0:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device, strict=False):
    """
    Load checkpoint with model and optimizer state dicts.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        strict: Whether to strictly enforce matching keys (default False for robustness)
    
    Returns:
        checkpoint dict with 'epoch', 'loss', 'model_state_dict', 'optimizer_state_dict'
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint



def get_optimizer(model_or_params, optimizer_name="Adam", lr=0.001):
    if hasattr(model_or_params, "parameters"):
        params = model_or_params.parameters()
    else:
        params = model_or_params

    if optimizer_name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif optimizer_name == "MuSGD":
        from ultralytics.optim import MuSGD
        return MuSGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Available options: Adam, AdamW, RMSprop, SGD, MuSGD.")
