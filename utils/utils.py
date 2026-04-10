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


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints", interval=1):
    if epoch % interval == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_loss_{loss:.4f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Example usage:
# logger = setup_logger()
# logger.info("This is an info message.")

# checkpoint_path = save_checkpoint(model, optimizer, epoch, loss)
# logger.info(f"Checkpoint saved at: {checkpoint_path}")

# 