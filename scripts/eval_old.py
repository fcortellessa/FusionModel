# scripts/eval.py

import torch
import yaml

from models.pose_model import PoseEstimationNet
from data.dataset import PoseDataset


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "pose"}
            target = batch["pose"].to(device)

            y, z, q = model(inputs)

            # TODO: compute metrics
            total_loss += q.norm().item()

    print("Eval metric:", total_loss / len(loader))


def main(config_path, checkpoint):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseDataset(cfg["data"], split="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = PoseEstimationNet(
        model_name=cfg["model"]["backbone"],
        streams=cfg["model"]["streams"],
        fusion=cfg["model"]["fusion"]
    ).to(device)

    model.load_state_dict(torch.load(checkpoint))

    evaluate(model, loader, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    args = parser.parse_args()

    main(args.config, args.ckpt)