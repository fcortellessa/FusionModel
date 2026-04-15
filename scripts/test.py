# test script
import time
import torch
import pandas as pd
from base_model.pose_estimation_model import PoseEstimationNet
from utils.utils import load_checkpoint, load_config



model_path = "checkpoints/best_model.pth"
config_path = "configs/pose_estimation.yaml"
test_data_path = "data/test_data.pt"


if __name__ == "__main__":
    cfg = load_config(config_path)
    model = PoseEstimationNet(
        model_name=cfg["model"]["backbone"],
        streams=cfg["model"]["streams"],
        fusion=cfg["model"]["fusion"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        rotation_parametrisation=cfg["model"]["head_type"],
        activation=cfg["model"]["activation"],
        rotation_layers=[cfg["model"]["mlp_quat_1"], cfg["model"]["mlp_quat_2"], cfg["model"]["mlp_quat_3"]],
        translation_layers=[cfg["model"]["mlp_trans_1"], cfg["model"]["mlp_trans_2"],cfg["model"]["mlp_trans_3"]],
        dropout_rot=cfg["training"]["dropout_rot"],
        dropout_trans=cfg["training"]["dropout_trans"],
        use_batchnorm=cfg["training"]["use_batchnorm"]
    )
    # load state dict
    model.load_state_dict(load_checkpoint(model_path, device="cpu")["model_state_dict"])
    print("Model loaded successfully!")

    # load test data
    test_data = torch.load(test_data_path)
    # run inference
    model.eval()
    csv_results = {
        "sample_id": [],
        "pred_translation": [],
        "pred_rotation": [],
        "pred_quaternion": [],
        "gt_pose": []
    }
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            inputs = {k: v.unsqueeze(0) for k, v in sample.items() if k != "pose"}
            target = sample["pose"].unsqueeze(0)

            y, z, q = model(inputs)
            # Store results for CSV export
            csv_results["sample_id"].append(i)
            csv_results["pred_translation"].append(y.cpu().numpy())
            csv_results["pred_rotation"].append(z.cpu().numpy())
            csv_results["pred_quaternion"].append(q.cpu().numpy())
            csv_results["gt_pose"].append(target.cpu().numpy())

    # Save results to CSV
    df = pd.DataFrame(csv_results)
    df.to_csv(f"results/results_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
