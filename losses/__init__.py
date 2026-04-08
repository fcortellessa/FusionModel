# # losses/__init__.py
# 

# import torch.nn as nn
# from .se2_losses import mse_loss, total_loss_2D_vector
# # 1. Define the Registry
# _LOSS_FACTORY = {
#     "mse": nn.MSELoss,
#     "l1": nn.L1Loss,
#     "weighted_pose": WeightedPoseLoss,
# }

# # 2. The Helper Function
# def get_loss(loss_config):
#     """
#     Args:
#         loss_config: A dict containing 'type' and optionally 'params'.
#                      e.g., {"type": "weighted_pose", "params": {"rot_weight": 0.5}}
#     """
#     name = loss_config["type"]
#     params = loss_config.get("params", {})

#     if name not in _LOSS_FACTORY:
#         raise ValueError(
#             f"Loss '{name}' not found. Available: {list(_LOSS_FACTORY.keys())}"
#         )
    
#     # Instantiate and return the loss object
#     return _LOSS_FACTORY[name](**params)