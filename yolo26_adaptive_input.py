"""
This code defines a multi-stream pose estimation model based on the YOLO architecture.
Key Features:
1. **Multi-Stream Input**: Supports multiple input streams (e.g., RGB camera, tactile sensor, depth camera) with flexible fusion strategies (concatenation, gated, attention).
Architecture:
- Each stream has its own YOLO backbone (encoder + neck) to extract multi-scale features (P3, P4, P5).
- Dimensionality reduction via 1x1 convolutions and spatial pooling to get fixed-size feature vectors.
- A fusion module combines features from all streams.
- Separate MLP heads predict translation (y, z) and rotation (quaternion).

2. **Dynamic Channel Alignment**: The model can adapt the first convolutional layer of the YOLO backbone to accept different numbers of input channels (e.g., 4 for RGB-D).
3. **Fusion Strategies**: The `FusionModule` supports simple concatenation, gated fusion (learned weights), and attention-based fusion.
"""

import os
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.optim import MuSGD
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from Dataloader.dataloader_preprocessed import get_dataloaders
from Loss.loss_det_both_weighted import total_loss, test_loss


class FusionModule(nn.Module):
    def __init__(self, dim, method="concat", num_streams=2):
        super().__init__()
        self.method = method
        self.num_streams = num_streams
        if method == "gated":
            self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, features_list):
        """
        Args:
            features_list: List of [Batch, Dim] tensors from each stream
        """
        if len(features_list) == 1:
            return features_list[0]

        if self.method == "concat":
            # Result: [Batch, Dim * num_streams]
            return torch.cat(features_list, dim=1)

        elif self.method == "gated":
            # Result: [Batch, Dim] (Weighted average of streams)
            combined = torch.cat(features_list, dim=1)
            # weights shape: [Batch, num_streams]
            weights = self.gate(combined).unsqueeze(-1) 
            # stack: [Batch, num_streams, Dim]
            stacked = torch.stack(features_list, dim=1)
            return (stacked * weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[512], dropout=0.0, bn=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _adapt_backbone_channels(model, in_channels=4):
    """Modify first conv layer to accept N channels (especially for RGB-D = 4)."""
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    if first_conv is None:
        return model

    old_weight = first_conv.weight.data
    new_conv = nn.Conv2d(in_channels,
                         first_conv.out_channels,
                         kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride,
                         padding=first_conv.padding,
                         bias=(first_conv.bias is not None))

    # Copy existing weights
    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        if in_channels > 3:
            new_conv.weight[:, 3:] = old_weight[:, :1]  # copy R channel

    # Replace
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is first_conv:
                setattr(module, child_name, new_conv)
                return model

    return model



# -----------------------------
# Main Model
# -----------------------------
class PoseEstimationNet(nn.Module):
    def __init__(self,
                 model_name='yolo26n.pt',
                 streams=("mono_cam", "tactile", "depth_cam"),
                 fusion="concat",
                 freeze_backbone=False):
        super().__init__()

        self.streams = streams
        self.feature_extractors = nn.ModuleDict()   # yolo trunks (backbone + neck) for each stream
        self.in_channels = {}

        # 1. Load Base Model Template
        full_yolo = YOLO(model_name).model
        self.p345_indices = get_p345_indices(full_yolo)
        base_model = full_yolo.model[:-1]   # Extract backbone + neck

        for stream in streams:
            extractor = deepcopy(base_model)
            
            if stream == "depth_cam":
                extractor = _adapt_backbone_channels(extractor, 4)
            else:
                self.in_channels[stream] = 3
            
            if freeze_backbone:
                for p in extractor.parameters():
                    p.requires_grad = False
            
            self.feature_extractors[stream] = extractor

        # Extract feature dimensions using a dummy forward pass
        with torch.no_grad():
            stream0 = streams[0]
            dummy_input = torch.zeros(1, self.in_channels[stream0], 640, 640)
            feats = self.extract_features_raw(dummy_input, self.feature_extractors[stream0])

        dim_p3, dim_p4, dim_p5 = [f.shape[1] for f in feats]  # should be 64, 128, 256 for yolo26n.pt

        # 2. Dimensionality Reduction (1x1 Convolutions)
        # YOLO Nano channels: P3=64, P4=128, P5=256
        self.conv_p3 = nn.Conv2d(dim_p3, dim_p3//2, 1)  # ->(feats[1], feats[1]//2, 1) 
        self.conv_p4 = nn.Conv2d(dim_p4, dim_p4//2, 1)
        self.conv_p5 = nn.Conv2d(dim_p5, dim_p5//2, 1)

        # 3. Spatial Pooling: Downsampling layers to reach target spatial dimensions
        self.spatial_grid_size = (10, 10)
        self.down_p3 = nn.AdaptiveAvgPool2d(self.spatial_grid_size)
        self.down_p4 = nn.AdaptiveAvgPool2d(self.spatial_grid_size)
        self.down_p5 = nn.AdaptiveAvgPool2d(self.spatial_grid_size)

        # 4. Fusion
        # Compute feature dimension
        with torch.no_grad():
            single_stream_out = self.forward_single_stream(dummy_input, self.feature_extractors[stream0])
            self.single_stream_feat_dim = single_stream_out.shape[1]

        self.fusion = FusionModule(self.single_stream_feat_dim, fusion, num_streams=len(streams))
        if fusion == "concat":
            self.fused_dim = self.single_stream_feat_dim * len(self.streams)
        else:
            self.fused_dim = self.single_stream_feat_dim    # gated

        # 5. Regression Heads (MLPs) Predict Position and Rotation
        self.translation_head = MLPHead(self.fused_dim, 2, hidden=[512], dropout=0.2, bn=True)
        self.rotation_head = MLPHead(self.fused_dim, 4, hidden=[512], dropout=0.2, bn=True)
    
    def forward(self, x_dict):
        stream_features = []

        for stream in self.streams:
            x = x_dict[stream]
            extractor = self.feature_extractors[stream]
            # Get the flattened [Batch, Dim] feature for this stream
            feat = self.forward_single_stream(x, extractor)
            stream_features.append(feat)

        # 6. Multi-Stream Fusion (Now handles list of tensors)
        fused_features = self.fusion(stream_features)

        # 7. Final Prediction Head
        translation = self.translation_head(fused_features)
        rotation = self.rotation_head(fused_features)

        return translation[:, 0:1], translation[:, 1:2], rotation

    def forward_single_stream(self, x, extractor):
        # 1. Get raw multi-scale features (P3, P4, P5)
        p3, p4, p5 = self.extract_features_raw(x, extractor)

        # 2. Apply your 1x1 reductions and spatial pooling
        p3 = self.down_p3(self.conv_p3(p3)) # Output: (B, C/2, 20, 20)
        p4 = self.down_p4(self.conv_p4(p4)) # Output: (B, C/2, 5, 5)
        p5 = self.down_p5(self.conv_p5(p5)) # Output: (B, C/2, 5, 5)

        # 3. Flatten and concatenate the scales for this stream
        return torch.cat([p.view(x.size(0), -1) for p in [p3, p4, p5]], dim=1)
    
    def extract_features_raw(self, x, extractor):
        """
        Extracts P3, P4, P5 while respecting YOLO skip-connections.
        Raw = no 1x1 convs or pooling, just the direct outputs from the backbone/neck.
        """
        cache = []
        outputs = []
        
        # 'extractor' is your nn.Sequential (base_model[:-1])
        for i, layer in enumerate(extractor):
            # 1. Handle Skip Connections (The 'f' attribute)
            if layer.f != -1:  # If the input is NOT just the previous layer
                if isinstance(layer.f, int):
                    # Pull a single layer from the cache
                    x = cache[layer.f]
                else:
                    # Concatenate/Multiply multiple layers from the cache
                    # layer.f is a list like [-1, 6]
                    x = [x if j == -1 else cache[j] for j in layer.f]
            
            # 2. Forward pass through the layer
            x = layer(x)
            
            # 3. Store result in cache
            # IMPORTANT: Always append x to cache so indices match layer numbers
            cache.append(x)
            
            # 4. If this is a P3, P4, or P5 index, save it for our output
            if i in self.p345_indices:
                outputs.append(x)
            
        return outputs

    def forward_old(self, x_dict):
        """
        Args:
            x_dict: Dictionary mapping stream names (e.g., 'mono_cam') to tensors.
        Returns:
            Tensor: The final pose estimation (e.g., 7D vector).
        """
        stream_features = []

        for stream in self.streams:
            # 1. Get the input tensor for this stream
            x = x_dict[stream]
            extractor = self.feature_extractors[stream]

            # 2. Extract multi-scale raw features (P3, P4, P5)
            p3, p4, p5 = self.extract_features_raw(x, extractor)

            # 3. Apply Dimensionality Reduction (1x1 Convs)
            p3 = self.conv_p3(p3)
            p4 = self.conv_p4(p4)
            p5 = self.conv_p5(p5)

            # 4. Apply Spatial Pooling (Fixed Output Sizes)
            p3 = self.down_p3(p3) # Result: (B, C_new, 20, 20)
            p4 = self.down_p4(p4) # Result: (B, C_new, 5, 5)
            p5 = self.down_p5(p5) # Result: (B, C_new, 5, 5)

            # 5. Flatten and Concatenate scales for THIS stream
            # .view(x.size(0), -1) ensures we keep the Batch dimension
            feat_p3 = p3.view(p3.size(0), -1)
            feat_p4 = p4.view(p4.size(0), -1)
            feat_p5 = p5.view(p5.size(0), -1)
            
            combined_scales = torch.cat([feat_p3, feat_p4, feat_p5], dim=1)
            stream_features.append(combined_scales)

        # 6. Multi-Stream Fusion
        if len(stream_features) > 1:
            if self.fusion.method == "concat":
                fused_features = torch.cat(stream_features, dim=1)
            else:
                # For gated/attention, we fuse them iteratively or as a stack
                # Current FusionModule expects 2 inputs; for 3+ streams you'd need a loop
                fused_features = stream_features[0]
                for i in range(1, len(stream_features)):
                    fused_features = self.fusion(fused_features, stream_features[i])
        else:
            fused_features = stream_features[0]

        # 7. Final Prediction Head
        translation = self.translation_head(fused_features) # Output: [Batch, 2]
        rotation = self.rotation_head(fused_features)       # Output: [Batch, 4]

        # If your main block expects 3 outputs (y, z, q), 
        # you likely want to split translation into y and z.
        y_transl = translation[:, 0:1]
        z_transl = translation[:, 1:2]

        return y_transl, z_transl, rotation

def get_p345_indices(model):
    """
    Dynamically finds the indices of the P3, P4, and P5 feature maps.
    Works for YOLOv8, YOLO11, and YOLO26.
    """
    # The last layer in an Ultralytics detection model is typically the 'Detect' head
    # Its '.f' attribute contains the indices of the layers it receives (P3, P4, P5)
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        layers = model.model
    else:
        # It's already the Sequential list of layers
        layers = model

    # 2. The 'Detect' head is the very last layer (-1) 
    # but since you might have sliced it [:-1], we need the original full model 
    # or to look for the layer that has the '.f' attribute.
    detect_layer = None
    for layer in reversed(layers):
        if hasattr(layer, 'f'):
            detect_layer = layer
            break

    if detect_layer is not None:
        # Returns [P3_idx, P4_idx, P5_idx]
        return detect_layer.f 
    
    raise ValueError("Could not find a layer with 'f' attribute (Detect head) to determine indices.")


# -----------------------------
# Loss
# -----------------------------
class QuaternionLoss(nn.Module):
    def forward(self, pred, target):
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        dot = torch.sum(pred * target, dim=1)
        return (1 - dot ** 2).mean()


# -----------------------------
# Optimizer
# -----------------------------
def get_optimizer(model, lr=0.01, wd=0.0005):
    g_bnw, g_w, g_b = [], [], []

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            g_bnw.append(m.weight)
        elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
            g_w.append(m.weight)
        if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
            g_b.append(m.bias)

    optimizer = MuSGD(g_bnw + g_b, lr=lr, momentum=0.9, nesterov=True)
    optimizer.add_param_group({"params": g_w, "weight_decay": wd})
    return optimizer


# -----------------------------
# Main
# -----------------------------

def train_loop(model, data_loader, optimizer, loss_fn, device):
    """
    Standard training loop for one epoch.
    Args:
    - model: The PoseEstimationNet instance.
    - data_loader: DataLoader yielding batches of data.
    - optimizer: The optimizer instance.
    - loss_fn: The loss function instance (e.g., PoseLossManager).
    - device: The torch device (e.g., 'cuda' or 'cpu').
    """

    model.train()
    total_loss = 0  
    train_loss, train_loss_y, train_loss_z, train_loss_q = 0,0,0,0


    for batch in data_loader:
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        # Forward pass
        y_pred, z_pred, q_pred = model(batch)

        # Reconstruct the 7D Pose Vector [x, y, z, qx, qy, qz, qw]
        # Your loss function ignores x (index 0), so we use zeros for it.
        x_dummy = torch.zeros_like(y_pred)
        
        pose_pred = torch.cat([x_dummy, y_pred, z_pred, q_pred], dim=1)
        pose_true = batch["pose"].to(device)

        # Compute loss
        # transl_loss = F.mse_loss(y_pred, y_true) + F.mse_loss(z_pred, z_true)
        # rot_loss = loss_fn(q_pred, q_true)

        # Apply learned weighting: Loss = (1 / exp(log_var)) * L + log_var
        # This prevents the weights from going to zero and balances scales.
        # weighted_l_trans = transl_loss * torch.exp(-loss_fn.log_var_trans) + loss_fn.log_var_trans
        # weighted_l_rot = rot_loss * torch.exp(-loss_fn.log_var_rot) + loss_fn.log_var_rot   

        # total loss
        loss, loss_q, loss_y, loss_z = loss_fn(pose_true, pose_pred)
        
        train_loss += loss.item()
        train_loss_q += loss_q.item()
        train_loss_y += loss_y.item()
        train_loss_z += loss_z.item()
        
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)



def train_loop_old(model, data_loader, optimizer, loss_fn, device):
    """
    Standard training loop for one epoch.
    Args:
    - model: The PoseEstimationNet instance.
    - data_loader: DataLoader yielding batches of data.
    - optimizer: The optimizer instance.
    - loss_fn: The loss function instance (e.g., PoseLossManager).
    - device: The torch device (e.g., 'cuda' or 'cpu').
    """

    model.train()
    total_loss = 0
    for batch in data_loader:
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        y_pred, z_pred, q_pred = model(batch)
        y_true = batch['pose'][:, 1]
        z_true = batch['pose'][:, 2]
        q_true = batch['pose'][:, 3:7]

        # Compute loss
        # transl_loss = F.mse_loss(y_pred, y_true) + F.mse_loss(z_pred, z_true)
        # rot_loss = loss_fn(q_pred, q_true)

        # Apply learned weighting: Loss = (1 / exp(log_var)) * L + log_var
        # This prevents the weights from going to zero and balances scales.
        # weighted_l_trans = transl_loss * torch.exp(-loss_fn.log_var_trans) + loss_fn.log_var_trans
        # weighted_l_rot = rot_loss * torch.exp(-loss_fn.log_var_rot) + loss_fn.log_var_rot   

        # total loss
        loss = loss_fn(q_true, q_pred)


        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def test_loop(model, val_loader, loss_fn, device):
    """
    Standard evaluation loop for one epoch.
    Args:    
    - model: The PoseEstimationNet instance.
    - val_loader: DataLoader yielding batches of validation data.
    - loss_fn: The loss function instance (e.g., PoseLossManager).
    - device: The torch device (e.g., 'cuda' or 'cpu').
    """
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            y_pred, z_pred, q_pred = model(batch)
            x_dummy = torch.zeros_like(y_pred)
            pose_pred = torch.cat([x_dummy, y_pred, z_pred, q_pred], dim=1)
            
            pose_true = batch['pose']
            
            total_loss, _, _, _ = loss_fn(pose_true, pose_pred)
            total_val_loss += total_loss.item()

    return total_val_loss / len(val_loader)

if __name__ == "__main__":

    # 1. load and preprocess data 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_batch = {
        "mono_cam": torch.randn(4, 3, 640, 640),
        "tactile": torch.randn(4, 3, 640, 640),
        # "depth_cam": torch.randn(4, 4, 640, 640),
    }
    dummy_targets = {
        "y": torch.randn(4, 1),
        "z": torch.randn(4, 1),
        "q": F.normalize(torch.randn(4, 4), dim=1)
    }


    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir="Data/Data_split",
        batch_size=3,
        shuffle=True
    )

    # checkpoint and logging
    checkpoint_path = "logging/pose_estimation_checkpoint.pt"
    logging_dir = "logging/logging.txt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(logging_dir), exist_ok=True)

    # 2. create model, optimizer, loss function
    model = PoseEstimationNet(streams=("rgb_image", "digit_image"), fusion="concat").to(device)

    optim = get_optimizer(model)
    criterion = total_loss  # This is your custom loss function that includes learned weighting

    epochs = 25

    # 3. Training Loop

    for epoch in range(epochs):
        loss = train_loop(model, train_loader, optim, criterion, device)
        val_loss = test_loop(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")


    # 4. Testing Loop
    test_loss_total = test_loop(model, test_loader, criterion, device)

    # dummy_y, dummy_z, dummy_q = model(batch)
    # print(dummy_y.shape, dummy_z.shape, dummy_q.shape)





