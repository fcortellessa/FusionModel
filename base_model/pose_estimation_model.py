"""
This module defines the PoseEstimationNet class, which implements a multi-stream pose estimation model using a YOLO backbone for feature extraction and a custom head for pose regression. 
The model can handle multiple input streams (e.g., monocular RGB, tactile, depth) and supports different fusion strategies to combine features from these streams. 
The head can be configured to output poses in different formats (e.g., quaternion, 2D unit vector, SE3).
Key components:
- YOLO Backbone: The model uses a pretrained YOLO model as the backbone for feature extraction. The P3, P4, and P5 feature maps are extracted while respecting the skip connections defined in the YOLO architecture.
- Fusion Module: A configurable fusion module combines the features from different streams. The type of fusion
    (e.g., concatenation, attention-based) can be specified in the configuration.
- Pose Head: The head of the model takes the fused features and outputs the predicted pose. Different head architectures can be used depending on the desired output format for the pose.
The forward pass of the model processes each input stream through its respective backbone, extracts and processes the P3, P4, P5 features, fuses them, and then passes the fused features through the head to get the final pose prediction.


ToDo:
1. Learning Rate Scheduler: Implement a learning rate scheduler that adjusts the learning rate during training based on validation performance or a predefined schedule. This can help improve convergence and prevent overfitting.
2. Dropout and Regularization: Add dropout layers and other regularization techniques to the model to prevent overfitting, especially when training on smaller datasets.
3. Change optimizer: Experiment with different optimizers (e.g., AdamW, RMSprop) to see if they improve training stability and convergence.

"""
import os
import sys
import inspect
# from logging import config

import torch
import torch.nn as nn
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from base_model.yolo_trunk import load_yolo_trunk, get_p345_indices, align_channels, extract_features_raw
from base_model.mlp_head import MLPHead, PoseHead_with_quaternion, PoseHead_with_2D_unitVector, SE3_lie_group_PoseHead, PoseHead_with_angle, PoseHead_with_lie_algebra, SE3_lie_algebra_direct_PoseHead
from fusion.registry import build_fusion


class PoseEstimationNet(nn.Module):
    def __init__(self, model_name='yolo26n.pt', streams=("mono_cam", "tactile"), fusion="concat", freeze_backbone=False, rotation_parametrisation="quaternion", activation=nn.GELU(), 
                 rotation_layers = [256, 1024, 16], translation_layers = [1024, 1024, 0], dropout_rot = 0.0, dropout_trans = 0.0, use_batchnorm = True):
        super().__init__()

        self.streams = streams
        self.fusion_type = fusion
        self.feature_extractors = nn.ModuleDict()
        self.in_channels = {}

        # 1. YOLO feature extractor setup
        full, base_model = load_yolo_trunk(model_name)
        self.p345_indices = get_p345_indices(full)

        for stream in streams:
            extractor = deepcopy(base_model)

            if stream == "depth_cam":
                extractor = align_channels(extractor, 4)
                self.in_channels[stream] = 4
            else:
                self.in_channels[stream] = 3
            
            if freeze_backbone:
                for p in extractor.parameters():
                    p.requires_grad = False

            self.feature_extractors[stream] = extractor

        # infer dimensions
        with torch.no_grad():
            stream0 = streams[0]
            dummy_input = torch.zeros(1, self.in_channels[stream0], 640, 640)
            feats = extract_features_raw(dummy_input, self.feature_extractors[stream0], self.p345_indices)

        c3, c4, c5 = [f.shape[1] for f in feats]

        self.conv_p3 = nn.Conv2d(c3, c3//2, 1)
        self.conv_p4 = nn.Conv2d(c4, c4//2, 1)
        self.conv_p5 = nn.Conv2d(c5, c5//2, 1)

        self.spatial_grid_size = (10, 10)
        self.down_p3 = nn.AdaptiveAvgPool2d(self.spatial_grid_size)
        self.down_p4 = nn.AdaptiveAvgPool2d(self.spatial_grid_size)
        self.down_p5 = nn.AdaptiveAvgPool2d(self.spatial_grid_size)

        # 2. Fusion module
        # Determine spatial dimensions
        self.is_2d_fusion = (fusion == "spatial_gated_2d")

        if self.is_2d_fusion:
            # We treat the unified pyramid channel footprint as our 'dim'
            # Channels: (c3//2) + (c4//2) + (c5//2)
            self.total_spatial_channels = (c3//2) + (c4//2) + (c5//2)
            
            # Instantiate the 2D fusion via registry
            self.fusion = build_fusion(fusion, self.total_spatial_channels, len(streams))
            
            # Post-fusion convolution to mix features spatially across scales
            self.fusion_post_conv = nn.Sequential(
                nn.Conv2d(self.total_spatial_channels, self.total_spatial_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.total_spatial_channels) if use_batchnorm else nn.Identity(),
                nn.GELU()
            )
            # Unified grid size maps directly to the linear layers
            fused_dim = self.total_spatial_channels * self.spatial_grid_size[0] * self.spatial_grid_size[1]
        else:
            with torch.no_grad():
                dummy_feat = self.forward_single_stream(dummy_input, extractor)
                feat_dim = dummy_feat.shape[1]

            # build_fusion returns a single fusion module instance as specified by the config
            self.fusion = build_fusion(fusion, feat_dim, len(streams))
            # Concatenation-preserving fusions retain one complete vector per stream.
            concat_output_fusions = {
                "concat",
                "gated_concat",
                "residual_gated_concat",
                "channelwise_residual_gated_concat",
            }
            fused_dim = feat_dim * len(streams) if fusion in concat_output_fusions else feat_dim

        self._fusion_accepts_aux_losses = "aux_losses" in inspect.signature(self.fusion.forward).parameters
        self.weights = None
        self.last_fusion_weights = None
        self.last_fusion_targets = None


        # 3. Prediction Heads
        # Helper to process config values into a list
        def to_list(dims):
            return [d for d in dims if isinstance(d, int) and d > 0]
         
        self.activation = nn.GELU()
        self.rotation_parametrisation = rotation_parametrisation
        rot_hidden = to_list(rotation_layers)
        trans_hidden = to_list(translation_layers)

        # Rotation Head
        if rotation_parametrisation == "quaternion":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=4,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation
            )
        elif rotation_parametrisation == "2D_unit_vector":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=2,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation
            )
        elif rotation_parametrisation == "angle":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=1,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation
            )
        elif rotation_parametrisation == "lie_algebra":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=3,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation
            )
        elif rotation_parametrisation == "se3_direct":
            # SE(3) direct head outputs complete 6D SE(3) Lie algebra element
            self.se3_direct_head = SE3_lie_algebra_direct_PoseHead(
                in_dim=fused_dim
            )
        else:
            raise ValueError(f"Unsupported rotation parametrisation: {rotation_parametrisation}")
        
        # Translation Heads (only for non-SE3-direct modes)
        if rotation_parametrisation != "se3_direct":
            self.y_head = MLPHead(
                in_dim=fused_dim,
                out_dim=1,
                hidden_dims=trans_hidden,
                dropout=dropout_trans,
                bn=use_batchnorm,
                activation=self.activation
            )
            self.z_head = MLPHead(
                in_dim=fused_dim,
                out_dim=1,
                hidden_dims=trans_hidden,
                dropout=dropout_trans,
                bn=use_batchnorm,
                activation=self.activation
            )



    def forward_single_stream(self, x, extractor):
        p3, p4, p5 = extract_features_raw(x, extractor, self.p345_indices)

        p3 = self.down_p3(self.conv_p3(p3))
        p4 = self.down_p4(self.conv_p4(p4))
        p5 = self.down_p5(self.conv_p5(p5))

        return torch.cat([p.flatten(1) for p in [p3, p4, p5]], dim=1)


    def forward(self, x_dict, aux_losses=None):
        self.last_fusion_weights = None
        if self.is_2d_fusion:
            # --- 2D SPATIAL FUSION PIPELINE ---
            stream_p3s, stream_p4s, stream_p5s = [], [], []
            
            for stream in self.streams:
                x = x_dict[stream]
                extractor = self.feature_extractors[stream]
                
                # Extract raw features from backbone
                p3, p4, p5 = extract_features_raw(x, extractor, self.p345_indices)
                
                # Align maps to a unified spatial layout
                p3_f = self.down_p3(self.conv_p3(p3))
                p4_f = self.down_p4(self.conv_p4(p4))
                p5_f = self.down_p5(self.conv_p5(p5))
                
                # Concatenate scales per-stream to build a 2D spatial block
                spatial_block = torch.cat([p3_f, p4_f, p5_f], dim=1) # [B, total_spatial_channels, H, W]
                
                # Append the constructed spatial map
                stream_p3s.append(spatial_block) # Reusing list to store blocks

            # Call your registry module with a list of 2D Tensors!
            fusion_out = self.fusion(stream_p3s)
            fused_spatial, gate_weights = fusion_out  # gate_weights shape: [B, num_streams, 10, 10]
            
            # Mix structural elements locally, then flatten for heads
            fused_spatial = self.fusion_post_conv(fused_spatial)
            fused = fused_spatial.flatten(1)
            
            # Log average spatial weights across the 10x10 coordinates
            avg_fusion_weights = gate_weights.mean(dim=(0, 2, 3)) 
            self.last_fusion_weights = gate_weights

        else:
            # --- 1D FUSION PIPELINE ---
            features = []

            for stream in self.streams:
                x = x_dict[stream]
                extractor = self.feature_extractors[stream]
                feat = self.forward_single_stream(x, extractor)
                features.append(feat)

            # Fuse features (Shape: [Batch, Fused_Dim])
            if aux_losses is not None and self._fusion_accepts_aux_losses:
                fusion_out = self.fusion(features, aux_losses=aux_losses)
            else:
                fusion_out = self.fusion(features)

            fusion_targets = None
            if isinstance(fusion_out, tuple):
                if len(fusion_out) == 2:
                    fused, fusion_weights = fusion_out
                elif len(fusion_out) == 3:
                    fused, fusion_weights, fusion_targets = fusion_out
                else:
                    raise ValueError(
                        f"Fusion module '{self.fusion_type}' returned {len(fusion_out)} values; expected 1, 2, or 3."
                    )
                avg_fusion_weights = fusion_weights.mean(dim=0) if fusion_weights is not None else None
            else:
                fused = fusion_out
                avg_fusion_weights = None

            # Retain raw per-example gates for epoch statistics without changing the
            # public return value, which remains the mean gate for compatibility.
            self.last_fusion_weights = fusion_weights if isinstance(fusion_out, tuple) else None
            self.last_fusion_targets = fusion_targets

        # Handle SE(3) direct optimization
        if self.rotation_parametrisation == "se3_direct":
            # SE(3) direct: outputs 6D [tx, ty, tz, ωx, ωy, ωz]
            se3_pose = self.se3_direct_head(fused)
            
            # Extract translation and rotation for compatibility
            translation = se3_pose[:, 0:3]  # [x, y, z]
            rotation = se3_pose[:, 3:6]      # [ωx, ωy, ωz] - rotation vector
            
            return {
                "se3": se3_pose,             # Shape: [B, 6] complete SE(3)
                "rotation": rotation,        # Shape: [B, 3] rotation vector
                "trans": translation,        # Shape: [B, 3]
                "pose": se3_pose             # Shape: [B, 6] for SE(3)
            }, avg_fusion_weights
        else:
            # Run through separate heads
            rotation = self.rot_head(fused)
            if self.rotation_parametrisation == "quaternion":
                rotation = torch.nn.functional.normalize(rotation, p=2, dim=1, eps=1e-12)
            
            y = self.y_head(fused)
            z = self.z_head(fused)

            # Return as a dictionary or concatenated vector, x=0
            x_coord = torch.zeros_like(y)
            translation = torch.cat([x_coord, y, z], dim=-1)

            return {
                "rotation": rotation,   # Shape: [B, 4] for quat
                "trans": translation,    # Shape: [B, 3]
                "pose": torch.cat([translation, rotation], dim=-1)  # Shape: [B, 7]
            }, avg_fusion_weights  # return mean fusion weights for logging
