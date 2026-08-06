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
from fusion.adapters import TaskSpecificResidualGate, TaskSpecificChannelAttention


class PoseEstimationNet(nn.Module):
    def __init__(self, model_name='yolo26n.pt', streams=("mono_cam", "tactile"), fusion="concat", freeze_backbone=False, rotation_parametrisation="quaternion", activation=nn.GELU(),
                 rotation_layers = [256, 1024, 16], translation_layers = [1024, 1024, 0], dropout_rot = 0.0, dropout_trans = 0.0, use_batchnorm = True, task_adapter=None,
                 task_feature_adapter=False, task_feature_adapter_type="channel_attention", dual_fusion=False, translation_head_mode="separate",
                 spatial_grid_size=(10, 10)):
        super().__init__()

        if dual_fusion and rotation_parametrisation == "se3_direct":
            raise ValueError(
                "dual_fusion is incompatible with head_type 'se3_direct': the SE(3) direct "
                "head consumes a single joint feature vector, but dual_fusion produces two "
                "independent (rotation/translation) fused vectors with no shared 'fused' tensor."
            )

        if translation_head_mode not in ("separate", "joint"):
            raise ValueError(
                f"Unsupported translation_head_mode: {translation_head_mode}. "
                f"Supported options: separate, joint."
            )
        self.translation_head_mode = translation_head_mode

        self.streams = streams
        self.fusion_type = fusion
        self.dual_fusion = bool(dual_fusion)
        self.task_adapter_type = task_adapter
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

        # Learned stride-2 downsampling layers (from colleague's original design).
        # Two stride-2 convs per scale to progressively reduce spatial dimensions:
        # P3: 80×80 -> 40×40 -> 20×20
        # P4: 40×40 -> 20×20 -> 10×10 (then will further reduce to 5×5)
        # P5: 20×20 -> 10×10 -> 5×5
        # These are learned, allowing the network to decide what structure to preserve during downsampling.
        self.stride_p3_1 = nn.Conv2d(c3//2, c3//2, kernel_size=3, stride=2, padding=1)
        self.stride_p3_2 = nn.Conv2d(c3//2, c3//2, kernel_size=3, stride=2, padding=1)

        self.stride_p4_1 = nn.Conv2d(c4//2, c4//2, kernel_size=3, stride=2, padding=1)
        self.stride_p4_2 = nn.Conv2d(c4//2, c4//2, kernel_size=3, stride=2, padding=1)

        self.stride_p5_1 = nn.Conv2d(c5//2, c5//2, kernel_size=3, stride=2, padding=1)
        self.stride_p5_2 = nn.Conv2d(c5//2, c5//2, kernel_size=3, stride=2, padding=1)

        # 2. Fusion module
        # Determine spatial dimensions
        self.is_2d_fusion = (fusion == "spatial_gated_2d")

        if self.is_2d_fusion:
            # We treat the unified pyramid channel footprint as our 'dim'
            # Channels: (c3//2) + (c4//2) + (c5//2)
            self.total_spatial_channels = (c3//2) + (c4//2) + (c5//2)

            def build_post_conv():
                return nn.Sequential(
                    nn.Conv2d(self.total_spatial_channels, self.total_spatial_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.total_spatial_channels) if use_batchnorm else nn.Identity(),
                    nn.GELU()
                )

            if self.dual_fusion:
                # Independent fusion module + post-conv per task branch: RGB/tactile
                # features are combined separately for rotation and for translation,
                # each with its own learned parameters (no sharing between branches).
                self.fusion_rotation = build_fusion(fusion, self.total_spatial_channels, len(streams))
                self.fusion_translation = build_fusion(fusion, self.total_spatial_channels, len(streams))
                self.fusion_post_conv_rotation = build_post_conv()
                self.fusion_post_conv_translation = build_post_conv()
                fusion_for_aux_check = self.fusion_rotation
            else:
                # Instantiate the 2D fusion via registry
                self.fusion = build_fusion(fusion, self.total_spatial_channels, len(streams))
                # Post-fusion convolution to mix features spatially across scales
                self.fusion_post_conv = build_post_conv()
                fusion_for_aux_check = self.fusion

            # Unified grid size maps directly to the linear layers
            fused_dim = self.total_spatial_channels * self.spatial_grid_size[0] * self.spatial_grid_size[1]
        else:
            with torch.no_grad():
                dummy_feat = self.forward_single_stream(dummy_input, extractor)
                feat_dim = dummy_feat.shape[1]

            # Concatenation-preserving fusions retain one complete vector per stream.
            concat_output_fusions = {
                "concat",
                "gated_concat",
                "channelwise_gated_concat",
                "residual_gated_concat",
                "channelwise_residual_gated_concat",
            }

            if self.dual_fusion:
                # Independent fusion module per task branch (own parameters), fed
                # from the same shared per-stream backbone features.
                self.fusion_rotation = build_fusion(fusion, feat_dim, len(streams))
                self.fusion_translation = build_fusion(fusion, feat_dim, len(streams))
                fusion_for_aux_check = self.fusion_rotation
            else:
                # build_fusion returns a single fusion module instance as specified by the config
                self.fusion = build_fusion(fusion, feat_dim, len(streams))
                fusion_for_aux_check = self.fusion

            fused_dim = feat_dim * len(streams) if fusion in concat_output_fusions else feat_dim

        self._fusion_accepts_aux_losses = "aux_losses" in inspect.signature(fusion_for_aux_check.forward).parameters
        self.weights = None
        self.last_fusion_weights = None
        self.last_fusion_targets = None
        self.rot_adapter = None
        self.trans_adapter = None
        self.rot_adapter_activations = None
        self.trans_adapter_activations = None

        if task_adapter:
            if task_adapter == "task_specific_residual_gate":
                self.rot_adapter = TaskSpecificResidualGate(fused_dim)
                self.trans_adapter = TaskSpecificResidualGate(fused_dim)
            elif task_adapter == "task_specific_channel_attention":
                self.rot_adapter = TaskSpecificChannelAttention(fused_dim)
                self.trans_adapter = TaskSpecificChannelAttention(fused_dim)
            else:
                raise ValueError(f"Unknown task adapter: {task_adapter}")
            self.task_adapter_type = task_adapter

        # Optional task-specific feature adaptation applied after fusion, right before
        # the rotation/translation heads. These two adapters are independent instances
        # (no shared parameters) and do not replace the legacy `rot_adapter`/
        # `trans_adapter` mechanism above, which is kept for backward compatibility.
        # When disabled (default), no adapters are created and the model behaves
        # exactly like the original implementation.
        self.task_feature_adapter_enabled = bool(task_feature_adapter)
        self.rotation_adapter = None
        self.translation_adapter = None
        self.rotation_attention = None
        self.translation_attention = None

        if self.task_feature_adapter_enabled:
            if task_feature_adapter_type == "channel_attention":
                self.rotation_adapter = TaskSpecificChannelAttention(fused_dim)
                self.translation_adapter = TaskSpecificChannelAttention(fused_dim)
            elif task_feature_adapter_type == "residual_gate":
                self.rotation_adapter = TaskSpecificResidualGate(fused_dim)
                self.translation_adapter = TaskSpecificResidualGate(fused_dim)
            else:
                raise ValueError(f"Unknown task_feature_adapter_type: {task_feature_adapter_type}")
        self.task_feature_adapter_type = task_feature_adapter_type


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
        
        # Translation Head(s) (only for non-SE3-direct modes)
        if rotation_parametrisation != "se3_direct":
            if self.translation_head_mode == "joint":
                self.trans_head = MLPHead(
                    in_dim=fused_dim,
                    out_dim=2,
                    hidden_dims=trans_hidden,
                    dropout=dropout_trans,
                    bn=use_batchnorm,
                    activation=self.activation
                )
            else:
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

        # 1×1 channel reduction, then two stride-2 convolutions per scale for learned downsampling
        p3 = self.stride_p3_2(self.stride_p3_1(self.conv_p3(p3)))  # 80×80 -> 40×40 -> 20×20
        p4 = self.stride_p4_2(self.stride_p4_1(self.conv_p4(p4)))  # 40×40 -> 20×20 -> 10×10
        p5 = self.stride_p5_2(self.stride_p5_1(self.conv_p5(p5)))  # 20×20 -> 10×10 -> 5×5

        return torch.cat([p.flatten(1) for p in [p3, p4, p5]], dim=1)

    def _unpack_fusion_output(self, fusion_out):
        if not isinstance(fusion_out, tuple):
            return fusion_out, None, None
        if len(fusion_out) == 2:
            fused, fusion_weights = fusion_out
            return fused, fusion_weights, None
        if len(fusion_out) == 3:
            return fusion_out
        raise ValueError(
            f"Fusion module '{self.fusion_type}' returned {len(fusion_out)} values; expected 1, 2, or 3."
        )

    def forward(self, x_dict, aux_losses=None):
        self.last_fusion_weights = None
        self.last_fusion_targets = None

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

            if self.dual_fusion:
                # Independent fusion + post-conv per task branch, both fed from the
                # same backbone feature maps computed above.
                fused_spatial_rot, gate_weights_rot = self.fusion_rotation(stream_p3s)
                fused_spatial_trans, gate_weights_trans = self.fusion_translation(stream_p3s)

                fused_rot = self.fusion_post_conv_rotation(fused_spatial_rot).flatten(1)
                fused_trans = self.fusion_post_conv_translation(fused_spatial_trans).flatten(1)

                self.last_fusion_weights = {"rotation": gate_weights_rot, "translation": gate_weights_trans}
                avg_fusion_weights = {
                    "rotation": gate_weights_rot.mean(dim=(0, 2, 3)),
                    "translation": gate_weights_trans.mean(dim=(0, 2, 3)),
                }
            else:
                # Call your registry module with a list of 2D Tensors!
                fusion_out = self.fusion(stream_p3s)
                fused_spatial, gate_weights = fusion_out  # gate_weights shape: [B, num_streams, 10, 10]

                # Mix structural elements locally, then flatten for heads
                fused_spatial = self.fusion_post_conv(fused_spatial)
                fused_rot = fused_spatial.flatten(1)
                fused_trans = fused_rot

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

            if self.dual_fusion:
                # Independent fusion module per task branch, both fed from the same
                # per-stream backbone features computed above.
                if aux_losses is not None and self._fusion_accepts_aux_losses:
                    fusion_out_rot = self.fusion_rotation(features, aux_losses=aux_losses)
                    fusion_out_trans = self.fusion_translation(features, aux_losses=aux_losses)
                else:
                    fusion_out_rot = self.fusion_rotation(features)
                    fusion_out_trans = self.fusion_translation(features)

                fused_rot, fusion_weights_rot, fusion_targets_rot = self._unpack_fusion_output(fusion_out_rot)
                fused_trans, fusion_weights_trans, fusion_targets_trans = self._unpack_fusion_output(fusion_out_trans)

                avg_fusion_weights = {
                    "rotation": fusion_weights_rot.mean(dim=0) if fusion_weights_rot is not None else None,
                    "translation": fusion_weights_trans.mean(dim=0) if fusion_weights_trans is not None else None,
                }
                # Retain raw per-example gates/targets for epoch statistics, keyed by branch.
                self.last_fusion_weights = {"rotation": fusion_weights_rot, "translation": fusion_weights_trans}
                self.last_fusion_targets = {"rotation": fusion_targets_rot, "translation": fusion_targets_trans}
            else:
                # Fuse features (Shape: [Batch, Fused_Dim])
                if aux_losses is not None and self._fusion_accepts_aux_losses:
                    fusion_out = self.fusion(features, aux_losses=aux_losses)
                else:
                    fusion_out = self.fusion(features)

                fused, fusion_weights, fusion_targets = self._unpack_fusion_output(fusion_out)
                avg_fusion_weights = fusion_weights.mean(dim=0) if fusion_weights is not None else None
                fused_rot = fused
                fused_trans = fused

                # Retain raw per-example gates for epoch statistics without changing the
                # public return value, which remains the mean gate for compatibility.
                self.last_fusion_weights = fusion_weights
                self.last_fusion_targets = fusion_targets

        # Handle SE(3) direct optimization
        if self.rotation_parametrisation == "se3_direct":
            # SE(3) direct: outputs 6D [tx, ty, tz, ωx, ωy, ωz]. Not reachable with
            # dual_fusion=True: __init__ rejects that combination since this head
            # needs one joint feature vector, not separate rot/trans fused vectors.
            se3_pose = self.se3_direct_head(fused_rot)

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
            rot_adapter_activations = None
            trans_adapter_activations = None

            if self.rot_adapter:
                fused_rot, self.rot_adapter_activations = self.rot_adapter(fused_rot)
            if self.trans_adapter:
                fused_trans, self.trans_adapter_activations = self.trans_adapter(fused_trans)

            # Optional task-specific feature adaptation: two independent adapters
            # (no shared parameters) applied right before the rotation/translation
            # heads. When disabled, behavior is unchanged (identity passthrough).
            rotation_features = fused_rot
            translation_features = fused_trans
            if self.rotation_adapter is not None:
                rotation_features, self.rotation_attention = self.rotation_adapter(fused_rot)
            if self.translation_adapter is not None:
                translation_features, self.translation_attention = self.translation_adapter(fused_trans)

            # Run through separate heads
            rotation = self.rot_head(rotation_features)
            if self.rotation_parametrisation == "quaternion":
                rotation = torch.nn.functional.normalize(rotation, p=2, dim=1, eps=1e-12)
        
            if self.translation_head_mode == "joint":
                trans = self.trans_head(translation_features)
                y, z = trans[:, 0:1], trans[:, 1:2]
            else:
                y = self.y_head(translation_features)
                z = self.z_head(translation_features)

            # Return as a dictionary or concatenated vector, x=0
            x_coord = torch.zeros_like(y)
            translation = torch.cat([x_coord, y, z], dim=-1)

            output = {
                "rotation": rotation,   # Shape: [B, 4] for quat
                "trans": translation,    # Shape: [B, 3]
                "pose": torch.cat([translation, rotation], dim=-1)  # Shape: [B, 7]
            }

            # Optional attention logging: only present when task-specific feature
            # adaptation is enabled. When disabled, the output dict is unchanged
            # (no extra key added), preserving full backward compatibility. When
            # enabled, callers that want to log the attentions can look up this
            # key; callers (e.g. the existing training loop) that don't know about
            # it simply ignore the extra key, so no modification is required there.
            if self.task_feature_adapter_enabled:
                output["task_feature_adapter_attention"] = {
                    "rotation_attention": self.rotation_attention,
                    "translation_attention": self.translation_attention,
                }

            return output, avg_fusion_weights  # return mean fusion weights for logging
