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
                 spatial_grid_size=(10, 10), gate_init_prob=None, backbone_trainable=False, pre_head_grid_size=None,
                 pre_head_layernorm=False, head_bottleneck_rank=None, pre_head_channel_divisor=None):
        super().__init__()

        # Optional additional AdaptiveAvgPool2d applied to P3/P4/P5 (all three, same
        # target size) right before flattening into the fused vector, for the 1D
        # pipeline only (forward_single_stream) -- shrinks the bottleneck feeding the
        # prediction heads, whose first layer is otherwise a single dense projection
        # out of the full flattened size (e.g. rot_head's first layer is ~183.5M of
        # its ~192.9M total parameters at the current fixed 20x20/10x10/5x5 sizes).
        #
        # This is a NEW, separate mechanism -- NOT a fix/wiring of the existing
        # `spatial_grid_size` parameter above. spatial_grid_size is referenced only
        # in the (dead, unreachable) 2D-fusion fused_dim calculation further below,
        # which itself refers to `self.spatial_grid_size`, an attribute that is
        # never actually assigned anywhere in this class -- so spatial_grid_size has
        # never affected this (1D pipeline / gated_concat family) forward path at
        # all, regardless of its config value. Every existing config sets
        # spatial_grid_size: 10 without it ever doing anything for gated_concat/
        # channelwise_gated_concat*; reusing that key for this new, real mechanism
        # would silently change what "10" has always (inertly) meant. Hence a new,
        # separate name here instead.
        #
        # None (default) skips the extra pooling entirely -- fused_dim stays exactly
        # what the fixed stride convs above already produce, identical to every
        # existing run. Accepts a single int (square grid, e.g. 6 -> AdaptiveAvgPool2d(6))
        # or a [h, w] pair, matching spatial_grid_size's own documented convention.
        if pre_head_grid_size is None:
            self.pre_head_pool = None
        else:
            if isinstance(pre_head_grid_size, int):
                pre_head_grid_size = (pre_head_grid_size, pre_head_grid_size)
            else:
                pre_head_grid_size = tuple(pre_head_grid_size)
            self.pre_head_pool = nn.AdaptiveAvgPool2d(pre_head_grid_size)

        # Optional low-rank factorization of the head's first Linear layer -- see
        # MLPHead's bottleneck_rank docstring. None (default) is unchanged/backward
        # compatible; stored here and threaded into every MLPHead(...) construction
        # below (rot_head, trans_head/y_head/z_head).
        self.head_bottleneck_rank = head_bottleneck_rank

        # Optional channel reduction (1x1 conv) applied to P3/P4/P5 right before flatten,
        # on top of the existing conv_p3/p4/p5 halving -- unlike pre_head_grid_size (which
        # shrinks the head's input by discarding SPATIAL resolution, and empirically hurt
        # accuracy -- see final_ablation's paired-comparison results), this shrinks it via
        # the CHANNEL axis instead, leaving every spatial position intact. Each scale's
        # post-stride-conv channel count c (P3:32, P4:64, P5:128 by default) becomes
        # max(1, c // pre_head_channel_divisor). None (default, divisor unset) skips this
        # entirely -- identical to every existing run/config.
        self.pre_head_channel_divisor = pre_head_channel_divisor

        # New, explicitly-named opt-in for actually training the backbone (see the
        # extended comment at the freeze/unfreeze site below for why this exists
        # separately from freeze_backbone, which stays pinned to its historical --
        # always-frozen -- behavior for backward compatibility). Default False
        # preserves that historical behavior exactly; True is the first time this
        # project would ever actually fine-tune the backbone.
        self.backbone_trainable = bool(backbone_trainable)
        if freeze_backbone and self.backbone_trainable:
            raise ValueError(
                "freeze_backbone=True and backbone_trainable=True are contradictory: "
                "freeze_backbone historically has no unfreezing effect (see the "
                "feature_extractors setup below) and backbone_trainable is the only "
                "thing that can actually enable backbone gradients. Set "
                "freeze_backbone=False when using backbone_trainable=True."
            )

        # Optional initial sigmoid gate output for the gated_concat fusion family (see
        # fusion/gated_concat.py's _sigmoid_init_bias). None (default) leaves every
        # accepting fusion class at its own default of 0.5, identical to the previous
        # zero-bias init -- fully backward compatible. Ignored by fusion types that
        # don't expose a gate_init_prob kwarg (concat, attention, the softmax/sigmoid
        # weighted-sum "gated" family, spatial_gated_2d).
        self.gate_init_prob = gate_init_prob

        if dual_fusion and rotation_parametrisation == "se3_direct":
            raise ValueError(
                "dual_fusion is incompatible with head_type 'se3_direct': the SE(3) direct "
                "head consumes a single joint feature vector, but dual_fusion produces two "
                "independent (rotation/translation) fused vectors with no shared 'fused' tensor."
            )

        if pre_head_layernorm and rotation_parametrisation == "se3_direct":
            raise ValueError(
                "pre_head_layernorm is incompatible with head_type 'se3_direct': that head is "
                "consumed on a separate return path (forward()'s se3_direct branch) that never "
                "reaches the rotation/translation-adapter site where pre_head_layernorm is applied."
            )
        self.pre_head_layernorm = bool(pre_head_layernorm)

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
            
            # NOTE: ultralytics' YOLO(model_name) loader returns every parameter with
            # requires_grad=False already set (confirmed empirically -- YOLO('yolo26n.pt')
            # loads with 0/366 params trainable). freeze_backbone historically only ever
            # set requires_grad=False (when True) and never set it back to True when
            # False -- so every run in this repo's history that used freeze_backbone=False
            # (the near-universal default, incl. configs/default.yml and every fusion/gate
            # ablation this project has run) has actually trained with a FULLY FROZEN
            # backbone regardless of that flag's value. freeze_backbone therefore stays a
            # no-op below (both True and False produce an all-frozen backbone), preserving
            # exact backward compatibility with every existing run/comparison.
            #
            # backbone_trainable (new, default False) is the only thing that can actually
            # enable backbone gradients -- an explicit, separately-named opt-in rather than
            # a reinterpretation of freeze_backbone, so turning it on is a deliberate choice
            # that doesn't retroactively change what freeze_backbone=False has always meant.
            for p in extractor.parameters():
                p.requires_grad = self.backbone_trainable

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

        # Optional additional 1x1-conv channel reduction, applied after the stride convs
        # above (i.e. at the final 20x20/10x10/5x5 resolution) -- see
        # pre_head_channel_divisor's docstring at its __init__ argument. None (divisor
        # unset, default) leaves these as plain pass-throughs (nn.Identity()), so
        # forward_single_stream's behavior and every existing checkpoint are unaffected.
        def _channel_reduce(channels):
            if self.pre_head_channel_divisor is None:
                return nn.Identity()
            return nn.Conv2d(channels, max(1, channels // self.pre_head_channel_divisor), 1)

        self.pre_head_channel_conv_p3 = _channel_reduce(c3 // 2)
        self.pre_head_channel_conv_p4 = _channel_reduce(c4 // 2)
        self.pre_head_channel_conv_p5 = _channel_reduce(c5 // 2)

        # Per-scale flattened sizes within the 1D per-stream feature vector, i.e. the
        # boundaries of the [P3 | P4 | P5] blocks that forward_single_stream() later
        # concatenates (torch.cat([p.flatten(1) for p in [p3, p4, p5]], dim=1)).
        # Lets fusion-gate diagnostics (see scripts/train.py) break a channel-wise gate
        # down by which pyramid scale each channel came from, not just its raw index.
        # Includes pre_head_channel_conv_p3/p4/p5 so these dims reflect the reduced
        # channel counts when pre_head_channel_divisor is set (forward_single_stream
        # applies them at the same point, before flatten).
        with torch.no_grad():
            p3_d = self.pre_head_channel_conv_p3(self.stride_p3_2(self.stride_p3_1(self.conv_p3(feats[0]))))
            p4_d = self.pre_head_channel_conv_p4(self.stride_p4_2(self.stride_p4_1(self.conv_p4(feats[1]))))
            p5_d = self.pre_head_channel_conv_p5(self.stride_p5_2(self.stride_p5_1(self.conv_p5(feats[2]))))
        self.stream_scale_dims = {
            "p3": p3_d.shape[1] * p3_d.shape[2] * p3_d.shape[3],
            "p4": p4_d.shape[1] * p4_d.shape[2] * p4_d.shape[3],
            "p5": p5_d.shape[1] * p5_d.shape[2] * p5_d.shape[3],
        }
        # Channel count of each scale's block (the block is channel-major: C x H x W flattened),
        # so per-scale size // channels = grid cells per channel. Used by se_channel_* fusions.
        self.stream_scale_channels = {
            "p3": p3_d.shape[1],
            "p4": p4_d.shape[1],
            "p5": p5_d.shape[1],
        }

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
                self.fusion_rotation = build_fusion(fusion, self.total_spatial_channels, len(streams), gate_init_prob=self.gate_init_prob)
                self.fusion_translation = build_fusion(fusion, self.total_spatial_channels, len(streams), gate_init_prob=self.gate_init_prob)
                self.fusion_post_conv_rotation = build_post_conv()
                self.fusion_post_conv_translation = build_post_conv()
                fusion_for_aux_check = self.fusion_rotation
            else:
                # Instantiate the 2D fusion via registry
                self.fusion = build_fusion(fusion, self.total_spatial_channels, len(streams), gate_init_prob=self.gate_init_prob)
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
                "scale_gated_concat",
                "se_channel_gated_concat",
                "se_channel_gated_concat_renorm",
                "channelwise_gated_concat",
                "channelwise_gated_concat_wide",
                "channelwise_gated_concat_local",
                "channelwise_gated_concat_hybrid",
                "residual_gated_concat",
                "channelwise_residual_gated_concat",
                "channelwise_residual_gated_concat_wide",
                "channelwise_residual_gated_concat_local",
                "channelwise_residual_gated_concat_hybrid",
                "gated_concat_floor",
                "channelwise_gated_concat_floor",
            }

            if self.dual_fusion:
                # Independent fusion module per task branch (own parameters), fed
                # from the same shared per-stream backbone features.
                self.fusion_rotation = build_fusion(fusion, feat_dim, len(streams), gate_init_prob=self.gate_init_prob, scale_dims=self.stream_scale_dims, scale_channels=self.stream_scale_channels)
                self.fusion_translation = build_fusion(fusion, feat_dim, len(streams), gate_init_prob=self.gate_init_prob, scale_dims=self.stream_scale_dims, scale_channels=self.stream_scale_channels)
                fusion_for_aux_check = self.fusion_rotation
            else:
                # build_fusion returns a single fusion module instance as specified by the config
                self.fusion = build_fusion(fusion, feat_dim, len(streams), gate_init_prob=self.gate_init_prob, scale_dims=self.stream_scale_dims, scale_channels=self.stream_scale_channels)
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

        # Optional LayerNorm on the fused vector, applied right before the rotation/translation
        # heads (after any adapters above). Independent per-branch instances (own learned
        # gain/bias each), same pattern as rotation_adapter/translation_adapter. Targets a
        # different failure mode than pre_head_grid_size: pre_head_grid_size shrinks the
        # bottleneck's *dimensionality*; this normalizes its *magnitude* every forward pass,
        # decoupling the scale the fusion/backbone happens to produce from the scale of the
        # gradient the head's (by far the model's largest) first Linear layer receives --
        # see CLAUDE.md's "Gate granularity / naming" note and the mean-gate ablation
        # (scripts/evaluate/gate_ablation_test.py) for the diagnosis this targets. Default
        # False is a no-op / fully backward compatible (identity passthrough not even
        # constructed, so no new state_dict keys for any existing config/checkpoint).
        self.pre_head_norm_rot = nn.LayerNorm(fused_dim) if self.pre_head_layernorm else None
        self.pre_head_norm_trans = nn.LayerNorm(fused_dim) if self.pre_head_layernorm else None


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
                activation=self.activation,
            bottleneck_rank=self.head_bottleneck_rank
            )
        elif rotation_parametrisation == "2D_unit_vector":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=2,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation,
            bottleneck_rank=self.head_bottleneck_rank
            )
        elif rotation_parametrisation == "angle":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=1,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation,
            bottleneck_rank=self.head_bottleneck_rank
            )
        elif rotation_parametrisation == "lie_algebra":
            self.rot_head = MLPHead(
                in_dim=fused_dim,
                out_dim=3,
                hidden_dims=rot_hidden,
                dropout=dropout_rot,
                bn=use_batchnorm,
                activation=self.activation,
            bottleneck_rank=self.head_bottleneck_rank
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
                    activation=self.activation,
                bottleneck_rank=self.head_bottleneck_rank
                )
            else:
                self.y_head = MLPHead(
                    in_dim=fused_dim,
                    out_dim=1,
                    hidden_dims=trans_hidden,
                    dropout=dropout_trans,
                    bn=use_batchnorm,
                    activation=self.activation,
                bottleneck_rank=self.head_bottleneck_rank
                )
                self.z_head = MLPHead(
                    in_dim=fused_dim,
                    out_dim=1,
                    hidden_dims=trans_hidden,
                    dropout=dropout_trans,
                    bn=use_batchnorm,
                    activation=self.activation,
                bottleneck_rank=self.head_bottleneck_rank
                )



    def forward_single_stream(self, x, extractor):
        p3, p4, p5 = extract_features_raw(x, extractor, self.p345_indices)

        # 1×1 channel reduction, then two stride-2 convolutions per scale for learned downsampling
        p3 = self.stride_p3_2(self.stride_p3_1(self.conv_p3(p3)))  # 80×80 -> 40×40 -> 20×20
        p4 = self.stride_p4_2(self.stride_p4_1(self.conv_p4(p4)))  # 40×40 -> 20×20 -> 10×10
        p5 = self.stride_p5_2(self.stride_p5_1(self.conv_p5(p5)))  # 20×20 -> 10×10 -> 5×5

        # Optional additional channel reduction (see pre_head_channel_divisor in
        # __init__). nn.Identity() when unset -- no-op, identical to every existing run.
        p3 = self.pre_head_channel_conv_p3(p3)
        p4 = self.pre_head_channel_conv_p4(p4)
        p5 = self.pre_head_channel_conv_p5(p5)

        # Optional additional shrink of the pre-flatten bottleneck (see
        # pre_head_grid_size in __init__). None (default) skips this entirely,
        # preserving the fixed 20x20/10x10/5x5 sizes above exactly -- identical to
        # every existing run/config, none of which have ever actually varied this
        # (the long-dead, unrelated `spatial_grid_size` key never affected this
        # 1D pipeline at all -- see the extended note in __init__).
        if self.pre_head_pool is not None:
            p3 = self.pre_head_pool(p3)
            p4 = self.pre_head_pool(p4)
            p5 = self.pre_head_pool(p5)

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

            # Optional LayerNorm right before the heads (see its construction site for why).
            # Applied last, after every adapter above, so it normalizes exactly what the
            # heads' first Linear layer actually receives, whatever produced it.
            if self.pre_head_norm_rot is not None:
                rotation_features = self.pre_head_norm_rot(rotation_features)
            if self.pre_head_norm_trans is not None:
                translation_features = self.pre_head_norm_trans(translation_features)

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
