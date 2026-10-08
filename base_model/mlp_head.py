"""
MLP head for pose estimation. This module defines the MLPHead class, which is a simple feedforward neural network used as the head of the pose estimation model. 
Specific pose head classes that utilize the MLPHead in different ways for predicting translation and rotation components of the pose:
- PoseHead_with_quaternion: Predicts translation in (y, z) coordinates and rotation as a quaternion (4D).
- PoseHead_with_2D_unitVector: Predicts translation in (y, z) coordinates and rotation as a 2D unit vector (sin, cos).
- SE3_lie_group_PoseHead: Predicts translation in (x, y, z) coordinates and rotation as a 3D vector representing the Lie algebra of SE(3).
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[512], dropout=0.0, bn=False, activation=nn.GELU(),
                 bottleneck_rank=None):
        """
        bottleneck_rank: optional low-rank factorization of ONLY the first Linear layer (the one
        consuming `in_dim` -- by far the largest layer for this project's ~44800-dim fused vector,
        see CLAUDE.md's "Gate granularity / naming" note: the head's first layer alone holds ~90%
        of the whole model's parameters). Replaces nn.Linear(in_dim, hidden_dims[0])
        (in_dim * hidden_dims[0] params) with nn.Linear(in_dim, bottleneck_rank, bias=False) ->
        nn.Linear(bottleneck_rank, hidden_dims[0]) (in_dim*bottleneck_rank + bottleneck_rank*hidden_dims[0]
        params). Every one of the in_dim inputs still reaches the layer -- unlike spatial pooling
        (model.pre_head_grid_size), no information is structurally discarded -- only the RANK of the
        linear map is constrained, not what it can see. Later layers (hidden-to-hidden, and the final
        output layer) are left untouched: they're already small relative to the first layer, so
        factoring them would save little. None (default) is the original single dense layer,
        unchanged -- fully backward compatible with every existing config/checkpoint.
        """
        super().__init__()
        layers = []
        prev = in_dim

        for idx, h in enumerate(hidden_dims):
            if idx == 0 and bottleneck_rank is not None and 0 < bottleneck_rank < prev:
                layers.append(nn.Linear(prev, bottleneck_rank, bias=False))
                layers.append(nn.Linear(bottleneck_rank, h))
            else:
                layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    


class PoseHead_with_quaternion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.translation = MLPHead(in_dim, 2, hidden_dims=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 4, hidden_dims=[512], dropout=0.2, bn=True)

    def forward(self, x):
        """
        returns:
        - t_y: predicted translation in y direction (shape: [batch_size, 1])
        - t_z: predicted translation in z direction (shape: [batch_size, 1])
        - r_quat: predicted rotation as a quaternion (shape: [batch_size, 4])
        """
        t = self.translation(x)
        r = self.rotation(x)
        r = torch.nn.functional.normalize(r, p=2, dim=1, eps=1e-12)
        return t[:, 0:1], t[:, 1:2], r
    


class PoseHead_with_2D_unitVector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.translation = MLPHead(in_dim, 2, hidden_dims=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 2, hidden_dims=[512], dropout=0.2, bn=True)

    def forward(self, x):
        t = self.translation(x)
        r = self.rotation(x)
        return t[:, 0:1], t[:, 1:2], r[:, 0:1], r[:, 1:2]



class SE3_lie_group_PoseHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.translation = MLPHead(in_dim, 3, hidden_dims=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 3, hidden_dims=[512], dropout=0.2, bn=True)

    def forward(self, x):
        t = self.translation(x)
        r = self.rotation(x)
        return t[:, 0:1], t[:, 1:2], r[:, 0:1], r[:, 1:2], r[:, 2:3]


class PoseHead_with_angle(nn.Module):
    def __init__(self, in_dim):
        """
        Predicts rotation as a single angle (roll) around x-axis.
        Direct angle regression often works better than unit vector for rotation around a single axis.
        """
        super().__init__()
        self.translation = MLPHead(in_dim, 2, hidden_dims=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 1, hidden_dims=[512], dropout=0.2, bn=True)

    def forward(self, x):
        """
        returns:
        - t_y: predicted translation in y direction (shape: [batch_size, 1])
        - t_z: predicted translation in z direction (shape: [batch_size, 1])
        - r_angle: predicted rotation angle in radians (shape: [batch_size, 1])
        """
        t = self.translation(x)
        r = self.rotation(x)  # shape: [batch_size, 1]
        return t[:, 0:1], t[:, 1:2], r


class PoseHead_with_lie_algebra(nn.Module):
    def __init__(self, in_dim):
        """
        Predicts rotation as a 3D Lie algebra vector (axis-angle).
        Direction of the vector is the rotation axis, magnitude is the angle.
        Converted to quaternion via exponential map during training.
        """
        super().__init__()
        self.translation = MLPHead(in_dim, 2, hidden_dims=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 3, hidden_dims=[512], dropout=0.2, bn=True)

    def forward(self, x):
        """
        returns:
        - t_y: predicted translation in y direction (shape: [batch_size, 1])
        - t_z: predicted translation in z direction (shape: [batch_size, 1])
        - r_lie: predicted rotation as 3D axis-angle vector (shape: [batch_size, 3])
        """
        t = self.translation(x)
        r = self.rotation(x)  # shape: [batch_size, 3]
        return t[:, 0:1], t[:, 1:2], r


class SE3_lie_algebra_direct_PoseHead(nn.Module):
    def __init__(self, in_dim):
        """
        Predicts complete SE(3) pose directly in Lie algebra representation.
        This optimizes the entire SE(3) element as a single 6D vector,
        treating it as a unified optimization problem rather than separate
        translation and rotation objectives.
        
        Output is 6D: [ρ_x, ρ_y, ρ_z, ω_x, ω_y, ω_z]
        where ρ is the translation vector and ω is the rotation vector (axis-angle).
        """
        super().__init__()
        # Single MLP head that outputs the full 6D SE(3) Lie algebra element
        self.se3_head = MLPHead(in_dim, 6, hidden_dims=[512, 256], dropout=0.2, bn=True)

    def forward(self, x):
        """
        returns:
        - se3_pose: predicted complete SE(3) pose as 6D vector (shape: [batch_size, 6])
          Elements [0:3] are translation (x, y, z)
          Elements [3:6] are rotation as 3D axis-angle vector
        """
        se3_pose = self.se3_head(x)  # shape: [batch_size, 6]
        return se3_pose

