"""
MLP head for pose estimation. This module defines the MLPHead class, which is a simple feedforward neural network used as the head of the pose estimation model. 
Specific pose head classes that utilize the MLPHead in different ways for predicting translation and rotation components of the pose:
- PoseHead_with_quaternion: Predicts translation in (y, z) coordinates and rotation as a quaternion (4D).
- PoseHead_with_2D_unitVector: Predicts translation in (y, z) coordinates and rotation as a 2D unit vector (sin, cos).
- SE3_lie_group_PoseHead: Predicts translation in (x, y, z) coordinates and rotation as a 3D vector representing the Lie algebra of SE(3).
"""

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[512], dropout=0.0, bn=False, activation=nn.GELU()):
        super().__init__()
        layers = []
        prev = in_dim

        for h in hidden_dims:
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
        self.translation = MLPHead(in_dim, 2, hidden=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 4, hidden=[512], dropout=0.2, bn=True)

    def forward(self, x):
        """
        returns:
        - t_y: predicted translation in y direction (shape: [batch_size, 1])
        - t_z: predicted translation in z direction (shape: [batch_size, 1])
        - r_quat: predicted rotation as a quaternion (shape: [batch_size, 4])
        """
        t = self.translation(x)
        r = self.rotation(x)
        return t[:, 0:1], t[:, 1:2], r
    


class PoseHead_with_2D_unitVector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.translation = MLPHead(in_dim, 2, hidden=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 2, hidden=[512], dropout=0.2, bn=True)

    def forward(self, x):
        t = self.translation(x)
        r = self.rotation(x)
        return t[:, 0:1], t[:, 1:2], r[:, 0:1], r[:, 1:2]



class SE3_lie_group_PoseHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.translation = MLPHead(in_dim, 3, hidden=[512], dropout=0.2, bn=True)
        self.rotation = MLPHead(in_dim, 3, hidden=[512], dropout=0.2, bn=True)

    def forward(self, x):
        t = self.translation(x)
        r = self.rotation(x)
        return t[:, 0:1], t[:, 1:2], r[:, 0:1], r[:, 1:2], r[:, 2:3]
