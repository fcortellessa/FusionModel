import torch.nn as nn

class BaseFusion(nn.Module):
    def __init__(self, dim, num_streams):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
    
    def _validate(self, features_list):
        assert len(features_list) == self.num_streams, \
            f"Expected {self.num_streams} streams"

    def forward(self, features_list):
        raise NotImplementedError
    
