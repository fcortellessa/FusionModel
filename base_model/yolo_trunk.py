# models/yolo_backbone.py
"""
This module takes care of loading the YOLO backbone and extracting the P3, P4, P5 feature maps while respecting the skip connections.
Key functions:
- load_yolo_trunk: Loads the full YOLO model and extracts the backbone (all
    layers except the final detection head).
- get_p345_indices: Identifies which layers correspond to the P3, P4, P5 feature maps.
- extract_features: Given an input image, runs it through the backbone and extracts the P3
    P4, P5 features while correctly handling the skip connections defined by the 'f' attribute in YOLO layers.
- align_channels: Utility function to modify the first convolutional layer of the backbone to accept a
different number of input channels (e.g., for depth images with 4 channels instead of 3).

"""

import torch
import torch.nn as nn
from ultralytics import YOLO


def load_yolo_trunk(model_name):
    full = YOLO(model_name).model
    base = full.model[:-1]
    return full, base


def get_p345_indices(model):
    layers = model.model if hasattr(model, "model") else model

    for layer in reversed(layers):
        if hasattr(layer, "f"):
            return layer.f

    raise ValueError("Could not find P3/P4/P5 indices")

    
def extract_features_raw(x, extractor, p345_indices):
    """
    Extracts P3, P4, P5 while respecting YOLO skip-connections.
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
        if i in p345_indices:
            outputs.append(x)
        
    return outputs



def align_channels(model, in_channels=4):
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    if first_conv is None:
        return model

    old_weight = first_conv.weight.data

    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=(first_conv.bias is not None)
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        if in_channels > 3:
            new_conv.weight[:, 3:] = old_weight[:, :1]

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is first_conv:
                setattr(module, child_name, new_conv)
                return model

    return model


def extract_features(x, extractor, p345_indices):
    cache = []
    outputs = []

    for i, layer in enumerate(extractor):
        if layer.f != -1:
            if isinstance(layer.f, int):
                x = cache[layer.f]
            else:
                x = [x if j == -1 else cache[j] for j in layer.f]

        x = layer(x)
        cache.append(x)

        if i in p345_indices:
            outputs.append(x)

    return outputs