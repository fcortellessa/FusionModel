"""
Utility for computing model FLOPs (floating point operations).
Uses the 'thop' library for PyTorch model analysis.
"""

import torch


def compute_inference_flops(model, batch_size=1, input_height=640, input_width=640, device='cpu'):
    """
    Compute FLOPs for the PoseEstimationNet.

    Args:
        model: PoseEstimationNet instance
        batch_size: Batch size for FLOPs computation
        input_height: Input image height (default 640 for YOLO)
        input_width: Input image width (default 640 for YOLO)
        device: Device to compute on ('cpu' or 'cuda')

    Returns:
        dict with keys:
        - 'flops_total': Total FLOPs as integer
        - 'gflops_per_sample': GFLOPs per single sample (float)
        - 'gflops_per_batch': GFLOPs for entire batch (float)
        - 'params': Total parameters
        - 'params_millions': Parameters in millions
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError(
            "thop is not installed. Install it with: pip install thop\n"
            "Visit: https://github.com/Lyken17/pytorch-OpCounter"
        )

    model.to(device)
    model.eval()

    # Create dummy inputs matching the two streams (RGB + tactile)
    dummy_inputs = {
        'mono_cam': torch.randn(batch_size, 3, input_height, input_width, device=device),
        'tactile': torch.randn(batch_size, 3, input_height, input_width, device=device),
    }

    # Compute FLOPs using thop
    # profile returns (flops, params) where flops is the count for one forward pass
    flops, params = profile(model, inputs=(dummy_inputs,), verbose=False)

    # Convert to GFLOPs
    gflops_per_batch = flops / 1e9
    gflops_per_sample = gflops_per_batch / batch_size
    params_millions = params / 1e6

    return {
        'flops_total': flops,
        'gflops_per_batch': gflops_per_batch,
        'gflops_per_sample': gflops_per_sample,
        'params': params,
        'params_millions': params_millions,
    }
