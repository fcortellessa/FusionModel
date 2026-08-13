#!/usr/bin/env python3
"""
Quick smoke test for FLOPs computation.
Run: python test_flops.py
"""

import sys
sys.path.insert(0, '.')

from utils.flops_utils import compute_inference_flops
from base_model.pose_estimation_model import PoseEstimationNet
import torch

print("\nTesting FLOPs computation...")
print("=" * 60)

try:
    # Build a small test model
    model = PoseEstimationNet(
        model_name='yolo26n.pt',
        streams=('mono_cam', 'tactile'),
        fusion='gated',
        rotation_parametrisation='quaternion'
    )
    print("✓ Model created successfully")

    # Test with batch_size=32
    stats = compute_inference_flops(model, batch_size=32, device='cpu')
    print(f"✓ FLOPs computed successfully")
    print()
    print(f"  GFLOPs/sample:  {stats['gflops_per_sample']:.3f}")
    print(f"  GFLOPs/batch:   {stats['gflops_per_batch']:.3f}")
    print(f"  Params (total): {stats['params']:,}")
    print(f"  Params (M):     {stats['params_millions']:.2f}M")
    print()
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

except Exception as e:
    print(f"✗ Test failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
