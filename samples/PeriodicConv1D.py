# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating periodic (circular) 1D convolution with fused ReLU.

Compares:
- PyTorch: F.conv1d with circular padding + separate F.relu (unfused, two kernel launches)
- cuTile: Fused periodic conv + ReLU in a single kernel

This demonstrates the performance benefit of kernel fusion.
"""

import argparse
import cuda.tile as ct
import torch
import torch.nn.functional as F
from math import ceil

from samples.utils.benchmark import report_benchmark


ConstInt = ct.Constant[int]


# =============================================================================
# PyTorch Reference Implementation (Unfused)
# =============================================================================

def pytorch_periodic_conv1d_relu(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    PyTorch periodic 1D convolution with ReLU (NOT fused - two kernel launches).

    Args:
        input: Input tensor of shape (B, C_in, L)
        weight: Kernel weights of shape (C_out, C_in, 3)

    Returns:
        Output tensor of shape (B, C_out, L)
    """
    # Circular padding for kernel size 3: pad 1 on each side
    input_padded = F.pad(input, (1, 1), mode='circular')

    # Convolution (first kernel launch)
    output = F.conv1d(input_padded, weight, padding=0)

    # ReLU (second kernel launch - NOT fused)
    output = F.relu(output)

    return output


# =============================================================================
# cuTile Implementation (Fused Conv + ReLU)
# =============================================================================

@ct.kernel
def periodic_conv1d_relu_kernel(input, weight, output,
                                C_in: ConstInt,
                                TILE_C: ConstInt,
                                TILE_L: ConstInt):
    """
    cuTile kernel for periodic 1D convolution with fused ReLU.

    Grid indexing:
        - bid(0) = batch index
        - bid(1) = output channel tile index
        - bid(2) = spatial tile index

    Args:
        input: Input tensor of shape (B, C_in, L)
        weight: Kernel weights of shape (C_out, C_in, 3)
        output: Output tensor of shape (B, C_out, L)
        C_in: Number of input channels (compile-time constant)
        TILE_C: Tile size for output channels
        TILE_L: Tile size for spatial dimension
    """
    bid_b = ct.bid(0)   # Batch
    bid_c = ct.bid(1)   # Output channel tile
    bid_l = ct.bid(2)   # Spatial tile

    L = input.shape[2]

    # Output tile indices
    c_start = bid_c * TILE_C
    l_start = bid_l * TILE_L

    # Accumulator for output tile (TILE_C, TILE_L)
    acc = ct.zeros((TILE_C, TILE_L), dtype=ct.float32)

    # Accumulate over input channels
    for c_in in range(C_in):
        # Gather input patch with circular wrap: (TILE_L + 2,) for kernel size 3
        # We need positions [l_start - 1, l_start, ..., l_start + TILE_L] with wrap
        l_indices = (l_start - 1 + ct.arange(TILE_L + 2, dtype=ct.int32)) % L
        input_patch = ct.gather(input, (bid_b, c_in, l_indices))  # (TILE_L + 2,)

        # Load kernel weights for this input channel: (TILE_C, 3)
        weight_tile = ct.load(weight, (c_start, c_in, 0), (TILE_C, 1, 3),
                              padding_mode=ct.PaddingMode.ZERO)
        weight_tile = ct.reshape(weight_tile, (TILE_C, 3))

        # Convolve: for each of 3 kernel positions
        for k in range(3):
            # Extract shifted patch of length TILE_L
            patch_shifted = ct.extract(input_patch, index=(k,), shape=(TILE_L,))
            # Broadcast multiply and accumulate
            acc = acc + weight_tile[:, k:k+1] * patch_shifted[None, :]

    # Fused ReLU
    acc = ct.maximum(acc, 0.0)

    # Store output tile
    ct.store(output, (bid_b, c_start, l_start), acc)


def cutile_periodic_conv1d_relu(input: torch.Tensor, weight: torch.Tensor,
                                 tile_c: int = 32, tile_l: int = 32) -> torch.Tensor:
    """
    cuTile periodic 1D convolution with fused ReLU.

    Args:
        input: Input tensor of shape (B, C_in, L)
        weight: Kernel weights of shape (C_out, C_in, 3)
        tile_c: Tile size for output channels (must divide C_out evenly)
        tile_l: Tile size for spatial dimension (must divide L evenly)

    Returns:
        Output tensor of shape (B, C_out, L)
    """
    B, C_in, L = input.shape
    C_out = weight.shape[0]

    # Ensure tile sizes divide dimensions evenly
    assert C_out % tile_c == 0, f"C_out ({C_out}) must be divisible by tile_c ({tile_c})"
    assert L % tile_l == 0, f"L ({L}) must be divisible by tile_l ({tile_l})"

    # Allocate output
    output = torch.empty((B, C_out, L), device=input.device, dtype=input.dtype)

    # Calculate grid dimensions
    grid_b = B
    grid_c = C_out // tile_c
    grid_l = L // tile_l
    grid = (grid_b, grid_c, grid_l)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        periodic_conv1d_relu_kernel,
        (input, weight, output, C_in, tile_c, tile_l)
    )

    return output


# =============================================================================
# Correctness Verification
# =============================================================================

def verify_correctness(B: int, C_in: int, C_out: int, L: int,
                       tile_c: int = 32, tile_l: int = 32) -> bool:
    """
    Verify that cuTile and PyTorch implementations produce matching results.
    """
    # Create random input and weights
    input = torch.randn(B, C_in, L, device='cuda', dtype=torch.float32)
    weight = torch.randn(C_out, C_in, 3, device='cuda', dtype=torch.float32)

    # Compute using both implementations
    output_pytorch = pytorch_periodic_conv1d_relu(input, weight)
    output_cutile = cutile_periodic_conv1d_relu(input, weight, tile_c, tile_l)

    # Compare
    try:
        torch.testing.assert_close(output_cutile, output_pytorch, rtol=1e-4, atol=1e-4)
        return True
    except AssertionError as e:
        print(f"Verification failed: {e}")
        return False


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(B: int, C_in: int, C_out: int, L: int,
                  tile_c: int = 32, tile_l: int = 32) -> dict:
    """
    Benchmark both implementations and return timing results.
    """
    # Create random input and weights
    input = torch.randn(B, C_in, L, device='cuda', dtype=torch.float32)
    weight = torch.randn(C_out, C_in, 3, device='cuda', dtype=torch.float32)

    # Ensure CUDA is ready
    torch.cuda.synchronize()

    # Benchmark PyTorch
    def run_pytorch():
        return pytorch_periodic_conv1d_relu(input, weight)

    pytorch_result = report_benchmark(run_pytorch, ())

    # Benchmark cuTile
    def run_cutile():
        return cutile_periodic_conv1d_relu(input, weight, tile_c, tile_l)

    cutile_result = report_benchmark(run_cutile, ())

    return {
        "pytorch_ms": pytorch_result["mean_time_ms"],
        "cutile_ms": cutile_result["mean_time_ms"],
        "speedup": pytorch_result["mean_time_ms"] / cutile_result["mean_time_ms"]
    }


def run_sweep():
    """
    Run benchmark sweep over increasing channel sizes and input lengths.
    """
    B = 128  # Fixed batch size
    channels = [16, 32, 64, 128, 256, 512]
    lengths = [256, 512, 1024, 2048, 4096]

    print("\n" + "=" * 80)
    print("Periodic 1D Convolution + ReLU Benchmark")
    print("PyTorch: Unfused (conv + relu = 2 kernel launches)")
    print("cuTile:  Fused (conv + relu = 1 kernel launch)")
    print("=" * 80)
    print(f"\nBatch size: {B}")
    print(f"Kernel size: 3")
    print(f"Data type: float32\n")

    print(f"{'C_in/C_out':>12} {'Length':>10} {'PyTorch (ms)':>14} {'cuTile (ms)':>13} {'Speedup':>10}")
    print("-" * 65)

    for C in channels:
        for L in lengths:
            C_in = C
            C_out = C

            # Choose tile sizes that divide evenly
            tile_c = min(32, C_out)
            tile_l = min(32, L)

            # Ensure divisibility
            while C_out % tile_c != 0:
                tile_c //= 2
            while L % tile_l != 0:
                tile_l //= 2

            # Verify correctness first (with small batch for speed)
            if not verify_correctness(4, C_in, C_out, L, tile_c, tile_l):
                print(f"{C:>12} {L:>10} {'FAILED':>14}")
                continue

            # Run benchmark
            result = run_benchmark(B, C_in, C_out, L, tile_c, tile_l)

            print(f"{C:>12} {L:>10} {result['pytorch_ms']:>14.3f} "
                  f"{result['cutile_ms']:>13.3f} {result['speedup']:>10.2f}x")

    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def test():
    """
    Quick test to verify correctness with a small example.
    """
    B, C_in, C_out, L = 4, 32, 32, 256

    print("Testing periodic 1D convolution with fused ReLU...")
    print(f"  Input shape:  ({B}, {C_in}, {L})")
    print(f"  Weight shape: ({C_out}, {C_in}, 3)")
    print(f"  Output shape: ({B}, {C_out}, {L})")

    if verify_correctness(B, C_in, C_out, L):
        print("✓ Correctness test passed!")
    else:
        print("✗ Correctness test failed!")
        return

    # Quick benchmark
    result = run_benchmark(B, C_in, C_out, L)
    print(f"\nQuick benchmark:")
    print(f"  PyTorch: {result['pytorch_ms']:.3f} ms")
    print(f"  cuTile:  {result['cutile_ms']:.3f} ms")
    print(f"  Speedup: {result['speedup']:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periodic 1D Convolution + ReLU")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full benchmark sweep")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        test()

