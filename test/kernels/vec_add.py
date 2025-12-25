# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
import math


# Define a type alias for Constant integers.
# This helps in clearly indicating that certain kernel parameters are compile-time constants.
ConstInt = ct.Constant[int]


# --- Kernel 1: 1D Tiled Vector Add (Direct Load/Store) ---
@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ConstInt):
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.

    Each block processes a `TILE`-sized chunk of the vectors.
    This approach is efficient when the total dimension is a multiple of `TILE`,
    or when out-of-bounds accesses are implicitly handled by the calling context
    (e.g., by padding or ensuring input sizes match grid dimensions).

    Args:
        a: Input tensor A.
        b: Input tensor B.
        c: Output tensor for the sum (A + B).
        TILE (ConstInt): The size of the tile (chunk of data) processed by each
                         block. This must be a compile-time constant.
    """
    # Get the global ID of the current block along the first dimension.
    # In a 1D grid, this directly corresponds to the index of the tile.
    bid = ct.bid(0)

    # Load TILE-sized chunks from input vectors 'a' and 'b'.
    # `ct.load` automatically distributes the load operation across the threads
    # within the block, bringing the specified tile of data into shared memory
    # or registers. The `index=(bid,)` specifies which tile to load based on the block ID.
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Perform the element-wise addition on the loaded tiles.
    # This operation happens in parallel across the threads within the block.
    sum_tile = a_tile + b_tile

    # Store the resulting TILE-sized chunk back to the output vector 'c'.
    # `ct.store` writes the computed tile back to global memory, again
    # distributing the store operation across threads.
    ct.store(c, index=(bid,), tile=sum_tile)


# --- Kernel 2: 2D Tiled Matrix Add (Direct Load/Store) ---
@ct.kernel
def vec_add_kernel_2d(a, b, c, TILE_X: ConstInt, TILE_Y: ConstInt):
    """
    cuTile kernel for 2D element-wise matrix addition using direct tiled loads/stores.

    Each block computes a `TILE_X` x `TILE_Y` chunk of the matrices.
    Similar to the 1D direct kernel, this is efficient when dimensions are
    multiples of the tile sizes.

    Args:
        a: Input matrix A.
        b: Input matrix B.
        c: Output matrix for the sum (A + B).
        TILE_X (ConstInt): The tile dimension along the X-axis (rows).
        TILE_Y (ConstInt): The tile dimension along the Y-axis (columns).
    """
    # Get the global IDs of the current block along the X and Y axes.
    # `ct.bid(0)` for the first grid dimension (typically rows),
    # `ct.bid(1)` for the second grid dimension (typically columns).
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # Load `TILE_X` x `TILE_Y` chunks from input matrices 'a' and 'b'.
    # The `index=(bid_x, bid_y)` specifies the 2D tile to load.
    a_tile = ct.load(a, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))
    b_tile = ct.load(b, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))

    # Perform the element-wise addition on the loaded tiles.
    sum_tile = a_tile + b_tile

    # Store the resulting `TILE_X` x `TILE_Y` chunk back to the output matrix 'c'.
    ct.store(c, index=(bid_x, bid_y), tile=sum_tile)


# --- Kernel 3: 1D Tiled Vector Add (Gather/Scatter with Masking) ---
@ct.kernel
def vec_add_kernel_1d_gather(a, b, c, TILE: ConstInt):
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.

    Each block processes a `TILE`-sized chunk of the vectors.
    This approach is efficient when the total dimension is a multiple of `TILE`,
    or when out-of-bounds accesses are implicitly handled by the calling context
    (e.g., by padding or ensuring input sizes match grid dimensions).

    Args:
        a: Input tensor A.
        b: Input tensor B.
        c: Output tensor for the sum (A + B).
        TILE (ConstInt): The size of the tile (chunk of data) processed by each
                         block. This must be a compile-time constant.
    """
    # Get the global ID of the current block.
    bid = ct.bid(0)

    # Calculate indices for elements within the current block's tile.
    # `ct.arange(TILE, ...)` creates local offsets [0, 1, ..., TILE-1] for threads
    # within the block. `bid * TILE` shifts these to the correct global starting
    # point for this specific block.
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

    # Load elements using the calculated indices.
    # `ct.gather` only loads data within the array bounds, and zeroes any out-of-bounds elements.
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    # Perform the element-wise addition.
    sum_tile = a_tile + b_tile

    # Store the result using the same indices.
    # `ct.scatter()` only writes data to positions within the array bounds.
    ct.scatter(c, indices, sum_tile)


# --- Kernel 4: 2D Tiled Matrix Add (Gather/Scatter with Masking) ---
@ct.kernel
def vec_add_kernel_2d_gather(
    a, b, c,
    TILE_X: ConstInt, TILE_Y: ConstInt  # Tile dimensions for this block
):
    """
    cuTile kernel for 2D element-wise matrix addition using direct tiled loads/stores.

    Each block computes a `TILE_X` x `TILE_Y` chunk of the matrices.
    Similar to the 1D direct kernel, this is efficient when dimensions are
    multiples of the tile sizes.

    Args:
        a: Input matrix A.
        b: Input matrix B.
        c: Output matrix for the sum (A + B).
        TILE_X (ConstInt): The tile dimension along the X-axis (rows).
        TILE_Y (ConstInt): The tile dimension along the Y-axis (columns).
    """
    # Get the global IDs of the current block along the X and Y axes.
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # Calculate X and Y indices within the current block's tile.
    x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)
    y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)

    # Reshape the X and Y indices to (TILE_X, 1) and (1, TILE_Y), respectively.
    # This way, they can be broadcasted together to a common shape (TILE_X, TILE_Y).
    x = x[:, None]
    y = y[None, :]

    # Load elements using the calculated X and Y indices.
    # Both `a_tile` and `b_tile` have shape (TILE_X, TILE_Y).
    a_tile = ct.gather(a, (x, y))
    b_tile = ct.gather(b, (x, y))

    # Perform the element-wise addition.
    sum_tile = a_tile + b_tile

    # Store the result back to `c` using the same index tiles.
    # `ct.scatter()` only writes data to positions within the array bounds.
    ct.scatter(c, (x, y), sum_tile)


def vec_add(a, b, use_gather):
    assert a.shape == b.shape
    assert a.dim() <= 2

    c = torch.empty_like(a)

    if a.dim() == 1:
        TILE = min(1024, 2 ** math.ceil(math.log2(a.shape[0])))
        grid = (math.ceil(a.shape[0] / TILE), 1, 1)
        kernel = vec_add_kernel_1d_gather if use_gather else vec_add_kernel_1d
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE))
    else:
        c = torch.empty_like(a)

        kernel = vec_add_kernel_2d_gather if use_gather else vec_add_kernel_2d
        TILE_Y = min(1024, 2 ** math.ceil(math.log2(a.shape[1])))
        TILE_X = 1024 // TILE_Y
        grid = (math.ceil(a.shape[0] / TILE_X), math.ceil(a.shape[1] / TILE_Y), 1)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE_X, TILE_Y))

    return c
