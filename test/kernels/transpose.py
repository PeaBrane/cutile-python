# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct


# Define a type alias for Constant integers.
# This helps in clearly indicating that certain kernel parameters are compile-time constants,
# which cuTile uses for optimization and code generation.
ConstInt = ct.Constant[int]


@ct.kernel
def transpose_kernel(x, y,
                     tm: ConstInt,  # Tile size along M dimension (rows of original x)
                     tn: ConstInt):  # Tile size along N dimension (columns of original x)
    """
    cuTile kernel to transpose a 2D matrix by processing data in tiles.

    Each block is responsible for computing a `tn` x `tm` tile
    of the output (transposed) matrix `y`. This involves loading a `tm` x `tn`
    tile from the input matrix `x`, transposing it locally, and then storing
    the `tn` x `tm` result to the correct location in `y`.

    Args:
        x: Input matrix (M x N).
        y: Output matrix (N x M), which will be the transpose of x.
        tm (ConstInt): The height of the input tile (number of rows from x)
                       processed by this block.
        tn (ConstInt): The width of the input tile (number of columns from x)
                       processed by this block.
    """
    # Get the global IDs of the current block in a 2D grid.
    # `ct.bid(0)` gives the block ID along the X-axis of the grid, which corresponds
    # to the M-tile index (rows) of the original input matrix `x`.
    # `ct.bid(1)` gives the block ID along the Y-axis of the grid, which corresponds
    # to the N-tile index (columns) of the original input matrix `x`.
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    # Load a tile from the input matrix 'x'.
    # `ct.load` reads a `tm` x `tn` chunk of data from global memory `x`
    # at the specified `index=(bidx, bidy)`. This data is brought into
    # the block's local scope (e.g., shared memory or registers).
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))

    # Transpose the loaded tile.
    # `ct.transpose` without explicit axes defaults to swapping the last two dimensions.
    # For a 2D tile of shape (tm, tn), this operation transforms it into a (tn, tm) tile.
    transposed_tile = ct.transpose(input_tile)

    # Store the transposed tile to the output matrix 'y'.
    # Crucially, the store index for `y` must be swapped (`bidy`, `bidx`)
    # because `y` is the transpose of `x`. The `tile` argument provides
    # the `tn` x `tm` data to be written to global memory.
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)
