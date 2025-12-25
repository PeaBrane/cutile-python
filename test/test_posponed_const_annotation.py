# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Postpone evaluation of type annotations.
from __future__ import annotations

import cuda.tile as ct

ConstInt = ct.Constant[int]


def needs_constant(x: ct.Constant):
    pass


def needs_constant_int(x: ConstInt):
    pass


def needs_constant_bool(x: ct.Constant[bool]):
    pass


# TODO: Run with `mypy --check-untyped-defs` or another static type checker.
def test_constant_type_hints() -> None:
    int_constant: ct.Constant[int] = 42
    float_constant: ct.Constant[float] = 3.14
    bool_constant: ct.Constant[bool] = True

    needs_constant(int_constant)
    needs_constant(float_constant)
    needs_constant(bool_constant)
    needs_constant_int(int_constant)
    needs_constant_bool(bool_constant)
