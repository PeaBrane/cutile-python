# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import (Set, Callable, Tuple, get_origin, get_args, get_type_hints, Annotated, Any)

import cuda.tile._stub as ct


def _has_constant_annotation(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        _, *metadata = get_args(annotation)
        return any(isinstance(m, ct.ConstantAnnotation) for m in metadata)
    return False


def get_constant_annotations(pyfunc: Callable) -> Set[str]:
    const_annotations = set()
    sig = inspect.signature(pyfunc)
    resolved_hints = get_type_hints(pyfunc, globalns=pyfunc.__globals__, include_extras=True)
    for name, param in sig.parameters.items():
        param_annotation = resolved_hints.get(name, param.annotation)
        if _has_constant_annotation(param_annotation):
            const_annotations.add(name)
    return const_annotations


def get_constant_arg_flags(pyfunc: Callable) -> Tuple[bool, ...]:
    constant_arg_flags = []
    sig = inspect.signature(pyfunc)
    for _name, param in sig.parameters.items():
        if _has_constant_annotation(param.annotation):
            constant_arg_flags.append(True)
        else:
            constant_arg_flags.append(False)
    return tuple(constant_arg_flags)
