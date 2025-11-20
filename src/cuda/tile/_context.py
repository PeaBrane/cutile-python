# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass


@dataclass
class TileContextConfig:
    temp_dir: str
    log_keys: list[str]


def init_context_from_env(cls):
    config = TileContextConfig(
            temp_dir=get_temp_dir_from_env(),
            log_keys=get_log_keys_from_env()
            )
    return cls(config=config)


def get_log_keys_from_env():
    KEYS = {"CUTILEIR", "TILEIR"}
    env = os.environ.get('CUDA_TILE_LOGS', "")
    ret = []
    for x in env.split(","):
        x = x.upper().strip()
        if len(x) == 0:
            continue
        if x not in KEYS:
            raise RuntimeError(f"Unexpected value {x} in CUDA_TILE_LOGS, "
                               f"supported values are {KEYS}")
        ret.append(x)
    return ret


def _clean_tmp_dir(dir: str):
    shutil.rmtree(dir, ignore_errors=True)


def get_temp_dir_from_env():
    dir = os.environ.get('CUDA_TILE_TEMP_DIR', "")
    if dir == "":
        dir = tempfile.mkdtemp()
        atexit.register(_clean_tmp_dir, dir)
    return dir
